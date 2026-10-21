#!/usr/bin/env python3
"""Drive one brood pipeline stage from a YAML config.

Reads a stage config (e.g. ``configs/screening.yaml``,
``configs/representation.yaml`` or ``configs/heldout.yaml``), resolves the
per-model conda interpreter and GPU slot, runs tune → cv for each model (or
held-out prediction for a held-out stage), and renders the stage's figures.
Stages whose outputs already exist are skipped so interrupted runs resume
cheaply.

Usage
-----
    brood-run --config configs/screening.yaml --dataset GNEtolC
    brood-run --config configs/representation.yaml --dataset GNEtolC
    brood-run --config configs/heldout.yaml --dataset GNEtolC
    brood-run --config configs/screening.yaml --dataset GNEtolC --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from loguru import logger

from brood.core.logging import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[2]

TRAIN_MODULE = "brood.cli.train"
PLOT_MODULE = "brood.plotting.cv_comparison"
EVALUATE_MODULE = "brood.cli.evaluate"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive one brood pipeline stage from a YAML config."
    )
    parser.add_argument(
        "--config",
        default="configs/screening.yaml",
        help="Stage config (YAML)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data"),
        help="Data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(REPO_ROOT / "figures"),
        help="Figure directory (default: <repo>/figures)",
    )
    parser.add_argument("--dataset", default=None, help="Run one dataset only")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if their outputs exist",
    )
    return parser.parse_args()


_ACTIVE: set[subprocess.Popen] = set()


def _run(cmd: list[str], env: dict | None = None) -> int:
    proc = subprocess.Popen(cmd, env=env)
    _ACTIVE.add(proc)
    try:
        return proc.wait()
    finally:
        _ACTIVE.discard(proc)


def _kill_active() -> None:
    for proc in list(_ACTIVE):
        proc.terminate()


def _interpreter(env: str) -> list[str]:
    micromamba = shutil.which("micromamba") or "/home/bruno/bin/micromamba"
    return [micromamba, "run", "-n", env, "python"]


def _run_stage(cmd: list[str], gpu: int | None = None) -> int:
    """Run a command, optionally pinning it to one GPU."""
    if gpu is not None:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    else:
        env = None
    return _run(cmd, env)


def _model_command(
    model: str,
    mcfg: dict,
    dataset: str,
    data_dir: str,
    defaults: dict,
    cv: bool = False,
    tune: bool = False,
    tune_only: bool = False,
) -> list[str]:
    """Build a brood-train command list from a model config entry.

    Parameters
    ----------
    model : str
        Model value passed to ``--model`` (defaults to the run key prefix).
    mcfg : dict
        Model config entry with ``env`` and optional keys.
    dataset : str
        Dataset name.
    data_dir : str
        Data directory path.
    defaults : dict
        Stage defaults (currently only ``n_trials``).
    cv : bool
        Whether to append ``--cv`` and ``--n-trials`` (CV tunes per training
        training split).
    tune : bool
        Whether to append ``--tune`` and ``--n-trials``.
    tune_only : bool
        Whether to append ``--tune-only``.

    Returns
    -------
    list[str]
        Command tokens to run through ``_interpreter``.
    """
    cmd = _interpreter(mcfg["env"]) + [
        "-m",
        TRAIN_MODULE,
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
        "--model",
        model,
    ]
    if mcfg.get("representation"):
        cmd += ["--representation", mcfg["representation"]]
    if tune:
        cmd += ["--tune", "--n-trials", str(defaults.get("n_trials", 10))]
        if tune_only:
            cmd += ["--tune-only"]
    if cv:
        cmd += ["--cv", "--n-trials", str(defaults.get("n_trials", 10))]
    return cmd


def _evaluate_command(
    dataset: str,
    runs: dict[str, Path],
    data_dir: str,
    figures_dir: str,
) -> list[str]:
    """Build the held-out evaluation command.

    Every path is passed explicitly: ``--output-dir`` and ``--figures-dir``
    default to cwd-relative locations, so relying on the defaults would write
    results next to wherever the runner happened to be invoked. All runs go
    after a *single* ``--runs`` because it is nargs="+": repeating the flag
    would make argparse keep only the last pair.
    """
    cmd = _interpreter("brood") + [
        "-m",
        EVALUATE_MODULE,
        "--dataset",
        dataset,
        "--test-full-path",
        str(Path(data_dir) / "splits" / dataset / "test.csv"),
        "--output-dir",
        str(Path(data_dir) / "evaluations" / dataset),
        "--figures-dir",
        figures_dir,
        "--runs",
    ]
    cmd += [f"{model}={path}" for model, path in runs.items()]
    return cmd


def _stage(
    cmd: list[str],
    label: str,
    gpu: int | None = None,
) -> bool:
    """Run a stage; warn and continue on failure."""
    logger.info(f"[{label}] {' '.join(cmd)} (gpu={gpu})")
    rc = _run_stage(cmd, gpu)
    if rc != 0:
        logger.error(f"[{label}] failed with exit code {rc}")
        return False
    logger.info(f"[{label}] done")
    return True


def _ensure(
    output: Path,
    cmd: list[str],
    label: str,
    *,
    force: bool,
    gpu: int | None = None,
) -> bool:
    """Run *cmd* unless *output* exists; return whether it is now present."""
    if not force and output.exists():
        logger.info(f"[{label}] up to date")
        return True
    return _stage(cmd, label, gpu)


def _stale(golden: Path, inputs: list[Path]) -> bool:
    """Whether any input is newer than the final output."""
    return any(
        path.exists() and path.stat().st_mtime > golden.stat().st_mtime
        for path in inputs
    )


# A CV model stage: (run name, completion marker, command, GPU slot).
# The marker is ``metrics/cv.csv``: ``run_cv_model`` writes it once at the end,
# whereas the raw predictions file is rewritten after every fold, so a partial
# predictions file must not be read as a finished CV run.
_CvStage = tuple[str, Path, list[str], int | None]


def _gpu_lanes(stages: list[_CvStage]) -> dict[int | None, list[_CvStage]]:
    """Group stages into resource lanes keyed by GPU slot (None = CPU).

    Two CPU models each saturate every core (``n_jobs=-1``) and two models on
    the same GPU slot contend for the device, so models within a lane run
    serially while distinct lanes (CPU vs each GPU) run concurrently.
    """
    lanes: dict[int | None, list[_CvStage]] = {}
    for stage in stages:
        lanes.setdefault(stage[3], []).append(stage)
    return lanes


def _run_cv_models(
    name: str,
    stages: list[_CvStage],
    force: bool,
    parallel: bool,
) -> dict[str, Path]:
    """Run each model's CV stage, optionally overlapping CPU and GPU lanes."""

    def _drain(lane: list[_CvStage]) -> dict[str, Path]:
        done: dict[str, Path] = {}
        for run_name, cv_csv, cmd, gpu in lane:
            if _ensure(
                cv_csv,
                cmd,
                f"{name}/{run_name} cv",
                force=force,
                gpu=gpu,
            ):
                done[run_name] = cv_csv
        return done

    if not parallel:
        return _drain(stages)

    lanes = _gpu_lanes(stages)
    cv_outputs: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        for lane_outputs in pool.map(_drain, lanes.values()):
            cv_outputs.update(lane_outputs)
    return cv_outputs


def _resolve_datasets(
    cfg: dict,
    data_dir: Path,
    requested: str | None,
    heldout: bool,
) -> dict[str, dict]:
    """Resolve the dataset list, optionally limited to ``requested``."""
    model_config = cfg["models"]
    configured = cfg.get("datasets")
    if configured is None:
        split_name = "test.csv" if heldout else "cv_folds.csv"
        configured = sorted(
            path.parent.name
            for path in (data_dir / "splits").glob(f"*/{split_name}")
        )
    if requested:
        configured = [requested]
    return {
        name: {
            "models": model_config,
        }
        for name in configured
    }


def _run_heldout(
    name: str,
    dcfg: dict,
    data_dir: Path,
    figures_dir: Path,
    force: bool,
    defaults: dict,
    heldout_models: list[str],
) -> None:
    """Predict the held-out test set, then evaluate every run that scored.

    This is the only stage that reads the test split, so it is kept separate
    from the CV stages on purpose: nothing else can touch the test labels.
    """
    runs = data_dir / "runs" / name
    test_outputs: dict[str, Path] = {}

    for model_key in heldout_models:
        mcfg = dcfg["models"].get(model_key)
        if mcfg is None:
            logger.warning(
                f"[{name} heldout] {model_key} is not declared under "
                "datasets.models; skipping"
            )
            continue

        test_csv = runs / model_key / "predictions" / "test.csv"
        model = mcfg.get("model", model_key.split("_")[0])

        # Tuned runs need a full-training params.json for the refit. CV tunes
        # on each training portion, so this is the only place params.json is
        # produced.
        if mcfg.get("tune"):
            if not _ensure(
                runs / model_key / "params.json",
                _model_command(
                    model,
                    mcfg,
                    name,
                    str(data_dir),
                    defaults,
                    cv=False,
                    tune=True,
                    tune_only=True,
                ),
                f"{name}/{model_key} tune",
                force=force,
                gpu=mcfg.get("gpu"),
            ):
                continue

        # Reuse the tuned params.json for the held-out prediction.
        if not _ensure(
            test_csv,
            _model_command(
                model,
                mcfg,
                name,
                str(data_dir),
                defaults,
            ),
            f"{name}/{model_key} heldout",
            force=force,
            gpu=mcfg.get("gpu"),
        ):
            continue
        test_outputs[model_key] = test_csv

    if not test_outputs:
        logger.warning(f"[{name} heldout] no predictions to evaluate")
        return

    _stage(
        _evaluate_command(name, test_outputs, str(data_dir), str(figures_dir)),
        f"{name} evaluate",
    )


def run_dataset(
    name: str,
    dcfg: dict,
    data_dir: Path,
    figures_dir: Path,
    force: bool,
    defaults: dict,
    plot_models: list[str] | None,
    repr_plot_models: list[str] | None,
    heldout_models: list[str] | None = None,
    parallel: bool = False,
) -> None:
    """Run all model + plot stages for one dataset.

    Parameters
    ----------
    name : str
        Dataset name.
    dcfg : dict
        Dataset config entry with ``models``.
    data_dir : Path
        Root data directory.
    figures_dir : Path
        Root figures directory.
    force : bool
        Re-run stages even if outputs exist.
    defaults : dict
        Stage defaults (``n_trials``).
    plot_models : list[str] | None
        Runs to include in the leaderboard figure.
    repr_plot_models : list[str] | None
        Runs to include in the representation figures.
    heldout_models : list[str] | None
        If provided, switch to held-out prediction + evaluation.
    parallel : bool
        Overlap the CPU lane with each GPU lane's CV stages. Default is
        serial (CPU models already use every core).
    """
    splits = data_dir / "splits"
    runs = data_dir / "runs" / name

    if heldout_models:
        if not (splits / name / "test.csv").exists():
            logger.warning(
                f"[{name}] missing test split — run "
                "`brood-split` first; skipping dataset"
            )
            return
        _run_heldout(
            name,
            dcfg,
            data_dir,
            figures_dir,
            force,
            defaults,
            heldout_models,
        )
        return

    if not (splits / name / "cv_folds.csv").exists():
        logger.warning(
            f"[{name}] missing CV folds CSV — run "
            "`brood-split` first; skipping dataset"
        )
        return

    stages = [
        (
            model_key,
            runs / model_key / "metrics" / "cv.csv",
            _model_command(
                mcfg.get("model", model_key.split("_")[0]),
                mcfg,
                name,
                str(data_dir),
                defaults,
                cv=True,
                tune=False,
            ),
            mcfg.get("gpu"),
        )
        for model_key, mcfg in dcfg["models"].items()
    ]
    cv_outputs = _run_cv_models(name, stages, force, parallel)

    if not cv_outputs:
        return

    def _plot_cmd(models: list[str]) -> list[str]:
        return _interpreter("brood") + [
            "-m",
            PLOT_MODULE,
            "--dataset",
            name,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(figures_dir),
            "--models",
            *models,
        ]

    cv_figures_dir = figures_dir / name / "cv"

    if plot_models:
        leaderboard_inputs = [
            cv_outputs[m] for m in plot_models if m in cv_outputs
        ]
        if len(leaderboard_inputs) < 2:
            logger.warning(
                f"[{name} plot] need at least 2 models for leaderboard"
            )
        else:
            svg = cv_figures_dir / f"{name}_screening_ap.svg"
            if force or not svg.exists() or _stale(svg, leaderboard_inputs):
                cmd = _plot_cmd(plot_models)
                if repr_plot_models is None:
                    cmd += ["--skip-representation"]
                _stage(cmd, f"{name} plot")
            else:
                logger.info(f"[{name} plot] up to date")

    if repr_plot_models:
        repr_inputs = [
            cv_outputs[m] for m in repr_plot_models if m in cv_outputs
        ]
        if len(repr_inputs) < len(repr_plot_models):
            logger.warning(
                f"[{name} plot representations] missing "
                f"{len(repr_plot_models) - len(repr_inputs)} of "
                f"{len(repr_plot_models)} runs; skipping figure 2"
            )
        else:
            # This stage writes both representation figures, so both must be
            # present and fresh before it is skipped.
            repr_svgs = [
                cv_figures_dir / f"{name}_representation_comparison.svg",
                cv_figures_dir / f"{name}_representation_ap.svg",
            ]
            if (
                force
                or any(not svg.exists() for svg in repr_svgs)
                or any(_stale(svg, repr_inputs) for svg in repr_svgs)
            ):
                _stage(
                    _plot_cmd(repr_plot_models) + ["--skip-leaderboard"],
                    f"{name} plot representations",
                )
            else:
                logger.info(f"[{name} plot representations] up to date")


def main() -> None:
    """Run the pipeline stage defined in the YAML config."""
    setup_logging()
    args = _parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    defaults = cfg.get("defaults", {})
    plot_models = cfg.get("plot_models")
    repr_plot_models = cfg.get("representation_plot_models")
    heldout_models = cfg.get("heldout_models")
    parallel = bool(cfg.get("parallel", False))
    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    try:
        datasets = _resolve_datasets(
            cfg, data_dir, args.dataset, heldout=heldout_models is not None
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    if not datasets:
        logger.error("No datasets configured or split artifacts found")
        sys.exit(1)

    def _on_sigint(*_):
        _kill_active()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        for name, dcfg in datasets.items():
            dcfg["_config_path"] = args.config
            run_dataset(
                name,
                dcfg,
                data_dir,
                figures_dir,
                args.force,
                defaults,
                plot_models,
                repr_plot_models,
                heldout_models,
                parallel,
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted; killed running stages")
        sys.exit(130)

    logger.info("Pipeline stage finished")


if __name__ == "__main__":
    main()
