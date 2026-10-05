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
    python scripts/run_pipeline.py \
        --config configs/screening.yaml --dataset GNEtolC
    python scripts/run_pipeline.py \
        --config configs/representation.yaml --dataset GNEtolC
    python scripts/run_pipeline.py \
        --config configs/heldout.yaml --dataset GNEtolC
    python scripts/run_pipeline.py \
        --config configs/screening.yaml --dataset GNEtolC --force
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from loguru import logger

CPU_SLOTS = 2


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_cv_comparison.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"


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


# --- process management ----------------------------------------------------

_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


def _run(cmd: list[str], env: dict | None = None) -> int:
    proc = subprocess.Popen(cmd, env=env)
    with _ACTIVE_LOCK:
        _ACTIVE.add(proc)
    try:
        return proc.wait()
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(proc)


def _kill_active() -> None:
    with _ACTIVE_LOCK:
        for proc in list(_ACTIVE):
            proc.terminate()


def _interpreter(env: str) -> list[str]:
    micromamba = shutil.which("micromamba") or "/home/bruno/bin/micromamba"
    return [micromamba, "run", "-n", env, "python"]


class _Scheduler:
    """Bound concurrency: one stage at a time per GPU, CPU_SLOTS on CPU."""

    def __init__(self, gpus: list[int]):
        self.gpu_locks = {g: threading.Lock() for g in gpus}
        self.cpu = threading.Semaphore(CPU_SLOTS)

    def run(self, cmd: list[str], gpu: int | None) -> int:
        if gpu is None:
            with self.cpu:
                return _run(cmd)
        with self.gpu_locks[gpu]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            return _run(cmd, env)


# --- stage helpers ---------------------------------------------------------


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
    cmd = _interpreter(mcfg["env"]) + [
        str(TRAIN_SCRIPT),
        "--dataset",
        dataset,
        "--data-dir",
        data_dir,
        "--model",
        model,
    ]
    if mcfg.get("representation"):
        cmd += ["--representation", mcfg["representation"]]
    if mcfg.get("model") == "tabpfn" and mcfg.get("version"):
        cmd += ["--tabpfn-version", mcfg["version"]]
    if tune:
        cmd += ["--tune", "--n-trials", str(defaults.get("n_trials", 10))]
        if tune_only:
            cmd += ["--tune-only"]
    if cv:
        cmd += ["--cv"]
    if mcfg.get("args"):
        cmd += shlex.split(mcfg["args"])
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
        str(EVALUATE_SCRIPT),
        "--dataset",
        dataset,
        "--test-full-path",
        str(Path(data_dir) / "splits" / f"{dataset}_test.csv"),
        "--output-dir",
        str(Path(data_dir) / "evaluations" / dataset),
        "--figures-dir",
        figures_dir,
        "--runs",
    ]
    cmd += [f"{model}={path}" for model, path in runs.items()]
    return cmd


def _stage(
    sched: _Scheduler,
    cmd: list[str],
    label: str,
    gpu: int | None,
) -> bool:
    """Run a stage through the scheduler; warn and continue on failure."""
    logger.info(f"[{label}] {' '.join(cmd)} (gpu={gpu})")
    rc = sched.run(cmd, gpu)
    if rc != 0:
        logger.error(f"[{label}] failed with exit code {rc}")
        return False
    logger.success(f"[{label}] done")
    return True


def _stale(golden: Path, inputs: list[Path]) -> bool:
    """Whether any input is newer than the golden output."""
    return any(
        path.exists() and path.stat().st_mtime > golden.stat().st_mtime
        for path in inputs
    )


# --- dataset pipeline ------------------------------------------------------


def _run_heldout(
    name: str,
    dcfg: dict,
    data_dir: Path,
    figures_dir: Path,
    sched: _Scheduler,
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
        if force or not test_csv.exists():
            # No --tune and no --cv: reuse the tuned params.json and predict.
            if not _stage(
                sched,
                _model_command(
                    mcfg.get("model", model_key.split("_")[0]),
                    mcfg,
                    name,
                    str(data_dir),
                    defaults,
                ),
                f"{name}/{model_key} heldout",
                mcfg.get("gpu"),
            ):
                continue
        else:
            logger.info(f"[{name}/{model_key} heldout] up to date")
        test_outputs[model_key] = test_csv

    if not test_outputs:
        logger.warning(f"[{name} heldout] no predictions to evaluate")
        return

    _stage(
        sched,
        _evaluate_command(name, test_outputs, str(data_dir), str(figures_dir)),
        f"{name} evaluate",
        None,
    )


def run_dataset(
    name: str,
    dcfg: dict,
    data_dir: Path,
    figures_dir: Path,
    sched: _Scheduler,
    force: bool,
    defaults: dict,
    plot_models: list[str] | None,
    repr_plot_models: list[str] | None,
    heldout_models: list[str] | None = None,
) -> None:
    """Run all model + plot stages for one dataset."""
    splits = data_dir / "splits"
    runs = data_dir / "runs" / name

    if heldout_models:
        if not (splits / f"{name}_test.csv").exists():
            logger.warning(
                f"[{name}] missing test split — run "
                "`python scripts/split_datasets.py` first; skipping dataset"
            )
            return
        _run_heldout(
            name,
            dcfg,
            data_dir,
            figures_dir,
            sched,
            force,
            defaults,
            heldout_models,
        )
        return

    if not (splits / f"{name}_cv_folds.csv").exists():
        logger.warning(
            f"[{name}] missing CV folds CSV — run "
            "`python scripts/split_datasets.py` first; skipping dataset"
        )
        return

    cv_outputs: dict[str, Path] = {}

    for model_key, mcfg in dcfg["models"].items():
        run_name = model_key
        cv_csv = runs / run_name / "predictions" / "cv.csv"
        params = runs / run_name / "params.json"

        if mcfg.get("tune") and mcfg.get("cv_only"):
            if force or not params.exists():
                if not _stage(
                    sched,
                    _model_command(
                        mcfg.get("model", model_key.split("_")[0]),
                        mcfg,
                        name,
                        str(data_dir),
                        defaults,
                        cv=False,
                        tune=True,
                        tune_only=True,
                    ),
                    f"{name}/{run_name} tune",
                    mcfg.get("gpu"),
                ):
                    continue

        if force or not cv_csv.exists():
            model = mcfg.get("model", model_key.split("_")[0])
            if _stage(
                sched,
                _model_command(
                    model,
                    mcfg,
                    name,
                    str(data_dir),
                    defaults,
                    cv=True,
                    tune=False,
                ),
                f"{name}/{run_name} cv",
                mcfg.get("gpu"),
            ):
                cv_outputs[run_name] = cv_csv
        else:
            logger.info(f"[{name}/{run_name} cv] up to date")
            cv_outputs[run_name] = cv_csv

    if not cv_outputs:
        return

    def _plot_cmd(models: list[str]) -> list[str]:
        return _interpreter("brood") + [
            str(PLOT_SCRIPT),
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
            svg = cv_figures_dir / f"{name}_cv_ap.svg"
            if force or not svg.exists() or _stale(svg, leaderboard_inputs):
                cmd = _plot_cmd(plot_models)
                if repr_plot_models is None:
                    cmd += ["--skip-representation"]
                _stage(sched, cmd, f"{name} plot", None)
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
                cv_figures_dir / f"{name}_cv_representation.svg",
                cv_figures_dir / f"{name}_cv_representation_ap.svg",
            ]
            if (
                force
                or any(not svg.exists() for svg in repr_svgs)
                or any(_stale(svg, repr_inputs) for svg in repr_svgs)
            ):
                # Figure 1 belongs to the screening stage: without
                # --skip-leaderboard this stage would redraw the leaderboard
                # from representation models and overwrite it.
                _stage(
                    sched,
                    _plot_cmd(repr_plot_models) + ["--skip-leaderboard"],
                    f"{name} plot representations",
                    None,
                )
            else:
                logger.info(f"[{name} plot representations] up to date")


def main() -> None:
    """Run the pipeline stage defined in the YAML config."""
    args = _parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    defaults = cfg.get("defaults", {})
    plot_models = cfg.get("plot_models")
    repr_plot_models = cfg.get("representation_plot_models")
    heldout_models = cfg.get("heldout_models")
    datasets = cfg["datasets"]
    if args.dataset:
        if args.dataset not in datasets:
            logger.error(f"Unknown dataset {args.dataset}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}

    data_dir = Path(args.data_dir)
    figures_dir = Path(args.figures_dir)
    gpus = sorted(
        {
            mcfg["gpu"]
            for dcfg in datasets.values()
            for mcfg in dcfg["models"].values()
            if "gpu" in mcfg
        }
    )
    sched = _Scheduler(gpus)

    def _on_sigint(*_):
        _kill_active()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        for name, dcfg in datasets.items():
            run_dataset(
                name,
                dcfg,
                data_dir,
                figures_dir,
                sched,
                args.force,
                defaults,
                plot_models,
                repr_plot_models,
                heldout_models,
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted; killed running stages")
        sys.exit(130)

    logger.success("Pipeline stage finished")


if __name__ == "__main__":
    main()
