"""Tests for the pipeline orchestrator (brood/pipeline/runner.py)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from brood.data.splitter import assign_cv_folds

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"

DIVERSE_SMILES = [
    "C1CC1",
    "C1CC1C",
    "C1CCC1",
    "C1CCC1C",
    "C1CCCC1",
    "C1CCCC1C",
    "C1CCCCC1",
    "C1CCCCC1C",
    "C1CCCCCC1",
    "C1CCCCCC1C",
    "C1CCCCC2CCCCC12",
    "C1CCCCC2CCCCC1C2",
]

# 120 unique valid SMILES: 10 ring sizes x 12 substituent-chain lengths, so
# the molecules also span 10 distinct Murcko scaffolds for the scaffold split.
# SMILES must be unique: predictions are joined onto molecules by SMILES, so
# repeated keys would inflate the merged table instead of failing.
UNIQUE_SMILES = [
    "C1" + "C" * (3 + i % 10) + "1" + "C" * (i // 10) for i in range(120)
]


@pytest.fixture
def pipeline_root(tmp_path):
    """tmp data-dir + config with two cheap models (no tuning, no GPU)."""
    data = tmp_path / "data"
    splits = data / "splits"
    splits.mkdir(parents=True)
    smis = UNIQUE_SMILES
    y = [i % 2 for i in range(len(smis))]
    train = pd.DataFrame(
        {"standardized_smiles": smis[:100], "target": y[:100]}
    )
    test = pd.DataFrame(
        {
            "standardized_smiles": smis[100:],
            "target": y[100:],
            "distance_to_train": [0.5] * 20,
        }
    )
    train.to_csv(splits / "demo_train.csv", index=False)
    test.to_csv(splits / "demo_test.csv", index=False)
    assign_cv_folds(train, n_splits=3).to_csv(
        splits / "demo_cv_folds.csv", index=False
    )

    config = {
        "defaults": {"n_trials": 5},
        "datasets": {
            "demo": {
                "models": {
                    "similarity": {"env": "brood", "cv_only": True},
                    "rf_morgan2048": {
                        "env": "brood",
                        "model": "rf",
                        "representation": "morgan2048",
                        "cv_only": True,
                    },
                }
            }
        },
    }
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def _make_cv_only(root: Path, representation_plot_models: list[str] | None):
    """Rewrite the config so both models are CV-only."""
    path = root / "pipeline.yaml"
    config = yaml.safe_load(path.read_text())
    config.pop("representation_plot_models", None)
    if representation_plot_models is not None:
        config["representation_plot_models"] = representation_plot_models
    path.write_text(yaml.safe_dump(config))


def _run_pipeline(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(RUN_PIPELINE),
            "--config",
            str(root / "pipeline.yaml"),
            "--data-dir",
            str(root / "data"),
            "--figures-dir",
            str(root / "figures"),
            *args,
        ],
        capture_output=True,
        text=True,
    )


def _log(res: subprocess.CompletedProcess) -> str:
    return res.stdout + res.stderr


def test_pipeline_runs_all_stages(pipeline_root):
    """Split-ready data → CV for each model; rerun executes nothing."""
    data = pipeline_root / "data"
    res = _run_pipeline(pipeline_root)
    assert res.returncode == 0, res.stderr

    runs = data / "runs" / "demo"
    assert (runs / "rf_morgan2048" / "predictions" / "cv.csv").exists()
    assert (runs / "similarity" / "predictions" / "cv.csv").exists()

    res2 = _run_pipeline(pipeline_root)
    assert res2.returncode == 0, res2.stderr
    assert "up to date" in _log(res2)


def test_failed_model_does_not_block_others(pipeline_root):
    """A model that dies mid-run warns and lets the rest finish."""
    data = pipeline_root / "data"
    config = yaml.safe_load((pipeline_root / "pipeline.yaml").read_text())
    config["datasets"]["demo"]["models"]["similarity"]["model"] = (
        "does_not_exist"
    )
    (pipeline_root / "pipeline.yaml").write_text(yaml.safe_dump(config))

    res = _run_pipeline(pipeline_root)

    assert (
        data / "runs" / "demo" / "rf_morgan2048" / "predictions" / "cv.csv"
    ).exists()
    assert "failed with exit code" in _log(res)


def test_representation_plot_stage_runs_when_configured(pipeline_root):
    """A configured representation_plot_models adds a second plot stage."""
    _make_cv_only(pipeline_root, ["similarity", "rf_morgan2048"])
    res = _run_pipeline(pipeline_root)
    assert res.returncode == 0, res.stderr
    assert "plot representations" in _log(res)


def test_representation_plot_stage_absent_without_the_key(pipeline_root):
    """Screening-only configs emit the leaderboard and nothing more."""
    _make_cv_only(pipeline_root, None)
    res = _run_pipeline(pipeline_root)
    assert res.returncode == 0, res.stderr
    assert "plot representations" not in _log(res)


def test_representation_stage_does_not_overwrite_the_leaderboard(
    pipeline_root,
):
    """Regression: the representation figures must not clobber figure 1.

    All three figures land in ``<dataset>/cv``, so the representation stage has
    to claim only its own two. A sentinel leaderboard stands in for the
    screening stage's output and must survive untouched.
    """
    figures = pipeline_root / "figures" / "demo" / "cv"
    figures.mkdir(parents=True)
    leaderboard = figures / "demo_cv_ap.svg"
    leaderboard.write_text("<svg>screening leaderboard</svg>")

    _make_cv_only(pipeline_root, ["similarity", "rf_morgan2048"])
    res = _run_pipeline(pipeline_root)
    log = _log(res)

    assert res.returncode == 0, res.stderr
    # The plot stage must actually run, or the sentinel proves nothing.
    assert "plot representations" in log
    assert "--skip-leaderboard" in log
    assert (figures / "demo_cv_representation_ap.svg").exists()
    assert leaderboard.read_text() == "<svg>screening leaderboard</svg>"


def test_screening_stage_leaves_representation_figures_absent(pipeline_root):
    """Screening owns figure 1 alone: it must not draw the other two."""
    path = pipeline_root / "pipeline.yaml"
    config = yaml.safe_load(path.read_text())
    config.pop("representation_plot_models", None)
    config["plot_models"] = ["similarity", "rf_morgan2048"]
    path.write_text(yaml.safe_dump(config))

    res = _run_pipeline(pipeline_root)
    log = _log(res)

    assert res.returncode == 0, res.stderr
    assert "--skip-representation" in log
    figures = pipeline_root / "figures" / "demo" / "cv"
    assert (figures / "demo_cv_ap.svg").exists()
    assert not (figures / "demo_cv_representation_ap.svg").exists()


def _make_heldout(root: Path, models: list[str]) -> None:
    """Rewrite the config as a held-out stage over *models*.

    Keeps whatever the fixture already declared for a run (e.g. ``similarity``
    has no ``model`` key) and synthesises an entry only for a new name.
    """
    path = root / "pipeline.yaml"
    config = yaml.safe_load(path.read_text())
    config["plot_models"] = None
    config["representation_plot_models"] = None
    config["heldout_models"] = models
    declared = config["datasets"]["demo"]["models"]
    for name in models:
        declared.setdefault(
            name,
            {
                "env": "brood",
                "model": "rf",
                "representation": "morgan2048",
            },
        )
    path.write_text(yaml.safe_dump(config))


def test_heldout_stage_predicts_then_evaluates(pipeline_root):
    """Held-out runs write test.csv, then one evaluate stage scores them."""
    _make_heldout(pipeline_root, ["rf_morgan2048", "similarity"])
    res = _run_pipeline(pipeline_root)
    log = _log(res)

    assert res.returncode == 0, res.stderr
    runs = pipeline_root / "data" / "runs" / "demo"
    assert (runs / "rf_morgan2048" / "predictions" / "test.csv").exists()
    assert (runs / "similarity" / "predictions" / "test.csv").exists()

    # The held-out stage must not launch CV work or CV plots.
    assert "--cv" not in log
    assert "plot representations" not in log

    assert "evaluate" in log
    evaluations = pipeline_root / "data" / "evaluations" / "demo"
    assert (evaluations / "metrics.csv").exists()
    assert (evaluations / "merged_results.csv").exists()

    figures = pipeline_root / "figures" / "demo" / "heldout"
    assert (figures / "demo_heldout_pr.svg").exists()
    assert (figures / "rf_morgan2048_distance_vs_probability.svg").exists()


def test_heldout_stage_resumes_without_repredicting(pipeline_root):
    """Existing test.csv files are reused, not recomputed."""
    _make_heldout(pipeline_root, ["rf_morgan2048"])
    first = _run_pipeline(pipeline_root)
    assert first.returncode == 0, first.stderr

    second = _run_pipeline(pipeline_root)
    log = _log(second)

    assert second.returncode == 0, second.stderr
    assert "rf_morgan2048 heldout] up to date" in log


def test_heldout_stage_warns_on_an_undeclared_run(pipeline_root):
    """A run named only in heldout_models is skipped loudly, not silently."""
    _make_heldout(pipeline_root, ["rf_morgan2048", "not_declared"])
    path = pipeline_root / "pipeline.yaml"
    config = yaml.safe_load(path.read_text())
    del config["datasets"]["demo"]["models"]["not_declared"]
    path.write_text(yaml.safe_dump(config))

    res = _run_pipeline(pipeline_root)

    assert res.returncode == 0, res.stderr
    assert "not_declared is not declared" in _log(res)
    runs = pipeline_root / "data" / "runs" / "demo"
    assert not (runs / "not_declared").exists()
