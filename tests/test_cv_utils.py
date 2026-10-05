"""Tests for scaffold-CV evaluation helpers."""

import json

import pandas as pd
import pytest

from brood.cv.loop import load_params, run_cv_model
from brood.data.splitter import assign_cv_folds


@pytest.fixture
def cv_folds_dir(tmp_path):
    """tmp_path/data with a 3-fold, 6-scaffold CV folds CSV."""
    smis = [
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
    df = pd.DataFrame({"standardized_smiles": smis, "target": [0, 1] * 6})
    cv = assign_cv_folds(df, n_splits=3)
    splits = tmp_path / "splits"
    splits.mkdir()
    cv.to_csv(splits / "test_cv_folds.csv", index=False)
    return tmp_path


def test_run_cv_model_writes_rows(cv_folds_dir):
    """Trivial constant predictor still yields one row per usable fold."""

    def constant_predict(train_df, test_df):
        return pd.DataFrame(
            {
                "standardized_smiles": test_df["standardized_smiles"],
                "predicted_probability": [0.5] * len(test_df),
            }
        )

    out = run_cv_model(
        "test", cv_folds_dir, "dummy", constant_predict, n_folds=3
    )

    assert list(out.columns) == ["model", "fold", "ap", "bedroc", "brier"]
    assert len(out) == 3
    assert set(out["fold"]) == {0, 1, 2}
    reloaded = pd.read_csv(
        cv_folds_dir / "runs" / "test" / "dummy" / "metrics" / "cv.csv"
    )
    raw = pd.read_csv(
        cv_folds_dir / "runs" / "test" / "dummy" / "predictions" / "cv.csv"
    )
    assert (
        len(raw)
        == len(
            cv_folds_dir.joinpath("splits", "test_cv_folds.csv")
            .read_text()
            .splitlines()
        )
        - 1
    )
    pd.testing.assert_frame_equal(out, reloaded)


def test_load_params_overrides_defaults(cv_folds_dir):
    """Saved tuned params override defaults when present."""
    preds_dir = cv_folds_dir / "predictions" / "test"
    preds_dir.mkdir(parents=True)
    (preds_dir / "rf_params.json").write_text(json.dumps({"kbest__k": 42}))

    params = load_params(
        cv_folds_dir, "test", "rf", {"kbest__k": 512, "var__threshold": 0.01}
    )

    assert params["kbest__k"] == 42
    assert params["var__threshold"] == 0.01


def test_load_params_ignores_nested_cv_params(cv_folds_dir):
    """Outer-fold params must not be used as final model params."""
    run_dir = cv_folds_dir / "runs" / "test" / "rf"
    run_dir.mkdir(parents=True)
    (run_dir / "params.json").write_text(
        json.dumps({"0": {"rf__n_estimators": 10}})
    )

    params = load_params(
        cv_folds_dir,
        "test",
        "rf",
        {"rf__n_estimators": 100},
    )

    assert params == {"rf__n_estimators": 100}
