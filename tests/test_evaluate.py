"""Tests for the held-out evaluation CLI (brood/cli/evaluate.py).

The plotting itself is covered in ``tests/test_heldout_plots.py``; these tests
cover the CLI plumbing that joins predictions onto the test set.
"""

import argparse

import pandas as pd
import pytest

from brood.cli.evaluate import _merge_all_runs, _parse_runs


def _args(**overrides) -> argparse.Namespace:
    """Minimal args namespace for _merge_all_runs."""
    base = {
        "smiles_col": "standardized_smiles",
        "score_col": "predicted_probability",
        "label_col": "target",
        "distance_col": "distance_to_train",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN", "CCC"],
            "target": [0, 1, 0],
            "distance_to_train": [0.3, 0.4, 0.5],
        }
    )


def _write_preds(path, smiles, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"standardized_smiles": smiles, "predicted_probability": scores}
    ).to_csv(path, index=False)
    return path


def test_parse_runs_reads_every_pair():
    """All model=path pairs survive parsing, not just the last one."""
    runs = _parse_runs(["rf=a.csv", "tabpfn=b.csv", "chemprop=c.csv"])
    assert runs == {
        "rf": "a.csv",
        "tabpfn": "b.csv",
        "chemprop": "c.csv",
    }


@pytest.mark.parametrize("item", ["rf", "=a.csv", "rf="])
def test_parse_runs_rejects_malformed_entries(item):
    with pytest.raises(ValueError, match="Bad --runs entry"):
        _parse_runs([item])


def test_merge_all_runs_widens_one_column_per_run(test_df, tmp_path):
    """Each run gets its own <model>_predicted_probability column."""
    runs = {
        "rf": _write_preds(
            tmp_path / "rf.csv", ["CCO", "CCN", "CCC"], [0.1, 0.8, 0.2]
        ),
        "tabpfn": _write_preds(
            tmp_path / "tabpfn.csv", ["CCO", "CCN", "CCC"], [0.3, 0.9, 0.4]
        ),
    }
    merged, successful = _merge_all_runs(test_df, runs, _args())

    assert merged["rf_predicted_probability"].tolist() == [0.1, 0.8, 0.2]
    assert merged["tabpfn_predicted_probability"].tolist() == [0.3, 0.9, 0.4]
    assert len(successful) == 2
    # One row per test molecule: the join must not multiply rows.
    assert len(merged) == len(test_df)


def test_merge_all_runs_keeps_test_rows_a_run_did_not_score(test_df, tmp_path):
    """Left join: a molecule missing from a run stays, with a null score."""
    runs = {
        "rf": _write_preds(tmp_path / "rf.csv", ["CCO", "CCC"], [0.1, 0.2])
    }
    merged, successful = _merge_all_runs(test_df, runs, _args())

    assert len(merged) == len(test_df)
    assert merged["rf_predicted_probability"].isna().sum() == 1
    assert successful == [("rf", "rf_predicted_probability")]


def test_merge_all_runs_rejects_duplicate_test_smiles(tmp_path):
    """Repeated keys would inflate every metric, so this must raise."""
    duped = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCO"],
            "target": [0, 1],
            "distance_to_train": [0.3, 0.4],
        }
    )
    runs = {"rf": _write_preds(tmp_path / "rf.csv", ["CCO"], [0.5])}
    with pytest.raises(ValueError, match="repeated"):
        _merge_all_runs(duped, runs, _args())


def test_merge_all_runs_skips_a_missing_prediction_file(test_df, tmp_path):
    """A run with no file is warned about, and the others still merge."""
    runs = {
        "gone": tmp_path / "does_not_exist.csv",
        "rf": _write_preds(tmp_path / "rf.csv", ["CCO"], [0.5]),
    }
    merged, successful = _merge_all_runs(test_df, runs, _args())

    assert len(merged) == len(test_df)
    assert [name for name, _ in successful] == ["rf"]
