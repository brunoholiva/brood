"""Tests for dataset normalization."""

import pandas as pd

from brood.data.report import normalize_dataset


def test_normalize_dataset_canonicalizes_and_reports_quality():
    data = pd.DataFrame(
        {
            "smiles": ["OCC", "CCO", None, "not-smiles"],
            "label": [1, 1, 0, 0],
        }
    )

    normalized, report = normalize_dataset(data, "smiles", "label")

    assert list(normalized["standardized_smiles"]) == ["CCO"]
    assert report["missing_smiles"] == 1
    assert report["invalid_smiles"] == 1
    assert report["duplicate_rows"] == 1


def test_conflicting_duplicate_labels_are_removed():
    data = pd.DataFrame(
        {"standardized_smiles": ["CCO", "OCC"], "target": [0, 1]}
    )

    normalized, report = normalize_dataset(data)

    assert normalized.empty
    assert report["conflicting_labels"] == 1
