"""Tests for dataset normalization and scaffold reporting."""

import json

import pandas as pd
import pytest

from src.data_report import (
    build_dataset_report,
    normalize_dataset,
    save_dataset_report,
)


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


def test_conflicting_duplicate_labels_are_rejected():
    data = pd.DataFrame(
        {"standardized_smiles": ["CCO", "OCC"], "target": [0, 1]}
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        normalize_dataset(data)


def test_report_contains_scaffold_statistics(tmp_path):
    data = pd.DataFrame(
        {
            "standardized_smiles": ["C1CC1", "C1CC1C", "C1CCC1"],
            "target": [0, 1, 1],
        }
    )

    report = build_dataset_report(data)
    save_dataset_report(data, tmp_path)

    assert report["molecules"] == 3
    assert report["n_scaffolds"] >= 2
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "generic_scaffolds.png").exists()
    assert json.loads((tmp_path / "report.json").read_text())["molecules"] == 3
