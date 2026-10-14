"""Tests for the per-fold class-balance split diagnostic."""

import pandas as pd

from brood.plotting.split_diagnostics import plot_cv_class_balance


def test_class_balance_figure_is_written(tmp_path):
    """The diagnostic renders one SVG under <dataset>/splits."""
    cv_df = pd.DataFrame(
        {
            "repeat": [0, 0, 0, 0, 1, 1, 1, 1],
            "fold": [0, 0, 1, 1, 0, 0, 1, 1],
            "target": [1, 0, 0, 0, 1, 1, 0, 0],
        }
    )

    path = plot_cv_class_balance(cv_df, "demo", tmp_path)

    assert path is not None
    assert path == tmp_path / "demo" / "splits" / "demo_cv_class_balance.svg"
    assert path.exists()
    assert path.read_text().lstrip().startswith("<?xml")


def test_class_balance_without_repeat(tmp_path):
    """Single-split folds still render (repeat defaults to 0)."""
    cv_df = pd.DataFrame({"fold": [0, 0, 1, 1], "target": [1, 0, 1, 0]})

    path = plot_cv_class_balance(cv_df, "demo", tmp_path)

    assert path is not None and path.exists()
