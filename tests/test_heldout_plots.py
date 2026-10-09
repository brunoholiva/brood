"""Tests for the held-out figures (brood/plotting/heldout.py)."""

import pandas as pd
import pytest

from brood.plotting.heldout import (
    plot_distance_vs_probability,
    plot_pr_curves,
    run_color,
)


@pytest.fixture
def merged():
    """A held-out table with two runs, two actives, and one unscored run."""
    return pd.DataFrame(
        {
            "standardized_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC", "C6"],
            "target": [1, 0, 1, 0, 0, 0],
            "distance_to_train": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "good_predicted_probability": [0.9, 0.1, 0.8, 0.2, 0.05, 0.02],
            "flat_predicted_probability": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        }
    )


def test_pr_curves_cover_every_run(merged, tmp_path):
    """One curve per run, in the shared SVG style."""
    path = plot_pr_curves(
        merged,
        {
            "rf_maccs": "good_predicted_probability",
            "tabpfn_maccs": "flat_predicted_probability",
        },
        "demo",
        tmp_path,
    )
    assert path.name == "demo_heldout_pr.svg"
    assert path.exists()
    svg = path.read_text()
    assert "rf_maccs" in svg
    assert "tabpfn_maccs" in svg


def test_pr_curves_skip_a_run_without_actives_but_keep_the_others(
    merged, tmp_path
):
    """One unusable run must not hide the rest."""
    merged["target"] = 0
    merged.loc[0, "target"] = 1
    merged["dead_predicted_probability"] = float("nan")

    path = plot_pr_curves(
        merged,
        {
            "rf_maccs": "dead_predicted_probability",
            "tabpfn_maccs": "flat_predicted_probability",
        },
        "demo",
        tmp_path,
    )
    svg = path.read_text()
    assert "rf_maccs" not in svg
    assert "tabpfn_maccs" in svg


def test_pr_curves_fail_loudly_when_nothing_is_usable(merged, tmp_path):
    """No run has an active: raise instead of drawing a misleading figure."""
    merged["target"] = 0
    with pytest.raises(ValueError, match="cannot draw"):
        plot_pr_curves(
            merged,
            {"rf_maccs": "good_predicted_probability"},
            "demo",
            tmp_path,
        )


def test_pr_curves_ignore_null_scores(merged, tmp_path):
    """A run that scored only part of the test set still plots."""
    merged.loc[0, "good_predicted_probability"] = float("nan")
    path = plot_pr_curves(
        merged, {"rf_maccs": "good_predicted_probability"}, "demo", tmp_path
    )
    assert "n=5" in path.read_text()


def test_distance_plot_marks_actives_and_inactives(merged, tmp_path):
    """Per-run SVG named after the run, with both classes in the legend."""
    path = plot_distance_vs_probability(
        merged, "rf_maccs", "good_predicted_probability", tmp_path
    )
    assert path.name == "rf_maccs_distance_vs_probability.svg"
    svg = path.read_text()
    assert "active (n=2)" in svg
    assert "inactive (n=4)" in svg


def test_distance_plot_rejects_an_all_null_run(merged, tmp_path):
    """No complete rows is an error, not a blank figure."""
    merged["good_predicted_probability"] = float("nan")
    with pytest.raises(ValueError, match="no complete held-out rows"):
        plot_distance_vs_probability(
            merged, "rf_maccs", "good_predicted_probability", tmp_path
        )


def test_run_color_is_stable_and_wraps():
    """Colour depends on position only, so a panel never shifts colours."""
    assert run_color(0) == run_color(0)
    assert run_color(0) != run_color(1)
    assert run_color(0) == run_color(6)
