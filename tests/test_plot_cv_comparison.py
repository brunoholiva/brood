"""Tests for CV plots derived from raw fold predictions."""

import logging
import sys

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.libqsturng import qsturng

from brood.plotting.cv_comparison import (
    _assumptions,
    _compare,
    _fold_table,
    _load_cv_metrics,
    _model_color,
    _split_model,
    _warn_on_incomplete_folds,
    friedman_conover,
    rm_tukey_hsd,
)
from brood.plotting.style import LEARNER_COLORS


def _table(samples, models, values):
    """Build a sample x model table from a nested list of columns."""
    return pd.DataFrame(np.asarray(values).T, index=samples, columns=models)


def test_cv_metrics_are_loaded_from_run_metrics(tmp_path):
    path = tmp_path / "runs" / "demo" / "rf" / "metrics"
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["rf", "rf"],
            "repeat": [0, 0],
            "fold": [0, 1],
            "ap": [0.4, 0.6],
            "bedroc": [0.3, 0.5],
            "brier": [0.2, 0.15],
        }
    ).to_csv(path / "cv.csv", index=False)

    metrics = _load_cv_metrics(tmp_path, "demo")

    assert list(metrics.columns) == [
        "model",
        "repeat",
        "fold",
        "ap",
        "bedroc",
        "brier",
        "sample",
    ]
    assert len(metrics) == 2
    assert list(metrics["sample"]) == ["0_0", "0_1"]
    assert metrics["brier"].notna().all()


def test_rm_tukey_hsd_uses_model_by_sample_error_term():
    table = pd.DataFrame(
        {
            "a": [0.10, 0.20, 0.30, 0.40],
            "b": [0.12, 0.24, 0.28, 0.43],
            "c": [0.50, 0.44, 0.62, 0.57],
        },
        index=range(4),
    )

    comparison = rm_tukey_hsd(table)

    values = table.to_numpy()
    model_means = table.mean(axis=0).to_numpy()
    fold_means = table.mean(axis=1).to_numpy()
    residual = (
        values - fold_means[:, None] - model_means[None, :] + values.mean()
    )
    error_df = 2 * 3
    error_mse = (residual**2).sum() / error_df
    expected_q = qsturng(0.95, 3, error_df)
    expected_ci = expected_q / np.sqrt(2) * np.sqrt(error_mse / 4)

    np.testing.assert_allclose(comparison.means.to_numpy(), model_means)
    np.testing.assert_allclose(comparison.ci_half, expected_ci, rtol=2e-3)
    assert comparison.test == "parametric"
    assert comparison.omnibus_p < 0.05
    np.testing.assert_allclose(
        comparison.p_values.to_numpy(),
        comparison.p_values.to_numpy().T,
    )
    np.testing.assert_allclose(np.diag(comparison.p_values), 1.0)


def test_pair_comparison_reuses_the_full_estimator_means():
    """Fig 2 reads the pairwise p-value from the single full comparison."""
    table = pd.DataFrame(
        {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.2, 0.1, 0.4, 0.3], "c": [0.3] * 4},
        index=range(4),
    )
    full = rm_tukey_hsd(table)
    pair = rm_tukey_hsd(table[["a", "b"]])
    assert list(pair.means) == [full.means["a"], full.means["b"]]
    assert 0 <= pair.omnibus_p <= 1
    assert full.significant("a", "b") == bool(
        full.p_values.loc["a", "b"] < full.alpha
    )


def test_friedman_conover_returns_a_symmetric_p_matrix():
    table = _table(
        list(range(6)),
        ["a", "b", "c"],
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.1, 0.4, 0.3, 0.5, 0.9],
            [0.3, 0.4, 0.2, 0.6, 0.7, 0.8],
        ],
    )
    comparison = friedman_conover(table)
    assert comparison.test == "nonparametric"
    assert comparison.ci_half is None
    assert 0 <= comparison.omnibus_p <= 1
    np.testing.assert_allclose(
        comparison.p_values.to_numpy(),
        comparison.p_values.to_numpy().T,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tabpfn_morgan512", ("tabpfn", "morgan512")),
        ("rf_morgan_maccs_rdkit2d", ("rf", "morgan_maccs_rdkit2d")),
        ("chemprop", ("chemprop", "")),
        ("knn", ("knn", "")),
    ],
)
def test_split_model(name, expected):
    assert _split_model(name) == expected


def test_every_known_model_gets_its_colour():
    """Regression: a missing representation suffix used to send chemprop and
    knn to the unknown-run grey, so it looked identical in every
    figure and vanished from the leaderboard legend."""
    runs = [
        "chemprop",
        "knn",
        "tabpfn_morgan2048",
        "rf_morgan2048",
        "lgbm_morgan2048",
    ]
    colors = [_model_color(run) for run in runs]
    assert len(set(colors)) == len(runs)
    for run in runs:
        assert _model_color(run) == LEARNER_COLORS[_split_model(run)[0]]


def test_incomplete_runs_are_reported_not_silently_dropped(caplog):
    """A run missing samples/models would shrink the paired comparison."""
    metrics = pd.DataFrame(
        {
            "sample": [0, 1, 2, 3] * 2 + [0, 1],
            "model": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
            "ap": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45, 0.2, 0.3],
        }
    )

    table = _fold_table(metrics, "ap")
    with caplog.at_level(
        logging.WARNING, logger="brood.plotting.cv_comparison"
    ):
        _warn_on_incomplete_folds(metrics, table)

    assert table.shape == (2, 3)
    assert "runs with incomplete samples: c=2/4" in caplog.text
    assert "samples dropped from every model: [2, 3]" in caplog.text


def test_violated_sphericity_triggers_the_nonparametric_fallback():
    rng = np.random.default_rng(0)
    spread = np.diag([0.05, 0.2, 0.4, 0.8])
    values = (spread @ rng.normal(size=(4, 30))).T
    table = _table(list(range(30)), [f"m{i}" for i in range(4)], values.T)

    diag = _assumptions(table)
    assert diag["gg_epsilon"] < 0.75

    comparison, _ = _compare(table, "auto")
    assert comparison.test == "nonparametric"
    assert comparison.reason


def test_auto_stays_parametric_when_assumptions_hold():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(4, 40))
    table = _table(list(range(40)), [f"m{i}" for i in range(4)], values)

    comparison, _ = _compare(table, "auto")
    assert comparison.test == "parametric"

    forced, _ = _compare(table, "nonparametric")
    assert forced.test == "nonparametric"


def _render_inputs(rows):
    df = pd.DataFrame(rows)
    table = _fold_table(df, "ap")
    comparison, _ = _compare(table, "auto")
    return df, table, comparison


def test_main_renders_both_figures(tmp_path, monkeypatch):
    """Regression: main() used to crash when --chosen was not passed."""
    import brood.plotting.cv_comparison as plot_mod

    rows = [
        {"model": model, "sample": f"r0f{fold}", "ap": 0.1 * i + 0.01 * fold}
        for i, model in enumerate(
            [
                "knn",
                "lgbm_maccs",
                "lgbm_rdkit2d",
                "tabpfn_maccs",
                "tabpfn_rdkit2d",
            ]
        )
        for fold in (0, 1, 2)
    ]
    monkeypatch.setattr(
        plot_mod,
        "_load_cv_metrics",
        lambda data_dir, dataset, models=None: pd.DataFrame(rows),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_cv_comparison.py",
            "--dataset",
            "demo",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    plot_mod.main()

    out = tmp_path / "out" / "demo" / "cv"
    assert (out / "demo_cv_ap.svg").exists()
    assert (out / "demo_cv_representation.svg").exists()
    assert (out / "demo_cv_representation_ap.svg").exists()


def test_skip_leaderboard_draws_only_the_representation_figure(
    tmp_path, monkeypatch
):
    """Regression: the representation stage must not redraw figure 1.

    Both figures live in the same ``<dataset>/cv`` directory, so a stage that
    renders figure 2 without claiming figure 1 silently overwrites the
    screening leaderboard with representation models.
    """
    import brood.plotting.cv_comparison as plot_mod

    rows = [
        {"model": model, "sample": f"r0f{fold}", "ap": 0.1 * i + 0.01 * fold}
        for i, model in enumerate(
            ["knn", "lgbm_maccs", "tabpfn_maccs", "tabpfn_rdkit2d"]
        )
        for fold in (0, 1, 2)
    ]
    monkeypatch.setattr(
        plot_mod,
        "_load_cv_metrics",
        lambda data_dir, dataset, models=None: pd.DataFrame(rows),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_cv_comparison.py",
            "--dataset",
            "demo",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-leaderboard",
        ],
    )

    plot_mod.main()

    out = tmp_path / "out" / "demo" / "cv"
    assert not (out / "demo_cv_ap.svg").exists()
    assert (out / "demo_cv_representation.svg").exists()
    assert (out / "demo_cv_representation_ap.svg").exists()


def test_representation_figure_skips_one_sided_groups(tmp_path, monkeypatch):
    """A representation missing one learner is skipped, not crashed on."""
    import brood.plotting.cv_comparison as plot_mod

    rows = [
        {"model": model, "sample": f"r0f{fold}", "ap": 0.1 * i + 0.01 * fold}
        for i, model in enumerate(
            ["tabpfn_maccs", "lgbm_maccs", "tabpfn_rdkit2d"]
        )
        for fold in (0, 1, 2)
    ]
    df, table, comparison = _render_inputs(rows)
    out = tmp_path / "out"
    plot_mod._plot_representation(df, "demo", out, "ap", table, comparison)
    assert (out / "demo_cv_representation.svg").exists()


def test_representation_figure_skipped_when_no_pairs_exist(tmp_path, caplog):
    """No complete learner pair means no figure 2, without an exception."""
    import brood.plotting.cv_comparison as plot_mod

    rows = [
        {"model": model, "sample": f"r0f{fold}", "ap": 0.1 * i + 0.01 * fold}
        for i, model in enumerate(["tabpfn_maccs", "lgbm_rdkit2d"])
        for fold in (0, 1, 2)
    ]
    df, table, comparison = _render_inputs(rows)
    out = tmp_path / "out"
    with caplog.at_level(
        logging.WARNING, logger="brood.plotting.cv_comparison"
    ):
        plot_mod._plot_representation(df, "demo", out, "ap", table, comparison)
    assert not out.exists()
    assert "No learner x representation pairs" in caplog.text
