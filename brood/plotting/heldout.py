"""Held-out test-set figures.

Two figures over the merged prediction table produced by
:mod:`brood.cli.evaluate`:

1. ``<dataset>_heldout_pr.svg`` - precision-recall curves, one per run, on a
   single panel so the runs are directly comparable.
2. ``<run>_distance_vs_probability.svg`` - predicted probability against
   distance to the training set, actives in colour and inactives in grey, one
   panel per run. Extrapolation shows up as confident actives far from the
   training set, which is the failure mode the distance bins exist to catch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from brood.plotting.style import (
    FONTSIZE_SMALL,
    FONTSIZE_TITLE,
    LEARNER_COLORS,
    decorate_axis,
    run_label,
    run_learner,
    save_fig,
    set_axes,
    use_style,
)

logger = logging.getLogger(__name__)

# Must precede every ``plt.subplots()`` call: seaborn's set_style only reaches
# axes created after it runs.
use_style()

#: Colour-blind-safe pair for the active/inactive split in the scatter plots.
ACTIVE_COLOR = "#d62728"
INACTIVE_COLOR = "#7f8c8d"

#: Distinct hues for the PR curves, one per run in a comparison panel.
_CURVE_COLORS = (
    "#0173b2",
    "#de8f05",
    "#029e73",
    "#d55e00",
    "#cc78bc",
    "#ca9161",
)


def run_color(index: int) -> str:
    """Stable colour for the *index*-th run in a multi-run panel."""
    return _CURVE_COLORS[index % len(_CURVE_COLORS)]


def curve_colors(runs: list[str]) -> list[str]:
    """Colours for a panel of runs, one per run.

    Runs are coloured by model, so the same model keeps the same colour in
    every figure. A panel whose runs share a model, or that holds an
    unknown model, falls back to the positional palette: two curves in one
    panel must never get the same colour.
    """
    learners = [run_learner(run) for run in runs]
    if len(set(learners)) < len(learners) or "other" in learners:
        return [run_color(index) for index in range(len(runs))]
    return [LEARNER_COLORS[learner] for learner in learners]


def plot_pr_curves(
    merged: pd.DataFrame,
    score_cols: dict[str, str],
    dataset: str,
    output_dir: Path,
    label_col: str = "target",
) -> Path:
    """Precision-recall curves for every run, overlaid on one panel.

    ``score_cols`` maps run name to its score column in ``merged``. Runs with
    no scored actives are skipped with a warning rather than raising: one run
    that predicted nothing usable should not hide the others.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged test + predictions frame.
    score_cols : dict[str, str]
        Mapping from run name to score column name.
    dataset : str
        Dataset name for the title and filename.
    output_dir : Path
        Directory to write the SVG.
    label_col : str
        Column with true binary labels.

    Returns
    -------
    Path
        Path to the saved SVG.

    Raises
    ------
    ValueError
        If no run has both scores and actives.
    """
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)

    colors = curve_colors(list(score_cols))
    drawn = 0
    for index, (run, col) in enumerate(score_cols.items()):
        usable = merged[merged[col].notna() & merged[label_col].notna()]
        if usable.empty or usable[label_col].sum() == 0:
            logger.warning(
                f"{run}: no scored actives on the held-out set; "
                "skipping its PR curve"
            )
            continue

        y_true = usable[label_col].to_numpy()
        scores = usable[col].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(
            recall,
            precision,
            color=colors[index],
            lw=2.0,
            label=f"{run_label(run)}  AP={ap:.3f} (n={len(usable)})",
        )
        drawn += 1

    if not drawn:
        plt.close(fig)
        raise ValueError(
            "No run had both scores and actives on the held-out set; "
            "cannot draw precision-recall curves"
        )

    # Prevalence reference: the precision a random ranker would reach.
    prevalence = float(merged[label_col].mean())
    ax.axhline(
        prevalence,
        ls="--",
        color="#999999",
        lw=1.2,
        label=f"prevalence={prevalence:.3f}",
    )

    decorate_axis(ax)
    set_axes(ax, xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left", fontsize=FONTSIZE_SMALL)
    ax.set_title(dataset, fontsize=FONTSIZE_TITLE)

    path = Path(output_dir) / f"{dataset}_heldout_pr.svg"
    save_fig(fig, path)
    logger.info(f"PR curves saved to {path}")
    return path


def plot_distance_vs_probability(
    merged: pd.DataFrame,
    run: str,
    score_col: str,
    output_dir: Path,
    label_col: str = "target",
    distance_col: str = "distance_to_train",
) -> Path:
    """Predicted probability against distance to the training set.

    Actives are drawn in colour and inactives in grey, so the eye picks out
    where the confident actives actually sit. Inactives are drawn first so the
    actives land on top of the grey cloud.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged test + predictions frame.
    run : str
        Run name for the title and filename.
    score_col : str
        Column with predicted probabilities.
    output_dir : Path
        Directory to write the SVG.
    label_col : str
        Column with true binary labels.
    distance_col : str
        Column with distance-to-train values.

    Returns
    -------
    Path
        Path to the saved SVG.
    """
    frame = merged[[distance_col, score_col, label_col]].dropna()
    if frame.empty:
        raise ValueError(f"{run}: no complete held-out rows to plot")

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    is_active = frame[label_col].to_numpy() == 1
    x = frame[distance_col].to_numpy()
    y = frame[score_col].to_numpy()

    ax.scatter(
        x[~is_active],
        y[~is_active],
        s=14,
        color=INACTIVE_COLOR,
        alpha=0.30,
        linewidths=0,
        label=f"inactive (n={(~is_active).sum()})",
        zorder=2,
    )
    ax.scatter(
        x[is_active],
        y[is_active],
        s=22,
        color=ACTIVE_COLOR,
        alpha=0.75,
        linewidths=0,
        label=f"active (n={is_active.sum()})",
        zorder=3,
    )

    decorate_axis(ax)
    set_axes(
        ax,
        xlim=(0, float(x.max()) * 1.02),
        ylim=(0, 1),
        xlabel="Distance to training set (mean 5-NN Tanimoto)",
        ylabel="Predicted probability",
    )
    ax.legend(loc="upper right", fontsize=FONTSIZE_SMALL)
    ax.set_title(run_label(run), fontsize=FONTSIZE_TITLE)

    path = Path(output_dir) / f"{run}_distance_vs_probability.svg"
    save_fig(fig, path)
    logger.info(f"{run}: distance plot saved to {path}")
    return path
