#!/usr/bin/env python3
"""Cross-validation figures for average precision.

Reads per-fold metrics from ``data/runs/<dataset>/*/metrics/cv.csv`` and
renders three figures:

1. ``<dataset>_cv_ap.svg`` - every model's average precision, one faint dot
   per CV fold, coloured by learner family, against the max-similarity
   baseline. This is the screening leaderboard.
2. ``<dataset>_cv_representation.svg`` - average precision per molecular
   representation, for the later representation stage. Representations are
   nominal so nothing is connected across them; only the two learners at the
   same representation are joined, because that is a paired comparison.
3. ``<dataset>_cv_representation_ap.svg`` - figure 1's leaderboard applied to
   the representation runs, so every run is ranked in one place instead of
   only being read off figure 2's pairs. Same function as figure 1 with a
   different output name; it is a different run set, so its omnibus test and
   Tukey half-width are its own.

Screening passes ``--skip-representation`` and gets figure 1 alone; the
representation stage passes ``--skip-leaderboard`` and gets figures 2 and 3
alone. Each stage must name only the figure it owns, otherwise it overwrites
the other stage's figure with the wrong model set.

Folds are treated as blocks, so the Tukey HSD intervals on figure 1 use the
paired model-by-fold error term.

``--output-dir`` is the root figure directory: the dataset and ``cv`` are
appended automatically, so pass ``figures``, not ``figures/<dataset>/cv``.

Usage
-----
    brood-plot-cv --dataset stokes
    brood-plot-cv --dataset stokes --output-dir figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.stats.libqsturng import qsturng

from brood.core.logging import setup_logging
from brood.plotting.style import (
    LEARNER_COLORS,
    LEARNER_LABELS,
    decorate_axis,
    save_fig,
    set_axes,
    use_style,
)
from brood.representations import REPRESENTATION_REGISTRY

logger = logging.getLogger(__name__)

# Must precede every ``plt.subplots()`` call: seaborn's set_style only reaches
# axes created after it runs.
use_style()

#: Plot order for the representation axis (fixed, so runs stay comparable).
REPRESENTATION_ORDER = tuple(REPRESENTATION_REGISTRY)

#: Human labels for the representation axis.
REPRESENTATION_LABELS = {
    name: rep.label or name for name, rep in REPRESENTATION_REGISTRY.items()
}

#: Learners that have one run per representation and so get a line in fig. 2.
_FACTORIAL_LEARNERS = ("tabpfn", "rf")

#: Runs without a representation variant, drawn as reference lines in fig. 2.
_REFERENCE_LEARNERS = ("chemprop", "similarity")

#: Significance level for the blocked Tukey HSD intervals.
_ALPHA = 0.05


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--data-dir", default="data", help="Data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the SVGs (default: figures/<dataset>/cv)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only include these model run names in the comparison.",
    )
    parser.add_argument(
        "--skip-representation",
        action="store_true",
        help="Skip figures 2 and 3 (the representation-stage figures)",
    )
    parser.add_argument(
        "--skip-leaderboard",
        action="store_true",
        help="Skip figure 1 (the screening leaderboard)",
    )
    return parser.parse_args()


def _load_cv_metrics(
    data_dir: Path, dataset: str, models: list[str] | None = None
) -> pd.DataFrame:
    """Load per-fold CV metrics from each run's ``metrics/cv.csv``.

    These are the values :func:`brood.cv.loop.run_cv_model` already derived
    from the raw fold predictions, so the figures read one source instead of
    recomputing AP/BEDROC/Brier a second time.
    """
    files = sorted((data_dir / "runs" / dataset).glob("*/metrics/cv.csv"))
    if models is not None:
        selected = set(models)
        files = [p for p in files if p.parent.parent.name in selected]
    if not files:
        raise FileNotFoundError(
            f"No CV metrics CSVs found under {data_dir / 'runs' / dataset}"
        )
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _split_model(name: str) -> tuple[str, str]:
    """'tabpfn_morgan512' -> ('tabpfn', 'morgan512').

    Runs with no representation variant ('chemprop', 'similarity') map to
    their own learner family with an empty representation, so they keep their
    family colour instead of falling back to the unknown-run grey.
    """
    learner, _, rep = name.partition("_")
    if learner in LEARNER_COLORS:
        return learner, rep
    return "other", name


def _fold_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Fold x model matrix of one metric (incomplete folds dropped)."""
    table = (
        df.dropna(subset=[metric])
        .pivot(index="fold", columns="model", values=metric)
        .dropna()
    )
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Need at least 2 folds and 2 models")
    return table


def _warn_on_incomplete_folds(df: pd.DataFrame, table: pd.DataFrame) -> None:
    """Name any run whose folds are incomplete.

    A run with fewer folds than the rest forces ``dropna`` to discard those
    folds for *every* model, silently shrinking the comparison.
    """
    coverage = df.groupby("model")["fold"].nunique()
    expected = int(coverage.max())
    short = {model: int(n) for model, n in coverage.items() if n < expected}
    dropped = sorted(set(df["fold"]) - set(table.index))
    if short or dropped:
        detail = ", ".join(
            f"{m}={n}/{expected}" for m, n in sorted(short.items())
        )
        logger.warning(
            "Paired comparison uses only {}/{} folds across {} models; "
            "runs with incomplete folds: {}; folds dropped from every "
            "model: {}. Re-run them before reading the figure.".format(
                table.shape[0],
                expected,
                table.shape[1],
                detail or "none",
                dropped or "none",
            )
        )


def _blocked_test(
    table: pd.DataFrame, alpha: float = _ALPHA
) -> tuple[pd.Series, float, float]:
    """Blocked means, simultaneous Tukey half-width, and omnibus p-value.

    Each fold is a block shared by every model. The model-by-fold interaction
    is the error term for the repeated-measures Tukey comparison. Taking a
    sub-table (e.g. one representation's two learners) applies exactly the
    same estimator to that subset.
    """
    n_folds, n_models = table.shape

    values = table.to_numpy()
    model_means = table.mean(axis=0)
    fold_means = table.mean(axis=1).to_numpy()
    grand_mean = float(values.mean())
    residual = (
        values
        - fold_means[:, None]
        - model_means.to_numpy()[None, :]
        + grand_mean
    )
    error_df = (n_folds - 1) * (n_models - 1)
    error_mse = float((residual**2).sum() / error_df)
    critical = float(qsturng(1.0 - alpha, n_models, error_df))
    half_width = critical * np.sqrt(error_mse / n_folds) / 2

    model_ss = n_folds * ((model_means - grand_mean) ** 2).sum()
    model_mse = model_ss / (n_models - 1)
    omnibus_f = model_mse / error_mse if error_mse else np.inf
    omnibus_p = float(stats.f.sf(omnibus_f, n_models - 1, error_df))
    return model_means, float(half_width), omnibus_p


def _model_color(model: str) -> tuple:
    """Colourblind-safe colour for a run's learner family."""
    return LEARNER_COLORS.get(_split_model(model)[0], "#999999")


def _plot_leaderboard(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    metric: str,
    suffix: str | None = None,
) -> None:
    """Figures 1 and 3: mean metric per run with one dot per CV fold.

    Colour encodes the learner family. The bar through each mean is the
    simultaneous Tukey HSD half-width, so intervals overlapping the leader's
    are not significantly worse than it.
    """
    metric_df = df.dropna(subset=[metric])
    table = _fold_table(metric_df, metric)
    raw_means, half_width, omnibus_p = _blocked_test(table)
    _warn_on_incomplete_folds(df, table)
    means = raw_means.sort_values(ascending=False)

    fig, ax = plt.subplots(
        figsize=(10, 0.46 * len(means) + 2.0), constrained_layout=True
    )
    rng = np.random.default_rng(0)
    for y, model in enumerate(means.index):
        color = _model_color(model)
        vals = table[model].to_numpy()
        ax.scatter(
            vals,
            y + rng.uniform(-0.13, 0.13, len(vals)),
            s=22,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
        mean = means[model]
        ax.plot(
            [mean - half_width, mean + half_width],
            [y, y],
            color=color,
            lw=1.6,
            alpha=0.85,
            zorder=4,
        )
        ax.plot(
            [mean, mean],
            [y - 0.28, y + 0.28],
            color=color,
            lw=4.5,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.text(
            1.005,
            y,
            f"{mean:.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color="#444444",
        )

    ax.text(
        1.005,
        -0.9,
        "mean AP",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        fontweight="bold",
        color="#444444",
    )

    baseline = [m for m in means.index if _split_model(m)[0] == "similarity"]
    if baseline:
        ref = means[baseline[0]]
        ax.axvline(ref, ls="--", color=LEARNER_COLORS["similarity"], lw=1.4)
        ax.text(
            ref,
            len(means) - 0.4,
            " max-similarity",
            color=LEARNER_COLORS["similarity"],
            fontsize=9,
            va="bottom",
            ha="left",
        )

    ax.set_yticks(range(len(means)), means.index)
    ax.set_ylim(len(means) - 0.4, -1.3)
    ax.tick_params(axis="y", length=0)
    decorate_axis(ax)
    set_axes(
        ax,
        xlabel=(
            f"Average precision per CV fold (dots), mean (vertical bar) and "
            f"Tukey HSD interval (horizontal bar); n_folds={table.shape[0]}"
        ),
    )
    present = sorted({_split_model(m)[0] for m in table.columns} - {"other"})
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LEARNER_COLORS[learner],
                label=LEARNER_LABELS[learner],
            )
            for learner in present
        ],
        loc="lower right",
    )
    # Same quantity as figure 1 but a different run set, and therefore a
    # different omnibus test: mark it so the two are not confused.
    scope = " (representation stage)" if suffix else ""
    ax.set_title(
        f"{dataset}: development-set 5-fold cross-validation "
        f"average precision{scope}\n"
        f"omnibus F-test on blocked means, p={omnibus_p:.2g}, α={_ALPHA}",
        fontsize=12,
    )
    path = output_dir / f"{dataset}_cv_{suffix or metric}.svg"
    save_fig(fig, path)
    logger.info(f"{metric.upper()} leaderboard saved to {path}")


def _plot_representation(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    metric: str,
) -> None:
    """Figure 2: metric per representation, paired within each group.

    Representations are nominal, so nothing is connected across them. The
    two learners at the same representation are joined by a segment because
    that *is* a paired comparison, and its length is the learner effect. Each
    segment carries the paired difference and is drawn solid only when it
    exceeds the blocked Tukey half-width for that same pair.
    """
    metric_df = df.dropna(subset=[metric])
    table = _fold_table(metric_df, metric)
    _warn_on_incomplete_folds(df, table)
    means = table.mean(axis=0)

    pairs: dict[str, tuple[str, str]] = {}
    for rep in REPRESENTATION_ORDER:
        cols = [
            f"{learner}_{rep}"
            for learner in _FACTORIAL_LEARNERS
            if f"{learner}_{rep}" in table.columns
        ]
        if len(cols) == 2:
            pairs[rep] = (cols[0], cols[1])
    if not pairs:
        logger.warning(
            "No learner x representation pairs found; skipping figure 2"
        )
        return

    reps = list(pairs)
    xs = np.arange(len(reps))
    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    n_tied = 0

    for x, rep in zip(xs, reps):
        left, right = pairs[rep]
        _, half_width, _ = _blocked_test(table[[left, right]])
        lo, hi = float(means[left]), float(means[right])
        color = LEARNER_COLORS[_split_model(left)[0]]
        significant = abs(hi - lo) > 2 * half_width
        n_tied += not significant
        ax.plot(
            [x, x],
            [lo, hi],
            color=color,
            lw=2.4,
            ls="-" if significant else (0, (2, 2)),
            alpha=0.5,
            zorder=2,
        )
        for model, value in ((left, lo), (right, hi)):
            ax.plot(
                x,
                value,
                marker="o",
                markersize=9,
                color=_model_color(model),
                linestyle="none",
                zorder=4,
            )
        ax.text(
            x,
            max(lo, hi) + 0.004,
            f"{lo - hi:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=LEARNER_COLORS[learner],
            label=LEARNER_LABELS[learner],
        )
        for learner in _FACTORIAL_LEARNERS
    ]
    for learner in _REFERENCE_LEARNERS:
        if learner not in table.columns:
            continue
        value = means[learner]
        color = LEARNER_COLORS[learner]
        ax.axhline(value, ls="--", color=color, lw=1.4, zorder=1)
        ax.text(
            len(reps) - 0.6,
            value,
            f" {LEARNER_LABELS[learner]} {value:.3f}",
            color=color,
            fontsize=9,
            va="bottom",
            ha="right",
        )
        handles.append(
            Line2D([0], [0], color=color, ls="--", lw=1.4, label=learner)
        )
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))

    decorate_axis(ax)
    set_axes(
        ax,
        ylabel="Average precision",
        xlabel="Molecular representation (segment labelled Δ = TabPFN − RF)",
    )
    ax.set_xticks(xs, [REPRESENTATION_LABELS.get(rep, rep) for rep in reps])
    ax.set_xlim(-0.4, len(reps) - 0.6)
    ax.set_title(
        f"{dataset}: representation and learner, paired within each "
        f"group\n{metric.upper()}, n_folds={table.shape[0]}, "
        f"{len(reps) - n_tied}/{len(reps)} gaps distinguishable at "
        f"α={_ALPHA}",
        fontsize=12,
    )
    path = output_dir / f"{dataset}_cv_representation.svg"
    save_fig(fig, path)
    logger.info(f"{metric.upper()} representation figure saved to {path}")


def main() -> None:
    """Render the CV figures."""
    setup_logging()
    args = _parse_args()
    df = _load_cv_metrics(Path(args.data_dir), args.dataset, args.models)
    if "ap" not in df.columns:
        raise ValueError("No average precision available in the CV files")
    output_dir = Path(args.output_dir or "figures") / args.dataset / "cv"
    if not args.skip_leaderboard:
        _plot_leaderboard(df, args.dataset, output_dir, "ap")
    if not args.skip_representation:
        _plot_representation(df, args.dataset, output_dir, "ap")
        _plot_leaderboard(
            df,
            args.dataset,
            output_dir,
            "ap",
            suffix="representation_ap",
        )


if __name__ == "__main__":
    main()
