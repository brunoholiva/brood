#!/usr/bin/env python3
"""Tukey HSD CV comparison of model ranking metrics.

Reads ``data/cv/<dataset>/*.csv`` (columns: model, fold, ap, bedroc), plots
each model's mean per-fold metric with Tukey-adjusted confidence intervals,
colored by Tukey HSD significance
vs the best model (alpha=0.05): best=blue, not significantly different=grey,
significantly different=red. Dashed vertical lines bound the best model's CI.

Usage
-----
    python scripts/plot_cv_comparison.py --dataset stokes
    python scripts/plot_cv_comparison.py --dataset stokes --output-dir figures
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BEST_COLOR = "#1f77b4"
EQUAL_COLOR = "#7f7f7f"
DIFF_COLOR = "#d62728"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--data-dir", default="data", help="Data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Output directory for the SVG (default: figures)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Tukey HSD significance level (default: 0.05)",
    )
    parser.add_argument(
        "--metric",
        choices=["ap", "bedroc", "both"],
        default="both",
        help="Metric to plot (default: both)",
    )
    return parser.parse_args()


def _load_cv_ap(cv_dir: Path) -> pd.DataFrame:
    files = sorted(cv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CV result CSVs found in {cv_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _tukey_reject_map(res, groups_unique) -> dict[tuple[str, str], bool]:
    """Map ordered group pairs to their Tukey HSD rejection verdict."""
    pairs = list(itertools.combinations(groups_unique, 2))
    return {
        tuple(sorted(p)): bool(reject) for p, reject in zip(pairs, res.reject)
    }


def _tukey_mean_intervals(
    df: pd.DataFrame, metric: str, alpha: float
) -> dict[str, tuple[float, float]]:
    """Return simultaneous Tukey-adjusted intervals for group means."""
    grouped = df.groupby("model")[metric]
    means = grouped.mean()
    residual_df = len(df) - len(means)
    residual_ss = sum(
        ((values - values.mean()) ** 2).sum() for _, values in grouped
    )
    mse = residual_ss / max(residual_df, 1)
    critical = qsturng(1.0 - alpha, len(means), max(residual_df, 1))
    return {
        model: (
            float(mean - critical * np.sqrt(mse / (2 * len(values)))),
            float(mean + critical * np.sqrt(mse / (2 * len(values)))),
        )
        for model, values in grouped
        for mean in [values.mean()]
    }


def _plot_metric(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    metric: str,
    alpha: float,
) -> None:
    """Render one Tukey comparison figure."""
    metric_df = df.dropna(subset=[metric]).copy()
    res = pairwise_tukeyhsd(
        metric_df[metric].to_numpy(), metric_df["model"].to_numpy(), alpha
    )
    reject_map = _tukey_reject_map(res, res.groupsunique)
    means = (
        metric_df.groupby("model")[metric].mean().sort_values(ascending=False)
    )
    intervals = _tukey_mean_intervals(metric_df, metric, alpha)

    fig, ax = plt.subplots(figsize=(7, 5))
    pos = np.arange(len(means))
    best = means.index[0]
    for model, y in zip(means.index, pos):
        low, high = intervals[model]
        color = (
            BEST_COLOR
            if model == best
            else DIFF_COLOR
            if reject_map[tuple(sorted((model, best)))]
            else EQUAL_COLOR
        )
        ax.errorbar(
            means[model],
            y,
            xerr=[[means[model] - low], [high - means[model]]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=7,
        )

    best_low, best_high = intervals[best]
    ax.axvline(best_low, ls="--", color=BEST_COLOR, alpha=0.4)
    ax.axvline(best_high, ls="--", color=BEST_COLOR, alpha=0.4)
    ax.set_yticks(pos, means.index)
    ax.set_xlabel(
        f"Mean {metric.upper()} (Tukey-adjusted CI, "
        f"n_folds={metric_df['fold'].nunique()})"
    )
    ax.set_title(
        f"{dataset}: scaffold CV {metric.upper()} comparison "
        f"(Tukey HSD, α={alpha})"
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=BEST_COLOR,
                label="best",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=EQUAL_COLOR,
                label="not significant",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=DIFF_COLOR,
                label="significantly different",
            ),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    suffix = "" if metric == "ap" else f"_{metric}"
    out_path = output_dir / f"{dataset}_cv_comparison{suffix}.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.success(f"{metric.upper()} comparison saved to {out_path}")


def main() -> None:
    """Render the CV comparison figure(s)."""
    args = _parse_args()
    cv_dir = Path(args.data_dir) / "cv" / args.dataset

    df = _load_cv_ap(cv_dir)
    metrics = ["ap", "bedroc"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        _plot_metric(
            df, args.dataset, Path(args.output_dir), metric, args.alpha
        )


if __name__ == "__main__":
    main()
