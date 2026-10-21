"""Does the metric change the answer? A metric battery on saved CV runs.

The screening pipeline ranks its runs by average precision. This module
re-scores the same per-fold predictions with several other metrics and
shows how much the ranking of the runs depends on the metric: one panel
per metric, the runs in a fixed order, so a run that moves when the
metric changes is visible directly.

No model is re-trained: every metric is computed from the predictions
that the CV loop already wrote to disk.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from brood.core.artifacts import prediction_dir
from brood.evaluation.metrics import METRIC_DIRECTION, sample_metrics
from brood.plotting.cv_comparison import _compare, _fold_table
from brood.plotting.style import (
    FONTSIZE,
    FONTSIZE_SMALL,
    FONTSIZE_TITLE,
    run_label,
    run_sort_key,
    save_fig,
    use_style,
)

logger = logging.getLogger(__name__)

#: Panels of the figure: metric key, panel title, colour.
PANELS = (
    ("ap", "Average precision", "#6666FF"),
    ("auc", "ROC AUC", "#FF9933"),
    ("ef1", "Enrichment factor @1%", "#7BB601"),
    ("brier", "Brier score (lower is better)", "#D4609A"),
)

#: The representation the screening stage compares its models on.
SCREENING_REPRESENTATION = "morgan2048"


def screening_candidates(data_dir: Path, dataset: str) -> list[str]:
    """List the runs the screening stage would produce.

    These are the morgan2048 models plus the model-only runs (``knn``,
    ``chemprop``), whether or not they have been run yet.
    """
    root = data_dir / "runs" / dataset
    if not root.exists():
        return []
    runs = [p.name for p in root.iterdir() if p.is_dir()]
    keep = [
        run
        for run in runs
        if run.endswith(f"_{SCREENING_REPRESENTATION}") or "_" not in run
    ]
    return sorted(keep, key=run_sort_key)


def screening_runs(data_dir: Path, dataset: str) -> list[str]:
    """List the screening runs that have finished their cross-validation.

    A run without ``predictions/cv.csv`` was never scored, so it is
    skipped instead of breaking the analysis; :func:`skipped_runs` names
    the ones that were left out.
    """
    return [
        run
        for run in screening_candidates(data_dir, dataset)
        if (prediction_dir(data_dir, dataset, run) / "cv.csv").exists()
    ]


def skipped_runs(data_dir: Path, dataset: str) -> list[str]:
    """List the screening runs that have no CV predictions yet."""
    return [
        run
        for run in screening_candidates(data_dir, dataset)
        if not (prediction_dir(data_dir, dataset, run) / "cv.csv").exists()
    ]


def load_cv_predictions(
    data_dir: Path, dataset: str, run: str
) -> pd.DataFrame:
    """Per-fold predictions of one run, without the SMILES column."""
    path = prediction_dir(data_dir, dataset, run) / "cv.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No CV predictions for {run}: {path}. "
            f"Run 'brood-train --dataset {dataset} --model <model> --cv'."
        )
    return pd.read_csv(
        path,
        usecols=["target", "repeat", "fold", "predicted_probability"],
    )


def train_prevalence(data_dir: Path, dataset: str) -> dict[tuple, float]:
    """Prevalence of the training portion of every CV sample.

    Used as the decision threshold for MCC: a run is called positive on
    the top ``prevalence`` fraction of its ranking, which needs the
    training labels only and never the labels of the scored sample.
    """
    path = data_dir / "splits" / dataset / "cv_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CV folds CSV: {path}")
    folds = pd.read_csv(path, usecols=["target", "repeat", "fold"])
    prevalence = {}
    for repeat, block in folds.groupby("repeat"):
        for fold in block["fold"].unique():
            train = block[block["fold"] != fold]
            prevalence[(repeat, fold)] = float(train["target"].mean())
    return prevalence


def run_battery(data_dir: Path, dataset: str, runs: list[str]) -> pd.DataFrame:
    """Score every run with every metric, one row per CV sample."""
    prevalence = train_prevalence(data_dir, dataset)
    rows = []
    for run in runs:
        predictions = load_cv_predictions(data_dir, dataset, run)
        scored = 0
        for (repeat, fold), sample in predictions.groupby(["repeat", "fold"]):
            scores = sample["predicted_probability"].to_numpy()
            rate = prevalence.get(
                (repeat, fold), float(sample["target"].mean())
            )
            threshold = float(np.quantile(scores, 1.0 - rate))
            metrics = sample_metrics(
                sample["target"].to_numpy(), scores, threshold=threshold
            )
            rows.append(
                {"run": run, "repeat": repeat, "fold": fold, **metrics}
            )
            scored += 1
        logger.info(f"{run}: {scored} samples scored")

    samples = pd.DataFrame(rows)
    coverage = samples.groupby("run").size()
    expected = int(coverage.max())
    short = {run: int(n) for run, n in coverage.items() if n < expected}
    if short:
        detail = ", ".join(
            f"{run}={n}/{expected}" for run, n in sorted(short.items())
        )
        logger.warning(f"Dropping runs with incomplete CV samples: {detail}")
        complete = coverage[coverage == expected].index
        samples = samples[samples["run"].isin(complete)]
    return samples.reset_index(drop=True)


def summarise(samples: pd.DataFrame) -> pd.DataFrame:
    """Mean of every metric per run."""
    return samples.groupby("run")[list(METRIC_DIRECTION)].mean()


def rank_runs(table: pd.DataFrame, metric: str) -> list[str]:
    """Order the runs by one metric, best first."""
    ascending = METRIC_DIRECTION[metric] == "lower"
    return list(table[metric].sort_values(ascending=ascending).index)


def ap_ci_half(samples: pd.DataFrame) -> float:
    """Half-width of the simultaneous 95% interval around the AP means.

    Reuses the comparison machinery of the CV figures, so a metric that
    only swaps runs which are inside this interval is a swap of ties.
    """
    frame = samples.rename(columns={"run": "model"}).copy()
    frame["sample"] = (
        frame["repeat"].astype(str) + "_" + frame["fold"].astype(str)
    )
    if frame["model"].nunique() < 2 or frame["sample"].nunique() < 2:
        return 0.0
    comparison, _ = _compare(_fold_table(frame, "ap"), "auto")
    return float(comparison.ci_half or 0.0)


def format_report(
    table: pd.DataFrame,
    ci_half: float = 0.0,
    skipped: list[str] | None = None,
) -> str:
    """Text answer to "does the metric change the ranking?"."""
    if skipped:
        lines = ["Skipped runs: " + ", ".join(skipped)]
    else:
        lines = []
    lines.append("Mean metric per run:")
    lines.append(table.round(4).to_string())
    lines.append("")
    lines.append("Ranking by each metric (best first):")
    for metric in METRIC_DIRECTION:
        lines.append(f"  {metric:7s} " + " > ".join(rank_runs(table, metric)))
    lines.append("")

    ap_order = rank_runs(table, "ap")
    ap_rank = {run: i for i, run in enumerate(ap_order)}
    lines.append("Spearman rank correlation with AP:")
    for metric in METRIC_DIRECTION:
        order = rank_runs(table, metric)
        ranks = [ap_rank[run] for run in order]
        rho = stats.spearmanr(ranks, list(range(len(ranks)))).statistic
        lines.append(f"  {metric:7s} {rho:+.2f}")
    lines.append("")

    lines.append("Selection (runs kept for the held-out stage):")
    for k in (1, 2, 3):
        picks = {m: rank_runs(table, m)[:k] for m in METRIC_DIRECTION}
        ap_pick = set(picks["ap"])
        for metric, run_list in picks.items():
            if metric == "ap":
                continue
            if set(run_list) != ap_pick:
                lines.append(
                    f"  top {k} by {metric:7s} differs from AP: "
                    + ", ".join(run_list)
                )
        if all(set(v) == ap_pick for v in picks.values()):
            lines.append(f"  top {k} identical under every metric")
    lines.append("")

    if ci_half:
        lines.append(
            "AP gaps against the simultaneous 95% interval "
            f"(half-width {ci_half:.4f}):"
        )
        for first, second in zip(ap_order, ap_order[1:]):
            gap = table.loc[first, "ap"] - table.loc[second, "ap"]
            verdict = "separated" if gap > ci_half else "tied"
            lines.append(f"  {first} > {second}  gap {gap:+.4f}  {verdict}")
    return "\n".join(lines)


def decorate_box(ax, *, labelsize: int = FONTSIZE) -> None:
    """Close the panel into a box: four spines, no tick marks.

    The shared style leaves the top and right spine open, which reads as
    loose on a small panel; here every panel is a closed frame.
    """
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_color("black")
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)


def plot_metric_ranking(
    table: pd.DataFrame,
    dataset: str,
    path: Path,
    title: str | None = None,
) -> None:
    """One panel per metric, the runs in a fixed order (best AP first).

    The bars keep the row order of the primary metric and every bar
    carries its rank under that metric, so a run that moves when the
    metric changes shows a number that no longer matches its row.
    """
    use_style()
    order = rank_runs(table, "ap")
    n = len(order)
    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(13.0, 1.3 + 0.45 * n), sharey=True
    )
    positions = np.arange(n)
    for ax, (metric, panel_title, color) in zip(axes, PANELS):
        values = [table.loc[run, metric] for run in order]
        ranks = {
            run: rank
            for rank, run in enumerate(rank_runs(table, metric), start=1)
        }
        ax.barh(positions, values, color=color, height=0.72)
        span = max(values) if max(values) else 1.0
        for position, run, value in zip(positions, order, values):
            ax.text(
                value + 0.02 * span,
                position,
                str(ranks[run]),
                va="center",
                ha="left",
                fontsize=FONTSIZE_SMALL,
                color=color,
            )
        ax.set_xlim(0, 1.15 * span)
        ax.set_title(panel_title, fontsize=FONTSIZE, pad=12)
        decorate_box(ax, labelsize=FONTSIZE_SMALL)
    axes[0].invert_yaxis()
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(
        [run_label(run) for run in order], fontsize=FONTSIZE
    )
    fig.suptitle(
        title or dataset, x=0.01, ha="left", va="top", fontsize=FONTSIZE_TITLE
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, path)


def main() -> None:
    """Entry point of ``brood-plot-metrics``."""
    parser = argparse.ArgumentParser(
        description="Compare screening runs under several metrics."
    )
    parser.add_argument("--dataset", default="GNEtolC")
    parser.add_argument(
        "--runs",
        default=None,
        help="Comma separated run names "
        "(default: the screening runs of the dataset)",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Root figure directory; the script appends <dataset>/cv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_dir = Path(args.data_dir)
    dataset = args.dataset
    runs = (
        [run.strip() for run in args.runs.split(",") if run.strip()]
        if args.runs
        else screening_runs(data_dir, dataset)
    )
    missing = [] if args.runs else skipped_runs(data_dir, dataset)
    if missing:
        logger.warning(
            "Skipping runs without CV predictions: " + ", ".join(missing)
        )
    if not runs:
        logger.error(f"No screening runs found for {dataset}")
        return

    logger.info(f"Scoring {len(runs)} runs: {runs}")
    samples = run_battery(data_dir, dataset, runs)
    table = summarise(samples)
    dropped = [run for run in runs if run not in set(table.index)]
    if dropped:
        logger.warning(
            "Skipping runs with incomplete CV samples: " + ", ".join(dropped)
        )
    skipped = missing + [f"{run} (incomplete CV samples)" for run in dropped]

    table_path = data_dir / "runs" / dataset / "metric_comparison.csv"
    table.to_csv(table_path)
    logger.info(f"Metric table saved to {table_path}")

    figure_path = (
        Path(args.output_dir)
        / dataset
        / "cv"
        / f"{dataset}_metric_ranking.svg"
    )
    plot_metric_ranking(table, dataset, figure_path)
    logger.info(f"Figure saved to {figure_path}")

    print(format_report(table, ap_ci_half(samples), skipped))


if __name__ == "__main__":
    main()
