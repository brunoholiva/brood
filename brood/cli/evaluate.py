#!/usr/bin/env python3
"""Batch BYOP evaluation runner.

Evaluates predictions for multiple models against a held-out test set
and saves a merged results CSV + a flat metrics CSV.

Usage
-----
    python scripts/evaluate.py --dataset stokes
        --test-full-path data/splits/stokes_test.csv
        --runs rf=data/runs/stokes/rf/predictions/test.csv
               chemprop=data/runs/stokes/chemprop/predictions/test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from brood.evaluation.byop import (
    assert_unique_smiles,
    evaluate_predictions,
    load_predictions,
)
from brood.plotting.heldout import (
    plot_distance_vs_probability,
    plot_pr_curves,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch BYOP evaluation runner."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--test-full-path",
        required=True,
        help="Path to the held-out test CSV",
    )
    parser.add_argument(
        "--runs",
        required=True,
        nargs="+",
        help="model=/path/to/predictions.csv pairs",
        metavar="model=path",
    )
    parser.add_argument(
        "--smiles-col",
        default="standardized_smiles",
        help="SMILES column name in prediction files",
    )
    parser.add_argument(
        "--score-col",
        default="predicted_probability",
        help="Score column name in prediction files",
    )
    parser.add_argument(
        "--label-col",
        default="target",
        help="Label column name",
    )
    parser.add_argument(
        "--distance-col",
        default="distance_to_train",
        help="Distance column name",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir for CSVs. Defaults to data/evaluations/<dataset>/",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Root directory for plots (default: figures)",
    )
    return parser.parse_args()


def _parse_runs(items: list[str]) -> dict[str, str]:
    """Parse `model=path` pairs into {name: path}."""
    runs: dict[str, str] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"Bad --runs entry (want model=path): {item}")
        runs[name] = path
    return runs


def _merge_all_runs(
    test_df: pd.DataFrame,
    runs: dict[str, str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load all predictions and left-join into a wide dataframe."""
    assert_unique_smiles(test_df, "standardized_smiles", "test set")
    merged = test_df.copy()
    successful_runs: list[tuple[str, str]] = []

    for name, preds_path_str in runs.items():
        preds_path = Path(preds_path_str)
        if not preds_path.exists():
            logger.warning(f"{name}: predictions not found at {preds_path}")
            continue

        preds = load_predictions(
            preds_path,
            smiles_col=args.smiles_col,
            score_col=args.score_col,
        )

        score_col_renamed = f"{name}_{args.score_col}"
        preds = preds.rename(columns={args.score_col: score_col_renamed})
        logger.info(
            f"{name}: {len(preds)} predictions loaded "
            f"({preds[score_col_renamed].notna().sum()} non-null)"
        )

        merged = merged.merge(
            preds[[args.smiles_col, score_col_renamed]],
            left_on="standardized_smiles",
            right_on=args.smiles_col,
            how="left",
        )
        successful_runs.append((name, score_col_renamed))

    return merged, successful_runs


def _evaluate(
    merged: pd.DataFrame,
    successful_runs: list[tuple[str, str]],
    args: argparse.Namespace,
    figure_dir: Path,
) -> list[dict]:
    """Evaluate each model and return metrics rows."""
    metrics_rows: list[dict] = []
    plotted: dict[str, str] = {}

    for name, score_col_renamed in successful_runs:
        eval_df = merged.dropna(subset=[score_col_renamed]).copy()

        if len(eval_df) == 0:
            logger.warning(
                f"{name}: no test molecules with predictions, skipping"
            )
            continue

        logger.info(f"{name}: {len(eval_df)} test molecules with predictions")

        result = evaluate_predictions(
            eval_df,
            score_col=score_col_renamed,
            label_col=args.label_col,
            distance_col=args.distance_col,
        )
        try:
            plot_distance_vs_probability(
                eval_df,
                name,
                score_col_renamed,
                figure_dir,
                label_col=args.label_col,
                distance_col=args.distance_col,
            )
        except ValueError as err:
            logger.warning(str(err))
        plotted[name] = score_col_renamed

        row: dict = {"model": name, "global_n": result.global_.n}
        row["global_ap"] = result.global_.average_precision
        row["global_bedroc"] = result.global_.bedroc
        row["global_brier"] = result.global_.brier

        for b in result.by_bin:
            row[f"{b.bin}_ap"] = b.average_precision
            row[f"{b.bin}_bedroc"] = b.bedroc
            row[f"{b.bin}_brier"] = b.brier
            row[f"{b.bin}_n"] = b.n
            row[f"{b.bin}_n_actives"] = b.n_actives

        metrics_rows.append(row)

        logger.success(
            f"{name}  AP={row['global_ap']:.4f}  "
            f"BEDROC={row['global_bedroc']:.4f}  n={row['global_n']}"
        )

    # One PR panel over every run that produced a usable scatter.
    if plotted:
        try:
            plot_pr_curves(
                merged,
                plotted,
                args.dataset,
                figure_dir,
                label_col=args.label_col,
            )
        except ValueError as err:
            logger.warning(str(err))

    return metrics_rows


def _save_and_report(
    merged: pd.DataFrame,
    metrics_rows: list[dict],
    output_dir: Path,
) -> None:
    """Save merged results and metrics CSVs, print summary table."""
    merged_path = output_dir / "merged_results.csv"
    merged.to_csv(merged_path, index=False)
    logger.info(f"Merged results ({len(merged)} rows) saved to {merged_path}")

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        logger.info(f"Metrics saved to {output_dir / 'metrics.csv'}")

        logger.info("─" * 65)
        logger.info(
            f"{'model':<20} {'AP':<10} {'BEDROC':<10} {'BRIER':<10} {'n':<6}"
        )
        logger.info("─" * 65)
        for row in metrics_rows:
            logger.info(
                f"{row['model']:<20} {row['global_ap']:<10.4f} "
                f"{row['global_bedroc']:<10.4f} "
                f"{row['global_brier']:<10.4f} {row['global_n']:<6}"
            )
        logger.info("─" * 65)
    else:
        logger.warning("No runs completed.")


def main() -> None:
    """Evaluate multiple model predictions against the held-out test set."""
    args = _parse_args()
    runs = _parse_runs(args.runs)
    if not runs:
        sys.exit("Nothing to evaluate: --runs is empty.")
    test_full_path = Path(args.test_full_path)

    if not test_full_path.exists():
        logger.error(f"test_full_path not found: {test_full_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_full_path)
    n_test_actives = int(test_df["target"].sum())
    logger.info(
        f"{args.dataset}: evaluating {len(test_df)} test "
        f"({n_test_actives} actives)"
    )

    output_dir = Path(args.output_dir or f"data/evaluations/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(args.figures_dir) / args.dataset / "heldout"
    figure_dir.mkdir(parents=True, exist_ok=True)

    merged, successful_runs = _merge_all_runs(test_df, runs, args)
    if not successful_runs:
        logger.warning("No runs completed.")
        return

    metrics_rows = _evaluate(merged, successful_runs, args, figure_dir)
    _save_and_report(merged, metrics_rows, output_dir)


if __name__ == "__main__":
    main()
