#!/usr/bin/env python3
"""Batch BYOP evaluation runner.

Reads a JSON runlist, evaluates predictions for multiple models,
and saves a merged results CSV + a flat metrics CSV.

Usage
-----
    python scripts/evaluate.py file.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.byop import evaluate_predictions, load_predictions
from src.tracking import log_experiment
from src.types import EvalConfig, SplitConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch BYOP evaluation runner."
    )
    parser.add_argument("runlist", help="Path to JSON runlist")
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
        help="Output dir for CSVs. Defaults to data/predictions/<dataset>/",
    )
    return parser.parse_args()


def _load_runlist(path: str) -> dict:
    with open(path) as f:
        runlist = json.load(f)
    for key in ("dataset", "test_full_path", "runs"):
        if key not in runlist:
            raise KeyError(f"Missing required key '{key}' in runlist.")
    if not runlist["runs"]:
        raise ValueError("Runlist 'runs' list is empty.")
    return runlist


def _build_extra_params(test_full_path: str | Path) -> dict:
    """Derive train path from test path and load training stats."""
    test_path = Path(test_full_path)
    if test_path.name.endswith("_test.csv"):
        train_full_path = str(test_path).replace("_test.csv", "_train.csv")
    else:
        train_full_path = str(test_path.parent / "train.csv")

    params = {}
    if Path(train_full_path).exists():
        train_df = pd.read_csv(train_full_path)
        params["n_train"] = len(train_df)
        params["n_train_actives"] = int(train_df["target"].sum())
    else:
        logger.warning(f"Train path not found: {train_full_path}")
    return params


def _merge_all_runs(
    test_df: pd.DataFrame,
    runlist: dict,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Load all predictions and left-join into a wide dataframe."""
    merged = test_df.copy()
    successful_runs: list[tuple[str, str]] = []

    for run in runlist["runs"]:
        name = run["name"]
        preds_path = Path(run["predictions_path"])
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


def _evaluate_and_log(
    merged: pd.DataFrame,
    successful_runs: list[tuple[str, str]],
    runlist: dict,
    args: argparse.Namespace,
    extra_params: dict,
) -> list[dict]:
    """Evaluate each model, log to MLflow, and return metrics rows."""
    tags = runlist.get("tags", {})
    dataset = runlist["dataset"]
    if "dataset" not in tags:
        tags["dataset"] = dataset

    split_cfg = SplitConfig()
    eval_cfg = EvalConfig()
    metrics_rows: list[dict] = []

    for name, score_col_renamed in successful_runs:
        run_tags = {**tags, "model": name}
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

        log_experiment(
            run_name=f"{dataset}/{name}",
            result=result,
            split_cfg=split_cfg,
            eval_cfg=eval_cfg,
            tags=run_tags,
            extra_params=extra_params,
        )

        row: dict = {"model": name, "global_n": result.global_.n}
        row["global_ap"] = result.global_.average_precision
        row["global_bedroc"] = result.global_.bedroc

        for b in result.by_bin:
            row[f"{b.bin}_ap"] = b.average_precision
            row[f"{b.bin}_bedroc"] = b.bedroc
            row[f"{b.bin}_n"] = b.n
            row[f"{b.bin}_n_actives"] = b.n_actives

        metrics_rows.append(row)

        logger.success(
            f"{name}  AP={row['global_ap']:.4f}  "
            f"BEDROC={row['global_bedroc']:.4f}  n={row['global_n']}"
        )

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

        logger.info("─" * 50)
        logger.info(f"{'model':<20} {'AP':<10} {'BEDROC':<10} {'n':<6}")
        logger.info("─" * 50)
        for row in metrics_rows:
            logger.info(
                f"{row['model']:<20} {row['global_ap']:<10.4f} "
                f"{row['global_bedroc']:<10.4f} {row['global_n']:<6}"
            )
        logger.info("─" * 50)
    else:
        logger.warning("No runs completed.")


def main() -> None:
    """Run batch evaluation from a JSON runlist."""
    args = _parse_args()
    runlist = _load_runlist(args.runlist)
    dataset = runlist["dataset"]
    test_full_path = Path(runlist["test_full_path"])

    if not test_full_path.exists():
        logger.error(f"test_full_path not found: {test_full_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_full_path)
    n_test_actives = int(test_df["target"].sum())
    logger.info(
        f"{dataset}: evaluating {len(test_df)} test ({n_test_actives} actives)"
    )

    extra_params = {"n_test": len(test_df), "n_test_actives": n_test_actives}
    extra_params.update(_build_extra_params(test_full_path))

    output_dir = Path(args.output_dir or f"data/predictions/{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, successful_runs = _merge_all_runs(test_df, runlist, args)
    if not successful_runs:
        logger.warning("No runs completed.")
        return

    metrics_rows = _evaluate_and_log(
        merged, successful_runs, runlist, args, extra_params
    )
    _save_and_report(merged, metrics_rows, output_dir)


if __name__ == "__main__":
    main()
