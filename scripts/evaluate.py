#!/usr/bin/env python3
"""Batch BYOP evaluation runner.

Reads a JSON file and evaluates + logs predictions for multiple
model runs against a single dataset split.

Usage
-----
    python scripts/run_batch.py file.json
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

from src.byop import (  # noqa: E402
    evaluate_predictions,
    load_predictions,
    merge_predictions,
)
from src.tracking import log_experiment  # noqa: E402
from src.types import EvalConfig, SplitConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch BYOP evaluation runner."
    )
    parser.add_argument("manifest", help="Path to JSON manifest")
    parser.add_argument(
        "--smiles-col",
        default="standardized_smiles",
        help="SMILES column name (default: standardized_smiles)",
    )
    parser.add_argument(
        "--score-col",
        default="predicted_probability",
        help="Prediction score column (default: predicted_probability)",
    )
    parser.add_argument(
        "--label-col",
        default="target",
        help="Label column name (default: target)",
    )
    parser.add_argument(
        "--distance-col",
        default="distance_to_train",
        help="Distance column name (default: distance_to_train)",
    )
    return parser.parse_args()


def _load_manifest(path: str) -> dict:
    with open(path) as f:
        manifest = json.load(f)
    for key in ("dataset", "test_full_path", "runs"):
        if key not in manifest:
            raise KeyError(f"Missing required key '{key}' in manifest.")
    if not manifest["runs"]:
        raise ValueError("Manifest 'runs' list is empty.")
    return manifest


def main() -> None:
    """Run batch evaluation from a JSON manifest."""
    args = _parse_args()
    manifest = _load_manifest(args.manifest)
    dataset = manifest["dataset"]
    test_full_path = Path(manifest["test_full_path"])
    tags = manifest.get("tags", {})
    if "dataset" not in tags:
        tags["dataset"] = dataset

    if not test_full_path.exists():
        logger.error(f"test_full_path not found: {test_full_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_full_path)
    n_test_actives = int(test_df["target"].sum())
    logger.info(f"{dataset}: evaluating {len(test_df)} test molecules "
                f"({n_test_actives} actives)")

    # Derive train split path from test path
    test_path = Path(test_full_path)
    if test_path.name.endswith("_test.csv"):
        train_full_path = str(test_path).replace("_test.csv", "_train.csv")
    else:
        train_full_path = str(test_path.parent / "train.csv")

    extra_params = {"n_test": len(test_df), "n_test_actives": n_test_actives}
    if Path(train_full_path).exists():
        train_df = pd.read_csv(train_full_path)
        n_train_actives = int(train_df["target"].sum())
        extra_params["n_train"] = len(train_df)
        extra_params["n_train_actives"] = n_train_actives

    split_cfg = SplitConfig()
    eval_cfg = EvalConfig()

    results = []
    for run in manifest["runs"]:
        name = run["name"]
        preds_path = Path(run["predictions_path"])
        if not preds_path.exists():
            logger.warning(f"{name}: predictions not found at {preds_path}")
            continue

        run_tags = {**tags, "model": name}
        run_name = f"{dataset}/{name}"

        preds = load_predictions(
            preds_path,
            smiles_col=args.smiles_col,
            score_col=args.score_col,
        )
        logger.info(f"{name}: {len(preds)} predictions loaded")

        merged = merge_predictions(
            test_df,
            preds,
            smiles_col=args.smiles_col,
            score_col=args.score_col,
        )

        if len(merged) == 0:
            logger.warning(f"{name}: no matching test molecules, skipping")
            continue

        logger.info(f"{name}: {len(merged)} test molecules matched")

        result = evaluate_predictions(
            merged,
            score_col=args.score_col,
            label_col=args.label_col,
            distance_col=args.distance_col,
        )

        log_experiment(
            run_name=run_name,
            result=result,
            split_cfg=split_cfg,
            eval_cfg=eval_cfg,
            tags=run_tags,
            extra_params=extra_params,
        )

        g = result.global_
        results.append((name, g))

        logger.success(
            f"{name}  AP={g.average_precision:.4f}  "
            f"BEDROC={g.bedroc:.4f}  n={g.n}"
        )

    if results:
        logger.info("─" * 50)
        logger.info(f"{'model':<20} {'AP':<10} {'BEDROC':<10} {'n':<6}")
        logger.info("─" * 50)
        for name, g in results:
            logger.info(
                f"{name:<20} {g.average_precision:<10.4f} "
                f"{g.bedroc:<10.4f} {g.n:<6}"
            )
        logger.info("─" * 50)
    else:
        logger.warning("No runs completed.")


if __name__ == "__main__":
    main()
