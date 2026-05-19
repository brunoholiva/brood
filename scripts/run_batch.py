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

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.byop import (  # noqa: E402
    evaluate_predictions,
    load_predictions,
    merge_predictions,
)
from src.tracking import log_experiment
from src.types import EvalConfig, SplitConfig 


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
        print(f"Error: test_full_path not found: {test_full_path}")
        sys.exit(1)

    test_df = pd.read_csv(test_full_path)

    split_cfg = SplitConfig()
    eval_cfg = EvalConfig()

    summary = []
    for run in manifest["runs"]:
        name = run["name"]
        preds_path = Path(run["predictions_path"])
        if not preds_path.exists():
            print(f"  SKIP  {name}  — predictions not found: {preds_path}")
            continue

        run_tags = {**tags, "model": name}
        run_name = f"{dataset}/{name}"

        preds = load_predictions(
            preds_path,
            smiles_col=args.smiles_col,
            score_col=args.score_col,
        )
        merged = merge_predictions(
            test_df,
            preds,
            smiles_col=args.smiles_col,
            score_col=args.score_col,
        )

        if len(merged) == 0:
            print(f"  SKIP  {name}  — no matching predictions.")
            continue

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
        )

        g = result.global_
        summary.append(
            f"  OK    {name:<20}  AP={g.average_precision:.4f}  "
            f"BEDROC={g.bedroc:.4f}  n={g.n}"
        )

    if summary:
        print("\nBatch complete.\n" + "\n".join(summary))
    else:
        print("\nNo runs completed.")


if __name__ == "__main__":
    main()
