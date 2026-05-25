#!/usr/bin/env python3
"""Similarity baseline: score = max Tanimoto similarity to any training active.

For each test molecule, computes Morgan fingerprints and finds the maximum
Tanimoto similarity to any active molecule in the training set. That
similarity becomes the predicted probability.

Usage
-----
    python training/run_maxsim_baseline.py --dataset dataset_name
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.fingerprints import MorganFingerprintTransformer

BATCH_SIZE = 5000


def _build_fingerprints(smiles):
    return MorganFingerprintTransformer().transform(smiles)


def _score_via_similarity(test_fps, active_fps, batch_size=BATCH_SIZE):
    """Score each test molecule by max Tanimoto sim to any training active."""
    scores = []
    n_test = test_fps.shape[0]
    for start in tqdm(
        range(0, n_test, batch_size),
        desc="Scoring by similarity",
        unit="batch",
    ):
        end = min(start + batch_size, n_test)
        dists = pairwise_distances(
            test_fps[start:end], active_fps, metric="jaccard", n_jobs=-1
        )
        scores.extend((1.0 - dists).max(axis=1).tolist())
    return scores


def _load_data(
    train_path: Path, test_path: Path, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run split_datasets.py first.")
        sys.exit(1)
    if not test_path.exists():
        logger.error(f"{test_path} not found. Run split_datasets.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    logger.info(
        f"{dataset}: train={len(train_df)} ({n_train_actives} actives)"
    )
    logger.info(f"{dataset}: test={len(test_df)} ({n_test_actives} actives)")

    return train_df, test_df


def _save_predictions(preds, out_dir, filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    preds.to_csv(out_path, index=False)
    logger.success(f"Predictions saved to {out_path}")
    return out_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Similarity baseline: score by max Tanimoto to active."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g. stokes)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for distance computation (default: {BATCH_SIZE})",
    )
    return parser.parse_args()


def main():
    """Score test molecules by similarity to training actives."""
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    train_df, test_df = _load_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
    )

    active_mask = train_df["target"] == 1
    train_actives = train_df[active_mask]
    if len(train_actives) == 0:
        logger.error("No active molecules in training set — cannot score.")
        sys.exit(1)

    logger.info(
        f"Generating fingerprints ({len(train_actives)} actives, "
        f"{len(test_df)} test)..."
    )
    train_active_fps = _build_fingerprints(
        train_actives["standardized_smiles"]
    )
    test_fps = _build_fingerprints(test_df["standardized_smiles"])

    logger.info("Scoring test molecules by similarity to actives...")
    scores = _score_via_similarity(test_fps, train_active_fps, args.batch_size)

    preds = pd.DataFrame(
        {
            "standardized_smiles": test_df["standardized_smiles"],
            "predicted_probability": scores,
        }
    )

    _save_predictions(
        preds, data_dir / "predictions" / dataset, "similarity.csv"
    )


if __name__ == "__main__":
    main()
