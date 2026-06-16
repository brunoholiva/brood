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

import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.argparse_utils import build_base_arg_parser
from src.data_utils import load_split_data, save_predictions
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


def _parse_args():
    parser = build_base_arg_parser(
        description="Similarity baseline: score by max Tanimoto to active."
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

    train_df, test_df = load_split_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
        require_cluster_id=False,
    )

    active_mask = train_df["target"] == 1
    train_actives = train_df[active_mask]
    if len(train_actives) == 0:
        logger.error("No active molecules in training set — cannot score.")
        sys.exit(1)

    logger.info(
        f"{dataset}: generating fingerprints — {len(train_actives)} actives, "
        f"{len(test_df)} test"
    )
    train_active_fps = _build_fingerprints(
        train_actives["standardized_smiles"]
    )
    test_fps = _build_fingerprints(test_df["standardized_smiles"])

    logger.info(
        "{}: scoring {} test molecules by Tanimoto sim to {} actives".format(
            dataset, len(test_df), len(train_actives)
        )
    )
    scores = _score_via_similarity(test_fps, train_active_fps, args.batch_size)

    preds = pd.DataFrame(
        {
            "standardized_smiles": test_df["standardized_smiles"],
            "predicted_probability": scores,
        }
    )

    save_predictions(
        preds, data_dir / "predictions" / dataset, "similarity.csv"
    )


if __name__ == "__main__":
    main()
