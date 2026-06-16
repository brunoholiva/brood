#!/usr/bin/env python3
"""Train a TabPFN classifier on RDKit2D + MACCS features.

Uses in-context learning (no gradient descent). Generates RDKit2D
normalized descriptors + MACCS keys as input features.

Usage
-----
    micromamba activate tabfpn
    python training/run_tabpfn.py --dataset chembl_ecoli
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tabpfn import TabPFNClassifier

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data_utils import load_split_data, save_predictions
from src.descriptors import DescriptorTransformer


def _default_params(device: str) -> dict:
    resolved = device
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": resolved,
        "ignore_pretraining_limits": True,
    }


def _train_and_predict(
    dataset: str,
    train_smiles: list[str],
    train_targets: list[int],
    test_smiles: list[str],
    params: dict,
) -> pd.DataFrame:
    """Featurize, fit TabPFN (in-context), and return predictions."""
    logger.info(
        "{} | TabPFN: featurizing {} training molecules".format(
            dataset, len(train_smiles)
        )
    )
    transformer = DescriptorTransformer()
    X_train = transformer.transform(train_smiles)
    y_train = np.array(train_targets)
    logger.info(
        f"{dataset} | TabPFN: {X_train.shape[0]} training molecules, "
        f"{X_train.shape[1]} features"
    )

    logger.info(f"{dataset} | TabPFN: fitting (in-context learning)")
    model = TabPFNClassifier(
        device=params["device"],
        ignore_pretraining_limits=params["ignore_pretraining_limits"],
    )
    model.fit(X_train, y_train)

    logger.info(
        f"{dataset} | TabPFN: featurizing {len(test_smiles)} test molecules"
    )
    X_test = transformer.transform(test_smiles)
    logger.info(
        f"{dataset} | TabPFN: {X_test.shape[0]} test molecules, "
        f"{X_test.shape[1]} features"
    )

    probs = model.predict_proba(X_test)[:, 1]

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles,
            "predicted_probability": probs,
        }
    )


def main() -> None:
    """Train TabPFN, predict, save predictions."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train TabPFN on RDKit2D+MACCS features and predict."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device (default: auto-detect)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = args.dataset

    train_df, test_df = load_split_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
        require_cluster_id=False,
    )

    params = _default_params(args.device)

    preds = _train_and_predict(
        dataset,
        train_df["standardized_smiles"].tolist(),
        train_df["target"].tolist(),
        test_df["standardized_smiles"].tolist(),
        params,
    )

    save_predictions(preds, data_dir / "predictions" / dataset, "tabpfn.csv")


if __name__ == "__main__":
    main()
