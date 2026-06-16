"""Data loading and I/O utilities for training scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def load_split_data(
    train_path: Path,
    test_path: Path,
    dataset: str,
    require_cluster_id: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate train/test CSVs from split files.

    Parameters
    ----------
    train_path : Path
        Path to training CSV.
    test_path : Path
        Path to test CSV.
    dataset : str
        Dataset name for logging.
    require_cluster_id : bool
        If True, error if ``cluster_id`` column is missing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with loaded data.
    """
    for p, label in [(train_path, "train"), (test_path, "test")]:
        if not p.exists():
            logger.error(
                f"{label} path not found: {p}. Run split_datasets.py first."
            )
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    if "cluster_id" in train_df.columns:
        n_clusters = train_df["cluster_id"].nunique()
        logger.info(
            f"{dataset}: train={len(train_df)} ({n_train_actives} actives, "
            f"{n_clusters} Butina clusters)"
        )
    else:
        if require_cluster_id:
            logger.error(
                f"{train_path} missing 'cluster_id' column. Re-run "
                "scripts/split_datasets.py to regenerate splits."
            )
            sys.exit(1)
        logger.info(
            f"{dataset}: train={len(train_df)} ({n_train_actives} actives)"
        )

    logger.info(f"{dataset}: test={len(test_df)} ({n_test_actives} actives)")

    return train_df, test_df


def save_predictions(
    preds: pd.DataFrame, out_dir: Path, filename: str
) -> Path:
    """Write predictions CSV and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    preds.to_csv(out_path, index=False)
    logger.success(f"Predictions saved to {out_path}")
    return out_path
