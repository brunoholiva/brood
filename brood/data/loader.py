"""Data loading and I/O utilities for training scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MissingScaffoldColumnError(ValueError):
    """Raised when a split file lacks the required scaffold column."""


class MissingSplitFileError(FileNotFoundError):
    """Raised when a train or test split file is missing."""


def load_split_data(
    train_path: Path,
    test_path: Path,
    dataset: str,
    require_scaffold: bool = True,
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
    require_scaffold : bool
        If True, error if ``scaffold`` column is missing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with loaded data.

    Raises
    ------
    MissingSplitFileError
        If either split file does not exist.
    MissingScaffoldColumnError
        If ``require_scaffold`` is True and the train file lacks a
        ``scaffold`` column.
    """
    for p, label in [(train_path, "train"), (test_path, "test")]:
        if not p.exists():
            raise MissingSplitFileError(
                f"{label} path not found: {p}. Run `brood-split` first."
            )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    if "scaffold" in train_df.columns:
        n_scaffolds = train_df["scaffold"].nunique()
        logger.debug(
            f"{dataset}: train={len(train_df)} ({n_train_actives} actives, "
            f"{n_scaffolds} scaffolds)"
        )
    else:
        if require_scaffold:
            raise MissingScaffoldColumnError(
                f"{train_path} missing 'scaffold' column. Re-run "
                "`brood-split` to regenerate splits."
            )
        logger.debug(
            f"{dataset}: train={len(train_df)} ({n_train_actives} actives)"
        )

    logger.debug(f"{dataset}: test={len(test_df)} ({n_test_actives} actives)")

    return train_df, test_df


def save_predictions(
    preds: pd.DataFrame, out_dir: Path, filename: str
) -> Path:
    """Write predictions CSV and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    preds.to_csv(out_path, index=False)
    logger.info(f"Predictions saved to {out_path}")
    return out_path
