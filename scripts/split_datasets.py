#!/usr/bin/env python3
"""Batch split all datasets in ``data/raw/`` into train/test.

Writes ``data/splits/<dataset>_train.csv`` and
``data/splits/<dataset>_test.csv`` for every CSV in the raw data
directory.

Usage
-----
    python scripts/split_datasets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.splitter import taylor_butina_split  # noqa: E402

TARGET_ALIASES = [
    "target",
    "antimicrobial_activity",
    "activity",
    "label",
    "y",
]
SMILES_ALIASES = [
    "standardized_smiles",
    "SMILES",
    "smiles",
    "canonical_smiles",
]


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _dataset_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_cleaned", "_processed", "_raw"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _split_csv(
    path: Path,
    smiles_col: str | None,
    target_col: str | None,
    splits_dir: Path,
) -> None:
    df = pd.read_csv(path)
    dataset = _dataset_name(path)

    if smiles_col is None:
        smiles_col = _find_column(df, SMILES_ALIASES)
    if smiles_col is None:
        logger.warning(f"{dataset}: no SMILES column found, skipping")
        return

    if target_col is None:
        target_col = _find_column(df, TARGET_ALIASES)
    if target_col is None:
        logger.warning(f"{dataset}: no target column found, skipping")
        return

    n_total = len(df)
    n_total_actives = int(df[target_col].sum())
    logger.info(
        f"{dataset}: {n_total} molecules loaded, "
        f"{n_total_actives} actives"
    )

    df = df.rename(
        columns={target_col: "target", smiles_col: "standardized_smiles"}
    )

    before = len(df)
    df = df.dropna(subset=["standardized_smiles"])
    dropped = before - len(df)

    if dropped:
        logger.info(f"{dataset}: {dropped} NaN SMILES dropped")

    if len(df) < 2:
        logger.warning(
            f"{dataset}: only {len(df)} rows after NaN drop, skipping"
        )
        return

    try:
        train_df, test_df = taylor_butina_split(df, n_jobs=10)
    except Exception as e:
        logger.error(f"{dataset}: split failed — {e}")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / f"{dataset}_train.csv", index=False)
    test_df.to_csv(splits_dir / f"{dataset}_test.csv", index=False)

    n_train = len(train_df)
    n_test = len(test_df)
    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    logger.success(
        f"{dataset}: train={n_train} ({n_train_actives} actives), "
        f"test={n_test} ({n_test_actives} actives)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch split all datasets in data/."
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory with dataset CSVs (default: data/raw)",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Output directory for splits (default: data/splits)",
    )
    parser.add_argument(
        "--smiles-col", default=None, help="SMILES column name"
    )
    parser.add_argument(
        "--target-col", default=None, help="Target column name"
    )
    return parser.parse_args()


def main() -> None:
    """Run batch split on all CSVs in data/."""
    args = _parse_args()
    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        logger.warning(f"No CSVs found in {data_dir}/")
        return

    logger.info(f"Splitting {len(csvs)} dataset(s)...")
    for path in csvs:
        _split_csv(path, args.smiles_col, args.target_col, splits_dir)


if __name__ == "__main__":
    main()
