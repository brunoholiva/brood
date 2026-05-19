#!/usr/bin/env python3
"""Batch split all datasets in ``data/`` into train/test.

Writes ``splits/<dataset>/{train,test_full,test_features}.csv`` for
every CSV in the data directory.

Usage
-----
    python scripts/split_datasets.py
    python scripts/split_datasets.py --data-dir data --splits-dir splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.splitter import taylor_butina_split

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
        print(f"  SKIP  {dataset}  — no SMILES column found.")
        return

    # find target column
    if target_col is None:
        target_col = _find_column(df, TARGET_ALIASES)
    if target_col is None:
        print(f"  SKIP  {dataset}  — no target column found.")
        return

    df = df.rename(
        columns={target_col: "target", smiles_col: "standardized_smiles"}
    )

    before = len(df)
    df = df.dropna(subset=["standardized_smiles"])
    dropped = before - len(df)

    if len(df) < 2:
        print(f"  SKIP  {dataset}  — only {len(df)} rows after dropping NaN.")
        return

    try:
        train_df, test_df = taylor_butina_split(df)
    except Exception as e:
        print(f"  FAIL  {dataset}  — {e}")
        return

    out_dir = splits_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test_full.csv", index=False)
    test_df[["standardized_smiles", "distance_to_train"]].to_csv(
        out_dir / "test_features.csv", index=False
    )

    n_train = len(train_df)
    n_test = len(test_df)
    n_actives = test_df["target"].sum()
    msg = (
        f"  OK    {dataset:<20}  train={n_train}  test={n_test}  "
        f"actives={n_actives}"
    )
    if dropped:
        msg += f"  ({dropped} NaN dropped)"
    print(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch split all datasets in data/."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory with dataset CSVs"
    )
    parser.add_argument(
        "--splits-dir", default="splits", help="Output directory for splits"
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
        print(f"No CSVs found in {data_dir}/")
        return

    print(f"Splitting {len(csvs)} dataset(s)...")
    for path in csvs:
        _split_csv(path, args.smiles_col, args.target_col, splits_dir)


if __name__ == "__main__":
    main()
