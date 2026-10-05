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
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from brood.data.report import normalize_dataset
from brood.data.splitter import (
    assign_cv_folds,
    scaffold_split,
    validate_cv_folds,
    validate_partitions,
)

CV_FOLDS = 5

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

    df, quality = normalize_dataset(df, smiles_col, target_col)
    logger.info(f"{dataset}: quality report {quality}")

    if len(df) < 2:
        logger.warning(
            f"{dataset}: only {len(df)} rows after NaN drop, skipping"
        )
        return

    try:
        train_df, test_df = scaffold_split(df)
    except Exception as e:
        logger.error(f"{dataset}: split failed — {e}")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)
    split_checks = validate_partitions(train_df, test_df)

    train_df.to_csv(splits_dir / f"{dataset}_train.csv", index=False)
    test_df.to_csv(splits_dir / f"{dataset}_test.csv", index=False)

    cv_df = assign_cv_folds(train_df, n_splits=CV_FOLDS)
    cv_path = splits_dir / f"{dataset}_cv_folds.csv"
    cv_df[["standardized_smiles", "target", "scaffold", "fold"]].to_csv(
        cv_path, index=False
    )
    cv_checks = validate_cv_folds(cv_df)
    actual_folds = cv_checks["folds"]
    logger.success(f"CV folds ({actual_folds}) saved to {cv_path}")

    manifest = train_df.assign(partition="train").merge(
        cv_df[["standardized_smiles", "fold"]],
        on="standardized_smiles",
        how="left",
    )
    test_manifest = test_df.assign(partition="test", fold=pd.NA)
    manifest = pd.concat([manifest, test_manifest], ignore_index=True)
    manifest.to_csv(splits_dir / f"{dataset}_manifest.csv", index=False)
    (splits_dir / f"{dataset}_manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "test_fraction_requested": 0.1,
                "cv_folds": actual_folds,
                "random_state": 42,
                "distance": "mean 5-nearest Morgan-Tanimoto distance",
                "molecules": len(manifest),
                "test_molecules": int((manifest["partition"] == "test").sum()),
                "split_checks": split_checks,
                "cv_checks": cv_checks,
            },
            indent=2,
        )
    )

    _log_split_stats(dataset, train_df, test_df, cv_df)

    n_train = len(train_df)
    n_test = len(test_df)
    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    logger.success(
        f"{dataset}: train={n_train} ({n_train_actives} actives), "
        f"test={n_test} ({n_test_actives} actives)"
    )


def _log_split_stats(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_df: pd.DataFrame,
) -> None:
    """Log sanity statistics about a completed split."""
    n_scaff_overlap = len(set(train_df["scaffold"]) & set(test_df["scaffold"]))
    dist = test_df["distance_to_train"]
    logger.success(
        f"{dataset}: train={len(train_df)} "
        f"({train_df['target'].mean():.1%} actives, "
        f"{train_df['scaffold'].nunique()} scaffolds) | "
        f"test={len(test_df)} ({test_df['target'].mean():.1%} actives, "
        f"{test_df['scaffold'].nunique()} scaffolds) | "
        f"scaffold overlap={n_scaff_overlap}"
    )
    logger.info(
        f"{dataset}: test distance_to_train — "
        f"min={dist.min():.3f} mean={dist.mean():.3f} max={dist.max():.3f}"
    )
    cv_checks = validate_cv_folds(cv_df)
    fold_stats = [
        f"fold{fold}: {len(fdf)} ({fdf['target'].mean():.1%} actives, "
        f"{fdf['scaffold'].nunique()} scaffolds)"
        for fold, fdf in cv_df.groupby("fold", sort=True)
    ]
    logger.info(
        f"{dataset}: CV folds ({cv_checks['folds']}) — "
        + " | ".join(fold_stats)
        + f" | scaffold overlap={cv_checks['scaffold_overlap']}"
        + f" | active-fraction spread="
        f"{cv_checks['active_fraction_spread']:.1%}"
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
