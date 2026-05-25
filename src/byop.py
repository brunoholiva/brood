"""Bring-your-own-predictions helpers for brood evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluator import evaluate
from .types import BinMetrics, BroodResult, RankingMetrics


def load_predictions(
    path: str | Path,
    smiles_col: str = "standardized_smiles",
    score_col: str = "predicted_probability",
) -> pd.DataFrame:
    """Load a prediction CSV and validate required columns."""
    preds = pd.read_csv(path)
    missing = [c for c in (smiles_col, score_col) if c not in preds.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    return preds


def merge_predictions(
    test_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    score_col: str = "predicted_probability",
) -> pd.DataFrame:
    """Merge predictions onto test set by SMILES.

    All columns from ``test_df`` are retained. Predictions not matching
    a test SMILES are silently dropped.
    """
    if "standardized_smiles" not in test_df.columns:
        raise KeyError("'standardized_smiles' missing from test set.")
    merged = test_df.merge(
        preds_df[[smiles_col, score_col]],
        left_on="standardized_smiles",
        right_on=smiles_col,
        how="inner",
    )
    if smiles_col != "standardized_smiles":
        merged = merged.drop(columns=[smiles_col])
    return merged


def evaluate_predictions(
    merged_df: pd.DataFrame,
    score_col: str = "predicted_probability",
    label_col: str = "target",
    distance_col: str = "distance_to_train",
) -> BroodResult:
    """Evaluate predictions and return a BroodResult."""
    eval_result = evaluate(
        merged_df,
        score_col=score_col,
        label_col=label_col,
        distance_col=distance_col,
    )
    return BroodResult(
        global_=RankingMetrics(**eval_result["global"]),
        by_bin=[BinMetrics(**b) for b in eval_result["by_bin"]],
        test_df=merged_df,
        pipeline=None,
    )
