"""Bring-your-own-predictions helpers for brood evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from brood.core.types import BroodResult
from brood.evaluation.evaluator import evaluate


def assert_unique_smiles(
    df: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    label: str = "frame",
) -> None:
    """Raise if *df* has repeated SMILES.

    Predictions are joined onto molecules by SMILES, so a repeated key turns
    the join into a cartesian product: rows multiply, and every downstream
    metric and figure is computed over the inflated table without any error.
    Fail loudly instead.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"'{smiles_col}' missing from {label}.")
    n_dupes = int(df[smiles_col].duplicated().sum())
    if n_dupes:
        raise ValueError(
            f"{label} has {n_dupes} repeated '{smiles_col}' value(s) "
            f"({len(df)} rows, {df[smiles_col].nunique()} unique). "
            "Joining on SMILES would duplicate rows and corrupt the "
            "metrics; deduplicate first."
        )


def load_predictions(
    path: str | Path,
    smiles_col: str = "standardized_smiles",
    score_col: str = "predicted_probability",
) -> pd.DataFrame:
    """Load a prediction CSV and validate required columns.

    Parameters
    ----------
    path : str | Path
        Path to the predictions CSV.
    smiles_col : str
        Column name containing SMILES identifiers.
    score_col : str
        Column name containing predicted scores.

    Returns
    -------
    pd.DataFrame
        The loaded predictions with validated columns and unique SMILES.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If the SMILES column contains duplicated values.
    """
    preds = pd.read_csv(path)
    missing = [c for c in (smiles_col, score_col) if c not in preds.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    assert_unique_smiles(preds, smiles_col, f"predictions in {path}")
    return preds


def merge_predictions(
    test_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    score_col: str = "predicted_probability",
) -> pd.DataFrame:
    """Merge predictions onto test set by SMILES.

    All columns from ``test_df`` are retained. Predictions not matching
    a test SMILES are silently dropped. Both sides must have unique SMILES,
    so the join stays one-to-one.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-set DataFrame with a ``standardized_smiles`` column.
    preds_df : pd.DataFrame
        Predictions DataFrame with SMILES and score columns.
    smiles_col : str
        SMILES column name in ``preds_df``.
    score_col : str
        Score column name in ``preds_df``.

    Returns
    -------
    pd.DataFrame
        Inner-joined frame with test columns plus the prediction score.

    Raises
    ------
    ValueError
        If either input has duplicated SMILES values.
    """
    assert_unique_smiles(test_df, "standardized_smiles", "test set")
    assert_unique_smiles(preds_df, smiles_col, "predictions")
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
    """Evaluate predictions and return a BroodResult.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged test + predictions DataFrame.
    score_col : str
        Column containing predicted probabilities.
    label_col : str
        Column containing true binary labels.
    distance_col : str
        Column containing distance-to-train values.

    Returns
    -------
    BroodResult
        Container with global and per-bin metrics plus the scored frame.
    """
    return evaluate(
        merged_df,
        score_col=score_col,
        label_col=label_col,
        distance_col=distance_col,
    )
