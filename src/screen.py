"""High-level API: screen() for one-shot OOD evaluation."""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from .evaluator import evaluate
from .fingerprints import MorganFingerprintTransformer
from .splitter import taylor_butina_split
from .types import (
    BinMetrics,
    ColumnConfig,
    EvalConfig,
    RankingMetrics,
    ScreenResult,
    SplitConfig,
)


def screen(
    df: pd.DataFrame,
    model,
    column_config: Optional[ColumnConfig] = None,
    split_config: Optional[SplitConfig] = None,
    eval_config: Optional[EvalConfig] = None,
    random_state: Optional[int] = None,
) -> ScreenResult:
    """One-shot OOD evaluation: split -> train -> predict -> evaluate.

    Wraps the 4-step workflow (split, train, predict, evaluate) into a
    single call. Bare sklearn estimators are automatically wrapped in a
    Pipeline with MorganFingerprintTransformer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``smiles_col`` and ``target_col``.
    model : sklearn estimator or Pipeline
        Any sklearn-compatible model. If it is not already a Pipeline, it
        will be auto-wrapped with MorganFingerprintTransformer. For non-
        sklearn models, use the modular API instead.
    column_config : ColumnConfig, optional
        Column name configuration. Defaults to ``ColumnConfig()``.
    split_config : SplitConfig, optional
        Split configuration (train_size, threshold, etc.).
        Defaults to ``SplitConfig()``.
    eval_config : EvalConfig, optional
        Evaluation configuration (bins, bedrock_alpha).
        Defaults to ``EvalConfig()``.
    random_state : int or None, optional
        Seed for reproducibility. Applied to the model if it has a
        ``random_state`` attribute. Not used by the splitter (Taylor-
        Butina is deterministic).

    Returns
    -------
    ScreenResult
        ``ScreenResult(global_=RankingMetrics, by_bin=[BinMetrics, ...],
        test_df=DataFrame, pipeline=Pipeline)``
    """
    col_cfg = column_config or ColumnConfig()
    split_cfg = split_config or SplitConfig()
    eval_cfg = eval_config or EvalConfig()

    split_df = _validate_input(df, col_cfg.smiles_col, col_cfg.target_col)
    train_df, test_df = taylor_butina_split(
        split_df,
        train_size=split_cfg.train_size,
        test_size=split_cfg.test_size,
        threshold=split_cfg.threshold,
        approximate=split_cfg.approximate,
        distance_cutoff=split_cfg.distance_cutoff,
        n_jobs=split_cfg.n_jobs,
    )

    if len(test_df) == 0:
        warnings.warn("Test set is empty after distance filtering.")
        return _empty_result(test_df)

    pipeline = _maybe_wrap_model(model, random_state)
    pipeline.fit(train_df["standardized_smiles"], train_df["target"])
    _predict_scores(pipeline, train_df, test_df, col_cfg.score_col)

    eval_result = evaluate(
        test_df,
        score_col=col_cfg.score_col,
        label_col="target",
        distance_col="distance_to_train",
        bins=eval_cfg.distance_bins,
        bedrock_alpha=eval_cfg.bedrock_alpha,
    )

    return ScreenResult(
        global_=RankingMetrics(**eval_result["global"]),
        by_bin=[BinMetrics(**b) for b in eval_result["by_bin"]],
        test_df=test_df,
        pipeline=pipeline,
    )


def _validate_input(
    df: pd.DataFrame, smiles_col: str, target_col: str
) -> pd.DataFrame:
    """Check required columns and return a clean DataFrame."""
    for col in (smiles_col, target_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    return pd.DataFrame(
        {
            "standardized_smiles": df[smiles_col],
            "target": df[target_col],
        }
    )


def _maybe_wrap_model(model, random_state: Optional[int] = None) -> Pipeline:
    """Wrap a bare sklearn estimator in a Pipeline.

    If ``model`` is already a Pipeline, returns it unchanged.
    Otherwise wraps it with a MorganFingerprintTransformer step.
    """
    if not isinstance(model, Pipeline):
        if random_state is not None and hasattr(model, "random_state"):
            model.random_state = random_state
        return Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("clf", model),
            ]
        )
    return model


def _predict_scores(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
) -> None:
    """Assign predicted probabilities to test_df (in-place).

    Handles the edge case where ``predict_proba`` returns a single
    column (model saw only one class during training).
    """
    probas = pipeline.predict_proba(test_df["standardized_smiles"])
    if probas.shape[1] == 1:
        unique_cls = train_df["target"].unique()
        if len(unique_cls) == 1:
            test_df[score_col] = float(unique_cls[0])
        else:
            test_df[score_col] = float("nan")
    else:
        test_df[score_col] = probas[:, 1]


def _empty_result(test_df: pd.DataFrame) -> ScreenResult:
    """Return a ScreenResult for an empty test set."""
    return ScreenResult(
        global_=RankingMetrics(
            n=0,
            average_precision=float("nan"),
            bedroc=float("nan"),
        ),
        by_bin=[],
        test_df=test_df,
        pipeline=None,
    )
