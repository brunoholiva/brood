"""Scaffold-aware cross-validation evaluation shared across models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger

from .evaluator import _compute_ranking_metrics


def load_cv_folds(data_dir: Path, dataset: str) -> pd.DataFrame:
    """Load the fixed scaffold-CV folds for a dataset.

    Returns a frame with ``standardized_smiles``, ``target``, ``scaffold``,
    and ``fold`` columns (written by ``scripts/split_datasets.py``).
    """
    path = data_dir / "splits" / f"{dataset}_cv_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CV folds CSV: {path}")
    return pd.read_csv(path)


def load_params(
    data_dir: Path,
    dataset: str,
    model_name: str,
    default: dict,
) -> dict:
    """Return Optuna-tuned params overlaid on ``default``, if they exist.

    Tuned params are saved to
    ``data/predictions/<dataset>/<model>_params.json`` by
    :func:`src.tuning_utils.run_optuna_tuning`.
    """
    path = data_dir / "predictions" / dataset / f"{model_name}_params.json"
    if path.exists():
        with open(path) as f:
            default = {**default, **json.load(f)}
        logger.info(f"Using tuned params from {path}")
    return default


# ponytail: identity of each fold's test split
def run_cv_model(
    dataset: str,
    data_dir: Path,
    model_name: str,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    n_folds: int = 5,
) -> pd.DataFrame:
    """Evaluate a model on the fixed scaffold-CV folds.

    ``predict_fn(train_df, test_df)`` fits on ``train_df`` and returns a
    DataFrame with ``predicted_probability`` for every row of ``test_df``.

    Writes ``data/cv/<dataset>/<model_name>.csv`` with one ``(model, fold,
    ap, bedroc)`` row per fold and returns it.
    """
    cv_folds = load_cv_folds(data_dir, dataset)
    rows = []
    for fold in range(n_folds):
        train_df = cv_folds[cv_folds["fold"] != fold]
        test_df = cv_folds[cv_folds["fold"] == fold]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        if train_df["target"].nunique() < 2 or test_df["target"].nunique() < 2:
            logger.warning(
                f"{model_name}: fold {fold} has a single class, skipping"
            )
            continue

        preds = predict_fn(train_df, test_df)
        y_true = test_df["target"].to_numpy()
        ranking = _compute_ranking_metrics(
            y_true, preds["predicted_probability"].to_numpy(), 20.0
        )
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "ap": ranking.average_precision,
                "bedroc": ranking.bedroc,
            }
        )
        logger.info(
            f"{model_name}: fold {fold} — "
            f"AP={ranking.average_precision:.4f} "
            f"BEDROC={ranking.bedroc:.4f}"
        )

    if not rows:
        raise RuntimeError(f"{model_name}: no usable CV folds")

    out = pd.DataFrame(rows)
    out_dir = data_dir / "cv" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.csv"
    out.to_csv(out_path, index=False)
    logger.success(f"CV results for {model_name} saved to {out_path}")
    return out
