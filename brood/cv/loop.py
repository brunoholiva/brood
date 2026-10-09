"""Scaffold-aware cross-validation evaluation shared across models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from brood.core.artifacts import prediction_dir, run_dir
from brood.evaluation.evaluator import _compute_ranking_metrics

logger = logging.getLogger(__name__)


def load_cv_folds(data_dir: Path, dataset: str) -> pd.DataFrame:
    """Load the fixed scaffold-CV folds for a dataset.

    Returns a frame with ``standardized_smiles``, ``target``, ``scaffold``,
    and ``fold`` columns (written by ``brood-split``). The
    row order matches the development-split CSV, so a cached feature matrix
    can be indexed with the fold frame's index.
    """
    path = data_dir / "splits" / dataset / "cv_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CV folds CSV: {path}")
    return pd.read_csv(path)


def load_params(
    data_dir: Path,
    dataset: str,
    model_name: str,
    default: dict,
) -> dict:
    """Return tuned params overlaid on ``default``, if they exist.

    Tuned params are saved to ``data/runs/<dataset>/<model>/params.json`` by
    the training script. There is no nested-CV variant: one fixed parameter
    set is reused by every outer fold, which is what the study protocol
    assumes.
    """
    path = run_dir(data_dir, dataset, model_name) / "params.json"
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        if all(not isinstance(value, dict) for value in saved.values()):
            default = {**default, **saved}
            logger.info(f"Using tuned params from {path}")
        else:
            logger.warning(
                f"Ignoring nested-CV params at {path}; using defaults."
            )
    return default


def run_cv_model(
    dataset: str,
    data_dir: Path,
    model_name: str,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    n_folds: int = 5,
) -> pd.DataFrame:
    """Evaluate a model on the fixed scaffold-CV folds of the dev set.

    ``predict_fn(train_df, test_df)`` fits on ``train_df`` and returns a
    DataFrame with ``predicted_probability`` for every row of ``test_df``.

    Writes raw fold predictions to the model run directory and derived
    ``metrics/cv.csv`` with one metric row per fold. The held-out test set is
    never read here.

    Parameters
    ----------
    dataset : str
        Dataset name.
    data_dir : Path
        Root data directory containing ``splits/`` and ``runs/``.
    model_name : str
        Run name used for output paths.
    predict_fn : Callable
        Function taking train/test frames and returning predictions.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    pd.DataFrame
        Concatenated CV predictions for all folds.
    """
    cv_folds = load_cv_folds(data_dir, dataset)
    rows = []
    prediction_rows = []
    predictions_path = prediction_dir(data_dir, dataset, model_name) / "cv.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()
    for fold in range(n_folds):
        train_df = cv_folds[cv_folds["fold"] != fold]
        test_df = cv_folds[cv_folds["fold"] == fold]
        if train_df.empty or test_df.empty:
            continue

        if train_df["target"].nunique() < 2 or test_df["target"].nunique() < 2:
            logger.warning(
                f"{model_name}: fold {fold} has a single class, skipping"
            )
            continue

        preds = predict_fn(train_df, test_df)
        prediction_rows.extend(
            pd.DataFrame(
                {
                    "standardized_smiles": test_df[
                        "standardized_smiles"
                    ].to_numpy(),
                    "target": test_df["target"].to_numpy(),
                    "fold": fold,
                    "predicted_probability": preds[
                        "predicted_probability"
                    ].to_numpy(),
                }
            ).to_dict("records")
        )
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
        ranking = _compute_ranking_metrics(
            test_df["target"].to_numpy(),
            preds["predicted_probability"].to_numpy(),
            20.0,
        )
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "ap": ranking.average_precision,
                "bedroc": ranking.bedroc,
                "brier": ranking.brier,
            }
        )
        logger.info(
            f"{model_name}: fold {fold} — "
            f"AP={ranking.average_precision:.4f} "
            f"BEDROC={ranking.bedroc:.4f} "
            f"Brier={ranking.brier:.4f}"
        )

    if not rows:
        raise RuntimeError(f"{model_name}: no usable CV folds")

    out = pd.DataFrame(rows)
    out_dir = run_dir(data_dir, dataset, model_name) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"CV results for {model_name} saved to {out_path}")
    return out
