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
    """Load the repeated scaffold-CV folds for a dataset.

    Returns a long frame with ``standardized_smiles``, ``target``,
    ``scaffold``, ``repeat``, and ``fold`` columns (written by
    ``brood-split``): every molecule appears once per repeat.
    ``molecule_index`` is restored as the frame index, so ``frame.index``
    gives each molecule's positional row in the training split and can
    index a cached feature matrix directly.
    """
    path = data_dir / "splits" / dataset / "cv_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CV folds CSV: {path}")
    cv_folds = pd.read_csv(path)
    if "molecule_index" in cv_folds.columns:
        cv_folds = cv_folds.set_index("molecule_index")
    return cv_folds


def load_params(
    data_dir: Path,
    dataset: str,
    model_name: str,
    default: dict,
) -> dict:
    """Return tuned params overlaid on ``default``, if they exist.

    Tuned params are saved to ``data/runs/<dataset>/<model>/params.json`` by
    the full-train tuning step. Cross-validation no longer reads this file:
    tunable backends re-tune on each CV training portion via ``tune_on``.
    This loader therefore only feeds the held-out refit and untuned models.
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
) -> pd.DataFrame:
    """Evaluate a model on every repeated scaffold-CV split.

    ``predict_fn(train_df, test_df)`` fits on ``train_df`` (the CV training
    portion) and returns a DataFrame with ``predicted_probability`` for every
    row of ``test_df`` (the held-out fold). It is called once per
    (repeat, fold) sample, so a repeated 5x5 CV yields 25 samples.

    Writes raw predictions to the model run directory and derived
    ``metrics/cv.csv`` with one metric row per sample. The held-out test set
    is never read here.

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

    Returns
    -------
    pd.DataFrame
        Per-sample CV metrics for the model.
    """
    cv_folds = load_cv_folds(data_dir, dataset)
    has_repeat = "repeat" in cv_folds.columns
    repeats = sorted(cv_folds["repeat"].unique()) if has_repeat else [0]
    folds = sorted(cv_folds["fold"].unique())

    rows = []
    prediction_rows = []
    predictions_path = prediction_dir(data_dir, dataset, model_name) / "cv.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()
    for repeat in repeats:
        repeat_df = (
            cv_folds[cv_folds["repeat"] == repeat] if has_repeat else cv_folds
        )
        for fold in folds:
            train_df = repeat_df[repeat_df["fold"] != fold]
            test_df = repeat_df[repeat_df["fold"] == fold]
            if train_df.empty or test_df.empty:
                continue

            if (
                train_df["target"].nunique() < 2
                or test_df["target"].nunique() < 2
            ):
                logger.warning(
                    f"{model_name}: repeat {repeat} fold {fold} has a "
                    "single class, skipping"
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
                        "repeat": repeat,
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
                    "repeat": repeat,
                    "fold": fold,
                    "ap": ranking.average_precision,
                    "bedroc": ranking.bedroc,
                    "brier": ranking.brier,
                }
            )
            logger.info(
                f"{model_name}: repeat {repeat} fold {fold} — "
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
