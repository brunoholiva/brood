"""Optuna hyperparameter tuning utilities."""

from __future__ import annotations

from typing import Any, Callable

import mlflow
import optuna
from loguru import logger


def run_optuna_tuning(
    objective: Callable[[optuna.Trial], float],
    dataset: str,
    model_name: str,
    n_trials: int,
    random_state: int,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search and log results to MLflow.

    Parameters
    ----------
    objective : Callable[[optuna.Trial], float]
        Optuna objective function (maximized).
    dataset : str
        Dataset name for logging.
    model_name : str
        Model name for MLflow tags.
    n_trials : int
        Number of Optuna trials.
    random_state : int
        Random seed for TPESampler.
    n_clusters : int | None
        If provided, logs cv_strategy as "Butina GroupKFold".

    Returns
    -------
    dict[str, Any]
        Best hyperparameters from Optuna.
    """
    if n_clusters is not None:
        logger.info(
            f"{dataset} | Optuna: {n_trials} trials, "
            f"5-fold Butina GroupKFold ({n_clusters} clusters)"
        )
    else:
        logger.info(f"{dataset} | Optuna: {n_trials} trials, 5-fold CV")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_cv_ap = study.best_value

    logger.success(f"Tuning complete: best CV AP = {best_cv_ap:.4f}")
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(f"{dataset} | Best params: {param_str}")

    with mlflow.start_run(
        run_name=f"tuning/{model_name}/{dataset}", nested=False
    ):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_average_precision", best_cv_ap)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv_folds", 5)
        if n_clusters is not None:
            mlflow.log_param("cv_strategy", "Butina GroupKFold")
        mlflow.set_tags(
            {"dataset": dataset, "model": model_name, "stage": "tuning"}
        )

    return best_params
