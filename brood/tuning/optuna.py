"""Optuna hyperparameter tuning utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import optuna

logger = logging.getLogger(__name__)


def run_optuna_tuning(
    objective: Callable[[optuna.Trial], float],
    dataset: str,
    model_name: str,
    n_trials: int,
    random_state: int,
    params_path: Path | None = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search and save best params to JSON.

    Parameters
    ----------
    objective : Callable[[optuna.Trial], float]
        Optuna objective function (maximized).
    dataset : str
        Dataset name for logging.
    model_name : str
        Model name for the params filename.
    n_trials : int
        Number of Optuna trials.
    random_state : int
        Random seed for TPESampler.
    params_path : Path | None
        If provided, saves the best parameters. When it ends in ``.json``
        the file is used verbatim; otherwise the file
        ``<model_name>_params.json`` is written under this directory.

    Returns
    -------
    dict[str, Any]
        Best hyperparameters from Optuna.
    """
    logger.info(f"{dataset} | Optuna: {n_trials} trials")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_cv_ap = study.best_value

    logger.info(f"Tuning complete: best CV AP = {best_cv_ap:.4f}")
    param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(f"{dataset} | Best params: {param_str}")

    if params_path is not None:
        import json

        out_path = (
            params_path
            if params_path.suffix == ".json"
            else params_path / f"{model_name}_params.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(best_params, f, indent=2, sort_keys=True)
        logger.info(f"Best params saved to {out_path}")

    return best_params
