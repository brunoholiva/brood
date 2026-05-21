#!/usr/bin/env python3
"""Train a Random Forest on Morgan fingerprints and save predictions.

Pipeline: Morgan fingerprints → VarianceThreshold → SelectKBest → RF.

Optionally tune hyperparameters with Optuna (--tune).

Usage
-----
    python training/run_rf.py --dataset stokes
    python training/run_rf.py --dataset stokes --tune --n-trials 50
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings(
    "ignore", message="k=.* is greater than n_features=.*"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.fingerprints import MorganFingerprintTransformer  # noqa: E402


def _build_pipeline(**kwargs) -> Pipeline:
    """Build an RF pipeline with feature selection."""
    return Pipeline([
        ("fps", MorganFingerprintTransformer()),
        ("var", VarianceThreshold()),
        ("kbest", SelectKBest(f_classif)),
        ("rf", RandomForestClassifier(random_state=42)),
    ]).set_params(**kwargs)


def _default_params() -> dict:
    return {
        "var__threshold": 0.01,
        "kbest__k": 1024,
        "rf__n_estimators": 100,
        "rf__max_depth": None,
        "rf__min_samples_split": 2,
        "rf__min_samples_leaf": 1,
    }


def _objective(trial, smiles, targets) -> float:
    """Optuna objective: mean 5-fold CV AP."""
    params = {
        "var__threshold": trial.suggest_float(
            "var__threshold", 0.0, 0.05
        ),
        "kbest__k": trial.suggest_int("kbest__k", 100, 2048, step=100),
        "rf__n_estimators": trial.suggest_int(
            "rf__n_estimators", 50, 500
        ),
        "rf__max_depth": trial.suggest_int("rf__max_depth", 3, 30),
        "rf__min_samples_split": trial.suggest_int(
            "rf__min_samples_split", 2, 20
        ),
        "rf__min_samples_leaf": trial.suggest_int(
            "rf__min_samples_leaf", 1, 10
        ),
    }
    pipe = _build_pipeline(**params)
    scores = cross_val_score(
        pipe, smiles, targets, cv=5, scoring="average_precision", n_jobs=-1
    )
    return scores.mean()


def _load_data(
    train_path: Path, test_path: Path, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate train/test CSVs."""
    if not train_path.exists():
        logger.error(f"{train_path} not found. Run split_datasets.py first.")
        sys.exit(1)
    if not test_path.exists():
        logger.error(f"{test_path} not found. Run split_datasets.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_train_actives = int(train_df["target"].sum())
    n_test_actives = int(test_df["target"].sum())

    logger.info(
        f"{dataset}: train={len(train_df)} ({n_train_actives} actives)"
    )
    logger.info(
        f"{dataset}: test={len(test_df)} ({n_test_actives} actives)"
    )

    return train_df, test_df


def _run_tuning(
    smiles, targets, dataset: str, n_trials: int, random_state: int
) -> dict:
    """Run Optuna hyperparameter search and log results to MLflow."""
    logger.info(f"Tuning with Optuna ({n_trials} trials, 5-fold CV)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        lambda trial: _objective(trial, smiles, targets),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_cv_ap = study.best_value

    logger.success(f"Tuning done: best CV AP = {best_cv_ap:.4f}")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    with mlflow.start_run(
        run_name=f"tuning/rf/{dataset}", nested=False
    ):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_average_precision", best_cv_ap)
        mlflow.log_param("n_trials", n_trials)
        mlflow.set_tags(
            {"dataset": dataset, "model": "rf", "stage": "tuning"}
        )

    return best_params


def _train_and_predict(
    train_smiles, train_targets, test_smiles, params: dict
) -> pd.DataFrame:
    """Build pipeline, train, and return predictions."""
    pipe = _build_pipeline(**params)
    logger.info("Training final model...")
    pipe.fit(train_smiles, train_targets)

    return pd.DataFrame({
        "standardized_smiles": test_smiles,
        "predicted_probability": pipe.predict_proba(test_smiles)[:, 1],
    })


def _save_predictions(
    preds: pd.DataFrame, out_dir: Path, filename: str
) -> Path:
    """Write predictions CSV and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    preds.to_csv(out_path, index=False)
    logger.success(f"Predictions saved to {out_path}")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF on Morgan fingerprints and predict."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g. stokes)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state (default: 42)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning before final training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of tuning trials (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    """Train RF, predict, save predictions."""
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset

    train_df, test_df = _load_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
    )

    params = _default_params()
    if args.tune:
        params = _run_tuning(
            train_df["standardized_smiles"],
            train_df["target"],
            dataset,
            args.n_trials,
            args.random_state,
        )

    params["rf__random_state"] = args.random_state

    preds = _train_and_predict(
        train_df["standardized_smiles"],
        train_df["target"],
        test_df["standardized_smiles"],
        params,
    )

    _save_predictions(
        preds, data_dir / "predictions" / dataset, "rf.csv"
    )


if __name__ == "__main__":
    main()
