#!/usr/bin/env python3
r"""Train a Random Forest on Morgan fingerprints or RDKit2D+MACCS descriptors.

Pipeline: featurizer → VarianceThreshold → SelectKBest → RF.
Optionally tune hyperparameters with Optuna (--tune).

Usage
-----
    python training/run_rf.py --dataset stokes
    python training/run_rf.py --dataset stokes --tune --n-trials 50
    python training/run_rf.py --dataset stokes --featurizer descriptors
    python training/run_rf.py --dataset stokes --featurizer descriptors
    python training/run_rf.py --dataset stokes --featurizer descriptors \
        --tune --n-trials 50
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", message="k=.* is greater than n_features=.*")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.argparse_utils import build_base_arg_parser
from src.data_utils import load_split_data, save_predictions
from src.descriptors import DescriptorTransformer
from src.fingerprints import MorganFingerprintTransformer
from src.tuning_utils import run_optuna_tuning


def _build_pipeline(featurizer="morgan", **kwargs) -> Pipeline:
    if featurizer == "morgan":
        featurizer_step = ("fps", MorganFingerprintTransformer())
    else:
        featurizer_step = ("desc", DescriptorTransformer())
    return Pipeline(
        [
            featurizer_step,
            ("var", VarianceThreshold()),
            ("kbest", SelectKBest(f_classif)),
            ("rf", RandomForestClassifier(random_state=42)),
        ]
    ).set_params(**kwargs)


def _default_params() -> dict:
    return {
        "var__threshold": 0.01,
        "kbest__k": 1024,
        "rf__n_estimators": 100,
        "rf__max_depth": None,
        "rf__min_samples_split": 2,
        "rf__min_samples_leaf": 1,
    }


def _objective(trial, train_df: pd.DataFrame, featurizer: str) -> float:
    """Optuna objective: mean 5-fold CV AP (Butina GroupKFold if available)."""
    params = {
        "var__threshold": trial.suggest_float("var__threshold", 0.0, 0.05),
        "kbest__k": trial.suggest_int("kbest__k", 100, 2048, step=100),
        "rf__n_estimators": trial.suggest_int("rf__n_estimators", 50, 500),
        "rf__max_depth": trial.suggest_int("rf__max_depth", 3, 50),
        "rf__min_samples_split": trial.suggest_int(
            "rf__min_samples_split", 2, 20
        ),
        "rf__min_samples_leaf": trial.suggest_int(
            "rf__min_samples_leaf", 1, 10
        ),
    }
    pipe = _build_pipeline(featurizer=featurizer, **params)

    smiles = train_df["standardized_smiles"]
    targets = train_df["target"]

    if "cluster_id" in train_df.columns:
        cv = GroupKFold(n_splits=5)
        groups = train_df["cluster_id"]
        scores = cross_val_score(
            pipe,
            smiles,
            targets,
            groups=groups,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
    else:
        scores = cross_val_score(
            pipe,
            smiles,
            targets,
            cv=5,
            scoring="average_precision",
            n_jobs=-1,
        )
    return float(scores.mean())


def _train_and_predict(
    train_smiles,
    train_targets,
    test_smiles,
    params: dict,
    featurizer: str = "morgan",
) -> pd.DataFrame:
    feat_name = "Morgan" if featurizer == "morgan" else "descriptors"
    logger.info(f"RF ({feat_name}): fitting on {len(train_smiles)} molecules")
    pipe = _build_pipeline(featurizer=featurizer, **params)
    pipe.fit(train_smiles, train_targets)

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles,
            "predicted_probability": pipe.predict_proba(test_smiles)[:, 1],
        }
    )


def _parse_args():
    parser = build_base_arg_parser(
        description="Train RF on Morgan fps or descriptors and predict."
    )
    parser.add_argument(
        "--featurizer",
        default="morgan",
        choices=["morgan", "descriptors"],
        help="Featurization method (default: morgan)",
    )
    return parser.parse_args()


def main() -> None:
    """Train RF, predict, save predictions."""
    args = _parse_args()
    data_dir = Path(args.data_dir)
    dataset = args.dataset
    featurizer = args.featurizer

    train_df, test_df = load_split_data(
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
        dataset,
        require_cluster_id=False,
    )

    params = _default_params()
    if args.tune:
        n_clusters = (
            train_df["cluster_id"].nunique()
            if "cluster_id" in train_df.columns
            else None
        )
        model_name = "rf" if featurizer == "morgan" else "rf_descriptors"
        params = run_optuna_tuning(
            lambda trial: _objective(trial, train_df, featurizer),
            dataset,
            model_name=model_name,
            n_trials=args.n_trials,
            random_state=args.random_state,
            n_clusters=n_clusters,
        )

    params["rf__random_state"] = args.random_state

    preds = _train_and_predict(
        train_df["standardized_smiles"],
        train_df["target"],
        test_df["standardized_smiles"],
        params,
        featurizer=featurizer,
    )

    filename = "rf.csv" if featurizer == "morgan" else "rf_descriptors.csv"
    save_predictions(preds, data_dir / "predictions" / dataset, filename)


if __name__ == "__main__":
    main()
