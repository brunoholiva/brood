#!/usr/bin/env python3
r"""Train a Random Forest on Morgan fingerprints or combined descriptors.

Pipeline: featurizer → VarianceThreshold → SelectKBest → RF.
Optionally tune hyperparameters with Optuna (--tune).

Featurizers:
    morgan      — Morgan fingerprints (ECFP4, 512 bits)
    descriptors — Morgan(512) + RDKit2D(200) + MACCS(167) = 879-dim

Usage
-----
    python training/run_rf.py --dataset stokes
    python training/run_rf.py --dataset stokes --tune --n-trials 50
    python training/run_rf.py --dataset stokes --featurizer descriptors
    python training/run_rf.py --dataset stokes --featurizer descriptors \
        --tune --n-trials 50
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", message="k=.* is greater than n_features=.*")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.argparse_utils import build_base_arg_parser
from src.artifacts import save_metadata
from src.cv_utils import load_params, run_cv_model
from src.data_utils import load_split_data, save_predictions
from src.featurizers import MoleculeFeaturizer, MorganFingerprintTransformer
from src.tuning_utils import run_optuna_tuning


def _build_pipeline(featurizer="morgan", **kwargs) -> Pipeline:
    if featurizer == "morgan":
        featurizer_step = ("fps", MorganFingerprintTransformer())
    else:
        featurizer_step = ("desc", MoleculeFeaturizer())
    return Pipeline(
        [
            featurizer_step,
            ("var", VarianceThreshold()),
            ("kbest", SelectKBest(f_classif)),
            (
                "rf",
                RandomForestClassifier(
                    class_weight="balanced_subsample", random_state=42
                ),
            ),
        ]
    ).set_params(**kwargs)


def _default_params() -> dict:
    return {
        "var__threshold": 0.01,
        "kbest__k": 512,
        "rf__n_estimators": 100,
        "rf__max_depth": None,
        "rf__min_samples_split": 2,
        "rf__min_samples_leaf": 1,
    }


def _objective(
    trial,
    train_df: pd.DataFrame,
    featurizer: str,
    n_splits: int = 3,
) -> float:
    """Optuna objective: mean inner scaffold CV AP."""
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

    if "scaffold" in train_df.columns:
        n_splits = min(
            n_splits,
            train_df["scaffold"].nunique(),
            train_df["target"].value_counts().min(),
        )
        if n_splits < 2:
            return 0.0
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        groups = train_df["scaffold"]
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
    out_dir: Path | None = None,
) -> pd.DataFrame:
    feat_name = "Morgan" if featurizer == "morgan" else "descriptors"
    logger.info(f"RF ({feat_name}): fitting on {len(train_smiles)} molecules")
    pipe = _build_pipeline(featurizer=featurizer, **params)
    pipe.fit(train_smiles, train_targets)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name = "rf" if featurizer == "morgan" else "rf_descriptors"
        model_path = out_dir / f"{model_name}.joblib"
        joblib.dump(pipe, model_path)
        save_metadata(
            out_dir,
            model_name,
            {
                "model": model_name,
                "artifact": model_path.name,
                "featurizer": featurizer,
                "params": params,
            },
        )
        logger.success(f"Model saved to {model_path}")

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
        require_scaffold=False,
    )

    params = _default_params()
    model_name = "rf" if featurizer == "morgan" else "rf_descriptors"
    if args.tune and not args.cv:
        params = run_optuna_tuning(
            lambda trial: _objective(trial, train_df, featurizer),
            dataset,
            model_name=model_name,
            n_trials=args.n_trials,
            random_state=args.random_state,
            params_path=data_dir / "predictions" / dataset,
        )

    params["rf__random_state"] = args.random_state

    if args.cv:
        if args.tune:
            from src.cv_utils import run_nested_cv_model

            run_nested_cv_model(
                dataset,
                data_dir,
                model_name,
                lambda tr, te, fold_params: _train_and_predict(
                    tr["standardized_smiles"],
                    tr["target"],
                    te["standardized_smiles"],
                    {**fold_params, "rf__random_state": args.random_state},
                    featurizer=featurizer,
                ),
                lambda tr: run_optuna_tuning(
                    lambda trial: _objective(trial, tr, featurizer),
                    dataset,
                    model_name,
                    args.n_trials,
                    args.random_state,
                ),
            )
            return
        params = load_params(data_dir, dataset, model_name, params)
        run_cv_model(
            dataset,
            data_dir,
            model_name,
            lambda tr, te: _train_and_predict(
                tr["standardized_smiles"],
                tr["target"],
                te["standardized_smiles"],
                params,
                featurizer=featurizer,
            ),
        )
        return

    out_dir = data_dir / "predictions" / dataset
    preds = _train_and_predict(
        train_df["standardized_smiles"],
        train_df["target"],
        test_df["standardized_smiles"],
        params,
        featurizer=featurizer,
        out_dir=out_dir,
    )

    filename = "rf.csv" if featurizer == "morgan" else "rf_descriptors.csv"
    save_predictions(preds, out_dir, filename)


if __name__ == "__main__":
    main()
