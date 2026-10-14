"""Fixed-representation sklearn learners and their shared tuning protocol.

Every learner here consumes one cached representation matrix. Tuning is a
small Optuna study (:func:`tune_learner`) run either once on the full
training set (``tune``, writes ``params.json``) or on each CV training
training split (``tune_on``, nothing persisted).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from brood.core.artifacts import run_dir
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.data.loader import load_split_data
from brood.models.base import ModelBackend
from brood.representations.cache import (
    get_features,
    get_train_features,
    has_cached_features,
    has_cached_train,
)
from brood.representations.registry import get_representation

logger = logging.getLogger(__name__)

#: Tuning folds used by every Optuna trial.
INNER_SPLITS = 5


@dataclass(frozen=True)
class Learner:
    """One model over one fixed representation."""

    name: str
    estimator: object
    search_space: Callable
    scaling: bool = False

    def build(self, params: dict) -> Pipeline:
        """Return an unfitted pipeline for one configuration."""
        steps = []
        if self.scaling:
            steps.append(("scale", StandardScaler()))
        steps.append((self.name, clone(self.estimator)))
        relevant = {
            key: value
            for key, value in params.items()
            if key.startswith(f"{self.name}__")
        }
        return Pipeline(steps).set_params(**relevant)


def _rf_space(trial, n_features: int) -> dict:
    return {
        "rf__n_estimators": trial.suggest_int("rf__n_estimators", 200, 1000),
        "rf__max_depth": trial.suggest_categorical(
            "rf__max_depth", [None, 8, 16, 32]
        ),
        "rf__min_samples_leaf": trial.suggest_int(
            "rf__min_samples_leaf", 1, 8
        ),
        "rf__max_features": trial.suggest_categorical(
            "rf__max_features", ["sqrt", 0.1, 0.3]
        ),
    }


def _logreg_space(trial, n_features: int) -> dict:
    return {
        "logreg__C": trial.suggest_float("logreg__C", 1e-3, 1e2, log=True),
    }


def _lgbm_space(trial, n_features: int) -> dict:
    return {
        "lgbm__n_estimators": trial.suggest_int(
            "lgbm__n_estimators", 200, 1500
        ),
        "lgbm__learning_rate": trial.suggest_float(
            "lgbm__learning_rate", 0.01, 0.2, log=True
        ),
        "lgbm__num_leaves": trial.suggest_int("lgbm__num_leaves", 15, 255),
        "lgbm__min_child_samples": trial.suggest_int(
            "lgbm__min_child_samples", 10, 100
        ),
    }


def _screening_learners() -> list[Learner]:
    """Return the fixed-representation learners available in this env."""
    learners = [
        Learner(
            name="logreg",
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=2000
            ),
            search_space=_logreg_space,
            scaling=True,
        ),
        Learner(
            name="rf",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            search_space=_rf_space,
        ),
    ]
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        pass
    else:
        learners.append(
            Learner(
                name="lgbm",
                estimator=LGBMClassifier(
                    is_unbalance=True,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
                search_space=_lgbm_space,
            )
        )
    return learners


_LEARNERS: dict[str, Learner] = {
    learner.name: learner for learner in _screening_learners()
}


def learner_by_name(name: str) -> Learner | None:
    """Look up one screening model by name."""
    return _LEARNERS.get(name)


def list_sklearn_learners() -> list[str]:
    """Return available sklearn learner names."""
    return list(_LEARNERS)


def tune_learner(
    learner: Learner,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    random_state: int = 42,
) -> dict:
    """Search ``learner``'s small space with tuning CV on AP.

    Parameters
    ----------
    learner : Learner
        Learner object with ``search_space`` and ``build`` methods.
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary target vector.
    groups : np.ndarray
        Scaffold group labels for each row.
    n_trials : int
        Number of Optuna trials.
    random_state : int
        Random seed for fold shuffling and the Optuna sampler.

    Returns
    -------
    dict
        Best hyperparameters found by Optuna.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_features = int(x.shape[1])
    n_splits = max(
        2,
        min(
            INNER_SPLITS,
            len(set(groups.tolist())),
            int(np.bincount(y).min()),
        ),
    )
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    def _objective(trial):
        params = learner.search_space(trial, n_features)
        scores = cross_val_score(
            learner.build(params),
            x,
            y,
            groups=groups,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)
    return learner.search_space(
        optuna.trial.FixedTrial(study.best_params), n_features
    )


class SklearnBackend(ModelBackend):
    """Backend for logistic regression, random forest, and LightGBM."""

    supports_tuning = True
    needs_features = True

    def __init__(self, name: str):
        self.name = name
        self._learner = learner_by_name(name)
        if self._learner is None:
            raise ValueError(
                f"Unknown sklearn learner {name!r}. "
                f"Available: {list_sklearn_learners()}"
            )

    def default_params(self) -> dict[str, Any]:
        """Return fallback hyperparameters for this model."""
        defaults = {
            "logreg__C": 1.0,
            "rf__n_estimators": 500,
            "rf__max_depth": None,
            "rf__min_samples_leaf": 1,
            "rf__max_features": "sqrt",
            "lgbm__n_estimators": 500,
            "lgbm__learning_rate": 0.05,
            "lgbm__num_leaves": 31,
            "lgbm__min_child_samples": 20,
        }
        return {k: v for k, v in defaults.items() if k.startswith(self.name)}

    def _train_matrix(
        self,
        dataset: str,
        representation: str,
        data_dir: Path,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Return train features and the raw train frame (test untouched)."""
        train_df = pd.read_csv(data_dir / "splits" / dataset / "train.csv")
        cached = has_cached_train(data_dir, dataset, representation)
        transformer = None if cached else get_representation(representation)
        x_train, _, _ = get_train_features(
            dataset, representation, transformer, data_dir
        )
        return x_train, train_df

    def _heldout_matrix(
        self,
        dataset: str,
        representation: str,
        data_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Return train/test features and raw split dataframes."""
        train_df, test_df = load_split_data(
            data_dir / "splits" / dataset / "train.csv",
            data_dir / "splits" / dataset / "test.csv",
            dataset,
            require_scaffold=False,
        )
        cached = has_cached_features(data_dir, dataset, representation)
        transformer = None if cached else get_representation(representation)
        x_train, _, x_test, _, _ = get_features(
            dataset, representation, transformer, data_dir
        )
        return x_train, x_test, train_df, test_df

    def tune_on(
        self,
        dataset: str,
        data_dir: Path,
        train_df: pd.DataFrame,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Tune on one training slice with tuning CV.

        The cached train matrix is indexed by ``train_df``'s positions, so
        this works both for the full training set and for a CV
        training split. Nothing is persisted.
        """
        if representation is None:
            raise ValueError(f"{self.name} requires a representation")
        x_all, _ = self._train_matrix(dataset, representation, data_dir)
        x = x_all[train_df.index.to_numpy()]
        groups = (
            train_df["scaffold"]
            if "scaffold" in train_df.columns
            else train_df.index.to_series()
        )
        logger.info(
            f"{self.name}: tuning {n_trials} trials on {len(x)} molecules, "
            f"{x.shape[1]} features"
        )
        tuned = tune_learner(
            self._learner,
            x,
            train_df[TARGET_COL].to_numpy(),
            groups.to_numpy(),
            n_trials=n_trials,
            random_state=random_state,
        )
        params = {**self.default_params(), **tuned}
        params[f"{self.name}__random_state"] = random_state
        return params

    def tune(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Tune on the full training set and save params."""
        if representation is None:
            raise ValueError(f"{self.name} requires a representation")
        train_df = pd.read_csv(data_dir / "splits" / dataset / "train.csv")
        params = self.tune_on(
            dataset, data_dir, train_df, representation, n_trials, random_state
        )
        out = run_dir(data_dir, dataset, self.run_name(representation))
        out.mkdir(parents=True, exist_ok=True)
        (out / "params.json").write_text(
            json.dumps(params, indent=2, sort_keys=True)
        )
        logger.info(f"{self.run_name(representation)}: params saved")
        return params

    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit on the full training set and predict the test set."""
        if representation is None:
            raise ValueError(f"{self.name} requires a representation")
        x_train, x_test, train_df, test_df = self._heldout_matrix(
            dataset, representation, data_dir
        )
        name = self.run_name(representation)
        logger.info(f"{name}: fitting on {len(x_train)} training molecules")
        pipe = self._learner.build(params)
        pipe.fit(x_train, train_df[TARGET_COL].to_numpy())
        probabilities = pipe.predict_proba(x_test)[:, 1]

        out_dir = run_dir(data_dir, dataset, name)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / "model.joblib")
        return pd.DataFrame(
            {
                SMILES_COL: test_df[SMILES_COL].to_numpy(),
                PREDICTION_COL: probabilities,
            }
        )

    def predict_fold(
        self,
        dataset: str,
        data_dir: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit on one CV fold's training slice and predict its test slice."""
        if representation is None:
            raise ValueError(f"{self.name} requires a representation")
        x_train, _ = self._train_matrix(dataset, representation, data_dir)
        pipe = self._learner.build(params)
        train_idx = train_df.index.to_numpy()
        test_idx = test_df.index.to_numpy()
        pipe.fit(x_train[train_idx], train_df[TARGET_COL].to_numpy())
        probabilities = pipe.predict_proba(x_train[test_idx])[:, 1]
        return pd.DataFrame(
            {
                SMILES_COL: test_df[SMILES_COL].to_numpy(),
                PREDICTION_COL: probabilities,
            }
        )
