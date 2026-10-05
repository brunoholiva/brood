"""Fixed-representation sklearn learners and their shared tuning protocol.

Every learner here consumes one cached representation matrix. Tuning is light
and not nested: one small Optuna study per family writes ``params.json``; the
shared outer 5-fold CV reuses those fixed parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from brood.core.artifacts import run_dir, save_metadata
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.data.loader import load_split_data
from brood.models.base import ModelBackend
from brood.representations.cache import get_features, has_cached_features
from brood.representations.registry import get_representation

#: Inner folds used by every Optuna trial.
INNER_SPLITS = 5


@dataclass(frozen=True)
class Learner:
    """One model family over one fixed representation."""

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
    """Look up one screening learner by family name."""
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
    """Search ``learner``'s small space with inner scaffold CV on AP."""
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
        """Return fallback hyperparameters for this learner family."""
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

    def _load_params(
        self, data_dir: Path, dataset: str, representation: str | None
    ) -> dict[str, Any]:
        """Overlay saved tuned params on the defaults."""
        name = self.run_name(representation)
        path = run_dir(data_dir, dataset, name) / "params.json"
        params = self.default_params()
        if path.exists():
            params.update(json.loads(path.read_text()))
            logger.info(f"Using tuned params from {path}")
        return params

    def _load_features(
        self,
        dataset: str,
        representation: str,
        data_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cached (X_train, X_test) for one representation."""
        cached = has_cached_features(data_dir, dataset, representation)
        transformer = None if cached else get_representation(representation)
        x_train, _, x_test, _, _ = get_features(
            dataset, representation, transformer, data_dir
        )
        return x_train, x_test

    def _feature_matrix(
        self,
        dataset: str,
        representation: str,
        data_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Return cached train/test features and raw split dataframes."""
        train_df, test_df = load_split_data(
            data_dir / "splits" / f"{dataset}_train.csv",
            data_dir / "splits" / f"{dataset}_test.csv",
            dataset,
            require_scaffold=False,
        )
        x_train, x_test = self._load_features(
            dataset, representation, data_dir
        )
        return x_train, x_test, train_df, test_df

    def tune(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Search hyperparameters with inner scaffold CV and save params."""
        if representation is None:
            raise ValueError(f"{self.name} requires a representation")
        x_train, _, train_df, _ = self._feature_matrix(
            dataset, representation, data_dir
        )
        groups = (
            train_df["scaffold"]
            if "scaffold" in train_df.columns
            else train_df.index.to_series()
        )
        logger.info(
            f"{self.run_name(representation)}: tuning {n_trials} trials on "
            f"{len(x_train)} molecules, {x_train.shape[1]} features"
        )
        tuned = tune_learner(
            self._learner,
            x_train,
            train_df[TARGET_COL].to_numpy(),
            groups.to_numpy(),
            n_trials=n_trials,
            random_state=random_state,
        )
        params = {**self.default_params(), **tuned}
        params[f"{self.name}__random_state"] = random_state
        if self.name == "logreg":
            params["logreg__random_state"] = random_state

        out = run_dir(data_dir, dataset, self.run_name(representation))
        out.mkdir(parents=True, exist_ok=True)
        (out / "params.json").write_text(
            json.dumps(params, indent=2, sort_keys=True)
        )
        logger.success(f"{self.run_name(representation)}: params saved")
        return params

    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit on the full development set and predict the test set."""
        if representation is None:
            raise ValueError(f"{self.name} requires a representation")
        x_train, x_test, train_df, test_df = self._feature_matrix(
            dataset, representation, data_dir
        )
        name = self.run_name(representation)
        logger.info(f"{name}: fitting on {len(x_train)} development molecules")
        pipe = self._learner.build(params)
        pipe.fit(x_train, train_df[TARGET_COL].to_numpy())
        probabilities = pipe.predict_proba(x_test)[:, 1]

        out_dir = run_dir(data_dir, dataset, name)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / "model.joblib")
        save_metadata(
            out_dir,
            "metadata",
            {
                "model": name,
                "params": params,
                "representation": representation,
            },
        )
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
        x_train, _, train_full_df, _ = self._feature_matrix(
            dataset, representation, data_dir
        )
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


def model_name(learner_name: str, representation: str) -> str:
    """Run name for one learner x representation pair."""
    return f"{learner_name}_{representation}"


def default_params() -> dict[str, Any]:
    """Pre-specified defaults used when no tuned params exist."""
    backend = SklearnBackend("rf")
    return backend.default_params()
