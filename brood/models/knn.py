"""Morgan-fingerprint K-nearest-neighbor backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from brood.core.artifacts import prediction_dir, run_dir
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.data.loader import load_split_data, save_predictions
from brood.models.base import ModelBackend
from brood.representations.morgan import MorganFingerprintTransformer

K_VALUES = (1, 3, 5)
WEIGHTS = ("uniform", "distance")
INNER_SPLITS = 5


def _build_fingerprints(smiles: list[str]) -> np.ndarray:
    return (
        MorganFingerprintTransformer(fp_size=2048)
        .transform(smiles)
        .astype(bool)
    )


def _build_model(n_neighbors: int, weights: str) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="jaccard",
        algorithm="brute",
        n_jobs=-1,
    )


def _valid_k_values(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, random_state: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
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
    splits = list(cv.split(x, y, groups))
    min_train_size = min(len(train_idx) for train_idx, _ in splits)
    values = [k for k in K_VALUES if k <= min_train_size]
    return splits, values or [1]


def _tune(
    train_df: pd.DataFrame,
    n_trials: int,
    random_state: int,
) -> dict[str, Any]:
    import optuna

    x = _build_fingerprints(train_df[SMILES_COL].tolist())
    y = train_df[TARGET_COL].to_numpy()
    groups = train_df.get("scaffold", train_df.index.to_series()).to_numpy()
    splits, k_values = _valid_k_values(x, y, groups, random_state)

    def objective(trial):
        model = _build_model(
            trial.suggest_categorical("n_neighbors", k_values),
            trial.suggest_categorical("weights", list(WEIGHTS)),
        )
        scores = cross_val_score(
            model,
            x,
            y,
            groups=groups,
            cv=splits,
            scoring="average_precision",
            n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return {
        "n_neighbors": study.best_params["n_neighbors"],
        "weights": study.best_params["weights"],
    }


class KNNBackend(ModelBackend):
    """KNN using Morgan fingerprints and Jaccard distance."""

    name = "knn"
    supports_tuning = True
    needs_features = False

    def default_params(self) -> dict[str, Any]:
        """Return the default neighbor count and weighting method."""
        return {"n_neighbors": 5, "weights": "uniform"}

    def _predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        train_fps = _build_fingerprints(train_df[SMILES_COL].tolist())
        test_fps = _build_fingerprints(test_df[SMILES_COL].tolist())
        model = _build_model(params["n_neighbors"], params["weights"])
        model.fit(train_fps, train_df[TARGET_COL].to_numpy())
        probabilities = model.predict_proba(test_fps)
        active_column = list(model.classes_).index(1)
        return pd.DataFrame(
            {
                SMILES_COL: test_df[SMILES_COL].to_numpy(),
                PREDICTION_COL: probabilities[:, active_column],
            }
        )

    def tune(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Tune KNN on the full training set and save the best parameters."""
        train_df = pd.read_csv(data_dir / "splits" / dataset / "train.csv")
        params = self.tune_on(
            dataset,
            data_dir,
            train_df,
            representation,
            n_trials,
            random_state,
        )
        out = run_dir(data_dir, dataset, self.name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "params.json").write_text(json.dumps(params, indent=2) + "\n")
        return params

    def tune_on(
        self,
        dataset: str,
        data_dir: Path,
        train_df: pd.DataFrame,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Tune KNN on one CV training portion without saving parameters."""
        del dataset, data_dir, representation
        return _tune(train_df, n_trials, random_state)

    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit KNN on the training set and predict the held-out test set."""
        train_df, test_df = load_split_data(
            data_dir / "splits" / dataset / "train.csv",
            data_dir / "splits" / dataset / "test.csv",
            dataset,
            require_scaffold=False,
        )
        preds = self._predict(train_df, test_df, params)
        save_predictions(
            preds, prediction_dir(data_dir, dataset, self.name), "test.csv"
        )
        return preds

    def predict_fold(
        self,
        dataset: str,
        data_dir: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit KNN on one CV training portion and predict its held-out fold."""
        del dataset, data_dir, representation
        return self._predict(train_df, test_df, params)
