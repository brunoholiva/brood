"""Morgan-fingerprint K-nearest-neighbor backend."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, pairwise_distances
from sklearn.model_selection import StratifiedGroupKFold

from brood.core.artifacts import prediction_dir, run_dir
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.data.loader import load_split_data, save_predictions
from brood.models.base import ModelBackend
from brood.representations.cache import (
    get_train_features,
    has_cached_train,
)
from brood.representations.morgan import MorganFingerprintTransformer

logger = logging.getLogger(__name__)

K_VALUES = (1, 3, 5)
WEIGHTS = ("uniform", "distance")
INNER_SPLITS = 5
K_MAX = max(K_VALUES)
DISTANCE_BLOCK_SIZE = 128
CACHE_VERSION = "morgan2048-top5-v1"


def _build_fingerprints(smiles: list[str]) -> np.ndarray:
    return (
        MorganFingerprintTransformer(fp_size=2048)
        .transform(smiles)
        .astype(bool)
    )


def _valid_splits(
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
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
    return list(cv.split(np.zeros(len(y)), y, groups))


def _cache_path(
    data_dir: Path,
    dataset: str,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> Path:
    digest = hashlib.sha256(
        CACHE_VERSION.encode()
        + query_ids.astype(np.int64, copy=False).tobytes()
        + candidate_ids.astype(np.int64, copy=False).tobytes()
    ).hexdigest()[:20]
    cache_dir = data_dir / "features" / dataset / "knn_neighbors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{digest}.npz"


def _nearest_neighbors(
    query_features: np.ndarray,
    candidate_features: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        return cached["indices"], cached["similarities"]

    n_neighbors = min(K_MAX, len(candidate_ids))
    neighbor_ids = np.empty((len(query_ids), n_neighbors), dtype=np.int64)
    similarities = np.empty((len(query_ids), n_neighbors), dtype=np.float32)
    logger.info(
        "KNN: calculating %d neighbors for %d queries against %d candidates",
        n_neighbors,
        len(query_ids),
        len(candidate_ids),
    )
    for start in range(0, len(query_ids), DISTANCE_BLOCK_SIZE):
        stop = min(start + DISTANCE_BLOCK_SIZE, len(query_ids))
        distances = pairwise_distances(
            query_features[start:stop],
            candidate_features,
            metric="jaccard",
            n_jobs=-1,
        )
        nearest = np.argpartition(distances, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        order = np.argsort(nearest_distances, axis=1)
        nearest = np.take_along_axis(nearest, order, axis=1)
        nearest_distances = np.take_along_axis(
            nearest_distances, order, axis=1
        )
        neighbor_ids[start:stop] = candidate_ids[nearest]
        similarities[start:stop] = 1.0 - nearest_distances

    if cache_path is not None:
        temporary = cache_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, indices=neighbor_ids, similarities=similarities
            )
        temporary.replace(cache_path)
        logger.info("KNN: cached neighbors to %s", cache_path)
    return neighbor_ids, similarities


def _score_neighbors(
    labels: np.ndarray,
    neighbor_ids: np.ndarray,
    similarities: np.ndarray,
    n_neighbors: int,
    weights: str,
) -> np.ndarray:
    similarities = similarities[:, :n_neighbors]
    neighbor_labels = labels[neighbor_ids[:, :n_neighbors]]
    if weights == "uniform":
        return neighbor_labels.mean(axis=1)

    distances = 1.0 - similarities
    zero_distance = distances == 0
    scores = np.empty(len(neighbor_labels), dtype=float)
    has_exact_match = zero_distance.any(axis=1)
    if has_exact_match.any():
        exact_labels = neighbor_labels[has_exact_match]
        exact_mask = zero_distance[has_exact_match]
        scores[has_exact_match] = (exact_labels * exact_mask).sum(
            axis=1
        ) / exact_mask.sum(axis=1)
    if (~has_exact_match).any():
        non_exact_distances = distances[~has_exact_match]
        weights_array = 1.0 / non_exact_distances
        scores[~has_exact_match] = (
            neighbor_labels[~has_exact_match] * weights_array
        ).sum(axis=1) / weights_array.sum(axis=1)
    return scores


def _tune(
    train_df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    data_dir: Path,
    dataset: str,
    n_trials: int,
    random_state: int,
) -> dict[str, Any]:
    import optuna

    global_ids = train_df.index.to_numpy(dtype=np.int64)
    y = labels[global_ids]
    groups = train_df.get("scaffold", train_df.index.to_series()).to_numpy()
    splits = _valid_splits(y, groups, random_state)
    cached_folds = []
    for inner_fold, (train_idx, validation_idx) in enumerate(splits):
        candidate_ids = global_ids[train_idx]
        query_ids = global_ids[validation_idx]
        cache_path = _cache_path(
            data_dir,
            dataset,
            query_ids,
            candidate_ids,
        )
        neighbor_ids, similarities = _nearest_neighbors(
            features[query_ids],
            features[candidate_ids],
            query_ids,
            candidate_ids,
            cache_path,
        )
        cached_folds.append(
            (query_ids, neighbor_ids, similarities, inner_fold)
        )
        logger.info("KNN: inner fold %d neighbor cache ready", inner_fold)

    def objective(trial):
        n_neighbors = trial.suggest_categorical("n_neighbors", K_VALUES)
        weights = trial.suggest_categorical("weights", list(WEIGHTS))
        scores = []
        for query_ids, neighbor_ids, similarities, _ in cached_folds:
            predictions = _score_neighbors(
                labels,
                neighbor_ids,
                similarities,
                n_neighbors,
                weights,
            )
            scores.append(
                average_precision_score(labels[query_ids], predictions)
            )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(
            {
                "n_neighbors": list(K_VALUES),
                "weights": list(WEIGHTS),
            }
        ),
    )
    study.optimize(
        objective,
        n_trials=min(n_trials, len(K_VALUES) * len(WEIGHTS)),
        show_progress_bar=True,
    )
    return {
        "n_neighbors": study.best_params["n_neighbors"],
        "weights": study.best_params["weights"],
    }


class KNNBackend(ModelBackend):
    """KNN using Morgan fingerprints and Jaccard distance."""

    name = "knn"
    supports_tuning = True
    needs_features = False

    def __init__(self):
        self._feature_key = None
        self._features = None
        self._labels = None

    def default_params(self) -> dict[str, Any]:
        """Return the default neighbor count and weighting method."""
        return {"n_neighbors": 5, "weights": "uniform"}

    def _load_training_features(
        self, dataset: str, data_dir: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (str(data_dir), dataset)
        if self._feature_key == key:
            return self._features, self._labels
        cached = has_cached_train(data_dir, dataset, "morgan2048")
        transformer = (
            None if cached else MorganFingerprintTransformer(fp_size=2048)
        )
        features, labels, _ = get_train_features(
            dataset, "morgan2048", transformer, data_dir
        )
        self._feature_key = key
        self._features = np.asarray(features, dtype=bool)
        self._labels = labels.astype(int, copy=False)
        return self._features, self._labels

    def _predict_frames(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        train_features = _build_fingerprints(train_df[SMILES_COL].tolist())
        test_features = _build_fingerprints(test_df[SMILES_COL].tolist())
        train_ids = np.arange(len(train_df), dtype=np.int64)
        test_ids = np.arange(len(test_df), dtype=np.int64)
        neighbor_ids, similarities = _nearest_neighbors(
            test_features, train_features, test_ids, train_ids
        )
        labels = train_df[TARGET_COL].to_numpy()
        probabilities = _score_neighbors(
            labels,
            neighbor_ids,
            similarities,
            params["n_neighbors"],
            params["weights"],
        )
        return pd.DataFrame(
            {
                SMILES_COL: test_df[SMILES_COL].to_numpy(),
                PREDICTION_COL: probabilities,
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
        del representation
        features, labels = self._load_training_features(dataset, data_dir)
        return _tune(
            train_df,
            features,
            labels,
            data_dir,
            dataset,
            n_trials,
            random_state,
        )

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
        preds = self._predict_frames(train_df, test_df, params)
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
        del representation
        features, labels = self._load_training_features(dataset, data_dir)
        train_ids = train_df.index.to_numpy(dtype=np.int64)
        test_ids = test_df.index.to_numpy(dtype=np.int64)
        cache_path = _cache_path(data_dir, dataset, test_ids, train_ids)
        neighbor_ids, similarities = _nearest_neighbors(
            features[test_ids],
            features[train_ids],
            test_ids,
            train_ids,
            cache_path,
        )
        probabilities = _score_neighbors(
            labels,
            neighbor_ids,
            similarities,
            params["n_neighbors"],
            params["weights"],
        )
        return pd.DataFrame(
            {
                SMILES_COL: test_df[SMILES_COL].to_numpy(),
                PREDICTION_COL: probabilities,
            }
        )
