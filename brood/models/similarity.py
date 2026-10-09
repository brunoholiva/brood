"""Max-Tanimoto-similarity baseline backend.

Score = maximum Tanimoto similarity between the test molecule and any active
in the training set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from brood.core.artifacts import prediction_dir
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.data.loader import load_split_data, save_predictions
from brood.models.base import ModelBackend
from brood.representations.morgan import MorganFingerprintTransformer

BATCH_SIZE = 5000


def _build_fingerprints(smiles: list[str]) -> np.ndarray:
    return MorganFingerprintTransformer().transform(smiles)


def _score_via_similarity(
    test_fps: np.ndarray, active_fps: np.ndarray, batch_size: int = BATCH_SIZE
) -> list[float]:
    """Score each test molecule by max Tanimoto sim to any training active."""
    scores = []
    n_test = test_fps.shape[0]
    for start in tqdm(
        range(0, n_test, batch_size),
        desc="Scoring by similarity",
        unit="batch",
    ):
        end = min(start + batch_size, n_test)
        dists = pairwise_distances(
            test_fps[start:end], active_fps, metric="jaccard", n_jobs=-1
        )
        scores.extend((1.0 - dists).max(axis=1).tolist())
    return scores


class SimilarityBackend(ModelBackend):
    """Backend for the max-similarity baseline."""

    name = "similarity"
    supports_tuning = False
    needs_features = False

    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size

    def default_params(self) -> dict[str, Any]:
        """Return the baseline's only configurable parameter."""
        return {"batch_size": self.batch_size}

    def _require_actives(self, train_df: pd.DataFrame) -> pd.DataFrame:
        actives = train_df[train_df[TARGET_COL] == 1]
        if actives.empty:
            raise ValueError(
                "No active molecules in training set — cannot score."
            )
        return actives

    def _predict(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        actives = self._require_actives(train_df)
        active_fps = _build_fingerprints(actives[SMILES_COL].tolist())
        test_fps = _build_fingerprints(test_df[SMILES_COL].tolist())
        scores = _score_via_similarity(test_fps, active_fps, self.batch_size)
        return pd.DataFrame(
            {
                SMILES_COL: test_df[SMILES_COL].to_numpy(),
                PREDICTION_COL: scores,
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
        """Raise an error: the similarity baseline is not tunable."""
        raise RuntimeError("The similarity baseline is not tunable.")

    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Score the held-out test set by max Tanimoto to training actives."""
        train_df, test_df = load_split_data(
            data_dir / "splits" / dataset / "train.csv",
            data_dir / "splits" / dataset / "test.csv",
            dataset,
            require_scaffold=False,
        )
        preds = self._predict(train_df, test_df)
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
        """Score one CV fold by max Tanimoto to its training actives."""
        return self._predict(train_df, test_df)
