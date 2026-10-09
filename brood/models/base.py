"""Model backend interface for brood.

A model backend knows how to tune, predict on held-out data, and predict one
CV fold for one model family. The unified ``brood-train`` driver looks up
backends by name in :mod:`brood.models.registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ModelBackend(ABC):
    """One trainable model family.

    Implementations must be importable without their optional dependencies
    raising (e.g. chemprop, tabpfn) so that the registry can be inspected in
    any environment. Optional dependencies should be imported lazily inside
    the methods that need them.
    """

    name: str
    #: Whether this model consumes a pre-computed feature matrix. Graph models
    #: and the similarity baseline operate on SMILES directly.
    needs_features: bool = True
    #: Whether this model supports hyperparameter tuning.
    supports_tuning: bool = True

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return fallback hyperparameters used when no params.json exists."""

    @abstractmethod
    def tune(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        n_trials: int,
        random_state: int,
    ) -> dict[str, Any]:
        """Search hyperparameters and write params.json.

        Parameters
        ----------
        dataset : str
            Dataset name.
        data_dir : Path
            Root data directory.
        representation : str | None
            Representation name, or None for SMILES-based models.
        n_trials : int
            Number of Optuna trials.
        random_state : int
            Random seed for the sampler.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters.
        """

    @abstractmethod
    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit on the full development set and predict the held-out test set.

        Parameters
        ----------
        dataset : str
            Dataset name.
        data_dir : Path
            Root data directory.
        representation : str | None
            Representation name, or None for SMILES-based models.
        params : dict[str, Any]
            Hyperparameters/arguments for the model.

        Returns
        -------
        pd.DataFrame
            Frame with ``standardized_smiles`` and ``predicted_probability``.
        """

    @abstractmethod
    def predict_fold(
        self,
        dataset: str,
        data_dir: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit on ``train_df`` and predict ``test_df`` for one CV fold.

        Parameters
        ----------
        dataset : str
            Dataset name.
        data_dir : Path
            Root data directory.
        train_df : pd.DataFrame
            Training slice for this fold.
        test_df : pd.DataFrame
            Test slice for this fold.
        representation : str | None
            Representation name, or None for SMILES-based models.
        params : dict[str, Any]
            Hyperparameters/arguments for the model.

        Returns
        -------
        pd.DataFrame
            Frame with ``standardized_smiles`` and ``predicted_probability``.
        """

    def run_name(self, representation: str | None) -> str:
        """Canonical run directory name for this backend + representation."""
        if representation is None:
            return self.name
        return f"{self.name}_{representation}"
