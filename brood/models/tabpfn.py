"""TabPFN model backend and training utilities.

TabPFN uses in-context learning (no gradient descent). It is not tuned; the
checkpoint version is a declared methodological choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from brood.core.artifacts import prediction_dir, run_dir, save_metadata
from brood.core.constants import PREDICTION_COL, SMILES_COL, TARGET_COL
from brood.cv.loop import load_cv_folds
from brood.data.loader import save_predictions
from brood.models.base import ModelBackend
from brood.representations.cache import get_features
from brood.representations.registry import get_representation

_DEFAULT_VERSION = "v3.5"
_MAX_PRODUCT = 200_000_000


def resolve_device(device: str) -> str:
    """Resolve 'auto' to cuda or cpu; pass through others."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def default_tabpfn_params(device: str = "auto", random_state: int = 0) -> dict:
    """Return default TabPFNClassifier kwargs."""
    return {
        "device": resolve_device(device),
        "random_state": random_state,
    }


def check_product_limit(
    n_train: int, n_features: int, max_product: int = _MAX_PRODUCT
) -> None:
    """Warn if n×d product exceeds TabPFN's practical limit."""
    product = n_train * n_features
    if product > max_product:
        logger.warning(
            f"n×d product {product:,} exceeds recommended {max_product:,}. "
            "TabPFN may be slow or feature positional encodings may degrade."
        )


def _model_version(version: str | None):
    from tabpfn.constants import ModelVersion

    return ModelVersion(version or _DEFAULT_VERSION)


def _run_name(representation: str, version: str | None) -> str:
    name = f"tabpfn_{representation}"
    version = version or _DEFAULT_VERSION
    if version != _DEFAULT_VERSION:
        name = f"{name}_{version}"
    return name


def train_and_predict_tabpfn(
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    test_smiles: list[str],
    *,
    tabpfn_params: dict[str, Any] | None = None,
    model_version: Any = None,
) -> pd.DataFrame:
    """Fit TabPFN on pre-computed features and return predictions."""
    from tabpfn import TabPFNClassifier

    params = {**default_tabpfn_params(), **(tabpfn_params or {})}
    model_version = model_version or _model_version(None)

    check_product_limit(X_train.shape[0], X_train.shape[1])
    logger.info(
        f"{dataset} | TabPFN {model_version.value}: "
        "fitting (in-context learning)"
    )
    model = TabPFNClassifier.create_default_for_version(
        model_version, **params
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return pd.DataFrame({SMILES_COL: test_smiles, PREDICTION_COL: probs})


class TabPFNBackend(ModelBackend):
    """Backend for TabPFN in-context learning."""

    supports_tuning = False
    needs_features = True
    name = "tabpfn"

    def __init__(self, device: str = "auto", version: str | None = None):
        self.device = device
        self.version = version or _DEFAULT_VERSION

    def default_params(self) -> dict[str, Any]:
        """Return pinned TabPFN parameters (device + random state)."""
        return default_tabpfn_params(device=self.device, random_state=42)

    def _make_featurizer(self, representation: str):
        if representation == "chemeleon":
            return get_representation(
                "chemeleon", device=resolve_device(self.device)
            )
        return get_representation(representation)

    def _load_features(
        self,
        dataset: str,
        representation: str,
        data_dir: Path,
        force: bool = False,
    ):
        featurizer = self._make_featurizer(representation)
        return get_features(
            dataset,
            representation,
            featurizer,
            data_dir,
            force=force,
        )

    def tune(self, *args, **kwargs) -> dict[str, Any]:
        """TabPFN does not support hyperparameter tuning."""
        raise RuntimeError(
            "TabPFN uses fixed in-context params; tuning is not supported."
        )

    def run_name(self, representation: str | None) -> str:
        """Return the run name, with a version suffix when needed."""
        if representation is None:
            raise ValueError("tabpfn requires a representation")
        return _run_name(representation, self.version)

    def predict_held_out(
        self,
        dataset: str,
        data_dir: Path,
        representation: str | None,
        params: dict[str, Any],
    ) -> pd.DataFrame:
        """Fit TabPFN on the full development set and predict the test set."""
        if representation is None:
            raise ValueError("tabpfn requires a representation")
        x_train, y_train, x_test, _, smiles_test = self._load_features(
            dataset, representation, data_dir
        )
        preds = train_and_predict_tabpfn(
            dataset,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            test_smiles=list(smiles_test),
            tabpfn_params=params,
            model_version=_model_version(self.version),
        )
        name = self.run_name(representation)
        save_predictions(
            preds, prediction_dir(data_dir, dataset, name), "test.csv"
        )
        save_metadata(
            run_dir(data_dir, dataset, name),
            "metadata",
            {
                "model": name,
                "representation": representation,
                "version": self.version,
                "params": params,
            },
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
        """Fit TabPFN on one CV fold and predict its test slice."""
        if representation is None:
            raise ValueError("tabpfn requires a representation")
        x_train, _, _, _, _ = self._load_features(
            dataset, representation, data_dir
        )
        cv_folds = load_cv_folds(data_dir, dataset)
        if len(cv_folds) != len(x_train):
            raise ValueError(
                f"Cached train features ({len(x_train)} rows) do not match "
                f"CV folds ({len(cv_folds)} rows); re-split and re-run."
            )
        train_idx = train_df.index.to_numpy()
        test_idx = test_df.index.to_numpy()
        return train_and_predict_tabpfn(
            dataset,
            X_train=x_train[train_idx],
            y_train=train_df[TARGET_COL].to_numpy(),
            X_test=x_train[test_idx],
            test_smiles=test_df[SMILES_COL].tolist(),
            tabpfn_params=params,
            model_version=_model_version(self.version),
        )
