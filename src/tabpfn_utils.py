"""Shared TabPFN training utilities.

Reusable logic for featurizing, fitting, and predicting with TabPFN
across different featurization strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tabpfn import TabPFNClassifier

_MAX_PRODUCT = 200_000_000


def resolve_device(device: str) -> str:
    """Resolve 'auto' to cuda or cpu; pass through others."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def default_tabpfn_params(device: str = "auto") -> dict:
    """Return default TabPFNClassifier kwargs with resolved device."""
    resolved = resolve_device(device)
    return {
        "device": resolved,
        "ignore_pretraining_limits": True,
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


def _logits_to_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis."""
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def train_and_predict_tabpfn(
    dataset: str,
    train_smiles: list[str] | None = None,
    train_targets: list[int] | None = None,
    test_smiles: list[str] | None = None,
    featurizer=None,
    *,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    tabpfn_params: dict | None = None,
    balance_probabilities: bool = False,
) -> pd.DataFrame:
    """Featurize (or use pre-computed), fit TabPFN, return predictions.

    Parameters
    ----------
    dataset : str
        Dataset name for logging.
    train_smiles : list of str, optional
        Training SMILES. Required if ``X_train`` is not provided.
    train_targets : list of int, optional
        Training binary labels. Required if ``y_train`` is not provided.
    test_smiles : list of str, optional
        Test SMILES. Required if ``X_test`` is not provided.
    featurizer : TransformerMixin, optional
        Stateless sklearn-compatible featurizer. Required if
        ``X_train``/``X_test`` are not provided.
    X_train : np.ndarray, optional
        Pre-computed training features (n_train, n_features).
    y_train : np.ndarray, optional
        Pre-computed training labels (n_train,).
    X_test : np.ndarray, optional
        Pre-computed test features (n_test, n_features).
    tabpfn_params : dict or None
        Parameters passed to TabPFNClassifier. If None, uses defaults.
    balance_probabilities : bool
        Passed to TabPFNClassifier.

    Returns
    -------
    pd.DataFrame
        Columns: standardized_smiles, predicted_probability,
        predicted_probability_std.
    """
    if tabpfn_params is None:
        tabpfn_params = default_tabpfn_params()

    if X_train is not None:
        pass
    elif featurizer is not None and train_smiles is not None:
        logger.info(
            "{} | TabPFN: featurizing {} training molecules".format(
                dataset, len(train_smiles)
            )
        )
        X_train = featurizer.transform(train_smiles)
        if train_targets is not None:
            y_train = np.array(train_targets)
            feat_name = type(featurizer).__name__
            logger.info(
                f"{dataset} | TabPFN ({feat_name}): "
                f"{X_train.shape[0]} molecules, "
                f"{X_train.shape[1]} features"
            )
    else:
        raise ValueError(
            "Provide either pre-computed X_train or "
            "(train_smiles + featurizer)"
        )

    check_product_limit(X_train.shape[0], X_train.shape[1])

    logger.info(f"{dataset} | TabPFN: fitting (in-context learning)")
    model = TabPFNClassifier(
        device=tabpfn_params["device"],
        ignore_pretraining_limits=tabpfn_params["ignore_pretraining_limits"],
        balance_probabilities=balance_probabilities,
    )
    model.fit(X_train, y_train)

    if X_test is not None:
        pass
    elif featurizer is not None and test_smiles is not None:
        logger.info(
            f"{dataset} | TabPFN: featurizing "
            f"{len(test_smiles)} test molecules"
        )
        X_test = featurizer.transform(test_smiles)
    else:
        raise ValueError(
            "Provide either pre-computed X_test or "
            "(test_smiles + featurizer)"
        )

    raw_logits = model.predict_raw_logits(X_test)
    temperature = model.softmax_temperature_
    per_est_probs = _logits_to_probs(raw_logits, temperature)[:, :, 1]
    mean_probs = per_est_probs.mean(axis=0)
    std_probs = per_est_probs.std(axis=0)

    if test_smiles is None:
        test_smiles_list = [str(s) for s in range(len(X_test))]
    else:
        test_smiles_list = test_smiles

    return pd.DataFrame(
        {
            "standardized_smiles": test_smiles_list,
            "predicted_probability": mean_probs,
            "predicted_probability_std": std_probs,
        }
    )
