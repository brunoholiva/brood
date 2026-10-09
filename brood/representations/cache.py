"""Pre-compute and cache featurized datasets for fast experiment iteration.

Caches ``X_train.npy``, ``y_train.npy``, ``smiles_train.npy``,
``X_test.npy``, and ``smiles_test.npy`` under
``data/features/<dataset>/<featurizer_name>/``.

The train half is cached independently of the test half so the CV paths can
featurize only ``train.csv`` and never touch the held-out split.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _data_path(data_dir: Path, dataset: str, featurizer_name: str) -> Path:
    """Return the cache directory for a given dataset + featurizer."""
    return data_dir / "features" / dataset / featurizer_name


def _train_paths(cache_dir: Path):
    return (
        cache_dir / "X_train.npy",
        cache_dir / "y_train.npy",
        cache_dir / "smiles_train.npy",
    )


def _test_paths(cache_dir: Path):
    return (
        cache_dir / "X_test.npy",
        cache_dir / "smiles_test.npy",
    )


def _split_paths(data_dir: Path, dataset: str) -> tuple[Path, Path]:
    return (
        data_dir / "splits" / dataset / "train.csv",
        data_dir / "splits" / dataset / "test.csv",
    )


def _all_exist(paths) -> bool:
    return all(path.exists() for path in paths)


def _load_train(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_path, y_path, smiles_path = _train_paths(cache_dir)
    return (
        np.load(x_path),
        np.load(y_path),
        np.load(smiles_path, allow_pickle=True),
    )


def _save_train(
    cache_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    smiles_train: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path, smiles_path = _train_paths(cache_dir)
    np.save(x_path, x_train)
    np.save(y_path, y_train)
    np.save(smiles_path, smiles_train)


def has_cached_features(
    data_dir: Path, dataset: str, featurizer_name: str
) -> bool:
    """Check whether cached train *and* test features exist."""
    cache_dir = _data_path(data_dir, dataset, featurizer_name)
    if not cache_dir.is_dir():
        return False
    return _all_exist(_train_paths(cache_dir) + _test_paths(cache_dir))


def has_cached_train(
    data_dir: Path, dataset: str, featurizer_name: str
) -> bool:
    """Check whether cached train features exist (test split may be absent)."""
    cache_dir = _data_path(data_dir, dataset, featurizer_name)
    return _all_exist(_train_paths(cache_dir))


def get_train_features(
    dataset: str,
    featurizer_name: str,
    featurizer,
    data_dir: Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cached or freshly-computed **train-only** features.

    Reads only ``splits/<dataset>/train.csv``. The held-out test split is
    never touched, so this is the accessor for tuning and cross-validation.

    Returns
    -------
    tuple of np.ndarray
        ``(X_train, y_train, smiles_train)``
    """
    cache_dir = _data_path(data_dir, dataset, featurizer_name)
    if not force and _all_exist(_train_paths(cache_dir)):
        logger.info(
            f"{dataset} | Loading cached train features from {cache_dir}"
        )
        return _load_train(cache_dir)

    train_path, _ = _split_paths(data_dir, dataset)
    train_df = pd.read_csv(train_path)
    logger.info(f"{dataset} | Featurizing {len(train_df)} train molecules")
    x_train = featurizer.transform(train_df["standardized_smiles"].tolist())
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    smiles_train = train_df["standardized_smiles"].to_numpy()
    _save_train(cache_dir, x_train, y_train, smiles_train)
    logger.info(f"{dataset} | Cached train features to {cache_dir}")
    return x_train, y_train, smiles_train


def get_features(
    dataset: str,
    featurizer_name: str,
    featurizer,
    data_dir: Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cached or freshly-computed train **and** test features.

    Reads both splits. This is the held-out accessor; CV must use
    :func:`get_train_features` instead so the test split stays untouched.

    Returns
    -------
    tuple of np.ndarray
        ``(X_train, y_train, X_test, smiles_train, smiles_test)``
    """
    cache_dir = _data_path(data_dir, dataset, featurizer_name)
    if not force and has_cached_features(data_dir, dataset, featurizer_name):
        logger.info(f"{dataset} | Loading cached features from {cache_dir}")
        x_train, y_train, smiles_train = _load_train(cache_dir)
        x_path, smiles_path = _test_paths(cache_dir)
        x_test = np.load(x_path)
        smiles_test = np.load(smiles_path, allow_pickle=True)
        return (x_train, y_train, x_test, smiles_train, smiles_test)

    if not force and has_cached_train(data_dir, dataset, featurizer_name):
        x_train, y_train, smiles_train = _load_train(cache_dir)
    else:
        train_path, _ = _split_paths(data_dir, dataset)
        train_df = pd.read_csv(train_path)
        logger.info(f"{dataset} | Featurizing {len(train_df)} train molecules")
        x_train = featurizer.transform(
            train_df["standardized_smiles"].tolist()
        )
        y_train = train_df["target"].to_numpy(dtype=np.float64)
        smiles_train = train_df["standardized_smiles"].to_numpy()
        _save_train(cache_dir, x_train, y_train, smiles_train)

    _, test_path = _split_paths(data_dir, dataset)
    test_df = pd.read_csv(test_path)
    logger.info(f"{dataset} | Featurizing {len(test_df)} test molecules")
    x_test = featurizer.transform(test_df["standardized_smiles"].tolist())
    smiles_test = test_df["standardized_smiles"].to_numpy()

    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, smiles_path = _test_paths(cache_dir)
    np.save(x_path, x_test)
    np.save(smiles_path, smiles_test)
    logger.info(f"{dataset} | Cached features to {cache_dir}")
    return (x_train, y_train, x_test, smiles_train, smiles_test)
