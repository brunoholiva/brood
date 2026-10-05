"""Pre-compute and cache featurized datasets for fast experiment iteration.

Caches ``X_train.npy``, ``y_train.npy``, ``X_test.npy``,
``smiles_train.npy``, ``smiles_test.npy`` under
``data/features/<dataset>/<featurizer_name>/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


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
        data_dir / "splits" / f"{dataset}_train.csv",
        data_dir / "splits" / f"{dataset}_test.csv",
    )


def has_cached_features(
    data_dir: Path, dataset: str, featurizer_name: str
) -> bool:
    """Check whether cached features exist for this dataset+featurizer."""
    cache_dir = _data_path(data_dir, dataset, featurizer_name)
    if not cache_dir.is_dir():
        return False
    x_train_path, y_train_path, smiles_train_path = _train_paths(cache_dir)
    x_test_path, smiles_test_path = _test_paths(cache_dir)
    return all(
        p.exists()
        for p in [
            x_train_path,
            y_train_path,
            smiles_train_path,
            x_test_path,
            smiles_test_path,
        ]
    )


def get_features(
    dataset: str,
    featurizer_name: str,
    featurizer,
    data_dir: Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cached or freshly-computed features for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name (matches split filenames).
    featurizer_name : str
        Cache subdirectory name, e.g. ``"morgan_512_descriptors"``.
    featurizer : TransformerMixin
        Stateless sklearn-compatible featurizer (used only if cache
        misses).
    data_dir : Path
        Root data directory containing ``splits/`` and ``features/``.
    force : bool
        If ``True``, recompute and overwrite cache.

    Returns
    -------
    tuple of np.ndarray
        ``(X_train, y_train, X_test, smiles_train, smiles_test)``
    """
    cache_dir = _data_path(data_dir, dataset, featurizer_name)

    if not force and has_cached_features(data_dir, dataset, featurizer_name):
        logger.info(f"{dataset} | Loading cached features from {cache_dir}")
        x_train = np.load(cache_dir / "X_train.npy")
        y_train = np.load(cache_dir / "y_train.npy")
        smiles_train = np.load(
            cache_dir / "smiles_train.npy", allow_pickle=True
        )
        x_test = np.load(cache_dir / "X_test.npy")
        smiles_test = np.load(cache_dir / "smiles_test.npy", allow_pickle=True)
        return (x_train, y_train, x_test, smiles_train, smiles_test)

    train_path, test_path = _split_paths(data_dir, dataset)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        f"{dataset} | Featurizing {len(train_df)} train, "
        f"{len(test_df)} test molecules"
    )
    x_train = featurizer.transform(train_df["standardized_smiles"].tolist())
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    smiles_train = train_df["standardized_smiles"].to_numpy()

    x_test = featurizer.transform(test_df["standardized_smiles"].tolist())
    smiles_test = test_df["standardized_smiles"].to_numpy()

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_train.npy", x_train)
    np.save(cache_dir / "y_train.npy", y_train)
    np.save(cache_dir / "smiles_train.npy", smiles_train)
    np.save(cache_dir / "X_test.npy", x_test)
    np.save(cache_dir / "smiles_test.npy", smiles_test)
    logger.success(f"{dataset} | Cached features to {cache_dir}")

    return (x_train, y_train, x_test, smiles_train, smiles_test)
