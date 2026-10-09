"""Smoke tests that featurized rows align with SMILES and target order."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brood.representations import get_representation


@pytest.mark.parametrize("name", ["morgan2048", "maccs", "rdkit2d"])
def test_batch_features_align_with_smiles_order(name):
    """Batch-transformed rows match individually transformed molecules."""
    smiles = ["CCO", "CCCO", "c1ccccc1"]
    targets = [0, 1, 0]
    df = pd.DataFrame({"standardized_smiles": smiles, "target": targets})

    featurizer = get_representation(name)
    x_batch = featurizer.transform(df["standardized_smiles"].tolist())
    y = df["target"].to_numpy(dtype=np.float64)

    assert x_batch.shape[0] == len(smiles)
    assert np.array_equal(y, np.array(targets, dtype=np.float64))

    for i, smi in enumerate(smiles):
        x_one = featurizer.transform([smi])
        assert x_one.shape[0] == 1
        assert np.array_equal(x_batch[i], x_one[0])


def test_feature_cache_preserves_smiles_and_target_order(tmp_path):
    """Feature cache returns arrays in the same order as the input split."""
    from brood.representations.cache import get_features

    smiles = ["CCO", "CCCO", "c1ccccc1"]
    targets = [0, 1, 0]

    splits = tmp_path / "splits" / "demo"
    splits.mkdir(parents=True)
    train = pd.DataFrame({"standardized_smiles": smiles, "target": targets})
    # Empty test set is enough for this alignment check.
    test = train.iloc[:0].copy()
    train.to_csv(splits / "train.csv", index=False)
    test.to_csv(splits / "test.csv", index=False)

    featurizer = get_representation("morgan2048")
    x_train, y_train, x_test, smiles_train, smiles_test = get_features(
        "demo", "morgan2048", featurizer, tmp_path
    )

    assert list(smiles_train) == smiles
    assert np.array_equal(y_train, np.array(targets, dtype=np.float64))
    assert x_train.shape[0] == len(smiles)
    assert x_test.shape[0] == 0
    assert smiles_test.shape[0] == 0

    # A second call reuses the cache but the order is unchanged.
    x_train2, y_train2, _, smiles_train2, _ = get_features(
        "demo", "morgan2048", None, tmp_path
    )
    assert np.array_equal(x_train, x_train2)
    assert np.array_equal(y_train, y_train2)
    assert list(smiles_train2) == list(smiles_train)
