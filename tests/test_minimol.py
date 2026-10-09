"""Tests for the Minimol representation.

Requires the minimol conda environment (torch, graphium, minimol).
Skipped automatically in other environments.
"""

import importlib.util

import pytest

if importlib.util.find_spec("minimol") is None:
    pytest.skip("requires minimol (minimol env)", allow_module_level=True)

import numpy as np

from brood.representations import get_representation

SMILES = [
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc(-c2cccnc2)cc1",
]


@pytest.fixture(scope="module")
def featurizer():
    """One shared transformer so the model loads only once."""
    return get_representation("minimol")


def test_transform_shape(featurizer):
    out = featurizer.transform(SMILES)
    assert out.shape == (len(SMILES), 512)
    assert np.isfinite(out).all()


def test_empty_input(featurizer):
    out = featurizer.transform([])
    assert out.shape == (0, 512)


def test_invalid_smiles_raises(featurizer):
    with pytest.raises(ValueError, match="Invalid SMILES at position 1"):
        featurizer.transform(["CCO", "not_a_smiles", "c1ccccc1"])


def test_batch_matches_single_rows(featurizer):
    batch = featurizer.transform(SMILES)
    singles = np.stack([featurizer.transform([s])[0] for s in SMILES])
    np.testing.assert_allclose(batch, singles, rtol=1e-4, atol=1e-5)
