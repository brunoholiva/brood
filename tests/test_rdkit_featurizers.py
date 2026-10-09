"""Tests for RDKit featurizers and the registry."""

import numpy as np
import pytest

from brood.representations import REPRESENTATION_REGISTRY, get_representation

SMILES = [
    "CCO",
    "CCN",
    "c1ccccc1O",
    "CCCNC(=O)C",
    "O=C(O)c1ccccc1N",
    "C1CC1",
    "CC(F)Cl",
]

# non-chemeleon representations and their expected output dimensions
DIMS = {
    "morgan512": 512,
    "maccs": 167,
    "rdkit2d": 200,
    "morgan_maccs_rdkit2d": 879,
}


@pytest.mark.parametrize("name", sorted(DIMS))
class TestRegistry:
    """Registry exposes every non-chemeleon representation with right dims."""

    def test_known_representation(self, name):
        assert name in REPRESENTATION_REGISTRY

    def test_transform_shape(self, name):
        feats = get_representation(name).transform(SMILES)
        assert feats.shape == (len(SMILES), DIMS[name])

    def test_no_nan(self, name):
        feats = get_representation(name).transform(SMILES)
        assert not np.isnan(feats).any()

    def test_deterministic(self, name):
        t = get_representation(name)
        assert np.array_equal(t.transform(SMILES), t.transform(SMILES))

    def test_empty_input(self, name):
        assert get_representation(name).transform([]).shape == (0, DIMS[name])


@pytest.mark.parametrize("name", sorted(DIMS))
def test_invalid_smiles_raises(name):
    """Invalid SMILES raise a ValueError naming its position."""
    with pytest.raises(ValueError, match="Invalid SMILES at position 1"):
        get_representation(name).transform(["CCO", "not_a_molecule"])


def test_unknown_representation_raises():
    with pytest.raises(ValueError, match="Unknown representation"):
        get_representation("nonexistent")
