"""Molecular featurizer transformers and registry for brood."""

from src.featurizers.concatenated import (
    ConcatenatedFeaturizer as ConcatenatedFeaturizer,
)
from src.featurizers.maccs import MACCSFingerprintTransformer
from src.featurizers.rdkit2d import RDKit2DDescriptorTransformer

FEATURIZER_REGISTRY = {
    "maccs": {
        "cls": MACCSFingerprintTransformer,
        "dim": 167,
        "env": "tabpfn",
    },
    "rdkit2d": {
        "cls": RDKit2DDescriptorTransformer,
        "dim": 200,
        "env": "tabpfn",
    },
}


def get_featurizer(name: str, **kwargs):
    """Instantiate a featurizer by registry name with optional kwargs."""
    entry = FEATURIZER_REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown featurizer {name!r}. "
            f"Available: {list(FEATURIZER_REGISTRY)}"
        )
    return entry["cls"](**kwargs)
