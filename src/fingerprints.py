"""SMILES-to-fingerprint transformers for molecular ML pipelines."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, TransformerMixin


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to Morgan fingerprint bit vectors (ECFP4).

    Stateless sklearn transformer for use in Pipeline.

    Parameters
    ----------
    radius : int, default=2
        Morgan fingerprint radius.
    fp_size : int, default=2048
        Number of fingerprint bits.
    """

    def __init__(self, radius=2, fp_size=2048):
        self.radius = radius
        self.fp_size = fp_size

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Convert SMILES strings to (n_samples, fp_size) int64 matrix."""
        generator = AllChem.GetMorganGenerator(
            radius=self.radius, fpSize=self.fp_size
        )
        mols = [Chem.MolFromSmiles(s) for s in X]
        _validate_mols(mols)
        if len(mols) == 0:
            return np.zeros((0, self.fp_size), dtype=np.int64)
        return np.array([generator.GetFingerprint(m) for m in mols])


def _validate_mols(mols):
    """Raise ValueError if any RDKit Mol parsed as None."""
    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {i}")
