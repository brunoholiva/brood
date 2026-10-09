"""MACCS key fingerprint transformer."""

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from brood.representations._utils import require_mols


class MACCSFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to MACCS key bit vectors (167 bits).

    Stateless sklearn transformer for use in Pipeline.
    """

    def fit(self, X=None, y=None):
        """No-op: transformer is stateless."""
        return self

    def transform(self, X):
        """Convert SMILES to (n_samples, 167) int64 MACCS bit matrix."""
        mols = require_mols(X)
        if not mols:
            return np.zeros((0, 167), dtype=np.int64)

        def _maccs(mol):
            arr = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
            return arr

        vecs = [
            _maccs(mol)
            for mol in tqdm(mols, desc="MACCS", unit="mol", leave=True)
        ]
        return np.array(vecs, dtype=np.int64)
