"""MACCS key fingerprint transformer."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class MACCSFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to MACCS key bit vectors (167 bits).

    Stateless sklearn transformer for use in Pipeline.
    """

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Convert SMILES to (n_samples, 167) int64 MACCS bit matrix."""
        mols = [Chem.MolFromSmiles(s) for s in X]
        _validate_mols(mols)
        if not mols:
            return np.zeros((0, 167), dtype=np.int64)
        vecs = []
        for mol in tqdm(mols, desc="MACCS", unit="mol", leave=True):
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
            vecs.append(arr)
        return np.array(vecs, dtype=np.int64)


def _validate_mols(mols):
    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {i}")
