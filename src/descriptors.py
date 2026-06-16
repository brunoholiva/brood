"""SMILES-to-descriptor transformers for molecular ML pipelines."""

import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin


class DescriptorTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to RDKit2D + MACCS descriptor vectors (367-dim).

    Stateless sklearn transformer for use in Pipeline.
    NaN values are replaced with 0.0.
    """

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Convert SMILES to (n_samples, 367) float64 descriptor matrix."""
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        vecs = []
        for s in X:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {s}")

            rdkit2d = generator.process(s)[1:]
            maccs_vec = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(maccs_vec, maccs_arr)

            merged = np.concatenate((rdkit2d, maccs_arr))
            vecs.append(np.nan_to_num(merged, nan=0.0))

        if not vecs:
            return np.zeros((0, 367), dtype=np.float64)
        return np.array(vecs)
