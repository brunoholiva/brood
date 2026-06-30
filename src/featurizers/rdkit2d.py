"""RDKit2D normalized descriptor transformer."""

import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class RDKit2DDescriptorTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to RDKit2D descriptor vectors (200-dim).

    Stateless sklearn transformer for use in Pipeline.
    NaN values are replaced with 0.0.
    """

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Convert SMILES to (n_samples, 200) float64 RDKit2D matrix."""
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        vecs = []
        for s in tqdm(X, desc="RDKit2D", unit="mol", leave=True):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {s}")
            rdkit2d = generator.process(s)[1:]
            vecs.append(np.nan_to_num(rdkit2d, nan=0.0))
        if not vecs:
            return np.zeros((0, 200), dtype=np.float64)
        return np.array(vecs, dtype=np.float64)
