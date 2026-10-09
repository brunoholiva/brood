"""Morgan fingerprint transformer (ECFP4)."""

import numpy as np
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from brood.representations._utils import require_mols


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to Morgan fingerprint vectors (ECFP4).

    Stateless sklearn transformer for use in Pipeline.

    Parameters
    ----------
    radius : int, default=2
        Morgan fingerprint radius.
    fp_size : int, default=512
        Number of fingerprint bits.
    """

    def __init__(self, radius=2, fp_size=512):
        self.radius = radius
        self.fp_size = fp_size

    def fit(self, X=None, y=None):
        """No-op: transformer is stateless."""
        return self

    def transform(self, X):
        """Convert SMILES strings to (n_samples, fp_size) int64 matrix."""
        mols = require_mols(X)
        if not mols:
            return np.zeros((0, self.fp_size), dtype=np.int64)

        generator = AllChem.GetMorganGenerator(
            radius=self.radius, fpSize=self.fp_size
        )
        desc = f"Morgan {self.fp_size}"
        fps = [
            generator.GetFingerprint(m)
            for m in tqdm(mols, desc=desc, unit="mol", leave=True)
        ]
        return np.array(fps, dtype=np.int64)
