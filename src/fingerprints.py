"""SMILES-to-fingerprint transformers for molecular ML pipelines."""

import warnings

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", message="please use MorganGenerator"
)


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES to Morgan fingerprint vectors (ECFP4).

    Stateless sklearn transformer for use in Pipeline.

    Parameters
    ----------
    radius : int, default=2
        Morgan fingerprint radius.
    fp_size : int, default=2048
        Number of fingerprint bits.
    use_counts : bool, default=False
        If True, produce count vectors instead of bit vectors.
    """

    def __init__(self, radius=2, fp_size=2048, use_counts=False):
        self.radius = radius
        self.fp_size = fp_size
        self.use_counts = use_counts

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Convert SMILES strings to (n_samples, fp_size) int64 matrix."""
        mols = [Chem.MolFromSmiles(s) for s in X]
        _validate_mols(mols)
        if len(mols) == 0:
            return np.zeros((0, self.fp_size), dtype=np.int64)

        if self.use_counts:
            desc = f"Morgan(count) {self.fp_size}"
            vecs = []
            for mol in tqdm(mols, desc=desc, unit="mol", leave=True):
                fp = AllChem.GetHashedMorganFingerprint(
                    mol, radius=self.radius, nBits=self.fp_size
                )
                arr = np.zeros((self.fp_size,), dtype=np.int64)
                DataStructs.ConvertToNumpyArray(fp, arr)
                vecs.append(arr)
            return np.array(vecs, dtype=np.int64)
        else:
            generator = AllChem.GetMorganGenerator(
                radius=self.radius, fpSize=self.fp_size
            )
            desc = f"Morgan {self.fp_size}"
            fps = [
                generator.GetFingerprint(m)
                for m in tqdm(mols, desc=desc, unit="mol", leave=True)
            ]
            return np.array(fps, dtype=np.int64)


def _validate_mols(mols):
    """Raise ValueError if any RDKit Mol parsed as None."""
    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {i}")
