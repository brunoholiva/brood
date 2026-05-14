"""SMILES-to-fingerprint transformers for molecular ML pipelines."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.base import BaseEstimator, TransformerMixin


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES strings to Morgan fingerprint bit vectors.

    Uses ECFP4 (Morgan radius 2, 2048 bits) by default, matching the
    fingerprint used internally by the Taylor-Butina splitter.

    Designed for use in :class:`~sklearn.pipeline.Pipeline`:

    .. code:: python

        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("fps", MorganFingerprintTransformer()),
            ("clf", RandomForestClassifier()),
        ])
        pipe.fit(smiles, targets)

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
        """No-op. Stateless transformer.

        Parameters
        ----------
        X : array-like of str
            SMILES strings.
        y : array-like, optional
            Ignored.

        Returns
        -------
        self : MorganFingerprintTransformer
        """
        return self

    def transform(self, X):
        """Convert SMILES strings to Morgan fingerprint bit matrix.

        Parameters
        ----------
        X : array-like of str
            SMILES strings.

        Returns
        -------
        fps : ndarray of shape (n_samples, fp_size) and dtype int64
        """
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
