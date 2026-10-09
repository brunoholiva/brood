"""Minimol representation as an sklearn transformer.

Wraps the graphcore-research/minimol model so it produces 512-dim
float32 embeddings for a list of SMILES strings. The underlying model is
loaded lazily on the first ``transform`` call.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin


class MinimolFingerprint(BaseEstimator, TransformerMixin):
    """sklearn transformer that featurizes molecules with Minimol embeddings.

    Minimol is the 10M-parameter pretrained fingerprinting model from
    graphcore-research/minimol. Embeddings are 512-dim float vectors.
    The model is loaded lazily on first ``transform`` call.

    Note: upstream Minimol never moves the model to GPU (Lightning is
    bypassed), so this runs on CPU at roughly 25-30 mol/s.

    Parameters
    ----------
    batch_size : int, default=100
        Molecules per Minimol forward pass.
    """

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from minimol import Minimol

            self._model = Minimol(batch_size=self.batch_size)
        return self._model

    def fit(self, X=None, y=None):
        """No-op: model is loaded lazily during transform."""
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        """Featurize SMILES strings into Minimol embeddings.

        Parameters
        ----------
        X : list of str
            SMILES strings.

        Returns
        -------
        np.ndarray of shape (n_molecules, 512)
        """
        smiles = list(X)
        for i, s in enumerate(smiles):
            if Chem.MolFromSmiles(s) is None:
                raise ValueError(f"Invalid SMILES at position {i}")
        if not smiles:
            return np.zeros((0, 512), dtype=np.float32)

        model = self._load_model()
        results = model(smiles)
        if len(results) != len(smiles):
            raise ValueError(
                f"Minimol returned {len(results)} embeddings "
                f"for {len(smiles)} molecules; refusing to misalign rows"
            )
        return np.stack([np.asarray(r) for r in results])
