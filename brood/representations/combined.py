"""Combined Morgan + RDKit2D + MACCS featurizer (879-dim default).

Parses each SMILES once and reuses the Mol object for all three
feature types.  Inspired by EVA's MoleculeFeaturizer.
"""

from __future__ import annotations

import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

N_RDKIT_DESCRIPTORS = 200
N_MACCS = 167


class MoleculeFeaturizer(BaseEstimator, TransformerMixin):
    """Morgan fingerprints + RDKit2D normalized descriptors + MACCS keys.

    Concatenates three representations into a single feature vector:
    - Morgan fingerprint (default 512 bits, radius 2)
    - RDKit 2D normalized descriptors (200-dim, via descriptastorus)
    - MACCS structural keys (167-bit)

    Default output dimension: 512 + 200 + 167 = 879.

    Parameters
    ----------
    fp_size : int, default=512
        Morgan fingerprint bits.
    radius : int, default=2
        Morgan fingerprint radius.
    """

    def __init__(self, fp_size: int = 512, radius: int = 2):
        self.fp_size = fp_size
        self.radius = radius

    @property
    def n_features(self) -> int:
        """Total number of output features."""
        return self.fp_size + N_RDKIT_DESCRIPTORS + N_MACCS

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Convert SMILES to (n, n_features) float64 matrix.

        Parameters
        ----------
        X : list of str
            SMILES strings.

        Returns
        -------
        np.ndarray of shape (len(X), n_features)
        """
        n = len(X)
        if n == 0:
            return np.zeros((0, self.n_features), dtype=np.float64)

        generator = AllChem.GetMorganGenerator(
            radius=self.radius, fpSize=self.fp_size
        )
        rdkit_gen = rdNormalizedDescriptors.RDKit2DNormalized()

        vecs = []
        for i, s in enumerate(
            tqdm(X, desc="Featurize", unit="mol", leave=True)
        ):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES at position {i}")

            # Morgan
            morgan = np.array(generator.GetFingerprint(mol), dtype=np.float64)

            # RDKit2D
            rdkit2d = np.nan_to_num(rdkit_gen.process(s)[1:], nan=0.0)

            # MACCS
            maccs_arr = np.zeros((N_MACCS,))
            DataStructs.ConvertToNumpyArray(
                MACCSkeys.GenMACCSKeys(mol), maccs_arr
            )

            vecs.append(np.concatenate((morgan, rdkit2d, maccs_arr)))

        return np.array(vecs, dtype=np.float64)
