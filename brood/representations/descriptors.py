"""RDKit descriptor transformers (non-fingerprint representations).

Four stateless sklearn transformer families built from RDKit:

- ``PhysChemDescriptorTransformer``: compact interpretable physchem
  properties (logP, TPSA, HBD/HBA, rotatable bonds, rings, ...).
- ``TopologicalDescriptorTransformer``: graph/topology invariants
  (Chi connectivity, Kappa shape, Bertz/Balaban/Ipc complexity).
- ``EStateDescriptorTransformer``: per-atom intrinsic EState aggregates
  plus EState VSA bins.
- ``Autocorr2DTransformer``: Moreau–Broto 2D autocorrelation (192 dims).

All values are NaN-safe (NaN replaced with 0.0) so descriptors feed
directly into TabPFN/RF without preprocessing.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.EState import EStateIndices
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

_DESCRIPTORS = dict(Descriptors.descList)
_DESCRIPTORS.update(
    {
        "FormalCharge": lambda mol: float(Chem.GetFormalCharge(mol)),
        "NumHeavyAtoms": lambda mol: float(mol.GetNumHeavyAtoms()),
    }
)

PHYSCHEM_NAMES = (
    "MolWt",
    "ExactMolWt",
    "HeavyAtomMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumHeteroatoms",
    "NumHeavyAtoms",
    "NumValenceElectrons",
    "RingCount",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "FormalCharge",
    "NumRadicalElectrons",
    "HallKierAlpha",
)

TOPOLOGICAL_NAMES = (
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "HallKierAlpha",
    "BertzCT",
    "BalabanJ",
    "Ipc",
    "NOCount",
    "NHOHCount",
)


class _DescriptorTransformer(BaseEstimator, TransformerMixin):
    """Base: stateless, NaN-safe matrix of named RDKit descriptors."""

    _names: tuple[str, ...] = ()
    _dim = 0
    _desc = "Descriptors"

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def _mol_features(self, mol) -> np.ndarray:
        return np.array([_DESCRIPTORS[name](mol) for name in self._names])

    def _n_features(self) -> int:
        return self._dim or len(self._names)

    def transform(self, X):
        """Convert SMILES to (n_samples, n_features) float64 matrix."""
        n = len(X)
        out = np.zeros((n, self._n_features()), dtype=np.float64)
        for i, s in enumerate(
            tqdm(X, desc=self._desc, unit="mol", leave=True)
        ):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES at position {i}")
            out[i] = np.nan_to_num(self._mol_features(mol), nan=0.0)
        return out


class PhysChemDescriptorTransformer(_DescriptorTransformer):
    """Compact interpretable physicochemical property vector (20-dim)."""

    _names = PHYSCHEM_NAMES
    _desc = "PhysChem"


class TopologicalDescriptorTransformer(_DescriptorTransformer):
    """RDKit graph/topology invariants (21-dim)."""

    _names = TOPOLOGICAL_NAMES
    _desc = "Topological"


class EStateDescriptorTransformer(_DescriptorTransformer):
    """Electrotopological state: atom aggregates + EState-VSA bins (15-dim)."""

    _names = tuple(f"VSA_EState{i}" for i in range(1, 11))
    _dim = 15
    _desc = "EState"

    def _mol_features(self, mol) -> np.ndarray:
        states = EStateIndices(mol)
        agg = np.array(
            [
                states.sum(),
                states.mean(),
                states.std(),
                states.min(),
                states.max(),
            ],
            dtype=np.float64,
        )
        vsa = np.array([_DESCRIPTORS[name](mol) for name in self._names])
        return np.concatenate((agg, vsa))


class Autocorr2DTransformer(_DescriptorTransformer):
    """Moreau-Broto 2D autocorrelation descriptors (192-dim)."""

    _dim = 192
    _desc = "Autocorr2D"

    def _mol_features(self, mol) -> np.ndarray:
        return np.asarray(rdmd.CalcAUTOCORR2D(mol), dtype=np.float64)
