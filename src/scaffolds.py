"""Generic Murcko scaffold computation for scaffold-based splitting."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.logger().setLevel(RDLogger.ERROR)


def generic_scaffold(smiles: str) -> str:
    """Return the generic Murcko scaffold SMILES for a molecule.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule.

    Returns
    -------
    str
        Canonical SMILES of the generic (decorated) Murcko scaffold.

    Raises
    ------
    ValueError
        If the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumHeavyAtoms() == 0:
        return Chem.MolToSmiles(mol)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic)


def assign_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``scaffold`` column to a DataFrame copy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``standardized_smiles`` column.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``scaffold`` column.
    """
    out = df.copy()
    out["scaffold"] = [
        generic_scaffold(smi) for smi in df["standardized_smiles"]
    ]
    return out
