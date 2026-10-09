"""Shared RDKit utilities for molecular representation transformers."""

from __future__ import annotations

from rdkit import Chem


def require_mols(smiles: list[str]) -> list[Chem.Mol]:
    """Parse SMILES and raise on the first invalid entry.

    Parameters
    ----------
    smiles : list[str]
        Input SMILES strings.

    Returns
    -------
    list[Chem.Mol]
        Parsed RDKit molecules in the same order.

    Raises
    ------
    ValueError
        If any SMILES cannot be parsed, naming its position.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    for i, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"Invalid SMILES at position {i}")
    return mols
