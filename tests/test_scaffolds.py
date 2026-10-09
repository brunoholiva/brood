"""Tests for generic Murcko scaffold computation."""

from __future__ import annotations

import pandas as pd

from brood.data.scaffolds import assign_scaffolds, generic_scaffold

# Hypervalent metal complexes: RDKit parses them but cannot re-sanitize the
# generic scaffold (MakeScaffoldGeneric -> RemoveHs), so the non-generic
# scaffold must be used instead of raising.
METAL_COMPLEXES = [
    "O=C1O[Al]23(OC1=O)(OC(=O)C(=O)O2)OC(=O)C(=O)O3",
    "C1CN[Co-4]23(N1)(NCCN2)NCCN3",
]


def test_generic_scaffold_falls_back_on_sanitize_failure():
    """Metal complexes return a scaffold string instead of raising."""
    for smiles in METAL_COMPLEXES:
        scaffold = generic_scaffold(smiles)
        assert isinstance(scaffold, str)
        assert scaffold


def test_assign_scaffolds_preserves_rows_on_fallback():
    """Every molecule keeps its row; none is dropped or left unassigned."""
    df = pd.DataFrame(
        {
            "standardized_smiles": [*METAL_COMPLEXES, "c1ccccc1"],
            "target": [0, 1, 0],
        }
    )
    out = assign_scaffolds(df)
    assert len(out) == len(df)
    assert out["scaffold"].notna().all()
