"""Dataset normalization."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem


def normalize_dataset(
    df: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    target_col: str = "target",
) -> tuple[pd.DataFrame, dict]:
    """Canonicalize SMILES and return data-quality statistics.

    Invalid molecules are excluded from the returned frame and counted in the
    report. Canonical SMILES with conflicting labels are removed entirely.
    """
    for column in (smiles_col, target_col):
        if column not in df.columns:
            raise KeyError(f"'{column}' column is required")

    out = df[[smiles_col, target_col]].copy()
    out = out.rename(
        columns={smiles_col: "standardized_smiles", target_col: "target"}
    )
    missing_smiles = int(out["standardized_smiles"].isna().sum())
    out = out.dropna(subset=["standardized_smiles"])

    canonical = []
    invalid = 0
    for smiles in out["standardized_smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            invalid += 1
            canonical.append(None)
        else:
            canonical.append(Chem.MolToSmiles(mol))
    out["standardized_smiles"] = canonical
    out = out.dropna(subset=["standardized_smiles"])

    out["target"] = pd.to_numeric(out["target"], errors="raise").astype(int)
    if not set(out["target"].unique()).issubset({0, 1}):
        raise ValueError("target must contain only binary values 0 and 1")

    label_counts = out.groupby("standardized_smiles")["target"].nunique()
    conflicting_smiles = label_counts[label_counts.gt(1)].index
    conflicting = int(len(conflicting_smiles))
    if conflicting:
        out = out[~out["standardized_smiles"].isin(conflicting_smiles)]
    duplicate_rows = int(out.duplicated("standardized_smiles").sum())
    out = out.drop_duplicates("standardized_smiles").reset_index(drop=True)
    report = {
        "input_rows": int(len(df)),
        "missing_smiles": missing_smiles,
        "invalid_smiles": invalid,
        "conflicting_labels": conflicting,
        "duplicate_rows": duplicate_rows,
        "molecules": int(len(out)),
        "actives": int(out["target"].sum()),
        "inactive": int((out["target"] == 0).sum()),
    }
    return out, report
