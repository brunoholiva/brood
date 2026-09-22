"""Dataset validation and scaffold report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from .scaffolds import assign_scaffolds


def normalize_dataset(
    df: pd.DataFrame,
    smiles_col: str = "standardized_smiles",
    target_col: str = "target",
) -> tuple[pd.DataFrame, dict]:
    """Canonicalize SMILES and return data-quality statistics.

    Invalid molecules are excluded from the returned frame and counted in the
    report. Duplicate canonical SMILES with conflicting labels are rejected.
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

    conflicting = (
        out.groupby("standardized_smiles")["target"].nunique().gt(1).sum()
    )
    if conflicting:
        raise ValueError(
            f"Found {conflicting} canonical SMILES with conflicting labels"
        )
    duplicate_rows = int(out.duplicated("standardized_smiles").sum())
    out = out.drop_duplicates("standardized_smiles").reset_index(drop=True)
    report = {
        "input_rows": int(len(df)),
        "missing_smiles": missing_smiles,
        "invalid_smiles": invalid,
        "duplicate_rows": duplicate_rows,
        "molecules": int(len(out)),
        "actives": int(out["target"].sum()),
        "inactive": int((out["target"] == 0).sum()),
    }
    return out, report


def build_dataset_report(df: pd.DataFrame) -> dict:
    """Return label and generic Murcko scaffold statistics."""
    annotated = assign_scaffolds(df)
    scaffold_sizes = annotated.groupby("scaffold").agg(
        molecules=("target", "size"), actives=("target", "sum")
    )
    return {
        "molecules": int(len(annotated)),
        "actives": int(annotated["target"].sum()),
        "inactive": int((annotated["target"] == 0).sum()),
        "active_fraction": float(annotated["target"].mean()),
        "n_scaffolds": int(len(scaffold_sizes)),
        "scaffold_size_min": int(scaffold_sizes["molecules"].min()),
        "scaffold_size_max": int(scaffold_sizes["molecules"].max()),
        "scaffold_size_mean": float(scaffold_sizes["molecules"].mean()),
        "scaffolds": [
            {
                "scaffold": str(scaffold),
                "molecules": int(row.molecules),
                "actives": int(row.actives),
            }
            for scaffold, row in scaffold_sizes.sort_values(
                "molecules", ascending=False
            ).iterrows()
        ],
    }


def save_dataset_report(
    df: pd.DataFrame, output_dir: Path, max_scaffolds: int = 100
) -> dict:
    """Write JSON statistics and a generic-scaffold image grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_dataset_report(df)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    scaffolds = report["scaffolds"][:max_scaffolds]
    molecules = [Chem.MolFromSmiles(item["scaffold"]) for item in scaffolds]
    if molecules:
        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=5,
            subImgSize=(220, 180),
            legends=[
                f"n={item['molecules']} active={item['actives']}"
                for item in scaffolds
            ],
        )
        image.save(output_dir / "generic_scaffolds.png")
    return report
