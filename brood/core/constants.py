"""Shared column-name constants for the brood pipeline.

Centralising the column names keeps CSV-reading and -writing modules
consistent and greppable.
"""

from __future__ import annotations

SMILES_COL = "standardized_smiles"
TARGET_COL = "target"
PREDICTION_COL = "predicted_probability"
