"""Small helpers for inspectable model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_dir(data_dir: Path, dataset: str, model: str) -> Path:
    """Return the canonical artifact directory for one model run."""
    return data_dir / "runs" / dataset / model


def prediction_dir(data_dir: Path, dataset: str, model: str) -> Path:
    """Return the canonical prediction directory for one model run."""
    return run_dir(data_dir, dataset, model) / "predictions"


def save_metadata(
    output_dir: Path, name: str, metadata: dict[str, Any]
) -> Path:
    """Write one JSON metadata file next to a model or prediction artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str)
    )
    return path
