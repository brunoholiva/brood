"""Small helpers for inspectable model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
