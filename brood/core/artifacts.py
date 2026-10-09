"""Canonical artifact directory helpers."""

from __future__ import annotations

from pathlib import Path


def run_dir(data_dir: Path, dataset: str, model: str) -> Path:
    """Return the canonical artifact directory for one model run."""
    return data_dir / "runs" / dataset / model


def prediction_dir(data_dir: Path, dataset: str, model: str) -> Path:
    """Return the canonical prediction directory for one model run."""
    return run_dir(data_dir, dataset, model) / "predictions"
