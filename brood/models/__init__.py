"""Model backends for brood."""

from __future__ import annotations

from brood.models.base import ModelBackend
from brood.models.registry import get_model, is_sklearn_model, list_models

__all__ = [
    "ModelBackend",
    "get_model",
    "list_models",
    "is_sklearn_model",
]
