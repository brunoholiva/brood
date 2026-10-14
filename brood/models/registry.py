"""Central model registry for brood.

Backends are imported lazily so that the registry can be inspected in any
environment without pulling in optional dependencies (chemprop, tabpfn,
lightgbm).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brood.models.base import ModelBackend


class _LazyBackend:
    """Placeholder that loads a backend class on first access."""

    def __init__(self, module: str, class_name: str, kwargs: dict | None):
        self.module = module
        self.class_name = class_name
        self.kwargs = kwargs or {}
        self._cls = None

    def _load(self):
        if self._cls is None:
            module = importlib.import_module(self.module)
            self._cls = getattr(module, self.class_name)
        return self._cls

    def build(self, **extra_kwargs) -> "ModelBackend":
        """Instantiate the lazily-loaded backend."""
        cls = self._load()
        kwargs = {**self.kwargs, **extra_kwargs}
        return cls(**kwargs)


#: Single source of truth for model backends.
MODEL_REGISTRY: dict[str, _LazyBackend] = {
    "rf": _LazyBackend(
        "brood.models.sklearn", "SklearnBackend", {"name": "rf"}
    ),
    "lgbm": _LazyBackend(
        "brood.models.sklearn", "SklearnBackend", {"name": "lgbm"}
    ),
    "tabpfn": _LazyBackend("brood.models.tabpfn", "TabPFNBackend", {}),
    "tabpfn_regression": _LazyBackend(
        "brood.models.tabpfn", "TabPFNRegressionBackend", {}
    ),
    "chemprop": _LazyBackend("brood.models.chemprop", "ChempropBackend", {}),
    "knn": _LazyBackend("brood.models.knn", "KNNBackend", {}),
}


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(MODEL_REGISTRY)


def get_model(name: str, **kwargs) -> "ModelBackend":
    """Instantiate a model backend by name.

    Parameters
    ----------
    name : str
        Registered model name.
    **kwargs
        Extra constructor kwargs (e.g. ``device`` or ``version`` for TabPFN).

    Returns
    -------
    ModelBackend
        Configured backend instance.
    """
    entry = MODEL_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown model {name!r}. Available: {list_models()}")
    return entry.build(**kwargs)


def is_sklearn_model(name: str) -> bool:
    """Whether ``name`` is one of the fixed-representation sklearn learners."""
    return name in ("rf", "lgbm")
