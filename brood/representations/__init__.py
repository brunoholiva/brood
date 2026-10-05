"""Molecular representation transformers and registry for brood."""

from __future__ import annotations

from brood.representations.registry import (
    REPRESENTATION_REGISTRY,
    get_representation,
    list_representations,
    representation_env,
)

#: Backward-compatible alias.
FEATURIZER_REGISTRY = {
    name: (rep.module, rep.class_name, rep.default_kwargs)
    for name, rep in REPRESENTATION_REGISTRY.items()
}

#: Backward-compatible alias.
get_featurizer = get_representation

__all__ = [
    "FEATURIZER_REGISTRY",
    "REPRESENTATION_REGISTRY",
    "get_featurizer",
    "get_representation",
    "list_representations",
    "representation_env",
]

_PUBLIC_ATTRS = {
    rep.class_name: (rep.module, rep.class_name)
    for rep in REPRESENTATION_REGISTRY.values()
}


def __getattr__(name: str):
    """Lazily expose transformer classes used across training scripts."""
    import importlib

    entry = _PUBLIC_ATTRS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def __dir__():
    return sorted(
        set(globals())
        | set(_PUBLIC_ATTRS)
        | {"get_representation", "get_featurizer", "list_representations"}
    )
