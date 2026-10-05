"""Molecular representation registry for brood."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Representation:
    """One registered molecular representation.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"morgan2048"``.
    module : str
        Fully-qualified module path inside ``brood.representations``.
    class_name : str
        Transformer class name inside that module.
    default_kwargs : dict
        Default constructor kwargs (e.g. fingerprint radius/size).
    dims : int or None
        Output dimension, when known and fixed.
    env : str or None
        Conda env required for this representation, if not the base one.
    """

    name: str
    module: str
    class_name: str
    default_kwargs: dict[str, Any]
    dims: int | None = None
    env: str | None = None
    label: str | None = None


#: Single source of truth for all molecular representations.
REPRESENTATION_REGISTRY: dict[str, Representation] = {
    "morgan512": Representation(
        name="morgan512",
        module="brood.representations.morgan",
        class_name="MorganFingerprintTransformer",
        default_kwargs={"radius": 2, "fp_size": 512},
        dims=512,
        label="Morgan\n512",
    ),
    "morgan2048": Representation(
        name="morgan2048",
        module="brood.representations.morgan",
        class_name="MorganFingerprintTransformer",
        default_kwargs={"radius": 2, "fp_size": 2048},
        dims=2048,
        label="Morgan\n2048",
    ),
    "maccs": Representation(
        name="maccs",
        module="brood.representations.maccs",
        class_name="MACCSFingerprintTransformer",
        default_kwargs={},
        dims=167,
        label="MACCS",
    ),
    "rdkit2d": Representation(
        name="rdkit2d",
        module="brood.representations.rdkit2d",
        class_name="RDKit2DDescriptorTransformer",
        default_kwargs={},
        dims=200,
        label="RDKit2D",
    ),
    "physchem": Representation(
        name="physchem",
        module="brood.representations.descriptors",
        class_name="PhysChemDescriptorTransformer",
        default_kwargs={},
    ),
    "topological": Representation(
        name="topological",
        module="brood.representations.descriptors",
        class_name="TopologicalDescriptorTransformer",
        default_kwargs={},
    ),
    "estate": Representation(
        name="estate",
        module="brood.representations.descriptors",
        class_name="EStateDescriptorTransformer",
        default_kwargs={},
    ),
    "autocorr2d": Representation(
        name="autocorr2d",
        module="brood.representations.descriptors",
        class_name="Autocorr2DTransformer",
        default_kwargs={},
    ),
    "morgan_maccs_rdkit2d": Representation(
        name="morgan_maccs_rdkit2d",
        module="brood.representations.combined",
        class_name="MoleculeFeaturizer",
        default_kwargs={"fp_size": 512},
        dims=879,
        label="Morgan\n+MACCS\n+RDKit2D",
    ),
    "chemeleon": Representation(
        name="chemeleon",
        module="brood.representations.chemeleon",
        class_name="CheMeleonFingerprint",
        default_kwargs={},
        dims=2048,
        env="chemeleon",
        label="CheMeleon",
    ),
}


def _load(module: str, attr: str):
    return getattr(importlib.import_module(module), attr)


def list_representations() -> list[str]:
    """Return all registered representation names."""
    return list(REPRESENTATION_REGISTRY)


def get_representation(name: str, **kwargs: Any):
    """Instantiate a representation transformer by name.

    Parameters
    ----------
    name : str
        Registered representation name.
    **kwargs
        Override constructor kwargs.

    Returns
    -------
    TransformerMixin
        Unfitted transformer instance.
    """
    rep = REPRESENTATION_REGISTRY.get(name)
    if rep is None:
        raise ValueError(
            f"Unknown representation {name!r}. "
            f"Available: {list_representations()}"
        )
    cls = _load(rep.module, rep.class_name)
    return cls(**{**rep.default_kwargs, **kwargs})


def representation_env(name: str) -> str | None:
    """Return the special conda env required for a representation, if any."""
    rep = REPRESENTATION_REGISTRY.get(name)
    return rep.env if rep else None
