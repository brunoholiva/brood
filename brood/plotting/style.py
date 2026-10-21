"""Shared figure style for the CV result plots.

Minimal white-background style: one hand-picked colour per model, no top/right
spines, left/bottom spines pushed outward, and no tick marks.

This module is the single source of truth for the figures' colours, fonts and
run labels: :func:`run_label` turns a run directory name such as
``tabpfn_chemeleon`` into the display name ``TabPFN + CheMeleon``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from brood.models.registry import MODEL_REGISTRY
from brood.representations import REPRESENTATION_REGISTRY

FONTSIZE = 18

FONTSIZE_SMALL = 14

FONTSIZE_TITLE = 20

#: Canonical order of the models on a leaderboard axis.
MODEL_ORDER = tuple(MODEL_REGISTRY)

#: Canonical order of the representations on the representation axis.
REPRESENTATION_ORDER = tuple(REPRESENTATION_REGISTRY)

#: One colour per model; grey is reserved for the naive baseline.
LEARNER_COLORS = {
    "tabpfn": "#6666FF",
    "lgbm": "#FF9933",
    "rf": "#7BB601",
    "chemprop": "#BB3909",
    "knn": "#666666",
}

LEARNER_LABELS = {
    "tabpfn": "TabPFN",
    "rf": "Random forest",
    "chemprop": "Chemprop",
    "knn": "KNN",
    "lgbm": "LightGBM",
}

#: Human labels for the representation axis.
REPRESENTATION_LABELS = {
    name: rep.label or name for name, rep in REPRESENTATION_REGISTRY.items()
}

#: Registry labels may span lines; a run name is a single line.
_NEWLINE = "\n"


def use_style() -> None:
    """Apply the white-background style and typography defaults."""
    sns.set_style("white")
    rcParams["font.size"] = FONTSIZE
    rcParams["font.family"] = "DejaVu Sans"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["svg.fonttype"] = "none"


def decorate_axis(ax, *, labelsize: int = FONTSIZE) -> None:
    """Hide the top/right spines and offset the visible ones outward."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)


def set_axes(ax, xlim=None, ylim=None, xlabel=None, ylabel=None) -> None:
    """Set limits and axis labels with the shared label padding."""
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=14)


def run_learner(run: str) -> str:
    """Model type of a run name, or ``"other"`` when it is unknown.

    ``'tabpfn_morgan512'`` -> ``'tabpfn'``. Runs with no representation
    variant (``'chemprop'``, ``'knn'``) map to their own model type, so
    they keep their model colour instead of falling back to the
    unknown-run grey.
    """
    learner, _, _ = run.partition("_")
    if learner in LEARNER_COLORS:
        return learner
    return "other"


def _flatten(label: str) -> str:
    r"""Registry label on one line.

    ``'Morgan\n2048'`` -> ``'Morgan 2048'`` and ``'Morgan\n+MACCS'`` ->
    ``'Morgan + MACCS'``: a line that starts with ``+`` is a new term, any
    other line continues the previous one.
    """
    parts = label.split(_NEWLINE)
    flat = parts[0]
    for part in parts[1:]:
        if part.startswith("+"):
            flat += f" + {part.lstrip('+')}"
        else:
            flat += f" {part}"
    return flat


def representation_label(name: str) -> str:
    """Representation label on one line, e.g. ``'Morgan 2048'``."""
    return _flatten(REPRESENTATION_LABELS.get(name, name))


def run_label(run: str) -> str:
    """Display name of a run.

    ``'tabpfn_chemeleon'`` -> ``'TabPFN + CheMeleon'`` and ``'knn'`` ->
    ``'KNN'``. Unknown run names are returned unchanged, so an unlabelled
    run still shows something meaningful.
    """
    learner, _, representation = run.partition("_")
    label = LEARNER_LABELS.get(learner)
    if label is None:
        return run
    if not representation:
        return label
    rep_label = REPRESENTATION_LABELS.get(representation)
    if rep_label is None:
        return f"{label} + {representation.replace('_', ' ')}"
    return f"{label} + {_flatten(rep_label)}"


def run_sort_key(run: str) -> tuple[int, int]:
    """Sort key giving every dataset the same leaderboard order.

    Representations come first in registry order, then models in registry
    order, so a model x representation block stays together and a run
    without a representation (``'knn'``) sorts last. Unknown names sort
    last too.
    """
    learner, _, representation = run.partition("_")
    rep_index = (
        REPRESENTATION_ORDER.index(representation)
        if representation in REPRESENTATION_ORDER
        else len(REPRESENTATION_ORDER)
    )
    model_index = (
        MODEL_ORDER.index(learner)
        if learner in MODEL_ORDER
        else len(MODEL_ORDER)
    )
    return (rep_index, model_index)


def save_fig(fig, path) -> None:
    """Write an SVG, creating the parent directory if needed."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
