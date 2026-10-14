"""Shared figure style for the CV result plots.

Minimal white-background style: colorblind-safe palette, no top/right spines,
left/bottom spines pushed outward, and no tick marks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

_PALETTE = sns.color_palette("colorblind")

#: One colour per model; grey is reserved for the naive baseline.
LEARNER_COLORS = {
    "tabpfn": _PALETTE[0],
    "rf": _PALETTE[1],
    "chemprop": _PALETTE[2],
    "knn": _PALETTE[7],
    "lgbm": _PALETTE[4],
}

LEARNER_LABELS = {
    "tabpfn": "TabPFN",
    "rf": "Random forest",
    "chemprop": "Chemprop",
    "knn": "KNN",
    "lgbm": "LightGBM",
}


def use_style() -> None:
    """Apply the white-background style and typography defaults."""
    sns.set_style("white")
    rcParams["legend.loc"] = "best"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["svg.fonttype"] = "none"


def decorate_axis(ax, *, labelsize: str = "large") -> None:
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


def save_fig(fig, path) -> None:
    """Write an SVG, creating the parent directory if needed."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
