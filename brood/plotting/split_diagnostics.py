"""Split diagnostics: per-fold class balance for the repeated CV folds.

Renders one stacked bar per (repeat, fold): the active fraction on top of the
inactive fraction, both scaled to full bar height, with a dashed line at the
dataset's overall active fraction. A fold whose stack deviates from that line
is not class-balanced, which threatens both model fitting and the paired
statistical comparison downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from brood.plotting.style import (
    FONTSIZE_SMALL,
    FONTSIZE_TITLE,
    decorate_axis,
    save_fig,
    set_axes,
    use_style,
)

logger = logging.getLogger(__name__)

_ACTIVE_COLOR = "#0072B2"
_INACTIVE_COLOR = "#D9D9D9"
_REFERENCE_COLOR = "#D55E00"

use_style()


def plot_cv_class_balance(
    cv_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> Path | None:
    """Stacked active/inactive fraction per CV fold, grouped by repeat.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Frame with ``target`` and ``fold`` columns, plus an optional
        ``repeat`` column.
    dataset : str
        Dataset name, used in the title and filename.
    output_dir : Path
        Root figure directory; ``<dataset>/splits`` is appended.

    Returns
    -------
    Path | None
        Path to the written SVG, or None if there was nothing to plot.
    """
    df = cv_df.copy()
    if "repeat" not in df.columns:
        df["repeat"] = 0
    if df.empty:
        logger.warning(f"{dataset}: no CV folds to plot")
        return None

    groups = list(df.groupby(["repeat", "fold"], sort=True))
    labels = [f"r{repeat}f{fold}" for (repeat, fold), _ in groups]
    totals = np.array([len(frame) for _, frame in groups], dtype=float)
    actives = np.array([frame["target"].sum() for _, frame in groups])
    active_fraction = actives / totals
    overall = float(df["target"].mean())
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(max(7.0, 0.5 * len(groups) + 2.0), 5.2))
    ax.bar(x, 1.0 - active_fraction, color=_INACTIVE_COLOR, width=0.8)
    ax.bar(
        x,
        active_fraction,
        bottom=1.0 - active_fraction,
        color=_ACTIVE_COLOR,
        width=0.8,
    )
    ax.axhline(overall, ls="--", color=_REFERENCE_COLOR, lw=1.4, zorder=4)
    ax.text(
        len(groups) - 0.5,
        overall,
        f" overall {overall:.1%}",
        color=_REFERENCE_COLOR,
        fontsize=FONTSIZE_SMALL,
        va="bottom",
        ha="right",
    )
    for xi, (total, fraction) in enumerate(zip(totals, active_fraction)):
        ax.text(
            xi,
            1.01,
            f"n={int(total)}\n{fraction:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#444444",
        )

    ax.set_ylim(0, 1.18)
    ax.set_xticks(x, labels, rotation=90)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    decorate_axis(ax)
    set_axes(
        ax,
        xlabel="CV fold (repeat × fold)",
        ylabel="Class fraction",
    )
    ax.set_title(dataset, fontsize=FONTSIZE_TITLE)
    ax.legend(
        handles=[
            Line2D([0], [0], color=_ACTIVE_COLOR, lw=8, label="active"),
            Line2D([0], [0], color=_INACTIVE_COLOR, lw=8, label="inactive"),
            Line2D(
                [0],
                [0],
                color=_REFERENCE_COLOR,
                ls="--",
                lw=1.4,
                label="overall active fraction",
            ),
        ],
        loc="upper right",
    )
    path = output_dir / dataset / "splits" / f"{dataset}_cv_class_balance.svg"
    save_fig(fig, path)
    logger.info(f"CV class-balance figure saved to {path}")
    return path
