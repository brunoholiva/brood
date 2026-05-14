"""Dataclasses for structured brood results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class RankingMetrics:
    """Ranking performance for a set of molecules.

    Attributes
    ----------
    n : int
        Number of molecules in this set.
    average_precision : float
        Average precision (area under precision-recall curve).
    bedroc : float
        Boltzmann-Enhanced Discrimination of ROC (alpha=20.0).
    """

    n: int
    average_precision: float
    bedroc: float


@dataclass
class BinMetrics:
    """Ranking metrics for a specific distance-to-train bin.

    Attributes
    ----------
    bin : str
        Bin label (e.g. "near", "medium", "far").
    n : int
        Number of molecules in this bin.
    average_precision : float
        Average precision within this bin.
    bedroc : float
        BEDROC score within this bin.
    """

    bin: str
    n: int
    average_precision: float
    bedroc: float


@dataclass
class ScreenResult:
    """Full result of a brood screening experiment.

    Attributes
    ----------
    global_ : RankingMetrics
        Ranking metrics across all test molecules.
    by_bin : list of BinMetrics
        Per-bin ranking metrics.
    test_df : pd.DataFrame
        Test set with predictions and distance_to_train.
    pipeline : Pipeline or None
        Trained pipeline. None if test set was empty.
    """

    global_: RankingMetrics
    by_bin: list[BinMetrics]
    test_df: pd.DataFrame
    pipeline: Optional[Pipeline]
