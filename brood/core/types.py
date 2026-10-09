"""Dataclasses for structured brood results and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RankingMetrics:
    """Ranking and probability performance for a set of molecules.

    Attributes
    ----------
    n : int
        Number of molecules in this set.
    average_precision : float
        Average precision (area under precision-recall curve).
    bedroc : float
        Boltzmann-Enhanced Discrimination of ROC (alpha=20.0).
    brier : float
        Mean squared probability error (lower is better).
    """

    n: int
    average_precision: float
    bedroc: float
    brier: float


@dataclass
class BinMetrics:
    """Metrics for a specific distance-to-train bin.

    Attributes
    ----------
    bin : str
        Bin label (e.g. "near", "medium", "far").
    n : int
        Number of molecules in this bin.
    n_actives : int
        Number of active molecules in this bin.
    average_precision : float
        Average precision within this bin.
    bedroc : float
        BEDROC score within this bin.
    brier : float
        Mean squared probability error within this bin.
    """

    bin: str
    n: int
    n_actives: int
    average_precision: float
    bedroc: float
    brier: float


@dataclass
class BroodResult:
    """Full result of a brood experiment.

    Attributes
    ----------
    global_ : RankingMetrics
        Ranking metrics across all test molecules.
    by_bin : list of BinMetrics
        Per-bin ranking metrics.
    test_df : pd.DataFrame
        Test set with predictions and distance_to_train.
    """

    global_: RankingMetrics
    by_bin: list[BinMetrics]
    test_df: pd.DataFrame
