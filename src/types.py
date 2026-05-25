"""Dataclasses for structured brood results and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class ColumnConfig:
    """Column name configuration for screening.

    Attributes
    ----------
    smiles_col : str
        Column with SMILES strings.
    target_col : str
        Column with binary labels (0/1).
    score_col : str
        Column name for predicted probabilities in the output.
    """

    smiles_col: str = "standardized_smiles"
    target_col: str = "target"
    score_col: str = "predicted_probability"


@dataclass
class SplitConfig:
    """Taylor-Butina split configuration.

    Attributes
    ----------
    train_size : float
        Fraction of data for training.
    test_size : float
        Fraction of data for testing.
    threshold : float
        Tanimoto distance threshold for clustering.
    approximate : bool
        Use approximate similarity (NNDescent) for clustering.
    n_jobs : int or None
        Number of parallel jobs for the splitter.
    """

    train_size: float = 0.8
    test_size: float = 0.2
    threshold: float = 0.65
    approximate: bool = True
    n_jobs: Optional[int] = None


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Attributes
    ----------
    distance_bins : list of (low, high, name) or None
        Distance bins for per-bin metrics. Defaults to near (0-0.3),
        medium (0.3-0.6), far (0.6+).
    bedrock_alpha : float
        BEDROC early-enrichment parameter.
    """

    distance_bins: Optional[list] = None
    bedrock_alpha: float = 20.0


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
    n_actives : int
        Number of active molecules in this bin.
    average_precision : float
        Average precision within this bin.
    bedroc : float
        BEDROC score within this bin.
    """

    bin: str
    n: int
    n_actives: int
    average_precision: float
    bedroc: float


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
    pipeline : Pipeline or None
        Trained pipeline. None if test set was empty.
    """

    global_: RankingMetrics
    by_bin: list[BinMetrics]
    test_df: pd.DataFrame
    pipeline: Optional[Pipeline]
