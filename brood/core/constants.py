"""Shared constants for the brood pipeline.

Centralising column names and default values keeps the various CSV-reading
and -writing modules consistent and greppable.
"""

from __future__ import annotations

#: Random seed used throughout the pipeline unless overridden.
DEFAULT_RANDOM_STATE = 42

#: Default number of scaffold-CV folds.
DEFAULT_N_FOLDS = 5

#: Default number of Optuna tuning trials.
DEFAULT_N_TRIALS = 10

#: Column names used in split/prediction/evaluation dataframes.
SMILES_COL = "standardized_smiles"
TARGET_COL = "target"
SCAFFOLD_COL = "scaffold"
FOLD_COL = "fold"
DISTANCE_COL = "distance_to_train"
PREDICTION_COL = "predicted_probability"

#: Subdirectory names under the data root.
SPLITS_DIR = "splits"
FEATURES_DIR = "features"
RUNS_DIR = "runs"
EVALUATIONS_DIR = "evaluations"

#: Default BEDROC early-enrichment parameter.
BEDROC_ALPHA = 20.0

#: Distance bins for the default distance-aware evaluation.
DEFAULT_BINS = [
    (0.0, 0.3, "near"),
    (0.3, 0.6, "medium"),
    (0.6, 1.01, "far"),
]
