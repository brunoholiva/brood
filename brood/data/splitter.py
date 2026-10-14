"""Scaffold-based OOD splitting and CV fold assignment.

Molecules are grouped by their generic Murcko scaffold. The held-out
test set is one scaffold-based, label-stratified slice (default 10%),
and each repeated-CV split keeps each scaffold in one fold via
:class:`StratifiedGroupKFold`.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from .scaffolds import assign_scaffolds

logger = logging.getLogger(__name__)

K_NEIGHBORS = 5
BATCH_SIZE = 5000
DEFAULT_RANDOM_STATE = 42
LABEL_SHIFT_TOLERANCE = 0.1


def compute_morgan_fingerprints(
    smiles_list: list[str], radius: int = 2, fp_size: int = 2048
) -> list:
    """Return RDKit Morgan fingerprints for distance calculations."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    if any(mol is None for mol in mols):
        raise ValueError("Invalid SMILES encountered during fingerprinting.")
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps_bitvect = [generator.GetFingerprint(mol) for mol in mols]
    return fps_bitvect


def _score_batch(batch_fps, train_fps):
    """Mean 5-NN Tanimoto distance for one batch of test molecules."""
    k = min(K_NEIGHBORS, len(train_fps))
    mean_dists = []
    for test_fp in batch_fps:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(test_fp, train_fps),
            dtype=float,
        )
        distances = 1.0 - similarities
        nearest = np.partition(distances, k - 1)[:k]
        mean_dists.append(float(nearest.mean()))
    return mean_dists


def _score_test_set(test_fps, train_fps):
    """Score test molecules, returning distance_to_train per position.

    Processes in batches to control memory usage. Positions align with
    ``test_fps`` order.
    """
    all_dists = []
    for start in tqdm(
        range(0, len(test_fps), BATCH_SIZE),
        desc="Scoring test molecules",
        unit="batch",
    ):
        end = min(start + BATCH_SIZE, len(test_fps))
        all_dists.extend(_score_batch(test_fps[start:end], train_fps))
    return all_dists


def compute_test_distances(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    fps: list,
    smiles: list[str],
) -> pd.DataFrame:
    """Compute distance_to_train for test molecules and merge into test_df.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-set frame with a ``standardized_smiles`` column.
    train_df : pd.DataFrame
        Training-set frame with a ``standardized_smiles`` column.
    fps : list
        Morgan fingerprints aligned with ``smiles``.
    smiles : list[str]
        Ordered SMILES corresponding to ``fps``.

    Returns
    -------
    pd.DataFrame
        ``test_df`` with an added ``distance_to_train`` column.
    """
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}
    train_fps = [
        fps[smiles_to_idx[s]] for s in train_df["standardized_smiles"]
    ]
    test_fps = [fps[smiles_to_idx[s]] for s in test_df["standardized_smiles"]]

    return test_df.assign(
        distance_to_train=_score_test_set(test_fps, train_fps)
    )


def _format_output(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select final columns for output."""
    test_df = test_df[
        ["standardized_smiles", "target", "scaffold", "distance_to_train"]
    ]
    train_df = train_df[["standardized_smiles", "target", "scaffold"]]
    return train_df, test_df


def _require_scaffold(df: pd.DataFrame) -> pd.DataFrame:
    """Validate scaffold grouping, assigning it if necessary."""
    if "scaffold" not in df.columns:
        df = assign_scaffolds(df)
    n_scaffolds = df["scaffold"].nunique()
    if n_scaffolds < 2:
        raise ValueError(
            f"Need at least 2 distinct scaffolds for a scaffold split, "
            f"found {n_scaffolds}."
        )
    return df


def validate_partitions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_shift_tolerance: float = LABEL_SHIFT_TOLERANCE,
) -> dict:
    """Assert scaffold isolation and report label-distribution shift.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training frame with ``scaffold`` and ``target`` columns.
    test_df : pd.DataFrame
        Test frame with ``scaffold`` and ``target`` columns.
    label_shift_tolerance : float
        Maximum tolerable absolute active-fraction difference.

    Returns
    -------
    dict
        Report with keys ``scaffold_overlap``, ``train_active_fraction``,
        ``test_active_fraction``, ``absolute_label_shift``, and
        ``label_shift_within_tolerance``.

    Raises
    ------
    ValueError
        If any scaffold appears in both partitions.
    """
    overlap = set(train_df["scaffold"]) & set(test_df["scaffold"])
    if overlap:
        raise ValueError(
            f"Scaffold leakage detected: {len(overlap)} shared scaffolds"
        )
    train_fraction = float(train_df["target"].mean())
    test_fraction = float(test_df["target"].mean())
    shift = abs(train_fraction - test_fraction)
    return {
        "scaffold_overlap": len(overlap),
        "train_active_fraction": train_fraction,
        "test_active_fraction": test_fraction,
        "absolute_label_shift": shift,
        "label_shift_within_tolerance": shift <= label_shift_tolerance,
    }


def validate_cv_folds(
    cv_df: pd.DataFrame,
    label_shift_tolerance: float = LABEL_SHIFT_TOLERANCE,
) -> dict:
    """Assert scaffold isolation and report label shift across CV folds.

    Isolation is checked *within each repeat*: a scaffold may land in a
    different fold in another repeat (that is repeated CV), but never twice
    inside one repeat.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Frame with ``fold``, ``scaffold``, and ``target`` columns, plus an
        optional ``repeat`` column.
    label_shift_tolerance : float
        Maximum tolerable active-fraction spread across folds.

    Returns
    -------
    dict
        Report with keys ``folds``, ``repeats``, ``scaffold_overlap``,
        ``active_fraction_by_fold``, ``active_fraction_spread``, and
        ``active_fraction_within_tolerance``.

    Raises
    ------
    ValueError
        If any scaffold appears in multiple folds of the same repeat.
    """
    has_repeat = "repeat" in cv_df.columns
    folds = sorted(cv_df["fold"].unique())
    repeats = sorted(cv_df["repeat"].unique()) if has_repeat else [0]
    key = ["scaffold", "repeat"] if has_repeat else ["scaffold"]
    scaffold_folds = cv_df.groupby(key)["fold"].nunique()
    shared_scaffolds = int((scaffold_folds > 1).sum())
    if shared_scaffolds:
        raise ValueError(
            f"CV scaffold leakage detected: {shared_scaffolds} scaffold-"
            "repeat pairs appear in multiple folds."
        )

    fractions = cv_df.groupby("fold")["target"].mean()
    spread = float(fractions.max() - fractions.min())
    return {
        "folds": len(folds),
        "repeats": len(repeats),
        "scaffold_overlap": shared_scaffolds,
        "active_fraction_by_fold": {
            int(fold): float(fractions[fold]) for fold in folds
        },
        "active_fraction_spread": spread,
        "active_fraction_within_tolerance": spread <= label_shift_tolerance,
    }


def _shuffle_split_by_scaffold(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random scaffold-disjoint split of a DataFrame.

    Randomly holds out ``test_size`` of the scaffolds (group-disjoint).
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    groups = df["scaffold"].to_numpy()
    scaffold_ids = np.unique(groups)
    n_groups = len(scaffold_ids)
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 distinct scaffolds, found {n_groups}."
        )

    n_test = max(1, min(n_groups - 1, int(round(test_size * n_groups))))
    rng = np.random.default_rng(random_state)
    test_scaffolds = rng.choice(scaffold_ids, size=n_test, replace=False)
    test_mask = np.isin(groups, test_scaffolds)
    return df.iloc[~test_mask].copy(), df.iloc[test_mask].copy()


def split_train_test(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a scaffold-annotated DataFrame into train/test by scaffold."""
    df = _require_scaffold(df)
    return _shuffle_split_by_scaffold(
        df, test_size=1.0 - train_size, random_state=random_state
    )


def scaffold_split(
    df: pd.DataFrame,
    test_frac: float = 0.1,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOD train/test split by generic Murcko scaffold.

    1. Groups molecules by generic Murcko scaffold.
    2. Carves a label-stratified, scaffold-disjoint held-out test
       (``test_frac``) with :class:`StratifiedGroupKFold`.
    3. Computes ``distance_to_train`` for test (mean 5-NN Tanimoto dist).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``standardized_smiles`` and ``target`` columns.
    test_frac : float
        Target fraction of molecules held out as test. Scaffolds remain
        indivisible, so the achieved fraction may differ slightly.
    random_state : int
        Seed for deterministic fold assignment.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with a ``scaffold`` column in both, plus
        ``distance_to_train`` in the test set.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    for col in ("standardized_smiles", "target"):
        if col not in df.columns:
            raise KeyError(f"'{col}' column is required.")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}.")

    df = _require_scaffold(df)

    logger.info("Generating Morgan fingerprints for distance scoring...")
    smiles = df["standardized_smiles"].tolist()
    fps = compute_morgan_fingerprints(smiles)

    groups = df["scaffold"].to_numpy()
    targets = df["target"].to_numpy()
    n_groups = len(set(groups))
    n_splits = max(2, min(10, int(round(1.0 / test_frac))))
    min_class_count = min(np.bincount(targets))
    n_splits = min(n_splits, n_groups, min_class_count)
    if n_splits < 2:
        raise ValueError(
            "Cannot scaffold-split: too few scaffolds or too few "
            "members per class."
        )

    # StratifiedGroupKFold gives feasible scaffold-disjoint candidates, but
    # its first fold is not guaranteed to contain 10% of the molecules.
    # Choose the candidate closest to the requested molecule fraction.
    candidates = []
    for seed in range(random_state, random_state + 32):
        kfold = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        for train_candidate, test_candidate in kfold.split(
            df, targets, groups
        ):
            test_fraction = len(test_candidate) / len(df)
            active_fraction = targets[test_candidate].mean()
            score = abs(test_fraction - test_frac) + 0.25 * abs(
                active_fraction - targets.mean()
            )
            candidates.append((score, train_candidate, test_candidate))
    _, train_idx, test_idx = min(candidates, key=lambda item: item[0])

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    validate_partitions(train_df, test_df)

    logger.info(
        f"Split: train={len(train_df)} ({train_df['scaffold'].nunique()} "
        f"scaffolds), test={len(test_df)} "
        f"({test_df['scaffold'].nunique()} scaffolds)"
    )

    logger.info("Computing distance_to_train for test molecules...")
    test_df = compute_test_distances(test_df, train_df, fps, smiles)
    train_df, test_df = _format_output(train_df, test_df)

    logger.info(f"Scoring done: {len(test_df)} test molecules")
    return train_df, test_df


def _balanced_folds(
    groups: np.ndarray,
    targets: np.ndarray,
    n_splits: int,
    base_seed: int,
    width: int = 32,
) -> np.ndarray:
    """Return the most label-balanced scaffold-based fold assignment.

    Searches ``width`` shuffle seeds starting at ``base_seed`` and keeps the
    assignment whose per-fold active fractions are closest together.
    """
    candidates = []
    for seed in range(base_seed, base_seed + width):
        kfold = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        candidate = np.zeros(len(groups), dtype=int)
        for fold_id, (_, val_idx) in enumerate(
            kfold.split(np.arange(len(groups)), targets, groups)
        ):
            candidate[val_idx] = fold_id
        fractions = [
            targets[candidate == fold_id].mean()
            for fold_id in range(n_splits)
        ]
        candidates.append((max(fractions) - min(fractions), candidate))
    return min(candidates, key=lambda item: item[0])[1]


def assign_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Assign repeated scaffold-based CV fold labels to a DataFrame.

    Produces ``n_repeats`` independent splits, each an ``n_splits``-fold
    scaffold-based, label-balanced split. The result is a long frame: every
    molecule appears once per repeat with its ``repeat`` and ``fold``. One
    shared assignment is used for every model, so the 5x5 = 25 CV splits
    are identical across models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``scaffold`` and ``target`` columns.
    n_splits : int
        Number of CV folds per repeat.
    n_repeats : int
        Number of repeats.
    random_state : int
        Seed for deterministic fold assignment.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` repeated ``n_repeats`` times with ``repeat``, ``fold``,
        and ``molecule_index`` columns. ``molecule_index`` is the positional
        row in the training split, used to index cached feature matrices.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}.")
    df = _require_scaffold(df)
    n_groups = df["scaffold"].nunique()
    min_class_count = min(np.bincount(df["target"].to_numpy()))
    if n_splits > n_groups or n_splits > min_class_count:
        n_splits = min(n_splits, n_groups, min_class_count)
        logger.warning(f"Reducing CV folds to {n_splits} for this dataset.")

    groups = df["scaffold"].to_numpy()
    targets = df["target"].to_numpy()
    parts = []
    for repeat in range(n_repeats):
        part = df.copy()
        part["repeat"] = repeat
        part["molecule_index"] = np.arange(len(df))
        part["fold"] = _balanced_folds(
            groups, targets, n_splits, random_state + repeat * 100
        )
        parts.append(part)

    out = pd.concat(parts, ignore_index=False)
    validate_cv_folds(out)
    return out
