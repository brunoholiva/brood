"""Taylor-Butina distance-based OOD splitter."""

import warnings

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from .fingerprints import MorganFingerprintTransformer

warnings.filterwarnings(
    "ignore", message="Data was converted to boolean for metric jaccard"
)

K_NEIGHBORS = 5
BATCH_SIZE = 5000


def _smiles_to_mols(smiles_list: list[str]) -> list[Chem.Mol]:
    """Convert SMILES to RDKit Mol objects, dropping invalid ones."""
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
    return mols


def _validate_mols(mols: list, expected_count: int) -> None:
    """Raise ValueError if any molecules failed to parse."""
    if len(mols) != expected_count:
        raise ValueError(
            f"Invalid SMILES: {expected_count - len(mols)} molecules "
            "could not be parsed."
        )


def _compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    fp_size: int = 2048,
) -> tuple[list, np.ndarray]:
    """Compute Morgan fingerprints in both RDKit and numpy formats.

    Computes fingerprints once and returns both:
    - RDKit ExplicitBitVect objects (for Butina clustering)
    - numpy int64 array (for distance calculations)

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.
    radius : int
        Morgan fingerprint radius. Default: 2.
    fp_size : int
        Number of fingerprint bits. Default: 2048.

    Returns
    -------
    tuple[list, np.ndarray]
        (fps_bitvect, fps_numpy) where fps_bitvect is a list of RDKit
        ExplicitBitVect objects and fps_numpy is a numpy array of shape
        (n_molecules, fp_size).

    Raises
    ------
    ValueError
        If any SMILES fails to parse.
    """
    mols = _smiles_to_mols(smiles_list)
    _validate_mols(mols, len(smiles_list))

    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps_bitvect = [generator.GetFingerprint(mol) for mol in mols]
    fps_numpy = np.array(fps_bitvect)

    return fps_bitvect, fps_numpy


def butina_cluster(
    smiles_list: list[str],
    threshold: float = 0.65,
    radius: int = 2,
    fp_size: int = 2048,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Cluster molecules using the Taylor-Butina (sphere exclusion) algorithm.

    Returns cluster IDs (integers) for each molecule. Clusters are ordered
    by size descending (largest cluster = 0, next = 1, etc.).

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings to cluster.
    threshold : float, optional
        Tanimoto distance threshold for clustering. Molecules within this
        distance are assigned to the same cluster. Default: 0.65.
    radius : int, optional
        Radius for Morgan (ECFP) fingerprints. Default: 2.
    fp_size : int, optional
        Number of bits in fingerprint. Default: 2048.
    n_jobs : int or None, optional
        Currently unused (RDKit clustering is single-threaded). Included
        for API consistency.

    Returns
    -------
    np.ndarray
        Array of integer cluster IDs with shape (n_molecules,).
    """
    fps_bitvect, _ = _compute_morgan_fingerprints(
        smiles_list, radius=radius, fp_size=fp_size
    )
    return _cluster_from_fps(fps_bitvect, threshold)


def _cluster_from_fps(
    fps_bitvect: list,
    threshold: float,
) -> np.ndarray:
    """Perform Butina clustering on pre-computed RDKit fingerprint bit vectors.

    Internal helper used by :func:`butina_cluster` and
    :func:`taylor_butina_split`.
    """
    n = len(fps_bitvect)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(
            fps_bitvect[i], fps_bitvect[:i]
        )
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, threshold, isDistData=True)

    cluster_ids = np.zeros(n, dtype=int)

    sorted_clusters = sorted(clusters, key=lambda c: -len(c))

    for cluster_id, cluster_members in enumerate(sorted_clusters):
        for mol_idx in cluster_members:
            cluster_ids[mol_idx] = cluster_id

    return cluster_ids


def split_by_clusters(
    df: pd.DataFrame,
    train_size: float = 0.8,
    valid_size: float | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Split a DataFrame by cluster ID for OOD train/(valid)/test partitions.

    For out-of-distribution evaluation, the smallest clusters (structurally
    most novel scaffolds) are assigned to validation/test, while the largest
    clusters go to training. This ensures structural dissimilarity between
    partitions.

    The input DataFrame must have a ``cluster_id`` column (typically from
    calling :func:`butina_cluster` first).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``cluster_id`` column.
    train_size : float, optional
        Fraction for training partition (largest clusters). Default: 0.8.
    valid_size : float or None, optional
        If provided, creates a 3-way split: train (largest clusters),
        valid (medium), test (smallest). Fraction for validation.

    Returns
    -------
    tuple
        2-tuple ``(train_df, test_df)`` if ``valid_size`` is None,
        or 3-tuple ``(train_df, valid_df, test_df)`` otherwise.
    """
    cluster_sizes = df.groupby("cluster_id").size().sort_values(ascending=True)
    n_total = len(df)

    if valid_size is not None:
        test_size = 1.0 - train_size - valid_size
        n_test = int(test_size * n_total)
        n_valid = int(valid_size * n_total)

        test_clusters = []
        cum_test = 0
        for cid, size in cluster_sizes.items():
            if cum_test + size <= n_test:
                test_clusters.append(cid)
                cum_test += size
            else:
                break

        remaining = cluster_sizes.drop(test_clusters)

        valid_clusters = []
        cum_valid = 0
        for cid, size in remaining.items():
            if cum_valid + size <= n_valid:
                valid_clusters.append(cid)
                cum_valid += size
            else:
                break

        train_clusters = list(remaining.drop(valid_clusters).index)

        train_df = df[df["cluster_id"].isin(train_clusters)].copy()
        valid_df = df[df["cluster_id"].isin(valid_clusters)].copy()
        test_df = df[df["cluster_id"].isin(test_clusters)].copy()

        logger.info(
            f"Split: train={len(train_df)} ({len(train_clusters)} clusters), "
            f"valid={len(valid_df)} ({len(valid_clusters)} clusters), "
            f"test={len(test_df)} ({len(test_clusters)} clusters)"
        )

        return train_df, valid_df, test_df

    else:
        n_test = int((1.0 - train_size) * n_total)

        test_clusters = []
        cum_test = 0
        for cid, size in cluster_sizes.items():
            if cum_test + size <= n_test:
                test_clusters.append(cid)
                cum_test += size
            else:
                break

        train_clusters = list(cluster_sizes.drop(test_clusters).index)

        train_df = df[df["cluster_id"].isin(train_clusters)].copy()
        test_df = df[df["cluster_id"].isin(test_clusters)].copy()

        logger.info(
            f"Split: train={len(train_df)} ({len(train_clusters)} clusters), "
            f"test={len(test_df)} ({len(test_clusters)} clusters)"
        )

        return train_df, test_df


def _generate_fingerprints(smiles_list):
    return MorganFingerprintTransformer().transform(smiles_list)


def _score_batch(batch_fps, batch_smiles, batch_targets, train_fps):
    """Score one batch of test molecules and compute distance_to_train."""
    dists = pairwise_distances(
        batch_fps, train_fps, metric="jaccard", n_jobs=-1
    )

    k = min(K_NEIGHBORS, dists.shape[1])
    mean_dists = []
    for row in dists:
        nearest = np.partition(row, k - 1)[:k] if k < len(row) else row
        mean_dists.append(nearest.mean())

    return batch_smiles, batch_targets, mean_dists


def _score_test_set(test_smiles, test_targets, test_fps, train_fps):
    """Score test molecules and calculate distance_to_train.

    Processes test molecules in batches to control memory usage.
    """
    n_test = len(test_smiles)
    all_smiles = []
    all_targets = []
    all_dists = []

    for start in tqdm(
        range(0, n_test, BATCH_SIZE),
        desc="Scoring test molecules",
        unit="batch",
    ):
        end = min(start + BATCH_SIZE, n_test)
        fps = test_fps[start:end]
        smi = test_smiles[start:end]
        tgt = test_targets[start:end]

        s, t, d = _score_batch(fps, smi, tgt, train_fps)
        all_smiles.extend(s)
        all_targets.extend(t)
        all_dists.extend(d)

    return pd.DataFrame(
        {
            "standardized_smiles": all_smiles,
            "target": all_targets,
            "distance_to_train": all_dists,
        }
    )


def taylor_butina_split(
    df,
    train_size=0.8,
    threshold=0.65,
    n_jobs=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a Taylor-Butina OOD train/test split.

    This is a high-level convenience that:
    1. Clusters molecules by structural similarity (Butina sphere exclusion)
    2. Assigns largest clusters to training (in-distribution)
    3. Assigns smallest clusters to test (out-of-distribution)
    4. Computes ``distance_to_train`` for each test molecule (mean 5-NN
       Tanimoto distance to training set)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``standardized_smiles`` and ``target`` columns.
    train_size : float, optional
        Fraction of data for training (largest clusters). Default: 0.8.
    threshold : float, optional
        Tanimoto distance threshold for Butina clustering. Molecules within
        this distance are assigned to the same cluster. Default: 0.65.
    n_jobs : int or None, optional
        Unused (RDKit clustering is single-threaded). For API consistency.

    Returns
    -------
    train_df : pd.DataFrame
        Training set with ``standardized_smiles``, ``target``, and
        ``cluster_id`` columns.
    test_df : pd.DataFrame
        Test set with ``standardized_smiles``, ``target``, ``cluster_id``,
        and ``distance_to_train`` columns.
    """
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    smiles = df["standardized_smiles"].tolist()

    logger.info("Running Taylor-Butina clustering")
    logger.info("Generating Morgan fingerprints...")

    fps_bitvect, fps_numpy = _compute_morgan_fingerprints(smiles)

    logger.info("Clustering...")
    cluster_ids = _cluster_from_fps(fps_bitvect, threshold)
    n_clusters = len(set(cluster_ids))

    full_df = df.copy()
    full_df["cluster_id"] = cluster_ids

    train_df, test_df = split_by_clusters(full_df, train_size=train_size)

    logger.info(
        f"Clustering done: {n_clusters} clusters, "
        f"train={len(train_df)}, test={len(test_df)}"
    )

    logger.info("Computing distance_to_train for test molecules...")
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}

    train_smiles = train_df["standardized_smiles"].tolist()
    test_smiles = test_df["standardized_smiles"].tolist()
    test_targets = test_df["target"].tolist()

    train_fps = fps_numpy[[smiles_to_idx[s] for s in train_smiles]]
    test_fps = fps_numpy[[smiles_to_idx[s] for s in test_smiles]]

    logger.info("Scoring test set by distance to training set...")
    test_dist_df = _score_test_set(
        test_smiles, test_targets, test_fps, train_fps
    )

    test_df = test_df.merge(
        test_dist_df[["standardized_smiles", "distance_to_train"]],
        on="standardized_smiles",
        how="left",
    )

    cols = ["standardized_smiles", "target", "cluster_id", "distance_to_train"]
    test_df = test_df[cols]

    train_cols = ["standardized_smiles", "target", "cluster_id"]
    train_df = train_df[train_cols]

    logger.info(f"Scoring done: {len(test_df)} test molecules")

    return train_df, test_df
