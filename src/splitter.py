"""Distance-based OOD splitting using Taylor-Butina or Leader clustering.

For datasets < 50k: exact Butina sphere-exclusion (all-pairs distances).
For datasets >= 50k: Leader algorithm (streaming, O(n x k) memory).
"""

import warnings

import numpy as np
import pandas as pd
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", message="Data was converted to boolean for metric jaccard"
)

K_NEIGHBORS = 5
BATCH_SIZE = 5000
LARGE_DATASET_THRESHOLD = 50000


def _smiles_to_mols(smiles_list: list[str]) -> list[Chem.Mol]:
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
    return mols


def _validate_mols(mols: list, expected_count: int) -> None:
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
    """Morgan fingerprints in RDKit and numpy formats."""
    mols = _smiles_to_mols(smiles_list)
    _validate_mols(mols, len(smiles_list))

    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps_bitvect = [generator.GetFingerprint(mol) for mol in mols]
    fps_numpy = np.array(fps_bitvect)

    return fps_bitvect, fps_numpy


def _butina_cluster(
    fps_bitvect: list,
    threshold: float,
) -> np.ndarray:
    """Butina sphere-exclusion via all-pairs distance matrix (n < 50k)."""
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


def _leader_cluster(
    fps_bitvect: list,
    threshold: float,
) -> np.ndarray:
    """Streaming sphere-exclusion (Leader algorithm). O(n x k) memory.

    Each molecule is compared only to existing cluster centroids, not
    the full pairwise matrix. Used for n >= 50k.
    """
    n = len(fps_bitvect)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    sim_cutoff = 1.0 - threshold

    clusters: list[list[int]] = []
    centroids: list[DataStructs.ExplicitBitVect] = []

    for i, fps in enumerate(fps_bitvect):
        assigned = False
        for c_idx, centroid_fps in enumerate(centroids):
            sim = DataStructs.TanimotoSimilarity(fps, centroid_fps)
            if sim >= sim_cutoff:
                clusters[c_idx].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            centroids.append(fps)

    sorted_clusters = sorted(clusters, key=lambda c: -len(c))

    cluster_ids = np.zeros(n, dtype=int)
    for cid, members in enumerate(sorted_clusters):
        for idx in members:
            cluster_ids[idx] = cid

    return cluster_ids


def _cluster_from_fps(fps_bitvect: list, threshold: float) -> np.ndarray:
    """Dispatch to Butina or Leader clustering by dataset size."""
    if len(fps_bitvect) >= LARGE_DATASET_THRESHOLD:
        logger.info(
            f"{len(fps_bitvect)} molecules >= {LARGE_DATASET_THRESHOLD}, "
            "using Leader clustering (streaming)"
        )
        return _leader_cluster(fps_bitvect, threshold)
    return _butina_cluster(fps_bitvect, threshold)


def butina_cluster(
    smiles_list: list[str],
    threshold: float = 0.65,
    radius: int = 2,
    fp_size: int = 2048,
) -> np.ndarray:
    """Cluster molecules using sphere-exclusion (Butina / Leader).

    Returns cluster IDs ordered by descending size.
    Uses Leader algorithm for datasets >= 50k.
    """
    fps_bitvect, _ = _compute_morgan_fingerprints(
        smiles_list, radius=radius, fp_size=fp_size
    )
    return _cluster_from_fps(fps_bitvect, threshold)


def _accumulate_clusters(
    cluster_sizes: pd.Series,
    n_target: int,
) -> list:
    """Greedily accumulate smallest clusters up to n_target molecules."""
    selected = []
    cum = 0
    for cid, size in cluster_sizes.items():
        if cum + size <= n_target:
            selected.append(cid)
            cum += size
        else:
            break
    return selected


def split_train_test(
    df: pd.DataFrame,
    train_size: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by cluster ID into train/test partitions.

    Smallest clusters go to test, largest go to training.
    """
    cluster_sizes = df.groupby("cluster_id").size().sort_values(ascending=True)
    n_total = len(df)
    n_test = int((1.0 - train_size) * n_total)
    test_clusters = _accumulate_clusters(cluster_sizes, n_test)
    train_clusters = list(cluster_sizes.drop(test_clusters).index)

    train_df = df[df["cluster_id"].isin(train_clusters)].copy()
    test_df = df[df["cluster_id"].isin(test_clusters)].copy()

    logger.info(
        f"Split: train={len(train_df)} ({len(train_clusters)} clusters), "
        f"test={len(test_df)} ({len(test_clusters)} clusters)"
    )
    return train_df, test_df


def split_train_valid_test(
    df: pd.DataFrame,
    train_size: float = 0.8,
    valid_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by cluster ID into train/valid/test partitions.

    Smallest clusters go to test/valid, largest go to training.
    """
    test_size = 1.0 - train_size - valid_size
    cluster_sizes = df.groupby("cluster_id").size().sort_values(ascending=True)
    n_total = len(df)
    test_clusters = _accumulate_clusters(
        cluster_sizes, int(test_size * n_total)
    )
    remaining = cluster_sizes.drop(test_clusters)
    valid_clusters = _accumulate_clusters(remaining, int(valid_size * n_total))
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


def split_by_clusters(
    df: pd.DataFrame,
    train_size: float = 0.8,
    valid_size: float | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Split DataFrame by cluster ID for OOD train/(valid)/test partitions.

    .. deprecated::
        Use :func:`split_train_test` or :func:`split_train_valid_test` instead.
    """
    if valid_size is not None:
        return split_train_valid_test(df, train_size, valid_size)
    return split_train_test(df, train_size)


def _assign_clusters(
    df: pd.DataFrame, cluster_ids: np.ndarray
) -> pd.DataFrame:
    """Add cluster_id column to DataFrame copy."""
    full_df = df.copy()
    full_df["cluster_id"] = cluster_ids
    return full_df


def _score_batch(batch_fps, train_fps):
    """Mean 5-NN Tanimoto distance for one batch of test molecules."""
    dists = pairwise_distances(
        batch_fps, train_fps, metric="jaccard", n_jobs=1
    )
    k = min(K_NEIGHBORS, dists.shape[1])
    mean_dists = []
    for row in dists:
        nearest = np.partition(row, k - 1)[:k] if k < len(row) else row
        mean_dists.append(nearest.mean())
    return mean_dists


def _score_test_set(test_smiles, test_targets, test_fps, train_fps):
    """Score test molecules and calculate distance_to_train.

    Processes in batches to control memory usage.
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
        dists = _score_batch(test_fps[start:end], train_fps)
        all_smiles.extend(test_smiles[start:end])
        all_targets.extend(test_targets[start:end])
        all_dists.extend(dists)

    return pd.DataFrame(
        {
            "standardized_smiles": all_smiles,
            "target": all_targets,
            "distance_to_train": all_dists,
        }
    )


def _compute_test_distances(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    fps_numpy: np.ndarray,
    smiles: list[str],
) -> pd.DataFrame:
    """Compute distance_to_train for test molecules and merge into test_df."""
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}
    train_fps = fps_numpy[
        [smiles_to_idx[s] for s in train_df["standardized_smiles"]]
    ]
    test_fps = fps_numpy[
        [smiles_to_idx[s] for s in test_df["standardized_smiles"]]
    ]

    test_dist_df = _score_test_set(
        test_df["standardized_smiles"].tolist(),
        test_df["target"].tolist(),
        test_fps,
        train_fps,
    )

    return test_df.merge(
        test_dist_df[["standardized_smiles", "distance_to_train"]],
        on="standardized_smiles",
        how="left",
    )


def _format_output(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select final columns for output."""
    test_df = test_df[
        ["standardized_smiles", "target", "cluster_id", "distance_to_train"]
    ]
    train_df = train_df[["standardized_smiles", "target", "cluster_id"]]
    return train_df, test_df


def taylor_butina_split(
    df,
    train_size=0.8,
    threshold=0.65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOD train/test split using Taylor-Butina clustering.

    1. Clusters molecules by structural similarity (Butina / Leader)
    2. Largest clusters -> training, smallest -> test
    3. Computes ``distance_to_train`` for test (mean 5-NN Tanimoto dist)

    Uses Leader algorithm (streaming) for datasets >= 50k.
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

    full_df = _assign_clusters(df, cluster_ids)
    train_df, test_df = split_train_test(full_df, train_size=train_size)

    logger.info(
        f"Clustering done: {n_clusters} clusters, "
        f"train={len(train_df)}, test={len(test_df)}"
    )

    logger.info("Computing distance_to_train for test molecules...")
    test_df = _compute_test_distances(test_df, train_df, fps_numpy, smiles)

    train_df, test_df = _format_output(train_df, test_df)

    logger.info(f"Scoring done: {len(test_df)} test molecules")

    return train_df, test_df
