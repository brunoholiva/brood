"""Taylor-Butina distance-based OOD splitter."""

import warnings

import numpy as np
import pandas as pd
from loguru import logger
from skfp.model_selection import butina_train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from .fingerprints import MorganFingerprintTransformer

warnings.filterwarnings(
    "ignore", message="Data was converted to boolean for metric jaccard"
)

K_NEIGHBORS = 5
BATCH_SIZE = 5000


def _generate_fingerprints(smiles_list):
    return MorganFingerprintTransformer().transform(smiles_list)


def _filter_and_score_batch(
    batch_fps, batch_smiles, batch_targets, train_fps, distance_cutoff
):
    """Filter one batch of test molecules and compute distance_to_train."""
    dists = pairwise_distances(
        batch_fps, train_fps, metric="jaccard", n_jobs=-1
    )
    min_dists = dists.min(axis=1)
    keep = min_dists > distance_cutoff

    kept_smiles = [s for s, ok in zip(batch_smiles, keep) if ok]
    kept_targets = [t for t, ok in zip(batch_targets, keep) if ok]

    kept_dists = dists[keep]
    k = min(K_NEIGHBORS, kept_dists.shape[1]) if kept_dists.ndim == 2 else 0
    mean_dists = []
    for row in kept_dists:
        nearest = np.partition(row, k - 1)[:k] if k < len(row) else row
        mean_dists.append(nearest.mean())

    return kept_smiles, kept_targets, mean_dists


def _filter_and_score_test_set(
    test_smiles, test_targets, test_fps, train_fps, distance_cutoff
):
    """Filter test molecules by distance and calculate distance_to_train.

    Processes test molecules in batches to control memory usage.
    """
    n_test = len(test_smiles)
    all_smiles = []
    all_targets = []
    all_dists = []

    for start in tqdm(
        range(0, n_test, BATCH_SIZE),
        desc="Filtering test molecules",
        unit="batch",
    ):
        end = min(start + BATCH_SIZE, n_test)
        fps = test_fps[start:end]
        smi = test_smiles[start:end]
        tgt = test_targets[start:end]

        kept_s, kept_t, kept_d = _filter_and_score_batch(
            fps, smi, tgt, train_fps, distance_cutoff
        )
        all_smiles.extend(kept_s)
        all_targets.extend(kept_t)
        all_dists.extend(kept_d)

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
    test_size=0.2,
    threshold=0.65,
    approximate=True,
    distance_cutoff=0.2,
    n_jobs=None,
):
    """Split molecules into train/test using Taylor-Butina clustering.

    Molecules are clustered by scaffold similarity. The smallest clusters
    (most novel scaffolds) form the test set. Test molecules too similar
    to any training molecule (Tanimoto distance <= distance_cutoff) are
    excluded, and each retained test molecule gets a `distance_to_train`
    score (mean 5-NN Tanimoto distance to the training set).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``smiles`` and ``target`` columns.
    train_size : float, optional
        Fraction of data for training.
    test_size : float, optional
        Fraction of data for testing.
    threshold : float, optional
        Tanimoto distance threshold for Taylor-Butina clustering.
    approximate : bool, optional
        Use approximate similarity (NNDescent) for clustering. Falls
        back to exact for datasets under 5000 molecules.
    distance_cutoff : float, optional
        Test molecules with minimum Tanimoto distance to the training
        set at or below this value are removed from the test set.
    n_jobs : int or None, optional
        Number of parallel jobs.

    Returns
    -------
    train_df : pd.DataFrame
        Training set with ``smiles`` and ``target`` columns.
    test_df : pd.DataFrame
        Test set with ``smiles``, ``target``, and ``distance_to_train``
        columns.
    """
    smiles = df["standardized_smiles"].tolist()
    targets = df["target"].tolist()

    logger.info("Running Taylor-Butina clustering...")
    train_smiles, test_smiles, train_targets, test_targets = (
        butina_train_test_split(
            smiles,
            targets,
            train_size=train_size,
            test_size=test_size,
            threshold=threshold,
            approximate=approximate,
            n_jobs=n_jobs,
        )
    )
    logger.info(
        f"Clustering done: train={len(train_smiles)}, test={len(test_smiles)}"
    )

    logger.info("Generating fingerprints...")
    fps = _generate_fingerprints(smiles)
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}
    train_fps = fps[[smiles_to_idx[s] for s in train_smiles]]
    test_fps = fps[[smiles_to_idx[s] for s in test_smiles]]

    logger.info(
        f"Filtering test set (distance_cutoff={distance_cutoff})..."
    )
    test_df = _filter_and_score_test_set(
        test_smiles, test_targets, test_fps, train_fps, distance_cutoff
    )
    logger.info(
        f"Filtering done: {len(test_df)} test molecules retained"
    )

    train_df = pd.DataFrame(
        {
            "standardized_smiles": train_smiles,
            "target": train_targets,
        }
    )

    return train_df, test_df
