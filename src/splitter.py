"""Taylor-Butina distance-based OOD splitter."""

import numpy as np
import pandas as pd
from skfp.model_selection import butina_train_test_split
from sklearn.metrics.pairwise import pairwise_distances

from src.fingerprints import MorganFingerprintTransformer

K_NEIGHBORS = 5


def _generate_fingerprints(smiles_list):
    return MorganFingerprintTransformer().transform(smiles_list)


def _filter_and_score_test_set(
    test_smiles, test_targets, test_fps, train_fps, distance_cutoff
):
    """Filter test molecules by distance and calculate distance_to_train.

    Converts inputs to float64 once to avoid sklearn UserWarning on int
    data. Uses Jaccard distance (= Tanimoto distance for binary vectors).
    """
    kept_smiles = []
    kept_targets = []
    mean_dists = []

    train_fps_b = train_fps.astype(bool)

    for i, (smi, tgt) in enumerate(zip(test_smiles, test_targets)):
        dists = pairwise_distances(
            test_fps[i : i + 1].astype(bool),
            train_fps_b,
            metric="jaccard",
        )[0]

        min_dist = dists.min()
        if min_dist <= distance_cutoff:
            continue

        kept_smiles.append(smi)
        kept_targets.append(tgt)
        k = min(K_NEIGHBORS, len(dists))
        nearest = np.partition(dists, k - 1)[:k] if k < len(dists) else dists
        mean_dists.append(nearest.mean())

    return pd.DataFrame(
        {
            "standardized_smiles": kept_smiles,
            "target": kept_targets,
            "distance_to_train": mean_dists,
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

    fps = _generate_fingerprints(smiles)
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}
    train_fps = fps[[smiles_to_idx[s] for s in train_smiles]]
    test_fps = fps[[smiles_to_idx[s] for s in test_smiles]]

    test_df = _filter_and_score_test_set(
        test_smiles, test_targets, test_fps, train_fps, distance_cutoff
    )
    train_df = pd.DataFrame(
        {
            "standardized_smiles": train_smiles,
            "target": train_targets,
        }
    )

    return train_df, test_df
