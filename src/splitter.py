"""Taylor-Butina distance-based OOD splitter."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from skfp.model_selection import butina_train_test_split

MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048
K_NEIGHBORS = 5


def _generate_fingerprints(smiles_list):
    generator = AllChem.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE
        )
    fps = {}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fps[smiles] = generator.GetFingerprint(mol)

    return fps


def _filter_and_score_test_set(
    test_smiles, test_targets, smiles_to_fp, train_fps, distance_cutoff
):
    """Filter test molecules by distance and calculate distance_to_train."""
    retained_test_data = []

    for smi, tgt in zip(test_smiles, test_targets):
        test_fp = smiles_to_fp.get(smi)
        if not test_fp:
            continue

        similarities = BulkTanimotoSimilarity(test_fp, train_fps)

        distances = sorted([1.0 - s for s in similarities])

        if distances[0] > distance_cutoff:
            k = min(K_NEIGHBORS, len(distances))
            mean_dist = np.mean(distances[:k])

            retained_test_data.append(
                {
                    "standardized_smiles": smi,
                    "target": tgt,
                    "distance_to_train": mean_dist,
                }
            )

    return retained_test_data


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

    train_smiles, test_smiles, train_targets, test_targets = butina_train_test_split(
        smiles,
        targets,
        train_size=train_size,
        test_size=test_size,
        threshold=threshold,
        approximate=approximate,
        n_jobs=n_jobs,
    )

    smiles_to_fp = _generate_fingerprints(smiles)
    train_fps = [smiles_to_fp[s] for s in train_smiles]

    test_results = _filter_and_score_test_set(
        test_smiles, test_targets, smiles_to_fp, train_fps, distance_cutoff
    )

    train_df = pd.DataFrame(
        {
            "standardized_smiles": train_smiles,
            "target": train_targets,
        }
    )
    test_df = pd.DataFrame(test_results)

    return train_df, test_df
