"""brood — Out-of-Distribution Molecular Evaluation Framework.

OOD molecular evaluation framework for antibiotic discovery.
"""

import warnings

import pandas as pd
from sklearn.pipeline import Pipeline

from .evaluator import evaluate
from .fingerprints import MorganFingerprintTransformer
from .splitter import taylor_butina_split

__all__ = [
    "MorganFingerprintTransformer",
    "evaluate",
    "screen",
    "taylor_butina_split",
]


def screen(
    df,
    model,
    smiles_col="standardized_smiles",
    target_col="target",
    score_col="predicted_probability",
    train_size=0.8,
    test_size=0.2,
    threshold=0.65,
    approximate=True,
    distance_cutoff=0.2,
    distance_bins=None,
    bedrock_alpha=20.0,
    random_state=None,
    n_jobs=None,
):
    """One-shot OOD evaluation: split -> train -> predict -> evaluate.

    Wraps the 4-step workflow (split, train, predict, evaluate) into a
    single call. Bare sklearn estimators are automatically wrapped in a
    Pipeline with MorganFingerprintTransformer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``smiles_col`` and ``target_col``.
    model : sklearn estimator or Pipeline
        Any sklearn-compatible model. If it is not already a Pipeline, it
        will be auto-wrapped with MorganFingerprintTransformer. For non-
        sklearn models, use the modular API instead.
    smiles_col : str, default="standardized_smiles"
        Column with SMILES strings.
    target_col : str, default="target"
        Column with binary labels (0/1).
    score_col : str, default="predicted_probability"
        Column name for predicted probabilities in the output test_df.
    train_size : float, default=0.8
        Fraction of data for training.
    test_size : float, default=0.2
        Fraction of data for testing.
    threshold : float, default=0.65
        Tanimoto distance threshold for Taylor-Butina clustering.
    approximate : bool, default=True
        Use approximate similarity (NNDescent) for clustering.
    distance_cutoff : float, default=0.2
        Test molecules with min Tanimoto distance to training set at or
        below this value are excluded.
    distance_bins : list of (low, high, name), optional
        Bins for per-bin evaluation. Defaults to near (0-0.3),
        medium (0.3-0.6), far (0.6+).
    bedrock_alpha : float, default=20.0
        BEDROC early-enrichment parameter.
    random_state : int or None, optional
        Seed for reproducibility. Applied to the model if it has a
        ``random_state`` attribute. Not used by the splitter (Taylor-
        Butina is deterministic).
    n_jobs : int or None, optional
        Number of parallel jobs for the splitter.

    Returns
    -------
    dict
        ``{"global": {"n", "average_precision", "bedroc"},
        "by_bin": [{"bin", "n", "average_precision", "bedroc"}, ...],
        "test_df": pd.DataFrame, "pipeline": Pipeline}``
    """
    for col in (smiles_col, target_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    split_df = pd.DataFrame(
        {
            "standardized_smiles": df[smiles_col],
            "target": df[target_col],
        }
    )

    train_df, test_df = taylor_butina_split(
        split_df,
        train_size=train_size,
        test_size=test_size,
        threshold=threshold,
        approximate=approximate,
        distance_cutoff=distance_cutoff,
        n_jobs=n_jobs,
    )

    if len(test_df) == 0:
        warnings.warn("Test set is empty after distance filtering.")
        return {
            "global": {
                "n": 0,
                "average_precision": float("nan"),
                "bedroc": float("nan"),
            },
            "by_bin": [],
            "test_df": test_df,
            "pipeline": None,
        }

    if not isinstance(model, Pipeline):
        if random_state is not None and hasattr(model, "random_state"):
            model.random_state = random_state
        model = Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("clf", model),
            ]
        )

    model.fit(train_df["standardized_smiles"], train_df["target"])
    probas = model.predict_proba(test_df["standardized_smiles"])
    if probas.shape[1] == 1:
        unique_cls = train_df["target"].unique()
        if len(unique_cls) == 1:
            test_df[score_col] = float(unique_cls[0])
        else:
            test_df[score_col] = float("nan")
    else:
        test_df[score_col] = probas[:, 1]

    results = evaluate(
        test_df,
        score_col=score_col,
        label_col="target",
        distance_col="distance_to_train",
        bins=distance_bins,
        bedrock_alpha=bedrock_alpha,
    )
    results["test_df"] = test_df
    results["pipeline"] = model
    return results
