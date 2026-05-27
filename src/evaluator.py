"""Distance-aware evaluation metrics for OOD molecular screening."""

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import average_precision_score

DEFAULT_BINS = [
    (0.0, 0.3, "near"),
    (0.3, 0.6, "medium"),
    (0.6, 1.01, "far"),
]


def evaluate(
    df,
    score_col="predicted_probability",
    label_col="target",
    distance_col="distance_to_train",
    bins=None,
    bedroc_alpha=20.0,
):
    """Evaluate screening performance globally and binned by distance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``label_col``, ``score_col``, and ``distance_col``.
    score_col : str, optional
        Column name for predicted probabilities (positive class).
    label_col : str, optional
        Column name for true binary labels (0/1).
    distance_col : str, optional
        Column name for distance to training set.
    bins : list of (low, high, name), optional
        Distance bins for per-bin metrics. Defaults to near (0-0.3),
        medium (0.3-0.6), far (0.6+).
    bedroc_alpha : float, optional
        BEDROC early-enrichment parameter (higher = more weight
        on top-ranked molecules).

    Returns
    -------
    result : dict
        ``{"global": {...}, "by_bin": [{...}, ...]}``. Each entry
        contains ``n``, ``average_precision``, and ``bedroc``.
    """
    _validate_columns(df, score_col, label_col, distance_col)

    y_true = df[label_col].to_numpy()
    y_score = df[score_col].to_numpy()
    distances = df[distance_col].to_numpy()

    global_metrics = _compute_ranking_metrics(y_true, y_score, bedroc_alpha)

    bins = bins or DEFAULT_BINS
    by_bin = []
    for low, high, name in bins:
        mask = (distances >= low) & (distances < high)
        subset = df[mask]
        if len(subset) == 0:
            by_bin.append(_empty_bin(name))
            continue
        metrics = _compute_ranking_metrics(
            subset[label_col].to_numpy(),
            subset[score_col].to_numpy(),
            bedroc_alpha,
        )
        metrics["bin"] = name
        metrics["n"] = len(subset)
        metrics["n_actives"] = int(subset[label_col].sum())
        by_bin.append(metrics)

    return {"global": global_metrics, "by_bin": by_bin}


def _validate_columns(df, score_col, label_col, distance_col):
    required = [score_col, label_col, distance_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def _compute_ranking_metrics(y_true, y_score, alpha):
    """AP and BEDROC for one label/score pair."""
    ap = average_precision_score(y_true, y_score)

    order = np.argsort(y_score)[::-1]
    sorted_labels = y_true[order]
    scores_2d = sorted_labels.reshape(-1, 1)
    bedroc = float(CalcBEDROC(scores_2d, 0, alpha))

    return {"n": len(y_true), "average_precision": ap, "bedroc": bedroc}


def _empty_bin(name):
    return {
        "bin": name,
        "n": 0,
        "n_actives": 0,
        "average_precision": float("nan"),
        "bedroc": float("nan"),
    }
