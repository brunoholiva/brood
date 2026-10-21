"""A battery of ranking and probability metrics for one prediction sample.

The screening pipeline ranks models by average precision. This module adds
the other metrics the same predictions could have been scored with, so the
choice of metric can be measured instead of assumed.

Every metric is computed from ``(y_true, y_score)`` alone, which means a
metric can be added to an existing run without re-training anything: the
per-fold predictions are already on disk.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from brood.evaluation.evaluator import _compute_ranking_metrics

#: Direction that counts as "better" for every metric of the battery.
METRIC_DIRECTION = {
    "ap": "higher",
    "auc": "higher",
    "bedroc": "higher",
    "ef1": "higher",
    "mcc": "higher",
    "brier": "lower",
    "nll": "lower",
}

#: Screening budget used by the enrichment factor.
ENRICHMENT_FRACTION = 0.01


def enrichment_factor(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = ENRICHMENT_FRACTION,
) -> float:
    """Enrichment of actives in the top ``fraction`` of the ranking.

    A value of 1 means the top of the list is no better than random; 10
    means ten times more actives than the prevalence would give.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Predicted probabilities.
    fraction : float, optional
        Fraction of the ranked list that is screened, 0.01 = top 1%.

    Returns
    -------
    float
        Enrichment factor, or NaN when the sample has no actives.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = max(1, int(round(fraction * n)))
    top = np.argsort(y_score)[::-1][:k]
    prevalence = float(np.mean(y_true))
    if prevalence == 0:
        return float("nan")
    return float(y_true[top].sum() / k) / prevalence


def sample_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute every metric of the battery for one sample.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels.
    y_score : np.ndarray
        Predicted probabilities.
    threshold : float, optional
        Decision threshold for MCC. Callers should pass a threshold that
        was fixed without looking at the sample's labels, e.g. the
        training prevalence.

    Returns
    -------
    dict[str, float]
        Metric name to value. See :data:`METRIC_DIRECTION` for the
        direction that counts as better.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    ranking = _compute_ranking_metrics(y_true, y_score, 20.0)
    clipped = np.clip(y_score, 1e-7, 1.0 - 1e-7)
    return {
        "ap": float(ranking.average_precision),
        "auc": float(roc_auc_score(y_true, y_score)),
        "bedroc": float(ranking.bedroc),
        "brier": float(ranking.brier),
        "nll": float(log_loss(y_true, clipped)),
        "mcc": float(
            matthews_corrcoef(y_true, (y_score >= threshold).astype(int))
        ),
        "ef1": enrichment_factor(y_true, y_score, ENRICHMENT_FRACTION),
    }
