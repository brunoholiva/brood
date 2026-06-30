"""Concatenated featurizer for combining multiple molecular transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class ConcatenatedFeaturizer(BaseEstimator, TransformerMixin):
    """Concatenate outputs of multiple featurizers.

    Stateless sklearn transformer. Each sub-featurizer transforms the
    input independently; results are concatenated column-wise.

    Parameters
    ----------
    featurizers : list of TransformerMixin
        Featurizers to apply and concatenate.
    names : list of str, optional
        Names for progress bar labels.
    """

    def __init__(self, featurizers, names=None):
        self.featurizers = featurizers
        self.names = names or [type(f).__name__ for f in featurizers]

    def fit(self, X, y=None):
        """Stateless transformer; returns self."""
        return self

    def transform(self, X):
        """Transform X with each featurizer and concatenate."""
        parts = []
        for name, featurizer in tqdm(
            zip(self.names, self.featurizers),
            desc="Concatenate",
            unit="feat",
            leave=True,
        ):
            parts.append(featurizer.transform(X))
        return np.concatenate(parts, axis=1)
