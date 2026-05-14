"""Tests for the high-level screen() API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src import screen
from src.fingerprints import MorganFingerprintTransformer

SMILES = [
    "CCO",
    "CCN",
    "CCC",
    "CCCl",
    "CCBr",
    "CCI",
    "CCF",
    "CC=O",
    "CCCO",
    "CCN(CC)CC",
]
TARGETS = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]


@pytest.fixture
def df():
    """Create a DataFrame with 10 diverse SMILES."""
    return pd.DataFrame({"standardized_smiles": SMILES, "target": TARGETS})


class TestReturnStructure:
    """Verify the return dict structure."""

    def test_has_all_keys(self, df):
        """Result dict has global, by_bin, test_df, pipeline keys."""
        result = screen(df, RandomForestClassifier(random_state=42))
        for key in ("global", "by_bin", "test_df", "pipeline"):
            assert key in result

    def test_global_has_metrics(self, df):
        """Global entry has n, average_precision, bedroc."""
        result = screen(df, RandomForestClassifier(random_state=42))
        for key in ("n", "average_precision", "bedroc"):
            assert key in result["global"]

    def test_by_bin_is_list(self, df):
        """by_bin is a list."""
        result = screen(df, RandomForestClassifier(random_state=42))
        assert isinstance(result["by_bin"], list)

    def test_test_df_is_dataframe(self, df):
        """test_df is a DataFrame."""
        result = screen(df, RandomForestClassifier(random_state=42))
        assert isinstance(result["test_df"], pd.DataFrame)

    def test_pipeline_is_pipeline(self, df):
        """Check result pipeline is a Pipeline object."""
        result = screen(df, RandomForestClassifier(random_state=42))
        assert isinstance(result["pipeline"], Pipeline)


class TestAutoWrap:
    """Bare sklearn estimators are auto-wrapped."""

    def test_bare_estimator_wrapped(self, df):
        """Bare RF gets wrapped in a Pipeline."""
        result = screen(df, RandomForestClassifier(random_state=42))
        assert isinstance(result["pipeline"], Pipeline)

    def test_pipeline_preserved(self, df):
        """Pre-built Pipeline passes through without re-wrap."""
        pipe = Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )
        result = screen(df, pipe)
        assert result["pipeline"] is pipe


class TestColumnNames:
    """Custom column name mapping."""

    def test_custom_smiles_col(self):
        """Works with custom smiles column name."""
        df = pd.DataFrame({"smiles": SMILES, "target": TARGETS})
        result = screen(
            df,
            RandomForestClassifier(random_state=42),
            smiles_col="smiles",
        )
        assert result["global"]["n"] > 0

    def test_custom_target_col(self):
        """Works with custom target column name."""
        df = pd.DataFrame({"standardized_smiles": SMILES, "label": TARGETS})
        result = screen(
            df,
            RandomForestClassifier(random_state=42),
            target_col="label",
        )
        assert result["global"]["n"] > 0

    def test_missing_column_raises(self):
        """Missing column raises KeyError."""
        df = pd.DataFrame({"wrong": SMILES, "target": TARGETS})
        with pytest.raises(KeyError, match="smiles"):
            screen(
                df,
                RandomForestClassifier(random_state=42),
                smiles_col="smiles",
            )


class TestEdgeCases:
    """Edge cases for screen()."""

    def test_empty_test_set(self, df):
        """distance_cutoff=1.0 produces empty test set."""
        result = screen(
            df,
            RandomForestClassifier(random_state=42),
            distance_cutoff=1.0,
        )
        assert result["global"]["n"] == 0
        assert np.isnan(result["global"]["average_precision"])
        assert result["pipeline"] is None
        assert len(result["test_df"]) == 0

    def test_custom_bins(self, df):
        """Custom bin edges pass through to evaluate."""
        bins = [(0.0, 0.5, "close"), (0.5, 1.01, "far")]
        result = screen(
            df,
            RandomForestClassifier(random_state=42),
            distance_cutoff=0.0,
            distance_bins=bins,
        )
        names = [b["bin"] for b in result["by_bin"]]
        assert names == ["close", "far"]


class TestRandomState:
    """random_state parameter propagates to model."""

    def test_random_state_set_on_bare_estimator(self, df):
        """random_state is set on a bare model."""
        model = RandomForestClassifier()
        assert model.random_state is None
        screen(df, model, random_state=42)
        assert model.random_state == 42

    def test_random_state_overwrites_previous(self, df):
        """random_state from screen overwrites model's previous value."""
        model = RandomForestClassifier(random_state=7)
        screen(df, model, random_state=42)
        assert model.random_state == 42


class TestIntegration:
    """End-to-end metrics sanity."""

    def test_all_actives(self):
        """All-actives runs without error; predict_proba has 1 column."""
        df = pd.DataFrame(
            {"standardized_smiles": SMILES[:6], "target": [1] * 6}
        )
        result = screen(
            df,
            RandomForestClassifier(random_state=42),
            distance_cutoff=0.0,
        )
        assert "global" in result
        assert "by_bin" in result

    def test_all_inactives(self):
        """All-inactives runs without error; predict_proba has 1 column."""
        df = pd.DataFrame(
            {"standardized_smiles": SMILES[:6], "target": [0] * 6}
        )
        result = screen(
            df,
            RandomForestClassifier(random_state=42),
            distance_cutoff=0.0,
        )
        assert "global" in result
        assert "by_bin" in result

    def test_deterministic_reproducibility(self, df):
        """Same inputs produce same metrics."""
        rf = RandomForestClassifier(random_state=42)
        r1 = screen(df, rf, random_state=42)
        r2 = screen(df, rf, random_state=42)
        assert r1["global"]["average_precision"] == pytest.approx(
            r2["global"]["average_precision"]
        )
        assert r1["global"]["bedroc"] == pytest.approx(r2["global"]["bedroc"])
