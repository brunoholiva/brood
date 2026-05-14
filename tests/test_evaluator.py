"""Tests for the distance-aware evaluator."""

import numpy as np
import pandas as pd
import pytest

from src.evaluator import evaluate

N = 100
NP_RNG = np.random.default_rng(42)


@pytest.fixture
def perfect_df():
    """Perfect predictions: score == label, various distances."""
    return pd.DataFrame(
        {
            "target": [1, 1, 0, 0, 1, 0, 0, 1],
            "predicted_probability": [
                0.99,
                0.95,
                0.05,
                0.10,
                0.98,
                0.20,
                0.01,
                0.92,
            ],
            "distance_to_train": [
                0.1,
                0.2,
                0.15,
                0.25,
                0.5,
                0.7,
                0.8,
                0.9,
            ],
        }
    )


@pytest.fixture
def random_df():
    """Random scores for statistical tests."""
    rng = NP_RNG
    return pd.DataFrame(
        {
            "target": rng.integers(0, 2, size=N),
            "predicted_probability": rng.uniform(0, 1, size=N),
            "distance_to_train": rng.uniform(0, 1, size=N),
        }
    )


class TestGlobal:
    """Global metrics computation."""

    def test_perfect_predictions(self, perfect_df):
        """Perfect ranking gives AP=1 and BEDROC=1."""
        result = evaluate(perfect_df)
        g = result["global"]
        assert g["average_precision"] == pytest.approx(1.0)
        assert g["bedroc"] == pytest.approx(1.0)

    def test_output_keys(self, random_df):
        """Result dict has global and by_bin keys."""
        result = evaluate(random_df)
        assert "global" in result
        assert "by_bin" in result

    def test_global_has_required_fields(self, perfect_df):
        """Global entry has n, average_precision, bedroc."""
        g = evaluate(perfect_df)["global"]
        for key in ("n", "average_precision", "bedroc"):
            assert key in g

    def test_global_n_matches_input(self, random_df):
        """Global n equals total number of samples."""
        g = evaluate(random_df)["global"]
        assert g["n"] == N


class TestByBin:
    """Per-bin metrics structure and values."""

    def test_by_bin_is_list(self, perfect_df):
        """by_bin is a list of bin result dicts."""
        result = evaluate(perfect_df)
        assert isinstance(result["by_bin"], list)

    def test_three_default_bins(self, perfect_df):
        """Default bins are near, medium, far."""
        result = evaluate(perfect_df)
        names = [b["bin"] for b in result["by_bin"]]
        assert names == ["near", "medium", "far"]

    def test_each_bin_has_required_fields(self, perfect_df):
        """Each bin entry has bin, n, average_precision, bedroc."""
        result = evaluate(perfect_df)
        for entry in result["by_bin"]:
            for key in ("bin", "n", "average_precision", "bedroc"):
                assert key in entry

    def test_bin_counts_sum_to_total(self, random_df):
        """Bin counts sum to global n."""
        result = evaluate(random_df)
        total_bin = sum(b["n"] for b in result["by_bin"])
        assert total_bin == result["global"]["n"]


class TestValidation:
    """Input validation."""

    def test_missing_score_col(self):
        """Missing score column raises KeyError."""
        df = pd.DataFrame({"target": [0, 1], "distance_to_train": [0.1, 0.2]})
        with pytest.raises(KeyError, match="predicted_probability"):
            evaluate(df)

    def test_missing_label_col(self):
        """Missing label column raises KeyError."""
        df = pd.DataFrame(
            {
                "predicted_probability": [0.5, 0.5],
                "distance_to_train": [0.1, 0.2],
            }
        )
        with pytest.raises(KeyError, match="target"):
            evaluate(df)

    def test_missing_distance_col(self):
        """Missing distance column raises KeyError."""
        df = pd.DataFrame(
            {
                "target": [0, 1],
                "predicted_probability": [0.5, 0.5],
            }
        )
        with pytest.raises(KeyError, match="distance_to_train"):
            evaluate(df)

    def test_custom_column_names(self):
        """Custom column names are used correctly."""
        df = pd.DataFrame(
            {
                "label": [1, 0],
                "score": [0.9, 0.1],
                "dist": [0.2, 0.8],
            }
        )
        result = evaluate(
            df,
            score_col="score",
            label_col="label",
            distance_col="dist",
        )
        assert "global" in result


class TestEdgeCases:
    """Empty bins and custom bins."""

    def test_empty_bin_returns_nan(self, perfect_df):
        """Bin with no molecules returns NaN metrics."""
        result = evaluate(perfect_df, bins=[(0.99, 1.0, "empty")])
        entry = result["by_bin"][0]
        assert entry["n"] == 0
        assert np.isnan(entry["average_precision"])
        assert np.isnan(entry["bedroc"])

    def test_custom_bins(self, perfect_df):
        """Custom bin edges produce expected bin names."""
        result = evaluate(
            perfect_df,
            bins=[(0.0, 0.5, "close"), (0.5, 1.01, "far")],
        )
        names = [b["bin"] for b in result["by_bin"]]
        assert names == ["close", "far"]


class TestIntegration:
    """End-to-end: splitter + model + evaluator."""

    def test_full_workflow(self):
        """Full pipeline: split, train, predict, evaluate."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        from src.fingerprints import MorganFingerprintTransformer
        from src.splitter import taylor_butina_split

        smiles = [
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
        targets = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
        df = pd.DataFrame({"standardized_smiles": smiles, "target": targets})
        train_df, test_df = taylor_butina_split(df, distance_cutoff=0.0)

        pipe = Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )
        pipe.fit(train_df["standardized_smiles"], train_df["target"])
        test_df["predicted_probability"] = pipe.predict_proba(
            test_df["standardized_smiles"]
        )[:, 1]

        result = evaluate(test_df)

        assert result["global"]["average_precision"] >= 0
        assert result["global"]["bedroc"] >= 0
        for entry in result["by_bin"]:
            assert entry["n"] >= 0
