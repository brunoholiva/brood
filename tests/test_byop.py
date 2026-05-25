"""Tests for the BYOP helpers in src/byop.py."""

import pandas as pd
import pytest

from src.byop import evaluate_predictions, load_predictions, merge_predictions

SMILES = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]


@pytest.fixture
def test_df():
    """Complete test set with targets and distances."""
    return pd.DataFrame(
        {
            "standardized_smiles": SMILES,
            "target": [0, 0, 1, 1, 0],
            "distance_to_train": [0.35, 0.42, 0.51, 0.68, 0.22],
            "metadata": ["a", "b", "c", "d", "e"],
        }
    )


@pytest.fixture
def preds_csv(tmp_path):
    """Write a temporary predictions CSV."""
    path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCC", "CCBr"],
            "predicted_probability": [0.1, 0.9, 0.3],
        }
    ).to_csv(path, index=False)
    return path


class TestLoadPredictions:
    """Verify load_predictions validation."""

    def test_loads_valid_csv(self, preds_csv):
        preds = load_predictions(preds_csv)
        assert list(preds["standardized_smiles"]) == ["CCO", "CCC", "CCBr"]
        assert list(preds["predicted_probability"]) == [0.1, 0.9, 0.3]

    def test_raises_on_missing_smiles_col(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"wrong": ["CCO"], "score": [0.5]}).to_csv(
            path, index=False
        )
        with pytest.raises(KeyError, match="standardized_smiles"):
            load_predictions(path)

    def test_raises_on_missing_score_col(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"standardized_smiles": ["CCO"], "wrong": [0.5]}).to_csv(
            path, index=False
        )
        with pytest.raises(KeyError, match="predicted_probability"):
            load_predictions(path)

    def test_custom_column_names(self, tmp_path):
        path = tmp_path / "custom.csv"
        pd.DataFrame({"smiles": ["CCO"], "prob": [0.5]}).to_csv(
            path, index=False
        )
        preds = load_predictions(path, smiles_col="smiles", score_col="prob")
        assert list(preds["smiles"]) == ["CCO"]
        assert list(preds["prob"]) == [0.5]


class TestMergePredictions:
    """Verify merge_predictions behavior."""

    def test_merges_correctly(self, test_df, preds_csv):
        preds = load_predictions(preds_csv)
        merged = merge_predictions(test_df, preds)

        assert "standardized_smiles" in merged.columns
        assert "target" in merged.columns
        assert "distance_to_train" in merged.columns
        assert "predicted_probability" in merged.columns
        assert "metadata" in merged.columns

        assert len(merged) == 3

    def test_inner_join_drops_unmatched(self, test_df, tmp_path):
        path = tmp_path / "partial.csv"
        pd.DataFrame(
            {
                "standardized_smiles": ["CCO", "NOT_A_SMILE"],
                "predicted_probability": [0.1, 0.9],
            }
        ).to_csv(path, index=False)
        preds = load_predictions(path)
        merged = merge_predictions(test_df, preds)
        assert list(merged["standardized_smiles"]) == ["CCO"]

    def test_keeps_all_test_columns(self, test_df, preds_csv):
        preds = load_predictions(preds_csv)
        merged = merge_predictions(test_df, preds)
        for col in test_df.columns:
            assert col in merged.columns

    def test_raises_when_no_smiles_in_test(self, preds_csv):
        bad = pd.DataFrame({"wrong": ["CCO"], "target": [0]})
        preds = load_predictions(preds_csv)
        with pytest.raises(KeyError, match="standardized_smiles"):
            merge_predictions(bad, preds)


class TestEvaluatePredictions:
    """Verify evaluate_predictions returns correct types."""

    def test_returns_brood_result(self, test_df, preds_csv):
        preds = load_predictions(preds_csv)
        merged = merge_predictions(test_df, preds)
        result = evaluate_predictions(merged)

        assert result.global_.n == 3
        assert 0.0 <= result.global_.average_precision <= 1.0
        assert len(result.by_bin) == 3
        assert result.test_df is merged
        assert result.pipeline is None

    def test_global_metrics_present(self, test_df, preds_csv):
        preds = load_predictions(preds_csv)
        merged = merge_predictions(test_df, preds)
        result = evaluate_predictions(merged)
        g = result.global_
        assert hasattr(g, "average_precision")
        assert hasattr(g, "bedroc")
        assert hasattr(g, "n")

    def test_bins_have_actives(self, test_df, preds_csv):
        preds = load_predictions(preds_csv)
        merged = merge_predictions(test_df, preds)
        result = evaluate_predictions(merged)
        for b in result.by_bin:
            assert hasattr(b, "n_actives")
