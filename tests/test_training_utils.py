"""Tests for brood shared training utilities."""

import pandas as pd
import pytest

from brood.data.loader import (
    MissingScaffoldColumnError,
    MissingSplitFileError,
    load_split_data,
    save_predictions,
)


class TestLoadSplitData:
    @pytest.fixture
    def train_csv(self, tmp_path):
        path = tmp_path / "train.csv"
        pd.DataFrame(
            {
                "standardized_smiles": ["CCO", "CCN"],
                "target": [0, 1],
                "scaffold": ["CCO", "CCN"],
            }
        ).to_csv(path, index=False)
        return path

    @pytest.fixture
    def test_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        pd.DataFrame(
            {
                "standardized_smiles": ["CCC"],
                "target": [0],
                "distance_to_train": [0.5],
            }
        ).to_csv(path, index=False)
        return path

    def test_loads_data(self, train_csv, test_csv):
        train_df, test_df = load_split_data(train_csv, test_csv, "test_ds")
        assert len(train_df) == 2
        assert len(test_df) == 1
        assert "scaffold" in train_df.columns

    def test_raises_on_missing_train(self, test_csv, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(MissingSplitFileError):
            load_split_data(missing, test_csv, "test_ds")

    def test_raises_on_missing_test(self, train_csv, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(MissingSplitFileError):
            load_split_data(train_csv, missing, "test_ds")

    def test_raises_on_missing_scaffold(self, train_csv, test_csv):
        df = pd.read_csv(train_csv)
        df.drop(columns=["scaffold"], inplace=True)
        df.to_csv(train_csv, index=False)
        with pytest.raises(MissingScaffoldColumnError):
            load_split_data(
                train_csv, test_csv, "test_ds", require_scaffold=True
            )

    def test_skips_scaffold_warning_if_not_required(self, train_csv, test_csv):
        df_train = pd.read_csv(train_csv)
        df_train.drop(columns=["scaffold"], inplace=True)
        df_train.to_csv(train_csv, index=False)
        train_df, test_df = load_split_data(
            train_csv, test_csv, "test_ds", require_scaffold=False
        )
        assert "scaffold" not in train_df.columns


class TestSavePredictions:
    def test_saves_csv(self, tmp_path):
        preds = pd.DataFrame(
            {
                "standardized_smiles": ["CCO"],
                "predicted_probability": [0.5],
            }
        )
        out_path = save_predictions(preds, tmp_path, "test.csv")
        assert out_path == tmp_path / "test.csv"
        loaded = pd.read_csv(out_path)
        assert list(loaded.columns) == [
            "standardized_smiles",
            "predicted_probability",
        ]
        assert loaded.iloc[0]["predicted_probability"] == 0.5

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        preds = pd.DataFrame(
            {
                "standardized_smiles": ["CCO"],
                "predicted_probability": [0.5],
            }
        )
        out_path = save_predictions(preds, nested, "test.csv")
        assert out_path.exists()

    def test_nrows(self, tmp_path):
        n = 5
        preds = pd.DataFrame(
            {
                "standardized_smiles": [f"C{i}" for i in range(n)],
                "predicted_probability": [float(i) / n for i in range(n)],
            }
        )
        out_path = save_predictions(preds, tmp_path, "test.csv")
        loaded = pd.read_csv(out_path)
        assert len(loaded) == n


class TestRunOptunaTuning:
    def test_returns_best_params(self):
        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return -((x - 0.3) ** 2)

        from brood.tuning.optuna import run_optuna_tuning

        best = run_optuna_tuning(
            objective,
            dataset="test_ds",
            model_name="test_model",
            n_trials=5,
            random_state=42,
        )
        assert "x" in best
        assert 0.0 <= best["x"] <= 1.0

    def test_saves_params_json(self, tmp_path):
        def objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return -((x - 0.3) ** 2)

        import json

        from brood.tuning.optuna import run_optuna_tuning

        run_optuna_tuning(
            objective,
            dataset="test_ds",
            model_name="test_model",
            n_trials=3,
            random_state=42,
            params_path=tmp_path,
        )
        out = tmp_path / "test_model_params.json"
        assert out.exists()
        params = json.loads(out.read_text())
        assert "x" in params
