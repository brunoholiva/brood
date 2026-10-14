"""Tests for the scaffold-based OOD splitter."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from brood.data.splitter import (
    assign_cv_folds,
    scaffold_split,
    split_train_test,
    validate_cv_folds,
)

DIVERSE_SMILES = [
    "C1CC1",
    "C1CC1C",
    "C1CCC1",
    "C1CCC1C",
    "C1CCCC1",
    "C1CCCC1C",
    "C1CCCCC1",
    "C1CCCCC1C",
    "C1CCCCCC1",
    "C1CCCCCC1C",
    "C1CCCCC2CCCCC12",
    "C1CCCCC2CCCCC1C2",
]


@pytest.fixture
def diverse_df():
    """DataFrame of 12 molecules spanning 6 scaffolds with balanced targets."""
    return pd.DataFrame(
        {
            "standardized_smiles": DIVERSE_SMILES,
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        }
    )


def _expected_5nn_distance(test_smi, train_smiles):
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = [
        generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles
    ]
    test_fp = generator.GetFingerprint(Chem.MolFromSmiles(test_smi))
    sims = BulkTanimotoSimilarity(test_fp, train_fps)
    distances = sorted(1.0 - s for s in sims)
    return float(np.mean(distances[:5]))


class TestReturnType:
    """Verify the function returns correct types."""

    def test_returns_dataframes(self, diverse_df):
        """Returns two pandas DataFrames."""
        train_df, test_df = scaffold_split(diverse_df)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_tuple_of_two(self, diverse_df):
        """Returns a 2-tuple."""
        result = scaffold_split(diverse_df)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestColumns:
    """Verify input/output column contracts."""

    def test_test_set_has_distance_to_train(self, diverse_df):
        """Test set gains the distance_to_train column."""
        _, test_df = scaffold_split(diverse_df)
        assert "distance_to_train" in test_df.columns

    def test_train_set_no_distance_column(self, diverse_df):
        """Train set does not have distance_to_train column."""
        train_df, _ = scaffold_split(diverse_df)
        assert "distance_to_train" not in train_df.columns

    def test_both_sets_have_scaffold(self, diverse_df):
        """Both train and test sets have the scaffold column."""
        train_df, test_df = scaffold_split(diverse_df)
        assert "scaffold" in train_df.columns
        assert "scaffold" in test_df.columns

    def test_test_set_has_all_original_columns(self, diverse_df):
        """Test set retains all original columns."""
        _, test_df = scaffold_split(diverse_df)
        for col in {"standardized_smiles", "target"}:
            assert col in test_df.columns


class TestDistanceValues:
    """Verify distance_to_train values are plausible."""

    def test_distance_is_float(self, diverse_df):
        """Column dtype is floating-point."""
        _, test_df = scaffold_split(diverse_df)
        assert test_df["distance_to_train"].dtype in (
            np.float64,
            np.float32,
            float,
        )

    def test_distance_in_unit_interval(self, diverse_df):
        """All distances are between 0 and 1 inclusive."""
        _, test_df = scaffold_split(diverse_df)
        d = test_df["distance_to_train"]
        assert (d >= 0.0).all()
        assert (d <= 1.0).all()

    def test_distance_matches_manual_5nn(self, diverse_df):
        """distance_to_train equals the mean 5-NN Tanimoto distance."""
        train_df, test_df = scaffold_split(diverse_df)
        assert len(train_df) >= 5
        if len(test_df) > 0:
            test_smi = test_df["standardized_smiles"].iloc[0]
            actual = test_df["distance_to_train"].iloc[0]
            expected = _expected_5nn_distance(
                test_smi, train_df["standardized_smiles"].tolist()
            )
            assert actual == pytest.approx(expected)


class TestDataIntegrity:
    """Verify data is correctly split and preserved."""

    def test_no_smiles_overlap(self, diverse_df):
        """Train and test sets have no shared molecules."""
        train_df, test_df = scaffold_split(diverse_df)
        train_smiles = set(train_df["standardized_smiles"])
        test_smiles = set(test_df["standardized_smiles"])
        assert train_smiles.isdisjoint(test_smiles)

    def test_target_values_preserved(self, diverse_df):
        """Target values match the original DataFrame for each molecule."""
        train_df, test_df = scaffold_split(diverse_df)
        original = diverse_df.set_index("standardized_smiles")
        for subset_df in (train_df, test_df):
            for _, row in subset_df.iterrows():
                expected = original.loc[row["standardized_smiles"], "target"]
                assert row["target"] == expected

    def test_total_count_preserved(self, diverse_df):
        """Train + test equals the original size."""
        train_df, test_df = scaffold_split(diverse_df)
        total = len(train_df) + len(test_df)
        assert total == len(diverse_df)

    def test_duplicate_smiles_do_not_inflate(self, diverse_df):
        """Duplicate SMILES must not blow up the test set via the merge."""
        dup_df = pd.concat(
            [diverse_df, diverse_df.iloc[[0]].copy()], ignore_index=True
        )
        train_df, test_df = scaffold_split(dup_df)
        total = len(train_df) + len(test_df)
        assert total == len(dup_df)

    def test_no_scaffold_overlap(self, diverse_df):
        """No scaffold appears in both train and test.

        This is critical for OOD evaluation: molecules sharing a generic
        Murcko scaffold should never span the train/test boundary.
        """
        train_df, test_df = scaffold_split(diverse_df)
        train_scaffolds = set(train_df["scaffold"])
        test_scaffolds = set(test_df["scaffold"])
        assert train_scaffolds.isdisjoint(test_scaffolds)


class TestDeterminism:
    """Verify repeated calls give the same result."""

    def test_deterministic_split(self, diverse_df):
        """Same input produces identical train/test splits."""
        result1 = scaffold_split(diverse_df)
        result2 = scaffold_split(diverse_df)

        train1, test1 = result1
        train2, test2 = result2

        assert list(train1["standardized_smiles"]) == list(
            train2["standardized_smiles"]
        )
        assert list(test1["standardized_smiles"]) == list(
            test2["standardized_smiles"]
        )
        assert list(test1["distance_to_train"]) == pytest.approx(
            list(test2["distance_to_train"])
        )


class TestInputValidation:
    """Verify invalid inputs raise appropriate errors."""

    def test_missing_smiles_column(self):
        """Raises KeyError when "standardized_smiles" column is absent."""
        df = pd.DataFrame({"wrong": ["CCO", "CCN"], "target": [0, 1]})
        with pytest.raises(KeyError):
            scaffold_split(df)

    def test_missing_target_column(self):
        """Raises KeyError when 'target' column is absent."""
        df = pd.DataFrame(
            {"standardized_smiles": ["CCO", "CCN"], "wrong": [0, 1]}
        )
        with pytest.raises(KeyError):
            scaffold_split(df)

    def test_empty_dataframe(self):
        """Raises ValueError on an empty DataFrame."""
        df = pd.DataFrame({"standardized_smiles": [], "target": []})
        with pytest.raises(ValueError):
            scaffold_split(df)


class TestSplitTrainTest:
    """Scaffold-based tuning splits used for early stopping."""

    def test_returns_two_dataframes(self, diverse_df):
        """split_train_test returns train/validation splits."""
        train_df, test_df = split_train_test(diverse_df)
        assert set(train_df["scaffold"]).isdisjoint(test_df["scaffold"])
        assert len(train_df) + len(test_df) == len(diverse_df)


class TestAssignCvFolds:
    """Repeated CV fold assignment for cross-model comparison."""

    def test_adds_repeat_and_fold_columns(self, diverse_df):
        """assign_cv_folds adds repeat and integer fold columns."""
        cv_df = assign_cv_folds(diverse_df, n_splits=3, n_repeats=2)
        assert "fold" in cv_df.columns
        assert "repeat" in cv_df.columns
        assert set(cv_df["fold"]) == {0, 1, 2}
        assert set(cv_df["repeat"]) == {0, 1}

    def test_repeats_cover_all_rows(self, diverse_df):
        """Every molecule appears once per repeat."""
        cv_df = assign_cv_folds(diverse_df, n_splits=3, n_repeats=4)
        assert len(cv_df) == 4 * len(diverse_df)

    def test_molecule_index_is_positional(self, diverse_df):
        """Each repeat carries a 0..N-1 positional molecule index."""
        cv_df = assign_cv_folds(diverse_df, n_splits=3, n_repeats=2)
        for _, frame in cv_df.groupby("repeat"):
            assert list(frame["molecule_index"]) == list(
                range(len(diverse_df))
            )

    def test_folds_scaffold_disjoint_within_repeat(self, diverse_df):
        """Each scaffold appears in exactly one fold per repeat."""
        cv_df = assign_cv_folds(diverse_df, n_splits=3, n_repeats=3)
        scaffold_folds = cv_df.groupby(["scaffold", "repeat"])[
            "fold"
        ].nunique()
        assert (scaffold_folds == 1).all()

    def test_deterministic(self, diverse_df):
        """Same input produces identical fold assignments."""
        cv1 = assign_cv_folds(diverse_df, n_splits=3, n_repeats=2)
        cv2 = assign_cv_folds(diverse_df, n_splits=3, n_repeats=2)
        assert list(cv1["fold"]) == list(cv2["fold"])
        assert list(cv1["repeat"]) == list(cv2["repeat"])

    def test_reports_fold_balance_and_scaffold_isolation(self, diverse_df):
        """CV validation summarizes the required grouped-CV invariants."""
        checks = validate_cv_folds(
            assign_cv_folds(diverse_df, n_splits=3, n_repeats=2)
        )
        assert checks["folds"] == 3
        assert checks["repeats"] == 2
        assert checks["scaffold_overlap"] == 0
        assert set(checks["active_fraction_by_fold"]) == {0, 1, 2}
