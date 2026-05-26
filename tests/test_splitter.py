"""Tests for the Taylor-Butina OOD splitter."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from src.splitter import butina_cluster, taylor_butina_split

DIVERSE_SMILES = [
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


@pytest.fixture
def diverse_df():
    """DataFrame of 10 SMILES with binary targets."""
    return pd.DataFrame(
        {
            "standardized_smiles": DIVERSE_SMILES,
            "target": [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        }
    )


class TestReturnType:
    """Verify the function returns correct types."""

    def test_returns_dataframes(self, diverse_df):
        """Returns two pandas DataFrames."""
        train_df, test_df = taylor_butina_split(diverse_df)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_tuple_of_two(self, diverse_df):
        """Returns a 2-tuple."""
        result = taylor_butina_split(diverse_df)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestColumns:
    """Verify input/output column contracts."""

    def test_test_set_has_distance_to_train(self, diverse_df):
        """Test set gains the distance_to_train column."""
        _, test_df = taylor_butina_split(diverse_df)
        assert "distance_to_train" in test_df.columns

    def test_train_set_no_distance_column(self, diverse_df):
        """Train set does not have distance_to_train column.

        (But it does have cluster_id, which is now included in both
        train and test outputs.)
        """
        train_df, _ = taylor_butina_split(diverse_df)
        assert "distance_to_train" not in train_df.columns
        assert "cluster_id" in train_df.columns

    def test_test_set_has_all_original_columns(self, diverse_df):
        """Test set retains all original columns."""
        _, test_df = taylor_butina_split(diverse_df)
        for col in {"standardized_smiles", "target"}:
            assert col in test_df.columns


class TestDistanceValues:
    """Verify distance_to_train values are plausible."""

    def test_distance_is_float(self, diverse_df):
        """Column dtype is floating-point."""
        _, test_df = taylor_butina_split(diverse_df)
        assert test_df["distance_to_train"].dtype in (
            np.float64,
            np.float32,
            float,
        )

    def test_distance_in_unit_interval(self, diverse_df):
        """All distances are between 0 and 1 inclusive."""
        _, test_df = taylor_butina_split(diverse_df)
        d = test_df["distance_to_train"]
        assert (d >= 0.0).all()
        assert (d <= 1.0).all()

    def test_distance_requires_multiple_train_mols(self, diverse_df):
        """With enough training molecules, 5-NN should use exactly 5."""
        train_df, test_df = taylor_butina_split(diverse_df)
        assert len(train_df) >= 5
        if len(test_df) > 0:
            k = 5
            generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
            train_fps = [
                generator.GetFingerprint(Chem.MolFromSmiles(s))
                for s in train_df["standardized_smiles"]
            ]

            test_fp = generator.GetFingerprint(
                Chem.MolFromSmiles(test_df["standardized_smiles"].iloc[0])
            )
            sims = BulkTanimotoSimilarity(test_fp, train_fps)
            distances = sorted([1.0 - s for s in sims])
            expected = np.mean(distances[:k])
            actual = test_df["distance_to_train"].iloc[0]
            assert actual == pytest.approx(expected)


class TestDataIntegrity:
    """Verify data is correctly partitioned and preserved."""

    def test_no_smiles_overlap(self, diverse_df):
        """Train and test sets have no shared molecules."""
        train_df, test_df = taylor_butina_split(diverse_df)
        train_smiles = set(train_df["standardized_smiles"])
        test_smiles = set(test_df["standardized_smiles"])
        assert train_smiles.isdisjoint(test_smiles)

    def test_target_values_preserved(self, diverse_df):
        """Target values match the original DataFrame for each molecule."""
        train_df, test_df = taylor_butina_split(diverse_df)
        original = diverse_df.set_index("standardized_smiles")
        for subset_df in (train_df, test_df):
            for _, row in subset_df.iterrows():
                expected = original.loc[row["standardized_smiles"], "target"]
                assert row["target"] == expected

    def test_total_count_preserved(self, diverse_df):
        """Total (train + test) does not exceed original size."""
        train_df, test_df = taylor_butina_split(diverse_df)
        total = len(train_df) + len(test_df)
        assert total <= len(diverse_df)

    def test_without_removed_molecules(self, diverse_df):
        """No molecules are lost from the split (train + test <= original)."""
        train_df, test_df = taylor_butina_split(diverse_df)
        combined = pd.concat([train_df, test_df], ignore_index=True)
        assert len(combined) <= len(diverse_df)


class TestDeterminism:
    """Verify repeated calls give the same result."""

    def test_deterministic_split(self, diverse_df):
        """Same input produces identical train/test splits."""
        result1 = taylor_butina_split(diverse_df)
        result2 = taylor_butina_split(diverse_df)

        train1, test1 = result1
        train2, test2 = result2

        assert list(train1["standardized_smiles"]) == list(
            train2["standardized_smiles"]
        )
        assert list(train1["target"]) == list(train2["target"])
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
            taylor_butina_split(df)

    def test_missing_target_column(self):
        """Raises KeyError when 'target' column is absent."""
        df = pd.DataFrame(
            {"standardized_smiles": ["CCO", "CCN"], "wrong": [0, 1]}
        )
        with pytest.raises(KeyError):
            taylor_butina_split(df)

    def test_empty_dataframe(self):
        """Raises ValueError on an empty DataFrame."""
        df = pd.DataFrame({"standardized_smiles": [], "target": []})
        with pytest.raises(ValueError):
            taylor_butina_split(df)


SIMILAR_SMILES = [
    "c1ccccc1",
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1O",
]


class TestButinaCluster:
    """Tests for the standalone butina_cluster() function."""

    def test_returns_integer_array(self):
        """Returns numpy array of integer cluster IDs."""
        cluster_ids = butina_cluster(SIMILAR_SMILES)
        assert isinstance(cluster_ids, np.ndarray)
        assert cluster_ids.dtype.kind in ("i", "u")
        assert len(cluster_ids) == 4

    def test_similar_molecules_share_cluster(self):
        """Similar molecules (benzene derivatives) go to same cluster.

        Uses a loose threshold so similar compounds cluster together.
        """
        cluster_ids = butina_cluster(SIMILAR_SMILES, threshold=0.8)
        assert len(set(cluster_ids)) == 1

    def test_diverse_molecules_form_multiple_clusters(self):
        """Diverse set of molecules form multiple clusters."""
        cluster_ids = butina_cluster(DIVERSE_SMILES, threshold=0.3)
        assert len(set(cluster_ids)) > 1

    def test_deterministic_clustering(self):
        """Same SMILES produce same cluster IDs."""
        c1 = butina_cluster(DIVERSE_SMILES)
        c2 = butina_cluster(DIVERSE_SMILES)
        np.testing.assert_array_equal(c1, c2)

    def test_invalid_smiles_raises(self):
        """Invalid SMILES raises ValueError."""
        with pytest.raises(ValueError):
            butina_cluster(["CCO", "invalid_smiles_123"])

    def test_empty_list_returns_empty(self):
        """Empty input returns empty array."""
        result = butina_cluster([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_single_molecule_single_cluster(self):
        """One molecule returns cluster ID 0."""
        result = butina_cluster(["CCO"])
        assert list(result) == [0]


class TestClusterIdIntegrity:
    """Test cluster_id column behavior in split outputs."""

    def test_train_has_cluster_id(self, diverse_df):
        """Train DataFrame has cluster_id column."""
        train_df, _ = taylor_butina_split(diverse_df)
        assert "cluster_id" in train_df.columns

    def test_test_has_cluster_id(self, diverse_df):
        """Test DataFrame has cluster_id column."""
        _, test_df = taylor_butina_split(diverse_df)
        assert "cluster_id" in test_df.columns

    def test_no_cluster_overlap_between_train_test(self, diverse_df):
        """No cluster appears in both train and test sets.

        This is critical for OOD evaluation: molecules from the same
        scaffold cluster should not appear in both training and test.
        """
        train_df, test_df = taylor_butina_split(diverse_df)
        train_clusters = set(train_df["cluster_id"])
        test_clusters = set(test_df["cluster_id"])
        assert train_clusters.isdisjoint(test_clusters)

    def test_cluster_ids_preserved(self, diverse_df):
        """Molecules keep their cluster IDs in split outputs."""
        full_smiles = diverse_df["standardized_smiles"].tolist()
        full_cluster_ids = butina_cluster(full_smiles)
        smiles_to_cluster = dict(zip(full_smiles, full_cluster_ids))

        train_df, test_df = taylor_butina_split(diverse_df)

        for _, row in train_df.iterrows():
            assert (
                smiles_to_cluster[row["standardized_smiles"]]
                == row["cluster_id"]
            )

        for _, row in test_df.iterrows():
            assert (
                smiles_to_cluster[row["standardized_smiles"]]
                == row["cluster_id"]
            )
