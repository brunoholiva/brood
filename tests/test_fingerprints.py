"""Tests for MorganFingerprintTransformer."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.fingerprints import MorganFingerprintTransformer, _validate_mols

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


class TestTransformer:
    """MorganFingerprintTransformer stateless transform behavior."""

    def test_fit_returns_self(self):
        """Fit returns the transformer instance."""
        t = MorganFingerprintTransformer()
        assert t.fit(SMILES) is t

    def test_fit_accepts_y_none(self):
        """Fit accepts optional y=None (sklearn convention)."""
        t = MorganFingerprintTransformer()
        assert t.fit(SMILES, y=None) is t

    def test_transform_shape(self):
        """Transform returns (n, 2048) for default fp_size."""
        t = MorganFingerprintTransformer()
        fps = t.transform(SMILES)
        assert fps.shape == (len(SMILES), 2048)

    def test_transform_dtype_binary(self):
        """Transform returns binary int64 array."""
        t = MorganFingerprintTransformer()
        fps = t.transform(SMILES)
        assert fps.dtype == np.int64
        assert set(np.unique(fps)).issubset({0, 1})

    def test_invalid_smiles_raises(self):
        """Invalid SMILES raises ValueError with position."""
        t = MorganFingerprintTransformer()
        with pytest.raises(ValueError, match="Invalid SMILES at position 1"):
            t.transform(["CCO", "not_a_molecule"])

    def test_configurable_radius(self):
        """Radius and fp_size parameters affect output."""
        t = MorganFingerprintTransformer(radius=1, fp_size=1024)
        fps = t.transform(["CCO"])
        assert fps.shape == (1, 1024)

    def test_empty_input(self):
        """Empty input returns empty array with correct shape."""
        t = MorganFingerprintTransformer()
        fps = t.transform([])
        assert fps.shape == (0, 2048)

    def test_deterministic(self):
        """Same input produces identical output."""
        t = MorganFingerprintTransformer()
        fps1 = t.transform(SMILES)
        fps2 = t.transform(SMILES)
        assert np.array_equal(fps1, fps2)


class TestPipeline:
    """End-to-end: transformer in sklearn Pipeline."""

    def test_pipeline_fit_predict(self):
        """Pipeline with transformer + RF can fit and predict."""
        pipe = Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )
        targets = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
        pipe.fit(SMILES, targets)
        preds = pipe.predict(SMILES)
        assert preds.shape == (len(SMILES),)

    def test_pipeline_predict_proba(self):
        """Pipeline returns probabilities via predict_proba."""
        pipe = Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )
        targets = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
        pipe.fit(SMILES, targets)
        probs = pipe.predict_proba(SMILES)
        assert probs.shape == (len(SMILES), 2)

    def test_pipeline_with_splitter(self):
        """Full workflow: splitter → pipeline predict."""
        import pandas as pd

        from src.splitter import taylor_butina_split

        df = pd.DataFrame(
            {
                "standardized_smiles": SMILES,
                "target": [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
            }
        )
        train_df, test_df = taylor_butina_split(df, distance_cutoff=0.0)

        pipe = Pipeline(
            [
                ("fps", MorganFingerprintTransformer()),
                ("rf", RandomForestClassifier(random_state=42)),
            ]
        )
        pipe.fit(train_df["standardized_smiles"], train_df["target"])

        probs = pipe.predict_proba(test_df["standardized_smiles"])
        assert len(probs) == len(test_df)


class TestValidateMols:
    """_validate_mols helper function."""

    def test_all_valid(self):
        """Valid mols raise no error."""
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN"]]
        _validate_mols(mols)

    def test_invalid_raises(self):
        """None mol raises ValueError with index."""
        from rdkit import Chem

        mols = [Chem.MolFromSmiles("CCO"), None]
        with pytest.raises(ValueError, match="Invalid SMILES at position 1"):
            _validate_mols(mols)
