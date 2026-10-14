"""Tests for fixed-representation screening learners."""

import pandas as pd

from brood.models.registry import get_model
from brood.models.sklearn import learner_by_name


def test_screening_run_names_are_stable():
    backend = get_model("rf")
    assert backend.run_name("morgan2048") == "rf_morgan2048"
    assert backend.run_name("morgan512") == "rf_morgan512"


def test_run_names_are_unique_across_the_representation_stage():
    names = {
        get_model(learner).run_name(rep)
        for learner in ("rf", "tabpfn")
        for rep in ("morgan2048", "morgan512", "maccs", "rdkit2d")
    }
    assert len(names) == 8


def test_knn_has_small_tuning_space():
    backend = get_model("knn")
    assert backend.default_params() == {
        "n_neighbors": 5,
        "weights": "uniform",
    }


def test_knn_predicts_probabilities_from_both_classes():
    backend = get_model("knn")
    train_df = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN", "CCC", "c1ccccc1"],
            "target": [1, 0, 0, 1],
            "scaffold": ["a", "b", "c", "d"],
        }
    )
    test_df = pd.DataFrame({"standardized_smiles": ["CCCO"]})

    preds = backend.predict_fold(
        "demo",
        None,
        train_df,
        test_df,
        None,
        {"n_neighbors": 3, "weights": "uniform"},
    )

    assert len(preds) == 1
    assert 0 <= preds.iloc[0]["predicted_probability"] <= 1


def test_rf_build_uses_fixed_model_parameters():
    learner = learner_by_name("rf")
    assert learner is not None
    backend = get_model("rf")
    pipe = learner.build(backend.default_params())
    assert pipe.get_params()["rf__n_estimators"] == 500
