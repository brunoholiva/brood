"""Tests for fixed-representation screening learners."""

import numpy as np
import pandas as pd

from brood.models.knn import _nearest_neighbors, _score_neighbors, _tune
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

    preds = backend._predict_frames(
        train_df, test_df, {"n_neighbors": 3, "weights": "uniform"}
    )

    assert len(preds) == 1
    assert 0 <= preds.iloc[0]["predicted_probability"] <= 1


def test_knn_distance_scores_stay_within_probability_bounds():
    scores = _score_neighbors(
        np.array([1, 1, 1]),
        np.array([[0, 1, 2]]),
        np.array([[0.99999994, 0.9999999, 0.5]], dtype=np.float32),
        3,
        "distance",
    )

    assert np.all((scores >= 0) & (scores <= 1))


def test_knn_neighbor_cache_is_reused(tmp_path):
    query = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
    candidates = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    query_ids = np.array([10, 11])
    candidate_ids = np.array([20, 21, 22])
    cache = tmp_path / "neighbors.npz"

    indices, similarities = _nearest_neighbors(
        query, candidates, query_ids, candidate_ids, cache
    )
    cached_indices, cached_similarities = _nearest_neighbors(
        np.zeros_like(query),
        np.zeros_like(candidates),
        query_ids,
        candidate_ids,
        cache,
    )

    np.testing.assert_array_equal(indices, cached_indices)
    np.testing.assert_array_equal(similarities, cached_similarities)


def test_knn_tuning_uses_cached_neighbors(tmp_path):
    train_df = pd.DataFrame(
        {
            "target": [0, 1] * 5,
            "scaffold": [f"s{i}" for i in range(10)],
        }
    )
    features = np.eye(10, 4, dtype=bool)
    labels = train_df["target"].to_numpy()

    params = _tune(train_df, features, labels, tmp_path, "demo", 1, 42)

    assert params["n_neighbors"] in (1, 3, 5)
    assert params["weights"] in ("uniform", "distance")
    assert len(list((tmp_path / "features" / "demo").rglob("*.npz"))) == 5


def test_rf_build_uses_fixed_model_parameters():
    learner = learner_by_name("rf")
    assert learner is not None
    backend = get_model("rf")
    pipe = learner.build(backend.default_params())
    assert pipe.get_params()["rf__n_estimators"] == 500
