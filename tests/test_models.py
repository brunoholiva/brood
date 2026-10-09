"""Tests for fixed-representation screening learners."""

import numpy as np

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


def test_logreg_build_ignores_other_family_defaults():
    learner = learner_by_name("logreg")
    assert learner is not None
    backend = get_model("logreg")
    pipe = learner.build(backend.default_params())
    pipe.fit(np.eye(4), np.array([0, 1, 0, 1]))
    assert pipe.predict_proba(np.eye(4)[:2])[:, 1].shape == (2,)


def test_rf_build_uses_fixed_family_parameters():
    learner = learner_by_name("rf")
    assert learner is not None
    backend = get_model("rf")
    pipe = learner.build(backend.default_params())
    assert pipe.get_params()["rf__n_estimators"] == 500
