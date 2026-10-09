"""Tests for the Polaris adapter."""

import numpy as np

from brood.polaris import (
    benchmark_id,
    benchmark_inputs,
    evaluate_predictions,
    prediction_arguments,
)


def test_benchmark_id_uses_polaris_slug_format():
    assert benchmark_id("pgp_broccatelli") == "tdcommons/pgp-broccatelli"


def test_score_metrics_use_probabilities():
    predictions = np.array([0.2, 0.8])

    assert np.array_equal(
        prediction_arguments("roc_auc", predictions)["y_prob"], predictions
    )
    assert np.array_equal(
        prediction_arguments("pr_auc", predictions)["y_prob"], predictions
    )
    assert np.array_equal(
        prediction_arguments("mean_absolute_error", predictions)["y_pred"],
        predictions,
    )


def test_benchmark_inputs_never_reads_test_targets():
    class Subset:
        inputs = np.array(["CCO"])
        targets = np.array([1])

    class HiddenTest(Subset):
        @property
        def targets(self):
            raise AssertionError("test targets must remain hidden")

    class Benchmark:
        def get_train_test_split(self):
            return Subset(), HiddenTest()

    train, targets, test = benchmark_inputs(Benchmark())

    assert list(train) == ["CCO"]
    assert list(targets) == [1]
    assert list(test) == ["CCO"]


def test_evaluation_does_not_upload():
    class Results:
        def upload_to_hub(self, **kwargs):
            raise AssertionError("evaluation must not upload")

    class Benchmark:
        def evaluate(self, **kwargs):
            assert set(kwargs) == {"y_prob"}
            return Results()

    assert (
        evaluate_predictions(
            Benchmark(), np.array([0.2, 0.8]), "roc_auc"
        ).__class__
        is Results
    )
