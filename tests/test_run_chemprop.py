"""Tests for brood.models.chemprop.

Requires chemprop conda environment (torch, lightning, chemprop).
Skipped automatically in other environments.
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("requires torch (chemprop env)", allow_module_level=True)


from brood.models.chemprop import (  # noqa: E402
    BATCH_SIZE,
    NUM_WORKERS,
    _build_model,
    _default_params,
)


class TestThroughputSettings:
    def test_batch_size_is_the_tuned_budget_constant(self):
        assert BATCH_SIZE == 256

    def test_dataloaders_use_workers(self):
        assert NUM_WORKERS > 0


class TestDefaultParams:
    def test_returns_dict(self):
        params = _default_params()
        assert isinstance(params, dict)

    def test_contains_expected_keys(self):
        params = _default_params()
        expected = {
            "depth",
            "message_hidden_dim",
            "dropout",
            "ffn_hidden_dim",
            "ffn_num_layers",
            "max_lr",
            "batch_size",
            "epochs",
        }
        assert expected.issubset(params.keys())

    def test_default_values(self):
        params = _default_params()
        assert params["depth"] == 3
        assert params["batch_size"] == BATCH_SIZE
        assert params["epochs"] == 30
        assert params["dropout"] == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("chemprop") is None,
    reason="requires chemprop (chemprop env)",
)
class TestModelBuilding:
    def test_pos_weight_is_wired_to_predictor_criterion(self):
        model = _build_model(pos_weight=2.0, depth=1)
        assert model.predictor.criterion.pos_weight == 2.0
