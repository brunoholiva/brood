"""Smoke tests for the dataclasses in src/types.py."""

import pandas as pd

from src.types import (
    BinMetrics,
    BroodResult,
    ColumnConfig,
    EvalConfig,
    RankingMetrics,
    SplitConfig,
)


class TestConfigs:
    """Config dataclasses instantiate with defaults."""

    def test_split_config_defaults(self):
        cfg = SplitConfig()
        assert cfg.train_size == 0.8
        assert cfg.threshold == 0.65

    def test_eval_config_defaults(self):
        cfg = EvalConfig()
        assert cfg.bedrock_alpha == 20.0
        assert cfg.distance_bins is None

    def test_column_config_defaults(self):
        cfg = ColumnConfig()
        assert cfg.smiles_col == "standardized_smiles"
        assert cfg.score_col == "predicted_probability"


class TestMetrics:
    """Metrics dataclasses accept expected fields."""

    def test_ranking_metrics(self):
        m = RankingMetrics(n=100, average_precision=0.5, bedroc=0.3)
        assert m.n == 100
        assert m.average_precision == 0.5
        assert m.bedroc == 0.3

    def test_bin_metrics(self):
        m = BinMetrics(
            bin="near", n=50, n_actives=5, average_precision=0.4, bedroc=0.2
        )
        assert m.bin == "near"
        assert m.n == 50
        assert m.n_actives == 5
        assert m.average_precision == 0.4


class TestBroodResult:
    """BroodResult is the top-level result container."""

    def test_minimal_instantiation(self):
        g = RankingMetrics(n=10, average_precision=0.5, bedroc=0.3)
        b = BinMetrics(
            bin="far", n=10, n_actives=2, average_precision=0.5, bedroc=0.3
        )
        df = pd.DataFrame({"a": [1]})
        result = BroodResult(global_=g, by_bin=[b], test_df=df, pipeline=None)
        assert result.global_.n == 10
        assert len(result.by_bin) == 1
        assert result.pipeline is None
