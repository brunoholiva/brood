"""Smoke tests for the dataclasses in brood.core.types."""

import pandas as pd

from brood.core.types import BinMetrics, BroodResult, RankingMetrics


class TestMetrics:
    """Metrics dataclasses accept expected fields."""

    def test_ranking_metrics(self):
        m = RankingMetrics(n=100, average_precision=0.5, bedroc=0.3, brier=0.1)
        assert m.n == 100
        assert m.average_precision == 0.5
        assert m.bedroc == 0.3
        assert m.brier == 0.1

    def test_bin_metrics(self):
        m = BinMetrics(
            bin="near",
            n=50,
            n_actives=5,
            average_precision=0.4,
            bedroc=0.2,
            brier=0.1,
        )
        assert m.bin == "near"
        assert m.n == 50
        assert m.n_actives == 5
        assert m.average_precision == 0.4


class TestBroodResult:
    """BroodResult is the top-level result container."""

    def test_minimal_instantiation(self):
        g = RankingMetrics(n=10, average_precision=0.5, bedroc=0.3, brier=0.1)
        b = BinMetrics(
            bin="far",
            n=10,
            n_actives=2,
            average_precision=0.5,
            bedroc=0.3,
            brier=0.1,
        )
        df = pd.DataFrame({"a": [1]})
        result = BroodResult(global_=g, by_bin=[b], test_df=df)
        assert result.global_.n == 10
        assert len(result.by_bin) == 1
        assert result.test_df is df
