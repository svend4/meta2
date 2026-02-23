"""Extra tests for puzzle_reconstruction.matching.score_aggregator."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.score_aggregator import (
    AggregatedScore,
    AggregationConfig,
    AggregationReport,
    aggregate_score_matrix,
    aggregate_scores,
    batch_aggregate_scores,
    filter_aggregated,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _agg_score(pair=(0, 1), score=0.5, sources=None, strategy="mean"):
    return AggregatedScore(pair=pair, score=score,
                           sources=sources or [score], strategy=strategy)


def _report(scores=None, strategy="mean"):
    sc = scores or [_agg_score(pair=(0, 1), score=0.7),
                    _agg_score(pair=(1, 2), score=0.4)]
    mean = sum(s.score for s in sc) / len(sc)
    return AggregationReport(scores=sc, n_pairs=len(sc),
                             strategy=strategy, mean_score=mean)


# ─── TestAggregationConfigExtra ──────────────────────────────────────────────

class TestAggregationConfigExtra:
    def test_strategy_mean(self):
        cfg = AggregationConfig(strategy="mean")
        assert cfg.strategy == "mean"

    def test_strategy_max(self):
        cfg = AggregationConfig(strategy="max")
        assert cfg.strategy == "max"

    def test_strategy_min(self):
        cfg = AggregationConfig(strategy="min")
        assert cfg.strategy == "min"

    def test_strategy_product(self):
        cfg = AggregationConfig(strategy="product")
        assert cfg.strategy == "product"

    def test_strategy_median(self):
        cfg = AggregationConfig(strategy="median")
        assert cfg.strategy == "median"

    def test_normalize_true(self):
        cfg = AggregationConfig(normalize=True)
        assert cfg.normalize is True

    def test_clip_min_half(self):
        cfg = AggregationConfig(clip_min=0.3, clip_max=1.0)
        assert cfg.clip_min == pytest.approx(0.3)

    def test_valid_weighted_weights(self):
        cfg = AggregationConfig(strategy="weighted", weights=[0.5, 0.5])
        assert cfg.weights == [0.5, 0.5]

    def test_rank_strategy(self):
        cfg = AggregationConfig(strategy="rank")
        assert cfg.strategy == "rank"


# ─── TestAggregatedScoreExtra ────────────────────────────────────────────────

class TestAggregatedScoreExtra:
    def test_n_sources_one(self):
        s = _agg_score(sources=[0.7])
        assert s.n_sources == 1

    def test_source_max_single(self):
        s = _agg_score(sources=[0.5])
        assert s.source_max == pytest.approx(0.5)

    def test_source_min_single(self):
        s = _agg_score(sources=[0.5])
        assert s.source_min == pytest.approx(0.5)

    def test_pair_stored(self):
        s = _agg_score(pair=(3, 7))
        assert s.pair == (3, 7)

    def test_strategy_stored(self):
        s = _agg_score(strategy="max")
        assert s.strategy == "max"

    def test_score_stored(self):
        s = _agg_score(score=0.88)
        assert s.score == pytest.approx(0.88)

    def test_multiple_sources(self):
        s = _agg_score(sources=[0.1, 0.5, 0.9])
        assert s.n_sources == 3
        assert s.source_max == pytest.approx(0.9)
        assert s.source_min == pytest.approx(0.1)


# ─── TestAggregationReportExtra ──────────────────────────────────────────────

class TestAggregationReportExtra:
    def test_n_pairs_stored(self):
        r = _report()
        assert r.n_pairs == len(r.scores)

    def test_strategy_stored(self):
        r = _report(strategy="max")
        assert r.strategy == "max"

    def test_best_pair_is_pair(self):
        r = _report()
        assert isinstance(r.best_pair, tuple)

    def test_best_score_max(self):
        r = _report()
        assert r.best_score == max(s.score for s in r.scores)

    def test_mean_score_positive(self):
        r = _report()
        assert r.mean_score > 0.0

    def test_empty_report_best_none(self):
        r = AggregationReport(scores=[], n_pairs=0, strategy="mean", mean_score=0.0)
        assert r.best_pair is None
        assert r.best_score == 0.0


# ─── TestAggregateScoresExtra ────────────────────────────────────────────────

class TestAggregateScoresExtra:
    def test_single_value_mean(self):
        r = aggregate_scores([0.7])
        assert r.score == pytest.approx(0.7)

    def test_two_values_mean(self):
        cfg = AggregationConfig(strategy="mean")
        r = aggregate_scores([0.3, 0.7], cfg)
        assert r.score == pytest.approx(0.5, abs=1e-9)

    def test_max_of_three(self):
        cfg = AggregationConfig(strategy="max")
        r = aggregate_scores([0.2, 0.8, 0.5], cfg)
        assert r.score == pytest.approx(0.8)

    def test_min_of_three(self):
        cfg = AggregationConfig(strategy="min")
        r = aggregate_scores([0.2, 0.8, 0.5], cfg)
        assert r.score == pytest.approx(0.2)

    def test_product_two(self):
        cfg = AggregationConfig(strategy="product")
        r = aggregate_scores([0.5, 0.4], cfg)
        assert r.score == pytest.approx(0.2, abs=1e-6)

    def test_median_odd(self):
        cfg = AggregationConfig(strategy="median")
        r = aggregate_scores([0.1, 0.5, 0.9], cfg)
        assert r.score == pytest.approx(0.5)

    def test_weighted_equal(self):
        cfg = AggregationConfig(strategy="weighted", weights=[1.0, 1.0])
        r = aggregate_scores([0.4, 0.8], cfg)
        assert r.score == pytest.approx(0.6, abs=1e-6)

    def test_clip_max_applied(self):
        cfg = AggregationConfig(strategy="mean", clip_max=0.6)
        r = aggregate_scores([0.8, 0.9], cfg)
        assert r.score <= 0.6

    def test_clip_min_applied(self):
        cfg = AggregationConfig(strategy="product", clip_min=0.3)
        r = aggregate_scores([0.1, 0.1], cfg)
        assert r.score >= 0.3

    def test_n_sources_stored(self):
        r = aggregate_scores([0.3, 0.7, 0.5])
        assert r.n_sources == 3


# ─── TestAggregateScoreMatrixExtra ───────────────────────────────────────────

class TestAggregateScoreMatrixExtra:
    def _mats(self, n=2, shape=(3, 3)):
        rng = np.random.default_rng(42)
        return [rng.uniform(0, 1, shape) for _ in range(n)]

    def test_single_matrix_passthrough(self):
        m = np.full((3, 3), 0.7)
        result = aggregate_score_matrix([m])
        assert np.allclose(result, 0.7)

    def test_mean_two_matrices(self):
        m1 = np.full((2, 2), 0.2)
        m2 = np.full((2, 2), 0.6)
        cfg = AggregationConfig(strategy="mean")
        result = aggregate_score_matrix([m1, m2], cfg)
        assert np.allclose(result, 0.4)

    def test_max_two_matrices(self):
        m1 = np.full((2, 2), 0.3)
        m2 = np.full((2, 2), 0.7)
        cfg = AggregationConfig(strategy="max")
        result = aggregate_score_matrix([m1, m2], cfg)
        assert np.allclose(result, 0.7)

    def test_min_two_matrices(self):
        m1 = np.full((2, 2), 0.3)
        m2 = np.full((2, 2), 0.7)
        cfg = AggregationConfig(strategy="min")
        result = aggregate_score_matrix([m1, m2], cfg)
        assert np.allclose(result, 0.3)

    def test_clip_applied(self):
        mats = [np.full((2, 2), 0.9)]
        cfg = AggregationConfig(clip_max=0.5)
        result = aggregate_score_matrix(mats, cfg)
        assert np.all(result <= 0.5)

    def test_values_in_range(self):
        result = aggregate_score_matrix(self._mats())
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-9

    def test_weighted_mismatch_raises(self):
        mats = self._mats(n=3)
        cfg = AggregationConfig(strategy="weighted", weights=[1.0, 1.0])
        with pytest.raises(ValueError):
            aggregate_score_matrix(mats, cfg)


# ─── TestBatchAggregateScoresExtra ───────────────────────────────────────────

class TestBatchAggregateScoresExtra:
    def test_returns_report(self):
        r = batch_aggregate_scores([(0, 1)], [[0.5, 0.7]])
        assert isinstance(r, AggregationReport)

    def test_n_pairs_one(self):
        r = batch_aggregate_scores([(0, 1)], [[0.6]])
        assert r.n_pairs == 1

    def test_mean_score_single(self):
        r = batch_aggregate_scores([(0, 1)], [[0.8]])
        assert r.mean_score == pytest.approx(0.8)

    def test_mean_score_two(self):
        r = batch_aggregate_scores([(0, 1), (1, 2)], [[0.4], [0.6]])
        assert r.mean_score == pytest.approx(0.5, abs=1e-9)

    def test_strategy_forwarded(self):
        cfg = AggregationConfig(strategy="max")
        r = batch_aggregate_scores([(0, 1)], [[0.3, 0.9]], cfg)
        assert r.strategy == "max"

    def test_best_pair_highest(self):
        r = batch_aggregate_scores(
            [(0, 1), (1, 2), (2, 3)],
            [[0.9], [0.3], [0.6]]
        )
        assert r.best_pair == (0, 1)

    def test_empty_report(self):
        r = batch_aggregate_scores([], [])
        assert r.n_pairs == 0
        assert r.mean_score == 0.0


# ─── TestFilterAggregatedExtra ───────────────────────────────────────────────

class TestFilterAggregatedExtra:
    def test_threshold_05_keeps_two(self):
        r = _report([_agg_score(score=0.9), _agg_score(score=0.6),
                     _agg_score(score=0.4)])
        result = filter_aggregated(r, 0.5)
        assert len(result) == 2

    def test_zero_threshold_all(self):
        r = _report()
        result = filter_aggregated(r, 0.0)
        assert len(result) == len(r.scores)

    def test_threshold_one_empty(self):
        r = _report()
        result = filter_aggregated(r, 1.0)
        assert len(result) == 0

    def test_returned_scores_above_threshold(self):
        r = _report([_agg_score(score=0.8), _agg_score(score=0.3)])
        result = filter_aggregated(r, 0.5)
        assert all(s.score >= 0.5 for s in result)

    def test_returns_list(self):
        r = _report()
        assert isinstance(filter_aggregated(r, 0.0), list)

    def test_exact_threshold_included(self):
        r = _report([_agg_score(score=0.5), _agg_score(score=0.3)])
        result = filter_aggregated(r, 0.5)
        assert any(s.score == pytest.approx(0.5) for s in result)
