"""Тесты для puzzle_reconstruction.matching.score_aggregator."""
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


# ─── TestAggregationConfig ────────────────────────────────────────────────────

class TestAggregationConfig:
    def test_defaults(self):
        cfg = AggregationConfig()
        assert cfg.strategy == "mean"
        assert cfg.weights is None
        assert cfg.clip_min == 0.0
        assert cfg.clip_max == 1.0
        assert cfg.normalize is False

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            AggregationConfig(strategy="unknown")

    def test_negative_clip_min_raises(self):
        with pytest.raises(ValueError):
            AggregationConfig(clip_min=-0.1)

    def test_clip_max_le_clip_min_raises(self):
        with pytest.raises(ValueError):
            AggregationConfig(clip_min=0.5, clip_max=0.5)

    def test_clip_max_lt_clip_min_raises(self):
        with pytest.raises(ValueError):
            AggregationConfig(clip_min=0.8, clip_max=0.2)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            AggregationConfig(strategy="weighted", weights=[-0.1, 0.5])

    def test_empty_weights_raises(self):
        with pytest.raises(ValueError):
            AggregationConfig(strategy="weighted", weights=[])

    def test_valid_strategies(self):
        for s in ("weighted", "mean", "max", "min", "median", "product", "rank"):
            cfg = AggregationConfig(strategy=s)
            assert cfg.strategy == s


# ─── TestAggregatedScore ──────────────────────────────────────────────────────

class TestAggregatedScore:
    def _make(self, score=0.7, sources=None):
        return AggregatedScore(
            pair=(0, 1),
            score=score,
            sources=sources if sources is not None else [0.6, 0.8],
            strategy="mean",
        )

    def test_basic_construction(self):
        s = self._make()
        assert s.pair == (0, 1)
        assert s.score == 0.7

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            AggregatedScore(pair=(0, 1), score=-0.1, sources=[], strategy="mean")

    def test_empty_strategy_raises(self):
        with pytest.raises(ValueError):
            AggregatedScore(pair=(0, 1), score=0.5, sources=[], strategy="")

    def test_n_sources(self):
        s = self._make(sources=[0.1, 0.2, 0.3])
        assert s.n_sources == 3

    def test_n_sources_empty(self):
        s = self._make(sources=[])
        assert s.n_sources == 0

    def test_source_max(self):
        s = self._make(sources=[0.1, 0.9, 0.5])
        assert abs(s.source_max - 0.9) < 1e-9

    def test_source_min(self):
        s = self._make(sources=[0.1, 0.9, 0.5])
        assert abs(s.source_min - 0.1) < 1e-9

    def test_source_max_empty(self):
        s = self._make(sources=[])
        assert s.source_max == 0.0

    def test_source_min_empty(self):
        s = self._make(sources=[])
        assert s.source_min == 0.0


# ─── TestAggregationReport ────────────────────────────────────────────────────

class TestAggregationReport:
    def _make_scores(self):
        return [
            AggregatedScore(pair=(0, 1), score=0.9, sources=[0.9], strategy="mean"),
            AggregatedScore(pair=(1, 2), score=0.3, sources=[0.3], strategy="mean"),
        ]

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            AggregationReport(scores=[], n_pairs=-1, strategy="mean", mean_score=0.0)

    def test_negative_mean_score_raises(self):
        with pytest.raises(ValueError):
            AggregationReport(scores=[], n_pairs=0, strategy="mean", mean_score=-0.1)

    def test_best_pair_empty(self):
        r = AggregationReport(scores=[], n_pairs=0, strategy="mean", mean_score=0.0)
        assert r.best_pair is None

    def test_best_score_empty(self):
        r = AggregationReport(scores=[], n_pairs=0, strategy="mean", mean_score=0.0)
        assert r.best_score == 0.0

    def test_best_pair_correct(self):
        scores = self._make_scores()
        r = AggregationReport(scores=scores, n_pairs=2, strategy="mean", mean_score=0.6)
        assert r.best_pair == (0, 1)

    def test_best_score_correct(self):
        scores = self._make_scores()
        r = AggregationReport(scores=scores, n_pairs=2, strategy="mean", mean_score=0.6)
        assert abs(r.best_score - 0.9) < 1e-9


# ─── TestAggregateScores ──────────────────────────────────────────────────────

class TestAggregateScores:
    def test_returns_aggregated_score(self):
        r = aggregate_scores([0.5, 0.7, 0.9])
        assert isinstance(r, AggregatedScore)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_scores([])

    def test_mean_strategy(self):
        cfg = AggregationConfig(strategy="mean")
        r = aggregate_scores([0.4, 0.6], cfg)
        assert abs(r.score - 0.5) < 1e-9

    def test_max_strategy(self):
        cfg = AggregationConfig(strategy="max")
        r = aggregate_scores([0.3, 0.8, 0.5], cfg)
        assert abs(r.score - 0.8) < 1e-9

    def test_min_strategy(self):
        cfg = AggregationConfig(strategy="min")
        r = aggregate_scores([0.3, 0.8, 0.5], cfg)
        assert abs(r.score - 0.3) < 1e-9

    def test_median_strategy(self):
        cfg = AggregationConfig(strategy="median")
        r = aggregate_scores([0.1, 0.5, 0.9], cfg)
        assert abs(r.score - 0.5) < 1e-9

    def test_product_strategy(self):
        cfg = AggregationConfig(strategy="product")
        r = aggregate_scores([0.5, 0.4], cfg)
        assert abs(r.score - 0.2) < 1e-9

    def test_weighted_strategy_basic(self):
        cfg = AggregationConfig(strategy="weighted", weights=[1.0, 3.0])
        r = aggregate_scores([0.0, 1.0], cfg)
        assert abs(r.score - 0.75) < 1e-9

    def test_weighted_length_mismatch_raises(self):
        cfg = AggregationConfig(strategy="weighted", weights=[0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            aggregate_scores([0.4, 0.6], cfg)

    def test_rank_strategy_returns_valid(self):
        cfg = AggregationConfig(strategy="rank")
        r = aggregate_scores([0.1, 0.5, 0.9], cfg)
        assert 0.0 <= r.score <= 1.0

    def test_clip_min_applied(self):
        cfg = AggregationConfig(strategy="product", clip_min=0.3)
        r = aggregate_scores([0.1, 0.1], cfg)
        assert r.score >= 0.3

    def test_clip_max_applied(self):
        cfg = AggregationConfig(strategy="mean", clip_max=0.5)
        r = aggregate_scores([0.8, 0.9], cfg)
        assert r.score <= 0.5

    def test_pair_stored(self):
        r = aggregate_scores([0.5], pair=(3, 7))
        assert r.pair == (3, 7)

    def test_sources_stored(self):
        r = aggregate_scores([0.4, 0.6, 0.8])
        assert r.sources == [0.4, 0.6, 0.8]

    def test_strategy_stored(self):
        cfg = AggregationConfig(strategy="max")
        r = aggregate_scores([0.5], cfg)
        assert r.strategy == "max"


# ─── TestAggregateScoreMatrix ─────────────────────────────────────────────────

class TestAggregateScoreMatrix:
    def _matrices(self, n=3, shape=(4, 4)):
        rng = np.random.default_rng(0)
        return [rng.uniform(0, 1, shape) for _ in range(n)]

    def test_returns_ndarray(self):
        result = aggregate_score_matrix(self._matrices())
        assert isinstance(result, np.ndarray)

    def test_output_shape_preserved(self):
        mats = self._matrices(shape=(5, 6))
        result = aggregate_score_matrix(mats)
        assert result.shape == (5, 6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_score_matrix([])

    def test_shape_mismatch_raises(self):
        m1 = np.ones((3, 3))
        m2 = np.ones((4, 4))
        with pytest.raises(ValueError):
            aggregate_score_matrix([m1, m2])

    def test_mean_strategy(self):
        m1 = np.full((2, 2), 0.4)
        m2 = np.full((2, 2), 0.8)
        cfg = AggregationConfig(strategy="mean")
        result = aggregate_score_matrix([m1, m2], cfg)
        assert np.allclose(result, 0.6)

    def test_max_strategy(self):
        m1 = np.full((2, 2), 0.3)
        m2 = np.full((2, 2), 0.9)
        cfg = AggregationConfig(strategy="max")
        result = aggregate_score_matrix([m1, m2], cfg)
        assert np.allclose(result, 0.9)

    def test_normalize_flag(self):
        mats = self._matrices()
        cfg = AggregationConfig(normalize=True)
        result = aggregate_score_matrix(mats, cfg)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_clip_applied(self):
        mats = [np.full((2, 2), 0.8)]
        cfg = AggregationConfig(clip_max=0.5)
        result = aggregate_score_matrix(mats, cfg)
        assert np.all(result <= 0.5)

    def test_weighted_length_mismatch_raises(self):
        mats = self._matrices(n=2)
        cfg = AggregationConfig(strategy="weighted", weights=[0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            aggregate_score_matrix(mats, cfg)


# ─── TestBatchAggregateScores ─────────────────────────────────────────────────

class TestBatchAggregateScores:
    def test_returns_report(self):
        pairs = [(0, 1), (1, 2)]
        sources = [[0.5, 0.7], [0.3, 0.9]]
        r = batch_aggregate_scores(pairs, sources)
        assert isinstance(r, AggregationReport)

    def test_n_pairs_correct(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        sources = [[0.5], [0.6], [0.7]]
        r = batch_aggregate_scores(pairs, sources)
        assert r.n_pairs == 3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_aggregate_scores([(0, 1)], [[0.5], [0.6]])

    def test_empty_returns_empty_report(self):
        r = batch_aggregate_scores([], [])
        assert r.n_pairs == 0
        assert r.mean_score == 0.0

    def test_mean_score_computed(self):
        pairs = [(0, 1), (1, 2)]
        sources = [[0.4], [0.6]]
        r = batch_aggregate_scores(pairs, sources)
        assert abs(r.mean_score - 0.5) < 1e-9

    def test_strategy_stored(self):
        cfg = AggregationConfig(strategy="max")
        r = batch_aggregate_scores([(0, 1)], [[0.3, 0.8]], cfg)
        assert r.strategy == "max"


# ─── TestFilterAggregated ─────────────────────────────────────────────────────

class TestFilterAggregated:
    def _report(self):
        scores = [
            AggregatedScore(pair=(0, 1), score=0.9, sources=[], strategy="mean"),
            AggregatedScore(pair=(1, 2), score=0.5, sources=[], strategy="mean"),
            AggregatedScore(pair=(2, 3), score=0.2, sources=[], strategy="mean"),
        ]
        return AggregationReport(scores=scores, n_pairs=3,
                                 strategy="mean", mean_score=0.53)

    def test_filters_below_threshold(self):
        result = filter_aggregated(self._report(), 0.5)
        assert all(s.score >= 0.5 for s in result)

    def test_exact_threshold_included(self):
        result = filter_aggregated(self._report(), 0.9)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_zero_threshold_returns_all(self):
        result = filter_aggregated(self._report(), 0.0)
        assert len(result) == 3

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_aggregated(self._report(), -0.1)

    def test_high_threshold_empty(self):
        result = filter_aggregated(self._report(), 1.0)
        assert len(result) == 0

    def test_returns_list(self):
        result = filter_aggregated(self._report(), 0.0)
        assert isinstance(result, list)
