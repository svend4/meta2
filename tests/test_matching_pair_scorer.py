"""Tests for puzzle_reconstruction/matching/pair_scorer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.pair_scorer import (
    ScoringWeights,
    PairScoreResult,
    aggregate_channels,
    score_pair,
    select_top_pairs,
    build_score_matrix,
    batch_score_pairs,
)


# ─── ScoringWeights ──────────────────────────────────────────────────────────

class TestScoringWeights:
    def test_default_construction(self):
        sw = ScoringWeights()
        assert sw.color == 1.0
        assert sw.texture == 1.0
        assert sw.geometry == 1.0
        assert sw.gradient == 1.0

    def test_total(self):
        sw = ScoringWeights(color=1.0, texture=2.0, geometry=0.5, gradient=0.5)
        assert abs(sw.total - 4.0) < 1e-9

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScoringWeights(color=-0.1)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            ScoringWeights(color=0.0, texture=0.0, geometry=0.0, gradient=0.0)

    def test_some_zero_allowed(self):
        sw = ScoringWeights(color=1.0, texture=0.0, geometry=0.0, gradient=0.0)
        assert sw.total == 1.0

    def test_as_dict(self):
        sw = ScoringWeights(color=2.0, texture=1.0, geometry=0.5, gradient=0.5)
        d = sw.as_dict()
        assert set(d.keys()) == {"color", "texture", "geometry", "gradient"}
        assert d["color"] == 2.0
        assert d["texture"] == 1.0

    def test_normalized(self):
        sw = ScoringWeights(color=2.0, texture=2.0, geometry=2.0, gradient=2.0)
        norm = sw.normalized()
        assert abs(norm.total - 1.0) < 1e-9
        assert abs(norm.color - 0.25) < 1e-9

    def test_normalized_uneven(self):
        sw = ScoringWeights(color=3.0, texture=1.0, geometry=0.0, gradient=0.0)
        norm = sw.normalized()
        assert abs(norm.total - 1.0) < 1e-9
        assert abs(norm.color - 0.75) < 1e-9
        assert abs(norm.texture - 0.25) < 1e-9


# ─── PairScoreResult ─────────────────────────────────────────────────────────

class TestPairScoreResult:
    def test_basic_creation(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.8)
        assert r.idx_a == 0
        assert r.idx_b == 1
        assert r.score == 0.8

    def test_score_zero(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.0)
        assert r.score == 0.0

    def test_score_one(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=1.0)
        assert r.score == 1.0

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            PairScoreResult(idx_a=0, idx_b=1, score=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            PairScoreResult(idx_a=0, idx_b=1, score=1.1)

    def test_n_channels_empty(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.5)
        assert r.n_channels == 0

    def test_n_channels_with_data(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.5,
                            channels={"color": 0.6, "texture": 0.4})
        assert r.n_channels == 2

    def test_pair_key_ordered(self):
        r = PairScoreResult(idx_a=5, idx_b=2, score=0.5)
        assert r.pair_key == (2, 5)

    def test_pair_key_same_order(self):
        r = PairScoreResult(idx_a=2, idx_b=7, score=0.5)
        assert r.pair_key == (2, 7)

    def test_dominant_channel_none_when_empty(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.5)
        assert r.dominant_channel is None

    def test_dominant_channel(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.5,
                            channels={"color": 0.3, "texture": 0.9, "geometry": 0.5})
        assert r.dominant_channel == "texture"

    def test_is_strong_match_true(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.7)
        assert r.is_strong_match is True

    def test_is_strong_match_false(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.69)
        assert r.is_strong_match is False

    def test_is_strong_match_boundary(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.7)
        assert r.is_strong_match is True


# ─── aggregate_channels ──────────────────────────────────────────────────────

class TestAggregateChannels:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_channels({})

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            aggregate_channels({"color": 1.1})

    def test_below_zero_raises(self):
        with pytest.raises(ValueError):
            aggregate_channels({"color": -0.1})

    def test_single_channel(self):
        score = aggregate_channels({"color": 0.8})
        assert abs(score - 0.8) < 1e-6

    def test_equal_weights_default(self):
        # With equal default weights, should be simple average
        score = aggregate_channels({"color": 0.6, "texture": 0.4})
        # weight = 1.0 for each (default ScoringWeights all 1.0)
        # (0.6*1 + 0.4*1) / (1+1) = 0.5
        assert abs(score - 0.5) < 1e-6

    def test_weighted_aggregate(self):
        weights = ScoringWeights(color=2.0, texture=1.0, geometry=1.0, gradient=1.0)
        score = aggregate_channels({"color": 1.0, "texture": 0.0}, weights=weights)
        # (1.0*2 + 0.0*1) / (2+1) = 2/3
        expected = 2.0 / 3.0
        assert abs(score - expected) < 1e-6

    def test_unknown_channel_gets_weight_one(self):
        weights = ScoringWeights(color=2.0, texture=2.0, geometry=2.0, gradient=2.0)
        # "unknown" channel not in weights → gets weight 1.0
        score = aggregate_channels({"unknown": 0.5}, weights=weights)
        assert abs(score - 0.5) < 1e-6

    def test_result_clipped_to_range(self):
        score = aggregate_channels({"color": 1.0})
        assert 0.0 <= score <= 1.0

    def test_all_zero_scores(self):
        score = aggregate_channels({"color": 0.0, "texture": 0.0})
        assert score == 0.0

    def test_all_one_scores(self):
        score = aggregate_channels({"color": 1.0, "texture": 1.0})
        assert abs(score - 1.0) < 1e-6


# ─── score_pair ──────────────────────────────────────────────────────────────

class TestScorePair:
    def test_basic(self):
        result = score_pair(0, 1, {"color": 0.8, "texture": 0.6})
        assert isinstance(result, PairScoreResult)
        assert result.idx_a == 0
        assert result.idx_b == 1
        assert 0.0 <= result.score <= 1.0

    def test_channels_preserved(self):
        cs = {"color": 0.7, "geometry": 0.5}
        result = score_pair(2, 3, cs)
        assert result.channels == cs

    def test_with_weights(self):
        weights = ScoringWeights(color=1.0, texture=0.0, geometry=0.0, gradient=0.0)
        result = score_pair(0, 1, {"color": 0.9}, weights=weights)
        assert abs(result.score - 0.9) < 1e-6

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            score_pair(0, 1, {})


# ─── select_top_pairs ────────────────────────────────────────────────────────

class TestSelectTopPairs:
    def _make_results(self):
        return [
            PairScoreResult(0, 1, 0.9, {"color": 0.9}),
            PairScoreResult(1, 2, 0.5, {"color": 0.5}),
            PairScoreResult(2, 3, 0.7, {"color": 0.7}),
            PairScoreResult(3, 4, 0.3, {"color": 0.3}),
        ]

    def test_no_filter(self):
        results = self._make_results()
        top = select_top_pairs(results)
        assert len(top) == 4
        # Sorted descending
        scores = [r.score for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_threshold(self):
        results = self._make_results()
        top = select_top_pairs(results, threshold=0.6)
        assert all(r.score >= 0.6 for r in top)
        assert len(top) == 2  # 0.9 and 0.7

    def test_top_k(self):
        results = self._make_results()
        top = select_top_pairs(results, top_k=2)
        assert len(top) == 2
        assert top[0].score == 0.9
        assert top[1].score == 0.7

    def test_threshold_and_top_k(self):
        results = self._make_results()
        top = select_top_pairs(results, threshold=0.5, top_k=1)
        assert len(top) == 1
        assert top[0].score == 0.9

    def test_threshold_zero_top_k_zero_no_limit(self):
        results = self._make_results()
        top = select_top_pairs(results, threshold=0.0, top_k=0)
        assert len(top) == 4

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            select_top_pairs([], threshold=-0.1)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            select_top_pairs([], top_k=-1)

    def test_empty_results(self):
        top = select_top_pairs([])
        assert top == []

    def test_sorted_descending(self):
        results = self._make_results()
        top = select_top_pairs(results)
        scores = [r.score for r in top]
        assert scores[0] >= scores[1] >= scores[2] >= scores[3]


# ─── build_score_matrix ──────────────────────────────────────────────────────

class TestBuildScoreMatrix:
    def test_basic_shape(self):
        results = [PairScoreResult(0, 1, 0.8)]
        mat = build_score_matrix(results, n_fragments=3)
        assert mat.shape == (3, 3)

    def test_symmetric(self):
        results = [PairScoreResult(0, 1, 0.8)]
        mat = build_score_matrix(results, n_fragments=3)
        assert mat[0, 1] == mat[1, 0]

    def test_values_set_correctly(self):
        results = [
            PairScoreResult(0, 1, 0.8),
            PairScoreResult(1, 2, 0.6),
        ]
        mat = build_score_matrix(results, n_fragments=3)
        assert abs(mat[0, 1] - 0.8) < 1e-6
        assert abs(mat[1, 0] - 0.8) < 1e-6
        assert abs(mat[1, 2] - 0.6) < 1e-6
        assert abs(mat[2, 1] - 0.6) < 1e-6

    def test_missing_pairs_are_zero(self):
        results = [PairScoreResult(0, 1, 0.8)]
        mat = build_score_matrix(results, n_fragments=3)
        assert mat[0, 2] == 0.0
        assert mat[2, 0] == 0.0

    def test_dtype_float32(self):
        results = [PairScoreResult(0, 1, 0.5)]
        mat = build_score_matrix(results, n_fragments=2)
        assert mat.dtype == np.float32

    def test_n_fragments_one(self):
        mat = build_score_matrix([], n_fragments=1)
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            build_score_matrix([], n_fragments=0)

    def test_out_of_bounds_indices_ignored(self):
        results = [PairScoreResult(0, 10, 0.9)]  # idx 10 out of range
        mat = build_score_matrix(results, n_fragments=3)
        # No crash, value not set
        assert mat.max() == 0.0


# ─── batch_score_pairs ───────────────────────────────────────────────────────

class TestBatchScorePairs:
    def test_basic(self):
        pairs = [(0, 1), (1, 2)]
        css_list = [{"color": 0.8}, {"color": 0.5}]
        results = batch_score_pairs(pairs, css_list)
        assert len(results) == 2
        assert all(isinstance(r, PairScoreResult) for r in results)

    def test_empty(self):
        results = batch_score_pairs([], [])
        assert results == []

    def test_len_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_score_pairs([(0, 1)], [{}, {}])

    def test_indices_preserved(self):
        pairs = [(3, 7)]
        css_list = [{"color": 0.6}]
        results = batch_score_pairs(pairs, css_list)
        assert results[0].idx_a == 3
        assert results[0].idx_b == 7

    def test_with_weights(self):
        pairs = [(0, 1)]
        css_list = [{"color": 1.0}]
        weights = ScoringWeights(color=1.0, texture=0.0, geometry=0.0, gradient=0.0)
        results = batch_score_pairs(pairs, css_list, weights=weights)
        assert abs(results[0].score - 1.0) < 1e-6
