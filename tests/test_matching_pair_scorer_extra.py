"""Extra tests for puzzle_reconstruction.matching.pair_scorer."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.pair_scorer import (
    ScoringWeights,
    PairScoreResult,
    aggregate_channels,
    score_pair,
    select_top_pairs,
    build_score_matrix,
    batch_score_pairs,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _result(ia=0, ib=1, score=0.5, **channels):
    ch = channels or {"color": score}
    return PairScoreResult(idx_a=ia, idx_b=ib, score=score, channels=ch)


def _results_n(n=5):
    return [_result(ia=i, ib=i + 1, score=round(i / n, 3)) for i in range(n)]


# ─── TestScoringWeightsExtra ─────────────────────────────────────────────────

class TestScoringWeightsExtra:
    def test_all_equal_default_total(self):
        sw = ScoringWeights()
        assert sw.total == pytest.approx(4.0)

    def test_as_dict_keys(self):
        sw = ScoringWeights()
        d = sw.as_dict()
        assert "color" in d and "texture" in d

    def test_normalized_total_one(self):
        sw = ScoringWeights(color=3.0, texture=1.0, geometry=0.0, gradient=0.0)
        n = sw.normalized()
        assert n.total == pytest.approx(1.0)

    def test_normalized_values(self):
        sw = ScoringWeights(color=2.0, texture=2.0, geometry=0.0, gradient=0.0)
        n = sw.normalized()
        assert n.color == pytest.approx(0.5)

    def test_single_nonzero_weight(self):
        sw = ScoringWeights(color=5.0, texture=0.0, geometry=0.0, gradient=0.0)
        assert sw.total == pytest.approx(5.0)

    def test_all_large_values(self):
        sw = ScoringWeights(color=100.0, texture=100.0, geometry=100.0, gradient=100.0)
        assert sw.total == pytest.approx(400.0)

    def test_as_dict_values(self):
        sw = ScoringWeights(color=2.0, texture=3.0, geometry=0.0, gradient=1.0)
        d = sw.as_dict()
        assert d["texture"] == pytest.approx(3.0)

    def test_single_negative_raises(self):
        with pytest.raises(ValueError):
            ScoringWeights(texture=-0.01)


# ─── TestPairScoreResultExtra ────────────────────────────────────────────────

class TestPairScoreResultExtra:
    def test_score_midpoint(self):
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.5)
        assert r.score == pytest.approx(0.5)

    def test_channels_stored(self):
        ch = {"color": 0.6, "texture": 0.4}
        r = PairScoreResult(idx_a=0, idx_b=1, score=0.5, channels=ch)
        assert r.channels["color"] == pytest.approx(0.6)

    def test_n_channels_three(self):
        r = PairScoreResult(0, 1, 0.5, {"a": 0.5, "b": 0.5, "c": 0.5})
        assert r.n_channels == 3

    def test_pair_key_sorted(self):
        r = PairScoreResult(idx_a=9, idx_b=3, score=0.5)
        assert r.pair_key == (3, 9)

    def test_dominant_channel_max(self):
        r = PairScoreResult(0, 1, 0.5, {"color": 0.9, "texture": 0.1})
        assert r.dominant_channel == "color"

    def test_is_strong_match_exactly_07(self):
        r = PairScoreResult(0, 1, 0.7)
        assert r.is_strong_match is True

    def test_is_strong_match_just_below(self):
        r = PairScoreResult(0, 1, 0.699)
        assert r.is_strong_match is False

    def test_score_zero_valid(self):
        r = PairScoreResult(0, 1, 0.0)
        assert r.score == 0.0

    def test_score_one_valid(self):
        r = PairScoreResult(0, 1, 1.0)
        assert r.score == 1.0


# ─── TestAggregateChannelsExtra ──────────────────────────────────────────────

class TestAggregateChannelsExtra:
    def test_single_channel_returned(self):
        score = aggregate_channels({"texture": 0.6})
        assert score == pytest.approx(0.6)

    def test_equal_channels_average(self):
        score = aggregate_channels({"color": 0.4, "texture": 0.8})
        # default weights both 1.0 → (0.4 + 0.8) / 2 = 0.6
        assert score == pytest.approx(0.6, abs=1e-6)

    def test_three_channels_average(self):
        score = aggregate_channels({"a": 0.0, "b": 0.5, "c": 1.0})
        # default weight 1.0 for unknowns → (0 + 0.5 + 1.0) / 3
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_custom_weights_asymmetric(self):
        w = ScoringWeights(color=3.0, texture=1.0, geometry=0.0, gradient=0.0)
        score = aggregate_channels({"color": 0.0, "texture": 1.0}, weights=w)
        # (0*3 + 1*1)/(3+1) = 0.25
        assert score == pytest.approx(0.25, abs=1e-6)

    def test_result_in_range(self):
        score = aggregate_channels({"color": 0.3, "texture": 0.7})
        assert 0.0 <= score <= 1.0

    def test_all_ones(self):
        score = aggregate_channels({"color": 1.0, "texture": 1.0, "geometry": 1.0})
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_all_zeros(self):
        score = aggregate_channels({"color": 0.0, "texture": 0.0})
        assert score == 0.0


# ─── TestScorePairExtra ──────────────────────────────────────────────────────

class TestScorePairExtra:
    def test_returns_pair_score_result(self):
        r = score_pair(0, 1, {"color": 0.5})
        assert isinstance(r, PairScoreResult)

    def test_indices_stored(self):
        r = score_pair(3, 7, {"color": 0.5})
        assert r.idx_a == 3 and r.idx_b == 7

    def test_channels_preserved(self):
        cs = {"color": 0.9, "texture": 0.3}
        r = score_pair(0, 1, cs)
        assert r.channels == cs

    def test_score_in_range(self):
        r = score_pair(0, 1, {"color": 0.8, "texture": 0.4})
        assert 0.0 <= r.score <= 1.0

    def test_color_only_weight_matches(self):
        w = ScoringWeights(color=1.0, texture=0.0, geometry=0.0, gradient=0.0)
        r = score_pair(0, 1, {"color": 0.77}, weights=w)
        assert r.score == pytest.approx(0.77, abs=1e-6)

    def test_texture_only_weight_matches(self):
        w = ScoringWeights(color=0.0, texture=1.0, geometry=0.0, gradient=0.0)
        r = score_pair(0, 1, {"texture": 0.55}, weights=w)
        assert r.score == pytest.approx(0.55, abs=1e-6)


# ─── TestSelectTopPairsExtra ─────────────────────────────────────────────────

class TestSelectTopPairsExtra:
    def test_top_1_returns_best(self):
        results = _results_n(5)
        top = select_top_pairs(results, top_k=1)
        assert len(top) == 1
        assert top[0].score == max(r.score for r in results)

    def test_top_k_larger_than_count(self):
        results = _results_n(3)
        top = select_top_pairs(results, top_k=10)
        assert len(top) == 3

    def test_threshold_all_pass(self):
        results = _results_n(4)
        top = select_top_pairs(results, threshold=0.0)
        assert len(top) == 4

    def test_threshold_none_pass(self):
        results = _results_n(4)
        top = select_top_pairs(results, threshold=1.1)
        assert len(top) == 0

    def test_sorted_descending(self):
        results = _results_n(6)
        top = select_top_pairs(results)
        scores = [r.score for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        assert select_top_pairs([]) == []

    def test_single_result_returned(self):
        results = [_result(score=0.9)]
        top = select_top_pairs(results)
        assert len(top) == 1


# ─── TestBuildScoreMatrixExtra ───────────────────────────────────────────────

class TestBuildScoreMatrixExtra:
    def test_shape_large(self):
        results = [PairScoreResult(i, i + 1, 0.5) for i in range(5)]
        mat = build_score_matrix(results, n_fragments=7)
        assert mat.shape == (7, 7)

    def test_diagonal_zero(self):
        results = [PairScoreResult(0, 1, 0.8)]
        mat = build_score_matrix(results, n_fragments=3)
        assert mat[0, 0] == 0.0
        assert mat[1, 1] == 0.0

    def test_symmetric_multiple(self):
        results = [PairScoreResult(0, 2, 0.7), PairScoreResult(1, 3, 0.5)]
        mat = build_score_matrix(results, n_fragments=4)
        assert mat[0, 2] == pytest.approx(mat[2, 0])
        assert mat[1, 3] == pytest.approx(mat[3, 1])

    def test_all_pairs_set(self):
        n = 4
        results = [PairScoreResult(i, j, 0.5)
                   for i in range(n) for j in range(i + 1, n)]
        mat = build_score_matrix(results, n_fragments=n)
        assert mat.sum() > 0

    def test_dtype(self):
        results = [PairScoreResult(0, 1, 0.5)]
        mat = build_score_matrix(results, n_fragments=2)
        assert mat.dtype == np.float32

    def test_empty_results_all_zero(self):
        mat = build_score_matrix([], n_fragments=3)
        assert mat.max() == 0.0


# ─── TestBatchScorePairsExtra ────────────────────────────────────────────────

class TestBatchScorePairsExtra:
    def test_single_pair(self):
        results = batch_score_pairs([(0, 1)], [{"color": 0.7}])
        assert len(results) == 1
        assert isinstance(results[0], PairScoreResult)

    def test_score_in_range(self):
        results = batch_score_pairs([(0, 1)], [{"color": 0.6, "texture": 0.4}])
        assert 0.0 <= results[0].score <= 1.0

    def test_three_pairs(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        css = [{"color": 0.8}, {"texture": 0.5}, {"geometry": 0.3}]
        results = batch_score_pairs(pairs, css)
        assert len(results) == 3

    def test_indices_match(self):
        pairs = [(5, 9)]
        results = batch_score_pairs(pairs, [{"color": 0.5}])
        assert results[0].idx_a == 5 and results[0].idx_b == 9

    def test_channels_preserved(self):
        cs = {"color": 0.9}
        results = batch_score_pairs([(0, 1)], [cs])
        assert results[0].channels == cs

    def test_custom_weights(self):
        w = ScoringWeights(color=1.0, texture=0.0, geometry=0.0, gradient=0.0)
        results = batch_score_pairs([(0, 1)], [{"color": 0.88}], weights=w)
        assert results[0].score == pytest.approx(0.88, abs=1e-5)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            batch_score_pairs([(0, 1), (1, 2)], [{"color": 0.5}])
