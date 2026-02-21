"""Тесты для puzzle_reconstruction.matching.score_combiner."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.score_combiner import (
    ScoreVector,
    CombinedScore,
    weighted_combine,
    min_combine,
    max_combine,
    rank_combine,
    normalize_score_vectors,
    batch_combine,
)


# ─── TestScoreVector ──────────────────────────────────────────────────────────

class TestScoreVector:
    def test_basic_creation(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5, "b": 0.8})
        assert sv.idx1 == 0
        assert sv.idx2 == 1
        assert len(sv) == 2

    def test_pair_property(self):
        sv = ScoreVector(idx1=2, idx2=5, scores={"x": 0.3})
        assert sv.pair == (2, 5)

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=-1, idx2=0, scores={"a": 0.5})

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=-1, scores={"a": 0.5})

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": -0.1})

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": 1.1})

    def test_score_boundary_values(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0, "b": 1.0})
        assert sv.scores["a"] == 0.0
        assert sv.scores["b"] == 1.0

    def test_empty_scores(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        assert len(sv) == 0

    def test_len(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.1, "b": 0.2, "c": 0.3})
        assert len(sv) == 3


# ─── TestCombinedScore ────────────────────────────────────────────────────────

class TestCombinedScore:
    def test_basic_creation(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.7)
        assert cs.score == pytest.approx(0.7)

    def test_pair_property(self):
        cs = CombinedScore(idx1=3, idx2=7, score=0.5)
        assert cs.pair == (3, 7)

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=-1, idx2=0, score=0.5)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=-1, score=0.5)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=-0.01)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=1.01)

    def test_score_boundary_values(self):
        cs0 = CombinedScore(idx1=0, idx2=1, score=0.0)
        cs1 = CombinedScore(idx1=0, idx2=1, score=1.0)
        assert cs0.score == 0.0
        assert cs1.score == 1.0

    def test_contributions_stored(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.5, contributions={"a": 0.3})
        assert "a" in cs.contributions


# ─── TestWeightedCombine ──────────────────────────────────────────────────────

class TestWeightedCombine:
    def test_equal_weights(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.4, "b": 0.6})
        cs = weighted_combine(sv)
        assert cs.score == pytest.approx(0.5, abs=1e-6)

    def test_custom_weights(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0, "b": 1.0})
        cs = weighted_combine(sv, weights={"a": 0.0, "b": 1.0})
        assert cs.score == pytest.approx(1.0, abs=1e-6)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 0.75})
        cs = weighted_combine(sv)
        assert cs.score == pytest.approx(0.75, abs=1e-6)

    def test_empty_scores_raises(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        with pytest.raises(ValueError):
            weighted_combine(sv)

    def test_negative_weight_raises(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5})
        with pytest.raises(ValueError):
            weighted_combine(sv, weights={"a": -1.0})

    def test_score_clipped_to_range(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0})
        cs = weighted_combine(sv)
        assert 0.0 <= cs.score <= 1.0

    def test_contributions_present(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5, "b": 0.5})
        cs = weighted_combine(sv)
        assert set(cs.contributions.keys()) == {"a", "b"}

    def test_idx_preserved(self):
        sv = ScoreVector(idx1=3, idx2=7, scores={"a": 0.5})
        cs = weighted_combine(sv)
        assert cs.idx1 == 3 and cs.idx2 == 7


# ─── TestMinCombine ───────────────────────────────────────────────────────────

class TestMinCombine:
    def test_returns_minimum(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.8, "c": 0.5})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.3)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 0.6})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.6)

    def test_empty_scores_raises(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        with pytest.raises(ValueError):
            min_combine(sv)

    def test_all_equal(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5, "b": 0.5})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.5)

    def test_idx_preserved(self):
        sv = ScoreVector(idx1=4, idx2=9, scores={"a": 0.2})
        cs = min_combine(sv)
        assert cs.idx1 == 4 and cs.idx2 == 9


# ─── TestMaxCombine ───────────────────────────────────────────────────────────

class TestMaxCombine:
    def test_returns_maximum(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.8, "c": 0.5})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.8)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 0.4})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.4)

    def test_empty_scores_raises(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        with pytest.raises(ValueError):
            max_combine(sv)

    def test_all_equal(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.7, "b": 0.7})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.7)

    def test_idx_preserved(self):
        sv = ScoreVector(idx1=2, idx2=6, scores={"a": 0.9})
        cs = max_combine(sv)
        assert cs.idx1 == 2 and cs.idx2 == 6


# ─── TestRankCombine ──────────────────────────────────────────────────────────

class TestRankCombine:
    def _make_svs(self):
        return [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.9, "b": 0.8}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.5, "b": 0.5}),
            ScoreVector(idx1=0, idx2=3, scores={"a": 0.1, "b": 0.2}),
        ]

    def test_returns_list(self):
        svs = self._make_svs()
        result = rank_combine(svs)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_best_ranked_highest(self):
        svs = self._make_svs()
        result = rank_combine(svs)
        scores = {(r.idx1, r.idx2): r.score for r in result}
        assert scores[(0, 1)] > scores[(0, 2)] > scores[(0, 3)]

    def test_scores_in_range(self):
        svs = self._make_svs()
        for cs in rank_combine(svs):
            assert 0.0 <= cs.score <= 1.0

    def test_empty_list(self):
        assert rank_combine([]) == []

    def test_single_item_score_one(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5})
        result = rank_combine([sv])
        assert result[0].score == pytest.approx(1.0)

    def test_mismatched_keys_raises(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.5}),
            ScoreVector(idx1=0, idx2=2, scores={"b": 0.5}),
        ]
        with pytest.raises(ValueError):
            rank_combine(svs)


# ─── TestNormalizeScoreVectors ────────────────────────────────────────────────

class TestNormalizeScoreVectors:
    def test_returns_list_same_length(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.2}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.8}),
        ]
        result = normalize_score_vectors(svs)
        assert len(result) == 2

    def test_scores_in_01(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.1, "b": 0.9}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.5, "b": 0.3}),
            ScoreVector(idx1=0, idx2=3, scores={"a": 0.9, "b": 0.6}),
        ]
        result = normalize_score_vectors(svs)
        for sv in result:
            for v in sv.scores.values():
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_min_becomes_zero(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.2}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.8}),
        ]
        result = normalize_score_vectors(svs)
        scores = [sv.scores["a"] for sv in result]
        assert min(scores) == pytest.approx(0.0)

    def test_max_becomes_one(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.2}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.8}),
        ]
        result = normalize_score_vectors(svs)
        scores = [sv.scores["a"] for sv in result]
        assert max(scores) == pytest.approx(1.0)

    def test_all_equal_becomes_zero(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.5}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.5}),
        ]
        result = normalize_score_vectors(svs)
        for sv in result:
            assert sv.scores["a"] == pytest.approx(0.0)

    def test_empty_list(self):
        assert normalize_score_vectors([]) == []

    def test_idx_preserved(self):
        svs = [ScoreVector(idx1=2, idx2=5, scores={"a": 0.4})]
        result = normalize_score_vectors(svs)
        assert result[0].idx1 == 2 and result[0].idx2 == 5


# ─── TestBatchCombine ─────────────────────────────────────────────────────────

class TestBatchCombine:
    def _make_svs(self):
        return [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.9, "b": 0.8}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.3, "b": 0.4}),
            ScoreVector(idx1=0, idx2=3, scores={"a": 0.6, "b": 0.7}),
        ]

    def test_weighted_sorted_descending(self):
        svs = self._make_svs()
        result = batch_combine(svs, method="weighted")
        scores = [cs.score for cs in result]
        assert scores == sorted(scores, reverse=True)

    def test_min_method(self):
        svs = self._make_svs()
        result = batch_combine(svs, method="min")
        assert all(isinstance(cs, CombinedScore) for cs in result)

    def test_max_method(self):
        svs = self._make_svs()
        result = batch_combine(svs, method="max")
        assert all(isinstance(cs, CombinedScore) for cs in result)

    def test_rank_method(self):
        svs = self._make_svs()
        result = batch_combine(svs, method="rank")
        assert len(result) == 3
        scores = [cs.score for cs in result]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_method_raises(self):
        svs = self._make_svs()
        with pytest.raises(ValueError):
            batch_combine(svs, method="unknown")

    def test_empty_list(self):
        result = batch_combine([], method="weighted")
        assert result == []

    def test_length_preserved(self):
        svs = self._make_svs()
        result = batch_combine(svs, method="weighted")
        assert len(result) == len(svs)
