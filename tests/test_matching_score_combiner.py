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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sv(idx1=0, idx2=1, **scores) -> ScoreVector:
    if not scores:
        scores = {"color": 0.8, "texture": 0.6}
    return ScoreVector(idx1=idx1, idx2=idx2, scores=scores)


def _sv_list(n=3) -> list:
    """n ScoreVectors with ascending color scores."""
    return [
        ScoreVector(idx1=i, idx2=i + 10,
                    scores={"color": 0.2 + i * 0.2, "shape": 0.5})
        for i in range(n)
    ]


# ─── TestScoreVector ──────────────────────────────────────────────────────────

class TestScoreVector:
    def test_basic_fields(self):
        sv = _sv(idx1=2, idx2=5, color=0.7)
        assert sv.idx1 == 2
        assert sv.idx2 == 5
        assert sv.scores["color"] == pytest.approx(0.7)

    def test_pair_property(self):
        sv = _sv(idx1=3, idx2=7)
        assert sv.pair == (3, 7)

    def test_len(self):
        sv = _sv(color=0.5, texture=0.6, shape=0.7)
        assert len(sv) == 3

    def test_params_default_empty(self):
        sv = _sv()
        assert sv.params == {}

    def test_idx1_neg_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=-1, idx2=0, scores={"a": 0.5})

    def test_idx2_neg_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=-1, scores={"a": 0.5})

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": 1.1})

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": -0.1})

    def test_score_zero_ok(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0})
        assert sv.scores["a"] == 0.0

    def test_score_one_ok(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0})
        assert sv.scores["a"] == 1.0

    def test_empty_scores_ok(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        assert len(sv) == 0


# ─── TestCombinedScore ────────────────────────────────────────────────────────

class TestCombinedScore:
    def test_basic_fields(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.75)
        assert cs.score == pytest.approx(0.75)

    def test_pair_property(self):
        cs = CombinedScore(idx1=2, idx2=5, score=0.5)
        assert cs.pair == (2, 5)

    def test_idx1_neg_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=-1, idx2=0, score=0.5)

    def test_idx2_neg_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=-1, score=0.5)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=1.1)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=-0.1)

    def test_contributions_default_empty(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.5)
        assert cs.contributions == {}

    def test_contributions_stored(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.7,
                            contributions={"color": 0.5, "shape": 0.2})
        assert "color" in cs.contributions


# ─── TestWeightedCombine ──────────────────────────────────────────────────────

class TestWeightedCombine:
    def test_returns_combined_score(self):
        sv = _sv(color=0.8, texture=0.6)
        cs = weighted_combine(sv)
        assert isinstance(cs, CombinedScore)

    def test_equal_weights_average(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.4, "b": 0.8})
        cs = weighted_combine(sv)
        assert cs.score == pytest.approx(0.6, abs=1e-5)

    def test_custom_weights(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0, "b": 0.0})
        cs = weighted_combine(sv, weights={"a": 1.0, "b": 0.0})
        assert cs.score == pytest.approx(1.0, abs=1e-4)

    def test_score_in_range(self):
        sv = _sv(color=0.5, texture=0.5)
        cs = weighted_combine(sv)
        assert 0.0 <= cs.score <= 1.0

    def test_empty_scores_raises(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        with pytest.raises(ValueError):
            weighted_combine(sv)

    def test_negative_weight_raises(self):
        sv = _sv(color=0.5)
        with pytest.raises(ValueError):
            weighted_combine(sv, weights={"color": -1.0})

    def test_zero_total_weight_raises(self):
        sv = _sv(color=0.5)
        with pytest.raises(ValueError):
            weighted_combine(sv, weights={"color": 0.0})

    def test_ids_preserved(self):
        sv = ScoreVector(idx1=3, idx2=7, scores={"a": 0.5})
        cs = weighted_combine(sv)
        assert cs.idx1 == 3 and cs.idx2 == 7

    def test_contributions_stored(self):
        sv = _sv(color=0.8, texture=0.6)
        cs = weighted_combine(sv)
        assert "color" in cs.contributions
        assert "texture" in cs.contributions


# ─── TestMinMaxCombine ────────────────────────────────────────────────────────

class TestMinCombine:
    def test_returns_combined_score(self):
        cs = min_combine(_sv(color=0.8, texture=0.4))
        assert isinstance(cs, CombinedScore)

    def test_score_is_minimum(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.9, "c": 0.6})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.3)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.7})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            min_combine(ScoreVector(idx1=0, idx2=1, scores={}))


class TestMaxCombine:
    def test_returns_combined_score(self):
        cs = max_combine(_sv(color=0.8, texture=0.4))
        assert isinstance(cs, CombinedScore)

    def test_score_is_maximum(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.9, "c": 0.6})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.9)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.4})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.4)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            max_combine(ScoreVector(idx1=0, idx2=1, scores={}))

    def test_min_leq_max(self):
        sv = _sv(color=0.2, texture=0.8)
        assert min_combine(sv).score <= max_combine(sv).score


# ─── TestRankCombine ──────────────────────────────────────────────────────────

class TestRankCombine:
    def test_returns_list(self):
        result = rank_combine(_sv_list(3))
        assert isinstance(result, list)

    def test_length_matches(self):
        svs = _sv_list(4)
        assert len(rank_combine(svs)) == 4

    def test_empty_list_returns_empty(self):
        assert rank_combine([]) == []

    def test_single_item_score_one(self):
        sv = _sv(color=0.5, texture=0.5)
        result = rank_combine([sv])
        assert result[0].score == pytest.approx(1.0)

    def test_best_pair_highest_score(self):
        svs = _sv_list(3)
        result = rank_combine(svs)
        scores = [cs.score for cs in result]
        # The last sv has the highest color score → highest rank score
        best_idx = svs.index(max(svs, key=lambda sv: sv.scores["color"]))
        assert result[best_idx].score == max(scores)

    def test_scores_in_range(self):
        for cs in rank_combine(_sv_list(5)):
            assert 0.0 <= cs.score <= 1.0

    def test_mismatched_keys_raises(self):
        sv1 = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5})
        sv2 = ScoreVector(idx1=1, idx2=2, scores={"b": 0.5})
        with pytest.raises(ValueError):
            rank_combine([sv1, sv2])

    def test_all_combined_scores(self):
        for cs in rank_combine(_sv_list(3)):
            assert isinstance(cs, CombinedScore)


# ─── TestNormalizeScoreVectors ────────────────────────────────────────────────

class TestNormalizeScoreVectors:
    def test_returns_list(self):
        result = normalize_score_vectors(_sv_list(3))
        assert isinstance(result, list)

    def test_empty_list(self):
        assert normalize_score_vectors([]) == []

    def test_length_preserved(self):
        svs = _sv_list(4)
        assert len(normalize_score_vectors(svs)) == 4

    def test_all_score_vectors(self):
        for sv in normalize_score_vectors(_sv_list(3)):
            assert isinstance(sv, ScoreVector)

    def test_min_max_normalized(self):
        svs = [
            ScoreVector(idx1=i, idx2=i + 1, scores={"a": 0.2 + i * 0.3})
            for i in range(3)
        ]
        normalized = normalize_score_vectors(svs)
        a_scores = [sv.scores["a"] for sv in normalized]
        assert min(a_scores) == pytest.approx(0.0, abs=1e-9)
        assert max(a_scores) == pytest.approx(1.0, abs=1e-9)

    def test_all_same_scores_become_zero(self):
        svs = [ScoreVector(idx1=i, idx2=i + 1, scores={"a": 0.5})
               for i in range(3)]
        normalized = normalize_score_vectors(svs)
        for sv in normalized:
            assert sv.scores["a"] == pytest.approx(0.0)

    def test_ids_preserved(self):
        svs = _sv_list(3)
        normalized = normalize_score_vectors(svs)
        for orig, norm in zip(svs, normalized):
            assert orig.idx1 == norm.idx1
            assert orig.idx2 == norm.idx2


# ─── TestBatchCombine ─────────────────────────────────────────────────────────

class TestBatchCombine:
    def test_returns_list(self):
        result = batch_combine(_sv_list(3))
        assert isinstance(result, list)

    def test_empty_list(self):
        assert batch_combine([]) == []

    def test_sorted_descending(self):
        result = batch_combine(_sv_list(5))
        scores = [cs.score for cs in result]
        assert scores == sorted(scores, reverse=True)

    def test_method_weighted(self):
        result = batch_combine(_sv_list(3), method="weighted")
        assert len(result) == 3

    def test_method_min(self):
        result = batch_combine(_sv_list(3), method="min")
        assert len(result) == 3

    def test_method_max(self):
        result = batch_combine(_sv_list(3), method="max")
        assert len(result) == 3

    def test_method_rank(self):
        result = batch_combine(_sv_list(3), method="rank")
        assert len(result) == 3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_combine(_sv_list(3), method="unknown")

    def test_all_combined_scores(self):
        for cs in batch_combine(_sv_list(3)):
            assert isinstance(cs, CombinedScore)

    def test_scores_in_range(self):
        for cs in batch_combine(_sv_list(4)):
            assert 0.0 <= cs.score <= 1.0
