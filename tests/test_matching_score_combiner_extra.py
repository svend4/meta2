"""Extra tests for puzzle_reconstruction.matching.score_combiner."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sv(idx1=0, idx2=1, **scores):
    if not scores:
        scores = {"color": 0.8, "texture": 0.6}
    return ScoreVector(idx1=idx1, idx2=idx2, scores=scores)


def _sv_chain(n=4, base=0.2, step=0.15):
    return [
        ScoreVector(idx1=i, idx2=i + 10,
                    scores={"color": min(base + i * step, 1.0),
                            "shape": 0.5})
        for i in range(n)
    ]


# ─── TestScoreVectorExtra ────────────────────────────────────────────────────

class TestScoreVectorExtra:
    def test_pair_property(self):
        sv = _sv(idx1=4, idx2=9)
        assert sv.pair == (4, 9)

    def test_len_three_keys(self):
        sv = _sv(a=0.1, b=0.2, c=0.3)
        assert len(sv) == 3

    def test_params_default_empty(self):
        sv = _sv()
        assert sv.params == {}

    def test_zero_score_ok(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0})
        assert sv.scores["a"] == 0.0

    def test_one_score_ok(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0})
        assert sv.scores["a"] == 1.0

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": 1.01})

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=1, scores={"a": -0.01})

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=-1, idx2=0, scores={"a": 0.5})

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            ScoreVector(idx1=0, idx2=-1, scores={"a": 0.5})

    def test_empty_scores_ok(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        assert len(sv) == 0

    def test_multiple_scores_stored(self):
        sv = ScoreVector(idx1=0, idx2=1,
                         scores={"a": 0.3, "b": 0.5, "c": 0.7})
        assert sv.scores["c"] == pytest.approx(0.7)


# ─── TestCombinedScoreExtra ──────────────────────────────────────────────────

class TestCombinedScoreExtra:
    def test_pair_property(self):
        cs = CombinedScore(idx1=3, idx2=8, score=0.6)
        assert cs.pair == (3, 8)

    def test_score_stored(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.42)
        assert cs.score == pytest.approx(0.42)

    def test_zero_score_ok(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.0)
        assert cs.score == 0.0

    def test_one_score_ok(self):
        cs = CombinedScore(idx1=0, idx2=1, score=1.0)
        assert cs.score == 1.0

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=1.1)

    def test_below_zero_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=0, idx2=1, score=-0.1)

    def test_contributions_stored(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.7,
                           contributions={"color": 0.4, "shape": 0.3})
        assert cs.contributions["color"] == pytest.approx(0.4)

    def test_contributions_default_empty(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.5)
        assert cs.contributions == {}

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            CombinedScore(idx1=-1, idx2=0, score=0.5)


# ─── TestWeightedCombineExtra ────────────────────────────────────────────────

class TestWeightedCombineExtra:
    def test_returns_combined_score(self):
        cs = weighted_combine(_sv(color=0.8, texture=0.6))
        assert isinstance(cs, CombinedScore)

    def test_equal_weights_average(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.4, "b": 0.8})
        cs = weighted_combine(sv)
        assert cs.score == pytest.approx(0.6, abs=1e-5)

    def test_custom_weights_all_on_a(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0, "b": 0.0})
        cs = weighted_combine(sv, weights={"a": 1.0, "b": 0.0})
        assert cs.score == pytest.approx(1.0, abs=1e-4)

    def test_custom_weights_all_on_b(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.2, "b": 0.9})
        cs = weighted_combine(sv, weights={"a": 0.0, "b": 1.0})
        assert cs.score == pytest.approx(0.9, abs=1e-5)

    def test_score_in_range(self):
        cs = weighted_combine(_sv(color=0.5, texture=0.5))
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

    def test_contributions_contain_keys(self):
        cs = weighted_combine(_sv(color=0.8, texture=0.6))
        assert "color" in cs.contributions
        assert "texture" in cs.contributions


# ─── TestMinCombineExtra ─────────────────────────────────────────────────────

class TestMinCombineExtra:
    def test_returns_combined_score(self):
        assert isinstance(min_combine(_sv(color=0.8, texture=0.4)), CombinedScore)

    def test_is_minimum(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.9, "c": 0.6})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.3)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 0.65})
        assert min_combine(sv).score == pytest.approx(0.65)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            min_combine(ScoreVector(idx1=0, idx2=1, scores={}))

    def test_min_leq_max(self):
        sv = _sv(color=0.2, texture=0.8)
        assert min_combine(sv).score <= max_combine(sv).score

    def test_ids_preserved(self):
        sv = ScoreVector(idx1=5, idx2=9, scores={"a": 0.4})
        cs = min_combine(sv)
        assert cs.idx1 == 5 and cs.idx2 == 9


# ─── TestMaxCombineExtra ─────────────────────────────────────────────────────

class TestMaxCombineExtra:
    def test_returns_combined_score(self):
        assert isinstance(max_combine(_sv(color=0.8, texture=0.4)), CombinedScore)

    def test_is_maximum(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.9, "c": 0.6})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.9)

    def test_single_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 0.4})
        assert max_combine(sv).score == pytest.approx(0.4)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            max_combine(ScoreVector(idx1=0, idx2=1, scores={}))

    def test_max_geq_min(self):
        sv = _sv(color=0.3, texture=0.7)
        assert max_combine(sv).score >= min_combine(sv).score

    def test_ids_preserved(self):
        sv = ScoreVector(idx1=6, idx2=10, scores={"a": 0.5})
        cs = max_combine(sv)
        assert cs.idx1 == 6 and cs.idx2 == 10

    def test_all_equal_scores(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5, "b": 0.5})
        assert max_combine(sv).score == pytest.approx(0.5)


# ─── TestRankCombineExtra ────────────────────────────────────────────────────

class TestRankCombineExtra:
    def test_returns_list(self):
        result = rank_combine(_sv_chain(3))
        assert isinstance(result, list)

    def test_length_matches(self):
        svs = _sv_chain(5)
        assert len(rank_combine(svs)) == 5

    def test_empty_returns_empty(self):
        assert rank_combine([]) == []

    def test_single_item_score_one(self):
        sv = _sv(color=0.5, texture=0.5)
        result = rank_combine([sv])
        assert result[0].score == pytest.approx(1.0)

    def test_scores_in_range(self):
        for cs in rank_combine(_sv_chain(4)):
            assert 0.0 <= cs.score <= 1.0

    def test_all_combined_scores(self):
        for cs in rank_combine(_sv_chain(3)):
            assert isinstance(cs, CombinedScore)

    def test_mismatched_keys_raises(self):
        sv1 = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5})
        sv2 = ScoreVector(idx1=1, idx2=2, scores={"b": 0.5})
        with pytest.raises(ValueError):
            rank_combine([sv1, sv2])

    def test_best_has_highest_rank_score(self):
        svs = _sv_chain(4)
        result = rank_combine(svs)
        scores = [cs.score for cs in result]
        best_idx = max(range(len(svs)),
                       key=lambda i: svs[i].scores["color"])
        assert result[best_idx].score == max(scores)


# ─── TestNormalizeScoreVectorsExtra ──────────────────────────────────────────

class TestNormalizeScoreVectorsExtra:
    def test_returns_list(self):
        assert isinstance(normalize_score_vectors(_sv_chain(3)), list)

    def test_empty_list(self):
        assert normalize_score_vectors([]) == []

    def test_length_preserved(self):
        svs = _sv_chain(4)
        assert len(normalize_score_vectors(svs)) == 4

    def test_all_score_vectors(self):
        for sv in normalize_score_vectors(_sv_chain(3)):
            assert isinstance(sv, ScoreVector)

    def test_min_becomes_zero(self):
        svs = [
            ScoreVector(idx1=i, idx2=i + 1, scores={"a": 0.2 + i * 0.3})
            for i in range(3)
        ]
        normalized = normalize_score_vectors(svs)
        a_scores = [sv.scores["a"] for sv in normalized]
        assert min(a_scores) == pytest.approx(0.0, abs=1e-9)

    def test_max_becomes_one(self):
        svs = [
            ScoreVector(idx1=i, idx2=i + 1, scores={"a": 0.2 + i * 0.3})
            for i in range(3)
        ]
        normalized = normalize_score_vectors(svs)
        a_scores = [sv.scores["a"] for sv in normalized]
        assert max(a_scores) == pytest.approx(1.0, abs=1e-9)

    def test_constant_scores_become_zero(self):
        svs = [ScoreVector(idx1=i, idx2=i + 1, scores={"a": 0.7})
               for i in range(4)]
        normalized = normalize_score_vectors(svs)
        for sv in normalized:
            assert sv.scores["a"] == pytest.approx(0.0)

    def test_ids_preserved(self):
        svs = _sv_chain(3)
        normalized = normalize_score_vectors(svs)
        for orig, norm in zip(svs, normalized):
            assert orig.idx1 == norm.idx1
            assert orig.idx2 == norm.idx2


# ─── TestBatchCombineExtra ───────────────────────────────────────────────────

class TestBatchCombineExtra:
    def test_returns_list(self):
        assert isinstance(batch_combine(_sv_chain(3)), list)

    def test_empty_list(self):
        assert batch_combine([]) == []

    def test_sorted_descending(self):
        result = batch_combine(_sv_chain(5))
        scores = [cs.score for cs in result]
        assert scores == sorted(scores, reverse=True)

    def test_method_weighted(self):
        assert len(batch_combine(_sv_chain(3), method="weighted")) == 3

    def test_method_min(self):
        assert len(batch_combine(_sv_chain(3), method="min")) == 3

    def test_method_max(self):
        assert len(batch_combine(_sv_chain(3), method="max")) == 3

    def test_method_rank(self):
        assert len(batch_combine(_sv_chain(3), method="rank")) == 3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_combine(_sv_chain(3), method="unknown")

    def test_all_combined_scores(self):
        for cs in batch_combine(_sv_chain(3)):
            assert isinstance(cs, CombinedScore)

    def test_scores_in_range(self):
        for cs in batch_combine(_sv_chain(4)):
            assert 0.0 <= cs.score <= 1.0

    def test_single_item_result(self):
        sv = _sv(color=0.7, texture=0.5)
        result = batch_combine([sv])
        assert len(result) == 1
        assert isinstance(result[0], CombinedScore)
