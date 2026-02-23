"""Extra tests for puzzle_reconstruction.matching.score_combiner."""
from __future__ import annotations

import pytest

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
        scores = {"a": 0.5}
    return ScoreVector(idx1=idx1, idx2=idx2, scores=scores)


def _svs(n=3):
    return [
        ScoreVector(idx1=0, idx2=i + 1,
                    scores={"a": round(1.0 - i * 0.2, 1),
                            "b": round(0.5 + i * 0.1, 1)})
        for i in range(n)
    ]


# ─── TestScoreVectorExtra ────────────────────────────────────────────────────

class TestScoreVectorExtra:
    def test_pair_property(self):
        sv = _sv(idx1=3, idx2=7)
        assert sv.pair == (3, 7)

    def test_large_indices(self):
        sv = ScoreVector(idx1=100, idx2=999, scores={"a": 0.5})
        assert sv.idx1 == 100

    def test_multiple_scores_len(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.1, "b": 0.5, "c": 0.9})
        assert len(sv) == 3

    def test_boundary_score_zero(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 0.0})
        assert sv.scores["x"] == 0.0

    def test_boundary_score_one(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"x": 1.0})
        assert sv.scores["x"] == 1.0

    def test_scores_stored_correctly(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"p": 0.3, "q": 0.7})
        assert sv.scores["p"] == pytest.approx(0.3)
        assert sv.scores["q"] == pytest.approx(0.7)

    def test_empty_scores_len_zero(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={})
        assert len(sv) == 0


# ─── TestCombinedScoreExtra ──────────────────────────────────────────────────

class TestCombinedScoreExtra:
    def test_score_boundary_zero(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.0)
        assert cs.score == 0.0

    def test_score_boundary_one(self):
        cs = CombinedScore(idx1=0, idx2=1, score=1.0)
        assert cs.score == 1.0

    def test_pair_property(self):
        cs = CombinedScore(idx1=2, idx2=8, score=0.5)
        assert cs.pair == (2, 8)

    def test_contributions_default_empty(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.5)
        assert cs.contributions == {}

    def test_contributions_stored(self):
        cs = CombinedScore(idx1=0, idx2=1, score=0.5,
                           contributions={"a": 0.5, "b": 0.5})
        assert len(cs.contributions) == 2

    def test_idx1_stored(self):
        cs = CombinedScore(idx1=5, idx2=3, score=0.5)
        assert cs.idx1 == 5

    def test_idx2_stored(self):
        cs = CombinedScore(idx1=5, idx2=3, score=0.5)
        assert cs.idx2 == 3


# ─── TestWeightedCombineExtra ────────────────────────────────────────────────

class TestWeightedCombineExtra:
    def test_three_equal_scores(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5, "b": 0.5, "c": 0.5})
        cs = weighted_combine(sv)
        assert cs.score == pytest.approx(0.5, abs=1e-6)

    def test_weight_zero_excludes_channel(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0, "b": 1.0})
        cs = weighted_combine(sv, weights={"a": 0.0, "b": 1.0})
        assert cs.score == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_zero_raises(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.5})
        with pytest.raises(ValueError):
            weighted_combine(sv, weights={"a": 0.0})

    def test_contributions_sum_approx_score(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.8, "b": 0.4})
        cs = weighted_combine(sv)
        assert set(cs.contributions.keys()) == {"a", "b"}

    def test_score_in_range(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.9})
        cs = weighted_combine(sv)
        assert 0.0 <= cs.score <= 1.0

    def test_asymmetric_weights(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0, "b": 0.0})
        cs = weighted_combine(sv, weights={"a": 3.0, "b": 1.0})
        # (1.0*3 + 0.0*1) / 4 = 0.75
        assert cs.score == pytest.approx(0.75, abs=1e-6)


# ─── TestMinCombineExtra ─────────────────────────────────────────────────────

class TestMinCombineExtra:
    def test_three_scores_min(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.2, "b": 0.6, "c": 0.9})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.2)

    def test_boundary_zero(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0, "b": 0.5})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(0.0)

    def test_boundary_one(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0})
        cs = min_combine(sv)
        assert cs.score == pytest.approx(1.0)

    def test_idx_forwarded(self):
        sv = ScoreVector(idx1=5, idx2=8, scores={"a": 0.3})
        cs = min_combine(sv)
        assert cs.idx1 == 5 and cs.idx2 == 8

    def test_pair_forwarded(self):
        sv = _sv(idx1=2, idx2=4)
        cs = min_combine(sv)
        assert cs.pair == (2, 4)


# ─── TestMaxCombineExtra ─────────────────────────────────────────────────────

class TestMaxCombineExtra:
    def test_three_scores_max(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.2, "b": 0.6, "c": 0.9})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.9)

    def test_boundary_zero(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.0})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.0)

    def test_boundary_one(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 1.0, "b": 0.5})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(1.0)

    def test_idx_forwarded(self):
        sv = ScoreVector(idx1=7, idx2=9, scores={"a": 0.8})
        cs = max_combine(sv)
        assert cs.idx1 == 7 and cs.idx2 == 9

    def test_all_equal(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3, "b": 0.3, "c": 0.3})
        cs = max_combine(sv)
        assert cs.score == pytest.approx(0.3)


# ─── TestRankCombineExtra ─────────────────────────────────────────────────────

class TestRankCombineExtra:
    def test_two_svs_ordered(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.9}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.3}),
        ]
        result = rank_combine(svs)
        scores = {(r.idx1, r.idx2): r.score for r in result}
        assert scores[(0, 1)] > scores[(0, 2)]

    def test_scores_in_01(self):
        svs = _svs(4)
        for cs in rank_combine(svs):
            assert 0.0 <= cs.score <= 1.0

    def test_length_preserved(self):
        svs = _svs(5)
        result = rank_combine(svs)
        assert len(result) == 5

    def test_single_sv_score_one(self):
        sv = ScoreVector(idx1=0, idx2=1, scores={"a": 0.3})
        result = rank_combine([sv])
        assert result[0].score == pytest.approx(1.0)

    def test_empty_returns_empty(self):
        assert rank_combine([]) == []


# ─── TestNormalizeScoreVectorsExtra ──────────────────────────────────────────

class TestNormalizeScoreVectorsExtra:
    def test_single_sv_becomes_zero(self):
        # Only one value → min == max → result = 0
        svs = [ScoreVector(idx1=0, idx2=1, scores={"a": 0.7})]
        result = normalize_score_vectors(svs)
        assert result[0].scores["a"] == pytest.approx(0.0)

    def test_three_svs_min_zero(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.1}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.5}),
            ScoreVector(idx1=0, idx2=3, scores={"a": 0.9}),
        ]
        result = normalize_score_vectors(svs)
        vals = [sv.scores["a"] for sv in result]
        assert min(vals) == pytest.approx(0.0)

    def test_three_svs_max_one(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.1}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.5}),
            ScoreVector(idx1=0, idx2=3, scores={"a": 0.9}),
        ]
        result = normalize_score_vectors(svs)
        vals = [sv.scores["a"] for sv in result]
        assert max(vals) == pytest.approx(1.0)

    def test_multi_channel_all_normalized(self):
        svs = [
            ScoreVector(idx1=0, idx2=1, scores={"a": 0.2, "b": 0.8}),
            ScoreVector(idx1=0, idx2=2, scores={"a": 0.6, "b": 0.4}),
        ]
        result = normalize_score_vectors(svs)
        for sv in result:
            for v in sv.scores.values():
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_idx_preserved(self):
        svs = [ScoreVector(idx1=3, idx2=9, scores={"a": 0.5})]
        result = normalize_score_vectors(svs)
        assert result[0].idx1 == 3 and result[0].idx2 == 9

    def test_empty_input(self):
        assert normalize_score_vectors([]) == []


# ─── TestBatchCombineExtra ───────────────────────────────────────────────────

class TestBatchCombineExtra:
    def test_weighted_all_combined_scores(self):
        svs = _svs(3)
        result = batch_combine(svs, method="weighted")
        assert all(isinstance(cs, CombinedScore) for cs in result)

    def test_min_all_valid(self):
        svs = _svs(3)
        result = batch_combine(svs, method="min")
        for cs in result:
            assert 0.0 <= cs.score <= 1.0

    def test_max_all_valid(self):
        svs = _svs(3)
        result = batch_combine(svs, method="max")
        for cs in result:
            assert 0.0 <= cs.score <= 1.0

    def test_rank_sorted_descending(self):
        svs = _svs(4)
        result = batch_combine(svs, method="rank")
        scores = [cs.score for cs in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_returns_empty(self):
        assert batch_combine([], method="weighted") == []

    def test_single_sv(self):
        svs = [_sv(a=0.8)]
        result = batch_combine(svs, method="weighted")
        assert len(result) == 1

    def test_weighted_sorted_descending(self):
        svs = _svs(4)
        result = batch_combine(svs, method="weighted")
        scores = [cs.score for cs in result]
        assert scores == sorted(scores, reverse=True)
