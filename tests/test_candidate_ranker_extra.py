"""Extra tests for puzzle_reconstruction/matching/candidate_ranker.py"""
import numpy as np
import pytest

from puzzle_reconstruction.matching.candidate_ranker import (
    CandidatePair,
    batch_rank,
    deduplicate_pairs,
    filter_by_score,
    rank_pairs,
    score_pair,
    top_k,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pair(i1, i2, s, **meta):
    return CandidatePair(idx1=i1, idx2=i2, score=float(s), meta=dict(meta))


def _pairs(*scores):
    return [_pair(i, i + 1, s) for i, s in enumerate(scores)]


# ─── TestCandidatePairExtra ───────────────────────────────────────────────────

class TestCandidatePairExtra:
    def test_large_indices(self):
        p = _pair(10000, 99999, 0.5)
        assert p.idx1 == 10000
        assert p.idx2 == 99999

    def test_score_near_one(self):
        p = _pair(0, 1, 0.9999)
        assert p.score == pytest.approx(0.9999)

    def test_score_near_zero(self):
        p = _pair(0, 1, 0.0001)
        assert p.score == pytest.approx(0.0001)

    def test_meta_empty_by_default(self):
        p = _pair(0, 1, 0.5)
        assert p.meta == {}

    def test_meta_multiple_keys(self):
        p = _pair(0, 1, 0.5, a=1, b=2, c=3)
        assert p.meta["a"] == 1
        assert p.meta["b"] == 2
        assert p.meta["c"] == 3

    def test_sorting_multiple_pairs(self):
        pairs = [_pair(i, i + 1, float(i) / 10) for i in range(5)]
        ranked = sorted(pairs)
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score

    def test_same_score_no_exception(self):
        p1 = _pair(0, 1, 0.5)
        p2 = _pair(2, 3, 0.5)
        # Both orderings should be possible without exception
        _ = sorted([p1, p2])

    def test_idx1_idx2_independent(self):
        p = _pair(3, 7, 0.6)
        assert p.idx1 != p.idx2


# ─── TestScorePairExtra ───────────────────────────────────────────────────────

class TestScorePairExtra:
    def test_large_indices(self):
        p = score_pair(500, 999, 0.8)
        assert p.idx1 == 500
        assert p.idx2 == 999

    def test_meta_empty_default(self):
        p = score_pair(0, 1, 0.7)
        assert p.meta == {}

    def test_many_meta_kwargs(self):
        p = score_pair(0, 1, 0.5, side=2, method="clahe", weight=0.3)
        assert p.meta["side"] == 2
        assert p.meta["method"] == "clahe"
        assert p.meta["weight"] == pytest.approx(0.3)

    def test_score_float_conversion(self):
        p = score_pair(0, 1, 1)  # int input
        assert isinstance(p.score, float)

    def test_returns_candidate_pair(self):
        assert isinstance(score_pair(0, 1, 0.5), CandidatePair)

    def test_score_zero(self):
        p = score_pair(0, 1, 0.0)
        assert p.score == pytest.approx(0.0)

    def test_score_one(self):
        p = score_pair(0, 1, 1.0)
        assert p.score == pytest.approx(1.0)


# ─── TestRankPairsExtra ───────────────────────────────────────────────────────

class TestRankPairsExtra:
    def test_ten_pairs_sorted(self):
        import random
        random.seed(42)
        scores = [random.random() for _ in range(10)]
        pairs = _pairs(*scores)
        ranked = rank_pairs(pairs)
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score

    def test_strict_descending_distinct(self):
        pairs = _pairs(0.1, 0.5, 0.9, 0.3, 0.7)
        ranked = rank_pairs(pairs)
        assert ranked[0].score == pytest.approx(0.9)
        assert ranked[-1].score == pytest.approx(0.1)

    def test_input_not_modified(self):
        original = _pairs(0.9, 0.1, 0.5)
        scores_before = [p.score for p in original]
        rank_pairs(original)
        assert [p.score for p in original] == scores_before

    def test_returns_all_pairs(self):
        pairs = _pairs(*[i * 0.1 for i in range(8)])
        assert len(rank_pairs(pairs)) == 8

    def test_single_pair_preserved(self):
        pairs = [_pair(3, 9, 0.42)]
        ranked = rank_pairs(pairs)
        assert ranked[0].idx1 == 3
        assert ranked[0].idx2 == 9


# ─── TestFilterByScoreExtra ───────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_threshold_zero_keeps_all_nonzero(self):
        pairs = _pairs(0.1, 0.5, 0.9)
        result = filter_by_score(pairs, threshold=0.0)
        assert len(result) == 3

    def test_threshold_one_removes_all(self):
        pairs = _pairs(0.3, 0.7, 0.9)
        result = filter_by_score(pairs, threshold=1.0)
        assert len(result) == 0

    def test_filtered_still_sorted(self):
        pairs = _pairs(0.2, 0.6, 0.8, 0.9, 0.4)
        result = filter_by_score(pairs, threshold=0.5)
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_large_list(self):
        pairs = [_pair(i, i + 1, i / 100.0) for i in range(100)]
        result = filter_by_score(pairs, threshold=0.5)
        assert all(p.score > 0.5 for p in result)

    def test_returns_candidate_pairs(self):
        pairs = _pairs(0.6, 0.7)
        for p in filter_by_score(pairs, threshold=0.5):
            assert isinstance(p, CandidatePair)

    def test_single_pair_above_threshold(self):
        result = filter_by_score([_pair(0, 1, 0.8)], threshold=0.5)
        assert len(result) == 1

    def test_single_pair_at_threshold_excluded(self):
        result = filter_by_score([_pair(0, 1, 0.5)], threshold=0.5)
        assert len(result) == 0


# ─── TestTopKExtra ────────────────────────────────────────────────────────────

class TestTopKExtra:
    def test_k_equals_length(self):
        pairs = _pairs(0.3, 0.7, 0.5)
        assert len(top_k(pairs, k=3)) == 3

    def test_k_negative_empty(self):
        pairs = _pairs(0.5, 0.8)
        result = top_k(pairs, k=-1)
        assert result == []

    def test_top_3_correct_order(self):
        pairs = _pairs(0.1, 0.9, 0.5, 0.7, 0.3)
        result = top_k(pairs, k=3)
        assert result[0].score == pytest.approx(0.9)
        assert result[1].score == pytest.approx(0.7)
        assert result[2].score == pytest.approx(0.5)

    def test_deduplicate_five_pairs_chain(self):
        # Chain: (0,1), (1,2), (2,3), (3,4), (4,5) — greedy picks (0,1), (2,3), (4,5)
        pairs = [_pair(i, i + 1, 1.0 - i * 0.1) for i in range(5)]
        result = top_k(pairs, k=5, deduplicate=True)
        seen = set()
        for p in result:
            assert p.idx1 not in seen
            assert p.idx2 not in seen
            seen.add(p.idx1)
            seen.add(p.idx2)

    def test_no_deduplicate_allows_overlap(self):
        pairs = [_pair(0, 1, 0.9), _pair(0, 2, 0.8), _pair(0, 3, 0.7)]
        result = top_k(pairs, k=3, deduplicate=False)
        assert len(result) == 3

    def test_result_is_list(self):
        assert isinstance(top_k(_pairs(0.5, 0.8), k=1), list)

    def test_large_k_clips_to_available(self):
        pairs = _pairs(0.3, 0.7)
        assert len(top_k(pairs, k=1000)) == 2


# ─── TestDeduplicatePairsExtra ────────────────────────────────────────────────

class TestDeduplicatePairsExtra:
    def test_three_non_conflicting(self):
        pairs = [_pair(0, 1, 0.9), _pair(2, 3, 0.8), _pair(4, 5, 0.7)]
        result = deduplicate_pairs(pairs)
        assert len(result) == 3

    def test_all_conflicting_keeps_best(self):
        pairs = [_pair(0, 1, 0.3), _pair(0, 1, 0.9), _pair(0, 2, 0.6)]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_result_sorted_by_score(self):
        pairs = [_pair(0, 1, 0.6), _pair(2, 3, 0.9), _pair(4, 5, 0.3)]
        result = deduplicate_pairs(pairs)
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_indices_unique_in_output(self):
        pairs = [_pair(i, i + 1, float(i) / 10) for i in range(6)]
        result = deduplicate_pairs(pairs)
        seen = set()
        for p in result:
            assert p.idx1 not in seen
            assert p.idx2 not in seen
            seen.add(p.idx1)
            seen.add(p.idx2)

    def test_single_pair_returned(self):
        result = deduplicate_pairs([_pair(5, 10, 0.8)])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)

    def test_ten_non_overlapping(self):
        pairs = [_pair(2 * i, 2 * i + 1, 0.5 + i * 0.01) for i in range(10)]
        result = deduplicate_pairs(pairs)
        assert len(result) == 10


# ─── TestBatchRankExtra ───────────────────────────────────────────────────────

class TestBatchRankExtra:
    def _mat(self, n=4, seed=3):
        rng = np.random.default_rng(seed)
        m = rng.uniform(0.1, 1.0, (n, n)).astype(np.float32)
        np.fill_diagonal(m, 0.0)
        return m

    def test_5x5_matrix(self):
        m = self._mat(5)
        result = batch_rank(m)
        assert isinstance(result, list)
        assert all(isinstance(p, CandidatePair) for p in result)

    def test_all_scores_from_matrix(self):
        m = np.zeros((3, 3), dtype=np.float32)
        m[0, 1] = 0.8
        m[1, 0] = 0.8
        r = batch_rank(m, symmetric=True)
        assert len(r) == 1
        assert r[0].score == pytest.approx(0.8)

    def test_sorted_descending_large(self):
        m = self._mat(6, seed=7)
        result = batch_rank(m)
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_threshold_half(self):
        m = self._mat(4)
        result = batch_rank(m, threshold=0.5)
        assert all(p.score > 0.5 for p in result)

    def test_symmetric_indices_ordered(self):
        m = self._mat(5)
        result = batch_rank(m, symmetric=True)
        for p in result:
            assert p.idx1 < p.idx2

    def test_2x2_symmetric(self):
        m = np.array([[0.0, 0.6], [0.6, 0.0]], dtype=np.float32)
        result = batch_rank(m, symmetric=True)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.6)

    def test_full_diagonal_excluded(self):
        m = np.eye(4, dtype=np.float32)
        result = batch_rank(m, threshold=0.0)
        assert all(p.idx1 != p.idx2 for p in result)

    def test_8x8_no_crash(self):
        m = self._mat(8)
        result = batch_rank(m)
        assert len(result) > 0
