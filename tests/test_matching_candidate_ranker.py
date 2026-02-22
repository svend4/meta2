"""Тесты для puzzle_reconstruction.matching.candidate_ranker."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.candidate_ranker import (
    CandidatePair,
    score_pair,
    rank_pairs,
    filter_by_score,
    top_k,
    deduplicate_pairs,
    batch_rank,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pair(i=0, j=1, score=0.8) -> CandidatePair:
    return CandidatePair(idx1=i, idx2=j, score=score)


def _pairs(n=4) -> list:
    """Create n pairs with descending scores."""
    return [CandidatePair(idx1=i, idx2=i + 10, score=1.0 - i * 0.1)
            for i in range(n)]


# ─── TestCandidatePair ────────────────────────────────────────────────────────

class TestCandidatePair:
    def test_basic_fields(self):
        p = _pair(3, 7, 0.9)
        assert p.idx1 == 3
        assert p.idx2 == 7
        assert p.score == pytest.approx(0.9)

    def test_default_meta_empty(self):
        p = _pair()
        assert p.meta == {}

    def test_meta_stored(self):
        p = CandidatePair(idx1=0, idx2=1, score=0.5, meta={"method": "dtw"})
        assert p.meta["method"] == "dtw"

    def test_lt_higher_score_is_less(self):
        p1 = _pair(score=0.9)
        p2 = _pair(score=0.7)
        # p1 < p2 when sorted descending
        assert p1 < p2

    def test_sorting_descending(self):
        pairs = [_pair(score=0.3), _pair(score=0.9), _pair(score=0.6)]
        s = sorted(pairs)
        assert s[0].score == pytest.approx(0.9)

    def test_score_zero_ok(self):
        p = CandidatePair(idx1=0, idx2=1, score=0.0)
        assert p.score == 0.0

    def test_score_one_ok(self):
        p = CandidatePair(idx1=0, idx2=1, score=1.0)
        assert p.score == 1.0


# ─── TestScorePair ────────────────────────────────────────────────────────────

class TestScorePair:
    def test_returns_candidate_pair(self):
        p = score_pair(0, 1, 0.8)
        assert isinstance(p, CandidatePair)

    def test_ids_stored(self):
        p = score_pair(3, 5, 0.7)
        assert p.idx1 == 3
        assert p.idx2 == 5

    def test_score_stored(self):
        p = score_pair(0, 1, 0.75)
        assert p.score == pytest.approx(0.75)

    def test_meta_as_kwargs(self):
        p = score_pair(0, 1, 0.8, method="ncc", channel="color")
        assert p.meta["method"] == "ncc"
        assert p.meta["channel"] == "color"

    def test_score_cast_to_float(self):
        p = score_pair(0, 1, 1)  # int input
        assert isinstance(p.score, float)

    def test_empty_meta(self):
        p = score_pair(0, 1, 0.5)
        assert p.meta == {}


# ─── TestRankPairs ────────────────────────────────────────────────────────────

class TestRankPairs:
    def test_returns_list(self):
        assert isinstance(rank_pairs(_pairs(3)), list)

    def test_sorted_descending(self):
        pairs = [_pair(score=0.3), _pair(score=0.9), _pair(score=0.6)]
        result = rank_pairs(pairs)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        assert rank_pairs([]) == []

    def test_single_pair(self):
        p = _pair(score=0.7)
        result = rank_pairs([p])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)

    def test_length_preserved(self):
        pairs = _pairs(5)
        assert len(rank_pairs(pairs)) == 5

    def test_all_same_score(self):
        pairs = [_pair(score=0.5) for _ in range(4)]
        result = rank_pairs(pairs)
        assert len(result) == 4


# ─── TestFilterByScore ────────────────────────────────────────────────────────

class TestFilterByScore:
    def test_returns_list(self):
        assert isinstance(filter_by_score(_pairs(3)), list)

    def test_threshold_applied(self):
        pairs = [_pair(score=0.3), _pair(score=0.7), _pair(score=0.9)]
        result = filter_by_score(pairs, threshold=0.5)
        assert all(p.score > 0.5 for p in result)

    def test_default_threshold_half(self):
        pairs = [_pair(score=0.4), _pair(score=0.6)]
        result = filter_by_score(pairs)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.6)

    def test_all_below_threshold(self):
        pairs = [_pair(score=0.1), _pair(score=0.2)]
        assert filter_by_score(pairs, threshold=0.5) == []

    def test_all_above_threshold(self):
        pairs = [_pair(score=0.8), _pair(score=0.9)]
        result = filter_by_score(pairs, threshold=0.0)
        assert len(result) == 2

    def test_boundary_excluded(self):
        # score > threshold (strict), so score == threshold is excluded
        pairs = [_pair(score=0.5)]
        result = filter_by_score(pairs, threshold=0.5)
        assert result == []

    def test_result_sorted_desc(self):
        pairs = [_pair(score=0.6), _pair(score=0.9), _pair(score=0.7)]
        result = filter_by_score(pairs, threshold=0.0)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)


# ─── TestDeduplicatePairs ─────────────────────────────────────────────────────

class TestDeduplicatePairs:
    def test_returns_list(self):
        assert isinstance(deduplicate_pairs(_pairs(3)), list)

    def test_no_overlap_all_kept(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=2, idx2=3, score=0.8),
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 2

    def test_overlap_one_removed(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=0, idx2=2, score=0.8),  # idx1=0 already used
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_highest_score_wins(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.7),
            CandidatePair(idx1=0, idx2=2, score=0.9),
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
        assert result[0].idx2 == 2

    def test_empty_list(self):
        assert deduplicate_pairs([]) == []

    def test_result_is_subset(self):
        pairs = _pairs(5)
        result = deduplicate_pairs(pairs)
        pair_ids = set()
        for p in result:
            assert p.idx1 not in pair_ids
            assert p.idx2 not in pair_ids
            pair_ids.add(p.idx1)
            pair_ids.add(p.idx2)

    def test_single_pair_kept(self):
        pairs = [CandidatePair(idx1=0, idx2=1, score=0.5)]
        assert len(deduplicate_pairs(pairs)) == 1


# ─── TestTopK ─────────────────────────────────────────────────────────────────

class TestTopK:
    def test_returns_list(self):
        assert isinstance(top_k(_pairs(5), k=3), list)

    def test_k_respected(self):
        pairs = _pairs(10)
        result = top_k(pairs, k=4)
        assert len(result) == 4

    def test_k_more_than_pairs(self):
        pairs = _pairs(3)
        result = top_k(pairs, k=10)
        assert len(result) == 3

    def test_k_zero_empty(self):
        pairs = _pairs(5)
        assert top_k(pairs, k=0) == []

    def test_top_scores_returned(self):
        pairs = [_pair(score=0.3), _pair(score=0.9), _pair(score=0.6)]
        result = top_k(pairs, k=2)
        scores = {p.score for p in result}
        assert 0.9 in scores

    def test_deduplicate_false(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=0, idx2=2, score=0.8),
        ]
        result = top_k(pairs, k=2, deduplicate=False)
        assert len(result) == 2

    def test_deduplicate_true(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=0, idx2=2, score=0.8),
        ]
        result = top_k(pairs, k=2, deduplicate=True)
        # Only first pair kept due to dedup
        assert len(result) == 1

    def test_empty_pairs_empty_result(self):
        assert top_k([], k=5) == []


# ─── TestBatchRank ────────────────────────────────────────────────────────────

class TestBatchRank:
    def _mat(self, n=4, seed=0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mat = rng.uniform(0, 1, (n, n)).astype(np.float32)
        np.fill_diagonal(mat, 0.0)
        return mat

    def test_returns_list(self):
        mat = self._mat(4)
        result = batch_rank(mat)
        assert isinstance(result, list)

    def test_all_candidate_pairs(self):
        mat = self._mat(4)
        result = batch_rank(mat)
        assert all(isinstance(p, CandidatePair) for p in result)

    def test_sorted_descending(self):
        mat = self._mat(5)
        result = batch_rank(mat)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_symmetric_upper_tri_only(self):
        mat = self._mat(4)
        result = batch_rank(mat, symmetric=True)
        for p in result:
            assert p.idx1 < p.idx2

    def test_symmetric_false_includes_both(self):
        mat = self._mat(4)
        result = batch_rank(mat, symmetric=False)
        # Should include both (i,j) and (j,i)
        assert any(p.idx1 > p.idx2 for p in result)

    def test_threshold_applied(self):
        mat = self._mat(4)
        result = batch_rank(mat, threshold=0.5)
        assert all(p.score > 0.5 for p in result)

    def test_non_square_raises(self):
        mat = np.ones((3, 4))
        with pytest.raises(ValueError):
            batch_rank(mat)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            batch_rank(np.ones(4))

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            batch_rank(np.ones((3, 3, 3)))

    def test_identity_matrix_no_pairs(self):
        mat = np.eye(5)
        result = batch_rank(mat, threshold=0.5)
        # Diagonal is 0, off-diagonal is 0 — all filtered out
        assert result == []

    def test_n1_matrix_empty(self):
        mat = np.array([[0.0]])
        result = batch_rank(mat)
        assert result == []
