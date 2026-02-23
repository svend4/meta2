"""Extra tests for puzzle_reconstruction.matching.candidate_ranker."""
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


def _pair(i=0, j=1, score=0.8) -> CandidatePair:
    return CandidatePair(idx1=i, idx2=j, score=score)


def _pairs(n=4) -> list:
    return [CandidatePair(idx1=i, idx2=i + 10, score=1.0 - i * 0.1) for i in range(n)]


# ─── TestCandidatePairExtra ──────────────────────────────────────────────────

class TestCandidatePairExtra:
    def test_eq_same_fields(self):
        a = _pair(1, 2, 0.5)
        b = _pair(1, 2, 0.5)
        assert a.idx1 == b.idx1 and a.idx2 == b.idx2 and a.score == b.score

    def test_meta_dict_mutation(self):
        p = CandidatePair(idx1=0, idx2=1, score=0.5, meta={"a": 1})
        p.meta["b"] = 2
        assert "b" in p.meta

    def test_negative_score_allowed(self):
        p = CandidatePair(idx1=0, idx2=1, score=-0.5)
        assert p.score == pytest.approx(-0.5)

    def test_large_score_allowed(self):
        p = CandidatePair(idx1=0, idx2=1, score=100.0)
        assert p.score == pytest.approx(100.0)

    def test_same_indices_allowed(self):
        p = CandidatePair(idx1=5, idx2=5, score=1.0)
        assert p.idx1 == p.idx2

    def test_repr_has_info(self):
        p = _pair(3, 7, 0.9)
        r = repr(p)
        assert "3" in r or "CandidatePair" in r

    def test_sorting_stable(self):
        pairs = [_pair(i=i, score=0.5) for i in range(5)]
        s = sorted(pairs)
        assert len(s) == 5

    def test_lt_equal_scores(self):
        a = _pair(score=0.5)
        b = _pair(score=0.5)
        # should not raise
        _ = (a < b) or (b < a) or True


# ─── TestScorePairExtra ──────────────────────────────────────────────────────

class TestScorePairExtra:
    def test_returns_float_score(self):
        p = score_pair(0, 1, 0.5)
        assert isinstance(p.score, float)

    def test_int_score_cast(self):
        p = score_pair(0, 1, 0)
        assert isinstance(p.score, float)

    def test_numpy_score_cast(self):
        p = score_pair(0, 1, np.float32(0.75))
        assert isinstance(p.score, float)

    def test_multiple_meta_keys(self):
        p = score_pair(0, 1, 0.8, method="ncc", channel="gray", n_matches=42)
        assert p.meta["n_matches"] == 42
        assert len(p.meta) == 3

    def test_idx_preserved(self):
        p = score_pair(100, 200, 0.1)
        assert p.idx1 == 100
        assert p.idx2 == 200

    def test_zero_score(self):
        p = score_pair(0, 1, 0.0)
        assert p.score == pytest.approx(0.0)

    def test_one_score(self):
        p = score_pair(0, 1, 1.0)
        assert p.score == pytest.approx(1.0)


# ─── TestRankPairsExtra ──────────────────────────────────────────────────────

class TestRankPairsExtra:
    def test_descending_order(self):
        pairs = [_pair(score=0.1), _pair(score=0.9), _pair(score=0.5)]
        result = rank_pairs(pairs)
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_two_equal_scores(self):
        pairs = [_pair(score=0.5), _pair(score=0.5)]
        result = rank_pairs(pairs)
        assert len(result) == 2

    def test_already_sorted(self):
        pairs = [_pair(score=0.9), _pair(score=0.7), _pair(score=0.3)]
        result = rank_pairs(pairs)
        assert [p.score for p in result] == pytest.approx([0.9, 0.7, 0.3])

    def test_reverse_sorted_input(self):
        pairs = [_pair(score=0.1), _pair(score=0.5), _pair(score=0.9)]
        result = rank_pairs(pairs)
        assert result[0].score == pytest.approx(0.9)

    def test_single_element(self):
        result = rank_pairs([_pair(score=0.42)])
        assert len(result) == 1

    def test_large_list(self):
        pairs = [_pair(score=i / 100.0) for i in range(100)]
        result = rank_pairs(pairs)
        assert len(result) == 100
        assert result[0].score >= result[-1].score

    def test_meta_preserved(self):
        p = CandidatePair(idx1=0, idx2=1, score=0.5, meta={"key": "val"})
        result = rank_pairs([p])
        assert result[0].meta["key"] == "val"


# ─── TestFilterByScoreExtra ──────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_threshold_zero_keeps_positive(self):
        pairs = [_pair(score=0.0), _pair(score=0.1)]
        result = filter_by_score(pairs, threshold=0.0)
        assert len(result) == 1

    def test_threshold_one_excludes_all(self):
        pairs = [_pair(score=0.9), _pair(score=0.99)]
        result = filter_by_score(pairs, threshold=1.0)
        assert result == []

    def test_empty_input(self):
        assert filter_by_score([], threshold=0.5) == []

    def test_all_pass(self):
        pairs = [_pair(score=0.8), _pair(score=0.9)]
        result = filter_by_score(pairs, threshold=0.1)
        assert len(result) == 2

    def test_result_is_sorted(self):
        pairs = [_pair(score=0.6), _pair(score=0.8), _pair(score=0.7)]
        result = filter_by_score(pairs, threshold=0.5)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_negative_threshold(self):
        pairs = [_pair(score=0.0), _pair(score=0.1)]
        result = filter_by_score(pairs, threshold=-1.0)
        assert len(result) == 2

    def test_meta_preserved(self):
        p = CandidatePair(idx1=0, idx2=1, score=0.9, meta={"m": 1})
        result = filter_by_score([p], threshold=0.0)
        assert result[0].meta["m"] == 1


# ─── TestDeduplicatePairsExtra ───────────────────────────────────────────────

class TestDeduplicatePairsExtra:
    def test_three_overlapping_keeps_best(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.7),
            CandidatePair(idx1=0, idx2=2, score=0.9),
            CandidatePair(idx1=0, idx2=3, score=0.5),
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_no_overlap(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.5),
            CandidatePair(idx1=2, idx2=3, score=0.6),
            CandidatePair(idx1=4, idx2=5, score=0.7),
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 3

    def test_idx2_overlap(self):
        pairs = [
            CandidatePair(idx1=0, idx2=5, score=0.9),
            CandidatePair(idx1=1, idx2=5, score=0.8),
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1

    def test_chain_dedup(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=1, idx2=2, score=0.8),
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_single(self):
        result = deduplicate_pairs([_pair(score=0.5)])
        assert len(result) == 1

    def test_empty(self):
        assert deduplicate_pairs([]) == []

    def test_returns_candidate_pairs(self):
        result = deduplicate_pairs(_pairs(3))
        assert all(isinstance(p, CandidatePair) for p in result)


# ─── TestTopKExtra ───────────────────────────────────────────────────────────

class TestTopKExtra:
    def test_k_1(self):
        pairs = _pairs(5)
        result = top_k(pairs, k=1)
        assert len(result) == 1

    def test_k_negative_empty(self):
        pairs = _pairs(5)
        result = top_k(pairs, k=-1)
        assert result == []

    def test_top_scores(self):
        pairs = [_pair(score=s) for s in [0.1, 0.5, 0.9, 0.3, 0.7]]
        result = top_k(pairs, k=2)
        scores = {p.score for p in result}
        assert 0.9 in scores

    def test_dedup_true_reduces(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=0, idx2=2, score=0.8),
            CandidatePair(idx1=3, idx2=4, score=0.7),
        ]
        result = top_k(pairs, k=3, deduplicate=True)
        assert len(result) <= 2

    def test_dedup_false_keeps_all(self):
        pairs = [
            CandidatePair(idx1=0, idx2=1, score=0.9),
            CandidatePair(idx1=0, idx2=2, score=0.8),
        ]
        result = top_k(pairs, k=2, deduplicate=False)
        assert len(result) == 2

    def test_empty(self):
        assert top_k([], k=5) == []

    def test_returns_list(self):
        assert isinstance(top_k(_pairs(3), k=2), list)

    def test_k_equal_len(self):
        pairs = _pairs(3)
        result = top_k(pairs, k=3)
        assert len(result) == 3


# ─── TestBatchRankExtra ──────────────────────────────────────────────────────

class TestBatchRankExtra:
    def _mat(self, n=4, seed=0):
        rng = np.random.default_rng(seed)
        mat = rng.uniform(0, 1, (n, n)).astype(np.float32)
        np.fill_diagonal(mat, 0.0)
        return mat

    def test_n2_matrix(self):
        mat = np.array([[0.0, 0.8], [0.6, 0.0]], dtype=np.float32)
        result = batch_rank(mat)
        assert len(result) >= 1

    def test_symmetric_no_self_pairs(self):
        mat = self._mat(4)
        result = batch_rank(mat, symmetric=True)
        for p in result:
            assert p.idx1 != p.idx2

    def test_symmetric_false_may_have_reverse(self):
        mat = self._mat(3)
        result = batch_rank(mat, symmetric=False)
        idx_pairs = {(p.idx1, p.idx2) for p in result}
        has_reverse = any((j, i) in idx_pairs for (i, j) in idx_pairs)
        assert has_reverse

    def test_all_zeros_no_pairs(self):
        mat = np.zeros((3, 3), dtype=np.float32)
        result = batch_rank(mat, threshold=0.0)
        assert result == []

    def test_threshold_filters(self):
        mat = self._mat(5)
        full = batch_rank(mat, threshold=0.0)
        filtered = batch_rank(mat, threshold=0.5)
        assert len(filtered) <= len(full)

    def test_all_pairs_candidate_pair(self):
        mat = self._mat(3)
        result = batch_rank(mat)
        assert all(isinstance(p, CandidatePair) for p in result)

    def test_result_descending(self):
        mat = self._mat(4)
        result = batch_rank(mat)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            batch_rank(np.ones((2, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            batch_rank(np.ones(5))

    def test_large_matrix(self):
        mat = self._mat(20)
        result = batch_rank(mat, symmetric=True)
        assert len(result) > 0
