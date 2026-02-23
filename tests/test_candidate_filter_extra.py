"""Extra tests for puzzle_reconstruction.assembly.candidate_filter."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.candidate_filter import (
    Candidate,
    FilterResult,
    batch_filter,
    deduplicate_candidates,
    filter_by_rank,
    filter_by_threshold,
    filter_top_k,
    merge_candidate_lists,
    normalize_scores,
)


def _cand(idx1=0, idx2=1, score=0.8):
    return Candidate(idx1=idx1, idx2=idx2, score=score)


def _cands(*specs):
    return [Candidate(idx1=i1, idx2=i2, score=s) for i1, i2, s in specs]


# ─── TestCandidateExtra ─────────────────────────────────────────────────────

class TestCandidateExtra:
    def test_fields(self):
        c = _cand(3, 7, 0.5)
        assert c.idx1 == 3 and c.idx2 == 7
        assert c.score == pytest.approx(0.5)

    def test_pair_property(self):
        c = _cand(2, 5, 0.6)
        assert c.pair == (2, 5)

    def test_default_params(self):
        assert _cand().params == {}

    def test_score_zero_ok(self):
        c = _cand(0, 1, 0.0)
        assert c.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        c = _cand(0, 1, 1.0)
        assert c.score == pytest.approx(1.0)

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            _cand(-1, 0, 0.5)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            _cand(0, -1, 0.5)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            _cand(0, 1, -0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            _cand(0, 1, 1.1)

    def test_same_indices_ok(self):
        c = _cand(5, 5, 0.5)
        assert c.idx1 == c.idx2


# ─── TestFilterResultExtra ──────────────────────────────────────────────────

class TestFilterResultExtra:
    def test_fields(self):
        fr = FilterResult(candidates=[], n_kept=3, n_removed=2)
        assert fr.n_kept == 3 and fr.n_removed == 2

    def test_len(self):
        fr = FilterResult(candidates=[_cand(), _cand(1, 2, 0.7)], n_kept=2, n_removed=0)
        assert len(fr) == 2

    def test_default_params(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=0)
        assert fr.params == {}

    def test_negative_n_kept_raises(self):
        with pytest.raises(ValueError):
            FilterResult(candidates=[], n_kept=-1, n_removed=0)

    def test_negative_n_removed_raises(self):
        with pytest.raises(ValueError):
            FilterResult(candidates=[], n_kept=0, n_removed=-1)


# ─── TestFilterByThresholdExtra ─────────────────────────────────────────────

class TestFilterByThresholdExtra:
    def test_all_above(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8))
        fr = filter_by_threshold(cands, 0.5)
        assert fr.n_kept == 2

    def test_all_below(self):
        cands = _cands((0, 1, 0.2), (1, 2, 0.1))
        fr = filter_by_threshold(cands, 0.5)
        assert fr.n_kept == 0

    def test_sorted_desc(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.9), (2, 3, 0.6))
        fr = filter_by_threshold(cands, 0.0)
        scores = [c.score for c in fr.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_zero_keeps_all(self):
        cands = _cands((0, 1, 0.0), (1, 2, 0.5))
        fr = filter_by_threshold(cands, 0.0)
        assert fr.n_kept == 2

    def test_threshold_one_strict(self):
        cands = _cands((0, 1, 1.0), (1, 2, 0.9))
        fr = filter_by_threshold(cands, 1.0)
        assert fr.n_kept == 1

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_threshold([], 1.5)

    def test_empty(self):
        fr = filter_by_threshold([], 0.5)
        assert fr.n_kept == 0 and fr.n_removed == 0

    def test_n_removed_correct(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.8), (2, 3, 0.6))
        fr = filter_by_threshold(cands, 0.5)
        assert fr.n_removed == 1


# ─── TestFilterTopKExtra ────────────────────────────────────────────────────

class TestFilterTopKExtra:
    def test_k_less_than_total(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.7))
        fr = filter_top_k(cands, k=2)
        assert fr.n_kept == 2

    def test_k_more_than_total(self):
        cands = _cands((0, 1, 0.8),)
        fr = filter_top_k(cands, k=10)
        assert fr.n_kept == 1

    def test_best_first(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.9))
        fr = filter_top_k(cands, k=1)
        assert fr.candidates[0].score == pytest.approx(0.9)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=0)

    def test_empty(self):
        fr = filter_top_k([], k=5)
        assert fr.n_kept == 0

    def test_n_removed(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.7))
        fr = filter_top_k(cands, k=1)
        assert fr.n_removed == 2

    def test_k_equal_total(self):
        cands = _cands((0, 1, 0.5), (1, 2, 0.7))
        fr = filter_top_k(cands, k=2)
        assert fr.n_kept == 2 and fr.n_removed == 0


# ─── TestFilterByRankExtra ──────────────────────────────────────────────────

class TestFilterByRankExtra:
    def test_half(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6))
        fr = filter_by_rank(cands, rank_threshold=0.5)
        assert fr.n_kept == 2

    def test_full_keeps_all(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8))
        fr = filter_by_rank(cands, rank_threshold=1.0)
        assert fr.n_kept == 2

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            filter_by_rank([], rank_threshold=0.0)

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_rank([], rank_threshold=1.5)

    def test_empty(self):
        fr = filter_by_rank([], rank_threshold=0.5)
        assert fr.n_kept == 0

    def test_quarter(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6))
        fr = filter_by_rank(cands, rank_threshold=0.25)
        assert fr.n_kept == 1


# ─── TestDeduplicateCandidatesExtra ─────────────────────────────────────────

class TestDeduplicateCandidatesExtra:
    def test_no_dups(self):
        cands = _cands((0, 1, 0.8), (2, 3, 0.7))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 2 and fr.n_removed == 0

    def test_exact_dup(self):
        cands = _cands((0, 1, 0.8), (0, 1, 0.6))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 1

    def test_symmetric_dup(self):
        cands = _cands((0, 1, 0.8), (1, 0, 0.6))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 1

    def test_best_score_kept(self):
        cands = _cands((0, 1, 0.4), (0, 1, 0.9), (0, 1, 0.7))
        fr = deduplicate_candidates(cands)
        assert fr.candidates[0].score == pytest.approx(0.9)

    def test_empty(self):
        fr = deduplicate_candidates([])
        assert fr.n_kept == 0

    def test_triple_dup(self):
        cands = _cands((0, 1, 0.5), (0, 1, 0.6), (0, 1, 0.7))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 1 and fr.n_removed == 2


# ─── TestNormalizeScoresExtra ───────────────────────────────────────────────

class TestNormalizeScoresExtra:
    def test_min_zero(self):
        cands = _cands((0, 1, 0.2), (1, 2, 0.8))
        scores = [c.score for c in normalize_scores(cands)]
        assert min(scores) == pytest.approx(0.0)

    def test_max_one(self):
        cands = _cands((0, 1, 0.2), (1, 2, 0.8))
        scores = [c.score for c in normalize_scores(cands)]
        assert max(scores) == pytest.approx(1.0)

    def test_all_equal_zero(self):
        cands = _cands((0, 1, 0.5), (1, 2, 0.5))
        assert all(c.score == pytest.approx(0.0) for c in normalize_scores(cands))

    def test_empty(self):
        assert normalize_scores([]) == []

    def test_indices_preserved(self):
        cands = _cands((3, 7, 0.4), (1, 2, 0.8))
        result = normalize_scores(cands)
        assert result[0].idx1 == 3 and result[0].idx2 == 7

    def test_single(self):
        result = normalize_scores(_cands((0, 1, 0.5)))
        assert result[0].score == pytest.approx(0.0)

    def test_returns_list(self):
        assert isinstance(normalize_scores(_cands((0, 1, 0.5))), list)


# ─── TestMergeCandidateListsExtra ───────────────────────────────────────────

class TestMergeCandidateListsExtra:
    def test_merges_all(self):
        l1 = _cands((0, 1, 0.8))
        l2 = _cands((2, 3, 0.7))
        assert len(merge_candidate_lists([l1, l2])) == 2

    def test_dedup_default(self):
        l1 = _cands((0, 1, 0.8))
        l2 = _cands((0, 1, 0.6))
        merged = merge_candidate_lists([l1, l2])
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(0.8)

    def test_no_dedup(self):
        l1 = _cands((0, 1, 0.8))
        l2 = _cands((0, 1, 0.6))
        assert len(merge_candidate_lists([l1, l2], dedup=False)) == 2

    def test_sorted(self):
        l1 = _cands((0, 1, 0.3))
        l2 = _cands((2, 3, 0.9))
        merged = merge_candidate_lists([l1, l2], dedup=False)
        assert merged[0].score >= merged[1].score

    def test_empty_lists(self):
        assert len(merge_candidate_lists([[], []])) == 0

    def test_single_list(self):
        l1 = _cands((0, 1, 0.5), (1, 2, 0.7))
        assert len(merge_candidate_lists([l1])) == 2


# ─── TestBatchFilterExtra ───────────────────────────────────────────────────

class TestBatchFilterExtra:
    def test_returns_list(self):
        assert isinstance(batch_filter([_cands((0, 1, 0.8))], threshold=0.5), list)

    def test_length(self):
        result = batch_filter([_cands((0, 1, 0.8)), _cands((1, 2, 0.3))],
                              threshold=0.5)
        assert len(result) == 2

    def test_threshold_applied(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.8))
        result = batch_filter([cands], threshold=0.5)
        assert result[0].n_kept == 1

    def test_top_k(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7))
        result = batch_filter([cands], threshold=0.0, top_k=2)
        assert result[0].n_kept == 2

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            batch_filter([], threshold=1.5)

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError):
            batch_filter([], threshold=0.5, top_k=0)

    def test_empty(self):
        assert batch_filter([], threshold=0.5) == []

    def test_all_filter_results(self):
        cands = _cands((0, 1, 0.8),)
        result = batch_filter([cands], threshold=0.5)
        assert isinstance(result[0], FilterResult)
