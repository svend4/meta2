"""Tests for puzzle_reconstruction.assembly.candidate_filter."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _cand(idx1: int, idx2: int, score: float) -> Candidate:
    return Candidate(idx1=idx1, idx2=idx2, score=score)


def _cands(*specs):
    return [_cand(i1, i2, s) for i1, i2, s in specs]


# ─── Candidate ───────────────────────────────────────────────────────────────

class TestCandidate:
    def test_fields_stored(self):
        c = _cand(0, 1, 0.8)
        assert c.idx1 == 0
        assert c.idx2 == 1
        assert c.score == pytest.approx(0.8)

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

    def test_score_zero_ok(self):
        c = _cand(0, 1, 0.0)
        assert c.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        c = _cand(0, 1, 1.0)
        assert c.score == pytest.approx(1.0)

    def test_pair_property(self):
        c = _cand(3, 7, 0.5)
        assert c.pair == (3, 7)

    def test_default_params_empty(self):
        c = _cand(0, 1, 0.5)
        assert c.params == {}


# ─── FilterResult ─────────────────────────────────────────────────────────────

class TestFilterResult:
    def test_fields_stored(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=5)
        assert fr.n_kept == 0
        assert fr.n_removed == 5

    def test_negative_n_kept_raises(self):
        with pytest.raises(ValueError):
            FilterResult(candidates=[], n_kept=-1, n_removed=0)

    def test_negative_n_removed_raises(self):
        with pytest.raises(ValueError):
            FilterResult(candidates=[], n_kept=0, n_removed=-1)

    def test_len(self):
        fr = FilterResult(candidates=[_cand(0, 1, 0.5)], n_kept=1, n_removed=0)
        assert len(fr) == 1

    def test_default_params_empty(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=0)
        assert fr.params == {}


# ─── filter_by_threshold ─────────────────────────────────────────────────────

class TestFilterByThreshold:
    def test_all_above_threshold_kept(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8))
        fr = filter_by_threshold(cands, 0.7)
        assert fr.n_kept == 2

    def test_none_above_threshold_removed(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.2))
        fr = filter_by_threshold(cands, 0.5)
        assert fr.n_kept == 0
        assert fr.n_removed == 2

    def test_sorted_by_score_descending(self):
        cands = _cands((0, 1, 0.5), (1, 2, 0.9), (2, 3, 0.7))
        fr = filter_by_threshold(cands, 0.0)
        scores = [c.score for c in fr.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_zero_keeps_all(self):
        cands = _cands((0, 1, 0.0), (1, 2, 1.0))
        fr = filter_by_threshold(cands, 0.0)
        assert fr.n_kept == 2

    def test_threshold_one_keeps_only_one(self):
        cands = _cands((0, 1, 0.9), (1, 2, 1.0))
        fr = filter_by_threshold(cands, 1.0)
        assert fr.n_kept == 1

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_threshold(_cands((0, 1, 0.5)), 1.5)

    def test_n_removed_correct(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.8), (2, 3, 0.6))
        fr = filter_by_threshold(cands, 0.5)
        assert fr.n_removed == 1

    def test_empty_input(self):
        fr = filter_by_threshold([], 0.5)
        assert fr.n_kept == 0
        assert fr.n_removed == 0


# ─── filter_top_k ─────────────────────────────────────────────────────────────

class TestFilterTopK:
    def test_top_k_correct(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.7))
        fr = filter_top_k(cands, k=2)
        assert fr.n_kept == 2

    def test_k_above_total_returns_all(self):
        cands = _cands((0, 1, 0.8), (1, 2, 0.6))
        fr = filter_top_k(cands, k=10)
        assert fr.n_kept == 2

    def test_best_score_first(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.9), (2, 3, 0.6))
        fr = filter_top_k(cands, k=1)
        assert fr.candidates[0].score == pytest.approx(0.9)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k(_cands((0, 1, 0.5)), k=0)

    def test_empty_input(self):
        fr = filter_top_k([], k=5)
        assert fr.n_kept == 0

    def test_n_removed_correct(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.5), (2, 3, 0.7))
        fr = filter_top_k(cands, k=2)
        assert fr.n_removed == 1


# ─── filter_by_rank ───────────────────────────────────────────────────────────

class TestFilterByRank:
    def test_half_kept(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.6))
        fr = filter_by_rank(cands, rank_threshold=0.5)
        assert fr.n_kept == 2

    def test_full_rank_keeps_all(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8))
        fr = filter_by_rank(cands, rank_threshold=1.0)
        assert fr.n_kept == 2

    def test_zero_rank_raises(self):
        with pytest.raises(ValueError):
            filter_by_rank(_cands((0, 1, 0.5)), rank_threshold=0.0)

    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_rank(_cands((0, 1, 0.5)), rank_threshold=1.5)

    def test_empty_input(self):
        fr = filter_by_rank([], rank_threshold=0.5)
        assert fr.n_kept == 0


# ─── deduplicate_candidates ───────────────────────────────────────────────────

class TestDeduplicateCandidates:
    def test_no_duplicates_unchanged(self):
        cands = _cands((0, 1, 0.8), (1, 2, 0.7))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 2
        assert fr.n_removed == 0

    def test_duplicate_pair_removed(self):
        cands = _cands((0, 1, 0.8), (0, 1, 0.6))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 1
        assert fr.candidates[0].score == pytest.approx(0.8)

    def test_symmetric_pair_deduped(self):
        cands = _cands((0, 1, 0.8), (1, 0, 0.6))
        fr = deduplicate_candidates(cands)
        assert fr.n_kept == 1
        assert fr.candidates[0].score == pytest.approx(0.8)

    def test_best_score_kept(self):
        cands = _cands((2, 3, 0.4), (2, 3, 0.9), (2, 3, 0.7))
        fr = deduplicate_candidates(cands)
        assert fr.candidates[0].score == pytest.approx(0.9)

    def test_empty_input(self):
        fr = deduplicate_candidates([])
        assert fr.n_kept == 0


# ─── normalize_scores ─────────────────────────────────────────────────────────

class TestNormalizeScores:
    def test_returns_list(self):
        cands = _cands((0, 1, 0.4), (1, 2, 0.8))
        result = normalize_scores(cands)
        assert isinstance(result, list)

    def test_min_maps_to_zero(self):
        cands = _cands((0, 1, 0.2), (1, 2, 0.8))
        result = normalize_scores(cands)
        scores = [c.score for c in result]
        assert min(scores) == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        cands = _cands((0, 1, 0.2), (1, 2, 0.8))
        result = normalize_scores(cands)
        scores = [c.score for c in result]
        assert max(scores) == pytest.approx(1.0)

    def test_all_equal_returns_zero(self):
        cands = _cands((0, 1, 0.5), (1, 2, 0.5))
        result = normalize_scores(cands)
        assert all(c.score == pytest.approx(0.0) for c in result)

    def test_empty_input_returns_empty(self):
        assert normalize_scores([]) == []

    def test_indices_preserved(self):
        cands = _cands((3, 7, 0.4), (1, 2, 0.8))
        result = normalize_scores(cands)
        assert result[0].idx1 == 3
        assert result[0].idx2 == 7


# ─── merge_candidate_lists ────────────────────────────────────────────────────

class TestMergeCandidateLists:
    def test_merges_all(self):
        l1 = _cands((0, 1, 0.8))
        l2 = _cands((1, 2, 0.7))
        merged = merge_candidate_lists([l1, l2])
        assert len(merged) == 2

    def test_dedup_by_default(self):
        l1 = _cands((0, 1, 0.8))
        l2 = _cands((0, 1, 0.6))
        merged = merge_candidate_lists([l1, l2])
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(0.8)

    def test_no_dedup_keeps_all(self):
        l1 = _cands((0, 1, 0.8))
        l2 = _cands((0, 1, 0.6))
        merged = merge_candidate_lists([l1, l2], dedup=False)
        assert len(merged) == 2

    def test_empty_lists(self):
        merged = merge_candidate_lists([[], []])
        assert len(merged) == 0

    def test_sorted_by_score(self):
        l1 = _cands((0, 1, 0.3))
        l2 = _cands((1, 2, 0.9))
        merged = merge_candidate_lists([l1, l2], dedup=False)
        assert merged[0].score >= merged[1].score


# ─── batch_filter ─────────────────────────────────────────────────────────────

class TestBatchFilter:
    def test_returns_list(self):
        result = batch_filter([_cands((0, 1, 0.8))], threshold=0.5)
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_filter([_cands((0, 1, 0.8)), _cands((1, 2, 0.3))],
                              threshold=0.5)
        assert len(result) == 2

    def test_threshold_applied(self):
        cands = _cands((0, 1, 0.3), (1, 2, 0.8))
        result = batch_filter([cands], threshold=0.5)
        assert result[0].n_kept == 1

    def test_top_k_applied(self):
        cands = _cands((0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7))
        result = batch_filter([cands], threshold=0.0, top_k=2)
        assert result[0].n_kept == 2

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            batch_filter([_cands((0, 1, 0.5))], threshold=1.5)

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError):
            batch_filter([_cands((0, 1, 0.5))], threshold=0.5, top_k=0)

    def test_empty_input_returns_empty(self):
        assert batch_filter([], threshold=0.5) == []
