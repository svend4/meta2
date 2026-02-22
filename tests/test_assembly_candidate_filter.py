"""Tests for puzzle_reconstruction/assembly/candidate_filter.py"""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.candidate_filter import (
    Candidate,
    FilterResult,
    filter_by_threshold,
    filter_top_k,
    filter_by_rank,
    deduplicate_candidates,
    normalize_scores,
    merge_candidate_lists,
    batch_filter,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_cand(idx1, idx2, score):
    return Candidate(idx1=idx1, idx2=idx2, score=score)


# ─── Candidate ────────────────────────────────────────────────────────────────

class TestCandidate:
    def test_basic_creation(self):
        c = make_cand(0, 1, 0.8)
        assert c.idx1 == 0
        assert c.idx2 == 1
        assert c.score == pytest.approx(0.8)

    def test_idx1_negative_raises(self):
        with pytest.raises(ValueError):
            Candidate(idx1=-1, idx2=0, score=0.5)

    def test_idx2_negative_raises(self):
        with pytest.raises(ValueError):
            Candidate(idx1=0, idx2=-1, score=0.5)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            Candidate(idx1=0, idx2=1, score=1.1)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            Candidate(idx1=0, idx2=1, score=-0.1)

    def test_score_zero_valid(self):
        c = make_cand(0, 1, 0.0)
        assert c.score == pytest.approx(0.0)

    def test_score_one_valid(self):
        c = make_cand(0, 1, 1.0)
        assert c.score == pytest.approx(1.0)

    def test_pair_property(self):
        c = make_cand(3, 7, 0.6)
        assert c.pair == (3, 7)

    def test_params_default_empty(self):
        c = make_cand(0, 1, 0.5)
        assert c.params == {}

    def test_params_stored(self):
        c = Candidate(idx1=0, idx2=1, score=0.5, params={"key": "val"})
        assert c.params["key"] == "val"


# ─── FilterResult ─────────────────────────────────────────────────────────────

class TestFilterResult:
    def test_basic_creation(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=0)
        assert fr.n_kept == 0
        assert fr.n_removed == 0

    def test_n_kept_negative_raises(self):
        with pytest.raises(ValueError):
            FilterResult(candidates=[], n_kept=-1, n_removed=0)

    def test_n_removed_negative_raises(self):
        with pytest.raises(ValueError):
            FilterResult(candidates=[], n_kept=0, n_removed=-1)

    def test_len_returns_n_kept(self):
        fr = FilterResult(candidates=[make_cand(0, 1, 0.5)], n_kept=1, n_removed=2)
        assert len(fr) == 1

    def test_params_default_empty(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=0)
        assert fr.params == {}


# ─── filter_by_threshold ─────────────────────────────────────────────────────

class TestFilterByThreshold:
    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_threshold([], threshold=1.1)

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            filter_by_threshold([], threshold=-0.1)

    def test_returns_filter_result(self):
        cands = [make_cand(0, 1, 0.8)]
        r = filter_by_threshold(cands, threshold=0.5)
        assert isinstance(r, FilterResult)

    def test_all_pass(self):
        cands = [make_cand(0, 1, 0.9), make_cand(1, 2, 0.8)]
        r = filter_by_threshold(cands, threshold=0.5)
        assert r.n_kept == 2
        assert r.n_removed == 0

    def test_none_pass(self):
        cands = [make_cand(0, 1, 0.3), make_cand(1, 2, 0.2)]
        r = filter_by_threshold(cands, threshold=0.5)
        assert r.n_kept == 0
        assert r.n_removed == 2

    def test_sorted_desc(self):
        cands = [make_cand(0, 1, 0.5), make_cand(1, 2, 0.9), make_cand(2, 3, 0.7)]
        r = filter_by_threshold(cands, threshold=0.0)
        scores = [c.score for c in r.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_n_kept_plus_removed_equals_input(self):
        cands = [make_cand(i, i + 1, float(i) / 10.0) for i in range(10)]
        r = filter_by_threshold(cands, threshold=0.5)
        assert r.n_kept + r.n_removed == len(cands)

    def test_exact_threshold_included(self):
        cands = [make_cand(0, 1, 0.5)]
        r = filter_by_threshold(cands, threshold=0.5)
        assert r.n_kept == 1

    def test_empty_input(self):
        r = filter_by_threshold([], threshold=0.5)
        assert r.n_kept == 0
        assert r.n_removed == 0


# ─── filter_top_k ─────────────────────────────────────────────────────────────

class TestFilterTopK:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=-1)

    def test_returns_filter_result(self):
        cands = [make_cand(0, 1, 0.8)]
        r = filter_top_k(cands, k=1)
        assert isinstance(r, FilterResult)

    def test_k_clips_to_available(self):
        cands = [make_cand(0, 1, 0.8)]
        r = filter_top_k(cands, k=10)
        assert r.n_kept == 1

    def test_k_limits_results(self):
        cands = [make_cand(i, i + 1, float(i) / 10.0) for i in range(10)]
        r = filter_top_k(cands, k=3)
        assert r.n_kept == 3

    def test_sorted_desc(self):
        cands = [make_cand(i, i + 1, float(i) / 10.0) for i in range(5)]
        r = filter_top_k(cands, k=3)
        scores = [c.score for c in r.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_top_candidate_is_best(self):
        cands = [make_cand(0, 1, 0.3), make_cand(1, 2, 0.9), make_cand(2, 3, 0.6)]
        r = filter_top_k(cands, k=1)
        assert r.candidates[0].score == pytest.approx(0.9)

    def test_empty_input(self):
        r = filter_top_k([], k=5)
        assert r.n_kept == 0


# ─── filter_by_rank ───────────────────────────────────────────────────────────

class TestFilterByRank:
    def test_rank_zero_raises(self):
        with pytest.raises(ValueError):
            filter_by_rank([], rank_threshold=0.0)

    def test_rank_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_rank([], rank_threshold=1.1)

    def test_rank_one_keeps_all(self):
        cands = [make_cand(i, i + 1, float(i + 1) / 10.0) for i in range(5)]
        r = filter_by_rank(cands, rank_threshold=1.0)
        assert r.n_kept == 5

    def test_empty_input(self):
        r = filter_by_rank([], rank_threshold=0.5)
        assert r.n_kept == 0

    def test_returns_filter_result(self):
        cands = [make_cand(0, 1, 0.7)]
        r = filter_by_rank(cands, rank_threshold=0.5)
        assert isinstance(r, FilterResult)

    def test_half_rank_keeps_at_least_one(self):
        cands = [make_cand(i, i + 1, float(i + 1) / 10.0) for i in range(4)]
        r = filter_by_rank(cands, rank_threshold=0.5)
        assert r.n_kept >= 1 and r.n_kept <= 4


# ─── deduplicate_candidates ───────────────────────────────────────────────────

class TestDeduplicateCandidates:
    def test_no_duplicates_unchanged_count(self):
        cands = [make_cand(0, 1, 0.8), make_cand(1, 2, 0.7)]
        r = deduplicate_candidates(cands)
        assert r.n_kept == 2

    def test_mirror_pair_deduplicated(self):
        c1 = make_cand(0, 1, 0.8)
        c2 = make_cand(1, 0, 0.6)  # mirror of c1
        r = deduplicate_candidates([c1, c2])
        assert r.n_kept == 1

    def test_best_score_wins(self):
        c1 = make_cand(0, 1, 0.8)
        c2 = make_cand(1, 0, 0.6)
        r = deduplicate_candidates([c1, c2])
        assert r.candidates[0].score == pytest.approx(0.8)

    def test_empty_input(self):
        r = deduplicate_candidates([])
        assert r.n_kept == 0

    def test_n_removed_counts_duplicates(self):
        c1 = make_cand(0, 1, 0.8)
        c2 = make_cand(1, 0, 0.6)
        r = deduplicate_candidates([c1, c2])
        assert r.n_removed == 1

    def test_returns_filter_result(self):
        r = deduplicate_candidates([make_cand(0, 1, 0.5)])
        assert isinstance(r, FilterResult)

    def test_sorted_desc(self):
        cands = [make_cand(0, 1, 0.3), make_cand(2, 3, 0.9), make_cand(4, 5, 0.6)]
        r = deduplicate_candidates(cands)
        scores = [c.score for c in r.candidates]
        assert scores == sorted(scores, reverse=True)


# ─── normalize_scores ─────────────────────────────────────────────────────────

class TestNormalizeScores:
    def test_empty_returns_empty(self):
        result = normalize_scores([])
        assert result == []

    def test_returns_list_of_candidates(self):
        cands = [make_cand(0, 1, 0.3), make_cand(1, 2, 0.9)]
        result = normalize_scores(cands)
        assert isinstance(result, list)
        assert all(isinstance(c, Candidate) for c in result)

    def test_range_01(self):
        cands = [make_cand(0, 1, 0.3), make_cand(1, 2, 0.9), make_cand(2, 3, 0.6)]
        result = normalize_scores(cands)
        scores = [c.score for c in result]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)

    def test_all_equal_scores_returns_zero(self):
        cands = [make_cand(0, 1, 0.5), make_cand(1, 2, 0.5)]
        result = normalize_scores(cands)
        assert all(c.score == pytest.approx(0.0) for c in result)

    def test_length_preserved(self):
        cands = [make_cand(i, i + 1, float(i) / 5.0) for i in range(5)]
        result = normalize_scores(cands)
        assert len(result) == 5

    def test_indices_preserved(self):
        cands = [make_cand(3, 7, 0.4), make_cand(5, 9, 0.8)]
        result = normalize_scores(cands)
        assert result[0].idx1 == 3
        assert result[0].idx2 == 7


# ─── merge_candidate_lists ────────────────────────────────────────────────────

class TestMergeCandidateLists:
    def test_empty_lists(self):
        result = merge_candidate_lists([])
        assert result == []

    def test_single_list_unchanged(self):
        cands = [make_cand(0, 1, 0.8), make_cand(1, 2, 0.5)]
        result = merge_candidate_lists([cands])
        assert len(result) == 2

    def test_merges_two_lists(self):
        l1 = [make_cand(0, 1, 0.8)]
        l2 = [make_cand(2, 3, 0.6)]
        result = merge_candidate_lists([l1, l2], dedup=False)
        assert len(result) == 2

    def test_dedup_removes_mirror(self):
        l1 = [make_cand(0, 1, 0.8)]
        l2 = [make_cand(1, 0, 0.5)]
        result = merge_candidate_lists([l1, l2], dedup=True)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)

    def test_sorted_desc(self):
        l1 = [make_cand(0, 1, 0.3)]
        l2 = [make_cand(2, 3, 0.9)]
        l3 = [make_cand(4, 5, 0.6)]
        result = merge_candidate_lists([l1, l2, l3], dedup=False)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)


# ─── batch_filter ─────────────────────────────────────────────────────────────

class TestBatchFilter:
    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            batch_filter([], threshold=1.5)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            batch_filter([[]], threshold=0.5, top_k=0)

    def test_returns_list_of_filter_results(self):
        lists = [[make_cand(0, 1, 0.8)], [make_cand(1, 2, 0.4)]]
        results = batch_filter(lists, threshold=0.5)
        assert isinstance(results, list)
        assert all(isinstance(r, FilterResult) for r in results)

    def test_length_equals_input(self):
        lists = [[make_cand(0, 1, 0.8)], [make_cand(1, 2, 0.9)], []]
        results = batch_filter(lists)
        assert len(results) == 3

    def test_empty_input(self):
        results = batch_filter([])
        assert results == []

    def test_top_k_applied(self):
        cands = [make_cand(i, i + 1, 1.0) for i in range(10)]
        results = batch_filter([cands], threshold=0.0, top_k=3)
        assert results[0].n_kept <= 3

    def test_threshold_applied(self):
        cands = [make_cand(0, 1, 0.3), make_cand(1, 2, 0.9)]
        results = batch_filter([cands], threshold=0.5)
        assert results[0].n_kept == 1
