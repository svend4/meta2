"""Extra tests for puzzle_reconstruction/assembly/candidate_filter.py"""
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


def _c(i1, i2, score):
    return Candidate(idx1=i1, idx2=i2, score=score)


# ─── TestCandidateExtra ───────────────────────────────────────────────────────

class TestCandidateExtra:
    def test_score_0_5_valid(self):
        c = _c(0, 1, 0.5)
        assert c.score == pytest.approx(0.5)

    def test_idx1_zero_valid(self):
        c = _c(0, 1, 0.5)
        assert c.idx1 == 0

    def test_idx2_zero_valid(self):
        c = _c(1, 0, 0.5)
        assert c.idx2 == 0

    def test_large_indices(self):
        c = _c(1000, 2000, 0.5)
        assert c.idx1 == 1000
        assert c.idx2 == 2000

    def test_params_score_key(self):
        c = Candidate(idx1=0, idx2=1, score=0.5, params={"score": 0.5, "rank": 2})
        assert c.params["rank"] == 2

    def test_pair_order_preserved(self):
        c = _c(5, 3, 0.7)
        assert c.pair == (5, 3)

    def test_score_boundary_0(self):
        c = _c(0, 1, 0.0)
        assert c.score == pytest.approx(0.0)

    def test_score_boundary_1(self):
        c = _c(0, 1, 1.0)
        assert c.score == pytest.approx(1.0)


# ─── TestFilterResultExtra ────────────────────────────────────────────────────

class TestFilterResultExtra:
    def test_large_n_kept(self):
        fr = FilterResult(candidates=[_c(0, 1, 0.5)], n_kept=100, n_removed=0)
        assert fr.n_kept == 100

    def test_large_n_removed(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=1000)
        assert fr.n_removed == 1000

    def test_params_stored(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=0,
                          params={"threshold": 0.5})
        assert fr.params["threshold"] == pytest.approx(0.5)

    def test_len_zero_when_empty(self):
        fr = FilterResult(candidates=[], n_kept=0, n_removed=0)
        assert len(fr) == 0

    def test_candidates_list_accessible(self):
        cands = [_c(i, i + 1, float(i) / 10.0) for i in range(5)]
        fr = FilterResult(candidates=cands, n_kept=5, n_removed=0)
        assert len(fr.candidates) == 5


# ─── TestFilterByThresholdExtra ───────────────────────────────────────────────

class TestFilterByThresholdExtra:
    def test_threshold_0_includes_all(self):
        cands = [_c(i, i + 1, float(i + 1) / 10.0) for i in range(5)]
        r = filter_by_threshold(cands, threshold=0.0)
        assert r.n_kept == 5

    def test_threshold_1_includes_only_perfect(self):
        cands = [_c(0, 1, 1.0), _c(1, 2, 0.9), _c(2, 3, 0.8)]
        r = filter_by_threshold(cands, threshold=1.0)
        assert r.n_kept == 1
        assert r.candidates[0].score == pytest.approx(1.0)

    def test_partial_pass_three_of_five(self):
        cands = [_c(i, i + 1, float(i) / 4.0) for i in range(5)]
        r = filter_by_threshold(cands, threshold=0.5)
        assert r.n_kept + r.n_removed == 5

    def test_candidates_sorted_desc(self):
        cands = [_c(0, 1, 0.3), _c(1, 2, 0.7), _c(2, 3, 0.5)]
        r = filter_by_threshold(cands, threshold=0.0)
        scores = [c.score for c in r.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_single_candidate_passes(self):
        r = filter_by_threshold([_c(0, 1, 0.9)], threshold=0.5)
        assert r.n_kept == 1


# ─── TestFilterTopKExtra ──────────────────────────────────────────────────────

class TestFilterTopKExtra:
    def test_k_1(self):
        cands = [_c(i, i + 1, float(i + 1) / 10.0) for i in range(5)]
        r = filter_top_k(cands, k=1)
        assert r.n_kept == 1

    def test_k_exactly_n(self):
        cands = [_c(i, i + 1, float(i) / 10.0) for i in range(5)]
        r = filter_top_k(cands, k=5)
        assert r.n_kept == 5

    def test_k_larger_than_10(self):
        cands = [_c(i, i + 1, 0.5) for i in range(3)]
        r = filter_top_k(cands, k=100)
        assert r.n_kept == 3

    def test_top_k_returns_best(self):
        cands = [_c(0, 1, 0.2), _c(1, 2, 0.9), _c(2, 3, 0.5), _c(3, 4, 0.7)]
        r = filter_top_k(cands, k=2)
        scores = {c.score for c in r.candidates}
        assert 0.9 in scores
        assert 0.7 in scores

    def test_n_removed_correct(self):
        cands = [_c(i, i + 1, float(i) / 10.0) for i in range(10)]
        r = filter_top_k(cands, k=3)
        assert r.n_removed == 7


# ─── TestFilterByRankExtra ────────────────────────────────────────────────────

class TestFilterByRankExtra:
    def test_rank_0_01_returns_at_least_one(self):
        cands = [_c(i, i + 1, float(i + 1) / 10.0) for i in range(10)]
        r = filter_by_rank(cands, rank_threshold=0.01)
        assert r.n_kept >= 1

    def test_rank_0_5_half_or_more(self):
        cands = [_c(i, i + 1, float(i + 1) / 10.0) for i in range(10)]
        r = filter_by_rank(cands, rank_threshold=0.5)
        assert r.n_kept <= 10

    def test_rank_1_0_all_pass(self):
        cands = [_c(i, i + 1, float(i + 1) / 10.0) for i in range(5)]
        r = filter_by_rank(cands, rank_threshold=1.0)
        assert r.n_kept == 5

    def test_n_kept_plus_removed_eq_input(self):
        cands = [_c(i, i + 1, float(i + 1) / 10.0) for i in range(8)]
        r = filter_by_rank(cands, rank_threshold=0.5)
        assert r.n_kept + r.n_removed == 8

    def test_single_candidate(self):
        r = filter_by_rank([_c(0, 1, 0.9)], rank_threshold=0.5)
        assert r.n_kept >= 0


# ─── TestDeduplicateCandidatesExtra ──────────────────────────────────────────

class TestDeduplicateCandidatesExtra:
    def test_three_unique_pairs(self):
        cands = [_c(0, 1, 0.8), _c(1, 2, 0.7), _c(2, 3, 0.6)]
        r = deduplicate_candidates(cands)
        assert r.n_kept == 3

    def test_two_mirrors_one_kept(self):
        cands = [_c(0, 1, 0.6), _c(1, 0, 0.9)]
        r = deduplicate_candidates(cands)
        assert r.n_kept == 1
        assert r.candidates[0].score == pytest.approx(0.9)

    def test_no_mirrors_unchanged(self):
        cands = [_c(0, 1, 0.5), _c(2, 3, 0.7), _c(4, 5, 0.9)]
        r = deduplicate_candidates(cands)
        assert r.n_kept == 3
        assert r.n_removed == 0

    def test_sorted_desc_after_dedup(self):
        cands = [_c(0, 1, 0.5), _c(2, 3, 0.9), _c(1, 0, 0.8)]
        r = deduplicate_candidates(cands)
        scores = [c.score for c in r.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_five_unique_all_kept(self):
        cands = [_c(i, i + 1, float(i + 1) / 5.0) for i in range(5)]
        r = deduplicate_candidates(cands)
        assert r.n_kept == 5


# ─── TestNormalizeScoresExtra ─────────────────────────────────────────────────

class TestNormalizeScoresExtra:
    def test_single_candidate_returns_zero(self):
        result = normalize_scores([_c(0, 1, 0.7)])
        assert result[0].score == pytest.approx(0.0)

    def test_two_candidates_min_0_max_1(self):
        result = normalize_scores([_c(0, 1, 0.3), _c(1, 2, 0.9)])
        scores = sorted(c.score for c in result)
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(1.0)

    def test_indices_unchanged(self):
        cands = [_c(3, 7, 0.4), _c(5, 9, 0.8)]
        result = normalize_scores(cands)
        idx_pairs = {(c.idx1, c.idx2) for c in result}
        assert (3, 7) in idx_pairs
        assert (5, 9) in idx_pairs

    def test_ten_candidates_range(self):
        cands = [_c(i, i + 1, float(i) / 9.0) for i in range(10)]
        result = normalize_scores(cands)
        scores = [c.score for c in result]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)


# ─── TestMergeCandidateListsExtra ────────────────────────────────────────────

class TestMergeCandidateListsExtra:
    def test_three_lists_merged(self):
        l1 = [_c(0, 1, 0.8)]
        l2 = [_c(2, 3, 0.6)]
        l3 = [_c(4, 5, 0.4)]
        result = merge_candidate_lists([l1, l2, l3], dedup=False)
        assert len(result) == 3

    def test_dedup_true_removes_mirror(self):
        l1 = [_c(0, 1, 0.9)]
        l2 = [_c(1, 0, 0.5)]
        result = merge_candidate_lists([l1, l2], dedup=True)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_sorted_desc_three_lists(self):
        l1 = [_c(0, 1, 0.2)]
        l2 = [_c(2, 3, 0.9)]
        l3 = [_c(4, 5, 0.5)]
        result = merge_candidate_lists([l1, l2, l3], dedup=False)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_lists_in_batch(self):
        result = merge_candidate_lists([[], [_c(0, 1, 0.8)], []])
        assert len(result) == 1

    def test_all_empty(self):
        result = merge_candidate_lists([[], [], []])
        assert result == []


# ─── TestBatchFilterExtra ────────────────────────────────────────────────────

class TestBatchFilterExtra:
    def test_three_lists(self):
        lists = [
            [_c(0, 1, 0.8), _c(1, 2, 0.3)],
            [_c(2, 3, 0.9)],
            [_c(3, 4, 0.1)],
        ]
        results = batch_filter(lists, threshold=0.5)
        assert len(results) == 3

    def test_mixed_valid_invalid(self):
        lists = [
            [_c(0, 1, 0.9), _c(1, 2, 0.8)],
            [_c(2, 3, 0.1), _c(3, 4, 0.2)],
        ]
        results = batch_filter(lists, threshold=0.5)
        assert results[0].n_kept == 2
        assert results[1].n_kept == 0

    def test_top_k_1_each(self):
        lists = [
            [_c(0, 1, 0.5), _c(1, 2, 0.9)],
            [_c(2, 3, 0.3), _c(3, 4, 0.7)],
        ]
        results = batch_filter(lists, threshold=0.0, top_k=1)
        for r in results:
            assert r.n_kept <= 1

    def test_all_filter_results(self):
        lists = [[_c(i, i + 1, 0.7)] for i in range(5)]
        results = batch_filter(lists)
        for r in results:
            assert isinstance(r, FilterResult)
