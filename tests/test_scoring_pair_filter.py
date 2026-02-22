"""Тесты для puzzle_reconstruction/scoring/pair_filter.py."""
import pytest
import numpy as np

from puzzle_reconstruction.scoring.pair_filter import (
    FilterConfig,
    CandidatePair,
    FilterReport,
    filter_by_score,
    filter_by_inlier_count,
    filter_top_k,
    deduplicate_pairs,
    filter_top_k_per_fragment,
    filter_pairs,
    merge_filter_results,
    batch_filter,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_pair(a, b, score=0.5, n_inliers=10, rank=0):
    return CandidatePair(id_a=a, id_b=b, score=score,
                         n_inliers=n_inliers, rank=rank)


def make_pairs(n=5, base_score=0.5):
    return [make_pair(i, i+1, score=base_score + i*0.1) for i in range(n)]


# ─── FilterConfig ─────────────────────────────────────────────────────────────

class TestFilterConfig:
    def test_defaults(self):
        cfg = FilterConfig()
        assert cfg.method == "combined"
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.min_inliers == 0
        assert cfg.max_pairs == 100
        assert cfg.top_k_per_id == 5

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            FilterConfig(method="invalid")

    def test_valid_methods(self):
        for m in ("score", "inlier", "rank", "combined"):
            cfg = FilterConfig(method=m)
            assert cfg.method == m

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            FilterConfig(min_score=-0.1)

    def test_negative_min_inliers_raises(self):
        with pytest.raises(ValueError, match="min_inliers"):
            FilterConfig(min_inliers=-1)

    def test_max_pairs_zero_raises(self):
        with pytest.raises(ValueError, match="max_pairs"):
            FilterConfig(max_pairs=0)

    def test_top_k_per_id_zero_raises(self):
        with pytest.raises(ValueError, match="top_k_per_id"):
            FilterConfig(top_k_per_id=0)

    def test_zero_min_score_valid(self):
        cfg = FilterConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_zero_min_inliers_valid(self):
        cfg = FilterConfig(min_inliers=0)
        assert cfg.min_inliers == 0


# ─── CandidatePair ────────────────────────────────────────────────────────────

class TestCandidatePair:
    def test_creation(self):
        p = CandidatePair(id_a=0, id_b=1, score=0.7, n_inliers=15, rank=2)
        assert p.id_a == 0
        assert p.id_b == 1
        assert p.score == pytest.approx(0.7)
        assert p.n_inliers == 15
        assert p.rank == 2

    def test_negative_id_a_raises(self):
        with pytest.raises(ValueError, match="id_a"):
            CandidatePair(id_a=-1, id_b=0, score=0.5)

    def test_negative_id_b_raises(self):
        with pytest.raises(ValueError, match="id_b"):
            CandidatePair(id_a=0, id_b=-1, score=0.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="score"):
            CandidatePair(id_a=0, id_b=1, score=-0.1)

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError, match="n_inliers"):
            CandidatePair(id_a=0, id_b=1, score=0.5, n_inliers=-1)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError, match="rank"):
            CandidatePair(id_a=0, id_b=1, score=0.5, rank=-1)

    def test_pair_property_ordered(self):
        p = CandidatePair(id_a=5, id_b=2, score=0.5)
        assert p.pair == (2, 5)

    def test_pair_property_already_ordered(self):
        p = CandidatePair(id_a=1, id_b=3, score=0.5)
        assert p.pair == (1, 3)

    def test_zero_score_valid(self):
        p = CandidatePair(id_a=0, id_b=1, score=0.0)
        assert p.score == 0.0

    def test_zero_n_inliers_valid(self):
        p = CandidatePair(id_a=0, id_b=1, score=0.5, n_inliers=0)
        assert p.n_inliers == 0


# ─── FilterReport ─────────────────────────────────────────────────────────────

class TestFilterReport:
    def test_creation(self):
        r = FilterReport(n_input=10, n_output=6, n_rejected=4, method="score")
        assert r.n_input == 10
        assert r.n_output == 6
        assert r.n_rejected == 4
        assert r.method == "score"

    def test_negative_n_input_raises(self):
        with pytest.raises(ValueError, match="n_input"):
            FilterReport(n_input=-1)

    def test_negative_n_output_raises(self):
        with pytest.raises(ValueError, match="n_output"):
            FilterReport(n_output=-1)

    def test_negative_n_rejected_raises(self):
        with pytest.raises(ValueError, match="n_rejected"):
            FilterReport(n_rejected=-1)

    def test_rejection_rate_zero_input(self):
        r = FilterReport(n_input=0, n_output=0, n_rejected=0)
        assert r.rejection_rate == pytest.approx(0.0)

    def test_rejection_rate(self):
        r = FilterReport(n_input=10, n_output=6, n_rejected=4)
        assert r.rejection_rate == pytest.approx(0.4)

    def test_rejection_rate_all_rejected(self):
        r = FilterReport(n_input=5, n_output=0, n_rejected=5)
        assert r.rejection_rate == pytest.approx(1.0)


# ─── filter_by_score ──────────────────────────────────────────────────────────

class TestFilterByScore:
    def test_returns_list(self):
        pairs = make_pairs()
        result = filter_by_score(pairs)
        assert isinstance(result, list)

    def test_all_pass_at_zero(self):
        pairs = make_pairs(3, base_score=0.1)
        result = filter_by_score(pairs, min_score=0.0)
        assert len(result) == 3

    def test_filters_below_threshold(self):
        pairs = [make_pair(0, 1, score=0.3), make_pair(1, 2, score=0.8)]
        result = filter_by_score(pairs, min_score=0.5)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([], min_score=-0.1)

    def test_empty_list_returns_empty(self):
        result = filter_by_score([], min_score=0.5)
        assert result == []

    def test_boundary_score_included(self):
        pairs = [make_pair(0, 1, score=0.5)]
        result = filter_by_score(pairs, min_score=0.5)
        assert len(result) == 1


# ─── filter_by_inlier_count ───────────────────────────────────────────────────

class TestFilterByInlierCount:
    def test_returns_list(self):
        pairs = make_pairs()
        result = filter_by_inlier_count(pairs)
        assert isinstance(result, list)

    def test_filters_below_min(self):
        pairs = [make_pair(0, 1, n_inliers=3), make_pair(1, 2, n_inliers=10)]
        result = filter_by_inlier_count(pairs, min_inliers=5)
        assert len(result) == 1
        assert result[0].n_inliers == 10

    def test_negative_min_inliers_raises(self):
        with pytest.raises(ValueError):
            filter_by_inlier_count([], min_inliers=-1)

    def test_zero_min_inliers_all_pass(self):
        pairs = make_pairs(3)
        result = filter_by_inlier_count(pairs, min_inliers=0)
        assert len(result) == 3

    def test_empty_list(self):
        result = filter_by_inlier_count([])
        assert result == []


# ─── filter_top_k ─────────────────────────────────────────────────────────────

class TestFilterTopK:
    def test_returns_top_k(self):
        pairs = [make_pair(i, i+1, score=float(i)) for i in range(10)]
        result = filter_top_k(pairs, k=3)
        assert len(result) == 3

    def test_sorted_descending(self):
        pairs = [make_pair(i, i+1, score=float(i)*0.1) for i in range(5)]
        result = filter_top_k(pairs, k=5)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_less_than_1_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=0)

    def test_k_greater_than_list_returns_all(self):
        pairs = make_pairs(3)
        result = filter_top_k(pairs, k=10)
        assert len(result) == 3

    def test_k_1_returns_highest(self):
        pairs = [make_pair(0, 1, score=0.3), make_pair(1, 2, score=0.9)]
        result = filter_top_k(pairs, k=1)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)


# ─── deduplicate_pairs ────────────────────────────────────────────────────────

class TestDeduplicatePairs:
    def test_removes_reverse_duplicate(self):
        p1 = make_pair(0, 1, score=0.8)
        p2 = make_pair(1, 0, score=0.5)  # symmetric duplicate
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1

    def test_keeps_highest_score(self):
        p1 = make_pair(0, 1, score=0.8)
        p2 = make_pair(1, 0, score=0.5)
        result = deduplicate_pairs([p1, p2])
        assert result[0].score == pytest.approx(0.8)

    def test_unique_pairs_unchanged(self):
        pairs = [make_pair(0, 1), make_pair(1, 2), make_pair(0, 2)]
        result = deduplicate_pairs(pairs)
        assert len(result) == 3

    def test_empty_returns_empty(self):
        result = deduplicate_pairs([])
        assert result == []

    def test_same_pair_twice_keeps_one(self):
        p1 = make_pair(0, 1, score=0.5)
        p2 = make_pair(0, 1, score=0.7)
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)


# ─── filter_top_k_per_fragment ────────────────────────────────────────────────

class TestFilterTopKPerFragment:
    def test_k_less_than_1_raises(self):
        with pytest.raises(ValueError):
            filter_top_k_per_fragment([], k=0)

    def test_reduces_output_size(self):
        # Fragment 0 and 1 share many pairs; with k=1 output is subset of input
        pairs = [
            make_pair(0, 1, score=0.9),
            make_pair(0, 2, score=0.7),
            make_pair(1, 2, score=0.5),
            make_pair(0, 3, score=0.3),
            make_pair(1, 3, score=0.2),
        ]
        result = filter_top_k_per_fragment(pairs, k=1)
        # Result must be a subset
        assert len(result) <= len(pairs)

    def test_empty_returns_empty(self):
        result = filter_top_k_per_fragment([], k=3)
        assert result == []

    def test_returns_list(self):
        pairs = make_pairs(3)
        result = filter_top_k_per_fragment(pairs, k=2)
        assert isinstance(result, list)


# ─── filter_pairs ─────────────────────────────────────────────────────────────

class TestFilterPairs:
    def test_returns_tuple(self):
        pairs = make_pairs(5)
        result = filter_pairs(pairs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_list_and_report(self):
        pairs = make_pairs(5)
        filtered, report = filter_pairs(pairs)
        assert isinstance(filtered, list)
        assert isinstance(report, FilterReport)

    def test_report_n_input(self):
        pairs = make_pairs(5)
        _, report = filter_pairs(pairs)
        assert report.n_input == 5

    def test_report_n_output_consistent(self):
        pairs = make_pairs(5)
        filtered, report = filter_pairs(pairs)
        assert report.n_output == len(filtered)

    def test_report_n_rejected(self):
        pairs = make_pairs(5)
        filtered, report = filter_pairs(pairs)
        assert report.n_rejected == 5 - len(filtered)

    def test_score_filter_applied(self):
        pairs = [make_pair(0, 1, score=0.1), make_pair(1, 2, score=0.9)]
        cfg = FilterConfig(method="score", min_score=0.5)
        filtered, _ = filter_pairs(pairs, cfg=cfg)
        for p in filtered:
            assert p.score >= 0.5

    def test_empty_input_returns_empty(self):
        filtered, report = filter_pairs([])
        assert filtered == []
        assert report.n_input == 0

    def test_none_cfg_uses_defaults(self):
        pairs = make_pairs(3)
        filtered, _ = filter_pairs(pairs, cfg=None)
        assert isinstance(filtered, list)


# ─── merge_filter_results ─────────────────────────────────────────────────────

class TestMergeFilterResults:
    def test_empty_lists_returns_empty(self):
        result = merge_filter_results([])
        assert result == []

    def test_single_list(self):
        pairs = make_pairs(3)
        result = merge_filter_results([pairs])
        assert len(result) == 3

    def test_deduplicates_across_lists(self):
        p1 = [make_pair(0, 1, score=0.8)]
        p2 = [make_pair(1, 0, score=0.5)]  # duplicate
        result = merge_filter_results([p1, p2])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)

    def test_merges_two_lists(self):
        p1 = [make_pair(0, 1)]
        p2 = [make_pair(2, 3)]
        result = merge_filter_results([p1, p2])
        assert len(result) == 2


# ─── batch_filter ─────────────────────────────────────────────────────────────

class TestBatchFilter:
    def test_empty_batch_returns_empty(self):
        result = batch_filter([])
        assert result == []

    def test_length_matches_input(self):
        batch = [make_pairs(3) for _ in range(4)]
        result = batch_filter(batch)
        assert len(result) == 4

    def test_each_result_is_tuple(self):
        batch = [make_pairs(2), make_pairs(3)]
        result = batch_filter(batch)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_reports_correct_n_input(self):
        batch = [make_pairs(3), make_pairs(5)]
        result = batch_filter(batch)
        assert result[0][1].n_input == 3
        assert result[1][1].n_input == 5
