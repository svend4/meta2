"""Тесты для puzzle_reconstruction.scoring.pair_filter."""
import pytest

from puzzle_reconstruction.scoring.pair_filter import (
    CandidatePair,
    FilterConfig,
    FilterReport,
    batch_filter,
    deduplicate_pairs,
    filter_by_inlier_count,
    filter_by_score,
    filter_pairs,
    filter_top_k,
    filter_top_k_per_fragment,
    merge_filter_results,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pair(id_a, id_b, score=1.0, n_inliers=5, rank=0):
    return CandidatePair(id_a=id_a, id_b=id_b, score=score,
                         n_inliers=n_inliers, rank=rank)


def _pairs_list():
    return [
        _pair(0, 1, score=0.9, n_inliers=10),
        _pair(0, 2, score=0.7, n_inliers=6),
        _pair(1, 2, score=0.5, n_inliers=4),
        _pair(2, 3, score=0.3, n_inliers=2),
        _pair(3, 4, score=0.1, n_inliers=1),
    ]


# ─── TestFilterConfig ─────────────────────────────────────────────────────────

class TestFilterConfig:
    def test_defaults(self):
        cfg = FilterConfig()
        assert cfg.method == "combined"
        assert cfg.min_score == 0.0
        assert cfg.min_inliers == 0
        assert cfg.max_pairs >= 1
        assert cfg.top_k_per_id >= 1

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            FilterConfig(method="unknown")

    def test_negative_min_score(self):
        with pytest.raises(ValueError):
            FilterConfig(min_score=-0.1)

    def test_negative_min_inliers(self):
        with pytest.raises(ValueError):
            FilterConfig(min_inliers=-1)

    def test_max_pairs_zero(self):
        with pytest.raises(ValueError):
            FilterConfig(max_pairs=0)

    def test_top_k_per_id_zero(self):
        with pytest.raises(ValueError):
            FilterConfig(top_k_per_id=0)

    def test_valid_method_score(self):
        cfg = FilterConfig(method="score")
        assert cfg.method == "score"

    def test_valid_method_inlier(self):
        cfg = FilterConfig(method="inlier")
        assert cfg.method == "inlier"


# ─── TestCandidatePair ────────────────────────────────────────────────────────

class TestCandidatePair:
    def test_basic_construction(self):
        p = CandidatePair(id_a=0, id_b=1, score=0.8)
        assert p.id_a == 0
        assert p.id_b == 1
        assert p.score == 0.8

    def test_defaults(self):
        p = CandidatePair(id_a=0, id_b=1, score=0.5)
        assert p.n_inliers == 0
        assert p.rank == 0

    def test_negative_id_a_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=-1, id_b=0, score=0.5)

    def test_negative_id_b_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=-1, score=0.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=-0.1)

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=0.5, n_inliers=-1)

    def test_pair_property_canonical(self):
        p = CandidatePair(id_a=5, id_b=2, score=0.5)
        assert p.pair == (2, 5)

    def test_pair_property_already_ordered(self):
        p = CandidatePair(id_a=1, id_b=3, score=0.5)
        assert p.pair == (1, 3)


# ─── TestFilterReport ─────────────────────────────────────────────────────────

class TestFilterReport:
    def test_defaults(self):
        r = FilterReport()
        assert r.n_input == 0
        assert r.n_output == 0
        assert r.n_rejected == 0

    def test_rejection_rate_zero_input(self):
        r = FilterReport(n_input=0, n_output=0, n_rejected=0)
        assert r.rejection_rate == 0.0

    def test_rejection_rate_half(self):
        r = FilterReport(n_input=10, n_output=5, n_rejected=5)
        assert abs(r.rejection_rate - 0.5) < 1e-10

    def test_rejection_rate_all(self):
        r = FilterReport(n_input=4, n_output=0, n_rejected=4)
        assert abs(r.rejection_rate - 1.0) < 1e-10

    def test_negative_n_input_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_input=-1)

    def test_negative_n_output_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_output=-1)

    def test_negative_n_rejected_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_rejected=-1)


# ─── TestFilterByScore ────────────────────────────────────────────────────────

class TestFilterByScore:
    def test_all_pass_zero_threshold(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, 0.0)
        assert len(result) == len(pairs)

    def test_none_pass_high_threshold(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, 10.0)
        assert len(result) == 0

    def test_partial_filter(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, 0.5)
        assert all(p.score >= 0.5 for p in result)

    def test_exact_threshold_included(self):
        p = _pair(0, 1, score=0.5)
        result = filter_by_score([p], 0.5)
        assert len(result) == 1

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_score(_pairs_list(), -0.1)

    def test_empty_input(self):
        assert filter_by_score([], 0.5) == []


# ─── TestFilterByInlierCount ──────────────────────────────────────────────────

class TestFilterByInlierCount:
    def test_all_pass_zero(self):
        pairs = _pairs_list()
        result = filter_by_inlier_count(pairs, 0)
        assert len(result) == len(pairs)

    def test_high_threshold_filters(self):
        pairs = _pairs_list()
        result = filter_by_inlier_count(pairs, 100)
        assert len(result) == 0

    def test_partial_filter(self):
        pairs = _pairs_list()
        result = filter_by_inlier_count(pairs, 5)
        assert all(p.n_inliers >= 5 for p in result)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_by_inlier_count(_pairs_list(), -1)

    def test_empty_input(self):
        assert filter_by_inlier_count([], 3) == []


# ─── TestFilterTopK ───────────────────────────────────────────────────────────

class TestFilterTopK:
    def test_returns_top_k(self):
        pairs = _pairs_list()
        result = filter_top_k(pairs, 3)
        assert len(result) == 3

    def test_sorted_by_score_descending(self):
        pairs = _pairs_list()
        result = filter_top_k(pairs, 3)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_list(self):
        pairs = _pairs_list()
        result = filter_top_k(pairs, 100)
        assert len(result) == len(pairs)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k(_pairs_list(), 0)

    def test_k_one(self):
        pairs = _pairs_list()
        result = filter_top_k(pairs, 1)
        assert len(result) == 1
        best_score = max(p.score for p in pairs)
        assert result[0].score == best_score

    def test_empty_input(self):
        result = filter_top_k([], 5)
        assert result == []


# ─── TestDeduplicatePairs ─────────────────────────────────────────────────────

class TestDeduplicatePairs:
    def test_no_duplicates_unchanged_count(self):
        pairs = _pairs_list()
        result = deduplicate_pairs(pairs)
        assert len(result) == len(pairs)

    def test_symmetric_duplicate_removed(self):
        p1 = _pair(0, 1, score=0.9)
        p2 = _pair(1, 0, score=0.5)  # symmetric duplicate, lower score
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1

    def test_best_score_kept(self):
        p1 = _pair(0, 1, score=0.3)
        p2 = _pair(1, 0, score=0.9)
        result = deduplicate_pairs([p1, p2])
        assert result[0].score == 0.9

    def test_empty_input(self):
        assert deduplicate_pairs([]) == []

    def test_single_pair_unchanged(self):
        p = _pair(0, 1, score=0.7)
        result = deduplicate_pairs([p])
        assert len(result) == 1

    def test_multiple_duplicates(self):
        pairs = [_pair(i % 3, (i + 1) % 3, score=float(i)) for i in range(6)]
        result = deduplicate_pairs(pairs)
        keys = [p.pair for p in result]
        assert len(keys) == len(set(keys))


# ─── TestFilterTopKPerFragment ────────────────────────────────────────────────

class TestFilterTopKPerFragment:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k_per_fragment(_pairs_list(), 0)

    def test_empty_input(self):
        result = filter_top_k_per_fragment([], 3)
        assert result == []

    def test_large_k_keeps_all(self):
        pairs = _pairs_list()
        result = filter_top_k_per_fragment(pairs, 100)
        assert set(id(p) for p in result) == set(id(p) for p in pairs)

    def test_returns_list(self):
        result = filter_top_k_per_fragment(_pairs_list(), 2)
        assert isinstance(result, list)

    def test_k_one_limits_each_fragment(self):
        # Each fragment should appear in at most k=1 pairs
        pairs = [_pair(0, 1, score=0.9), _pair(0, 2, score=0.8),
                 _pair(0, 3, score=0.7)]
        result = filter_top_k_per_fragment(pairs, 1)
        # Fragment 0 participates at most 1 time
        count_0 = sum(1 for p in result if p.id_a == 0 or p.id_b == 0)
        assert count_0 <= 1


# ─── TestFilterPairs ──────────────────────────────────────────────────────────

class TestFilterPairs:
    def test_returns_tuple(self):
        result = filter_pairs(_pairs_list())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_default_cfg(self):
        filtered, report = filter_pairs(_pairs_list())
        assert isinstance(report, FilterReport)
        assert report.n_input == len(_pairs_list())

    def test_score_method(self):
        cfg = FilterConfig(method="score", min_score=0.5)
        filtered, _ = filter_pairs(_pairs_list(), cfg)
        assert all(p.score >= 0.5 for p in filtered)

    def test_inlier_method(self):
        cfg = FilterConfig(method="inlier", min_inliers=5)
        filtered, _ = filter_pairs(_pairs_list(), cfg)
        assert all(p.n_inliers >= 5 for p in filtered)

    def test_combined_method(self):
        cfg = FilterConfig(method="combined", min_score=0.4, min_inliers=3)
        filtered, _ = filter_pairs(_pairs_list(), cfg)
        assert all(p.score >= 0.4 and p.n_inliers >= 3 for p in filtered)

    def test_report_counts_consistent(self):
        _, report = filter_pairs(_pairs_list())
        assert report.n_output + report.n_rejected == report.n_input

    def test_max_pairs_respected(self):
        cfg = FilterConfig(max_pairs=2)
        filtered, _ = filter_pairs(_pairs_list(), cfg)
        assert len(filtered) <= 2

    def test_empty_input(self):
        filtered, report = filter_pairs([])
        assert filtered == []
        assert report.n_input == 0


# ─── TestMergeFilterResults ───────────────────────────────────────────────────

class TestMergeFilterResults:
    def test_empty_list(self):
        result = merge_filter_results([])
        assert result == []

    def test_single_list(self):
        pairs = _pairs_list()
        result = merge_filter_results([pairs])
        assert len(result) == len(deduplicate_pairs(pairs))

    def test_two_disjoint_lists(self):
        list1 = [_pair(0, 1, score=0.9)]
        list2 = [_pair(2, 3, score=0.7)]
        result = merge_filter_results([list1, list2])
        assert len(result) == 2

    def test_overlapping_lists_deduplicated(self):
        list1 = [_pair(0, 1, score=0.9)]
        list2 = [_pair(1, 0, score=0.5)]  # symmetric duplicate
        result = merge_filter_results([list1, list2])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_returns_list(self):
        result = merge_filter_results([[]])
        assert isinstance(result, list)


# ─── TestBatchFilter ──────────────────────────────────────────────────────────

class TestBatchFilter:
    def test_empty_batch(self):
        result = batch_filter([])
        assert result == []

    def test_single_list(self):
        result = batch_filter([_pairs_list()])
        assert len(result) == 1
        filtered, report = result[0]
        assert isinstance(report, FilterReport)

    def test_multiple_lists(self):
        batch = [_pairs_list(), _pairs_list()[:3], _pairs_list()[:1]]
        result = batch_filter(batch)
        assert len(result) == 3

    def test_each_element_is_tuple(self):
        result = batch_filter([_pairs_list()])
        assert isinstance(result[0], tuple)

    def test_custom_cfg(self):
        cfg = FilterConfig(method="score", min_score=0.5)
        result = batch_filter([_pairs_list()], cfg)
        filtered, _ = result[0]
        assert all(p.score >= 0.5 for p in filtered)
