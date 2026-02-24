"""Extra tests for puzzle_reconstruction/scoring/pair_filter.py."""
from __future__ import annotations

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

def _pair(a: int = 0, b: int = 1, score: float = 0.8,
          n_inliers: int = 10, rank: int = 0) -> CandidatePair:
    return CandidatePair(id_a=a, id_b=b, score=score,
                         n_inliers=n_inliers, rank=rank)


def _pairs_list() -> list:
    return [
        _pair(0, 1, score=0.9, n_inliers=20),
        _pair(1, 2, score=0.7, n_inliers=15),
        _pair(0, 2, score=0.5, n_inliers=5),
        _pair(2, 3, score=0.3, n_inliers=2),
    ]


# ─── FilterConfig (extra) ─────────────────────────────────────────────────────

class TestFilterConfigExtra:
    def test_default_method(self):
        assert FilterConfig().method == "combined"

    def test_default_min_score(self):
        assert FilterConfig().min_score == pytest.approx(0.0)

    def test_default_min_inliers(self):
        assert FilterConfig().min_inliers == 0

    def test_default_max_pairs(self):
        assert FilterConfig().max_pairs == 100

    def test_default_top_k_per_id(self):
        assert FilterConfig().top_k_per_id == 5

    def test_custom_method_score(self):
        cfg = FilterConfig(method="score")
        assert cfg.method == "score"

    def test_custom_method_inlier(self):
        cfg = FilterConfig(method="inlier")
        assert cfg.method == "inlier"

    def test_custom_method_rank(self):
        cfg = FilterConfig(method="rank")
        assert cfg.method == "rank"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FilterConfig(method="magic")

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            FilterConfig(min_score=-0.1)

    def test_negative_min_inliers_raises(self):
        with pytest.raises(ValueError):
            FilterConfig(min_inliers=-1)

    def test_max_pairs_zero_raises(self):
        with pytest.raises(ValueError):
            FilterConfig(max_pairs=0)

    def test_top_k_per_id_zero_raises(self):
        with pytest.raises(ValueError):
            FilterConfig(top_k_per_id=0)

    def test_custom_values_stored(self):
        cfg = FilterConfig(min_score=0.5, min_inliers=10, max_pairs=50)
        assert cfg.min_score == pytest.approx(0.5)
        assert cfg.min_inliers == 10
        assert cfg.max_pairs == 50


# ─── CandidatePair (extra) ────────────────────────────────────────────────────

class TestCandidatePairExtra:
    def test_ids_stored(self):
        p = _pair(a=3, b=7)
        assert p.id_a == 3
        assert p.id_b == 7

    def test_score_stored(self):
        p = _pair(score=0.65)
        assert p.score == pytest.approx(0.65)

    def test_n_inliers_stored(self):
        p = _pair(n_inliers=25)
        assert p.n_inliers == 25

    def test_rank_stored(self):
        p = _pair(rank=3)
        assert p.rank == 3

    def test_negative_id_a_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=-1, id_b=0, score=0.5)

    def test_negative_id_b_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=-1, score=0.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=-0.1)

    def test_negative_inliers_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=0.5, n_inliers=-1)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=0.5, rank=-1)

    def test_pair_canonical_ordered(self):
        p = CandidatePair(id_a=5, id_b=2, score=0.7)
        assert p.pair == (2, 5)

    def test_pair_already_ordered(self):
        p = CandidatePair(id_a=1, id_b=9, score=0.7)
        assert p.pair == (1, 9)

    def test_pair_equal_ids(self):
        p = CandidatePair(id_a=3, id_b=3, score=0.5)
        assert p.pair == (3, 3)


# ─── FilterReport (extra) ─────────────────────────────────────────────────────

class TestFilterReportExtra:
    def test_default_n_input(self):
        assert FilterReport().n_input == 0

    def test_default_n_output(self):
        assert FilterReport().n_output == 0

    def test_default_n_rejected(self):
        assert FilterReport().n_rejected == 0

    def test_default_method(self):
        assert FilterReport().method == "combined"

    def test_negative_n_input_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_input=-1)

    def test_negative_n_output_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_output=-1)

    def test_negative_n_rejected_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_rejected=-1)

    def test_rejection_rate_zero_input(self):
        r = FilterReport(n_input=0, n_rejected=0)
        assert r.rejection_rate == pytest.approx(0.0)

    def test_rejection_rate_half(self):
        r = FilterReport(n_input=10, n_output=5, n_rejected=5)
        assert r.rejection_rate == pytest.approx(0.5)

    def test_rejection_rate_all(self):
        r = FilterReport(n_input=8, n_output=0, n_rejected=8)
        assert r.rejection_rate == pytest.approx(1.0)

    def test_rejection_rate_none(self):
        r = FilterReport(n_input=5, n_output=5, n_rejected=0)
        assert r.rejection_rate == pytest.approx(0.0)


# ─── filter_by_score (extra) ──────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_score([], 0.0), list)

    def test_empty_input(self):
        assert filter_by_score([], 0.5) == []

    def test_all_pass(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, min_score=0.0)
        assert len(result) == len(pairs)

    def test_all_filtered(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, min_score=1.5)
        assert result == []

    def test_partial_filter(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, min_score=0.6)
        assert all(p.score >= 0.6 for p in result)

    def test_exact_threshold_passes(self):
        p = _pair(score=0.5)
        result = filter_by_score([p], min_score=0.5)
        assert len(result) == 1

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([], min_score=-0.1)

    def test_order_preserved(self):
        pairs = _pairs_list()
        result = filter_by_score(pairs, min_score=0.0)
        assert [p.id_a for p in result] == [p.id_a for p in pairs]


# ─── filter_by_inlier_count (extra) ───────────────────────────────────────────

class TestFilterByInlierCountExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_inlier_count([], 0), list)

    def test_empty_input(self):
        assert filter_by_inlier_count([], 5) == []

    def test_all_pass(self):
        pairs = _pairs_list()
        result = filter_by_inlier_count(pairs, min_inliers=0)
        assert len(result) == len(pairs)

    def test_all_filtered(self):
        pairs = _pairs_list()
        result = filter_by_inlier_count(pairs, min_inliers=100)
        assert result == []

    def test_partial_filter(self):
        pairs = _pairs_list()
        result = filter_by_inlier_count(pairs, min_inliers=10)
        assert all(p.n_inliers >= 10 for p in result)

    def test_exact_threshold_passes(self):
        p = _pair(n_inliers=10)
        result = filter_by_inlier_count([p], min_inliers=10)
        assert len(result) == 1

    def test_negative_min_inliers_raises(self):
        with pytest.raises(ValueError):
            filter_by_inlier_count([], min_inliers=-1)


# ─── filter_top_k (extra) ─────────────────────────────────────────────────────

class TestFilterTopKExtra:
    def test_returns_list(self):
        assert isinstance(filter_top_k(_pairs_list(), 2), list)

    def test_returns_at_most_k(self):
        result = filter_top_k(_pairs_list(), k=2)
        assert len(result) <= 2

    def test_sorted_by_score_descending(self):
        pairs = _pairs_list()
        result = filter_top_k(pairs, k=3)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k(_pairs_list(), k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            filter_top_k(_pairs_list(), k=-1)

    def test_k_larger_than_list(self):
        pairs = _pairs_list()
        result = filter_top_k(pairs, k=100)
        assert len(result) == len(pairs)

    def test_empty_list(self):
        assert filter_top_k([], k=5) == []


# ─── deduplicate_pairs (extra) ────────────────────────────────────────────────

class TestDeduplicatePairsExtra:
    def test_returns_list(self):
        assert isinstance(deduplicate_pairs([]), list)

    def test_empty_input(self):
        assert deduplicate_pairs([]) == []

    def test_no_duplicates_unchanged(self):
        pairs = _pairs_list()
        result = deduplicate_pairs(pairs)
        assert len(result) == len(pairs)

    def test_symmetric_duplicate_removed(self):
        p1 = CandidatePair(id_a=0, id_b=1, score=0.8)
        p2 = CandidatePair(id_a=1, id_b=0, score=0.6)  # symmetric duplicate
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1

    def test_keeps_higher_score(self):
        p1 = CandidatePair(id_a=0, id_b=1, score=0.8)
        p2 = CandidatePair(id_a=1, id_b=0, score=0.6)
        result = deduplicate_pairs([p1, p2])
        assert result[0].score == pytest.approx(0.8)

    def test_exact_duplicate_keeps_one(self):
        p = _pair(0, 1, score=0.9)
        result = deduplicate_pairs([p, p])
        assert len(result) == 1

    def test_different_pairs_kept(self):
        p1 = _pair(0, 1, score=0.8)
        p2 = _pair(1, 2, score=0.7)
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 2


# ─── filter_top_k_per_fragment (extra) ────────────────────────────────────────

class TestFilterTopKPerFragmentExtra:
    def test_returns_list(self):
        assert isinstance(filter_top_k_per_fragment([], k=3), list)

    def test_empty_input(self):
        assert filter_top_k_per_fragment([], k=3) == []

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k_per_fragment([], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            filter_top_k_per_fragment([], k=-1)

    def test_single_pair_kept(self):
        result = filter_top_k_per_fragment([_pair()], k=1)
        assert len(result) == 1

    def test_many_pairs_filters_low_scored(self):
        # With k=1, only the top pair per fragment is "allowed"
        pairs = [
            _pair(0, 1, score=0.9),
            _pair(0, 2, score=0.1),  # low score — only allowed if top-1 for frag 2
            _pair(2, 3, score=0.1),  # low score — allowed if top-1 for frag 2 or 3
        ]
        result = filter_top_k_per_fragment(pairs, k=1)
        # (0,1) at 0.9 is top-1 for both frag 0 and frag 1 → always kept
        assert any(p.id_a == 0 and p.id_b == 1 for p in result)

    def test_k_larger_than_group_keeps_all(self):
        pairs = [_pair(0, 1, score=0.9), _pair(0, 2, score=0.7)]
        result = filter_top_k_per_fragment(pairs, k=10)
        assert len(result) == len(pairs)


# ─── filter_pairs (extra) ─────────────────────────────────────────────────────

class TestFilterPairsExtra:
    def test_returns_tuple(self):
        result = filter_pairs([])
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_list_and_report(self):
        pairs, report = filter_pairs(_pairs_list())
        assert isinstance(pairs, list)
        assert isinstance(report, FilterReport)

    def test_empty_input(self):
        pairs, report = filter_pairs([])
        assert pairs == []
        assert report.n_input == 0

    def test_n_input_matches(self):
        pl = _pairs_list()
        _, report = filter_pairs(pl)
        assert report.n_input == len(pl)

    def test_n_output_matches(self):
        pl = _pairs_list()
        pairs, report = filter_pairs(pl)
        assert report.n_output == len(pairs)

    def test_n_rejected_matches(self):
        pl = _pairs_list()
        pairs, report = filter_pairs(pl)
        assert report.n_rejected == report.n_input - report.n_output

    def test_none_cfg_uses_default(self):
        pairs, report = filter_pairs(_pairs_list(), cfg=None)
        assert isinstance(report, FilterReport)

    def test_method_score_filters(self):
        pl = _pairs_list()
        cfg = FilterConfig(method="score", min_score=0.8)
        result, _ = filter_pairs(pl, cfg)
        assert all(p.score >= 0.8 for p in result)

    def test_method_inlier_filters(self):
        pl = _pairs_list()
        cfg = FilterConfig(method="inlier", min_inliers=15)
        result, _ = filter_pairs(pl, cfg)
        assert all(p.n_inliers >= 15 for p in result)

    def test_max_pairs_respected(self):
        pl = _pairs_list()
        cfg = FilterConfig(max_pairs=2)
        result, _ = filter_pairs(pl, cfg)
        assert len(result) <= 2

    def test_report_method_matches_cfg(self):
        cfg = FilterConfig(method="score")
        _, report = filter_pairs(_pairs_list(), cfg)
        assert report.method == "score"


# ─── merge_filter_results (extra) ─────────────────────────────────────────────

class TestMergeFilterResultsExtra:
    def test_returns_list(self):
        assert isinstance(merge_filter_results([]), list)

    def test_empty_list(self):
        assert merge_filter_results([]) == []

    def test_empty_nested_lists(self):
        assert merge_filter_results([[], []]) == []

    def test_single_list_passthrough(self):
        pairs = _pairs_list()
        result = merge_filter_results([pairs])
        assert len(result) == len(deduplicate_pairs(pairs))

    def test_merges_two_lists(self):
        l1 = [_pair(0, 1, score=0.9)]
        l2 = [_pair(1, 2, score=0.7)]
        result = merge_filter_results([l1, l2])
        assert len(result) == 2

    def test_removes_cross_list_duplicates(self):
        p = _pair(0, 1, score=0.9)
        result = merge_filter_results([[p], [p]])
        assert len(result) == 1

    def test_keeps_higher_score_across_lists(self):
        p1 = CandidatePair(id_a=0, id_b=1, score=0.9)
        p2 = CandidatePair(id_a=1, id_b=0, score=0.5)
        result = merge_filter_results([[p1], [p2]])
        assert result[0].score == pytest.approx(0.9)


# ─── batch_filter (extra) ─────────────────────────────────────────────────────

class TestBatchFilterExtra:
    def test_returns_list(self):
        assert isinstance(batch_filter([]), list)

    def test_empty_pair_lists(self):
        assert batch_filter([]) == []

    def test_single_list(self):
        result = batch_filter([_pairs_list()])
        assert len(result) == 1

    def test_multiple_lists(self):
        result = batch_filter([_pairs_list(), _pairs_list()])
        assert len(result) == 2

    def test_each_element_is_tuple(self):
        result = batch_filter([_pairs_list()])
        assert isinstance(result[0], tuple)

    def test_each_tuple_has_pairs_and_report(self):
        pairs, report = batch_filter([_pairs_list()])[0]
        assert isinstance(pairs, list)
        assert isinstance(report, FilterReport)

    def test_cfg_applied_to_all(self):
        cfg = FilterConfig(min_score=0.8)
        results = batch_filter([_pairs_list(), _pairs_list()], cfg=cfg)
        for pairs, _ in results:
            assert all(p.score >= 0.8 for p in pairs)

    def test_none_cfg_uses_default(self):
        result = batch_filter([_pairs_list()], cfg=None)
        assert len(result) == 1
