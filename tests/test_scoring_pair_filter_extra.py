"""Extra tests for puzzle_reconstruction/scoring/pair_filter.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pair(a=0, b=1, score=0.7, n_inliers=10, rank=0) -> CandidatePair:
    return CandidatePair(id_a=a, id_b=b, score=score,
                         n_inliers=n_inliers, rank=rank)


def _cfg(**kw) -> FilterConfig:
    return FilterConfig(**kw)


# ─── FilterConfig ─────────────────────────────────────────────────────────────

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

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FilterConfig(method="invalid")

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

    def test_valid_methods(self):
        for m in ("score", "inlier", "rank", "combined"):
            cfg = FilterConfig(method=m)
            assert cfg.method == m


# ─── CandidatePair ────────────────────────────────────────────────────────────

class TestCandidatePairExtra:
    def test_ids_stored(self):
        p = _pair(a=3, b=7)
        assert p.id_a == 3 and p.id_b == 7

    def test_score_stored(self):
        p = _pair(score=0.85)
        assert p.score == pytest.approx(0.85)

    def test_n_inliers_stored(self):
        p = _pair(n_inliers=20)
        assert p.n_inliers == 20

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

    def test_negative_n_inliers_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=0.5, n_inliers=-1)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError):
            CandidatePair(id_a=0, id_b=1, score=0.5, rank=-1)

    def test_pair_canonical_ordered(self):
        p = _pair(a=5, b=2)
        assert p.pair == (2, 5)

    def test_pair_already_ordered(self):
        p = _pair(a=1, b=3)
        assert p.pair == (1, 3)


# ─── FilterReport ─────────────────────────────────────────────────────────────

class TestFilterReportExtra:
    def test_n_input_stored(self):
        r = FilterReport(n_input=10, n_output=7, n_rejected=3)
        assert r.n_input == 10

    def test_n_output_stored(self):
        r = FilterReport(n_input=10, n_output=7, n_rejected=3)
        assert r.n_output == 7

    def test_n_rejected_stored(self):
        r = FilterReport(n_input=10, n_output=7, n_rejected=3)
        assert r.n_rejected == 3

    def test_method_stored(self):
        r = FilterReport(method="score")
        assert r.method == "score"

    def test_negative_n_input_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_input=-1)

    def test_negative_n_output_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_output=-1)

    def test_negative_n_rejected_raises(self):
        with pytest.raises(ValueError):
            FilterReport(n_rejected=-1)

    def test_rejection_rate_zero_when_no_input(self):
        r = FilterReport()
        assert r.rejection_rate == pytest.approx(0.0)

    def test_rejection_rate_computed(self):
        r = FilterReport(n_input=10, n_output=3, n_rejected=7)
        assert r.rejection_rate == pytest.approx(0.7)

    def test_rejection_rate_one(self):
        r = FilterReport(n_input=5, n_output=0, n_rejected=5)
        assert r.rejection_rate == pytest.approx(1.0)


# ─── filter_by_score ──────────────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_score([_pair()]), list)

    def test_all_pass_with_zero_threshold(self):
        pairs = [_pair(score=0.1), _pair(score=0.9)]
        assert len(filter_by_score(pairs, 0.0)) == 2

    def test_filters_below_threshold(self):
        pairs = [_pair(score=0.3), _pair(score=0.8)]
        result = filter_by_score(pairs, 0.5)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)

    def test_empty_list(self):
        assert filter_by_score([], 0.5) == []

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([_pair()], -0.1)

    def test_none_pass_high_threshold(self):
        pairs = [_pair(score=0.4), _pair(score=0.5)]
        assert filter_by_score(pairs, 0.9) == []


# ─── filter_by_inlier_count ───────────────────────────────────────────────────

class TestFilterByInlierCountExtra:
    def test_returns_list(self):
        assert isinstance(filter_by_inlier_count([_pair()]), list)

    def test_all_pass_zero_threshold(self):
        pairs = [_pair(n_inliers=0), _pair(n_inliers=5)]
        assert len(filter_by_inlier_count(pairs, 0)) == 2

    def test_filters_below_threshold(self):
        pairs = [_pair(n_inliers=3), _pair(n_inliers=10)]
        result = filter_by_inlier_count(pairs, 5)
        assert len(result) == 1
        assert result[0].n_inliers == 10

    def test_empty_list(self):
        assert filter_by_inlier_count([], 5) == []

    def test_negative_min_inliers_raises(self):
        with pytest.raises(ValueError):
            filter_by_inlier_count([_pair()], -1)


# ─── filter_top_k ─────────────────────────────────────────────────────────────

class TestFilterTopKExtra:
    def test_returns_list(self):
        assert isinstance(filter_top_k([_pair()], 1), list)

    def test_limit_applied(self):
        pairs = [_pair(score=s) for s in [0.3, 0.5, 0.8, 0.9]]
        assert len(filter_top_k(pairs, 2)) == 2

    def test_sorted_descending(self):
        pairs = [_pair(a=i, b=i+1, score=float(i) / 10) for i in range(5)]
        result = filter_top_k(pairs, 3)
        scores = [p.score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([_pair()], 0)

    def test_k_exceeds_size(self):
        pairs = [_pair()]
        result = filter_top_k(pairs, 10)
        assert len(result) == 1


# ─── deduplicate_pairs ────────────────────────────────────────────────────────

class TestDeduplicatePairsExtra:
    def test_returns_list(self):
        assert isinstance(deduplicate_pairs([_pair()]), list)

    def test_no_duplicates_unchanged(self):
        pairs = [_pair(a=0, b=1), _pair(a=1, b=2)]
        assert len(deduplicate_pairs(pairs)) == 2

    def test_symmetric_duplicate_kept_higher_score(self):
        p1 = CandidatePair(id_a=0, id_b=1, score=0.5)
        p2 = CandidatePair(id_a=1, id_b=0, score=0.9)
        result = deduplicate_pairs([p1, p2])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_exact_duplicate_kept_once(self):
        p = _pair(a=0, b=1, score=0.7)
        result = deduplicate_pairs([p, p])
        assert len(result) == 1

    def test_empty_list(self):
        assert deduplicate_pairs([]) == []


# ─── filter_top_k_per_fragment ────────────────────────────────────────────────

class TestFilterTopKPerFragmentExtra:
    def test_returns_list(self):
        pairs = [_pair(a=0, b=1), _pair(a=0, b=2), _pair(a=0, b=3)]
        assert isinstance(filter_top_k_per_fragment(pairs, k=2), list)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k_per_fragment([_pair()], k=0)

    def test_k_one_limits_per_fragment(self):
        pairs = [
            CandidatePair(id_a=0, id_b=1, score=0.8),
            CandidatePair(id_a=0, id_b=2, score=0.9),
            CandidatePair(id_a=1, id_b=2, score=0.7),
        ]
        result = filter_top_k_per_fragment(pairs, k=1)
        # Fragment 0 can only be in top-1, same for others
        assert len(result) >= 1

    def test_empty_list(self):
        assert filter_top_k_per_fragment([], k=2) == []


# ─── filter_pairs ─────────────────────────────────────────────────────────────

class TestFilterPairsExtra:
    def _pairs(self):
        return [
            CandidatePair(id_a=0, id_b=1, score=0.9, n_inliers=15),
            CandidatePair(id_a=1, id_b=2, score=0.3, n_inliers=2),
            CandidatePair(id_a=0, id_b=2, score=0.7, n_inliers=8),
        ]

    def test_returns_tuple(self):
        result = filter_pairs(self._pairs())
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_list_and_report(self):
        pairs, report = filter_pairs(self._pairs())
        assert isinstance(pairs, list)
        assert isinstance(report, FilterReport)

    def test_none_cfg_uses_defaults(self):
        pairs, report = filter_pairs(self._pairs(), cfg=None)
        assert isinstance(report, FilterReport)

    def test_score_method_filters_by_score(self):
        cfg = FilterConfig(method="score", min_score=0.5)
        pairs, _ = filter_pairs(self._pairs(), cfg)
        assert all(p.score >= 0.5 for p in pairs)

    def test_inlier_method_filters_by_inliers(self):
        cfg = FilterConfig(method="inlier", min_inliers=5)
        pairs, _ = filter_pairs(self._pairs(), cfg)
        assert all(p.n_inliers >= 5 for p in pairs)

    def test_report_n_input(self):
        _, report = filter_pairs(self._pairs())
        assert report.n_input == 3

    def test_report_n_output_matches(self):
        pairs, report = filter_pairs(self._pairs())
        assert report.n_output == len(pairs)

    def test_report_n_rejected_consistent(self):
        pairs, report = filter_pairs(self._pairs())
        assert report.n_rejected == report.n_input - report.n_output

    def test_max_pairs_limits_output(self):
        cfg = FilterConfig(max_pairs=1)
        pairs, _ = filter_pairs(self._pairs(), cfg)
        assert len(pairs) <= 1

    def test_empty_input(self):
        pairs, report = filter_pairs([])
        assert pairs == []
        assert report.n_input == 0


# ─── merge_filter_results ─────────────────────────────────────────────────────

class TestMergeFilterResultsExtra:
    def test_returns_list(self):
        assert isinstance(merge_filter_results([[_pair()]]), list)

    def test_empty_input(self):
        assert merge_filter_results([]) == []

    def test_single_list_unchanged(self):
        pairs = [_pair(a=0, b=1), _pair(a=1, b=2)]
        result = merge_filter_results([pairs])
        assert len(result) == 2

    def test_merges_multiple_lists(self):
        l1 = [CandidatePair(id_a=0, id_b=1, score=0.7)]
        l2 = [CandidatePair(id_a=1, id_b=2, score=0.8)]
        result = merge_filter_results([l1, l2])
        assert len(result) == 2

    def test_deduplicates_across_lists(self):
        p = CandidatePair(id_a=0, id_b=1, score=0.7)
        q = CandidatePair(id_a=1, id_b=0, score=0.9)
        result = merge_filter_results([[p], [q]])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)


# ─── batch_filter ─────────────────────────────────────────────────────────────

class TestBatchFilterExtra:
    def test_returns_list(self):
        result = batch_filter([[_pair()]])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        result = batch_filter([[_pair()], [_pair(a=1, b=2)]])
        assert len(result) == 2

    def test_empty_batch(self):
        assert batch_filter([]) == []

    def test_each_element_tuple(self):
        result = batch_filter([[_pair()]])
        assert isinstance(result[0], tuple) and len(result[0]) == 2

    def test_none_cfg(self):
        result = batch_filter([[_pair()]], cfg=None)
        assert len(result) == 1
