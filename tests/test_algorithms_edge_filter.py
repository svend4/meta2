"""Tests for algorithms/edge_filter.py."""
import pytest

from puzzle_reconstruction.algorithms.edge_comparator import EdgeCompareResult
from puzzle_reconstruction.algorithms.edge_filter import (
    EdgeFilterConfig,
    apply_edge_filter,
    batch_filter_edges,
    deduplicate_pairs,
    filter_by_score,
    filter_compatible,
    filter_top_k,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_result(edge_id_a: int, edge_id_b: int, score: float) -> EdgeCompareResult:
    return EdgeCompareResult(
        edge_id_a=edge_id_a,
        edge_id_b=edge_id_b,
        dtw_dist=0.5,
        css_sim=0.7,
        fd_diff=0.1,
        ifs_sim=0.8,
        score=score,
    )


def make_results():
    """Create a standard set of 5 results with varied scores."""
    return [
        make_result(0, 1, 0.9),
        make_result(0, 2, 0.4),
        make_result(1, 2, 0.7),
        make_result(2, 3, 0.2),
        make_result(3, 4, 0.65),
    ]


# ─── EdgeFilterConfig ─────────────────────────────────────────────────────────

class TestEdgeFilterConfig:
    def test_defaults(self):
        cfg = EdgeFilterConfig()
        assert cfg.min_score is None
        assert cfg.top_k is None
        assert cfg.deduplicate is True
        assert cfg.only_compatible is False

    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            EdgeFilterConfig(min_score=1.1)

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            EdgeFilterConfig(min_score=-0.1)

    def test_min_score_zero_valid(self):
        cfg = EdgeFilterConfig(min_score=0.0)
        assert cfg.min_score == pytest.approx(0.0)

    def test_min_score_one_valid(self):
        cfg = EdgeFilterConfig(min_score=1.0)
        assert cfg.min_score == pytest.approx(1.0)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            EdgeFilterConfig(top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            EdgeFilterConfig(top_k=-1)

    def test_top_k_one_valid(self):
        cfg = EdgeFilterConfig(top_k=1)
        assert cfg.top_k == 1

    def test_custom_values(self):
        cfg = EdgeFilterConfig(
            min_score=0.5, top_k=3, deduplicate=False, only_compatible=True
        )
        assert cfg.min_score == pytest.approx(0.5)
        assert cfg.top_k == 3
        assert cfg.deduplicate is False
        assert cfg.only_compatible is True


# ─── filter_by_score ──────────────────────────────────────────────────────────

class TestFilterByScore:
    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            filter_by_score([], min_score=1.5)

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            filter_by_score([], min_score=-0.1)

    def test_empty_returns_empty(self):
        assert filter_by_score([], min_score=0.5) == []

    def test_all_pass_at_zero(self):
        results = make_results()
        filtered = filter_by_score(results, min_score=0.0)
        assert len(filtered) == len(results)

    def test_none_pass_at_one(self):
        results = make_results()
        # scores are all < 1.0 in make_results
        filtered = filter_by_score(results, min_score=1.0)
        assert len(filtered) == 0

    def test_threshold_inclusive(self):
        results = [make_result(0, 1, 0.5)]
        filtered = filter_by_score(results, min_score=0.5)
        assert len(filtered) == 1

    def test_order_preserved(self):
        results = make_results()
        filtered = filter_by_score(results, min_score=0.3)
        scores = [r.score for r in filtered]
        expected = [r.score for r in results if r.score >= 0.3]
        assert scores == expected

    def test_correct_results_kept(self):
        results = make_results()
        filtered = filter_by_score(results, min_score=0.6)
        for r in filtered:
            assert r.score >= 0.6


# ─── filter_top_k ─────────────────────────────────────────────────────────────

class TestFilterTopK:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k"):
            filter_top_k([], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k"):
            filter_top_k([], k=-1)

    def test_empty_returns_empty(self):
        assert filter_top_k([], k=5) == []

    def test_k_larger_than_len_returns_all(self):
        results = make_results()
        filtered = filter_top_k(results, k=100)
        assert len(filtered) == len(results)

    def test_exactly_k_returned(self):
        results = make_results()
        filtered = filter_top_k(results, k=3)
        assert len(filtered) == 3

    def test_sorted_descending(self):
        results = make_results()
        filtered = filter_top_k(results, k=4)
        scores = [r.score for r in filtered]
        assert scores == sorted(scores, reverse=True)

    def test_top_item_has_highest_score(self):
        results = make_results()
        filtered = filter_top_k(results, k=1)
        max_score = max(r.score for r in results)
        assert filtered[0].score == pytest.approx(max_score)

    def test_k_one(self):
        results = make_results()
        filtered = filter_top_k(results, k=1)
        assert len(filtered) == 1


# ─── filter_compatible ────────────────────────────────────────────────────────

class TestFilterCompatible:
    def test_empty_returns_empty(self):
        assert filter_compatible([]) == []

    def test_all_incompatible_returns_empty(self):
        results = [make_result(0, 1, 0.3), make_result(1, 2, 0.1)]
        assert filter_compatible(results) == []

    def test_all_compatible(self):
        results = [make_result(0, 1, 0.7), make_result(1, 2, 0.8), make_result(0, 2, 0.9)]
        filtered = filter_compatible(results)
        assert len(filtered) == 3

    def test_threshold_is_0_6(self):
        below = make_result(0, 1, 0.59)
        at = make_result(1, 2, 0.60)
        above = make_result(2, 3, 0.61)
        filtered = filter_compatible([below, at, above])
        scores = [r.score for r in filtered]
        assert 0.59 not in scores
        assert 0.60 in scores
        assert 0.61 in scores

    def test_mixed_keeps_only_compatible(self):
        results = make_results()
        filtered = filter_compatible(results)
        for r in filtered:
            assert r.is_compatible


# ─── deduplicate_pairs ────────────────────────────────────────────────────────

class TestDeduplicatePairs:
    def test_empty_returns_empty(self):
        assert deduplicate_pairs([]) == []

    def test_no_duplicates_unchanged(self):
        results = make_results()
        deduped = deduplicate_pairs(results)
        assert len(deduped) == len(results)

    def test_mirror_pair_deduplicated(self):
        r1 = make_result(0, 1, 0.8)
        r2 = make_result(1, 0, 0.5)  # mirror pair
        deduped = deduplicate_pairs([r1, r2])
        assert len(deduped) == 1

    def test_first_occurrence_kept(self):
        r1 = make_result(0, 1, 0.8)
        r2 = make_result(1, 0, 0.5)
        deduped = deduplicate_pairs([r1, r2])
        assert deduped[0].score == pytest.approx(0.8)

    def test_same_pair_key_deduplicated(self):
        r1 = make_result(2, 5, 0.6)
        r2 = make_result(2, 5, 0.9)
        deduped = deduplicate_pairs([r1, r2])
        assert len(deduped) == 1

    def test_order_preserved_for_unique(self):
        results = make_results()
        deduped = deduplicate_pairs(results)
        # No duplicates in make_results → order same
        assert [r.score for r in deduped] == [r.score for r in results]


# ─── apply_edge_filter ────────────────────────────────────────────────────────

class TestApplyEdgeFilter:
    def test_default_config_deduplicates(self):
        r1 = make_result(0, 1, 0.8)
        r2 = make_result(1, 0, 0.5)
        filtered = apply_edge_filter([r1, r2])
        assert len(filtered) == 1

    def test_none_config_uses_defaults(self):
        results = make_results()
        filtered = apply_edge_filter(results, cfg=None)
        assert isinstance(filtered, list)

    def test_min_score_applied(self):
        results = make_results()
        cfg = EdgeFilterConfig(min_score=0.6, deduplicate=False)
        filtered = apply_edge_filter(results, cfg=cfg)
        for r in filtered:
            assert r.score >= 0.6

    def test_top_k_applied(self):
        results = make_results()
        cfg = EdgeFilterConfig(top_k=2, deduplicate=False)
        filtered = apply_edge_filter(results, cfg=cfg)
        assert len(filtered) == 2

    def test_only_compatible_applied(self):
        results = make_results()
        cfg = EdgeFilterConfig(only_compatible=True, deduplicate=False)
        filtered = apply_edge_filter(results, cfg=cfg)
        for r in filtered:
            assert r.score >= 0.6

    def test_deduplicate_false_keeps_mirrors(self):
        r1 = make_result(0, 1, 0.8)
        r2 = make_result(1, 0, 0.5)
        cfg = EdgeFilterConfig(deduplicate=False)
        filtered = apply_edge_filter([r1, r2], cfg=cfg)
        assert len(filtered) == 2

    def test_empty_input(self):
        result = apply_edge_filter([])
        assert result == []

    def test_pipeline_order_min_then_topk(self):
        results = make_results()
        # min_score=0.6 keeps 3 results (0.9, 0.7, 0.65), then top_k=2
        cfg = EdgeFilterConfig(min_score=0.6, top_k=2, deduplicate=False)
        filtered = apply_edge_filter(results, cfg=cfg)
        assert len(filtered) == 2
        assert all(r.score >= 0.6 for r in filtered)


# ─── batch_filter_edges ───────────────────────────────────────────────────────

class TestBatchFilterEdges:
    def test_empty_batch_returns_empty(self):
        assert batch_filter_edges([]) == []

    def test_length_preserved(self):
        batches = [make_results(), make_results()[:3], []]
        result = batch_filter_edges(batches)
        assert len(result) == 3

    def test_each_batch_filtered(self):
        batches = [make_results(), make_results()]
        cfg = EdgeFilterConfig(min_score=0.6, deduplicate=False)
        result = batch_filter_edges(batches, cfg=cfg)
        for batch in result:
            for r in batch:
                assert r.score >= 0.6

    def test_returns_list_of_lists(self):
        result = batch_filter_edges([make_results()])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_none_config_uses_defaults(self):
        result = batch_filter_edges([make_results()], cfg=None)
        assert isinstance(result, list)
