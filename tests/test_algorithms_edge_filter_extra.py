"""Extra tests for puzzle_reconstruction.algorithms.edge_filter."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _res(a, b, score):
    return EdgeCompareResult(
        edge_id_a=a, edge_id_b=b,
        dtw_dist=0.5, css_sim=0.7, fd_diff=0.1, ifs_sim=0.8,
        score=score,
    )


def _results():
    return [
        _res(0, 1, 0.9), _res(0, 2, 0.4), _res(1, 2, 0.7),
        _res(2, 3, 0.2), _res(3, 4, 0.65),
    ]


# ─── TestEdgeFilterConfigExtra ────────────────────────────────────────────────

class TestEdgeFilterConfigExtra:
    def test_defaults(self):
        c = EdgeFilterConfig()
        assert c.min_score is None
        assert c.top_k is None
        assert c.deduplicate is True
        assert c.only_compatible is False

    def test_min_score_above_one_raises(self):
        with pytest.raises(ValueError):
            EdgeFilterConfig(min_score=1.1)

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError):
            EdgeFilterConfig(min_score=-0.01)

    def test_min_score_zero_ok(self):
        assert EdgeFilterConfig(min_score=0.0).min_score == pytest.approx(0.0)

    def test_min_score_one_ok(self):
        assert EdgeFilterConfig(min_score=1.0).min_score == pytest.approx(1.0)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeFilterConfig(top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError):
            EdgeFilterConfig(top_k=-5)

    def test_top_k_one_ok(self):
        assert EdgeFilterConfig(top_k=1).top_k == 1

    def test_custom_values(self):
        c = EdgeFilterConfig(min_score=0.3, top_k=5, deduplicate=False,
                              only_compatible=True)
        assert c.min_score == pytest.approx(0.3)
        assert c.top_k == 5
        assert c.deduplicate is False
        assert c.only_compatible is True

    def test_min_score_half(self):
        assert EdgeFilterConfig(min_score=0.5).min_score == pytest.approx(0.5)

    def test_top_k_large(self):
        assert EdgeFilterConfig(top_k=1000).top_k == 1000


# ─── TestFilterByScoreExtra ───────────────────────────────────────────────────

class TestFilterByScoreExtra:
    def test_above_one_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([], min_score=1.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            filter_by_score([], min_score=-0.1)

    def test_empty_returns_empty(self):
        assert filter_by_score([], min_score=0.5) == []

    def test_all_pass_at_zero(self):
        assert len(filter_by_score(_results(), min_score=0.0)) == 5

    def test_none_pass_at_one(self):
        assert len(filter_by_score(_results(), min_score=1.0)) == 0

    def test_inclusive(self):
        r = [_res(0, 1, 0.5)]
        assert len(filter_by_score(r, min_score=0.5)) == 1

    def test_order_preserved(self):
        results = _results()
        filtered = filter_by_score(results, min_score=0.3)
        expected = [r.score for r in results if r.score >= 0.3]
        assert [r.score for r in filtered] == expected

    def test_correct_kept(self):
        for r in filter_by_score(_results(), min_score=0.6):
            assert r.score >= 0.6

    def test_threshold_0_5(self):
        filtered = filter_by_score(_results(), min_score=0.5)
        assert all(r.score >= 0.5 for r in filtered)


# ─── TestFilterTopKExtra ──────────────────────────────────────────────────────

class TestFilterTopKExtra:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=-1)

    def test_empty_returns_empty(self):
        assert filter_top_k([], k=5) == []

    def test_k_gt_len_returns_all(self):
        assert len(filter_top_k(_results(), k=100)) == 5

    def test_exactly_k(self):
        assert len(filter_top_k(_results(), k=3)) == 3

    def test_sorted_desc(self):
        scores = [r.score for r in filter_top_k(_results(), k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_top_1_has_max(self):
        top = filter_top_k(_results(), k=1)
        assert top[0].score == pytest.approx(0.9)

    def test_top_2(self):
        top = filter_top_k(_results(), k=2)
        assert len(top) == 2
        assert top[0].score >= top[1].score


# ─── TestFilterCompatibleExtra ────────────────────────────────────────────────

class TestFilterCompatibleExtra:
    def test_empty_returns_empty(self):
        assert filter_compatible([]) == []

    def test_all_low_returns_empty(self):
        results = [_res(0, 1, 0.3), _res(1, 2, 0.1)]
        assert filter_compatible(results) == []

    def test_all_high(self):
        results = [_res(0, 1, 0.7), _res(1, 2, 0.8), _res(0, 2, 0.9)]
        assert len(filter_compatible(results)) == 3

    def test_at_0_6_included(self):
        filtered = filter_compatible([_res(0, 1, 0.60)])
        assert len(filtered) == 1

    def test_below_0_6_excluded(self):
        filtered = filter_compatible([_res(0, 1, 0.59)])
        assert len(filtered) == 0

    def test_mixed(self):
        filtered = filter_compatible(_results())
        for r in filtered:
            assert r.is_compatible


# ─── TestDeduplicatePairsExtra ────────────────────────────────────────────────

class TestDeduplicatePairsExtra:
    def test_empty(self):
        assert deduplicate_pairs([]) == []

    def test_no_dupes_same(self):
        assert len(deduplicate_pairs(_results())) == 5

    def test_mirror_deduplicated(self):
        r1 = _res(0, 1, 0.8)
        r2 = _res(1, 0, 0.5)
        assert len(deduplicate_pairs([r1, r2])) == 1

    def test_first_kept(self):
        r1 = _res(0, 1, 0.8)
        r2 = _res(1, 0, 0.5)
        assert deduplicate_pairs([r1, r2])[0].score == pytest.approx(0.8)

    def test_same_pair_deduplicated(self):
        r1 = _res(2, 5, 0.6)
        r2 = _res(2, 5, 0.9)
        assert len(deduplicate_pairs([r1, r2])) == 1

    def test_order_preserved(self):
        results = _results()
        scores = [r.score for r in deduplicate_pairs(results)]
        assert scores == [r.score for r in results]


# ─── TestApplyEdgeFilterExtra ─────────────────────────────────────────────────

class TestApplyEdgeFilterExtra:
    def test_default_deduplicates(self):
        r1 = _res(0, 1, 0.8)
        r2 = _res(1, 0, 0.5)
        assert len(apply_edge_filter([r1, r2])) == 1

    def test_none_cfg_ok(self):
        assert isinstance(apply_edge_filter(_results(), cfg=None), list)

    def test_min_score_applied(self):
        cfg = EdgeFilterConfig(min_score=0.6, deduplicate=False)
        for r in apply_edge_filter(_results(), cfg=cfg):
            assert r.score >= 0.6

    def test_top_k_applied(self):
        cfg = EdgeFilterConfig(top_k=2, deduplicate=False)
        assert len(apply_edge_filter(_results(), cfg=cfg)) == 2

    def test_only_compatible(self):
        cfg = EdgeFilterConfig(only_compatible=True, deduplicate=False)
        for r in apply_edge_filter(_results(), cfg=cfg):
            assert r.score >= 0.6

    def test_dedup_false_keeps_mirrors(self):
        r1 = _res(0, 1, 0.8)
        r2 = _res(1, 0, 0.5)
        cfg = EdgeFilterConfig(deduplicate=False)
        assert len(apply_edge_filter([r1, r2], cfg=cfg)) == 2

    def test_empty(self):
        assert apply_edge_filter([]) == []

    def test_pipeline_min_then_topk(self):
        cfg = EdgeFilterConfig(min_score=0.6, top_k=2, deduplicate=False)
        filtered = apply_edge_filter(_results(), cfg=cfg)
        assert len(filtered) == 2
        assert all(r.score >= 0.6 for r in filtered)


# ─── TestBatchFilterEdgesExtra ────────────────────────────────────────────────

class TestBatchFilterEdgesExtra:
    def test_empty_batch(self):
        assert batch_filter_edges([]) == []

    def test_length_preserved(self):
        batches = [_results(), _results()[:3], []]
        assert len(batch_filter_edges(batches)) == 3

    def test_each_batch_filtered(self):
        cfg = EdgeFilterConfig(min_score=0.6, deduplicate=False)
        for batch in batch_filter_edges([_results(), _results()], cfg=cfg):
            for r in batch:
                assert r.score >= 0.6

    def test_returns_list_of_lists(self):
        result = batch_filter_edges([_results()])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_none_cfg(self):
        assert isinstance(batch_filter_edges([_results()], cfg=None), list)
