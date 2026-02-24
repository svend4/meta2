"""Extra tests for puzzle_reconstruction/utils/shape_match_utils.py (iter-234)."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.shape_match_utils import (
    ShapeMatchConfig,
    ShapeMatchEntry,
    ShapeMatchSummary,
    make_match_entry,
    entries_from_results,
    summarise_matches,
    filter_good_matches,
    filter_poor_matches,
    filter_by_hu_dist,
    filter_match_by_score_range,
    top_k_match_entries,
    match_entry_stats,
    compare_match_summaries,
    batch_summarise_matches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(
    idx1: int = 0,
    idx2: int = 1,
    score: float = 0.7,
    hu_dist: float = 1.0,
    iou: float = 0.5,
    chamfer: float = 0.2,
    rank: int = 0,
) -> ShapeMatchEntry:
    return make_match_entry(
        idx1=idx1, idx2=idx2, score=score,
        hu_dist=hu_dist, iou=iou, chamfer=chamfer, rank=rank,
    )


def _entries_mixed() -> list:
    return [
        _entry(idx1=0, idx2=1, score=0.9, hu_dist=0.5, iou=0.8),
        _entry(idx1=1, idx2=2, score=0.3, hu_dist=5.0, iou=0.2),
        _entry(idx1=2, idx2=3, score=0.7, hu_dist=2.0, iou=0.6),
        _entry(idx1=3, idx2=4, score=0.4, hu_dist=12.0, iou=0.3),
        _entry(idx1=4, idx2=5, score=0.6, hu_dist=3.0, iou=0.5),
    ]


# ─── ShapeMatchConfig ───────────────────────────────────────────────────────

class TestShapeMatchConfigExtra:
    def test_default_min_score(self):
        assert ShapeMatchConfig().min_score == pytest.approx(0.0)

    def test_default_max_pairs(self):
        assert ShapeMatchConfig().max_pairs == 100

    def test_default_method(self):
        assert ShapeMatchConfig().method == "hu"

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(min_score=-0.1)

    def test_zero_max_pairs_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(max_pairs=0)

    def test_negative_max_pairs_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(max_pairs=-5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchConfig(method="invalid")

    def test_valid_method_hu(self):
        cfg = ShapeMatchConfig(method="hu")
        assert cfg.method == "hu"

    def test_valid_method_zernike(self):
        cfg = ShapeMatchConfig(method="zernike")
        assert cfg.method == "zernike"

    def test_valid_method_combined(self):
        cfg = ShapeMatchConfig(method="combined")
        assert cfg.method == "combined"


# ─── ShapeMatchEntry ────────────────────────────────────────────────────────

class TestShapeMatchEntryExtra:
    def test_fields_stored(self):
        e = _entry(idx1=3, idx2=7, score=0.85)
        assert e.idx1 == 3
        assert e.idx2 == 7
        assert e.score == pytest.approx(0.85)

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchEntry(idx1=-1, idx2=0, score=0.5)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchEntry(idx1=0, idx2=-1, score=0.5)

    def test_both_negative_raises(self):
        with pytest.raises(ValueError):
            ShapeMatchEntry(idx1=-1, idx2=-2, score=0.5)

    def test_is_good_above_threshold(self):
        assert _entry(score=0.8).is_good is True

    def test_is_good_below_threshold(self):
        assert _entry(score=0.3).is_good is False

    def test_is_good_at_boundary(self):
        assert _entry(score=0.5).is_good is False

    def test_is_good_just_above(self):
        assert _entry(score=0.51).is_good is True

    def test_meta_default_empty(self):
        e = _entry()
        assert e.meta == {}

    def test_zero_indices_valid(self):
        e = ShapeMatchEntry(idx1=0, idx2=0, score=0.5)
        assert e.idx1 == 0 and e.idx2 == 0


# ─── ShapeMatchSummary ──────────────────────────────────────────────────────

class TestShapeMatchSummaryExtra:
    def test_repr_contains_n(self):
        s = summarise_matches([_entry()])
        assert "n=1" in repr(s)

    def test_repr_contains_good(self):
        s = summarise_matches([_entry(score=0.8)])
        assert "good=1" in repr(s)

    def test_repr_contains_poor(self):
        s = summarise_matches([_entry(score=0.3)])
        assert "poor=1" in repr(s)

    def test_repr_contains_mean(self):
        s = summarise_matches([_entry(score=0.75)])
        assert "mean=" in repr(s)

    def test_repr_contains_max(self):
        s = summarise_matches([_entry(score=0.9)])
        assert "max=" in repr(s)

    def test_repr_contains_min(self):
        s = summarise_matches([_entry(score=0.1)])
        assert "min=" in repr(s)


# ─── make_match_entry ───────────────────────────────────────────────────────

class TestMakeMatchEntryExtra:
    def test_returns_entry(self):
        e = make_match_entry(0, 1, 0.7)
        assert isinstance(e, ShapeMatchEntry)

    def test_default_hu_dist(self):
        e = make_match_entry(0, 1, 0.5)
        assert e.hu_dist == pytest.approx(0.0)

    def test_default_iou(self):
        e = make_match_entry(0, 1, 0.5)
        assert e.iou == pytest.approx(0.0)

    def test_default_chamfer(self):
        e = make_match_entry(0, 1, 0.5)
        assert e.chamfer == pytest.approx(0.0)

    def test_default_rank(self):
        e = make_match_entry(0, 1, 0.5)
        assert e.rank == 0

    def test_meta_none_becomes_empty_dict(self):
        e = make_match_entry(0, 1, 0.5, meta=None)
        assert e.meta == {}

    def test_meta_dict_stored(self):
        e = make_match_entry(0, 1, 0.5, meta={"tag": "test"})
        assert e.meta == {"tag": "test"}

    def test_all_params_stored(self):
        e = make_match_entry(2, 5, 0.9, hu_dist=1.5, iou=0.8, chamfer=0.3, rank=7)
        assert e.idx1 == 2
        assert e.hu_dist == pytest.approx(1.5)
        assert e.rank == 7


# ─── entries_from_results ───────────────────────────────────────────────────

class TestEntriesFromResultsExtra:
    def test_empty_list(self):
        assert entries_from_results([]) == []

    def test_length_matches(self):
        results = [(0, 1, 0.5), (2, 3, 0.8), (4, 5, 0.6)]
        assert len(entries_from_results(results)) == 3

    def test_rank_assigned_sequentially(self):
        results = [(0, 1, 0.5), (2, 3, 0.8)]
        entries = entries_from_results(results)
        assert entries[0].rank == 0
        assert entries[1].rank == 1

    def test_scores_preserved(self):
        results = [(0, 1, 0.42)]
        entries = entries_from_results(results)
        assert entries[0].score == pytest.approx(0.42)

    def test_indices_preserved(self):
        results = [(10, 20, 0.5)]
        entries = entries_from_results(results)
        assert entries[0].idx1 == 10
        assert entries[0].idx2 == 20

    def test_returns_shape_match_entries(self):
        results = [(0, 1, 0.5)]
        assert isinstance(entries_from_results(results)[0], ShapeMatchEntry)


# ─── summarise_matches ──────────────────────────────────────────────────────

class TestSummariseMatchesExtra:
    def test_empty(self):
        s = summarise_matches([])
        assert s.n_total == 0
        assert s.mean_score == pytest.approx(0.0)

    def test_single_good(self):
        s = summarise_matches([_entry(score=0.8)])
        assert s.n_good == 1
        assert s.n_poor == 0

    def test_single_poor(self):
        s = summarise_matches([_entry(score=0.3)])
        assert s.n_good == 0
        assert s.n_poor == 1

    def test_n_total(self):
        entries = _entries_mixed()
        s = summarise_matches(entries)
        assert s.n_total == 5

    def test_good_poor_sum(self):
        entries = _entries_mixed()
        s = summarise_matches(entries)
        assert s.n_good + s.n_poor == s.n_total

    def test_mean_score_two_entries(self):
        entries = [_entry(score=0.4), _entry(score=0.8)]
        s = summarise_matches(entries)
        assert s.mean_score == pytest.approx(0.6)

    def test_min_max_scores(self):
        entries = _entries_mixed()
        s = summarise_matches(entries)
        assert s.min_score == pytest.approx(0.3)
        assert s.max_score == pytest.approx(0.9)

    def test_entries_stored_in_summary(self):
        entries = [_entry()]
        s = summarise_matches(entries)
        assert s.entries is entries


# ─── filter_good_matches ────────────────────────────────────────────────────

class TestFilterGoodMatchesExtra:
    def test_keeps_good_only(self):
        entries = _entries_mixed()
        result = filter_good_matches(entries)
        assert all(e.is_good for e in result)

    def test_count(self):
        entries = _entries_mixed()
        # scores: 0.9, 0.3, 0.7, 0.4, 0.6 -> good: 0.9, 0.7, 0.6
        assert len(filter_good_matches(entries)) == 3

    def test_empty_list(self):
        assert filter_good_matches([]) == []

    def test_all_good(self):
        entries = [_entry(score=0.8) for _ in range(3)]
        assert len(filter_good_matches(entries)) == 3

    def test_none_good(self):
        entries = [_entry(score=0.2) for _ in range(3)]
        assert len(filter_good_matches(entries)) == 0


# ─── filter_poor_matches ────────────────────────────────────────────────────

class TestFilterPoorMatchesExtra:
    def test_keeps_poor_only(self):
        entries = _entries_mixed()
        result = filter_poor_matches(entries)
        assert all(not e.is_good for e in result)

    def test_count(self):
        entries = _entries_mixed()
        # scores: 0.9, 0.3, 0.7, 0.4, 0.6 -> poor: 0.3, 0.4
        assert len(filter_poor_matches(entries)) == 2

    def test_empty_list(self):
        assert filter_poor_matches([]) == []

    def test_all_poor(self):
        entries = [_entry(score=0.1) for _ in range(3)]
        assert len(filter_poor_matches(entries)) == 3

    def test_none_poor(self):
        entries = [_entry(score=0.9) for _ in range(3)]
        assert len(filter_poor_matches(entries)) == 0


# ─── filter_by_hu_dist ──────────────────────────────────────────────────────

class TestFilterByHuDistExtra:
    def test_default_max_hu(self):
        entries = _entries_mixed()
        result = filter_by_hu_dist(entries)
        assert all(e.hu_dist <= 10.0 for e in result)

    def test_strict_threshold(self):
        entries = _entries_mixed()
        result = filter_by_hu_dist(entries, max_hu=1.0)
        assert len(result) == 1
        assert result[0].hu_dist == pytest.approx(0.5)

    def test_all_pass(self):
        entries = _entries_mixed()
        result = filter_by_hu_dist(entries, max_hu=100.0)
        assert len(result) == 5

    def test_none_pass(self):
        entries = _entries_mixed()
        result = filter_by_hu_dist(entries, max_hu=0.1)
        assert len(result) == 0

    def test_exact_boundary(self):
        entries = [_entry(hu_dist=5.0)]
        assert len(filter_by_hu_dist(entries, max_hu=5.0)) == 1

    def test_empty(self):
        assert filter_by_hu_dist([], max_hu=10.0) == []


# ─── filter_match_by_score_range ────────────────────────────────────────────

class TestFilterMatchByScoreRangeExtra:
    def test_default_range(self):
        entries = _entries_mixed()
        result = filter_match_by_score_range(entries)
        assert len(result) == 5

    def test_narrow_range(self):
        entries = _entries_mixed()
        result = filter_match_by_score_range(entries, lo=0.5, hi=0.8)
        # scores: 0.9, 0.3, 0.7, 0.4, 0.6 -> in [0.5, 0.8]: 0.7, 0.6
        assert len(result) == 2

    def test_empty_range(self):
        entries = _entries_mixed()
        result = filter_match_by_score_range(entries, lo=0.95, hi=1.0)
        assert len(result) == 0

    def test_exact_boundary_lo(self):
        entries = [_entry(score=0.5)]
        assert len(filter_match_by_score_range(entries, lo=0.5, hi=1.0)) == 1

    def test_exact_boundary_hi(self):
        entries = [_entry(score=0.8)]
        assert len(filter_match_by_score_range(entries, lo=0.0, hi=0.8)) == 1

    def test_empty(self):
        assert filter_match_by_score_range([], lo=0.0, hi=1.0) == []


# ─── top_k_match_entries ────────────────────────────────────────────────────

class TestTopKMatchEntriesExtra:
    def test_top_1(self):
        entries = _entries_mixed()
        result = top_k_match_entries(entries, k=1)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.9)

    def test_top_3(self):
        entries = _entries_mixed()
        result = top_k_match_entries(entries, k=3)
        assert len(result) == 3
        assert result[0].score >= result[1].score >= result[2].score

    def test_k_greater_than_n(self):
        entries = _entries_mixed()
        result = top_k_match_entries(entries, k=100)
        assert len(result) == len(entries)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_match_entries([], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            top_k_match_entries([], k=-1)

    def test_descending_order(self):
        entries = _entries_mixed()
        result = top_k_match_entries(entries, k=5)
        scores = [e.score for e in result]
        assert scores == sorted(scores, reverse=True)


# ─── match_entry_stats ──────────────────────────────────────────────────────

class TestMatchEntryStatsExtra:
    def test_empty(self):
        stats = match_entry_stats([])
        assert stats["n"] == 0
        assert stats["mean_score"] == pytest.approx(0.0)

    def test_n_count(self):
        entries = _entries_mixed()
        stats = match_entry_stats(entries)
        assert stats["n"] == 5

    def test_mean_score(self):
        entries = [_entry(score=0.4), _entry(score=0.8)]
        stats = match_entry_stats(entries)
        assert stats["mean_score"] == pytest.approx(0.6)

    def test_mean_hu_dist(self):
        entries = [_entry(hu_dist=2.0), _entry(hu_dist=4.0)]
        stats = match_entry_stats(entries)
        assert stats["mean_hu_dist"] == pytest.approx(3.0)

    def test_mean_iou(self):
        entries = [_entry(iou=0.3), _entry(iou=0.7)]
        stats = match_entry_stats(entries)
        assert stats["mean_iou"] == pytest.approx(0.5)

    def test_mean_chamfer(self):
        entries = [_entry(chamfer=1.0), _entry(chamfer=3.0)]
        stats = match_entry_stats(entries)
        assert stats["mean_chamfer"] == pytest.approx(2.0)


# ─── compare_match_summaries ────────────────────────────────────────────────

class TestCompareMatchSummariesExtra:
    def test_identical(self):
        s = summarise_matches(_entries_mixed())
        delta = compare_match_summaries(s, s)
        assert delta["mean_score_delta"] == pytest.approx(0.0)
        assert delta["n_total_delta"] == 0
        assert delta["n_good_delta"] == 0

    def test_a_better(self):
        a = summarise_matches([_entry(score=0.9)])
        b = summarise_matches([_entry(score=0.5)])
        delta = compare_match_summaries(a, b)
        assert delta["mean_score_delta"] > 0.0

    def test_b_better(self):
        a = summarise_matches([_entry(score=0.3)])
        b = summarise_matches([_entry(score=0.8)])
        delta = compare_match_summaries(a, b)
        assert delta["mean_score_delta"] < 0.0

    def test_max_score_delta(self):
        a = summarise_matches([_entry(score=0.9)])
        b = summarise_matches([_entry(score=0.6)])
        delta = compare_match_summaries(a, b)
        assert delta["max_score_delta"] == pytest.approx(0.3)

    def test_min_score_delta(self):
        a = summarise_matches([_entry(score=0.2)])
        b = summarise_matches([_entry(score=0.5)])
        delta = compare_match_summaries(a, b)
        assert delta["min_score_delta"] == pytest.approx(-0.3)

    def test_returns_dict(self):
        s = summarise_matches([_entry()])
        assert isinstance(compare_match_summaries(s, s), dict)


# ─── batch_summarise_matches ────────────────────────────────────────────────

class TestBatchSummariseMatchesExtra:
    def test_single_group(self):
        groups = [_entries_mixed()]
        result = batch_summarise_matches(groups)
        assert len(result) == 1
        assert isinstance(result[0], ShapeMatchSummary)

    def test_multiple_groups(self):
        groups = [
            [_entry(score=0.5)],
            [_entry(score=0.9)],
        ]
        result = batch_summarise_matches(groups)
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_summarise_matches([]) == []

    def test_group_with_empty_list(self):
        result = batch_summarise_matches([[]])
        assert len(result) == 1
        assert result[0].n_total == 0

    def test_summaries_independent(self):
        groups = [
            [_entry(score=0.2)],
            [_entry(score=0.8)],
        ]
        result = batch_summarise_matches(groups)
        assert result[0].mean_score == pytest.approx(0.2)
        assert result[1].mean_score == pytest.approx(0.8)

    def test_n_good_per_group(self):
        groups = [
            [_entry(score=0.9), _entry(score=0.3)],
            [_entry(score=0.8)],
        ]
        result = batch_summarise_matches(groups)
        assert result[0].n_good == 1
        assert result[1].n_good == 1
