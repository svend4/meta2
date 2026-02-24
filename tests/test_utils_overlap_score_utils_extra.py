"""Extra tests for puzzle_reconstruction/utils/overlap_score_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.overlap_score_utils import (
    OverlapScoreConfig,
    OverlapScoreEntry,
    OverlapSummary,
    make_overlap_entry,
    summarise_overlaps,
    filter_significant_overlaps,
    filter_by_area,
    top_k_overlaps,
    overlap_stats,
    penalty_score,
    batch_make_overlap_entries,
    group_by_fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _entry(idx1=0, idx2=1, iou=0.1, area=50.0, penalty=0.1) -> OverlapScoreEntry:
    return OverlapScoreEntry(idx1=idx1, idx2=idx2, iou=iou,
                             overlap_area=area, penalty=penalty)


# ─── OverlapScoreConfig ───────────────────────────────────────────────────────

class TestOverlapScoreConfigExtra:
    def test_default_iou_threshold(self):
        assert OverlapScoreConfig().iou_threshold == pytest.approx(0.05)

    def test_default_area_threshold(self):
        assert OverlapScoreConfig().area_threshold == pytest.approx(1.0)

    def test_iou_out_of_range_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreConfig(iou_threshold=1.5)

    def test_negative_area_threshold_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreConfig(area_threshold=-1.0)

    def test_penalise_self_overlap_default(self):
        assert OverlapScoreConfig().penalise_self_overlap is False


# ─── OverlapScoreEntry ────────────────────────────────────────────────────────

class TestOverlapScoreEntryExtra:
    def test_pair_property(self):
        e = _entry(idx1=3, idx2=7)
        assert e.pair == (3, 7)

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=-1, idx2=0, iou=0.1, overlap_area=5.0)

    def test_iou_out_of_range_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=0, idx2=1, iou=1.5, overlap_area=5.0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=0, idx2=1, iou=0.1, overlap_area=-1.0)

    def test_penalty_out_of_range_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=0, idx2=1, iou=0.1, overlap_area=5.0, penalty=1.5)

    def test_repr_contains_pair(self):
        assert "0" in repr(_entry())


# ─── OverlapSummary ───────────────────────────────────────────────────────────

class TestOverlapSummaryExtra:
    def test_negative_n_overlaps_raises(self):
        with pytest.raises(ValueError):
            OverlapSummary(entries=[], n_overlaps=-1, total_area=0.0,
                           max_iou=0.0, mean_penalty=0.0, is_valid=True)

    def test_negative_total_area_raises(self):
        with pytest.raises(ValueError):
            OverlapSummary(entries=[], n_overlaps=0, total_area=-5.0,
                           max_iou=0.0, mean_penalty=0.0, is_valid=True)


# ─── make_overlap_entry ───────────────────────────────────────────────────────

class TestMakeOverlapEntryExtra:
    def test_returns_entry(self):
        e = make_overlap_entry(0, 1, 0.1, 50.0)
        assert isinstance(e, OverlapScoreEntry)

    def test_significant_overlap_has_penalty(self):
        # Default thresholds: iou>=0.05, area>=1.0
        e = make_overlap_entry(0, 1, 0.2, 10.0)
        assert e.penalty == pytest.approx(0.2)

    def test_insignificant_overlap_zero_penalty(self):
        # iou below threshold
        e = make_overlap_entry(0, 1, 0.01, 10.0)
        assert e.penalty == pytest.approx(0.0)

    def test_meta_stored(self):
        e = make_overlap_entry(0, 1, 0.1, 5.0, meta={"note": "test"})
        assert e.meta["note"] == "test"


# ─── summarise_overlaps ───────────────────────────────────────────────────────

class TestSummariseOverlapsExtra:
    def test_no_significant_is_valid(self):
        # iou below threshold
        entries = [_entry(iou=0.01, area=0.5, penalty=0.0)]
        s = summarise_overlaps(entries)
        assert s.is_valid is True

    def test_significant_overlap_not_valid(self):
        entries = [_entry(iou=0.2, area=50.0, penalty=0.2)]
        s = summarise_overlaps(entries)
        assert s.is_valid is False

    def test_total_area_summed(self):
        entries = [_entry(iou=0.2, area=30.0, penalty=0.2),
                   _entry(iou=0.3, area=20.0, penalty=0.3)]
        s = summarise_overlaps(entries)
        assert s.total_area == pytest.approx(50.0)

    def test_empty_entries_valid(self):
        s = summarise_overlaps([])
        assert s.is_valid is True and s.n_overlaps == 0


# ─── filter_significant_overlaps ─────────────────────────────────────────────

class TestFilterSignificantOverlapsExtra:
    def test_filters_by_iou(self):
        entries = [_entry(iou=0.01), _entry(iou=0.2)]
        result = filter_significant_overlaps(entries, iou_threshold=0.1)
        assert len(result) == 1

    def test_empty_input(self):
        assert filter_significant_overlaps([]) == []


# ─── filter_by_area ───────────────────────────────────────────────────────────

class TestFilterByAreaExtra:
    def test_filters_by_area(self):
        entries = [_entry(area=0.5), _entry(area=100.0)]
        result = filter_by_area(entries, min_area=1.0)
        assert len(result) == 1


# ─── top_k_overlaps ───────────────────────────────────────────────────────────

class TestTopKOverlapsExtra:
    def test_returns_top_by_iou(self):
        entries = [_entry(iou=0.1), _entry(iou=0.5), _entry(iou=0.3)]
        top = top_k_overlaps(entries, 2)
        assert top[0].iou == pytest.approx(0.5)
        assert len(top) == 2

    def test_empty_input(self):
        assert top_k_overlaps([], 5) == []


# ─── overlap_stats ────────────────────────────────────────────────────────────

class TestOverlapStatsExtra:
    def test_empty_returns_zeros(self):
        s = overlap_stats([])
        assert s["n"] == 0

    def test_count_and_mean(self):
        entries = [_entry(iou=0.2), _entry(iou=0.4)]
        s = overlap_stats(entries)
        assert s["n"] == 2
        assert s["mean_iou"] == pytest.approx(0.3)

    def test_total_area(self):
        entries = [_entry(area=30.0), _entry(area=20.0)]
        s = overlap_stats(entries)
        assert s["total_area"] == pytest.approx(50.0)


# ─── penalty_score ────────────────────────────────────────────────────────────

class TestPenaltyScoreExtra:
    def test_empty_is_zero(self):
        assert penalty_score([]) == pytest.approx(0.0)

    def test_zero_penalty_entries(self):
        entries = [_entry(penalty=0.0), _entry(penalty=0.0)]
        assert penalty_score(entries) == pytest.approx(0.0)

    def test_nonzero_penalty(self):
        entries = [_entry(penalty=0.4), _entry(penalty=0.6)]
        assert penalty_score(entries) == pytest.approx(0.5)


# ─── batch_make_overlap_entries ───────────────────────────────────────────────

class TestBatchMakeOverlapEntriesExtra:
    def test_returns_list(self):
        result = batch_make_overlap_entries([(0, 1)], [0.1], [10.0])
        assert isinstance(result, list) and len(result) == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            batch_make_overlap_entries([(0, 1), (1, 2)], [0.1], [10.0])

    def test_empty_input(self):
        assert batch_make_overlap_entries([], [], []) == []


# ─── group_by_fragment ────────────────────────────────────────────────────────

class TestGroupByFragmentExtra:
    def test_groups_correctly(self):
        entries = [_entry(idx1=0, idx2=1), _entry(idx1=0, idx2=2),
                   _entry(idx1=1, idx2=2)]
        groups = group_by_fragment(entries)
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_empty_input(self):
        assert group_by_fragment({}) == {}
