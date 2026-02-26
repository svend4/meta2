"""Tests for puzzle_reconstruction.utils.overlap_score_utils"""
import pytest
import numpy as np

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


# ── OverlapScoreConfig ───────────────────────────────────────────────────────

def test_config_defaults():
    cfg = OverlapScoreConfig()
    assert cfg.iou_threshold == 0.05
    assert cfg.area_threshold == 1.0
    assert cfg.penalise_self_overlap is False


def test_config_iou_out_of_range():
    with pytest.raises(ValueError):
        OverlapScoreConfig(iou_threshold=1.5)
    with pytest.raises(ValueError):
        OverlapScoreConfig(iou_threshold=-0.1)


def test_config_negative_area_threshold():
    with pytest.raises(ValueError):
        OverlapScoreConfig(area_threshold=-1.0)


def test_config_boundary_iou():
    cfg0 = OverlapScoreConfig(iou_threshold=0.0)
    cfg1 = OverlapScoreConfig(iou_threshold=1.0)
    assert cfg0.iou_threshold == 0.0
    assert cfg1.iou_threshold == 1.0


# ── OverlapScoreEntry ────────────────────────────────────────────────────────

def test_entry_basic_creation():
    e = OverlapScoreEntry(idx1=0, idx2=1, iou=0.2, overlap_area=50.0)
    assert e.pair == (0, 1)
    assert e.penalty == 0.0


def test_entry_negative_idx_raises():
    with pytest.raises(ValueError):
        OverlapScoreEntry(idx1=-1, idx2=0, iou=0.1, overlap_area=10.0)


def test_entry_iou_out_of_range():
    with pytest.raises(ValueError):
        OverlapScoreEntry(idx1=0, idx2=1, iou=1.5, overlap_area=10.0)
    with pytest.raises(ValueError):
        OverlapScoreEntry(idx1=0, idx2=1, iou=-0.1, overlap_area=10.0)


def test_entry_negative_area_raises():
    with pytest.raises(ValueError):
        OverlapScoreEntry(idx1=0, idx2=1, iou=0.1, overlap_area=-1.0)


def test_entry_penalty_out_of_range_raises():
    with pytest.raises(ValueError):
        OverlapScoreEntry(idx1=0, idx2=1, iou=0.1, overlap_area=10.0, penalty=1.5)


def test_entry_repr():
    e = OverlapScoreEntry(idx1=0, idx2=1, iou=0.2, overlap_area=50.0)
    r = repr(e)
    assert "0" in r and "1" in r


# ── make_overlap_entry ───────────────────────────────────────────────────────

def test_make_overlap_entry_significant():
    cfg = OverlapScoreConfig(iou_threshold=0.1, area_threshold=5.0)
    e = make_overlap_entry(0, 1, iou=0.5, overlap_area=100.0, cfg=cfg)
    assert e.penalty == pytest.approx(0.5)


def test_make_overlap_entry_not_significant():
    cfg = OverlapScoreConfig(iou_threshold=0.5)
    e = make_overlap_entry(0, 1, iou=0.1, overlap_area=100.0, cfg=cfg)
    assert e.penalty == 0.0


def test_make_overlap_entry_default_cfg():
    e = make_overlap_entry(2, 3, iou=0.2, overlap_area=10.0)
    assert e.idx1 == 2
    assert e.idx2 == 3


def test_make_overlap_entry_with_meta():
    e = make_overlap_entry(0, 1, iou=0.3, overlap_area=50.0, meta={"tag": "test"})
    assert e.meta.get("tag") == "test"


# ── summarise_overlaps ───────────────────────────────────────────────────────

def test_summarise_overlaps_empty():
    summary = summarise_overlaps([])
    assert summary.n_overlaps == 0
    assert summary.is_valid is True
    assert summary.total_area == 0.0
    assert summary.max_iou == 0.0


def test_summarise_overlaps_significant():
    entries = [
        make_overlap_entry(0, 1, iou=0.6, overlap_area=100.0),
        make_overlap_entry(1, 2, iou=0.7, overlap_area=80.0),
    ]
    summary = summarise_overlaps(entries)
    assert summary.n_overlaps == 2
    assert summary.is_valid is False
    assert summary.total_area == pytest.approx(180.0)
    assert summary.max_iou == pytest.approx(0.7)


def test_summarise_overlaps_below_threshold():
    cfg = OverlapScoreConfig(iou_threshold=0.9)
    entries = [make_overlap_entry(0, 1, iou=0.1, overlap_area=5.0, cfg=cfg)]
    summary = summarise_overlaps(entries, cfg=cfg)
    assert summary.n_overlaps == 0
    assert summary.is_valid is True


def test_summarise_overlaps_mean_penalty():
    entries = [
        make_overlap_entry(0, 1, iou=0.6, overlap_area=50.0),
        make_overlap_entry(1, 2, iou=0.4, overlap_area=50.0),
    ]
    summary = summarise_overlaps(entries)
    assert 0.0 <= summary.mean_penalty <= 1.0


# ── filter_significant_overlaps ──────────────────────────────────────────────

def test_filter_significant_overlaps_basic():
    entries = [
        OverlapScoreEntry(0, 1, iou=0.1, overlap_area=10.0),
        OverlapScoreEntry(1, 2, iou=0.6, overlap_area=20.0),
    ]
    result = filter_significant_overlaps(entries, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].iou == 0.6


def test_filter_significant_overlaps_empty():
    assert filter_significant_overlaps([], iou_threshold=0.1) == []


# ── filter_by_area ───────────────────────────────────────────────────────────

def test_filter_by_area_basic():
    entries = [
        OverlapScoreEntry(0, 1, iou=0.2, overlap_area=5.0),
        OverlapScoreEntry(1, 2, iou=0.2, overlap_area=50.0),
    ]
    result = filter_by_area(entries, min_area=10.0)
    assert len(result) == 1
    assert result[0].overlap_area == 50.0


def test_filter_by_area_zero_keeps_all():
    entries = [OverlapScoreEntry(0, 1, iou=0.1, overlap_area=0.0)]
    result = filter_by_area(entries, min_area=0.0)
    assert len(result) == 1


# ── top_k_overlaps ───────────────────────────────────────────────────────────

def test_top_k_overlaps_returns_k():
    entries = [OverlapScoreEntry(i, i+1, iou=float(i)/10, overlap_area=1.0)
               for i in range(5)]
    top = top_k_overlaps(entries, k=3)
    assert len(top) == 3


def test_top_k_overlaps_sorted_descending():
    entries = [OverlapScoreEntry(i, i+1, iou=float(i)/10, overlap_area=1.0)
               for i in range(5)]
    top = top_k_overlaps(entries, k=5)
    ious = [e.iou for e in top]
    assert ious == sorted(ious, reverse=True)


def test_top_k_overlaps_larger_than_list():
    entries = [OverlapScoreEntry(0, 1, iou=0.3, overlap_area=10.0)]
    top = top_k_overlaps(entries, k=100)
    assert len(top) == 1


# ── overlap_stats ────────────────────────────────────────────────────────────

def test_overlap_stats_empty():
    stats = overlap_stats([])
    assert stats["n"] == 0
    assert stats["mean_iou"] == 0.0


def test_overlap_stats_keys():
    entries = [OverlapScoreEntry(0, 1, iou=0.3, overlap_area=20.0)]
    stats = overlap_stats(entries)
    for key in ("n", "mean_iou", "std_iou", "max_iou", "min_iou", "total_area", "mean_area"):
        assert key in stats


def test_overlap_stats_single_entry():
    entries = [OverlapScoreEntry(0, 1, iou=0.4, overlap_area=30.0)]
    stats = overlap_stats(entries)
    assert stats["n"] == 1
    assert stats["mean_iou"] == pytest.approx(0.4)
    assert stats["total_area"] == pytest.approx(30.0)


# ── penalty_score ────────────────────────────────────────────────────────────

def test_penalty_score_empty():
    assert penalty_score([]) == 0.0


def test_penalty_score_value():
    entries = [
        make_overlap_entry(0, 1, iou=0.6, overlap_area=50.0),
        make_overlap_entry(1, 2, iou=0.8, overlap_area=50.0),
    ]
    score = penalty_score(entries)
    assert 0.0 <= score <= 1.0


# ── batch_make_overlap_entries ───────────────────────────────────────────────

def test_batch_make_overlap_entries_basic():
    pairs = [(0, 1), (1, 2)]
    ious = [0.2, 0.4]
    areas = [10.0, 20.0]
    entries = batch_make_overlap_entries(pairs, ious, areas)
    assert len(entries) == 2


def test_batch_make_overlap_entries_length_mismatch():
    with pytest.raises(ValueError):
        batch_make_overlap_entries([(0, 1)], [0.1, 0.2], [10.0])


def test_batch_make_overlap_entries_empty():
    entries = batch_make_overlap_entries([], [], [])
    assert entries == []


# ── group_by_fragment ────────────────────────────────────────────────────────

def test_group_by_fragment_basic():
    entries = [
        OverlapScoreEntry(0, 1, iou=0.1, overlap_area=5.0),
        OverlapScoreEntry(0, 2, iou=0.2, overlap_area=8.0),
        OverlapScoreEntry(1, 2, iou=0.3, overlap_area=12.0),
    ]
    groups = group_by_fragment(entries)
    assert 0 in groups
    assert len(groups[0]) == 2
    assert 1 in groups
    assert len(groups[1]) == 1


def test_group_by_fragment_empty():
    assert group_by_fragment([]) == {}
