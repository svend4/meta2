"""Tests for puzzle_reconstruction.utils.shape_match_utils."""
import pytest
import numpy as np

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

np.random.seed(77)


# ─── ShapeMatchConfig ────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = ShapeMatchConfig()
    assert cfg.min_score == pytest.approx(0.0)
    assert cfg.max_pairs == 100
    assert cfg.method == "hu"


def test_config_invalid_min_score():
    with pytest.raises(ValueError):
        ShapeMatchConfig(min_score=-0.1)


def test_config_invalid_max_pairs():
    with pytest.raises(ValueError):
        ShapeMatchConfig(max_pairs=0)


def test_config_invalid_method():
    with pytest.raises(ValueError):
        ShapeMatchConfig(method="unknown")


def test_config_valid_methods():
    for method in ("hu", "zernike", "combined"):
        cfg = ShapeMatchConfig(method=method)
        assert cfg.method == method


# ─── ShapeMatchEntry ─────────────────────────────────────────────────────────

def test_entry_creation():
    e = make_match_entry(0, 1, 0.8, hu_dist=1.2, iou=0.7, chamfer=0.5, rank=1)
    assert e.idx1 == 0
    assert e.idx2 == 1
    assert e.score == pytest.approx(0.8)
    assert e.hu_dist == pytest.approx(1.2)
    assert e.iou == pytest.approx(0.7)
    assert e.rank == 1


def test_entry_is_good_true():
    e = make_match_entry(0, 1, 0.8)
    assert e.is_good is True


def test_entry_is_good_false():
    e = make_match_entry(0, 1, 0.3)
    assert e.is_good is False


def test_entry_is_good_boundary():
    # exactly 0.5: not > 0.5 → False
    e = make_match_entry(0, 1, 0.5)
    assert e.is_good is False


def test_entry_negative_idx_raises():
    with pytest.raises(ValueError):
        ShapeMatchEntry(idx1=-1, idx2=0, score=0.5)


def test_entry_negative_idx2_raises():
    with pytest.raises(ValueError):
        ShapeMatchEntry(idx1=0, idx2=-1, score=0.5)


# ─── make_match_entry ────────────────────────────────────────────────────────

def test_make_match_entry_meta():
    e = make_match_entry(2, 3, 0.6, meta={"key": "val"})
    assert e.meta["key"] == "val"


def test_make_match_entry_defaults():
    e = make_match_entry(0, 1, 0.7)
    assert e.hu_dist == pytest.approx(0.0)
    assert e.iou == pytest.approx(0.0)
    assert e.chamfer == pytest.approx(0.0)
    assert e.rank == 0
    assert e.meta == {}


# ─── entries_from_results ────────────────────────────────────────────────────

def test_entries_from_results_basic():
    results = [(0, 1, 0.8), (2, 3, 0.5), (4, 5, 0.3)]
    entries = entries_from_results(results)
    assert len(entries) == 3
    assert entries[0].idx1 == 0
    assert entries[0].score == pytest.approx(0.8)


def test_entries_from_results_ranks():
    results = [(0, 1, 0.8), (2, 3, 0.5)]
    entries = entries_from_results(results)
    assert entries[0].rank == 0
    assert entries[1].rank == 1


def test_entries_from_results_empty():
    entries = entries_from_results([])
    assert entries == []


# ─── summarise_matches ───────────────────────────────────────────────────────

def test_summarise_matches_empty():
    s = summarise_matches([])
    assert s.n_total == 0
    assert s.mean_score == pytest.approx(0.0)


def test_summarise_matches_basic():
    entries = [
        make_match_entry(0, 1, 0.8),
        make_match_entry(1, 2, 0.4),
        make_match_entry(2, 3, 0.9),
    ]
    s = summarise_matches(entries)
    assert s.n_total == 3
    assert s.n_good == 2
    assert s.n_poor == 1
    assert s.max_score == pytest.approx(0.9)
    assert s.min_score == pytest.approx(0.4)


def test_summarise_matches_mean_score():
    entries = [
        make_match_entry(0, 1, 0.6),
        make_match_entry(1, 2, 0.8),
    ]
    s = summarise_matches(entries)
    assert s.mean_score == pytest.approx(0.7)


def test_summarise_matches_repr():
    entries = [make_match_entry(0, 1, 0.7)]
    s = summarise_matches(entries)
    r = repr(s)
    assert "n=1" in r


# ─── filter_good_matches ─────────────────────────────────────────────────────

def test_filter_good_matches():
    entries = [
        make_match_entry(0, 1, 0.8),
        make_match_entry(1, 2, 0.3),
        make_match_entry(2, 3, 0.7),
    ]
    result = filter_good_matches(entries)
    assert len(result) == 2
    assert all(e.score > 0.5 for e in result)


# ─── filter_poor_matches ─────────────────────────────────────────────────────

def test_filter_poor_matches():
    entries = [
        make_match_entry(0, 1, 0.8),
        make_match_entry(1, 2, 0.3),
    ]
    result = filter_poor_matches(entries)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.3)


# ─── filter_by_hu_dist ───────────────────────────────────────────────────────

def test_filter_by_hu_dist():
    entries = [
        make_match_entry(0, 1, 0.8, hu_dist=2.0),
        make_match_entry(1, 2, 0.7, hu_dist=15.0),
        make_match_entry(2, 3, 0.6, hu_dist=8.0),
    ]
    result = filter_by_hu_dist(entries, max_hu=10.0)
    assert len(result) == 2
    assert all(e.hu_dist <= 10.0 for e in result)


# ─── filter_match_by_score_range ─────────────────────────────────────────────

def test_filter_match_by_score_range():
    entries = [
        make_match_entry(0, 1, 0.2),
        make_match_entry(1, 2, 0.6),
        make_match_entry(2, 3, 0.9),
    ]
    result = filter_match_by_score_range(entries, lo=0.5, hi=0.8)
    assert len(result) == 1
    assert result[0].score == pytest.approx(0.6)


def test_filter_match_by_score_range_all():
    entries = [make_match_entry(0, 1, 0.5), make_match_entry(1, 2, 0.8)]
    result = filter_match_by_score_range(entries, lo=0.0, hi=1.0)
    assert len(result) == 2


# ─── top_k_match_entries ─────────────────────────────────────────────────────

def test_top_k_match_entries_order():
    entries = [
        make_match_entry(0, 1, 0.3),
        make_match_entry(1, 2, 0.9),
        make_match_entry(2, 3, 0.6),
    ]
    top2 = top_k_match_entries(entries, k=2)
    assert len(top2) == 2
    assert top2[0].score == pytest.approx(0.9)


def test_top_k_match_entries_invalid_k():
    entries = [make_match_entry(0, 1, 0.5)]
    with pytest.raises(ValueError):
        top_k_match_entries(entries, k=0)


def test_top_k_match_entries_k_larger():
    entries = [make_match_entry(0, 1, 0.7)]
    result = top_k_match_entries(entries, k=10)
    assert len(result) == 1


# ─── match_entry_stats ───────────────────────────────────────────────────────

def test_match_entry_stats_empty():
    stats = match_entry_stats([])
    assert stats["n"] == 0
    assert stats["mean_score"] == pytest.approx(0.0)


def test_match_entry_stats_keys():
    entries = [make_match_entry(0, 1, 0.7, hu_dist=1.0, iou=0.5, chamfer=0.3)]
    stats = match_entry_stats(entries)
    for key in ("n", "mean_score", "mean_hu_dist", "mean_iou", "mean_chamfer"):
        assert key in stats


def test_match_entry_stats_values():
    entries = [
        make_match_entry(0, 1, 0.6, hu_dist=2.0, iou=0.4),
        make_match_entry(1, 2, 0.8, hu_dist=4.0, iou=0.6),
    ]
    stats = match_entry_stats(entries)
    assert stats["n"] == 2
    assert stats["mean_score"] == pytest.approx(0.7)
    assert stats["mean_hu_dist"] == pytest.approx(3.0)
    assert stats["mean_iou"] == pytest.approx(0.5)


# ─── compare_match_summaries ─────────────────────────────────────────────────

def test_compare_match_summaries_keys():
    s_a = summarise_matches([make_match_entry(0, 1, 0.8)])
    s_b = summarise_matches([make_match_entry(0, 1, 0.5),
                              make_match_entry(1, 2, 0.3)])
    result = compare_match_summaries(s_a, s_b)
    for key in ("n_total_delta", "n_good_delta", "mean_score_delta",
                "max_score_delta", "min_score_delta"):
        assert key in result


def test_compare_match_summaries_self():
    entries = [make_match_entry(0, 1, 0.7)]
    s = summarise_matches(entries)
    result = compare_match_summaries(s, s)
    assert result["n_total_delta"] == 0
    assert result["mean_score_delta"] == pytest.approx(0.0)


# ─── batch_summarise_matches ─────────────────────────────────────────────────

def test_batch_summarise_matches_length():
    groups = [
        [make_match_entry(0, 1, 0.8)],
        [make_match_entry(0, 1, 0.5), make_match_entry(1, 2, 0.3)],
        [],
    ]
    results = batch_summarise_matches(groups)
    assert len(results) == 3


def test_batch_summarise_matches_empty_group():
    results = batch_summarise_matches([[]])
    assert results[0].n_total == 0


def test_batch_summarise_matches_returns_summaries():
    groups = [[make_match_entry(0, 1, 0.6)]]
    results = batch_summarise_matches(groups)
    assert isinstance(results[0], ShapeMatchSummary)
