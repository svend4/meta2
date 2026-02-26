"""Tests for puzzle_reconstruction.utils.orient_skew_utils."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.orient_skew_utils import (
    OrientMatchConfig,
    OrientMatchEntry,
    OrientMatchSummary,
    SkewCorrConfig,
    SkewCorrEntry,
    SkewCorrSummary,
    make_orient_match_entry,
    summarise_orient_match_entries,
    filter_high_orient_matches,
    filter_low_orient_matches,
    filter_orient_by_score_range,
    filter_orient_by_max_angle,
    top_k_orient_match_entries,
    best_orient_match_entry,
    orient_match_stats,
    compare_orient_summaries,
    batch_summarise_orient_match_entries,
    make_skew_corr_entry,
    summarise_skew_corr_entries,
    filter_high_confidence_skew,
    filter_skew_by_method,
    filter_skew_by_angle_range,
    top_k_skew_entries,
    best_skew_entry,
    skew_corr_stats,
    compare_skew_summaries,
    batch_summarise_skew_corr_entries,
)

np.random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _orient_entries():
    return [
        make_orient_match_entry(0, 1, 45.0, 0.9, 36),
        make_orient_match_entry(1, 2, 90.0, 0.6, 18),
        make_orient_match_entry(2, 3, 135.0, 0.4, 36),
    ]


def _skew_entries():
    return [
        make_skew_corr_entry(0, 2.5, 0.8, "hough"),
        make_skew_corr_entry(1, -1.0, 0.95, "fft"),
        make_skew_corr_entry(2, 0.5, 0.6, "hough"),
    ]


# ── OrientMatchEntry ──────────────────────────────────────────────────────────

def test_make_orient_match_entry_fields():
    e = make_orient_match_entry(0, 1, 90.0, 0.8, 36)
    assert e.fragment_a == 0
    assert e.fragment_b == 1
    assert e.best_angle == pytest.approx(90.0)
    assert e.best_score == pytest.approx(0.8)
    assert e.n_angles_tested == 36


# ── summarise_orient_match_entries ────────────────────────────────────────────

def test_summarise_orient_empty():
    s = summarise_orient_match_entries([])
    assert s.n_entries == 0
    assert s.mean_score == pytest.approx(0.0)


def test_summarise_orient_normal():
    entries = _orient_entries()
    s = summarise_orient_match_entries(entries)
    assert s.n_entries == 3
    assert s.max_score == pytest.approx(0.9)
    assert s.min_score == pytest.approx(0.4)
    # only 0.9 >= 0.7
    assert s.high_score_count == 1


def test_summarise_orient_mean_score():
    entries = _orient_entries()
    s = summarise_orient_match_entries(entries)
    assert s.mean_score == pytest.approx((0.9 + 0.6 + 0.4) / 3)


def test_summarise_orient_mean_angle():
    entries = _orient_entries()
    s = summarise_orient_match_entries(entries)
    assert s.mean_angle == pytest.approx((45.0 + 90.0 + 135.0) / 3)


# ── Filters ───────────────────────────────────────────────────────────────────

def test_filter_high_orient_matches():
    entries = _orient_entries()
    high = filter_high_orient_matches(entries, threshold=0.7)
    assert all(e.best_score >= 0.7 for e in high)
    assert len(high) == 1


def test_filter_low_orient_matches():
    entries = _orient_entries()
    low = filter_low_orient_matches(entries, threshold=0.7)
    assert all(e.best_score < 0.7 for e in low)
    assert len(low) == 2


def test_filter_orient_by_score_range():
    entries = _orient_entries()
    ranged = filter_orient_by_score_range(entries, lo=0.5, hi=0.7)
    assert all(0.5 <= e.best_score <= 0.7 for e in ranged)


def test_filter_orient_by_max_angle():
    entries = _orient_entries()
    filtered = filter_orient_by_max_angle(entries, max_angle=90.0)
    assert all(e.best_angle <= 90.0 for e in filtered)
    assert len(filtered) == 2


def test_top_k_orient_match_entries():
    entries = _orient_entries()
    top = top_k_orient_match_entries(entries, 2)
    assert len(top) == 2
    assert top[0].best_score >= top[1].best_score


def test_best_orient_match_entry():
    entries = _orient_entries()
    best = best_orient_match_entry(entries)
    assert best.best_score == pytest.approx(0.9)


def test_best_orient_match_entry_empty():
    assert best_orient_match_entry([]) is None


def test_orient_match_stats_empty():
    stats = orient_match_stats([])
    assert stats["count"] == 0


def test_orient_match_stats_values():
    entries = _orient_entries()
    stats = orient_match_stats(entries)
    assert stats["min"] == pytest.approx(0.4)
    assert stats["max"] == pytest.approx(0.9)
    assert stats["count"] == pytest.approx(3.0)


def test_compare_orient_summaries():
    entries = _orient_entries()
    s1 = summarise_orient_match_entries(entries[:2])
    s2 = summarise_orient_match_entries(entries[1:])
    delta = compare_orient_summaries(s1, s2)
    assert "mean_score_delta" in delta
    assert "mean_angle_delta" in delta


def test_batch_summarise_orient_match_entries():
    groups = [_orient_entries()[:2], _orient_entries()[1:]]
    summaries = batch_summarise_orient_match_entries(groups)
    assert len(summaries) == 2


# ── SkewCorrEntry ─────────────────────────────────────────────────────────────

def test_make_skew_corr_entry_fields():
    e = make_skew_corr_entry(0, 3.5, 0.75, "hough")
    assert e.image_id == 0
    assert e.angle_deg == pytest.approx(3.5)
    assert e.confidence == pytest.approx(0.75)
    assert e.method == "hough"


def test_summarise_skew_empty():
    s = summarise_skew_corr_entries([])
    assert s.n_entries == 0
    assert s.dominant_method == ""


def test_summarise_skew_normal():
    entries = _skew_entries()
    s = summarise_skew_corr_entries(entries)
    assert s.n_entries == 3
    assert s.dominant_method == "hough"
    assert s.max_confidence == pytest.approx(0.95)
    assert s.min_confidence == pytest.approx(0.6)


def test_filter_high_confidence_skew():
    entries = _skew_entries()
    filtered = filter_high_confidence_skew(entries, threshold=0.8)
    assert all(e.confidence >= 0.8 for e in filtered)


def test_filter_skew_by_method():
    entries = _skew_entries()
    filtered = filter_skew_by_method(entries, "hough")
    assert all(e.method == "hough" for e in filtered)
    assert len(filtered) == 2


def test_filter_skew_by_angle_range():
    entries = _skew_entries()
    filtered = filter_skew_by_angle_range(entries, -1.5, 1.0)
    assert all(-1.5 <= e.angle_deg <= 1.0 for e in filtered)


def test_top_k_skew_entries():
    entries = _skew_entries()
    top = top_k_skew_entries(entries, 2)
    assert len(top) == 2
    assert top[0].confidence >= top[1].confidence


def test_best_skew_entry():
    entries = _skew_entries()
    best = best_skew_entry(entries)
    assert best.confidence == pytest.approx(0.95)


def test_best_skew_entry_empty():
    assert best_skew_entry([]) is None


def test_skew_corr_stats_empty():
    stats = skew_corr_stats([])
    assert stats["count"] == 0


def test_compare_skew_summaries():
    entries = _skew_entries()
    s1 = summarise_skew_corr_entries(entries[:2])
    s2 = summarise_skew_corr_entries(entries[1:])
    delta = compare_skew_summaries(s1, s2)
    assert "mean_confidence_delta" in delta
    assert "mean_angle_delta" in delta


def test_batch_summarise_skew_entries():
    groups = [_skew_entries()[:2], _skew_entries()[1:]]
    summaries = batch_summarise_skew_corr_entries(groups)
    assert len(summaries) == 2
