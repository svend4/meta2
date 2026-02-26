"""Tests for puzzle_reconstruction.utils.event_affine_utils."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.event_affine_utils import (
    EventRecordConfig,
    EventRecordEntry,
    EventRecordSummary,
    make_event_record_entry,
    summarise_event_record_entries,
    filter_error_events,
    filter_events_by_level,
    filter_events_by_name,
    filter_events_by_time_range,
    top_k_recent_events,
    latest_event_entry,
    event_record_stats,
    compare_event_summaries,
    batch_summarise_event_record_entries,
    AffineMatchConfig,
    AffineMatchEntry,
    AffineMatchSummary,
    make_affine_match_entry,
    summarise_affine_match_entries,
    filter_strong_affine_matches,
    filter_weak_affine_matches,
    filter_affine_by_inliers,
    filter_affine_with_transform,
    top_k_affine_match_entries,
    best_affine_match_entry,
    affine_match_stats,
    compare_affine_summaries,
    batch_summarise_affine_match_entries,
)

np.random.seed(88)


def _make_event_entries(n=8):
    levels = ["debug", "info", "warning", "error"]
    entries = []
    for i in range(n):
        lvl = levels[i % len(levels)]
        entries.append(EventRecordEntry(
            event_id=i, name=f"event_{i % 3}",
            level=lvl, timestamp=float(i * 10),
            is_error=(lvl == "error"),
        ))
    return entries


def _make_affine_entries(n=8):
    entries = []
    for i in range(n):
        s = float(np.random.uniform(0.1, 1.0))
        entries.append(AffineMatchEntry(
            idx1=i, idx2=i+1, score=s,
            n_inliers=int(s * 20),
            reprojection_error=float(np.random.uniform(0.5, 5.0)),
            has_transform=s > 0.3,
        ))
    return entries


# ── 1. EventRecordConfig defaults ────────────────────────────────────────────
def test_event_config_defaults():
    cfg = EventRecordConfig()
    assert cfg.min_level == "debug"
    assert cfg.namespace == "default"


# ── 2. make_event_record_entry ───────────────────────────────────────────────
def test_make_event_entry():
    e = make_event_record_entry(1, "load", "info", 100.0, False)
    assert e.event_id == 1
    assert e.name == "load"
    assert e.level == "info"
    assert e.timestamp == 100.0
    assert e.is_error is False


# ── 3. summarise_event_record_entries empty ───────────────────────────────────
def test_summarise_events_empty():
    s = summarise_event_record_entries([])
    assert s.n_entries == 0
    assert s.error_rate == 0.0
    assert s.time_span == 0.0


# ── 4. summarise_event_record_entries nonempty ────────────────────────────────
def test_summarise_events_nonempty():
    entries = _make_event_entries(8)
    s = summarise_event_record_entries(entries)
    assert s.n_entries == 8
    assert s.n_warnings >= 0
    assert s.time_span >= 0.0
    assert s.unique_names <= 8


# ── 5. filter_error_events ───────────────────────────────────────────────────
def test_filter_error_events():
    entries = _make_event_entries(8)
    errors = filter_error_events(entries)
    assert all(e.is_error for e in errors)


# ── 6. filter_events_by_level ────────────────────────────────────────────────
def test_filter_by_level():
    entries = _make_event_entries(8)
    info = filter_events_by_level(entries, "info")
    assert all(e.level == "info" for e in info)


# ── 7. filter_events_by_name ─────────────────────────────────────────────────
def test_filter_by_name():
    entries = _make_event_entries(9)
    named = filter_events_by_name(entries, "event_1")
    assert all(e.name == "event_1" for e in named)


# ── 8. filter_events_by_time_range ───────────────────────────────────────────
def test_filter_by_time_range():
    entries = _make_event_entries(10)
    filtered = filter_events_by_time_range(entries, 20.0, 60.0)
    assert all(20.0 <= e.timestamp <= 60.0 for e in filtered)


# ── 9. top_k_recent_events ───────────────────────────────────────────────────
def test_top_k_recent():
    entries = _make_event_entries(8)
    top3 = top_k_recent_events(entries, 3)
    assert len(top3) == 3
    timestamps = [e.timestamp for e in top3]
    assert timestamps == sorted(timestamps, reverse=True)


# ── 10. latest_event_entry ───────────────────────────────────────────────────
def test_latest_event():
    entries = _make_event_entries(8)
    latest = latest_event_entry(entries)
    assert latest is not None
    assert latest.timestamp == max(e.timestamp for e in entries)


def test_latest_event_empty():
    assert latest_event_entry([]) is None


# ── 11. event_record_stats ───────────────────────────────────────────────────
def test_event_stats():
    entries = _make_event_entries(8)
    stats = event_record_stats(entries)
    assert stats["count"] == 8.0
    assert stats["min"] <= stats["max"]
    assert "error_rate" in stats


def test_event_stats_empty():
    stats = event_record_stats([])
    assert stats["count"] == 0


# ── 12. compare_event_summaries ──────────────────────────────────────────────
def test_compare_event_summaries():
    ea = _make_event_entries(8)
    eb = _make_event_entries(4)
    sa = summarise_event_record_entries(ea)
    sb = summarise_event_record_entries(eb)
    delta = compare_event_summaries(sa, sb)
    assert "error_rate_delta" in delta
    assert "n_errors_delta" in delta


# ── 13. batch_summarise_event_record_entries ──────────────────────────────────
def test_batch_event_summaries():
    groups = [_make_event_entries(5), _make_event_entries(3)]
    summaries = batch_summarise_event_record_entries(groups)
    assert len(summaries) == 2
    assert summaries[0].n_entries == 5


# ── 14. AffineMatchConfig defaults ───────────────────────────────────────────
def test_affine_config_defaults():
    cfg = AffineMatchConfig()
    assert cfg.min_score == 0.0
    assert cfg.min_inliers == 0


# ── 15. make_affine_match_entry ──────────────────────────────────────────────
def test_make_affine_entry():
    e = make_affine_match_entry(0, 1, 0.7, 10, 1.5, True)
    assert e.idx1 == 0
    assert e.idx2 == 1
    assert e.score == 0.7
    assert e.n_inliers == 10
    assert e.has_transform is True


# ── 16. summarise_affine_match_entries empty ──────────────────────────────────
def test_summarise_affine_empty():
    s = summarise_affine_match_entries([])
    assert s.n_entries == 0
    assert s.mean_score == 0.0


# ── 17. summarise_affine_match_entries nonempty ───────────────────────────────
def test_summarise_affine_nonempty():
    entries = _make_affine_entries(10)
    s = summarise_affine_match_entries(entries)
    assert s.n_entries == 10
    assert s.min_score <= s.mean_score <= s.max_score


# ── 18. filter_strong_affine_matches ─────────────────────────────────────────
def test_filter_strong_affine():
    entries = _make_affine_entries(10)
    strong = filter_strong_affine_matches(entries, 0.5)
    assert all(e.score >= 0.5 for e in strong)


# ── 19. filter_weak_affine_matches ───────────────────────────────────────────
def test_filter_weak_affine():
    entries = _make_affine_entries(10)
    weak = filter_weak_affine_matches(entries, 0.5)
    assert all(e.score < 0.5 for e in weak)


# ── 20. filter_affine_by_inliers ─────────────────────────────────────────────
def test_filter_affine_inliers():
    entries = _make_affine_entries(10)
    filtered = filter_affine_by_inliers(entries, 5)
    assert all(e.n_inliers >= 5 for e in filtered)


# ── 21. filter_affine_with_transform ─────────────────────────────────────────
def test_filter_with_transform():
    entries = _make_affine_entries(10)
    with_t = filter_affine_with_transform(entries)
    assert all(e.has_transform for e in with_t)


# ── 22. top_k_affine_match_entries ───────────────────────────────────────────
def test_top_k_affine():
    entries = _make_affine_entries(10)
    top3 = top_k_affine_match_entries(entries, 3)
    assert len(top3) == 3
    scores = [e.score for e in top3]
    assert scores == sorted(scores, reverse=True)


# ── 23. best_affine_match_entry ──────────────────────────────────────────────
def test_best_affine():
    entries = _make_affine_entries(10)
    best = best_affine_match_entry(entries)
    assert best is not None
    assert best.score == max(e.score for e in entries)


def test_best_affine_empty():
    assert best_affine_match_entry([]) is None


# ── 24. affine_match_stats ───────────────────────────────────────────────────
def test_affine_stats():
    entries = _make_affine_entries(10)
    stats = affine_match_stats(entries)
    assert stats["count"] == 10.0
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_affine_stats_empty():
    stats = affine_match_stats([])
    assert stats["count"] == 0


# ── 25. compare_affine_summaries ─────────────────────────────────────────────
def test_compare_affine():
    ea = _make_affine_entries(8)
    eb = _make_affine_entries(6)
    sa = summarise_affine_match_entries(ea)
    sb = summarise_affine_match_entries(eb)
    delta = compare_affine_summaries(sa, sb)
    assert "mean_score_delta" in delta
    assert "mean_inliers_delta" in delta


# ── 26. batch_summarise_affine_match_entries ──────────────────────────────────
def test_batch_affine():
    groups = [_make_affine_entries(5), _make_affine_entries(3)]
    summaries = batch_summarise_affine_match_entries(groups)
    assert len(summaries) == 2
    assert summaries[0].n_entries == 5
