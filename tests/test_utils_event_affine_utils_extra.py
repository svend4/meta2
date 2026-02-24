"""Extra tests for puzzle_reconstruction/utils/event_affine_utils.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _evt(eid=0, name="test", level="info", ts=1.0, is_error=False) -> EventRecordEntry:
    return EventRecordEntry(event_id=eid, name=name, level=level,
                             timestamp=ts, is_error=is_error)


def _evts(n=5) -> list:
    return [_evt(eid=i, ts=float(i)) for i in range(n)]


def _aff(idx1=0, idx2=1, score=0.7, n_inliers=10) -> AffineMatchEntry:
    return AffineMatchEntry(idx1=idx1, idx2=idx2, score=score,
                             n_inliers=n_inliers, reprojection_error=1.5)


def _affs(n=4) -> list:
    return [_aff(idx1=i, score=float(i+1)/n) for i in range(n)]


# ─── EventRecordConfig ────────────────────────────────────────────────────────

class TestEventRecordConfigExtra:
    def test_default_min_level(self):
        assert EventRecordConfig().min_level == "debug"

    def test_default_namespace(self):
        assert EventRecordConfig().namespace == "default"

    def test_custom_values(self):
        cfg = EventRecordConfig(min_level="warning", namespace="pipeline")
        assert cfg.min_level == "warning" and cfg.namespace == "pipeline"


# ─── EventRecordEntry ─────────────────────────────────────────────────────────

class TestEventRecordEntryExtra:
    def test_stores_event_id(self):
        assert _evt(eid=5).event_id == 5

    def test_stores_name(self):
        assert _evt(name="fit").name == "fit"

    def test_stores_level(self):
        assert _evt(level="error").level == "error"

    def test_stores_timestamp(self):
        assert _evt(ts=42.5).timestamp == pytest.approx(42.5)

    def test_stores_is_error(self):
        assert _evt(is_error=True).is_error is True

    def test_default_is_error_false(self):
        e = EventRecordEntry(event_id=0, name="x", level="info", timestamp=0.0)
        assert e.is_error is False


# ─── make_event_record_entry ──────────────────────────────────────────────────

class TestMakeEventRecordEntryExtra:
    def test_returns_entry(self):
        e = make_event_record_entry(0, "test", "info", 1.0)
        assert isinstance(e, EventRecordEntry)

    def test_values_stored(self):
        e = make_event_record_entry(3, "fit", "error", 2.5, is_error=True)
        assert e.event_id == 3 and e.name == "fit" and e.is_error is True


# ─── summarise_event_record_entries ───────────────────────────────────────────

class TestSummariseEventRecordEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_event_record_entries(_evts()), EventRecordSummary)

    def test_n_entries_correct(self):
        assert summarise_event_record_entries(_evts(5)).n_entries == 5

    def test_empty_entries(self):
        s = summarise_event_record_entries([])
        assert s.n_entries == 0

    def test_error_count(self):
        entries = [_evt(is_error=True), _evt(is_error=False), _evt(is_error=True)]
        s = summarise_event_record_entries(entries)
        assert s.n_errors == 2

    def test_unique_names(self):
        entries = [_evt(name="a"), _evt(name="b"), _evt(name="a")]
        s = summarise_event_record_entries(entries)
        assert s.unique_names == 2


# ─── filter functions ─────────────────────────────────────────────────────────

class TestFilterEventRecordExtra:
    def test_filter_error_events(self):
        entries = [_evt(is_error=True), _evt(is_error=False)]
        result = filter_error_events(entries)
        assert all(e.is_error for e in result)

    def test_filter_by_level(self):
        entries = [_evt(level="error"), _evt(level="info")]
        result = filter_events_by_level(entries, "error")
        assert all(e.level == "error" for e in result)

    def test_filter_by_name(self):
        entries = [_evt(name="foo"), _evt(name="bar")]
        result = filter_events_by_name(entries, "foo")
        assert all(e.name == "foo" for e in result)

    def test_filter_by_time_range(self):
        entries = [_evt(ts=1.0), _evt(ts=3.0), _evt(ts=5.0)]
        result = filter_events_by_time_range(entries, 2.0, 4.0)
        assert all(2.0 <= e.timestamp <= 4.0 for e in result)

    def test_empty_input(self):
        assert filter_error_events([]) == []


# ─── top_k and latest ─────────────────────────────────────────────────────────

class TestTopKLatestEventExtra:
    def test_top_k_recent(self):
        entries = _evts(5)
        result = top_k_recent_events(entries, 2)
        assert len(result) == 2
        ts = [e.timestamp for e in result]
        assert ts == sorted(ts, reverse=True)

    def test_latest_returns_max_ts(self):
        entries = [_evt(ts=1.0), _evt(ts=5.0), _evt(ts=3.0)]
        latest = latest_event_entry(entries)
        assert latest.timestamp == pytest.approx(5.0)

    def test_latest_empty_is_none(self):
        assert latest_event_entry([]) is None


# ─── event_record_stats ───────────────────────────────────────────────────────

class TestEventRecordStatsExtra:
    def test_returns_dict(self):
        assert isinstance(event_record_stats(_evts()), dict)

    def test_keys_present(self):
        stats = event_record_stats(_evts(3))
        for k in ("count", "min", "max", "span"):
            assert k in stats

    def test_empty_entries(self):
        assert event_record_stats([])["count"] == 0


# ─── compare_event_summaries ──────────────────────────────────────────────────

class TestCompareEventSummariesExtra:
    def test_returns_dict(self):
        s = summarise_event_record_entries(_evts(3))
        assert isinstance(compare_event_summaries(s, s), dict)

    def test_identical_zero_delta(self):
        s = summarise_event_record_entries(_evts(3))
        d = compare_event_summaries(s, s)
        assert d["error_rate_delta"] == pytest.approx(0.0)


# ─── batch_summarise_event_record_entries ─────────────────────────────────────

class TestBatchSummariseEventRecordEntriesExtra:
    def test_returns_list(self):
        assert isinstance(batch_summarise_event_record_entries([_evts(2)]), list)

    def test_length_matches(self):
        result = batch_summarise_event_record_entries([_evts(2), _evts(3)])
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_summarise_event_record_entries([]) == []


# ─── AffineMatchConfig ────────────────────────────────────────────────────────

class TestAffineMatchConfigExtra:
    def test_default_min_score(self):
        assert AffineMatchConfig().min_score == pytest.approx(0.0)

    def test_default_min_inliers(self):
        assert AffineMatchConfig().min_inliers == 0

    def test_custom_values(self):
        cfg = AffineMatchConfig(min_score=0.5, min_inliers=10)
        assert cfg.min_score == pytest.approx(0.5) and cfg.min_inliers == 10


# ─── AffineMatchEntry ─────────────────────────────────────────────────────────

class TestAffineMatchEntryExtra:
    def test_stores_idx1_idx2(self):
        e = _aff(idx1=3, idx2=7)
        assert e.idx1 == 3 and e.idx2 == 7

    def test_stores_score(self):
        assert _aff(score=0.85).score == pytest.approx(0.85)

    def test_stores_n_inliers(self):
        assert _aff(n_inliers=20).n_inliers == 20

    def test_default_has_transform_true(self):
        e = AffineMatchEntry(idx1=0, idx2=1, score=0.5, n_inliers=5,
                              reprojection_error=1.0)
        assert e.has_transform is True


# ─── make_affine_match_entry ──────────────────────────────────────────────────

class TestMakeAffineMatchEntryExtra:
    def test_returns_entry(self):
        e = make_affine_match_entry(0, 1, 0.7, 10, 1.5)
        assert isinstance(e, AffineMatchEntry)

    def test_values_stored(self):
        e = make_affine_match_entry(2, 5, 0.9, 15, 0.5, has_transform=False)
        assert e.idx1 == 2 and e.score == pytest.approx(0.9)
        assert e.has_transform is False


# ─── summarise and filter affine ──────────────────────────────────────────────

class TestSummariseAffineMatchEntriesExtra:
    def test_returns_summary(self):
        assert isinstance(summarise_affine_match_entries(_affs()), AffineMatchSummary)

    def test_n_entries_correct(self):
        assert summarise_affine_match_entries(_affs(5)).n_entries == 5

    def test_empty_entries(self):
        s = summarise_affine_match_entries([])
        assert s.n_entries == 0

    def test_filter_strong(self):
        entries = [_aff(score=0.3), _aff(score=0.8)]
        result = filter_strong_affine_matches(entries, threshold=0.5)
        assert all(e.score >= 0.5 for e in result)

    def test_filter_weak(self):
        entries = [_aff(score=0.3), _aff(score=0.8)]
        result = filter_weak_affine_matches(entries, threshold=0.5)
        assert all(e.score < 0.5 for e in result)

    def test_filter_by_inliers(self):
        entries = [_aff(n_inliers=3), _aff(n_inliers=15)]
        result = filter_affine_by_inliers(entries, min_inliers=10)
        assert all(e.n_inliers >= 10 for e in result)

    def test_filter_with_transform(self):
        entries = [_aff(), AffineMatchEntry(idx1=0, idx2=1, score=0.5,
                                             n_inliers=5, reprojection_error=1.0,
                                             has_transform=False)]
        result = filter_affine_with_transform(entries)
        assert all(e.has_transform for e in result)

    def test_top_k_by_score(self):
        result = top_k_affine_match_entries(_affs(5), 3)
        assert len(result) == 3

    def test_best_returns_highest_score(self):
        entries = [_aff(score=0.2), _aff(score=0.9)]
        best = best_affine_match_entry(entries)
        assert best.score == pytest.approx(0.9)

    def test_best_empty_is_none(self):
        assert best_affine_match_entry([]) is None


# ─── affine_match_stats ───────────────────────────────────────────────────────

class TestAffineMatchStatsExtra:
    def test_returns_dict(self):
        assert isinstance(affine_match_stats(_affs()), dict)

    def test_keys_present(self):
        for k in ("count", "mean", "std", "min", "max"):
            assert k in affine_match_stats(_affs(3))

    def test_empty_entries(self):
        assert affine_match_stats([])["count"] == 0


# ─── compare_affine_summaries ─────────────────────────────────────────────────

class TestCompareAffineSummariesExtra:
    def test_returns_dict(self):
        s = summarise_affine_match_entries(_affs(3))
        assert isinstance(compare_affine_summaries(s, s), dict)

    def test_identical_zero_delta(self):
        s = summarise_affine_match_entries(_affs(3))
        d = compare_affine_summaries(s, s)
        assert d["mean_score_delta"] == pytest.approx(0.0)


# ─── batch_summarise_affine_match_entries ─────────────────────────────────────

class TestBatchSummariseAffineMatchEntriesExtra:
    def test_returns_list(self):
        assert isinstance(batch_summarise_affine_match_entries([_affs(2)]), list)

    def test_length_matches(self):
        result = batch_summarise_affine_match_entries([_affs(2), _affs(3)])
        assert len(result) == 2

    def test_empty_groups(self):
        assert batch_summarise_affine_match_entries([]) == []
