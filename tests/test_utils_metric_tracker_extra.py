"""Extra tests for puzzle_reconstruction/utils/metric_tracker.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.metric_tracker import (
    MetricRecord,
    MetricStats,
    TrackerConfig,
    MetricTracker,
    make_tracker,
    merge_trackers,
    compute_moving_average,
    export_metrics,
)


# ─── MetricRecord ─────────────────────────────────────────────────────────────

class TestMetricRecordExtra:
    def test_stores_name_and_value(self):
        r = MetricRecord(name="loss", value=0.5, step=0)
        assert r.name == "loss" and r.value == pytest.approx(0.5)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            MetricRecord(name="", value=1.0)

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            MetricRecord(name="acc", value=0.9, step=-1)

    def test_default_step_zero(self):
        r = MetricRecord(name="x", value=1.0)
        assert r.step == 0


# ─── MetricStats ──────────────────────────────────────────────────────────────

class TestMetricStatsExtra:
    def test_range_property(self):
        s = MetricStats(name="m", count=5, mean=0.5, std=0.1,
                        minimum=0.2, maximum=0.8, last=0.7)
        assert s.range == pytest.approx(0.6)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            MetricStats(name="", count=0, mean=0.0, std=0.0,
                        minimum=0.0, maximum=0.0, last=0.0)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            MetricStats(name="m", count=-1, mean=0.0, std=0.0,
                        minimum=0.0, maximum=0.0, last=0.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            MetricStats(name="m", count=1, mean=0.0, std=-0.1,
                        minimum=0.0, maximum=0.0, last=0.0)


# ─── TrackerConfig ────────────────────────────────────────────────────────────

class TestTrackerConfigExtra:
    def test_default_max_history(self):
        assert TrackerConfig().max_history == 0

    def test_default_namespace(self):
        assert TrackerConfig().namespace == "default"

    def test_negative_max_history_raises(self):
        with pytest.raises(ValueError):
            TrackerConfig(max_history=-1)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            TrackerConfig(namespace="")


# ─── MetricTracker ────────────────────────────────────────────────────────────

class TestMetricTrackerExtra:
    def test_record_and_last_value(self):
        t = MetricTracker()
        t.record("loss", 0.5)
        assert t.last_value("loss") == pytest.approx(0.5)

    def test_unknown_metric_last_value_none(self):
        t = MetricTracker()
        assert t.last_value("unknown") is None

    def test_metric_names(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("b", 2.0)
        assert set(t.metric_names()) == {"a", "b"}

    def test_has_metric(self):
        t = MetricTracker()
        t.record("x", 0.0)
        assert t.has_metric("x") is True
        assert t.has_metric("y") is False

    def test_size_all(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("b", 2.0)
        assert t.size() == 2

    def test_size_named(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("a", 2.0)
        assert t.size("a") == 2

    def test_clear_named(self):
        t = MetricTracker()
        t.record("a", 1.0)
        removed = t.clear("a")
        assert removed == 1
        assert not t.has_metric("a")

    def test_clear_all(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("b", 2.0)
        removed = t.clear()
        assert removed == 2

    def test_max_history_enforced(self):
        cfg = TrackerConfig(max_history=3)
        t = MetricTracker(cfg)
        for i in range(10):
            t.record("x", float(i))
        assert t.size("x") == 3

    def test_namespace(self):
        cfg = TrackerConfig(namespace="experiment")
        t = MetricTracker(cfg)
        assert t.namespace == "experiment"

    def test_stats_none_for_unknown(self):
        t = MetricTracker()
        assert t.stats("nonexistent") is None

    def test_stats_correctness(self):
        t = MetricTracker()
        for v in [1.0, 2.0, 3.0]:
            t.record("m", v)
        s = t.stats("m")
        assert s.count == 3
        assert s.mean == pytest.approx(2.0)
        assert s.minimum == pytest.approx(1.0)
        assert s.maximum == pytest.approx(3.0)

    def test_record_dict(self):
        t = MetricTracker()
        t.record_dict({"a": 0.1, "b": 0.2}, step=1)
        assert t.has_metric("a") and t.has_metric("b")

    def test_values_list(self):
        t = MetricTracker()
        t.record("v", 1.0)
        t.record("v", 2.0)
        assert t.values("v") == pytest.approx([1.0, 2.0])


# ─── make_tracker ─────────────────────────────────────────────────────────────

class TestMakeTrackerExtra:
    def test_returns_tracker(self):
        t = make_tracker()
        assert isinstance(t, MetricTracker)

    def test_custom_namespace(self):
        t = make_tracker(namespace="run1")
        assert t.namespace == "run1"

    def test_max_history_applied(self):
        t = make_tracker(max_history=2)
        for i in range(5):
            t.record("x", float(i))
        assert t.size("x") == 2


# ─── merge_trackers ───────────────────────────────────────────────────────────

class TestMergeTrackersExtra:
    def test_copies_records(self):
        src = make_tracker()
        src.record("a", 1.0)
        src.record("a", 2.0)
        dst = make_tracker()
        count = merge_trackers(dst, src)
        assert count == 2
        assert dst.size("a") == 2

    def test_empty_source(self):
        dst = make_tracker()
        count = merge_trackers(dst, make_tracker())
        assert count == 0


# ─── compute_moving_average ───────────────────────────────────────────────────

class TestComputeMovingAverageExtra:
    def test_empty_metric_returns_empty(self):
        t = make_tracker()
        assert compute_moving_average(t, "x") == []

    def test_window_lt_1_raises(self):
        t = make_tracker()
        t.record("x", 1.0)
        with pytest.raises(ValueError):
            compute_moving_average(t, "x", window=0)

    def test_same_length_as_history(self):
        t = make_tracker()
        for v in [1.0, 2.0, 3.0, 4.0]:
            t.record("m", v)
        ma = compute_moving_average(t, "m", window=2)
        assert len(ma) == 4

    def test_single_value_window_1(self):
        t = make_tracker()
        t.record("x", 5.0)
        ma = compute_moving_average(t, "x", window=1)
        assert ma == pytest.approx([5.0])


# ─── export_metrics ───────────────────────────────────────────────────────────

class TestExportMetricsExtra:
    def test_returns_dict(self):
        t = make_tracker()
        t.record("a", 1.0, step=0)
        result = export_metrics(t)
        assert isinstance(result, dict) and "a" in result

    def test_step_value_pairs(self):
        t = make_tracker()
        t.record("x", 0.5, step=3)
        pairs = export_metrics(t)["x"]
        assert pairs[0] == (3, pytest.approx(0.5))

    def test_empty_tracker(self):
        assert export_metrics(make_tracker()) == {}
