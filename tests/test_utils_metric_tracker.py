"""Tests for puzzle_reconstruction.utils.metric_tracker"""
import numpy as np
import pytest
from puzzle_reconstruction.utils.metric_tracker import (
    MetricRecord, MetricStats, TrackerConfig, MetricTracker,
    make_tracker, merge_trackers, compute_moving_average, export_metrics,
)


# ── MetricRecord ──────────────────────────────────────────────────────────────

def test_metric_record_empty_name_raises():
    with pytest.raises(ValueError):
        MetricRecord(name="", value=1.0)


def test_metric_record_negative_step_raises():
    with pytest.raises(ValueError):
        MetricRecord(name="loss", value=1.0, step=-1)


def test_metric_record_defaults():
    rec = MetricRecord(name="accuracy", value=0.95)
    assert rec.step == 0


# ── MetricStats ───────────────────────────────────────────────────────────────

def test_metric_stats_range():
    s = MetricStats(name="loss", count=3, mean=0.5,
                    std=0.1, minimum=0.3, maximum=0.7, last=0.6)
    assert abs(s.range - 0.4) < 1e-9


def test_metric_stats_empty_name_raises():
    with pytest.raises(ValueError):
        MetricStats(name="", count=1, mean=0.5, std=0.1,
                    minimum=0.3, maximum=0.7, last=0.6)


def test_metric_stats_negative_count_raises():
    with pytest.raises(ValueError):
        MetricStats(name="m", count=-1, mean=0.0, std=0.0,
                    minimum=0.0, maximum=0.0, last=0.0)


def test_metric_stats_negative_std_raises():
    with pytest.raises(ValueError):
        MetricStats(name="m", count=1, mean=0.5, std=-0.1,
                    minimum=0.4, maximum=0.6, last=0.5)


# ── TrackerConfig ─────────────────────────────────────────────────────────────

def test_tracker_config_negative_max_history_raises():
    with pytest.raises(ValueError):
        TrackerConfig(max_history=-1)


def test_tracker_config_empty_namespace_raises():
    with pytest.raises(ValueError):
        TrackerConfig(namespace="")


def test_tracker_config_defaults():
    cfg = TrackerConfig()
    assert cfg.max_history == 0
    assert cfg.namespace == "default"


# ── MetricTracker ─────────────────────────────────────────────────────────────

def test_metric_tracker_record_and_retrieve():
    t = MetricTracker()
    t.record("loss", 0.5, step=1)
    assert t.last_value("loss") == pytest.approx(0.5)


def test_metric_tracker_values_list():
    t = MetricTracker()
    for v in [0.1, 0.2, 0.3]:
        t.record("acc", v)
    assert t.values("acc") == [0.1, 0.2, 0.3]


def test_metric_tracker_empty_name_raises():
    t = MetricTracker()
    with pytest.raises(ValueError):
        t.record("", 1.0)


def test_metric_tracker_negative_step_raises():
    t = MetricTracker()
    with pytest.raises(ValueError):
        t.record("loss", 1.0, step=-1)


def test_metric_tracker_history_returns_list():
    t = MetricTracker()
    t.record("m", 0.5)
    h = t.history("m")
    assert isinstance(h, list)
    assert len(h) == 1


def test_metric_tracker_last_value_none_if_missing():
    t = MetricTracker()
    assert t.last_value("nonexistent") is None


def test_metric_tracker_has_metric():
    t = MetricTracker()
    assert not t.has_metric("x")
    t.record("x", 1.0)
    assert t.has_metric("x")


def test_metric_tracker_stats():
    t = MetricTracker()
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in vals:
        t.record("m", v)
    s = t.stats("m")
    assert isinstance(s, MetricStats)
    assert s.count == 5
    assert s.mean == pytest.approx(3.0)
    assert s.minimum == pytest.approx(1.0)
    assert s.maximum == pytest.approx(5.0)


def test_metric_tracker_stats_none_if_missing():
    t = MetricTracker()
    assert t.stats("ghost") is None


def test_metric_tracker_all_stats():
    t = MetricTracker()
    t.record("a", 1.0)
    t.record("b", 2.0)
    s = t.all_stats()
    assert "a" in s and "b" in s


def test_metric_tracker_clear_specific():
    t = MetricTracker()
    t.record("a", 1.0)
    t.record("b", 2.0)
    n = t.clear("a")
    assert n == 1
    assert not t.has_metric("a")
    assert t.has_metric("b")


def test_metric_tracker_clear_all():
    t = MetricTracker()
    t.record("a", 1.0)
    t.record("b", 2.0)
    n = t.clear()
    assert n == 2
    assert t.metric_names() == []


def test_metric_tracker_max_history():
    t = MetricTracker(TrackerConfig(max_history=3))
    for i in range(10):
        t.record("m", float(i))
    assert t.size("m") == 3


def test_metric_tracker_record_dict():
    t = MetricTracker()
    t.record_dict({"loss": 0.5, "acc": 0.95}, step=2)
    assert t.has_metric("loss")
    assert t.has_metric("acc")


def test_metric_tracker_namespace():
    t = MetricTracker(TrackerConfig(namespace="experiment1"))
    assert t.namespace == "experiment1"


def test_metric_tracker_size_global():
    t = MetricTracker()
    t.record("a", 1.0)
    t.record("a", 2.0)
    t.record("b", 3.0)
    assert t.size() == 3


# ── make_tracker ──────────────────────────────────────────────────────────────

def test_make_tracker_returns_metric_tracker():
    t = make_tracker(max_history=5, namespace="ns")
    assert isinstance(t, MetricTracker)
    assert t.namespace == "ns"


# ── merge_trackers ────────────────────────────────────────────────────────────

def test_merge_trackers_copies_all():
    src = MetricTracker()
    src.record("loss", 0.5)
    src.record("acc", 0.9)
    dst = MetricTracker()
    n = merge_trackers(dst, src)
    assert n == 2
    assert dst.has_metric("loss")
    assert dst.has_metric("acc")


# ── compute_moving_average ────────────────────────────────────────────────────

def test_compute_moving_average_basic():
    t = MetricTracker()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        t.record("m", v)
    ma = compute_moving_average(t, "m", window=3)
    assert len(ma) == 5
    assert ma[2] == pytest.approx(2.0)  # mean of [1,2,3]


def test_compute_moving_average_window_1():
    t = MetricTracker()
    vals = [1.0, 2.0, 3.0]
    for v in vals:
        t.record("m", v)
    ma = compute_moving_average(t, "m", window=1)
    assert ma == pytest.approx(vals)


def test_compute_moving_average_invalid_window_raises():
    t = MetricTracker()
    with pytest.raises(ValueError):
        compute_moving_average(t, "m", window=0)


def test_compute_moving_average_empty_metric():
    t = MetricTracker()
    ma = compute_moving_average(t, "nonexistent")
    assert ma == []


# ── export_metrics ────────────────────────────────────────────────────────────

def test_export_metrics_structure():
    t = MetricTracker()
    t.record("loss", 0.5, step=0)
    t.record("loss", 0.3, step=1)
    exported = export_metrics(t)
    assert "loss" in exported
    assert exported["loss"] == [(0, 0.5), (1, 0.3)]


def test_export_metrics_empty():
    t = MetricTracker()
    assert export_metrics(t) == {}
