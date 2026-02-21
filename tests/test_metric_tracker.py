"""Тесты для puzzle_reconstruction.utils.metric_tracker."""
import pytest

from puzzle_reconstruction.utils.metric_tracker import (
    MetricRecord,
    MetricStats,
    MetricTracker,
    TrackerConfig,
    compute_moving_average,
    export_metrics,
    make_tracker,
    merge_trackers,
)


# ─── TestMetricRecord ─────────────────────────────────────────────────────────

class TestMetricRecord:
    def test_basic_construction(self):
        r = MetricRecord(name="loss", value=0.5, step=10)
        assert r.name == "loss"
        assert r.value == 0.5
        assert r.step == 10

    def test_default_step(self):
        r = MetricRecord(name="acc", value=0.9)
        assert r.step == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            MetricRecord(name="", value=0.5)

    def test_negative_step_raises(self):
        with pytest.raises(ValueError):
            MetricRecord(name="loss", value=0.5, step=-1)

    def test_negative_value_allowed(self):
        r = MetricRecord(name="delta", value=-1.0, step=0)
        assert r.value == -1.0


# ─── TestMetricStats ──────────────────────────────────────────────────────────

class TestMetricStats:
    def _make(self, count=10, mean=0.5, std=0.1, minimum=0.2,
              maximum=0.9, last=0.6):
        return MetricStats(name="loss", count=count, mean=mean,
                           std=std, minimum=minimum, maximum=maximum,
                           last=last)

    def test_basic_construction(self):
        s = self._make()
        assert s.name == "loss"
        assert s.count == 10

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            MetricStats(name="", count=0, mean=0.0, std=0.0,
                        minimum=0.0, maximum=0.0, last=0.0)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            MetricStats(name="x", count=-1, mean=0.0, std=0.0,
                        minimum=0.0, maximum=0.0, last=0.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            MetricStats(name="x", count=1, mean=0.5, std=-0.1,
                        minimum=0.0, maximum=1.0, last=0.5)

    def test_range_property(self):
        s = self._make(minimum=0.2, maximum=0.9)
        assert abs(s.range - 0.7) < 1e-9

    def test_range_zero(self):
        s = self._make(minimum=0.5, maximum=0.5)
        assert s.range == 0.0


# ─── TestTrackerConfig ────────────────────────────────────────────────────────

class TestTrackerConfig:
    def test_defaults(self):
        cfg = TrackerConfig()
        assert cfg.max_history == 0
        assert cfg.namespace == "default"

    def test_negative_max_history_raises(self):
        with pytest.raises(ValueError):
            TrackerConfig(max_history=-1)

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError):
            TrackerConfig(namespace="")

    def test_zero_max_history_ok(self):
        cfg = TrackerConfig(max_history=0)
        assert cfg.max_history == 0


# ─── TestMetricTrackerRecord ──────────────────────────────────────────────────

class TestMetricTrackerRecord:
    def test_record_stores_value(self):
        t = MetricTracker()
        t.record("loss", 0.5, step=1)
        assert t.last_value("loss") == 0.5

    def test_record_empty_name_raises(self):
        t = MetricTracker()
        with pytest.raises(ValueError):
            t.record("", 0.5)

    def test_record_negative_step_raises(self):
        t = MetricTracker()
        with pytest.raises(ValueError):
            t.record("loss", 0.5, step=-1)

    def test_record_multiple_values(self):
        t = MetricTracker()
        for i in range(5):
            t.record("acc", float(i) * 0.1, step=i)
        assert len(t.values("acc")) == 5

    def test_record_dict(self):
        t = MetricTracker()
        t.record_dict({"loss": 0.4, "acc": 0.9}, step=2)
        assert t.last_value("loss") == 0.4
        assert t.last_value("acc") == 0.9

    def test_max_history_enforced(self):
        t = MetricTracker(TrackerConfig(max_history=3))
        for i in range(10):
            t.record("x", float(i))
        assert len(t.values("x")) == 3
        assert t.last_value("x") == 9.0

    def test_max_history_keeps_recent(self):
        t = MetricTracker(TrackerConfig(max_history=2))
        t.record("v", 1.0, step=0)
        t.record("v", 2.0, step=1)
        t.record("v", 3.0, step=2)
        vals = t.values("v")
        assert vals == [2.0, 3.0]


# ─── TestMetricTrackerRead ────────────────────────────────────────────────────

class TestMetricTrackerRead:
    def _tracker_with_data(self):
        t = MetricTracker()
        for i in range(5):
            t.record("loss", 1.0 / (i + 1), step=i)
            t.record("acc", float(i) * 0.2, step=i)
        return t

    def test_last_value_none_for_unknown(self):
        t = MetricTracker()
        assert t.last_value("unknown") is None

    def test_values_empty_for_unknown(self):
        t = MetricTracker()
        assert t.values("unknown") == []

    def test_history_returns_list_of_records(self):
        t = self._tracker_with_data()
        hist = t.history("loss")
        assert len(hist) == 5
        assert all(isinstance(r, MetricRecord) for r in hist)

    def test_history_returns_copy(self):
        t = self._tracker_with_data()
        hist = t.history("loss")
        hist.clear()
        assert len(t.history("loss")) == 5

    def test_metric_names_returns_all(self):
        t = self._tracker_with_data()
        names = t.metric_names()
        assert "loss" in names
        assert "acc" in names

    def test_has_metric_true(self):
        t = self._tracker_with_data()
        assert t.has_metric("loss") is True

    def test_has_metric_false_for_unknown(self):
        t = MetricTracker()
        assert t.has_metric("x") is False

    def test_namespace_property(self):
        t = MetricTracker(TrackerConfig(namespace="test_ns"))
        assert t.namespace == "test_ns"


# ─── TestMetricTrackerStats ───────────────────────────────────────────────────

class TestMetricTrackerStats:
    def test_stats_returns_metric_stats(self):
        t = MetricTracker()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            t.record("loss", v)
        s = t.stats("loss")
        assert isinstance(s, MetricStats)

    def test_stats_none_for_unknown(self):
        t = MetricTracker()
        assert t.stats("unknown") is None

    def test_stats_count_correct(self):
        t = MetricTracker()
        for v in range(7):
            t.record("x", float(v))
        assert t.stats("x").count == 7

    def test_stats_mean_correct(self):
        t = MetricTracker()
        for v in [0.0, 0.5, 1.0]:
            t.record("x", v)
        assert abs(t.stats("x").mean - 0.5) < 1e-9

    def test_stats_min_max_correct(self):
        t = MetricTracker()
        for v in [0.1, 0.9, 0.5]:
            t.record("x", v)
        s = t.stats("x")
        assert abs(s.minimum - 0.1) < 1e-9
        assert abs(s.maximum - 0.9) < 1e-9

    def test_stats_last_correct(self):
        t = MetricTracker()
        for v in [0.1, 0.3, 0.7]:
            t.record("x", v)
        assert abs(t.stats("x").last - 0.7) < 1e-9

    def test_all_stats_returns_dict(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("b", 2.0)
        all_s = t.all_stats()
        assert "a" in all_s and "b" in all_s


# ─── TestMetricTrackerClear ───────────────────────────────────────────────────

class TestMetricTrackerClear:
    def test_clear_specific_metric(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("b", 2.0)
        n = t.clear("a")
        assert n == 1
        assert t.has_metric("a") is False
        assert t.has_metric("b") is True

    def test_clear_all(self):
        t = MetricTracker()
        for _ in range(5):
            t.record("a", 1.0)
        for _ in range(3):
            t.record("b", 2.0)
        n = t.clear()
        assert n == 8
        assert t.metric_names() == []

    def test_clear_returns_count(self):
        t = MetricTracker()
        for _ in range(4):
            t.record("x", 0.5)
        n = t.clear("x")
        assert n == 4

    def test_clear_unknown_returns_zero(self):
        t = MetricTracker()
        assert t.clear("nonexistent") == 0

    def test_size_specific(self):
        t = MetricTracker()
        for _ in range(6):
            t.record("m", 0.1)
        assert t.size("m") == 6

    def test_size_total(self):
        t = MetricTracker()
        for _ in range(3):
            t.record("a", 1.0)
        for _ in range(4):
            t.record("b", 2.0)
        assert t.size() == 7

    def test_size_unknown_metric_zero(self):
        t = MetricTracker()
        assert t.size("nope") == 0


# ─── TestMakeTracker ──────────────────────────────────────────────────────────

class TestMakeTracker:
    def test_returns_metric_tracker(self):
        t = make_tracker()
        assert isinstance(t, MetricTracker)

    def test_namespace_set(self):
        t = make_tracker(namespace="my_ns")
        assert t.namespace == "my_ns"

    def test_max_history_set(self):
        t = make_tracker(max_history=5)
        for i in range(10):
            t.record("x", float(i))
        assert t.size("x") == 5


# ─── TestMergeTrackers ────────────────────────────────────────────────────────

class TestMergeTrackers:
    def test_copies_records_to_target(self):
        src = MetricTracker()
        src.record("loss", 0.5, step=0)
        src.record("loss", 0.4, step=1)
        tgt = MetricTracker()
        n = merge_trackers(tgt, src)
        assert n == 2
        assert len(tgt.values("loss")) == 2

    def test_existing_records_preserved(self):
        src = MetricTracker()
        src.record("a", 1.0)
        tgt = MetricTracker()
        tgt.record("b", 2.0)
        merge_trackers(tgt, src)
        assert tgt.has_metric("a")
        assert tgt.has_metric("b")

    def test_empty_source_returns_zero(self):
        assert merge_trackers(MetricTracker(), MetricTracker()) == 0


# ─── TestComputeMovingAverage ─────────────────────────────────────────────────

class TestComputeMovingAverage:
    def test_returns_list(self):
        t = MetricTracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t.record("x", v)
        ma = compute_moving_average(t, "x", window=3)
        assert isinstance(ma, list)
        assert len(ma) == 5

    def test_window_1_equals_values(self):
        t = MetricTracker()
        vals = [0.1, 0.5, 0.9]
        for v in vals:
            t.record("x", v)
        ma = compute_moving_average(t, "x", window=1)
        for a, b in zip(ma, vals):
            assert abs(a - b) < 1e-9

    def test_window_3_basic(self):
        t = MetricTracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t.record("x", v)
        ma = compute_moving_average(t, "x", window=3)
        assert abs(ma[2] - 2.0) < 1e-9
        assert abs(ma[4] - 4.0) < 1e-9

    def test_empty_metric_returns_empty(self):
        t = MetricTracker()
        assert compute_moving_average(t, "unknown") == []

    def test_window_lt_1_raises(self):
        t = MetricTracker()
        t.record("x", 1.0)
        with pytest.raises(ValueError):
            compute_moving_average(t, "x", window=0)


# ─── TestExportMetrics ────────────────────────────────────────────────────────

class TestExportMetrics:
    def test_returns_dict(self):
        t = MetricTracker()
        t.record("a", 1.0, step=0)
        result = export_metrics(t)
        assert isinstance(result, dict)

    def test_keys_match_metric_names(self):
        t = MetricTracker()
        t.record("loss", 0.5, step=0)
        t.record("acc", 0.9, step=0)
        result = export_metrics(t)
        assert "loss" in result
        assert "acc" in result

    def test_values_are_step_value_pairs(self):
        t = MetricTracker()
        t.record("m", 0.3, step=5)
        result = export_metrics(t)
        pairs = result["m"]
        assert pairs[0] == (5, 0.3)

    def test_empty_tracker_returns_empty_dict(self):
        t = MetricTracker()
        assert export_metrics(t) == {}

    def test_multiple_steps_ordered(self):
        t = MetricTracker()
        for i in range(4):
            t.record("v", float(i), step=i)
        pairs = export_metrics(t)["v"]
        assert pairs == [(0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0)]
