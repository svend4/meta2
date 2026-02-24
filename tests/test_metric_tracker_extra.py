"""Extra tests for puzzle_reconstruction.utils.metric_tracker."""
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


# ─── TestMetricRecordExtra ────────────────────────────────────────────────────

class TestMetricRecordExtra:
    def test_step_0_valid(self):
        r = MetricRecord(name="loss", value=0.5, step=0)
        assert r.step == 0

    def test_large_step(self):
        r = MetricRecord(name="loss", value=0.5, step=10000)
        assert r.step == 10000

    def test_large_positive_value(self):
        r = MetricRecord(name="score", value=1e6, step=0)
        assert r.value == pytest.approx(1e6)

    def test_zero_value(self):
        r = MetricRecord(name="x", value=0.0, step=0)
        assert r.value == pytest.approx(0.0)

    def test_name_stored(self):
        r = MetricRecord(name="accuracy", value=0.9)
        assert r.name == "accuracy"

    def test_various_values(self):
        for v in (-100.0, -1.0, 0.0, 0.5, 1.0, 100.0):
            r = MetricRecord(name="v", value=v)
            assert r.value == pytest.approx(v)


# ─── TestMetricStatsExtra ─────────────────────────────────────────────────────

class TestMetricStatsExtra:
    def _make(self, **kw):
        defaults = dict(name="loss", count=5, mean=0.5, std=0.1,
                        minimum=0.1, maximum=0.9, last=0.7)
        defaults.update(kw)
        return MetricStats(**defaults)

    def test_std_zero_ok(self):
        s = self._make(std=0.0)
        assert s.std == pytest.approx(0.0)

    def test_large_count(self):
        s = self._make(count=1000000)
        assert s.count == 1000000

    def test_min_equals_max(self):
        s = self._make(minimum=0.5, maximum=0.5)
        assert s.range == pytest.approx(0.0)

    def test_range_positive(self):
        s = self._make(minimum=0.1, maximum=0.9)
        assert s.range == pytest.approx(0.8)

    def test_name_stored(self):
        s = self._make(name="accuracy")
        assert s.name == "accuracy"

    def test_last_stored(self):
        s = self._make(last=0.42)
        assert s.last == pytest.approx(0.42)


# ─── TestTrackerConfigExtra ───────────────────────────────────────────────────

class TestTrackerConfigExtra:
    def test_custom_namespace(self):
        cfg = TrackerConfig(namespace="train")
        assert cfg.namespace == "train"

    def test_large_max_history(self):
        cfg = TrackerConfig(max_history=1000)
        assert cfg.max_history == 1000

    def test_max_history_100(self):
        cfg = TrackerConfig(max_history=100)
        assert cfg.max_history == 100

    def test_namespace_with_underscores(self):
        cfg = TrackerConfig(namespace="my_model_v2")
        assert cfg.namespace == "my_model_v2"


# ─── TestMetricTrackerRecordExtra ─────────────────────────────────────────────

class TestMetricTrackerRecordExtra:
    def test_record_100_values(self):
        t = MetricTracker()
        for i in range(100):
            t.record("x", float(i))
        assert t.size("x") == 100

    def test_different_metrics_same_step(self):
        t = MetricTracker()
        t.record("a", 1.0, step=5)
        t.record("b", 2.0, step=5)
        assert t.last_value("a") == pytest.approx(1.0)
        assert t.last_value("b") == pytest.approx(2.0)

    def test_record_dict_5_keys(self):
        t = MetricTracker()
        d = {f"m{i}": float(i) for i in range(5)}
        t.record_dict(d, step=0)
        for k, v in d.items():
            assert t.last_value(k) == pytest.approx(v)

    def test_overwrite_with_max_history_1(self):
        t = MetricTracker(TrackerConfig(max_history=1))
        t.record("v", 1.0, step=0)
        t.record("v", 2.0, step=1)
        assert t.last_value("v") == pytest.approx(2.0)
        assert t.size("v") == 1

    def test_negative_value_stored(self):
        t = MetricTracker()
        t.record("delta", -5.0, step=0)
        assert t.last_value("delta") == pytest.approx(-5.0)


# ─── TestMetricTrackerReadExtra ───────────────────────────────────────────────

class TestMetricTrackerReadExtra:
    def test_values_in_insertion_order(self):
        t = MetricTracker()
        vals = [0.1, 0.5, 0.9, 0.3, 0.7]
        for v in vals:
            t.record("x", v)
        assert t.values("x") == vals

    def test_history_steps_correct(self):
        t = MetricTracker()
        for i in range(5):
            t.record("x", float(i), step=i * 2)
        hist = t.history("x")
        assert [r.step for r in hist] == [0, 2, 4, 6, 8]

    def test_metric_names_returns_list(self):
        t = MetricTracker()
        t.record("a", 1.0)
        t.record("b", 2.0)
        names = t.metric_names()
        assert isinstance(names, list)

    def test_two_metrics_independent(self):
        t = MetricTracker()
        for i in range(3):
            t.record("a", float(i))
            t.record("b", float(i) * 2)
        assert t.size("a") == 3
        assert t.size("b") == 3

    def test_has_metric_false_after_clear(self):
        t = MetricTracker()
        t.record("x", 1.0)
        t.clear("x")
        assert t.has_metric("x") is False


# ─── TestMetricTrackerStatsExtra ──────────────────────────────────────────────

class TestMetricTrackerStatsExtra:
    def test_std_known(self):
        t = MetricTracker()
        for v in [0.0, 0.0, 2.0, 2.0]:
            t.record("x", v)
        s = t.stats("x")
        assert s.mean == pytest.approx(1.0)
        assert s.std >= 0.0

    def test_single_value_stats(self):
        t = MetricTracker()
        t.record("x", 0.5, step=0)
        s = t.stats("x")
        assert s.count == 1
        assert s.mean == pytest.approx(0.5)
        assert s.minimum == pytest.approx(0.5)
        assert s.maximum == pytest.approx(0.5)

    def test_negative_values_stats(self):
        t = MetricTracker()
        for v in [-1.0, -2.0, -3.0]:
            t.record("x", v)
        s = t.stats("x")
        assert s.minimum == pytest.approx(-3.0)
        assert s.maximum == pytest.approx(-1.0)
        assert s.mean == pytest.approx(-2.0)

    def test_all_stats_length(self):
        t = MetricTracker()
        for name in ("a", "b", "c"):
            t.record(name, 1.0)
        all_s = t.all_stats()
        assert len(all_s) == 3


# ─── TestMetricTrackerClearExtra ──────────────────────────────────────────────

class TestMetricTrackerClearExtra:
    def test_clear_then_rerecord(self):
        t = MetricTracker()
        for _ in range(5):
            t.record("x", 1.0)
        t.clear("x")
        t.record("x", 99.0)
        assert t.last_value("x") == pytest.approx(99.0)
        assert t.size("x") == 1

    def test_clear_all_then_reuse(self):
        t = MetricTracker()
        for name in ("a", "b", "c"):
            t.record(name, 1.0)
        t.clear()
        assert t.metric_names() == []
        t.record("new", 5.0)
        assert t.size("new") == 1

    def test_size_zero_after_clear(self):
        t = MetricTracker()
        for _ in range(3):
            t.record("x", 1.0)
        t.clear("x")
        assert t.size("x") == 0

    def test_clear_multiple_calls(self):
        t = MetricTracker()
        t.record("x", 1.0)
        n1 = t.clear("x")
        n2 = t.clear("x")
        assert n1 == 1
        assert n2 == 0


# ─── TestMakeTrackerExtra ─────────────────────────────────────────────────────

class TestMakeTrackerExtra:
    def test_default_namespace(self):
        t = make_tracker()
        assert t.namespace == "default"

    def test_two_trackers_independent(self):
        t1 = make_tracker(namespace="t1")
        t2 = make_tracker(namespace="t2")
        t1.record("x", 1.0)
        assert not t2.has_metric("x")

    def test_make_tracker_large_history(self):
        t = make_tracker(max_history=500)
        for i in range(600):
            t.record("x", float(i))
        assert t.size("x") == 500


# ─── TestMergeTrackersExtra ───────────────────────────────────────────────────

class TestMergeTrackersExtra:
    def test_multiple_metrics_merged(self):
        src = MetricTracker()
        for name in ("a", "b", "c"):
            src.record(name, 1.0)
        tgt = MetricTracker()
        n = merge_trackers(tgt, src)
        assert n == 3
        assert all(tgt.has_metric(name) for name in ("a", "b", "c"))

    def test_large_merge(self):
        src = MetricTracker()
        for i in range(50):
            src.record("m", float(i))
        tgt = MetricTracker()
        n = merge_trackers(tgt, src)
        assert n == 50
        assert tgt.size("m") == 50

    def test_merge_into_non_empty(self):
        src = MetricTracker()
        src.record("a", 1.0)
        tgt = MetricTracker()
        tgt.record("a", 0.5)
        merge_trackers(tgt, src)
        assert tgt.size("a") == 2


# ─── TestComputeMovingAverageExtra ────────────────────────────────────────────

class TestComputeMovingAverageExtra:
    def test_window_2(self):
        t = MetricTracker()
        for v in [1.0, 3.0, 5.0]:
            t.record("x", v)
        ma = compute_moving_average(t, "x", window=2)
        # ma[1] = (1+3)/2 = 2, ma[2] = (3+5)/2 = 4
        assert abs(ma[2] - 4.0) < 1e-9

    def test_window_exceeds_length(self):
        t = MetricTracker()
        for v in [1.0, 2.0]:
            t.record("x", v)
        ma = compute_moving_average(t, "x", window=100)
        assert len(ma) == 2

    def test_length_always_matches_history(self):
        t = MetricTracker()
        for v in range(10):
            t.record("y", float(v))
        for w in (1, 3, 5, 10):
            ma = compute_moving_average(t, "y", window=w)
            assert len(ma) == 10

    def test_single_value_window_any(self):
        t = MetricTracker()
        t.record("x", 7.0)
        ma = compute_moving_average(t, "x", window=5)
        assert ma[0] == pytest.approx(7.0)


# ─── TestExportMetricsExtra ───────────────────────────────────────────────────

class TestExportMetricsExtra:
    def test_negative_values_exported(self):
        t = MetricTracker()
        t.record("delta", -1.5, step=3)
        pairs = export_metrics(t)["delta"]
        assert pairs[0] == (3, -1.5)

    def test_mixed_steps_ordered(self):
        t = MetricTracker()
        for i in range(5):
            t.record("m", float(i) * 0.1, step=i * 5)
        pairs = export_metrics(t)["m"]
        steps = [s for s, _ in pairs]
        assert steps == sorted(steps)

    def test_multiple_metrics_all_in_export(self):
        t = MetricTracker()
        for name in ("loss", "acc", "lr"):
            t.record(name, 0.5, step=0)
        result = export_metrics(t)
        assert set(result.keys()) == {"loss", "acc", "lr"}

    def test_ten_steps_exported(self):
        t = MetricTracker()
        for i in range(10):
            t.record("v", float(i), step=i)
        pairs = export_metrics(t)["v"]
        assert len(pairs) == 10
