"""Extra tests for puzzle_reconstruction/utils/tracker_utils.py (iter-234)."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.tracker_utils import (
    TrackerConfig,
    StepRecord,
    IterTracker,
    create_iter_tracker,
    record_step,
    get_values,
    get_steps,
    get_best_record,
    get_worst_record,
    compute_delta,
    is_improving,
    find_plateau_start,
    smooth_values,
    tracker_stats,
    compare_trackers,
    merge_trackers,
    window_stats,
    top_k_records,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _config(**kw) -> TrackerConfig:
    return TrackerConfig(**kw)


def _tracker(values: list | None = None, name: str = "test") -> IterTracker:
    t = create_iter_tracker(TrackerConfig(name=name))
    if values:
        for i, v in enumerate(values):
            record_step(t, step=i, value=v)
    return t


def _tracker_ascending(n: int = 10) -> IterTracker:
    return _tracker([float(i) for i in range(n)])


def _tracker_plateau(n: int = 10, plateau_val: float = 5.0) -> IterTracker:
    vals = [float(i) for i in range(5)] + [plateau_val] * (n - 5)
    return _tracker(vals[:n])


# ─── TrackerConfig ──────────────────────────────────────────────────────────

class TestTrackerConfigExtra:
    def test_default_window(self):
        assert TrackerConfig().window == 5

    def test_default_tol(self):
        assert TrackerConfig().tol == pytest.approx(1e-5)

    def test_default_smooth_window(self):
        assert TrackerConfig().smooth_window == 3

    def test_default_keep_history(self):
        assert TrackerConfig().keep_history is True

    def test_default_name(self):
        assert TrackerConfig().name == "tracker"

    def test_custom_window(self):
        cfg = TrackerConfig(window=10)
        assert cfg.window == 10

    def test_custom_tol(self):
        cfg = TrackerConfig(tol=1e-3)
        assert cfg.tol == pytest.approx(1e-3)

    def test_custom_name(self):
        cfg = TrackerConfig(name="loss")
        assert cfg.name == "loss"


# ─── StepRecord ─────────────────────────────────────────────────────────────

class TestStepRecordExtra:
    def test_fields_stored(self):
        r = StepRecord(step=3, value=0.75)
        assert r.step == 3
        assert r.value == pytest.approx(0.75)

    def test_default_meta_empty(self):
        r = StepRecord(step=0, value=0.0)
        assert r.meta == {}

    def test_meta_stored(self):
        r = StepRecord(step=0, value=0.0, meta={"lr": 0.01})
        assert r.meta["lr"] == pytest.approx(0.01)

    def test_repr_contains_step(self):
        r = StepRecord(step=5, value=0.123456)
        assert "step=5" in repr(r)

    def test_repr_contains_value(self):
        r = StepRecord(step=0, value=0.123456)
        assert "0.123456" in repr(r)

    def test_repr_format(self):
        r = StepRecord(step=1, value=1.0)
        text = repr(r)
        assert text.startswith("StepRecord(")
        assert text.endswith(")")


# ─── IterTracker ────────────────────────────────────────────────────────────

class TestIterTrackerExtra:
    def test_empty_records(self):
        t = IterTracker()
        assert t.records == []

    def test_default_config(self):
        t = IterTracker()
        assert isinstance(t.config, TrackerConfig)

    def test_default_metadata(self):
        t = IterTracker()
        assert t.metadata == {}

    def test_repr_contains_n(self):
        t = _tracker([1.0, 2.0])
        assert "n=2" in repr(t)

    def test_repr_contains_name(self):
        t = _tracker([1.0], name="loss")
        assert "loss" in repr(t)


# ─── create_iter_tracker ────────────────────────────────────────────────────

class TestCreateIterTrackerExtra:
    def test_returns_tracker(self):
        assert isinstance(create_iter_tracker(), IterTracker)

    def test_none_config_uses_defaults(self):
        t = create_iter_tracker(config=None)
        assert t.config.window == 5

    def test_custom_config(self):
        cfg = TrackerConfig(window=10, name="custom")
        t = create_iter_tracker(config=cfg)
        assert t.config.window == 10
        assert t.config.name == "custom"

    def test_meta_stored(self):
        t = create_iter_tracker(experiment="exp1", lr=0.01)
        assert t.metadata["experiment"] == "exp1"
        assert t.metadata["lr"] == pytest.approx(0.01)

    def test_empty_records(self):
        t = create_iter_tracker()
        assert len(t.records) == 0

    def test_no_meta(self):
        t = create_iter_tracker()
        assert t.metadata == {}


# ─── record_step ────────────────────────────────────────────────────────────

class TestRecordStepExtra:
    def test_returns_tracker(self):
        t = create_iter_tracker()
        result = record_step(t, 0, 1.0)
        assert result is t

    def test_appends_record(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.0)
        assert len(t.records) == 1

    def test_multiple_steps(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.0)
        record_step(t, 1, 2.0)
        record_step(t, 2, 3.0)
        assert len(t.records) == 3

    def test_step_value_stored(self):
        t = create_iter_tracker()
        record_step(t, 5, 0.42)
        assert t.records[0].step == 5
        assert t.records[0].value == pytest.approx(0.42)

    def test_meta_stored(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.0, lr=0.01)
        assert t.records[0].meta["lr"] == pytest.approx(0.01)

    def test_keep_history_false(self):
        cfg = TrackerConfig(keep_history=False)
        t = create_iter_tracker(config=cfg)
        record_step(t, 0, 1.0)
        record_step(t, 1, 2.0)
        record_step(t, 2, 3.0)
        assert len(t.records) == 1
        assert t.records[0].step == 2

    def test_chaining(self):
        t = create_iter_tracker()
        record_step(record_step(t, 0, 1.0), 1, 2.0)
        assert len(t.records) == 2


# ─── get_values ─────────────────────────────────────────────────────────────

class TestGetValuesExtra:
    def test_returns_ndarray(self):
        t = _tracker([1.0, 2.0, 3.0])
        assert isinstance(get_values(t), np.ndarray)

    def test_dtype_float64(self):
        t = _tracker([1.0, 2.0])
        assert get_values(t).dtype == np.float64

    def test_values_correct(self):
        t = _tracker([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(get_values(t), [1.0, 2.0, 3.0])

    def test_empty_tracker(self):
        t = _tracker([])
        assert len(get_values(t)) == 0

    def test_shape(self):
        t = _tracker([1.0, 2.0, 3.0])
        assert get_values(t).shape == (3,)


# ─── get_steps ──────────────────────────────────────────────────────────────

class TestGetStepsExtra:
    def test_returns_ndarray(self):
        t = _tracker([1.0, 2.0])
        assert isinstance(get_steps(t), np.ndarray)

    def test_dtype_int64(self):
        t = _tracker([1.0, 2.0])
        assert get_steps(t).dtype == np.int64

    def test_values_correct(self):
        t = _tracker([10.0, 20.0, 30.0])
        np.testing.assert_array_equal(get_steps(t), [0, 1, 2])

    def test_empty_tracker(self):
        t = _tracker([])
        assert len(get_steps(t)) == 0

    def test_shape(self):
        t = _tracker([1.0, 2.0, 3.0])
        assert get_steps(t).shape == (3,)


# ─── get_best_record ───────────────────────────────────────────────────────

class TestGetBestRecordExtra:
    def test_returns_max(self):
        t = _tracker([1.0, 3.0, 2.0])
        best = get_best_record(t)
        assert best is not None
        assert best.value == pytest.approx(3.0)

    def test_empty_returns_none(self):
        t = _tracker([])
        assert get_best_record(t) is None

    def test_single(self):
        t = _tracker([0.42])
        assert get_best_record(t).value == pytest.approx(0.42)

    def test_returns_step_record(self):
        t = _tracker([1.0, 2.0])
        assert isinstance(get_best_record(t), StepRecord)

    def test_tie_returns_one(self):
        t = _tracker([5.0, 5.0])
        assert get_best_record(t).value == pytest.approx(5.0)


# ─── get_worst_record ──────────────────────────────────────────────────────

class TestGetWorstRecordExtra:
    def test_returns_min(self):
        t = _tracker([3.0, 1.0, 2.0])
        worst = get_worst_record(t)
        assert worst is not None
        assert worst.value == pytest.approx(1.0)

    def test_empty_returns_none(self):
        t = _tracker([])
        assert get_worst_record(t) is None

    def test_single(self):
        t = _tracker([0.42])
        assert get_worst_record(t).value == pytest.approx(0.42)

    def test_returns_step_record(self):
        t = _tracker([1.0, 2.0])
        assert isinstance(get_worst_record(t), StepRecord)

    def test_tie_returns_one(self):
        t = _tracker([5.0, 5.0])
        assert get_worst_record(t).value == pytest.approx(5.0)


# ─── compute_delta ──────────────────────────────────────────────────────────

class TestComputeDeltaExtra:
    def test_lag_less_than_1_raises(self):
        t = _tracker([1.0, 2.0])
        with pytest.raises(ValueError):
            compute_delta(t, lag=0)

    def test_lag_negative_raises(self):
        t = _tracker([1.0, 2.0])
        with pytest.raises(ValueError):
            compute_delta(t, lag=-1)

    def test_ascending_lag_1(self):
        t = _tracker([1.0, 2.0, 3.0, 4.0])
        delta = compute_delta(t, lag=1)
        np.testing.assert_allclose(delta, [1.0, 1.0, 1.0])

    def test_ascending_lag_2(self):
        t = _tracker([1.0, 2.0, 3.0, 4.0])
        delta = compute_delta(t, lag=2)
        np.testing.assert_allclose(delta, [2.0, 2.0])

    def test_empty_tracker(self):
        t = _tracker([])
        delta = compute_delta(t, lag=1)
        assert len(delta) == 0

    def test_single_value(self):
        t = _tracker([1.0])
        delta = compute_delta(t, lag=1)
        assert len(delta) == 0

    def test_lag_equals_n(self):
        t = _tracker([1.0, 2.0])
        delta = compute_delta(t, lag=2)
        assert len(delta) == 0

    def test_returns_ndarray(self):
        t = _tracker([1.0, 2.0, 3.0])
        assert isinstance(compute_delta(t, lag=1), np.ndarray)


# ─── is_improving ───────────────────────────────────────────────────────────

class TestIsImprovingExtra:
    def test_ascending_is_improving(self):
        t = _tracker_ascending(10)
        assert is_improving(t, window=3) is True

    def test_constant_not_improving(self):
        t = _tracker([5.0, 5.0, 5.0, 5.0])
        assert is_improving(t, window=3) is False

    def test_descending_not_improving(self):
        t = _tracker([4.0, 3.0, 2.0, 1.0])
        assert is_improving(t, window=3) is False

    def test_too_few_values(self):
        t = _tracker([1.0, 2.0])
        assert is_improving(t, window=5) is False

    def test_empty_tracker(self):
        t = _tracker([])
        assert is_improving(t, window=3) is False

    def test_custom_tol(self):
        t = _tracker([1.0, 1.0, 1.0001])
        assert is_improving(t, window=3, tol=0.01) is False

    def test_tol_boundary(self):
        t = _tracker([0.0, 0.0, 0.0, 0.01])
        assert is_improving(t, window=3, tol=0.001) is True


# ─── find_plateau_start ────────────────────────────────────────────────────

class TestFindPlateauStartExtra:
    def test_window_lt_2_raises(self):
        t = _tracker([1.0, 2.0])
        with pytest.raises(ValueError):
            find_plateau_start(t, window=1)

    def test_window_zero_raises(self):
        t = _tracker([1.0])
        with pytest.raises(ValueError):
            find_plateau_start(t, window=0)

    def test_constant_values(self):
        t = _tracker([5.0, 5.0, 5.0, 5.0, 5.0])
        result = find_plateau_start(t, window=3, tol=1e-5)
        assert result == 0

    def test_ascending_then_plateau(self):
        t = _tracker_plateau(10)
        result = find_plateau_start(t, window=3, tol=1e-5)
        # Plateau starts at step 4 (index 4, value 5.0)
        # The window [5.0, 5.0, 5.0] first appears at index 4
        assert result is not None
        assert result >= 4

    def test_no_plateau(self):
        t = _tracker_ascending(10)
        result = find_plateau_start(t, window=3, tol=0.0)
        assert result is None

    def test_too_few_values(self):
        t = _tracker([1.0, 2.0])
        result = find_plateau_start(t, window=5)
        assert result is None

    def test_empty_tracker(self):
        t = _tracker([])
        result = find_plateau_start(t, window=2)
        assert result is None


# ─── smooth_values ──────────────────────────────────────────────────────────

class TestSmoothValuesExtra:
    def test_window_lt_1_raises(self):
        with pytest.raises(ValueError):
            smooth_values(np.array([1.0, 2.0]), window=0)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError):
            smooth_values(np.array([1.0]), window=-1)

    def test_window_1_identity(self):
        vals = np.array([1.0, 2.0, 3.0])
        result = smooth_values(vals, window=1)
        np.testing.assert_allclose(result, vals)

    def test_same_length(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth_values(vals, window=3)
        assert len(result) == len(vals)

    def test_returns_float64(self):
        vals = np.array([1, 2, 3], dtype=np.int32)
        result = smooth_values(vals, window=3)
        assert result.dtype == np.float64

    def test_empty_array(self):
        result = smooth_values(np.array([]), window=3)
        assert len(result) == 0

    def test_constant_interior_unchanged(self):
        vals = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = smooth_values(vals, window=3)
        # Interior values (indices 1..3) stay at 5.0; edges may differ
        np.testing.assert_allclose(result[1:-1], 5.0, atol=1e-10)


# ─── tracker_stats ──────────────────────────────────────────────────────────

class TestTrackerStatsExtra:
    def test_empty_tracker(self):
        t = _tracker([])
        stats = tracker_stats(t)
        assert stats["n"] == 0

    def test_n_count(self):
        t = _tracker([1.0, 2.0, 3.0])
        assert tracker_stats(t)["n"] == 3

    def test_mean(self):
        t = _tracker([2.0, 4.0])
        assert tracker_stats(t)["mean"] == pytest.approx(3.0)

    def test_min_max(self):
        t = _tracker([1.0, 5.0, 3.0])
        stats = tracker_stats(t)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_std(self):
        t = _tracker([1.0, 1.0, 1.0])
        assert tracker_stats(t)["std"] == pytest.approx(0.0)

    def test_best_step(self):
        t = _tracker([1.0, 3.0, 2.0])
        assert tracker_stats(t)["best_step"] == 1

    def test_worst_step(self):
        t = _tracker([3.0, 1.0, 2.0])
        assert tracker_stats(t)["worst_step"] == 1

    def test_first_last_step(self):
        t = _tracker([1.0, 2.0, 3.0])
        stats = tracker_stats(t)
        assert stats["first_step"] == 0
        assert stats["last_step"] == 2


# ─── compare_trackers ──────────────────────────────────────────────────────

class TestCompareTrackersExtra:
    def test_identical_trackers(self):
        t = _tracker([1.0, 2.0, 3.0])
        result = compare_trackers(t, t)
        assert result["winner"] == "tie"
        assert result["delta_best"] == pytest.approx(0.0)
        assert result["delta_mean"] == pytest.approx(0.0)

    def test_a_wins(self):
        a = _tracker([5.0, 6.0])
        b = _tracker([1.0, 2.0])
        result = compare_trackers(a, b)
        assert result["winner"] == "a"
        assert result["delta_best"] > 0.0

    def test_b_wins(self):
        a = _tracker([1.0, 2.0])
        b = _tracker([5.0, 6.0])
        result = compare_trackers(a, b)
        assert result["winner"] == "b"
        assert result["delta_best"] < 0.0

    def test_both_empty(self):
        a = _tracker([])
        b = _tracker([])
        result = compare_trackers(a, b)
        assert result["winner"] == "tie"

    def test_best_a_and_best_b(self):
        a = _tracker([1.0, 5.0])
        b = _tracker([1.0, 3.0])
        result = compare_trackers(a, b)
        assert result["best_a"] == pytest.approx(5.0)
        assert result["best_b"] == pytest.approx(3.0)

    def test_delta_mean(self):
        a = _tracker([2.0, 4.0])  # mean 3.0
        b = _tracker([1.0, 3.0])  # mean 2.0
        result = compare_trackers(a, b)
        assert result["delta_mean"] == pytest.approx(1.0)


# ─── merge_trackers ────────────────────────────────────────────────────────

class TestMergeTrackersExtra:
    def test_merge_two(self):
        a = _tracker([1.0, 2.0])
        b = _tracker([3.0, 4.0])
        merged = merge_trackers([a, b])
        assert len(merged.records) == 4

    def test_merge_empty(self):
        merged = merge_trackers([])
        assert len(merged.records) == 0

    def test_merge_single(self):
        t = _tracker([1.0, 2.0])
        merged = merge_trackers([t])
        assert len(merged.records) == 2

    def test_merge_preserves_values(self):
        a = _tracker([1.0])
        b = _tracker([2.0])
        merged = merge_trackers([a, b])
        vals = get_values(merged)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(2.0)

    def test_merge_returns_new_tracker(self):
        a = _tracker([1.0])
        merged = merge_trackers([a])
        assert merged is not a

    def test_merge_three(self):
        a = _tracker([1.0])
        b = _tracker([2.0])
        c = _tracker([3.0])
        merged = merge_trackers([a, b, c])
        assert len(merged.records) == 3


# ─── window_stats ───────────────────────────────────────────────────────────

class TestWindowStatsExtra:
    def test_window_lt_1_raises(self):
        t = _tracker([1.0, 2.0])
        with pytest.raises(ValueError):
            window_stats(t, window=0)

    def test_window_negative_raises(self):
        t = _tracker([1.0])
        with pytest.raises(ValueError):
            window_stats(t, window=-1)

    def test_returns_list(self):
        t = _tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(window_stats(t, window=3), list)

    def test_number_of_windows(self):
        t = _tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        result = window_stats(t, window=3)
        assert len(result) == 3  # 5 - 3 + 1

    def test_each_window_has_keys(self):
        t = _tracker([1.0, 2.0, 3.0])
        result = window_stats(t, window=2)
        for w in result:
            assert "mean" in w
            assert "std" in w
            assert "min" in w
            assert "max" in w

    def test_constant_values(self):
        t = _tracker([5.0, 5.0, 5.0, 5.0])
        result = window_stats(t, window=3)
        for w in result:
            assert w["mean"] == pytest.approx(5.0)
            assert w["std"] == pytest.approx(0.0)

    def test_ascending_first_window(self):
        t = _tracker([1.0, 2.0, 3.0, 4.0])
        result = window_stats(t, window=3)
        assert result[0]["mean"] == pytest.approx(2.0)
        assert result[0]["min"] == pytest.approx(1.0)
        assert result[0]["max"] == pytest.approx(3.0)

    def test_empty_tracker(self):
        t = _tracker([])
        result = window_stats(t, window=3)
        assert result == []

    def test_fewer_values_than_window(self):
        t = _tracker([1.0, 2.0])
        result = window_stats(t, window=5)
        assert result == []


# ─── top_k_records ──────────────────────────────────────────────────────────

class TestTopKRecordsExtra:
    def test_top_1(self):
        t = _tracker([1.0, 3.0, 2.0])
        result = top_k_records(t, 1)
        assert len(result) == 1
        assert result[0].value == pytest.approx(3.0)

    def test_top_2(self):
        t = _tracker([1.0, 3.0, 2.0])
        result = top_k_records(t, 2)
        assert len(result) == 2
        assert result[0].value >= result[1].value

    def test_k_greater_than_n(self):
        t = _tracker([1.0, 2.0])
        result = top_k_records(t, 100)
        assert len(result) == 2

    def test_empty_tracker(self):
        t = _tracker([])
        assert top_k_records(t, 5) == []

    def test_descending_order(self):
        t = _tracker([3.0, 1.0, 4.0, 1.0, 5.0])
        result = top_k_records(t, 5)
        values = [r.value for r in result]
        assert values == sorted(values, reverse=True)

    def test_returns_step_records(self):
        t = _tracker([1.0, 2.0])
        result = top_k_records(t, 2)
        for r in result:
            assert isinstance(r, StepRecord)
