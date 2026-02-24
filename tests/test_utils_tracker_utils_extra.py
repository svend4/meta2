"""Extra tests for puzzle_reconstruction/utils/tracker_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

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

def _filled_tracker(vals=None):
    t = create_iter_tracker()
    for i, v in enumerate(vals or [1.0, 2.0, 3.0, 4.0, 5.0]):
        record_step(t, i, v)
    return t


# ─── TrackerConfig ────────────────────────────────────────────────────────────

class TestTrackerConfigExtra:
    def test_defaults(self):
        cfg = TrackerConfig()
        assert cfg.window == 5 and cfg.name == "tracker"


# ─── StepRecord ───────────────────────────────────────────────────────────────

class TestStepRecordExtra:
    def test_repr(self):
        r = StepRecord(step=3, value=0.5)
        assert "step=3" in repr(r) and "0.5" in repr(r)


# ─── IterTracker ──────────────────────────────────────────────────────────────

class TestIterTrackerExtra:
    def test_repr(self):
        t = create_iter_tracker()
        assert "n=0" in repr(t)


# ─── create_iter_tracker ──────────────────────────────────────────────────────

class TestCreateIterTrackerExtra:
    def test_returns_tracker(self):
        t = create_iter_tracker()
        assert isinstance(t, IterTracker) and len(t.records) == 0

    def test_custom_config(self):
        cfg = TrackerConfig(window=10, name="custom")
        t = create_iter_tracker(cfg)
        assert t.config.name == "custom"

    def test_metadata_stored(self):
        t = create_iter_tracker(task="assembly")
        assert t.metadata["task"] == "assembly"


# ─── record_step ──────────────────────────────────────────────────────────────

class TestRecordStepExtra:
    def test_appends(self):
        t = create_iter_tracker()
        record_step(t, 0, 1.0)
        assert len(t.records) == 1

    def test_returns_tracker(self):
        t = create_iter_tracker()
        assert record_step(t, 0, 1.0) is t

    def test_no_history_keeps_last(self):
        cfg = TrackerConfig(keep_history=False)
        t = create_iter_tracker(cfg)
        record_step(t, 0, 1.0)
        record_step(t, 1, 2.0)
        assert len(t.records) == 1 and t.records[0].value == pytest.approx(2.0)


# ─── get_values / get_steps ───────────────────────────────────────────────────

class TestGetValuesStepsExtra:
    def test_values_array(self):
        t = _filled_tracker([1.0, 2.0, 3.0])
        v = get_values(t)
        assert isinstance(v, np.ndarray) and len(v) == 3

    def test_steps_array(self):
        t = _filled_tracker([1.0, 2.0])
        s = get_steps(t)
        assert s[0] == 0 and s[1] == 1

    def test_empty_tracker(self):
        t = create_iter_tracker()
        assert len(get_values(t)) == 0


# ─── get_best / get_worst ────────────────────────────────────────────────────

class TestGetBestWorstExtra:
    def test_best(self):
        t = _filled_tracker([1.0, 5.0, 3.0])
        assert get_best_record(t).value == pytest.approx(5.0)

    def test_worst(self):
        t = _filled_tracker([1.0, 5.0, 3.0])
        assert get_worst_record(t).value == pytest.approx(1.0)

    def test_empty_returns_none(self):
        t = create_iter_tracker()
        assert get_best_record(t) is None
        assert get_worst_record(t) is None


# ─── compute_delta ────────────────────────────────────────────────────────────

class TestComputeDeltaExtra:
    def test_basic(self):
        t = _filled_tracker([1.0, 3.0, 6.0])
        d = compute_delta(t)
        assert np.allclose(d, [2.0, 3.0])

    def test_lag_too_small_raises(self):
        t = _filled_tracker()
        with pytest.raises(ValueError):
            compute_delta(t, lag=0)

    def test_lag_exceeds_length(self):
        t = _filled_tracker([1.0])
        d = compute_delta(t, lag=2)
        assert len(d) == 0


# ─── is_improving ─────────────────────────────────────────────────────────────

class TestIsImprovingExtra:
    def test_improving(self):
        t = _filled_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        assert is_improving(t, window=3) is True

    def test_not_improving(self):
        t = _filled_tracker([5.0, 4.0, 3.0])
        assert is_improving(t, window=3) is False

    def test_too_few_steps(self):
        t = _filled_tracker([1.0])
        assert is_improving(t, window=3) is False


# ─── find_plateau_start ───────────────────────────────────────────────────────

class TestFindPlateauStartExtra:
    def test_plateau(self):
        t = _filled_tracker([1.0, 2.0, 5.0, 5.0, 5.0])
        step = find_plateau_start(t, window=3, tol=0.01)
        assert step == 2

    def test_no_plateau(self):
        t = _filled_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_plateau_start(t, window=3, tol=0.01) is None

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            find_plateau_start(_filled_tracker(), window=1)


# ─── smooth_values ────────────────────────────────────────────────────────────

class TestSmoothValuesExtra:
    def test_same_length(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = smooth_values(v, window=3)
        assert len(s) == len(v)

    def test_window_one_identity(self):
        v = np.array([1.0, 2.0, 3.0])
        s = smooth_values(v, window=1)
        assert np.allclose(s, v)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_values(np.array([1.0]), window=0)


# ─── tracker_stats ────────────────────────────────────────────────────────────

class TestTrackerStatsExtra:
    def test_empty(self):
        t = create_iter_tracker()
        s = tracker_stats(t)
        assert s["n"] == 0

    def test_min_max(self):
        t = _filled_tracker([2.0, 8.0, 5.0])
        s = tracker_stats(t)
        assert s["min"] == pytest.approx(2.0) and s["max"] == pytest.approx(8.0)


# ─── compare_trackers ────────────────────────────────────────────────────────

class TestCompareTrackersExtra:
    def test_identical(self):
        t = _filled_tracker()
        d = compare_trackers(t, t)
        assert d["winner"] == "tie"

    def test_a_wins(self):
        a = _filled_tracker([10.0])
        b = _filled_tracker([5.0])
        d = compare_trackers(a, b)
        assert d["winner"] == "a"


# ─── merge_trackers ───────────────────────────────────────────────────────────

class TestMergeTrackersExtra:
    def test_merge_concatenates(self):
        t1 = _filled_tracker([1.0, 2.0])
        t2 = _filled_tracker([3.0, 4.0])
        merged = merge_trackers([t1, t2])
        assert len(merged.records) == 4


# ─── window_stats ─────────────────────────────────────────────────────────────

class TestWindowStatsExtra:
    def test_count(self):
        t = _filled_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
        ws = window_stats(t, window=3)
        assert len(ws) == 3

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            window_stats(_filled_tracker(), window=0)


# ─── top_k_records ────────────────────────────────────────────────────────────

class TestTopKRecordsExtra:
    def test_returns_k(self):
        t = _filled_tracker([1.0, 5.0, 3.0])
        top = top_k_records(t, 2)
        assert len(top) == 2 and top[0].value == pytest.approx(5.0)
