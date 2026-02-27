"""Tests for puzzle_reconstruction.utils.tracker_utils."""
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

np.random.seed(42)


def _build_tracker(values, name="test"):
    cfg = TrackerConfig(name=name)
    t = create_iter_tracker(cfg)
    for i, v in enumerate(values):
        record_step(t, i, v)
    return t


# ── TrackerConfig ─────────────────────────────────────────────────────────────

def test_tracker_config_defaults():
    cfg = TrackerConfig()
    assert cfg.window == 5
    assert cfg.tol == pytest.approx(1e-5)
    assert cfg.keep_history is True
    assert cfg.name == "tracker"


def test_tracker_config_custom():
    cfg = TrackerConfig(window=3, name="mytracker")
    assert cfg.name == "mytracker"
    assert cfg.window == 3


# ── create_iter_tracker ───────────────────────────────────────────────────────

def test_create_iter_tracker_empty():
    t = create_iter_tracker()
    assert len(t.records) == 0


def test_create_iter_tracker_with_meta():
    t = create_iter_tracker(source="test", run_id=1)
    assert t.metadata["source"] == "test"
    assert t.metadata["run_id"] == 1


def test_create_iter_tracker_default_config():
    t = create_iter_tracker()
    assert isinstance(t.config, TrackerConfig)


# ── record_step ───────────────────────────────────────────────────────────────

def test_record_step_appends():
    t = create_iter_tracker()
    record_step(t, 0, 1.0)
    record_step(t, 1, 2.0)
    assert len(t.records) == 2


def test_record_step_returns_tracker():
    t = create_iter_tracker()
    result = record_step(t, 0, 1.0)
    assert result is t


def test_record_step_no_history():
    cfg = TrackerConfig(keep_history=False)
    t = create_iter_tracker(cfg)
    record_step(t, 0, 1.0)
    record_step(t, 1, 2.0)
    assert len(t.records) == 1
    assert t.records[0].value == pytest.approx(2.0)


# ── get_values / get_steps ────────────────────────────────────────────────────

def test_get_values_empty():
    t = create_iter_tracker()
    assert len(get_values(t)) == 0


def test_get_values():
    t = _build_tracker([1.0, 2.0, 3.0])
    vals = get_values(t)
    np.testing.assert_array_almost_equal(vals, [1.0, 2.0, 3.0])


def test_get_steps():
    t = _build_tracker([0.5, 0.6, 0.7])
    steps = get_steps(t)
    np.testing.assert_array_equal(steps, [0, 1, 2])


# ── get_best_record / get_worst_record ────────────────────────────────────────

def test_get_best_record_none():
    t = create_iter_tracker()
    assert get_best_record(t) is None


def test_get_best_record():
    t = _build_tracker([1.0, 5.0, 3.0])
    best = get_best_record(t)
    assert best.value == pytest.approx(5.0)
    assert best.step == 1


def test_get_worst_record():
    t = _build_tracker([4.0, 1.0, 3.0])
    worst = get_worst_record(t)
    assert worst.value == pytest.approx(1.0)
    assert worst.step == 1


# ── compute_delta ─────────────────────────────────────────────────────────────

def test_compute_delta_basic():
    t = _build_tracker([1.0, 3.0, 6.0])
    d = compute_delta(t, lag=1)
    np.testing.assert_array_almost_equal(d, [2.0, 3.0])


def test_compute_delta_empty_result():
    t = _build_tracker([1.0])
    d = compute_delta(t, lag=2)
    assert len(d) == 0


def test_compute_delta_invalid_lag():
    t = _build_tracker([1.0, 2.0])
    with pytest.raises(ValueError):
        compute_delta(t, lag=0)


# ── is_improving ──────────────────────────────────────────────────────────────

def test_is_improving_true():
    t = _build_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    assert is_improving(t, window=3, tol=1e-6) is True


def test_is_improving_false():
    t = _build_tracker([5.0, 4.0, 3.0])
    assert is_improving(t, window=3) is False


def test_is_improving_too_short():
    t = _build_tracker([1.0, 2.0])
    assert is_improving(t, window=5) is False


# ── find_plateau_start ────────────────────────────────────────────────────────

def test_find_plateau_start_none():
    t = _build_tracker([1.0, 2.0, 3.0])
    result = find_plateau_start(t, window=5)
    assert result is None


def test_find_plateau_start_found():
    vals = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    t = _build_tracker(vals)
    idx = find_plateau_start(t, window=3, tol=1e-4)
    assert idx is not None


def test_find_plateau_start_invalid_window():
    t = _build_tracker([1.0, 2.0])
    with pytest.raises(ValueError):
        find_plateau_start(t, window=1)


# ── smooth_values ─────────────────────────────────────────────────────────────

def test_smooth_values_empty():
    result = smooth_values(np.array([]))
    assert len(result) == 0


def test_smooth_values_same_length():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = smooth_values(v, window=3)
    assert len(s) == len(v)


def test_smooth_values_window_one():
    v = np.array([1.0, 2.0, 3.0])
    s = smooth_values(v, window=1)
    np.testing.assert_array_almost_equal(s, v)


def test_smooth_values_invalid_window():
    with pytest.raises(ValueError):
        smooth_values(np.array([1.0, 2.0]), window=0)


# ── tracker_stats ─────────────────────────────────────────────────────────────

def test_tracker_stats_empty():
    t = create_iter_tracker()
    stats = tracker_stats(t)
    assert stats["n"] == 0


def test_tracker_stats():
    t = _build_tracker([1.0, 3.0, 2.0])
    stats = tracker_stats(t)
    assert stats["n"] == 3
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)
    assert stats["best_step"] == 1
    assert stats["worst_step"] == 0


# ── compare_trackers ──────────────────────────────────────────────────────────

def test_compare_trackers_winner_a():
    a = _build_tracker([1.0, 3.0, 5.0])
    b = _build_tracker([1.0, 2.0, 3.0])
    cmp = compare_trackers(a, b)
    assert cmp["winner"] == "a"
    assert cmp["delta_best"] > 0


def test_compare_trackers_tie():
    a = create_iter_tracker()
    b = create_iter_tracker()
    cmp = compare_trackers(a, b)
    assert cmp["winner"] == "tie"


# ── merge_trackers ────────────────────────────────────────────────────────────

def test_merge_trackers():
    a = _build_tracker([1.0, 2.0])
    b = _build_tracker([3.0, 4.0])
    merged = merge_trackers([a, b])
    assert len(merged.records) == 4


def test_merge_trackers_empty():
    merged = merge_trackers([])
    assert len(merged.records) == 0


# ── window_stats ──────────────────────────────────────────────────────────────

def test_window_stats_basic():
    t = _build_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = window_stats(t, window=3)
    assert len(stats) == 3  # 5 - 3 + 1
    assert "mean" in stats[0]
    assert "std" in stats[0]


def test_window_stats_invalid():
    t = _build_tracker([1.0, 2.0])
    with pytest.raises(ValueError):
        window_stats(t, window=0)


# ── top_k_records ─────────────────────────────────────────────────────────────

def test_top_k_records():
    t = _build_tracker([3.0, 1.0, 4.0, 1.5, 2.0])
    top = top_k_records(t, 3)
    assert len(top) == 3
    assert top[0].value == pytest.approx(4.0)


def test_top_k_records_exceeds_len():
    t = _build_tracker([1.0, 2.0])
    top = top_k_records(t, 10)
    assert len(top) == 2
