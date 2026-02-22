"""Extra tests for puzzle_reconstruction.assembly.score_tracker."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.score_tracker import (
    ScoreSnapshot,
    ScoreTracker,
    create_tracker,
    detect_convergence,
    extract_best_iteration,
    record_snapshot,
    smooth_scores,
    summarize_tracker,
)


def _filled_tracker(scores, n_placed=1):
    t = create_tracker()
    for i, s in enumerate(scores):
        record_snapshot(t, iteration=i, score=s, n_placed=n_placed)
    return t


# ─── ScoreSnapshot extras ─────────────────────────────────────────────────────

class TestScoreSnapshotExtra:
    def test_score_zero(self):
        s = ScoreSnapshot(iteration=0, score=0.0, n_placed=0)
        assert s.score == pytest.approx(0.0)

    def test_score_one(self):
        s = ScoreSnapshot(iteration=0, score=1.0, n_placed=0)
        assert s.score == pytest.approx(1.0)

    def test_n_placed_large(self):
        s = ScoreSnapshot(iteration=0, score=0.5, n_placed=1000)
        assert s.n_placed == 1000

    def test_extra_empty_by_default(self):
        s = ScoreSnapshot(iteration=5, score=0.5, n_placed=3)
        assert isinstance(s.extra, dict)
        assert len(s.extra) == 0

    def test_extra_multiple_keys(self):
        s = ScoreSnapshot(iteration=0, score=0.5, n_placed=1,
                          extra={"a": 1, "b": 2, "c": 3})
        assert len(s.extra) == 3
        assert s.extra["b"] == 2

    def test_repr_is_string(self):
        s = ScoreSnapshot(iteration=1, score=0.9, n_placed=2)
        assert isinstance(repr(s), str)

    def test_iteration_zero(self):
        s = ScoreSnapshot(iteration=0, score=0.5, n_placed=1)
        assert s.iteration == 0

    def test_large_iteration(self):
        s = ScoreSnapshot(iteration=99999, score=0.5, n_placed=1)
        assert s.iteration == 99999


# ─── ScoreTracker extras ──────────────────────────────────────────────────────

class TestScoreTrackerExtra:
    def test_snapshots_is_list(self):
        t = ScoreTracker()
        assert isinstance(t.snapshots, list)

    def test_params_is_dict(self):
        t = ScoreTracker()
        assert isinstance(t.params, dict)

    def test_multiple_params_stored(self):
        t = ScoreTracker(params={"a": 1, "b": 2, "c": "x"})
        assert t.params["a"] == 1
        assert t.params["c"] == "x"

    def test_snapshots_not_shared_between_instances(self):
        t1 = ScoreTracker()
        t2 = ScoreTracker()
        record_snapshot(t1, 0, 0.5, 1)
        assert len(t2.snapshots) == 0


# ─── create_tracker extras ────────────────────────────────────────────────────

class TestCreateTrackerExtra:
    def test_returns_fresh_tracker_each_time(self):
        t1 = create_tracker()
        t2 = create_tracker()
        assert t1 is not t2

    def test_no_params_by_default(self):
        t = create_tracker()
        assert t.params == {} or isinstance(t.params, dict)

    def test_params_with_int_and_float(self):
        t = create_tracker(n=5, tol=1e-4)
        assert t.params["n"] == 5
        assert t.params["tol"] == pytest.approx(1e-4)


# ─── record_snapshot extras ───────────────────────────────────────────────────

class TestRecordSnapshotExtra:
    def test_many_snapshots_appended(self):
        t = create_tracker()
        for i in range(50):
            record_snapshot(t, i, float(i) / 50, 1)
        assert len(t.snapshots) == 50

    def test_iteration_monotone_by_insertion(self):
        t = _filled_tracker([0.1, 0.5, 0.9, 0.3, 0.7])
        iters = [s.iteration for s in t.snapshots]
        assert iters == sorted(iters)

    def test_score_float_coercion(self):
        t = create_tracker()
        record_snapshot(t, 0, np.float64(0.75), 1)
        assert isinstance(t.snapshots[0].score, float)

    def test_n_placed_coercion(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, np.int32(7))
        assert isinstance(t.snapshots[0].n_placed, int)

    def test_extra_kwargs_all_stored(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.4, 2, alpha=0.1, beta=0.9, gamma=0.5)
        s = t.snapshots[0]
        assert s.extra["alpha"] == pytest.approx(0.1)
        assert s.extra["beta"] == pytest.approx(0.9)
        assert s.extra["gamma"] == pytest.approx(0.5)


# ─── detect_convergence extras ────────────────────────────────────────────────

class TestDetectConvergenceExtra:
    def test_single_point_returns_none(self):
        t = _filled_tracker([0.5])
        assert detect_convergence(t, window=2) is None

    def test_two_identical_window_2_converges(self):
        t = _filled_tracker([0.5, 0.5])
        result = detect_convergence(t, window=2, tol=1e-9)
        assert result is not None

    def test_returns_int(self):
        t = _filled_tracker([0.9, 0.9, 0.9, 0.9])
        result = detect_convergence(t, window=3, tol=1e-9)
        assert isinstance(result, int)

    def test_large_tol_always_converges(self):
        t = _filled_tracker([0.0, 0.5, 1.0, 0.9, 0.95])
        result = detect_convergence(t, window=3, tol=100.0)
        assert result is not None

    def test_convergence_at_tail(self):
        scores = [0.1, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8]
        t = _filled_tracker(scores)
        result = detect_convergence(t, window=4, tol=1e-9)
        assert result is not None
        assert result >= 2

    def test_all_zeros_converge(self):
        t = _filled_tracker([0.0] * 10)
        result = detect_convergence(t, window=5, tol=1e-9)
        assert result is not None


# ─── extract_best_iteration extras ───────────────────────────────────────────

class TestExtractBestIterationExtra:
    def test_all_zeros_returns_first(self):
        t = _filled_tracker([0.0, 0.0, 0.0])
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.0)

    def test_decreasing_scores_best_is_first(self):
        t = _filled_tracker([0.9, 0.7, 0.5, 0.3])
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.9)
        assert best.iteration == 0

    def test_best_score_is_maximum(self):
        scores = [0.3, 0.5, 0.7, 0.2, 0.6]
        t = _filled_tracker(scores)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(max(scores))

    def test_best_returns_snapshot_instance(self):
        t = _filled_tracker([0.4, 0.8])
        best = extract_best_iteration(t)
        assert isinstance(best, ScoreSnapshot)

    def test_n_placed_preserved_in_best(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.3, n_placed=5)
        record_snapshot(t, 1, 0.9, n_placed=7)
        best = extract_best_iteration(t)
        assert best.n_placed == 7


# ─── summarize_tracker extras ─────────────────────────────────────────────────

class TestSummarizeTrackerExtra:
    def test_single_snapshot_summary(self):
        t = _filled_tracker([0.5])
        s = summarize_tracker(t)
        assert s["n_snapshots"] == 1
        assert s["best_score"] == pytest.approx(0.5)
        assert s["worst_score"] == pytest.approx(0.5)

    def test_std_zero_for_uniform(self):
        t = _filled_tracker([0.5] * 5)
        s = summarize_tracker(t)
        assert s["std_score"] == pytest.approx(0.0, abs=1e-9)

    def test_std_positive_for_varied(self):
        t = _filled_tracker([0.0, 0.5, 1.0])
        s = summarize_tracker(t)
        assert s["std_score"] > 0.0

    def test_best_iteration_index_is_int(self):
        t = _filled_tracker([0.1, 0.9, 0.5])
        s = summarize_tracker(t)
        assert isinstance(s["best_iteration"], int)

    def test_first_last_iteration_equal_for_single(self):
        t = _filled_tracker([0.7])
        s = summarize_tracker(t)
        assert s["first_iteration"] == s["last_iteration"]

    def test_n_snapshots_correct(self):
        t = _filled_tracker([0.1] * 10)
        assert summarize_tracker(t)["n_snapshots"] == 10


# ─── smooth_scores extras ─────────────────────────────────────────────────────

class TestSmoothScoresExtra:
    def test_single_element_window_1(self):
        t = _filled_tracker([0.7])
        result = smooth_scores(t, window=1)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.7, abs=0.5)

    def test_large_window_equal_total_size(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        t = _filled_tracker(scores)
        result = smooth_scores(t, window=len(scores))
        assert len(result) == len(scores)

    def test_values_are_finite(self):
        scores = [0.2, 0.8, 0.4, 0.6]
        t = _filled_tracker(scores)
        result = smooth_scores(t, window=2)
        assert all(np.isfinite(v) for v in result)

    def test_ndarray_float64(self):
        t = _filled_tracker([0.1, 0.2, 0.3])
        result = smooth_scores(t, window=2)
        assert result.dtype == np.float64

    def test_monotone_input_approximately_preserved(self):
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        t = _filled_tracker(scores)
        result = smooth_scores(t, window=2)
        # Smoothed monotone series should still be roughly increasing in middle
        assert result[2] > result[0]

    def test_window_larger_than_data_ok(self):
        t = _filled_tracker([0.4, 0.6])
        result = smooth_scores(t, window=10)
        assert isinstance(result, np.ndarray)
        assert len(result) >= 2
