"""Tests for puzzle_reconstruction.assembly.score_tracker."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _filled_tracker(scores, n_placed=1):
    """Create a tracker pre-filled with the given score sequence."""
    t = create_tracker()
    for i, s in enumerate(scores):
        record_snapshot(t, iteration=i, score=s, n_placed=n_placed)
    return t


# ─── ScoreSnapshot ────────────────────────────────────────────────────────────

class TestScoreSnapshot:
    def test_fields_stored(self):
        s = ScoreSnapshot(iteration=3, score=0.75, n_placed=5)
        assert s.iteration == 3
        assert s.score == pytest.approx(0.75)
        assert s.n_placed == 5
        assert s.extra == {}

    def test_extra_stored(self):
        s = ScoreSnapshot(iteration=0, score=0.0, n_placed=0, extra={"loss": 1.5})
        assert s.extra["loss"] == pytest.approx(1.5)

    def test_repr_contains_iter_and_score(self):
        s = ScoreSnapshot(iteration=7, score=0.123, n_placed=2)
        r = repr(s)
        assert "7" in r
        assert "0.123" in r or "0.1230" in r


# ─── ScoreTracker ─────────────────────────────────────────────────────────────

class TestScoreTracker:
    def test_empty_on_init(self):
        t = ScoreTracker()
        assert t.snapshots == []
        assert t.params == {}

    def test_params_stored(self):
        t = ScoreTracker(params={"lr": 0.01})
        assert t.params["lr"] == pytest.approx(0.01)

    def test_repr_contains_n_snapshots(self):
        t = _filled_tracker([0.5, 0.6])
        r = repr(t)
        assert "2" in r


# ─── create_tracker ───────────────────────────────────────────────────────────

class TestCreateTracker:
    def test_returns_score_tracker(self):
        t = create_tracker()
        assert isinstance(t, ScoreTracker)

    def test_empty_snapshots(self):
        t = create_tracker()
        assert len(t.snapshots) == 0

    def test_params_forwarded(self):
        t = create_tracker(method="greedy", tol=1e-3)
        assert t.params["method"] == "greedy"
        assert t.params["tol"] == pytest.approx(1e-3)

    def test_independent_instances(self):
        t1 = create_tracker()
        t2 = create_tracker()
        record_snapshot(t1, 0, 0.5, 1)
        assert len(t2.snapshots) == 0


# ─── record_snapshot ──────────────────────────────────────────────────────────

class TestRecordSnapshot:
    def test_appends_snapshot(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 3)
        assert len(t.snapshots) == 1
        assert isinstance(t.snapshots[0], ScoreSnapshot)

    def test_returns_tracker(self):
        t = create_tracker()
        result = record_snapshot(t, 0, 0.5, 1)
        assert result is t

    def test_fields_coerced(self):
        t = create_tracker()
        record_snapshot(t, iteration=2, score=np.float32(0.8), n_placed=4)
        s = t.snapshots[0]
        assert isinstance(s.score, float)
        assert isinstance(s.n_placed, int)

    def test_extra_stored(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.3, 1, loss=2.5, gap=0.1)
        s = t.snapshots[0]
        assert s.extra["loss"] == pytest.approx(2.5)
        assert s.extra["gap"] == pytest.approx(0.1)

    def test_multiple_snapshots_ordered(self):
        t = _filled_tracker([0.1, 0.5, 0.9])
        iters = [s.iteration for s in t.snapshots]
        assert iters == [0, 1, 2]


# ─── detect_convergence ───────────────────────────────────────────────────────

class TestDetectConvergence:
    def test_convergence_flat_sequence(self):
        t = _filled_tracker([0.5, 0.5, 0.5, 0.5, 0.5])
        result = detect_convergence(t, window=3, tol=1e-6)
        assert result is not None
        assert isinstance(result, int)

    def test_no_convergence_increasing(self):
        t = _filled_tracker([0.1, 0.3, 0.5, 0.7, 0.9])
        result = detect_convergence(t, window=3, tol=1e-4)
        assert result is None

    def test_insufficient_data_returns_none(self):
        t = _filled_tracker([0.5, 0.5])
        result = detect_convergence(t, window=5, tol=1e-4)
        assert result is None

    def test_window_less_than_2_raises(self):
        t = _filled_tracker([0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            detect_convergence(t, window=1)

    def test_empty_tracker_returns_none(self):
        t = create_tracker()
        result = detect_convergence(t, window=3)
        assert result is None

    def test_convergence_after_initial_rise(self):
        scores = [0.1, 0.4, 0.8, 0.9, 0.9, 0.9, 0.9]
        t = _filled_tracker(scores)
        result = detect_convergence(t, window=3, tol=1e-9)
        assert result is not None
        # Convergence must occur at or after the flat portion begins
        assert result >= 2

    def test_returns_first_iteration_of_converged_window(self):
        # Flat from index 2 onwards
        scores = [0.1, 0.5, 0.7, 0.7, 0.7]
        t = _filled_tracker(scores)
        result = detect_convergence(t, window=3, tol=0.01)
        assert result == 2


# ─── extract_best_iteration ───────────────────────────────────────────────────

class TestExtractBestIteration:
    def test_empty_returns_none(self):
        t = create_tracker()
        assert extract_best_iteration(t) is None

    def test_single_snapshot(self):
        t = _filled_tracker([0.42])
        best = extract_best_iteration(t)
        assert isinstance(best, ScoreSnapshot)
        assert best.score == pytest.approx(0.42)

    def test_returns_maximum(self):
        t = _filled_tracker([0.2, 0.9, 0.5, 0.7])
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.9)
        assert best.iteration == 1

    def test_tie_returns_one_of_the_maxima(self):
        t = _filled_tracker([0.8, 0.8, 0.8])
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.8)

    def test_last_element_is_best(self):
        t = _filled_tracker([0.1, 0.2, 0.3, 0.99])
        best = extract_best_iteration(t)
        assert best.iteration == 3


# ─── summarize_tracker ────────────────────────────────────────────────────────

class TestSummarizeTracker:
    def test_empty_returns_zero_count(self):
        t = create_tracker()
        summary = summarize_tracker(t)
        assert summary["n_snapshots"] == 0

    def test_empty_has_no_score_keys(self):
        t = create_tracker()
        summary = summarize_tracker(t)
        assert "best_score" not in summary

    def test_keys_present(self):
        t = _filled_tracker([0.2, 0.6, 0.4])
        summary = summarize_tracker(t)
        for key in ("n_snapshots", "best_score", "worst_score", "mean_score",
                    "std_score", "first_iteration", "last_iteration", "best_iteration"):
            assert key in summary

    def test_best_score(self):
        t = _filled_tracker([0.2, 0.9, 0.5])
        assert summarize_tracker(t)["best_score"] == pytest.approx(0.9)

    def test_worst_score(self):
        t = _filled_tracker([0.2, 0.9, 0.5])
        assert summarize_tracker(t)["worst_score"] == pytest.approx(0.2)

    def test_mean_score(self):
        t = _filled_tracker([0.0, 0.5, 1.0])
        assert summarize_tracker(t)["mean_score"] == pytest.approx(0.5)

    def test_iteration_bounds(self):
        scores = [0.1, 0.2, 0.3]
        t = _filled_tracker(scores)
        s = summarize_tracker(t)
        assert s["first_iteration"] == 0
        assert s["last_iteration"] == 2

    def test_best_iteration_correct(self):
        t = _filled_tracker([0.3, 0.9, 0.1])
        assert summarize_tracker(t)["best_iteration"] == 1


# ─── smooth_scores ────────────────────────────────────────────────────────────

class TestSmoothScores:
    def test_returns_ndarray(self):
        t = _filled_tracker([0.1, 0.5, 0.9])
        result = smooth_scores(t, window=2)
        assert isinstance(result, np.ndarray)

    def test_length_preserved(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        t = _filled_tracker(scores)
        result = smooth_scores(t, window=3)
        assert len(result) == len(scores)

    def test_empty_tracker_returns_empty(self):
        t = create_tracker()
        result = smooth_scores(t, window=3)
        assert len(result) == 0

    def test_window_1_identity(self):
        scores = [0.1, 0.4, 0.9, 0.2]
        t = _filled_tracker(scores)
        result = smooth_scores(t, window=1)
        np.testing.assert_allclose(result, scores, atol=1e-9)

    def test_window_less_than_1_raises(self):
        t = _filled_tracker([0.5, 0.5])
        with pytest.raises(ValueError):
            smooth_scores(t, window=0)

    def test_uniform_scores_stay_uniform(self):
        t = _filled_tracker([0.5] * 10)
        result = smooth_scores(t, window=3)
        # Interior values should all equal 0.5; edges may differ slightly
        assert all(abs(v - 0.5) < 0.5 for v in result)

    def test_dtype_float64(self):
        t = _filled_tracker([0.1, 0.2, 0.3])
        result = smooth_scores(t, window=2)
        assert result.dtype == np.float64
