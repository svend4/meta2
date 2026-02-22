"""Тесты для puzzle_reconstruction/assembly/score_tracker.py."""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.score_tracker import (
    ScoreSnapshot,
    ScoreTracker,
    create_tracker,
    record_snapshot,
    detect_convergence,
    extract_best_iteration,
    summarize_tracker,
    smooth_scores,
)


# ─── ScoreSnapshot ────────────────────────────────────────────────────────────

class TestScoreSnapshot:
    def test_creation_basic(self):
        snap = ScoreSnapshot(iteration=0, score=0.5, n_placed=3)
        assert snap.iteration == 0
        assert snap.score == 0.5
        assert snap.n_placed == 3
        assert snap.extra == {}

    def test_creation_with_extra(self):
        snap = ScoreSnapshot(iteration=5, score=0.9, n_placed=10, extra={"loss": 0.1})
        assert snap.extra["loss"] == 0.1

    def test_fields_accessible(self):
        snap = ScoreSnapshot(iteration=1, score=0.75, n_placed=7)
        assert hasattr(snap, "iteration")
        assert hasattr(snap, "score")
        assert hasattr(snap, "n_placed")
        assert hasattr(snap, "extra")


# ─── ScoreTracker ─────────────────────────────────────────────────────────────

class TestScoreTracker:
    def test_creation_defaults(self):
        tracker = ScoreTracker()
        assert tracker.snapshots == []
        assert tracker.params == {}

    def test_creation_with_params(self):
        tracker = ScoreTracker(params={"tol": 1e-3})
        assert tracker.params["tol"] == 1e-3


# ─── create_tracker ───────────────────────────────────────────────────────────

class TestCreateTracker:
    def test_returns_empty_tracker(self):
        t = create_tracker()
        assert isinstance(t, ScoreTracker)
        assert t.snapshots == []
        assert t.params == {}

    def test_stores_params(self):
        t = create_tracker(tol=1e-4, window=5)
        assert t.params["tol"] == 1e-4
        assert t.params["window"] == 5

    def test_empty_snapshots(self):
        t = create_tracker()
        assert len(t.snapshots) == 0


# ─── record_snapshot ──────────────────────────────────────────────────────────

class TestRecordSnapshot:
    def test_mutates_tracker(self):
        t = create_tracker()
        returned = record_snapshot(t, 0, 0.5, 3)
        assert returned is t

    def test_appends_snapshot(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 3)
        assert len(t.snapshots) == 1

    def test_snapshot_values(self):
        t = create_tracker()
        record_snapshot(t, 7, 0.85, 10)
        snap = t.snapshots[0]
        assert snap.iteration == 7
        assert snap.score == pytest.approx(0.85)
        assert snap.n_placed == 10

    def test_multiple_snapshots(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.1, i)
        assert len(t.snapshots) == 5

    def test_extra_kwargs_stored(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 3, loss=0.12, precision=0.9)
        snap = t.snapshots[0]
        assert snap.extra["loss"] == pytest.approx(0.12)
        assert snap.extra["precision"] == pytest.approx(0.9)

    def test_score_converted_to_float(self):
        t = create_tracker()
        record_snapshot(t, 0, 1, 3)
        assert isinstance(t.snapshots[0].score, float)

    def test_n_placed_converted_to_int(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 3)
        assert isinstance(t.snapshots[0].n_placed, int)

    def test_append_order_preserved(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.3, 1)
        record_snapshot(t, 1, 0.7, 2)
        assert t.snapshots[0].score == pytest.approx(0.3)
        assert t.snapshots[1].score == pytest.approx(0.7)


# ─── detect_convergence ───────────────────────────────────────────────────────

class TestDetectConvergence:
    def test_window_less_than_2_raises(self):
        t = create_tracker()
        with pytest.raises(ValueError, match="window"):
            detect_convergence(t, window=1)

    def test_window_zero_raises(self):
        t = create_tracker()
        with pytest.raises(ValueError):
            detect_convergence(t, window=0)

    def test_empty_tracker_returns_none(self):
        t = create_tracker()
        assert detect_convergence(t, window=2) is None

    def test_too_few_snapshots_returns_none(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 1)
        assert detect_convergence(t, window=3) is None

    def test_converged_constant_scores(self):
        t = create_tracker()
        for i in range(6):
            record_snapshot(t, i, 0.9, i)
        result = detect_convergence(t, window=3, tol=1e-4)
        assert result is not None
        assert isinstance(result, int)

    def test_not_converged_increasing_scores(self):
        t = create_tracker()
        for i in range(10):
            record_snapshot(t, i, float(i) * 0.1, i)
        result = detect_convergence(t, window=3, tol=1e-4)
        assert result is None

    def test_returns_iteration_number_not_index(self):
        t = create_tracker()
        # Start from iteration 10, all same score
        for i in range(5):
            record_snapshot(t, 10 + i, 0.8, i)
        result = detect_convergence(t, window=3, tol=1e-4)
        # Should return iteration value, not 0-based index
        assert result == 10  # first window's first iteration

    def test_convergence_at_end(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.2, i)  # diverging
        for i in range(5, 10):
            record_snapshot(t, i, 0.95, i)  # converging
        result = detect_convergence(t, window=3, tol=1e-4)
        assert result is not None
        assert result >= 5

    def test_tolerance_boundary(self):
        t = create_tracker()
        # Small variations within tol
        scores = [0.90000, 0.90005, 0.90003, 0.90002, 0.90001]
        for i, s in enumerate(scores):
            record_snapshot(t, i, s, i)
        result = detect_convergence(t, window=3, tol=1e-2)
        assert result is not None

    def test_window_equals_2(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 1)
        record_snapshot(t, 1, 0.5, 2)
        result = detect_convergence(t, window=2, tol=1e-4)
        assert result is not None


# ─── extract_best_iteration ───────────────────────────────────────────────────

class TestExtractBestIteration:
    def test_empty_returns_none(self):
        t = create_tracker()
        assert extract_best_iteration(t) is None

    def test_single_snapshot(self):
        t = create_tracker()
        record_snapshot(t, 3, 0.75, 5)
        best = extract_best_iteration(t)
        assert best is not None
        assert best.score == pytest.approx(0.75)
        assert best.iteration == 3

    def test_returns_max_score(self):
        t = create_tracker()
        scores = [0.5, 0.9, 0.3, 0.7]
        for i, s in enumerate(scores):
            record_snapshot(t, i, s, i)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.9)
        assert best.iteration == 1

    def test_returns_snapshot_object(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 1)
        best = extract_best_iteration(t)
        assert isinstance(best, ScoreSnapshot)

    def test_last_max_among_equal_scores(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.8, 1)
        record_snapshot(t, 1, 0.8, 2)
        record_snapshot(t, 2, 0.8, 3)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.8)


# ─── summarize_tracker ────────────────────────────────────────────────────────

class TestSummarizeTracker:
    def test_empty_returns_n_snapshots_0(self):
        t = create_tracker()
        result = summarize_tracker(t)
        assert result == {"n_snapshots": 0}

    def test_returns_dict(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 1)
        assert isinstance(summarize_tracker(t), dict)

    def test_n_snapshots_correct(self):
        t = create_tracker()
        for i in range(4):
            record_snapshot(t, i, float(i) * 0.2, i)
        result = summarize_tracker(t)
        assert result["n_snapshots"] == 4

    def test_best_score(self):
        t = create_tracker()
        for i, s in enumerate([0.3, 0.9, 0.5]):
            record_snapshot(t, i, s, i)
        result = summarize_tracker(t)
        assert result["best_score"] == pytest.approx(0.9)

    def test_worst_score(self):
        t = create_tracker()
        for i, s in enumerate([0.3, 0.9, 0.5]):
            record_snapshot(t, i, s, i)
        result = summarize_tracker(t)
        assert result["worst_score"] == pytest.approx(0.3)

    def test_mean_score(self):
        t = create_tracker()
        for i, s in enumerate([0.2, 0.4, 0.6]):
            record_snapshot(t, i, s, i)
        result = summarize_tracker(t)
        assert result["mean_score"] == pytest.approx(0.4)

    def test_std_score(self):
        t = create_tracker()
        for i, s in enumerate([0.0, 0.5, 1.0]):
            record_snapshot(t, i, s, i)
        result = summarize_tracker(t)
        assert "std_score" in result
        assert result["std_score"] >= 0.0

    def test_first_and_last_iteration(self):
        t = create_tracker()
        record_snapshot(t, 10, 0.5, 1)
        record_snapshot(t, 20, 0.7, 2)
        result = summarize_tracker(t)
        assert result["first_iteration"] == 10
        assert result["last_iteration"] == 20

    def test_best_iteration(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.3, 1)
        record_snapshot(t, 5, 0.95, 2)
        record_snapshot(t, 10, 0.7, 3)
        result = summarize_tracker(t)
        assert result["best_iteration"] == 5

    def test_all_required_keys(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 1)
        result = summarize_tracker(t)
        required = {"n_snapshots", "best_score", "worst_score", "mean_score",
                    "std_score", "first_iteration", "last_iteration", "best_iteration"}
        assert required.issubset(result.keys())


# ─── smooth_scores ────────────────────────────────────────────────────────────

class TestSmoothScores:
    def test_window_less_than_1_raises(self):
        t = create_tracker()
        with pytest.raises(ValueError, match="window"):
            smooth_scores(t, window=0)

    def test_window_negative_raises(self):
        t = create_tracker()
        with pytest.raises(ValueError):
            smooth_scores(t, window=-1)

    def test_empty_tracker_returns_empty(self):
        t = create_tracker()
        result = smooth_scores(t)
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_returns_float64(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.1, i)
        result = smooth_scores(t)
        assert result.dtype == np.float64

    def test_same_length_as_snapshots(self):
        t = create_tracker()
        for i in range(7):
            record_snapshot(t, i, float(i) * 0.1, i)
        result = smooth_scores(t, window=3)
        assert len(result) == 7

    def test_constant_scores_middle_unchanged(self):
        t = create_tracker()
        for i in range(7):
            record_snapshot(t, i, 0.5, i)
        result = smooth_scores(t, window=3)
        # Middle values (not at edges) should be 0.5
        np.testing.assert_allclose(result[1:-1], 0.5, atol=1e-10)

    def test_window_1_returns_original(self):
        t = create_tracker()
        scores = [0.1, 0.5, 0.3, 0.8, 0.2]
        for i, s in enumerate(scores):
            record_snapshot(t, i, s, i)
        result = smooth_scores(t, window=1)
        expected = np.array(scores, dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_single_snapshot_returns_array(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.75, 1)
        result = smooth_scores(t, window=1)
        # window=1: kernel=[1.0], so same length as snapshots
        assert len(result) == 1
        assert result[0] == pytest.approx(0.75)
