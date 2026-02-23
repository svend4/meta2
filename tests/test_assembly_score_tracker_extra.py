"""Extra tests for puzzle_reconstruction/assembly/score_tracker.py"""
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


# ─── TestScoreSnapshotExtra ───────────────────────────────────────────────────

class TestScoreSnapshotExtra:
    def test_large_iteration(self):
        snap = ScoreSnapshot(iteration=10000, score=0.9, n_placed=100)
        assert snap.iteration == 10000

    def test_score_zero(self):
        snap = ScoreSnapshot(iteration=0, score=0.0, n_placed=0)
        assert snap.score == pytest.approx(0.0)

    def test_score_one(self):
        snap = ScoreSnapshot(iteration=0, score=1.0, n_placed=5)
        assert snap.score == pytest.approx(1.0)

    def test_n_placed_zero(self):
        snap = ScoreSnapshot(iteration=0, score=0.5, n_placed=0)
        assert snap.n_placed == 0

    def test_extra_multiple_keys(self):
        snap = ScoreSnapshot(iteration=1, score=0.7, n_placed=5,
                             extra={"a": 1, "b": 2.0, "c": "x"})
        assert snap.extra["a"] == 1
        assert snap.extra["b"] == pytest.approx(2.0)
        assert snap.extra["c"] == "x"

    def test_extra_empty_by_default(self):
        snap = ScoreSnapshot(iteration=0, score=0.5, n_placed=1)
        assert snap.extra == {}


# ─── TestScoreTrackerExtra ────────────────────────────────────────────────────

class TestScoreTrackerExtra:
    def test_params_multiple_keys(self):
        t = ScoreTracker(params={"tol": 1e-3, "window": 5, "max_iter": 100})
        assert t.params["tol"] == pytest.approx(1e-3)
        assert t.params["window"] == 5
        assert t.params["max_iter"] == 100

    def test_snapshots_list_empty(self):
        t = ScoreTracker()
        assert isinstance(t.snapshots, list)
        assert len(t.snapshots) == 0

    def test_params_default_empty(self):
        t = ScoreTracker()
        assert t.params == {}


# ─── TestCreateTrackerExtra ───────────────────────────────────────────────────

class TestCreateTrackerExtra:
    def test_many_params(self):
        t = create_tracker(tol=1e-4, window=10, max_iter=500, lr=0.01)
        assert t.params["tol"] == pytest.approx(1e-4)
        assert t.params["window"] == 10
        assert t.params["max_iter"] == 500
        assert t.params["lr"] == pytest.approx(0.01)

    def test_returns_score_tracker(self):
        assert isinstance(create_tracker(), ScoreTracker)

    def test_no_params(self):
        t = create_tracker()
        assert t.params == {}


# ─── TestRecordSnapshotExtra ──────────────────────────────────────────────────

class TestRecordSnapshotExtra:
    def test_ten_snapshots(self):
        t = create_tracker()
        for i in range(10):
            record_snapshot(t, i, float(i) / 10, i)
        assert len(t.snapshots) == 10

    def test_score_monotone_increasing(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.2, i)
        scores = [s.score for s in t.snapshots]
        assert scores == sorted(scores)

    def test_extra_independent_per_snapshot(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 3, loss=0.1)
        record_snapshot(t, 1, 0.6, 4, loss=0.05)
        assert t.snapshots[0].extra["loss"] == pytest.approx(0.1)
        assert t.snapshots[1].extra["loss"] == pytest.approx(0.05)

    def test_large_n_placed(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 9999)
        assert t.snapshots[0].n_placed == 9999

    def test_many_extra_kwargs(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 5, loss=0.1, acc=0.9, f1=0.8, prec=0.85, rec=0.75)
        snap = t.snapshots[0]
        assert snap.extra["loss"] == pytest.approx(0.1)
        assert snap.extra["acc"] == pytest.approx(0.9)
        assert snap.extra["f1"] == pytest.approx(0.8)

    def test_returns_same_tracker(self):
        t = create_tracker()
        ret = record_snapshot(t, 0, 0.5, 3)
        assert ret is t


# ─── TestDetectConvergenceExtra ───────────────────────────────────────────────

class TestDetectConvergenceExtra:
    def test_very_tight_tolerance_no_convergence(self):
        t = create_tracker()
        for i in range(8):
            record_snapshot(t, i, 0.9 + i * 1e-3, i)
        result = detect_convergence(t, window=3, tol=1e-10)
        assert result is None

    def test_loose_tolerance_converges(self):
        t = create_tracker()
        for i in range(6):
            record_snapshot(t, i, 0.9 + i * 1e-3, i)
        result = detect_convergence(t, window=3, tol=0.01)
        assert result is not None

    def test_window_exactly_3(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.8, 1)
        record_snapshot(t, 1, 0.8, 2)
        record_snapshot(t, 2, 0.8, 3)
        result = detect_convergence(t, window=3, tol=1e-4)
        assert result is not None

    def test_ten_constant_snapshots(self):
        t = create_tracker()
        for i in range(10):
            record_snapshot(t, i, 0.75, i)
        result = detect_convergence(t, window=5, tol=1e-4)
        assert result is not None
        assert isinstance(result, int)

    def test_converge_at_specific_iteration(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.3, i)  # varying
        for i in range(5, 12):
            record_snapshot(t, i, 0.99, i)  # constant
        result = detect_convergence(t, window=4, tol=1e-4)
        assert result is not None
        assert result >= 5

    def test_returns_int_type(self):
        t = create_tracker()
        for i in range(6):
            record_snapshot(t, i, 0.5, i)
        result = detect_convergence(t, window=3, tol=1e-4)
        assert isinstance(result, int)


# ─── TestExtractBestIterationExtra ────────────────────────────────────────────

class TestExtractBestIterationExtra:
    def test_ten_snapshots_returns_max(self):
        t = create_tracker()
        scores = [0.1, 0.3, 0.9, 0.5, 0.2, 0.7, 0.4, 0.6, 0.8, 0.0]
        for i, s in enumerate(scores):
            record_snapshot(t, i, s, i)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.9)
        assert best.iteration == 2

    def test_all_same_score(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, 0.6, i)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.6)

    def test_strictly_increasing_returns_last(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.25, i)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(1.0)

    def test_max_at_start(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.99, 10)
        for i in range(1, 5):
            record_snapshot(t, i, 0.3, i)
        best = extract_best_iteration(t)
        assert best.score == pytest.approx(0.99)
        assert best.iteration == 0

    def test_best_snapshot_n_placed(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 3)
        record_snapshot(t, 1, 0.9, 7)
        best = extract_best_iteration(t)
        assert best.n_placed == 7


# ─── TestSummarizeTrackerExtra ────────────────────────────────────────────────

class TestSummarizeTrackerExtra:
    def test_ten_snapshots_n_correct(self):
        t = create_tracker()
        for i in range(10):
            record_snapshot(t, i, float(i) * 0.1, i)
        r = summarize_tracker(t)
        assert r["n_snapshots"] == 10

    def test_first_last_iteration(self):
        t = create_tracker()
        for it in [5, 10, 15, 20, 25]:
            record_snapshot(t, it, 0.5, it)
        r = summarize_tracker(t)
        assert r["first_iteration"] == 5
        assert r["last_iteration"] == 25

    def test_best_iteration_middle(self):
        t = create_tracker()
        scores = [0.3, 0.5, 0.9, 0.4, 0.2]
        for i, s in enumerate(scores):
            record_snapshot(t, i * 2, s, i)
        r = summarize_tracker(t)
        assert r["best_iteration"] == 4  # iteration = index 2 * 2

    def test_mean_score_exact(self):
        t = create_tracker()
        for s in [0.2, 0.6, 1.0]:
            record_snapshot(t, 0, s, 0)
        r = summarize_tracker(t)
        assert r["mean_score"] == pytest.approx(0.6, abs=1e-9)

    def test_std_score_nonneg(self):
        t = create_tracker()
        for i, s in enumerate([0.1, 0.5, 0.9]):
            record_snapshot(t, i, s, i)
        r = summarize_tracker(t)
        assert r["std_score"] >= 0.0

    def test_all_keys_present(self):
        t = create_tracker()
        record_snapshot(t, 0, 0.5, 1)
        r = summarize_tracker(t)
        required = {"n_snapshots", "best_score", "worst_score", "mean_score",
                    "std_score", "first_iteration", "last_iteration", "best_iteration"}
        assert required.issubset(r.keys())


# ─── TestSmoothScoresExtra ────────────────────────────────────────────────────

class TestSmoothScoresExtra:
    def test_window_3_basic(self):
        t = create_tracker()
        for i in range(7):
            record_snapshot(t, i, float(i) * 0.1, i)
        out = smooth_scores(t, window=3)
        assert len(out) == 7

    def test_large_window(self):
        t = create_tracker()
        for i in range(20):
            record_snapshot(t, i, float(i) * 0.05, i)
        out = smooth_scores(t, window=7)
        assert len(out) == 20

    def test_increasing_signal_smoothed(self):
        t = create_tracker()
        for i in range(10):
            record_snapshot(t, i, float(i) * 0.1, i)
        out = smooth_scores(t, window=3)
        assert out[0] <= out[-1]

    def test_output_dtype_float64(self):
        t = create_tracker()
        for i in range(5):
            record_snapshot(t, i, float(i) * 0.2, i)
        out = smooth_scores(t, window=3)
        assert out.dtype == np.float64

    def test_window_1_passthrough(self):
        t = create_tracker()
        scores = [0.1, 0.5, 0.9, 0.3]
        for i, s in enumerate(scores):
            record_snapshot(t, i, s, i)
        out = smooth_scores(t, window=1)
        np.testing.assert_allclose(out, np.array(scores, dtype=np.float64), atol=1e-10)

    def test_spike_smoothed(self):
        t = create_tracker()
        scores = [0.5, 0.5, 5.0, 0.5, 0.5, 0.5, 0.5]
        for i, s in enumerate(scores):
            record_snapshot(t, i, s, i)
        out = smooth_scores(t, window=3)
        assert out[2] < 5.0

    def test_constant_scores_preserved(self):
        t = create_tracker()
        for i in range(9):
            record_snapshot(t, i, 0.7, i)
        out = smooth_scores(t, window=5)
        np.testing.assert_allclose(out[2:-2], 0.7, atol=1e-9)
