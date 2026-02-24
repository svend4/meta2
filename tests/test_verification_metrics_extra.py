"""Extra tests for puzzle_reconstruction/verification/metrics.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.metrics import (
    ReconstructionMetrics,
    BenchmarkResult,
    evaluate_reconstruction,
    compare_methods,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gt():
    """Ground truth: 3 fragments placed at known positions (close enough for adjacency)."""
    return {
        0: (np.array([0.0, 0.0]), 0.0),
        1: (np.array([50.0, 0.0]), 0.0),
        2: (np.array([0.0, 50.0]), 0.0),
    }


def _perfect_pred():
    """Prediction identical to ground truth."""
    return _gt()


def _shifted_pred():
    """Prediction with varying positional errors and angle offsets."""
    return {
        0: (np.array([0.0, 0.0]), 0.0),      # Reference matches GT
        1: (np.array([60.0, 5.0]), 0.1),      # Shifted vs GT [50,0]
        2: (np.array([5.0, 60.0]), -0.1),     # Shifted vs GT [0,50]
    }


def _bad_pred():
    """Prediction with large positional errors."""
    return {
        0: (np.array([500.0, 500.0]), 1.57),
        1: (np.array([700.0, 500.0]), 1.57),
        2: (np.array([500.0, 700.0]), 1.57),
    }


def _metrics(na=1.0, dc=1.0, perfect=True, rmse=0.0, ang=0.0,
             n=3, n_correct=3, n_total=3, emr=1.0):
    return ReconstructionMetrics(
        neighbor_accuracy=na, direct_comparison=dc, perfect=perfect,
        position_rmse=rmse, angular_error_deg=ang,
        n_fragments=n, n_correct_pairs=n_correct,
        n_total_pairs=n_total, edge_match_rate=emr,
    )


# ─── ReconstructionMetrics ───────────────────────────────────────────────────

class TestReconstructionMetricsExtra:
    def test_fields(self):
        m = _metrics()
        assert m.neighbor_accuracy == pytest.approx(1.0)
        assert m.direct_comparison == pytest.approx(1.0)
        assert m.perfect is True
        assert m.position_rmse == pytest.approx(0.0)
        assert m.angular_error_deg == pytest.approx(0.0)
        assert m.n_fragments == 3

    def test_summary(self):
        m = _metrics()
        s = m.summary()
        assert "3" in s
        assert "100.0%" in s

    def test_summary_imperfect(self):
        m = _metrics(na=0.5, dc=0.5, perfect=False, rmse=10.0, ang=5.0)
        s = m.summary()
        assert "50.0%" in s


# ─── BenchmarkResult ─────────────────────────────────────────────────────────

class TestBenchmarkResultExtra:
    def test_fields(self):
        m = _metrics()
        br = BenchmarkResult(method="greedy", metrics=m, runtime_sec=1.5)
        assert br.method == "greedy"
        assert br.runtime_sec == pytest.approx(1.5)


# ─── evaluate_reconstruction ─────────────────────────────────────────────────

class TestEvaluateReconstructionExtra:
    def test_perfect(self):
        m = evaluate_reconstruction(_perfect_pred(), _gt())
        assert m.direct_comparison == pytest.approx(1.0)
        assert m.position_rmse == pytest.approx(0.0)
        assert m.angular_error_deg == pytest.approx(0.0)
        assert m.perfect is True

    def test_shifted(self):
        m = evaluate_reconstruction(_shifted_pred(), _gt())
        assert m.position_rmse > 0.0
        assert m.angular_error_deg > 0.0

    def test_bad(self):
        m = evaluate_reconstruction(_bad_pred(), _gt())
        assert m.direct_comparison < 1.0

    def test_empty(self):
        m = evaluate_reconstruction({}, {})
        assert m.n_fragments == 0
        assert m.perfect is False

    def test_partial_overlap(self):
        pred = {0: (np.array([0.0, 0.0]), 0.0)}
        gt = {0: (np.array([0.0, 0.0]), 0.0), 1: (np.array([100.0, 0.0]), 0.0)}
        m = evaluate_reconstruction(pred, gt)
        assert m.n_fragments == 1

    def test_custom_adjacency(self):
        adj = [(0, 1), (1, 2)]
        m = evaluate_reconstruction(_perfect_pred(), _gt(), adjacency=adj)
        assert m.n_total_pairs == 2

    def test_custom_tolerance(self):
        m = evaluate_reconstruction(
            _shifted_pred(), _gt(),
            position_tolerance=1.0, angle_tolerance_deg=1.0,
        )
        # Tight tolerance → fewer correct
        assert m.direct_comparison < 1.0


# ─── compare_methods ─────────────────────────────────────────────────────────

class TestCompareMethodsExtra:
    def test_empty(self):
        s = compare_methods([])
        assert "Метод" in s

    def test_with_results(self):
        r1 = BenchmarkResult("greedy", _metrics(na=0.9), 1.0)
        r2 = BenchmarkResult("sa", _metrics(na=0.8), 2.0)
        s = compare_methods([r1, r2])
        assert "greedy" in s
        assert "sa" in s

    def test_sorted_by_na(self):
        r1 = BenchmarkResult("low", _metrics(na=0.3), 1.0)
        r2 = BenchmarkResult("high", _metrics(na=0.9), 2.0)
        s = compare_methods([r1, r2])
        lines = s.strip().split("\n")
        # high should appear before low (sorted by NA desc)
        high_idx = next(i for i, l in enumerate(lines) if "high" in l)
        low_idx = next(i for i, l in enumerate(lines) if "low" in l)
        assert high_idx < low_idx
