"""Тесты для puzzle_reconstruction.matching.dtw."""
import math
import pytest
import numpy as np
from puzzle_reconstruction.matching.dtw import (
    dtw_distance,
    dtw_distance_mirror,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _line(n=20, slope=1.0) -> np.ndarray:
    """Parametric 1-D curve as (N, 1) array."""
    t = np.linspace(0, 1, n)
    return (slope * t).reshape(-1, 1).astype(np.float64)


def _curve2d(n=20, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, 2)).astype(np.float64)


# ─── TestDtwDistance ──────────────────────────────────────────────────────────

class TestDtwDistance:
    def test_returns_float(self):
        assert isinstance(dtw_distance(_line(10), _line(10)), float)

    def test_identical_curves_zero(self):
        a = _line(10)
        assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric(self):
        a = _curve2d(15, seed=0)
        b = _curve2d(15, seed=1)
        d1 = dtw_distance(a, b)
        d2 = dtw_distance(b, a)
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_non_negative(self):
        assert dtw_distance(_line(10), _line(10, slope=2.0)) >= 0.0

    def test_empty_first_returns_inf(self):
        a = np.zeros((0, 1))
        b = _line(5)
        assert math.isinf(dtw_distance(a, b))

    def test_empty_second_returns_inf(self):
        a = _line(5)
        b = np.zeros((0, 1))
        assert math.isinf(dtw_distance(a, b))

    def test_different_lengths(self):
        a = _line(10)
        b = _line(20)
        d = dtw_distance(a, b)
        assert d >= 0.0
        assert not math.isinf(d)

    def test_2d_curves(self):
        a = _curve2d(10, 0)
        b = _curve2d(10, 1)
        d = dtw_distance(a, b)
        assert d >= 0.0

    def test_window_zero(self):
        a = _line(10)
        b = _line(10)
        d = dtw_distance(a, b, window=0)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_closer_curves_smaller_distance(self):
        base = _line(20)
        close = base + 0.01
        far = base + 1.0
        d_close = dtw_distance(base, close)
        d_far = dtw_distance(base, far)
        assert d_close < d_far

    def test_single_point_curves(self):
        a = np.array([[0.5]])
        b = np.array([[0.5]])
        assert dtw_distance(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_normalized_by_length(self):
        a = _line(10)
        b = _line(10, slope=2.0)
        d = dtw_distance(a, b)
        # Normalized result should be <= raw max distance
        assert d < 10.0

    def test_large_window(self):
        a = _line(15)
        b = _line(15)
        d = dtw_distance(a, b, window=100)
        assert d == pytest.approx(0.0, abs=1e-9)


# ─── TestDtwDistanceMirror ────────────────────────────────────────────────────

class TestDtwDistanceMirror:
    def test_returns_float(self):
        assert isinstance(dtw_distance_mirror(_line(10), _line(10)), float)

    def test_identical_curves_zero(self):
        a = _line(10)
        assert dtw_distance_mirror(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_mirror_curve_detected(self):
        n = 20
        t = np.linspace(0, 1, n).reshape(-1, 1)
        a = t.copy()
        b = t[::-1].copy()  # perfectly mirrored
        d_mirror = dtw_distance_mirror(a, b)
        d_direct = dtw_distance(a, b)
        # Mirror distance should be <= direct distance
        assert d_mirror <= d_direct + 1e-9

    def test_non_negative(self):
        a = _curve2d(10, 0)
        b = _curve2d(10, 1)
        assert dtw_distance_mirror(a, b) >= 0.0

    def test_leq_direct_distance(self):
        a = _curve2d(12, 0)
        b = _curve2d(12, 1)
        d_mirror = dtw_distance_mirror(a, b)
        d_direct = dtw_distance(a, b)
        assert d_mirror <= d_direct + 1e-9

    def test_different_lengths(self):
        a = _line(10)
        b = _line(15)
        d = dtw_distance_mirror(a, b)
        assert d >= 0.0

    def test_empty_returns_inf(self):
        a = np.zeros((0, 1))
        b = _line(5)
        assert math.isinf(dtw_distance_mirror(a, b))

    def test_2d_curves(self):
        a = _curve2d(10, 2)
        b = _curve2d(10, 3)
        d = dtw_distance_mirror(a, b)
        assert isinstance(d, float)
