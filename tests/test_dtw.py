"""Тесты для puzzle_reconstruction.matching.dtw."""
import math
import pytest
import numpy as np
from puzzle_reconstruction.matching.dtw import (
    dtw_distance,
    dtw_distance_mirror,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _line(n: int, d: int = 2, val: float = 0.0) -> np.ndarray:
    """Кривая из n точек в d-мерном пространстве (все точки одинаковы)."""
    return np.full((n, d), val, dtype=float)


def _ramp(n: int, d: int = 2) -> np.ndarray:
    """Кривая от 0 до n-1 по каждой оси."""
    return np.column_stack([np.arange(n, dtype=float)] * d)


def _sine(n: int, freq: float = 1.0, amp: float = 1.0) -> np.ndarray:
    """Одномерная синусоида, обёрнутая в (N,1)."""
    t = np.linspace(0, 2 * math.pi * freq, n)
    return (amp * np.sin(t)).reshape(-1, 1)


# ─── TestDtwDistance ──────────────────────────────────────────────────────────

class TestDtwDistance:
    def test_identical_curves_zero(self):
        a = _ramp(10)
        assert dtw_distance(a, a) == pytest.approx(0.0)

    def test_identical_copies_zero(self):
        a = _ramp(10)
        b = a.copy()
        assert dtw_distance(a, b) == pytest.approx(0.0)

    def test_empty_a_inf(self):
        a = np.empty((0, 2))
        b = _ramp(5)
        assert math.isinf(dtw_distance(a, b))

    def test_empty_b_inf(self):
        a = _ramp(5)
        b = np.empty((0, 2))
        assert math.isinf(dtw_distance(a, b))

    def test_both_empty_inf(self):
        a = np.empty((0, 2))
        b = np.empty((0, 2))
        assert math.isinf(dtw_distance(a, b))

    def test_returns_float(self):
        a = _ramp(5)
        b = _ramp(5)
        result = dtw_distance(a, b)
        assert isinstance(result, float)

    def test_non_negative(self):
        a = _ramp(10)
        b = _ramp(10) + 5.0
        assert dtw_distance(a, b) >= 0.0

    def test_different_lengths(self):
        a = _ramp(10)
        b = _ramp(15)
        dist = dtw_distance(a, b)
        assert dist >= 0.0
        assert not math.isinf(dist)

    def test_1d_curves(self):
        a = np.arange(5, dtype=float).reshape(-1, 1)
        b = np.arange(5, dtype=float).reshape(-1, 1)
        assert dtw_distance(a, b) == pytest.approx(0.0)

    def test_shifted_curve_positive_dist(self):
        a = _ramp(10)
        b = _ramp(10) + 100.0
        dist = dtw_distance(a, b)
        assert dist > 0.0

    def test_normalization_by_length(self):
        # Curve const=1 vs const=0 → cost per step ≈ sqrt(2) * window / (n+m)
        a = _line(20, val=0.0)
        b = _line(20, val=1.0)
        dist = dtw_distance(a, b)
        # dist should be finite and positive
        assert 0.0 < dist < 100.0

    def test_small_window_same_length(self):
        a = _ramp(30)
        b = _ramp(30)
        dist = dtw_distance(a, b, window=1)
        assert dist == pytest.approx(0.0)

    def test_large_window_no_error(self):
        a = _ramp(20)
        b = _ramp(25)
        dist = dtw_distance(a, b, window=100)
        assert 0.0 <= dist < math.inf

    def test_single_point_curves(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        dist = dtw_distance(a, b)
        # 1 step, norm by (1+1)=2: dist = 5/2 = 2.5
        assert dist == pytest.approx(5.0 / 2.0)

    def test_triangle_inequality_approx(self):
        a = _ramp(10)
        b = _ramp(10) + 1.0
        c = _ramp(10) + 2.0
        d_ab = dtw_distance(a, b)
        d_bc = dtw_distance(b, c)
        d_ac = dtw_distance(a, c)
        # Приблизительное неравенство треугольника
        assert d_ac <= d_ab + d_bc + 1e-9

    def test_symmetry(self):
        a = _ramp(8)
        b = _ramp(12) * 0.5
        assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))

    def test_constant_curve_distance(self):
        a = np.zeros((5, 1))
        b = np.ones((5, 1)) * 3.0
        dist = dtw_distance(a, b)
        assert dist > 0.0

    def test_sine_identical(self):
        a = _sine(20)
        b = _sine(20)
        assert dtw_distance(a, b) == pytest.approx(0.0)

    def test_sine_vs_cosine_positive(self):
        a = _sine(20)
        t = np.linspace(0, 2 * math.pi, 20)
        b = np.cos(t).reshape(-1, 1)
        assert dtw_distance(a, b) > 0.0

    def test_window_zero_becomes_length_diff(self):
        a = _ramp(5)
        b = _ramp(10)
        # window=0 → effective window = abs(5-10)=5; shouldn't raise
        dist = dtw_distance(a, b, window=0)
        assert dist >= 0.0

    def test_3d_curves(self):
        a = np.random.default_rng(0).random((8, 3))
        b = np.random.default_rng(1).random((8, 3))
        dist = dtw_distance(a, b)
        assert dist >= 0.0


# ─── TestDtwDistanceMirror ────────────────────────────────────────────────────

class TestDtwDistanceMirror:
    def test_identical_zero(self):
        a = _ramp(10)
        assert dtw_distance_mirror(a, a) == pytest.approx(0.0)

    def test_mirrored_low_dist(self):
        a = _ramp(10)
        b = a[::-1].copy()
        dist = dtw_distance_mirror(a, b)
        # Одна из двух (прямая или зеркальная) должна дать 0
        assert dist == pytest.approx(0.0)

    def test_returns_float(self):
        a = _ramp(5)
        b = _ramp(5)
        assert isinstance(dtw_distance_mirror(a, b), float)

    def test_non_negative(self):
        a = _ramp(8)
        b = _ramp(12) + 1.0
        assert dtw_distance_mirror(a, b) >= 0.0

    def test_leq_direct(self):
        # mirror ≤ direct (т.к. min(direct, mirrored))
        a = _ramp(10)
        b = _ramp(10)[::-1] + 0.5
        d_mirror = dtw_distance_mirror(a, b)
        d_direct = dtw_distance(a, b)
        assert d_mirror <= d_direct + 1e-9

    def test_empty_a_inf(self):
        a = np.empty((0, 2))
        b = _ramp(5)
        assert math.isinf(dtw_distance_mirror(a, b))

    def test_empty_b_inf(self):
        a = _ramp(5)
        b = np.empty((0, 2))
        assert math.isinf(dtw_distance_mirror(a, b))

    def test_different_lengths(self):
        a = _ramp(7)
        b = _ramp(13)
        dist = dtw_distance_mirror(a, b)
        assert dist >= 0.0
        assert not math.isinf(dist)

    def test_window_parameter(self):
        a = _ramp(20)
        b = _ramp(20)[::-1]
        d1 = dtw_distance_mirror(a, b, window=5)
        d2 = dtw_distance_mirror(a, b, window=50)
        # Оба должны быть >= 0 и не бесконечны
        assert d1 >= 0.0
        assert d2 >= 0.0

    def test_1d_mirrored(self):
        a = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        b = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])
        # Зеркальная копия → dist = 0
        assert dtw_distance_mirror(a, b) == pytest.approx(0.0)

    def test_sine_mirrored(self):
        a = _sine(30)
        b = a[::-1].copy()
        dist = dtw_distance_mirror(a, b)
        assert dist == pytest.approx(0.0)

    def test_random_curves(self):
        rng = np.random.default_rng(7)
        a = rng.random((12, 2))
        b = rng.random((12, 2))
        dist = dtw_distance_mirror(a, b)
        assert dist >= 0.0
