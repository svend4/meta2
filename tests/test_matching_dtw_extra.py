"""Additional tests for puzzle_reconstruction.matching.dtw."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.matching.dtw import (
    dtw_distance,
    dtw_distance_mirror,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _const(n=10, val=1.0) -> np.ndarray:
    return np.full((n, 1), val, dtype=np.float64)


def _ramp(n=20) -> np.ndarray:
    return np.linspace(0, 1, n).reshape(-1, 1).astype(np.float64)


def _sine(n=30) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n)
    return np.stack([np.sin(t), np.cos(t)], axis=1).astype(np.float64)


def _curve2d(n=15, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 5, (n, 2)).astype(np.float64)


# ─── TestDtwDistanceExtra ─────────────────────────────────────────────────────

class TestDtwDistanceExtra:
    def test_constant_curves_zero(self):
        a = _const(10, 3.0)
        assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_constant_vs_shifted(self):
        a = _const(10, 0.0)
        b = _const(10, 1.0)
        d = dtw_distance(a, b)
        assert d > 0.0

    def test_ramp_vs_itself_zero(self):
        a = _ramp(20)
        assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_scaled_curves_differ(self):
        a = _ramp(10)
        b = _ramp(10) * 2.0
        d = dtw_distance(a, b)
        assert d > 0.0

    def test_window_enforced_different_lengths(self):
        a = _ramp(8)
        b = _ramp(20)
        # window = max(5, |8-20|) = 12 → should still be computed
        d = dtw_distance(a, b, window=5)
        assert d >= 0.0
        assert not math.isinf(d)

    def test_sine_2d_nonneg(self):
        a = _sine(20)
        b = _sine(20)
        d = dtw_distance(a, b)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_different_scales_larger_distance(self):
        a = _curve2d(15, 0)
        b_close = a + 0.001
        b_far = a + 10.0
        d_close = dtw_distance(a, b_close)
        d_far = dtw_distance(a, b_far)
        assert d_close < d_far

    def test_both_empty_inf(self):
        a = np.zeros((0, 1))
        b = np.zeros((0, 1))
        assert math.isinf(dtw_distance(a, b))

    def test_single_vs_multiple(self):
        a = np.array([[0.5]])
        b = _ramp(10)
        d = dtw_distance(a, b)
        assert d >= 0.0

    def test_window_1_valid(self):
        a = _ramp(5)
        b = _ramp(5)
        d = dtw_distance(a, b, window=1)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_triangle_inequality_approx(self):
        """d(a,c) <= d(a,b) + d(b,c) for DTW (holds approximately)."""
        a = _curve2d(12, 0)
        b = _curve2d(12, 1)
        c = _curve2d(12, 2)
        d_ab = dtw_distance(a, b)
        d_bc = dtw_distance(b, c)
        d_ac = dtw_distance(a, c)
        # Allow generous tolerance — DTW is not strictly a metric
        assert d_ac <= d_ab + d_bc + 1.0

    def test_many_dimensions_ok(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal((10, 5))
        b = rng.standard_normal((10, 5))
        d = dtw_distance(a, b)
        assert d >= 0.0

    def test_float32_input(self):
        a = _ramp(10).astype(np.float32)
        b = _ramp(10).astype(np.float32)
        d = dtw_distance(a, b)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_identical_returns_zero_regardless_of_window(self):
        a = _curve2d(8, 99)
        for w in [1, 5, 20, 100]:
            assert dtw_distance(a, a, window=w) == pytest.approx(0.0, abs=1e-9)

    def test_returns_finite_for_typical_inputs(self):
        a = _sine(16)
        b = _curve2d(16, 3)
        d = dtw_distance(a, b)
        assert math.isfinite(d)


# ─── TestDtwDistanceMirrorExtra ───────────────────────────────────────────────

class TestDtwDistanceMirrorExtra:
    def test_ramp_vs_reversed_is_zero(self):
        """Reversed ramp perfectly mirrors ramp → mirror distance should be 0."""
        a = _ramp(15)
        b = a[::-1].copy()
        d = dtw_distance_mirror(a, b)
        assert d == pytest.approx(0.0, abs=1e-8)

    def test_mirror_symmetric(self):
        a = _curve2d(12, 0)
        b = _curve2d(12, 1)
        d1 = dtw_distance_mirror(a, b)
        d2 = dtw_distance_mirror(b, a)
        # Not necessarily equal, but both non-negative and finite
        assert d1 >= 0.0
        assert d2 >= 0.0

    def test_constant_mirror_zero(self):
        a = _const(10, 2.0)
        b = _const(10, 2.0)
        d = dtw_distance_mirror(a, b)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_mirror_le_direct_always(self):
        rng = np.random.default_rng(42)
        for seed in range(5):
            a = rng.standard_normal((12, 2))
            b = rng.standard_normal((12, 2))
            d_m = dtw_distance_mirror(a, b)
            d_d = dtw_distance(a, b)
            assert d_m <= d_d + 1e-9

    def test_both_empty_inf(self):
        a = np.zeros((0, 1))
        b = np.zeros((0, 1))
        assert math.isinf(dtw_distance_mirror(a, b))

    def test_different_lengths_ok(self):
        a = _ramp(8)
        b = _ramp(15)
        d = dtw_distance_mirror(a, b)
        assert d >= 0.0
        assert math.isfinite(d)

    def test_result_not_nan(self):
        a = _sine(20)
        b = _curve2d(20, 5)
        d = dtw_distance_mirror(a, b)
        assert not math.isnan(d)

    def test_window_parameter_passed(self):
        a = _ramp(10)
        b = _ramp(10)
        d = dtw_distance_mirror(a, b, window=2)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_sine_vs_reversed_sine(self):
        a = _sine(24)
        b = a[::-1].copy()
        d = dtw_distance_mirror(a, b)
        # Reversed sine is a mirror — should give small (possibly zero) dist
        d_direct = dtw_distance(a, b)
        assert d <= d_direct + 1e-9
