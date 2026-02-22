"""Additional tests for puzzle_reconstruction/matching/dtw.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.matching.dtw import (
    dtw_distance,
    dtw_distance_mirror,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ramp(n: int, d: int = 2) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=float)] * d)


def _const(n: int, d: int = 2, val: float = 0.0) -> np.ndarray:
    return np.full((n, d), val)


def _sine(n: int, freq: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi * freq, n)
    return np.column_stack([t, amp * np.sin(t)])


def _rand(n: int, d: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, d))


# ─── TestDTWDistanceExtra ─────────────────────────────────────────────────────

class TestDTWDistanceExtra:
    def test_float32_input_finite(self):
        a = _ramp(16).astype(np.float32)
        b = _ramp(16).astype(np.float32)
        assert dtw_distance(a, b) == pytest.approx(0.0)

    def test_float32_vs_float64(self):
        a32 = _ramp(10).astype(np.float32)
        a64 = _ramp(10).astype(np.float64)
        d32 = dtw_distance(a32, a32)
        d64 = dtw_distance(a64, a64)
        assert abs(d32 - d64) < 1e-6

    def test_identical_const_zero(self):
        a = _const(20, val=5.0)
        assert dtw_distance(a, a) == pytest.approx(0.0)

    def test_same_const_different_val_positive(self):
        a = _const(10, val=0.0)
        b = _const(10, val=7.0)
        assert dtw_distance(a, b) > 0.0

    def test_1d_float32(self):
        a = np.arange(10, dtype=np.float32).reshape(-1, 1)
        b = np.arange(10, dtype=np.float32).reshape(-1, 1)
        assert dtw_distance(a, b) == pytest.approx(0.0)

    def test_5d_curves(self):
        a = _rand(8, d=5, seed=0)
        b = _rand(8, d=5, seed=1)
        d = dtw_distance(a, b)
        assert d >= 0.0
        assert np.isfinite(d)

    def test_large_window_equals_small_for_identical(self):
        """Identical curves: any window gives distance=0."""
        a = _ramp(20)
        d_small = dtw_distance(a, a, window=1)
        d_large = dtw_distance(a, a, window=100)
        assert d_small == pytest.approx(0.0)
        assert d_large == pytest.approx(0.0)

    def test_n1_curve(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[4.0, 6.0]])
        d = dtw_distance(a, b)
        # single step, distance = 5.0, norm by (1+1)=2 → 2.5
        assert d == pytest.approx(5.0 / 2.0, abs=1e-9)

    def test_ramp_vs_shifted_ramp(self):
        a = _ramp(15)
        b = _ramp(15) + 3.0
        d = dtw_distance(a, b)
        assert d > 0.0
        assert np.isfinite(d)

    def test_different_lengths_n20_n5(self):
        a = _ramp(20)
        b = _ramp(5)
        d = dtw_distance(a, b)
        assert d >= 0.0
        assert not math.isinf(d)

    def test_reversed_ramp_nonneg(self):
        a = _ramp(10)
        b = a[::-1].copy()
        d = dtw_distance(a, b)
        assert d >= 0.0

    def test_finite_for_sine_vs_noise(self):
        a = _sine(32)
        b = _rand(32, d=2, seed=5)
        d = dtw_distance(a, b)
        assert np.isfinite(d)

    def test_window_0_handled(self):
        """window=0 → effective window = |n-m|, should not raise."""
        a = _ramp(8)
        b = _ramp(12)
        d = dtw_distance(a, b, window=0)
        assert d >= 0.0


# ─── TestDTWDistanceMirrorExtra ───────────────────────────────────────────────

class TestDTWDistanceMirrorExtra:
    def test_float32_input(self):
        a = _ramp(10).astype(np.float32)
        b = _ramp(10).astype(np.float32)
        d = dtw_distance_mirror(a, b)
        assert d == pytest.approx(0.0)

    def test_constant_same_zero(self):
        a = _const(15, val=3.0)
        d = dtw_distance_mirror(a, a)
        assert d == pytest.approx(0.0)

    def test_reversed_sine_zero(self):
        a = _sine(24)
        d = dtw_distance_mirror(a, a[::-1])
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_both_empty_inf(self):
        a = np.empty((0, 2))
        assert math.isinf(dtw_distance_mirror(a, a))

    def test_very_large_window(self):
        a = _ramp(10)
        b = _ramp(10)[::-1]
        d = dtw_distance_mirror(a, b, window=1000)
        assert d >= 0.0

    def test_5d_curves_nonneg(self):
        a = _rand(8, d=5, seed=0)
        b = _rand(8, d=5, seed=2)
        d = dtw_distance_mirror(a, b)
        assert d >= 0.0

    def test_different_lengths_finite(self):
        a = _ramp(6)
        b = _ramp(14) * 0.5
        d = dtw_distance_mirror(a, b)
        assert np.isfinite(d)

    def test_nonneg_for_random(self):
        for seed in range(5):
            a = _rand(12, seed=seed)
            b = _rand(12, seed=seed + 10)
            assert dtw_distance_mirror(a, b) >= 0.0

    def test_window_1_no_crash(self):
        a = _ramp(10)
        b = _ramp(10)[::-1]
        d = dtw_distance_mirror(a, b, window=1)
        assert d >= 0.0

    def test_mirror_result_le_direct(self):
        a = _sine(20)
        b = a[::-1].copy() + 0.01
        d_mirror = dtw_distance_mirror(a, b)
        d_direct = dtw_distance(a, b)
        assert d_mirror <= d_direct + 1e-9
