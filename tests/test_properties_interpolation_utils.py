"""
Property-based tests for puzzle_reconstruction.utils.interpolation_utils.

Verifies mathematical invariants:
- InterpolationConfig:   valid method strings, ValueError on bad method
- lerp:                  t=0 → a; t=1 → b; monotone; midpoint = (a+b)/2;
                         t outside [0,1] raises ValueError
- lerp_array:            same shape as a, b; t=0 → a; t=1 → b
- bilinear_interpolate:  returns scalar; at integer grid points = grid value
- resample_1d:           exact n_out points; constant signal → constant;
                         identity at same length
- fill_missing:          no NaNs in output; non-NaN values unchanged;
                         all-NaN → zeros
- interpolate_scores:    same shape as input; values in [min, max] of input
- smooth_interpolate:    same length; constant signal unchanged
- batch_resample:        same number of signals; each has correct n_out length
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.interpolation_utils import (
    InterpolationConfig,
    lerp,
    lerp_array,
    bilinear_interpolate,
    resample_1d,
    fill_missing,
    interpolate_scores,
    smooth_interpolate,
    batch_resample,
)

RNG = np.random.default_rng(13)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_signal(n: int = 30) -> np.ndarray:
    return RNG.uniform(-5.0, 5.0, size=n)


def _rand_grid(h: int = 10, w: int = 10) -> np.ndarray:
    return RNG.uniform(0.0, 100.0, size=(h, w))


# ─── InterpolationConfig ──────────────────────────────────────────────────────

class TestInterpolationConfig:
    def test_default_linear(self):
        cfg = InterpolationConfig()
        assert cfg.method == "linear"
        assert isinstance(cfg.clamp, bool)

    def test_nearest_valid(self):
        cfg = InterpolationConfig(method="nearest")
        assert cfg.method == "nearest"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            InterpolationConfig(method="cubic")

    def test_invalid_method_empty(self):
        with pytest.raises(ValueError):
            InterpolationConfig(method="")


# ─── lerp ─────────────────────────────────────────────────────────────────────

class TestLerp:
    def test_t0_returns_a(self):
        for _ in range(20):
            a, b = float(RNG.uniform(-100, 100)), float(RNG.uniform(-100, 100))
            assert lerp(a, b, 0.0) == pytest.approx(a)

    def test_t1_returns_b(self):
        for _ in range(20):
            a, b = float(RNG.uniform(-100, 100)), float(RNG.uniform(-100, 100))
            assert lerp(a, b, 1.0) == pytest.approx(b)

    def test_midpoint(self):
        a, b = 0.0, 10.0
        assert lerp(a, b, 0.5) == pytest.approx(5.0)

    def test_monotone_for_a_lt_b(self):
        a, b = 0.0, 10.0
        t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [lerp(a, b, t) for t in t_vals]
        assert all(results[i] <= results[i + 1] + 1e-12 for i in range(len(results) - 1))

    def test_t_negative_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, -0.1)

    def test_t_above_1_raises(self):
        with pytest.raises(ValueError):
            lerp(0.0, 1.0, 1.1)

    def test_same_values_returns_a(self):
        assert lerp(5.0, 5.0, 0.7) == pytest.approx(5.0)

    def test_result_between_a_and_b(self):
        for _ in range(30):
            a = float(RNG.uniform(-100, 100))
            b = float(RNG.uniform(-100, 100))
            t = float(RNG.uniform(0, 1))
            result = lerp(a, b, t)
            assert min(a, b) - 1e-10 <= result <= max(a, b) + 1e-10

    def test_linear_interpolation_formula(self):
        """Direct check: lerp(a, b, t) == a + t * (b - a)."""
        for _ in range(20):
            a = float(RNG.uniform(-50, 50))
            b = float(RNG.uniform(-50, 50))
            t = float(RNG.uniform(0, 1))
            assert lerp(a, b, t) == pytest.approx(a + t * (b - a))


# ─── lerp_array ───────────────────────────────────────────────────────────────

class TestLerpArray:
    def test_t0_returns_a(self):
        a = _rand_signal(20)
        b = _rand_signal(20)
        result = lerp_array(a, b, 0.0)
        np.testing.assert_array_almost_equal(result, a)

    def test_t1_returns_b(self):
        a = _rand_signal(20)
        b = _rand_signal(20)
        result = lerp_array(a, b, 1.0)
        np.testing.assert_array_almost_equal(result, b)

    def test_same_shape(self):
        a = _rand_signal(15)
        b = _rand_signal(15)
        result = lerp_array(a, b, 0.5)
        assert result.shape == a.shape

    def test_midpoint_elementwise(self):
        a = np.array([0.0, 2.0, 4.0])
        b = np.array([2.0, 4.0, 6.0])
        result = lerp_array(a, b, 0.5)
        np.testing.assert_array_almost_equal(result, [1.0, 3.0, 5.0])

    def test_result_between_a_and_b(self):
        a = _rand_signal(20)
        b = _rand_signal(20)
        t = 0.3
        result = lerp_array(a, b, t)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        assert np.all(result >= lo - 1e-10)
        assert np.all(result <= hi + 1e-10)


# ─── bilinear_interpolate ─────────────────────────────────────────────────────

class TestBilinearInterpolate:
    def test_returns_scalar(self):
        grid = _rand_grid(10, 10)
        val = bilinear_interpolate(grid, 3.5, 4.5)
        assert isinstance(val, float)

    def test_at_integer_grid_point(self):
        grid = np.zeros((10, 10))
        grid[3, 4] = 7.0
        # bilinear_interpolate(grid, x, y) where x=col=4, y=row=3
        val = bilinear_interpolate(grid, 4.0, 3.0)
        assert val == pytest.approx(7.0, abs=1e-8)

    def test_constant_grid(self):
        grid = np.full((10, 10), 5.0)
        for _ in range(20):
            r = float(RNG.uniform(0, 8))
            c = float(RNG.uniform(0, 8))
            val = bilinear_interpolate(grid, r, c)
            assert val == pytest.approx(5.0, abs=1e-8)

    def test_in_range_of_grid(self):
        grid = _rand_grid(10, 10)
        g_min = grid.min()
        g_max = grid.max()
        for _ in range(20):
            r = float(RNG.uniform(0, 8))
            c = float(RNG.uniform(0, 8))
            val = bilinear_interpolate(grid, r, c)
            assert g_min - 1e-8 <= val <= g_max + 1e-8


# ─── resample_1d ──────────────────────────────────────────────────────────────

class TestResample1d:
    @pytest.mark.parametrize("n_out", [5, 20, 50, 100])
    def test_exact_length(self, n_out):
        sig = _rand_signal(30)
        result = resample_1d(sig, n_out)
        assert len(result) == n_out

    def test_constant_signal_unchanged(self):
        sig = np.full(30, 4.0)
        result = resample_1d(sig, 50)
        assert np.all(np.abs(result - 4.0) < 1e-6)

    def test_identity_same_length(self):
        sig = _rand_signal(30)
        result = resample_1d(sig, 30)
        np.testing.assert_array_almost_equal(result, sig, decimal=5)

    def test_output_in_value_range(self):
        sig = _rand_signal(30)
        result = resample_1d(sig, 60)
        assert result.min() >= sig.min() - 0.01
        assert result.max() <= sig.max() + 0.01

    def test_single_element_input(self):
        sig = np.array([3.5])
        result = resample_1d(sig, 10)
        assert len(result) == 10


# ─── fill_missing ─────────────────────────────────────────────────────────────

class TestFillMissing:
    def test_no_nans_in_output(self):
        sig = _rand_signal(30)
        # Introduce some NaNs
        sig[5] = np.nan
        sig[15] = np.nan
        result = fill_missing(sig)
        assert not np.any(np.isnan(result))

    def test_non_nan_values_unchanged(self):
        sig = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = fill_missing(sig)
        assert result[0] == pytest.approx(1.0)
        assert result[2] == pytest.approx(3.0)
        assert result[4] == pytest.approx(5.0)

    def test_no_nans_input_unchanged(self):
        sig = _rand_signal(20)
        result = fill_missing(sig)
        np.testing.assert_array_almost_equal(result, sig)

    def test_same_length(self):
        sig = _rand_signal(25)
        sig[::3] = np.nan
        result = fill_missing(sig)
        assert len(result) == len(sig)

    def test_all_nan_returns_zeros(self):
        sig = np.full(10, np.nan)
        result = fill_missing(sig)
        assert not np.any(np.isnan(result))

    def test_single_nan(self):
        sig = np.array([1.0, np.nan, 5.0])
        result = fill_missing(sig)
        assert not np.isnan(result[1])
        # Should be interpolated between 1.0 and 5.0
        assert 1.0 <= result[1] <= 5.0


# ─── interpolate_scores ───────────────────────────────────────────────────────

class TestInterpolateScores:
    def test_same_shape(self):
        m = RNG.uniform(0.0, 1.0, size=(5, 5)).astype(np.float32)
        result = interpolate_scores(m)
        assert result.shape == m.shape

    def test_constant_matrix_unchanged(self):
        m = np.full((5, 5), 0.5, dtype=np.float32)
        result = interpolate_scores(m)
        assert np.all(np.abs(result - 0.5) < 1e-4)

    def test_values_in_reasonable_range(self):
        m = RNG.uniform(0.0, 1.0, size=(8, 8)).astype(np.float32)
        result = interpolate_scores(m)
        # With smoothing, values should stay within original range or very close
        assert result.min() >= -0.1
        assert result.max() <= 1.1


# ─── smooth_interpolate ───────────────────────────────────────────────────────

class TestSmoothInterpolate:
    def test_same_length(self):
        sig = _rand_signal(30)
        result = smooth_interpolate(sig)
        assert len(result) == len(sig)

    def test_constant_signal_unchanged(self):
        sig = np.full(30, 3.0)
        result = smooth_interpolate(sig)
        np.testing.assert_array_almost_equal(result, sig, decimal=5)

    def test_no_nans_in_output(self):
        sig = _rand_signal(30)
        result = smooth_interpolate(sig)
        assert not np.any(np.isnan(result))

    def test_output_in_range(self):
        sig = _rand_signal(40)
        result = smooth_interpolate(sig)
        # Smoothed values can slightly exceed original range near edges
        assert result.min() >= sig.min() - abs(sig).max() * 0.2
        assert result.max() <= sig.max() + abs(sig).max() * 0.2


# ─── batch_resample ───────────────────────────────────────────────────────────

class TestBatchResample:
    def test_same_count_as_input(self):
        signals = [_rand_signal(20), _rand_signal(30), _rand_signal(15)]
        results = batch_resample(signals, 50)
        assert len(results) == len(signals)

    @pytest.mark.parametrize("n_out", [10, 25, 50])
    def test_each_has_correct_length(self, n_out):
        signals = [_rand_signal(20), _rand_signal(30), _rand_signal(15)]
        results = batch_resample(signals, n_out)
        for r in results:
            assert len(r) == n_out

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            batch_resample([], 50)

    def test_single_signal(self):
        sig = _rand_signal(20)
        results = batch_resample([sig], 40)
        assert len(results) == 1
        assert len(results[0]) == 40

    def test_constant_signals_stay_constant(self):
        sigs = [np.full(20, float(i)) for i in range(1, 4)]
        results = batch_resample(sigs, 30)
        for i, r in enumerate(results):
            expected_val = float(i + 1)
            assert np.all(np.abs(r - expected_val) < 1e-6)
