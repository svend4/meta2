"""Extra tests for puzzle_reconstruction/utils/smoothing_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.smoothing_utils import (
    SmoothingParams,
    moving_average,
    gaussian_smooth,
    median_smooth,
    exponential_smooth,
    savgol_smooth,
    smooth_contour,
    apply_smoothing,
    batch_smooth,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sig(n=20, val=0.5) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


def _ramp(n=20) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _contour(n=10) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)]).astype(np.float32)


# ─── SmoothingParams ──────────────────────────────────────────────────────────

class TestSmoothingParamsExtra:
    def test_default_method(self):
        assert SmoothingParams().method == "moving_average"

    def test_default_window_size(self):
        assert SmoothingParams().window_size == 5

    def test_default_sigma(self):
        assert SmoothingParams().sigma == pytest.approx(1.0)

    def test_default_alpha(self):
        assert SmoothingParams().alpha == pytest.approx(0.3)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(method="cubic")

    def test_window_lt_3_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=4)

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(sigma=-1.0)

    def test_polyorder_zero_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(polyorder=0)

    def test_polyorder_ge_window_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=5, polyorder=5)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(alpha=0.0)

    def test_alpha_gt_one_raises(self):
        with pytest.raises(ValueError):
            SmoothingParams(alpha=1.1)

    def test_valid_all_methods(self):
        for m in ("moving_average", "gaussian", "median", "savgol", "exponential"):
            p = SmoothingParams(method=m)
            assert p.method == m


# ─── moving_average ───────────────────────────────────────────────────────────

class TestMovingAverageExtra:
    def test_returns_ndarray(self):
        assert isinstance(moving_average(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert moving_average(_ramp()).dtype == np.float64

    def test_same_length(self):
        assert len(moving_average(_ramp(12))) == 12

    def test_constant_unchanged(self):
        s = _sig(10, 3.0)
        np.testing.assert_allclose(moving_average(s), 3.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            moving_average(np.zeros((3, 4)))

    def test_window_lt_3_raises(self):
        with pytest.raises(ValueError):
            moving_average(_ramp(), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            moving_average(_ramp(), window_size=4)

    def test_empty_returns_empty(self):
        out = moving_average(np.array([]))
        assert len(out) == 0

    def test_larger_window_smoother(self):
        # Spike smoothed more by larger window
        s = np.zeros(20)
        s[10] = 10.0
        out3 = moving_average(s, window_size=3)
        out7 = moving_average(s, window_size=7)
        assert out7.max() <= out3.max()


# ─── gaussian_smooth ──────────────────────────────────────────────────────────

class TestGaussianSmoothExtra:
    def test_returns_ndarray(self):
        assert isinstance(gaussian_smooth(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert gaussian_smooth(_ramp()).dtype == np.float64

    def test_same_length(self):
        assert len(gaussian_smooth(_ramp(15))) == 15

    def test_constant_unchanged(self):
        s = _sig(10, 4.0)
        np.testing.assert_allclose(gaussian_smooth(s), 4.0, atol=1e-10)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            gaussian_smooth(np.zeros((3, 4)))

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            gaussian_smooth(_ramp(), sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            gaussian_smooth(_ramp(), sigma=-0.5)

    def test_empty_returns_empty(self):
        out = gaussian_smooth(np.array([]))
        assert len(out) == 0


# ─── median_smooth ────────────────────────────────────────────────────────────

class TestMedianSmoothExtra:
    def test_returns_ndarray(self):
        assert isinstance(median_smooth(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert median_smooth(_ramp()).dtype == np.float64

    def test_same_length(self):
        assert len(median_smooth(_ramp(12))) == 12

    def test_constant_unchanged(self):
        s = _sig(10, 2.0)
        np.testing.assert_allclose(median_smooth(s), 2.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            median_smooth(np.zeros((3, 4)))

    def test_window_lt_3_raises(self):
        with pytest.raises(ValueError):
            median_smooth(_ramp(), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            median_smooth(_ramp(), window_size=4)

    def test_empty_returns_empty(self):
        out = median_smooth(np.array([]))
        assert len(out) == 0

    def test_spike_suppressed(self):
        s = np.zeros(11)
        s[5] = 100.0
        out = median_smooth(s, window_size=3)
        assert out[5] < 100.0


# ─── exponential_smooth ───────────────────────────────────────────────────────

class TestExponentialSmoothExtra:
    def test_returns_ndarray(self):
        assert isinstance(exponential_smooth(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert exponential_smooth(_ramp()).dtype == np.float64

    def test_same_length(self):
        assert len(exponential_smooth(_ramp(10))) == 10

    def test_alpha_one_identity(self):
        s = _ramp(8)
        np.testing.assert_allclose(exponential_smooth(s, alpha=1.0), s)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(_ramp(), alpha=0.0)

    def test_alpha_gt_one_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(_ramp(), alpha=1.1)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(np.zeros((3, 4)))

    def test_empty_returns_empty(self):
        out = exponential_smooth(np.array([]))
        assert len(out) == 0

    def test_first_element_unchanged(self):
        s = np.array([5.0, 1.0, 1.0, 1.0])
        out = exponential_smooth(s, alpha=0.5)
        assert out[0] == pytest.approx(5.0)


# ─── savgol_smooth ────────────────────────────────────────────────────────────

class TestSavgolSmoothExtra:
    def test_returns_ndarray(self):
        assert isinstance(savgol_smooth(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert savgol_smooth(_ramp()).dtype == np.float64

    def test_same_length(self):
        assert len(savgol_smooth(_ramp(15))) == 15

    def test_linear_signal_inner_preserved(self):
        s = _ramp(20)
        out = savgol_smooth(s)
        # Inner points of linear signal preserved by savgol; edges may differ
        np.testing.assert_allclose(out[3:-3], s[3:-3], atol=1e-10)

    def test_window_lt_3_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(_ramp(), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(_ramp(), window_size=4)

    def test_polyorder_zero_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(_ramp(), polyorder=0)

    def test_polyorder_ge_window_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(_ramp(), window_size=5, polyorder=5)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(np.zeros((3, 4)))

    def test_empty_returns_empty(self):
        out = savgol_smooth(np.array([]))
        assert len(out) == 0


# ─── smooth_contour ───────────────────────────────────────────────────────────

class TestSmoothContourExtra:
    def test_returns_ndarray(self):
        assert isinstance(smooth_contour(_contour()), np.ndarray)

    def test_dtype_float32(self):
        assert smooth_contour(_contour()).dtype == np.float32

    def test_shape_preserved(self):
        c = _contour(12)
        assert smooth_contour(c).shape == (12, 2)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros((5, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros(10))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.zeros((0, 2)))

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(_contour(), sigma=0.0)

    def test_smooth_reduces_variation(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.5, (20, 2)).astype(np.float32)
        c = _contour(20) + noise
        out = smooth_contour(c, sigma=2.0)
        assert np.std(out) <= np.std(c) + 0.1


# ─── apply_smoothing ──────────────────────────────────────────────────────────

class TestApplySmoothingExtra:
    def test_returns_ndarray(self):
        p = SmoothingParams(method="gaussian")
        assert isinstance(apply_smoothing(_ramp(), p), np.ndarray)

    def test_same_length(self):
        p = SmoothingParams(method="moving_average")
        assert len(apply_smoothing(_ramp(12), p)) == 12

    def test_gaussian_dispatch(self):
        p = SmoothingParams(method="gaussian")
        out = apply_smoothing(_sig(8, 2.0), p)
        np.testing.assert_allclose(out, 2.0, atol=1e-10)

    def test_median_dispatch(self):
        p = SmoothingParams(method="median")
        out = apply_smoothing(_sig(8, 2.0), p)
        np.testing.assert_allclose(out, 2.0)

    def test_savgol_dispatch(self):
        p = SmoothingParams(method="savgol")
        out = apply_smoothing(_ramp(20), p)
        assert len(out) == 20

    def test_exponential_dispatch(self):
        p = SmoothingParams(method="exponential")
        out = apply_smoothing(_ramp(10), p)
        assert len(out) == 10


# ─── batch_smooth ─────────────────────────────────────────────────────────────

class TestBatchSmoothExtra:
    def test_returns_list(self):
        p = SmoothingParams()
        assert isinstance(batch_smooth([_ramp()], p), list)

    def test_length_matches(self):
        p = SmoothingParams()
        result = batch_smooth([_ramp(10), _ramp(15)], p)
        assert len(result) == 2

    def test_each_element_ndarray(self):
        p = SmoothingParams()
        for out in batch_smooth([_ramp(10), _ramp(8)], p):
            assert isinstance(out, np.ndarray)

    def test_lengths_preserved(self):
        p = SmoothingParams()
        result = batch_smooth([_ramp(10), _ramp(15)], p)
        assert len(result[0]) == 10
        assert len(result[1]) == 15
