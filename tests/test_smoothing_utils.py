"""Тесты для puzzle_reconstruction.utils.smoothing_utils."""
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


# ─── TestSmoothingParams ─────────────────────────────────────────────────────

class TestSmoothingParams:
    def test_default_values(self):
        p = SmoothingParams()
        assert p.method == "moving_average"
        assert p.window_size == 5
        assert p.sigma == 1.0
        assert p.polyorder == 2
        assert p.alpha == 0.3

    def test_valid_methods(self):
        for m in ("moving_average", "gaussian", "median", "savgol", "exponential"):
            p = SmoothingParams(method=m)
            assert p.method == m

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            SmoothingParams(method="unknown")

    def test_window_size_too_small(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=2)

    def test_window_size_even(self):
        with pytest.raises(ValueError):
            SmoothingParams(window_size=4)

    def test_sigma_nonpositive(self):
        with pytest.raises(ValueError):
            SmoothingParams(sigma=0.0)

    def test_polyorder_too_small(self):
        with pytest.raises(ValueError):
            SmoothingParams(polyorder=0)

    def test_polyorder_ge_window_size(self):
        with pytest.raises(ValueError):
            SmoothingParams(polyorder=5, window_size=5)

    def test_alpha_zero(self):
        with pytest.raises(ValueError):
            SmoothingParams(alpha=0.0)

    def test_alpha_gt_one(self):
        with pytest.raises(ValueError):
            SmoothingParams(alpha=1.1)

    def test_alpha_one_valid(self):
        p = SmoothingParams(alpha=1.0)
        assert p.alpha == 1.0


# ─── TestMovingAverage ────────────────────────────────────────────────────────

class TestMovingAverage:
    def test_returns_float64(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = moving_average(sig, window_size=3)
        assert out.dtype == np.float64

    def test_same_length(self):
        sig = np.arange(10, dtype=float)
        out = moving_average(sig, window_size=3)
        assert len(out) == len(sig)

    def test_constant_signal(self):
        sig = np.full(10, 5.0)
        out = moving_average(sig, window_size=5)
        np.testing.assert_allclose(out, 5.0)

    def test_smooths_values(self):
        sig = np.array([0.0, 10.0, 0.0, 10.0, 0.0], dtype=float)
        out = moving_average(sig, window_size=3)
        # Middle values should be between extremes
        assert out[2] < 10.0

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            moving_average(np.ones((3, 3)), window_size=3)

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            moving_average(np.ones(5), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            moving_average(np.ones(5), window_size=4)

    def test_empty_signal(self):
        out = moving_average(np.array([]), window_size=3)
        assert len(out) == 0


# ─── TestGaussianSmooth ───────────────────────────────────────────────────────

class TestGaussianSmooth:
    def test_returns_float64(self):
        sig = np.ones(10)
        out = gaussian_smooth(sig, sigma=1.0)
        assert out.dtype == np.float64

    def test_same_length(self):
        sig = np.arange(20, dtype=float)
        out = gaussian_smooth(sig, sigma=2.0)
        assert len(out) == len(sig)

    def test_constant_signal(self):
        sig = np.full(15, 3.0)
        out = gaussian_smooth(sig, sigma=1.5)
        np.testing.assert_allclose(out, 3.0, atol=1e-10)

    def test_sigma_nonpositive_raises(self):
        with pytest.raises(ValueError):
            gaussian_smooth(np.ones(5), sigma=0.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            gaussian_smooth(np.ones((3, 4)), sigma=1.0)

    def test_empty_signal(self):
        out = gaussian_smooth(np.array([]), sigma=1.0)
        assert len(out) == 0

    def test_reduces_variance(self):
        rng = np.random.default_rng(0)
        noisy = rng.normal(0, 5, size=100)
        smoothed = gaussian_smooth(noisy, sigma=3.0)
        assert smoothed.std() < noisy.std()


# ─── TestMedianSmooth ─────────────────────────────────────────────────────────

class TestMedianSmooth:
    def test_returns_float64(self):
        sig = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
        out = median_smooth(sig, window_size=3)
        assert out.dtype == np.float64

    def test_same_length(self):
        sig = np.arange(8, dtype=float)
        out = median_smooth(sig, window_size=3)
        assert len(out) == len(sig)

    def test_removes_spike(self):
        sig = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
        out = median_smooth(sig, window_size=3)
        # Spike at index 2 should be reduced
        assert out[2] < 100.0

    def test_constant_signal(self):
        sig = np.full(10, 7.0)
        out = median_smooth(sig, window_size=5)
        np.testing.assert_allclose(out, 7.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            median_smooth(np.ones((2, 5)), window_size=3)

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            median_smooth(np.ones(5), window_size=1)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            median_smooth(np.ones(5), window_size=4)

    def test_empty_signal(self):
        out = median_smooth(np.array([]), window_size=3)
        assert len(out) == 0


# ─── TestExponentialSmooth ────────────────────────────────────────────────────

class TestExponentialSmooth:
    def test_returns_float64(self):
        sig = np.array([1.0, 2.0, 3.0])
        out = exponential_smooth(sig, alpha=0.5)
        assert out.dtype == np.float64

    def test_same_length(self):
        sig = np.arange(10, dtype=float)
        out = exponential_smooth(sig, alpha=0.3)
        assert len(out) == len(sig)

    def test_first_element_unchanged(self):
        sig = np.array([5.0, 1.0, 1.0, 1.0])
        out = exponential_smooth(sig, alpha=0.5)
        assert out[0] == pytest.approx(5.0)

    def test_alpha_one_passthrough(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0])
        out = exponential_smooth(sig, alpha=1.0)
        np.testing.assert_allclose(out, sig)

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(np.ones(5), alpha=0.0)

    def test_alpha_gt_one_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(np.ones(5), alpha=1.5)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(np.ones((3, 3)), alpha=0.3)

    def test_empty_signal(self):
        out = exponential_smooth(np.array([]), alpha=0.5)
        assert len(out) == 0


# ─── TestSavgolSmooth ─────────────────────────────────────────────────────────

class TestSavgolSmooth:
    def test_returns_float64(self):
        sig = np.arange(10, dtype=float)
        out = savgol_smooth(sig, window_size=5, polyorder=2)
        assert out.dtype == np.float64

    def test_same_length(self):
        sig = np.arange(15, dtype=float)
        out = savgol_smooth(sig, window_size=5, polyorder=2)
        assert len(out) == len(sig)

    def test_linear_signal_preserved(self):
        sig = np.arange(10, dtype=float)
        out = savgol_smooth(sig, window_size=5, polyorder=1)
        np.testing.assert_allclose(out, sig, atol=1e-8)

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(np.ones(10), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(np.ones(10), window_size=4)

    def test_polyorder_zero_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(np.ones(10), window_size=5, polyorder=0)

    def test_polyorder_ge_window_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(np.ones(10), window_size=5, polyorder=5)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            savgol_smooth(np.ones((3, 5)), window_size=3, polyorder=1)

    def test_empty_signal(self):
        out = savgol_smooth(np.array([]), window_size=3, polyorder=1)
        assert len(out) == 0


# ─── TestSmoothContour ────────────────────────────────────────────────────────

class TestSmoothContour:
    def test_returns_float32(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [2, 1]], dtype=np.float32)
        out = smooth_contour(pts, sigma=0.5)
        assert out.dtype == np.float32

    def test_same_shape(self):
        pts = np.random.rand(20, 2).astype(np.float32)
        out = smooth_contour(pts, sigma=1.0)
        assert out.shape == pts.shape

    def test_constant_contour(self):
        pts = np.full((10, 2), 3.0, dtype=np.float32)
        out = smooth_contour(pts, sigma=1.0)
        np.testing.assert_allclose(out, 3.0, atol=1e-5)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.ones((5, 3)), sigma=1.0)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.ones(10), sigma=1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            smooth_contour(np.empty((0, 2), dtype=np.float32), sigma=1.0)

    def test_sigma_nonpositive_raises(self):
        pts = np.ones((5, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            smooth_contour(pts, sigma=0.0)


# ─── TestApplySmoothing ───────────────────────────────────────────────────────

class TestApplySmoothing:
    def _signal(self):
        return np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0])

    def test_moving_average(self):
        params = SmoothingParams(method="moving_average", window_size=3)
        out = apply_smoothing(self._signal(), params)
        assert len(out) == len(self._signal())
        assert out.dtype == np.float64

    def test_gaussian(self):
        params = SmoothingParams(method="gaussian", sigma=1.0)
        out = apply_smoothing(self._signal(), params)
        assert len(out) == len(self._signal())

    def test_median(self):
        params = SmoothingParams(method="median", window_size=3)
        out = apply_smoothing(self._signal(), params)
        assert len(out) == len(self._signal())

    def test_savgol(self):
        params = SmoothingParams(method="savgol", window_size=5, polyorder=2)
        out = apply_smoothing(self._signal(), params)
        assert len(out) == len(self._signal())

    def test_exponential(self):
        params = SmoothingParams(method="exponential", alpha=0.4)
        out = apply_smoothing(self._signal(), params)
        assert len(out) == len(self._signal())


# ─── TestBatchSmooth ──────────────────────────────────────────────────────────

class TestBatchSmooth:
    def test_returns_list(self):
        signals = [np.arange(10, dtype=float), np.ones(8)]
        params = SmoothingParams(method="moving_average", window_size=3)
        out = batch_smooth(signals, params)
        assert isinstance(out, list)

    def test_correct_length(self):
        signals = [np.arange(i + 5, dtype=float) for i in range(4)]
        params = SmoothingParams(method="gaussian", sigma=1.0)
        out = batch_smooth(signals, params)
        assert len(out) == len(signals)

    def test_empty_list(self):
        params = SmoothingParams()
        out = batch_smooth([], params)
        assert out == []

    def test_each_output_length(self):
        signals = [np.ones(n) for n in [5, 7, 9]]
        params = SmoothingParams(method="median", window_size=3)
        out = batch_smooth(signals, params)
        for orig, smoothed in zip(signals, out):
            assert len(smoothed) == len(orig)
