"""Tests for utils/smoothing_utils.py."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.smoothing_utils import (
    SmoothingParams,
    apply_smoothing,
    batch_smooth,
    exponential_smooth,
    gaussian_smooth,
    median_smooth,
    moving_average,
    savgol_smooth,
    smooth_contour,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_signal(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_constant(n=20, val=3.0):
    return np.full(n, val)


def make_contour(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2)).astype(np.float32)


# ─── SmoothingParams ──────────────────────────────────────────────────────────

class TestSmoothingParams:
    def test_defaults(self):
        p = SmoothingParams()
        assert p.method == "moving_average"
        assert p.window_size == 5
        assert p.sigma == pytest.approx(1.0)
        assert p.polyorder == 2
        assert p.alpha == pytest.approx(0.3)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Неизвестный"):
            SmoothingParams(method="lowess")

    def test_window_size_two_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            SmoothingParams(window_size=2)

    def test_window_size_even_raises(self):
        with pytest.raises(ValueError, match="нечётным"):
            SmoothingParams(window_size=4)

    def test_window_size_three_valid(self):
        p = SmoothingParams(window_size=3)
        assert p.window_size == 3

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SmoothingParams(sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            SmoothingParams(sigma=-1.0)

    def test_polyorder_zero_raises(self):
        with pytest.raises(ValueError, match="polyorder"):
            SmoothingParams(polyorder=0)

    def test_polyorder_ge_window_raises(self):
        with pytest.raises(ValueError, match="polyorder"):
            SmoothingParams(window_size=5, polyorder=5)

    def test_polyorder_window_minus_1_valid(self):
        p = SmoothingParams(window_size=5, polyorder=4)
        assert p.polyorder == 4

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SmoothingParams(alpha=0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            SmoothingParams(alpha=1.1)

    def test_alpha_one_valid(self):
        p = SmoothingParams(alpha=1.0)
        assert p.alpha == pytest.approx(1.0)

    def test_valid_methods(self):
        for method in ("moving_average", "gaussian", "median", "savgol", "exponential"):
            p = SmoothingParams(method=method)
            assert p.method == method


# ─── moving_average ───────────────────────────────────────────────────────────

class TestMovingAverage:
    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="одномерным"):
            moving_average(np.ones((3, 4)), window_size=3)

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            moving_average(np.ones(10), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="нечётным"):
            moving_average(np.ones(10), window_size=4)

    def test_empty_returns_empty(self):
        result = moving_average(np.array([]), window_size=3)
        assert len(result) == 0

    def test_returns_same_length(self):
        s = make_signal(n=20)
        result = moving_average(s, window_size=5)
        assert len(result) == 20

    def test_returns_float64(self):
        result = moving_average(make_signal(), window_size=3)
        assert result.dtype == np.float64

    def test_constant_signal_preserved(self):
        result = moving_average(make_constant(val=5.0), window_size=5)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)

    def test_window_1_invalid(self):
        with pytest.raises(ValueError):
            moving_average(np.ones(5), window_size=1)

    def test_smoothing_reduces_noise(self):
        # A very noisy signal should have lower std after smoothing
        rng = np.random.default_rng(0)
        noisy = rng.standard_normal(100)
        smoothed = moving_average(noisy, window_size=11)
        assert smoothed.std() < noisy.std()


# ─── gaussian_smooth ──────────────────────────────────────────────────────────

class TestGaussianSmooth:
    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="одномерным"):
            gaussian_smooth(np.ones((3, 4)))

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_smooth(np.ones(10), sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_smooth(np.ones(10), sigma=-1.0)

    def test_empty_returns_empty(self):
        result = gaussian_smooth(np.array([]))
        assert len(result) == 0

    def test_returns_same_length(self):
        s = make_signal(n=30)
        result = gaussian_smooth(s, sigma=2.0)
        assert len(result) == 30

    def test_returns_float64(self):
        result = gaussian_smooth(make_signal(), sigma=1.0)
        assert result.dtype == np.float64

    def test_constant_preserved(self):
        result = gaussian_smooth(make_constant(val=7.0), sigma=1.0)
        np.testing.assert_allclose(result, 7.0, atol=1e-6)

    def test_larger_sigma_more_smoothing(self):
        noisy = make_signal(n=100, seed=5)
        smooth1 = gaussian_smooth(noisy, sigma=0.5)
        smooth5 = gaussian_smooth(noisy, sigma=5.0)
        assert smooth5.std() < smooth1.std()


# ─── median_smooth ────────────────────────────────────────────────────────────

class TestMedianSmooth:
    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="одномерным"):
            median_smooth(np.ones((3, 4)))

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            median_smooth(np.ones(10), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="нечётным"):
            median_smooth(np.ones(10), window_size=4)

    def test_empty_returns_empty(self):
        result = median_smooth(np.array([]), window_size=3)
        assert len(result) == 0

    def test_returns_same_length(self):
        result = median_smooth(make_signal(n=20), window_size=5)
        assert len(result) == 20

    def test_returns_float64(self):
        result = median_smooth(make_signal(), window_size=3)
        assert result.dtype == np.float64

    def test_constant_preserved(self):
        result = median_smooth(make_constant(val=4.0), window_size=5)
        np.testing.assert_allclose(result, 4.0, atol=1e-10)

    def test_spike_removed(self):
        s = np.ones(11)
        s[5] = 100.0  # spike
        result = median_smooth(s, window_size=5)
        # Spike should be drastically reduced
        assert result[5] < 50.0


# ─── exponential_smooth ───────────────────────────────────────────────────────

class TestExponentialSmooth:
    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="одномерным"):
            exponential_smooth(np.ones((3, 4)))

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            exponential_smooth(np.ones(10), alpha=0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            exponential_smooth(np.ones(10), alpha=1.1)

    def test_empty_returns_empty(self):
        result = exponential_smooth(np.array([]))
        assert len(result) == 0

    def test_returns_same_length(self):
        result = exponential_smooth(make_signal(n=25), alpha=0.3)
        assert len(result) == 25

    def test_returns_float64(self):
        result = exponential_smooth(make_signal(), alpha=0.5)
        assert result.dtype == np.float64

    def test_first_element_preserved(self):
        s = make_signal()
        result = exponential_smooth(s, alpha=0.5)
        assert result[0] == pytest.approx(s[0])

    def test_alpha_one_returns_original(self):
        s = make_signal()
        result = exponential_smooth(s, alpha=1.0)
        np.testing.assert_allclose(result, s, atol=1e-10)

    def test_constant_preserved(self):
        result = exponential_smooth(make_constant(val=5.0), alpha=0.3)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)


# ─── savgol_smooth ────────────────────────────────────────────────────────────

class TestSavgolSmooth:
    def test_not_1d_raises(self):
        with pytest.raises(ValueError, match="одномерным"):
            savgol_smooth(np.ones((3, 4)))

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            savgol_smooth(np.ones(10), window_size=2)

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="нечётным"):
            savgol_smooth(np.ones(10), window_size=4)

    def test_polyorder_zero_raises(self):
        with pytest.raises(ValueError, match="polyorder"):
            savgol_smooth(np.ones(10), polyorder=0)

    def test_polyorder_ge_window_raises(self):
        with pytest.raises(ValueError, match="polyorder"):
            savgol_smooth(np.ones(10), window_size=5, polyorder=5)

    def test_empty_returns_empty(self):
        result = savgol_smooth(np.array([]))
        assert len(result) == 0

    def test_returns_same_length(self):
        result = savgol_smooth(make_signal(n=20), window_size=5, polyorder=2)
        assert len(result) == 20

    def test_returns_float64(self):
        result = savgol_smooth(make_signal(), window_size=5, polyorder=2)
        assert result.dtype == np.float64

    def test_constant_preserved(self):
        result = savgol_smooth(make_constant(val=3.0), window_size=5, polyorder=2)
        np.testing.assert_allclose(result, 3.0, atol=1e-6)

    def test_linear_signal_preserved_with_poly1(self):
        s = np.linspace(0.0, 10.0, 20)
        result = savgol_smooth(s, window_size=5, polyorder=1)
        # Linear signal should be preserved by any degree-1+ polynomial
        np.testing.assert_allclose(result, s, atol=0.5)


# ─── smooth_contour ───────────────────────────────────────────────────────────

class TestSmoothContour:
    def test_not_n2_raises(self):
        with pytest.raises(ValueError, match="форму"):
            smooth_contour(np.ones((5, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="форму"):
            smooth_contour(np.ones(10))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пустым"):
            smooth_contour(np.zeros((0, 2)))

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            smooth_contour(make_contour(), sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            smooth_contour(make_contour(), sigma=-1.0)

    def test_returns_same_shape(self):
        c = make_contour(n=25)
        result = smooth_contour(c, sigma=1.0)
        assert result.shape == c.shape

    def test_returns_float32(self):
        result = smooth_contour(make_contour(), sigma=1.0)
        assert result.dtype == np.float32

    def test_constant_contour_preserved(self):
        c = np.full((10, 2), 5.0, dtype=np.float32)
        result = smooth_contour(c, sigma=1.0)
        np.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_larger_sigma_more_smoothing(self):
        rng = np.random.default_rng(0)
        c = rng.standard_normal((50, 2)).astype(np.float32)
        s1 = smooth_contour(c, sigma=0.5)
        s5 = smooth_contour(c, sigma=5.0)
        # s5 should be smoother (lower std of differences from mean)
        assert np.std(s5) <= np.std(s1) + 0.1  # lenient


# ─── apply_smoothing ──────────────────────────────────────────────────────────

class TestApplySmoothing:
    def _apply(self, method, **kwargs):
        p = SmoothingParams(method=method, **kwargs)
        return apply_smoothing(make_signal(n=30), p)

    def test_moving_average(self):
        result = self._apply("moving_average", window_size=5)
        assert len(result) == 30

    def test_gaussian(self):
        result = self._apply("gaussian", sigma=1.0)
        assert len(result) == 30

    def test_median(self):
        result = self._apply("median", window_size=5)
        assert len(result) == 30

    def test_savgol(self):
        result = self._apply("savgol", window_size=5, polyorder=2)
        assert len(result) == 30

    def test_exponential(self):
        result = self._apply("exponential", alpha=0.4)
        assert len(result) == 30

    def test_returns_float64(self):
        p = SmoothingParams()
        result = apply_smoothing(make_signal(), p)
        assert result.dtype == np.float64


# ─── batch_smooth ─────────────────────────────────────────────────────────────

class TestBatchSmooth:
    def test_empty_list_returns_empty(self):
        p = SmoothingParams()
        assert batch_smooth([], p) == []

    def test_length_preserved(self):
        signals = [make_signal(n=10), make_signal(n=20), make_signal(n=15)]
        p = SmoothingParams()
        results = batch_smooth(signals, p)
        assert len(results) == 3

    def test_each_result_same_length_as_input(self):
        signals = [make_signal(n=n) for n in [10, 20, 30]]
        p = SmoothingParams()
        results = batch_smooth(signals, p)
        for s, r in zip(signals, results):
            assert len(r) == len(s)

    def test_each_result_float64(self):
        p = SmoothingParams(method="gaussian")
        results = batch_smooth([make_signal() for _ in range(3)], p)
        for r in results:
            assert r.dtype == np.float64
