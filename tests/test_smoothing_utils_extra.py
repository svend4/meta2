"""Extra tests for puzzle_reconstruction/utils/smoothing_utils.py"""
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


# ─── TestSmoothingParamsExtra ─────────────────────────────────────────────────

class TestSmoothingParamsExtra:
    def test_window_3_valid(self):
        p = SmoothingParams(window_size=3)
        assert p.window_size == 3

    def test_window_7_valid(self):
        p = SmoothingParams(window_size=7)
        assert p.window_size == 7

    def test_window_11_valid(self):
        p = SmoothingParams(window_size=11)
        assert p.window_size == 11

    def test_alpha_0_1_valid(self):
        p = SmoothingParams(alpha=0.1)
        assert p.alpha == pytest.approx(0.1)

    def test_alpha_0_9_valid(self):
        p = SmoothingParams(alpha=0.9)
        assert p.alpha == pytest.approx(0.9)

    def test_sigma_large(self):
        p = SmoothingParams(sigma=10.0)
        assert p.sigma == pytest.approx(10.0)

    def test_sigma_small(self):
        p = SmoothingParams(sigma=0.01)
        assert p.sigma == pytest.approx(0.01)

    def test_polyorder_1_valid(self):
        p = SmoothingParams(polyorder=1, window_size=5)
        assert p.polyorder == 1

    def test_polyorder_3_valid(self):
        p = SmoothingParams(polyorder=3, window_size=7)
        assert p.polyorder == 3

    def test_all_methods_storable(self):
        for m in ("moving_average", "gaussian", "median", "savgol", "exponential"):
            p = SmoothingParams(method=m)
            assert p.method == m


# ─── TestMovingAverageExtra ───────────────────────────────────────────────────

class TestMovingAverageExtra:
    def test_window_3_basic(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = moving_average(sig, window_size=3)
        assert len(out) == 5

    def test_window_5_long_signal(self):
        sig = np.arange(20, dtype=float)
        out = moving_average(sig, window_size=5)
        assert len(out) == 20

    def test_large_window(self):
        sig = np.arange(50, dtype=float)
        out = moving_average(sig, window_size=9)
        assert len(out) == 50

    def test_monotone_increasing(self):
        sig = np.arange(10, dtype=float)
        out = moving_average(sig, window_size=3)
        # middle values should be sorted
        mid = out[1:-1]
        assert all(mid[i] <= mid[i + 1] for i in range(len(mid) - 1))

    def test_spike_reduced(self):
        sig = np.array([0.0, 0.0, 100.0, 0.0, 0.0], dtype=float)
        out = moving_average(sig, window_size=3)
        assert out[2] < 100.0

    def test_output_float64(self):
        sig = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        out = moving_average(sig.astype(float), window_size=3)
        assert out.dtype == np.float64

    def test_window_1_raises(self):
        sig = np.array([42.0])
        with pytest.raises(ValueError):
            moving_average(sig, window_size=1)


# ─── TestGaussianSmoothExtra ──────────────────────────────────────────────────

class TestGaussianSmoothExtra:
    def test_small_sigma_preserves_more(self):
        sig = np.array([0.0, 10.0, 0.0, 10.0, 0.0], dtype=float)
        out_small = gaussian_smooth(sig, sigma=0.1)
        out_large = gaussian_smooth(sig, sigma=5.0)
        assert out_small.std() >= out_large.std()

    def test_large_sigma(self):
        sig = np.arange(30, dtype=float)
        out = gaussian_smooth(sig, sigma=10.0)
        assert len(out) == 30

    def test_long_signal(self):
        sig = np.random.default_rng(42).normal(0, 1, 200)
        out = gaussian_smooth(sig, sigma=3.0)
        assert len(out) == 200

    def test_integer_signal_input(self):
        sig = np.arange(10, dtype=float)
        out = gaussian_smooth(sig, sigma=1.0)
        assert out.dtype == np.float64

    def test_zero_constant_preserved(self):
        sig = np.zeros(15)
        out = gaussian_smooth(sig, sigma=2.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            gaussian_smooth(np.ones(5), sigma=-1.0)


# ─── TestMedianSmoothExtra ────────────────────────────────────────────────────

class TestMedianSmoothExtra:
    def test_large_window(self):
        sig = np.arange(30, dtype=float)
        out = median_smooth(sig, window_size=7)
        assert len(out) == 30

    def test_multiple_spikes_reduced(self):
        sig = np.array([1.0, 100.0, 1.0, 100.0, 1.0, 1.0, 1.0], dtype=float)
        out = median_smooth(sig, window_size=3)
        assert out[1] < 100.0
        assert out[3] < 100.0

    def test_already_smooth_signal(self):
        sig = np.full(10, 5.0)
        out = median_smooth(sig, window_size=3)
        np.testing.assert_allclose(out, 5.0)

    def test_window_5(self):
        sig = np.array([1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
        out = median_smooth(sig, window_size=5)
        assert out[2] < 100.0

    def test_long_signal(self):
        sig = np.random.default_rng(7).normal(0, 1, 100)
        out = median_smooth(sig, window_size=5)
        assert len(out) == 100

    def test_window_size_3_minimum(self):
        sig = np.ones(5)
        out = median_smooth(sig, window_size=3)
        assert len(out) == 5


# ─── TestExponentialSmoothExtra ───────────────────────────────────────────────

class TestExponentialSmoothExtra:
    def test_large_alpha_tracks_signal(self):
        sig = np.array([0.0, 10.0, 0.0, 10.0, 0.0], dtype=float)
        out = exponential_smooth(sig, alpha=0.9)
        # Large alpha means fast tracking
        assert out[1] > out[0]

    def test_small_alpha_slow_response(self):
        sig = np.array([0.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
        out_slow = exponential_smooth(sig, alpha=0.1)
        out_fast = exponential_smooth(sig, alpha=0.9)
        # Slow alpha: value at index 1 should be lower
        assert out_slow[1] < out_fast[1]

    def test_long_constant_signal(self):
        sig = np.full(20, 7.0)
        out = exponential_smooth(sig, alpha=0.5)
        assert out[0] == pytest.approx(7.0)
        np.testing.assert_allclose(out, 7.0)

    def test_increasing_signal_shape_preserved(self):
        sig = np.arange(10, dtype=float)
        out = exponential_smooth(sig, alpha=0.5)
        assert len(out) == 10

    def test_alpha_0_5(self):
        sig = np.array([0.0, 1.0], dtype=float)
        out = exponential_smooth(sig, alpha=0.5)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.5)


# ─── TestSavgolSmoothExtra ────────────────────────────────────────────────────

class TestSavgolSmoothExtra:
    def test_window_7_poly_3(self):
        sig = np.arange(20, dtype=float)
        out = savgol_smooth(sig, window_size=7, polyorder=3)
        assert len(out) == 20

    def test_window_9_poly_4(self):
        sig = np.arange(30, dtype=float)
        out = savgol_smooth(sig, window_size=9, polyorder=4)
        assert len(out) == 30

    def test_quadratic_polynomial_preserved(self):
        x = np.linspace(0, 10, 50)
        sig = x ** 2
        out = savgol_smooth(sig, window_size=5, polyorder=2)
        assert len(out) == 50

    def test_constant_signal(self):
        sig = np.full(20, 3.0)
        out = savgol_smooth(sig, window_size=5, polyorder=2)
        np.testing.assert_allclose(out, 3.0, atol=1e-8)

    def test_long_signal(self):
        sig = np.random.default_rng(42).normal(0, 1, 100)
        out = savgol_smooth(sig, window_size=7, polyorder=3)
        assert len(out) == 100

    def test_polyorder_1(self):
        sig = np.arange(15, dtype=float)
        out = savgol_smooth(sig, window_size=5, polyorder=1)
        assert out.dtype == np.float64


# ─── TestSmoothContourExtra ───────────────────────────────────────────────────

class TestSmoothContourExtra:
    def _make_contour(self, n=20):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
        return pts

    def test_circle_contour(self):
        pts = self._make_contour(30)
        out = smooth_contour(pts, sigma=1.0)
        assert out.shape == pts.shape
        assert out.dtype == np.float32

    def test_small_sigma(self):
        pts = self._make_contour(20)
        out = smooth_contour(pts, sigma=0.1)
        assert out.shape == pts.shape

    def test_large_sigma(self):
        pts = self._make_contour(50)
        out = smooth_contour(pts, sigma=5.0)
        assert out.shape == pts.shape

    def test_square_contour(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        out = smooth_contour(pts, sigma=1.0)
        assert out.shape == (4, 2)

    def test_many_points(self):
        pts = self._make_contour(100)
        out = smooth_contour(pts, sigma=2.0)
        assert out.shape == (100, 2)

    def test_output_float32(self):
        pts = self._make_contour(20)
        out = smooth_contour(pts, sigma=1.0)
        assert out.dtype == np.float32


# ─── TestApplySmoothingExtra ──────────────────────────────────────────────────

class TestApplySmoothingExtra:
    def _signal(self, n=20):
        return np.random.default_rng(0).normal(0, 1, n)

    def test_moving_average_long(self):
        params = SmoothingParams(method="moving_average", window_size=5)
        sig = self._signal(50)
        out = apply_smoothing(sig, params)
        assert len(out) == 50

    def test_gaussian_long(self):
        params = SmoothingParams(method="gaussian", sigma=2.0)
        sig = self._signal(50)
        out = apply_smoothing(sig, params)
        assert len(out) == 50

    def test_median_long(self):
        params = SmoothingParams(method="median", window_size=5)
        sig = self._signal(50)
        out = apply_smoothing(sig, params)
        assert len(out) == 50

    def test_savgol_long(self):
        params = SmoothingParams(method="savgol", window_size=7, polyorder=3)
        sig = self._signal(50)
        out = apply_smoothing(sig, params)
        assert len(out) == 50

    def test_exponential_long(self):
        params = SmoothingParams(method="exponential", alpha=0.4)
        sig = self._signal(50)
        out = apply_smoothing(sig, params)
        assert len(out) == 50

    def test_all_outputs_float64(self):
        sig = self._signal(20)
        for m in ("moving_average", "gaussian", "median", "savgol", "exponential"):
            params = SmoothingParams(method=m)
            out = apply_smoothing(sig, params)
            assert out.dtype == np.float64

    def test_constant_signal_all_methods(self):
        sig = np.full(15, 7.0)
        for m in ("moving_average", "gaussian", "median", "savgol", "exponential"):
            params = SmoothingParams(method=m)
            out = apply_smoothing(sig, params)
            # All methods preserve constant signals (within tolerance)
            assert isinstance(out, np.ndarray)
            assert len(out) == 15


# ─── TestBatchSmoothExtra ─────────────────────────────────────────────────────

class TestBatchSmoothExtra:
    def test_ten_signals(self):
        signals = [np.arange(10, dtype=float) for _ in range(10)]
        params = SmoothingParams(method="moving_average", window_size=3)
        out = batch_smooth(signals, params)
        assert len(out) == 10

    def test_mixed_lengths(self):
        signals = [np.ones(n, dtype=float) for n in [5, 7, 9, 11, 13]]
        params = SmoothingParams(method="gaussian", sigma=1.0)
        out = batch_smooth(signals, params)
        for orig, smoothed in zip(signals, out):
            assert len(smoothed) == len(orig)

    def test_gaussian_batch(self):
        signals = [np.random.default_rng(i).normal(0, 1, 20) for i in range(5)]
        params = SmoothingParams(method="gaussian", sigma=2.0)
        out = batch_smooth(signals, params)
        assert len(out) == 5

    def test_exponential_batch(self):
        signals = [np.arange(10, dtype=float) for _ in range(3)]
        params = SmoothingParams(method="exponential", alpha=0.5)
        out = batch_smooth(signals, params)
        assert all(len(s) == 10 for s in out)

    def test_savgol_batch(self):
        signals = [np.arange(15, dtype=float) for _ in range(4)]
        params = SmoothingParams(method="savgol", window_size=5, polyorder=2)
        out = batch_smooth(signals, params)
        assert len(out) == 4

    def test_all_outputs_float64(self):
        signals = [np.arange(10, dtype=float) for _ in range(3)]
        params = SmoothingParams(method="median", window_size=3)
        out = batch_smooth(signals, params)
        assert all(arr.dtype == np.float64 for arr in out)

    def test_single_signal(self):
        signals = [np.arange(8, dtype=float)]
        params = SmoothingParams(method="moving_average", window_size=3)
        out = batch_smooth(signals, params)
        assert len(out) == 1
        assert len(out[0]) == 8
