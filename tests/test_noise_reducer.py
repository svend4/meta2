"""Тесты для puzzle_reconstruction/preprocessing/noise_reducer.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_reducer import (
    NoiseReductionResult,
    estimate_noise,
    gaussian_reduce,
    median_reduce,
    bilateral_reduce,
    auto_reduce,
    batch_reduce,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 180
    img[:, :, 1] = 120
    img[:, :, 2] = 60
    return img


def _noisy(h=64, w=64, seed=99):
    """Высокошумное изображение (случайные пиксели)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _smooth(h=64, w=64):
    """Равномерно-гладкое изображение (нулевой шум)."""
    return np.full((h, w), 100, dtype=np.uint8)


# ─── NoiseReductionResult ─────────────────────────────────────────────────────

class TestNoiseReductionResult:
    def _make(self):
        return NoiseReductionResult(
            filtered=np.zeros((32, 32), dtype=np.uint8),
            method="gaussian",
            params={"ksize": 5},
            noise_estimate_before=10.0,
            noise_estimate_after=3.0,
        )

    def test_filtered_stored(self):
        r = self._make()
        assert r.filtered.shape == (32, 32)

    def test_method_stored(self):
        r = self._make()
        assert r.method == "gaussian"

    def test_params_stored(self):
        r = self._make()
        assert r.params["ksize"] == 5

    def test_params_default_empty(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="median",
        )
        assert isinstance(r.params, dict)

    def test_noise_before_stored(self):
        r = self._make()
        assert r.noise_estimate_before == pytest.approx(10.0)

    def test_noise_after_stored(self):
        r = self._make()
        assert r.noise_estimate_after == pytest.approx(3.0)

    def test_noise_defaults_zero(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="gaussian",
        )
        assert r.noise_estimate_before == pytest.approx(0.0)
        assert r.noise_estimate_after == pytest.approx(0.0)

    def test_repr_contains_method(self):
        assert "gaussian" in repr(self._make())

    def test_repr_contains_result(self):
        r = self._make()
        s = repr(r)
        assert "NoiseReductionResult" in s

    def test_filtered_ndarray(self):
        assert isinstance(self._make().filtered, np.ndarray)


# ─── estimate_noise ───────────────────────────────────────────────────────────

class TestEstimateNoise:
    def test_returns_float(self):
        assert isinstance(estimate_noise(_gray()), float)

    def test_nonneg(self):
        assert estimate_noise(_gray()) >= 0.0

    def test_constant_image_near_zero(self):
        assert estimate_noise(_smooth()) < 1.0

    def test_noisy_greater_than_constant(self):
        sigma_noisy   = estimate_noise(_noisy())
        sigma_constant = estimate_noise(_smooth())
        assert sigma_noisy > sigma_constant

    def test_gray_input(self):
        v = estimate_noise(_gray())
        assert isinstance(v, float)

    def test_bgr_input(self):
        v = estimate_noise(_bgr())
        assert v >= 0.0

    def test_all_zeros_near_zero(self):
        assert estimate_noise(np.zeros((32, 32), dtype=np.uint8)) < 1.0

    def test_checkerboard_high_noise(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[::2, ::2] = 255  # шахматный узор
        assert estimate_noise(img) > 1.0


# ─── gaussian_reduce ──────────────────────────────────────────────────────────

class TestGaussianReduce:
    def test_returns_result(self):
        assert isinstance(gaussian_reduce(_noisy()), NoiseReductionResult)

    def test_method(self):
        assert gaussian_reduce(_noisy()).method == "gaussian"

    def test_same_shape_gray(self):
        r = gaussian_reduce(_noisy(40, 60))
        assert r.filtered.shape == (40, 60)

    def test_same_shape_bgr(self):
        r = gaussian_reduce(_bgr(40, 60))
        assert r.filtered.shape == (40, 60, 3)

    def test_dtype_uint8(self):
        assert gaussian_reduce(_noisy()).filtered.dtype == np.uint8

    def test_ksize_stored(self):
        r = gaussian_reduce(_noisy(), ksize=7)
        assert r.params.get("ksize") == 7

    def test_sigma_stored(self):
        r = gaussian_reduce(_noisy(), sigma=1.5)
        assert r.params.get("sigma") == pytest.approx(1.5)

    def test_noise_before_nonneg(self):
        assert gaussian_reduce(_noisy()).noise_estimate_before >= 0.0

    def test_noise_after_nonneg(self):
        assert gaussian_reduce(_noisy()).noise_estimate_after >= 0.0

    def test_noise_reduces_for_noisy_image(self):
        r = gaussian_reduce(_noisy(), ksize=5)
        assert r.noise_estimate_after < r.noise_estimate_before

    def test_gray_input(self):
        r = gaussian_reduce(_gray())
        assert r.filtered.ndim == 2

    def test_bgr_input(self):
        r = gaussian_reduce(_bgr())
        assert r.filtered.ndim == 3

    def test_even_ksize_made_odd(self):
        r = gaussian_reduce(_noisy(), ksize=4)
        # Реализация должна обеспечить нечётность ядра
        assert r.filtered.shape == (64, 64)


# ─── median_reduce ────────────────────────────────────────────────────────────

class TestMedianReduce:
    def test_returns_result(self):
        assert isinstance(median_reduce(_noisy()), NoiseReductionResult)

    def test_method(self):
        assert median_reduce(_noisy()).method == "median"

    def test_same_shape_gray(self):
        r = median_reduce(_noisy(50, 70))
        assert r.filtered.shape == (50, 70)

    def test_same_shape_bgr(self):
        r = median_reduce(_bgr(50, 70))
        assert r.filtered.shape == (50, 70, 3)

    def test_dtype_uint8(self):
        assert median_reduce(_noisy()).filtered.dtype == np.uint8

    def test_ksize_stored(self):
        r = median_reduce(_noisy(), ksize=3)
        assert r.params.get("ksize") == 3

    def test_noise_before_nonneg(self):
        assert median_reduce(_noisy()).noise_estimate_before >= 0.0

    def test_noise_after_nonneg(self):
        assert median_reduce(_noisy()).noise_estimate_after >= 0.0

    def test_noise_reduces(self):
        r = median_reduce(_noisy(), ksize=5)
        assert r.noise_estimate_after < r.noise_estimate_before

    def test_gray_input(self):
        assert median_reduce(_gray()).filtered.ndim == 2

    def test_bgr_input(self):
        assert median_reduce(_bgr()).filtered.ndim == 3

    def test_smooth_image_unchanged(self):
        img = _smooth()
        r   = median_reduce(img, ksize=3)
        np.testing.assert_array_equal(r.filtered, img)


# ─── bilateral_reduce ─────────────────────────────────────────────────────────

class TestBilateralReduce:
    def test_returns_result(self):
        assert isinstance(bilateral_reduce(_noisy()), NoiseReductionResult)

    def test_method(self):
        assert bilateral_reduce(_noisy()).method == "bilateral"

    def test_same_shape_gray(self):
        r = bilateral_reduce(_noisy(32, 48))
        assert r.filtered.shape == (32, 48)

    def test_same_shape_bgr(self):
        r = bilateral_reduce(_bgr(32, 48))
        assert r.filtered.shape == (32, 48, 3)

    def test_dtype_uint8(self):
        assert bilateral_reduce(_noisy()).filtered.dtype == np.uint8

    def test_params_d(self):
        r = bilateral_reduce(_noisy(), d=5)
        assert r.params.get("d") == 5

    def test_params_sigma_color(self):
        r = bilateral_reduce(_noisy(), sigma_color=50.0)
        assert r.params.get("sigma_color") == pytest.approx(50.0)

    def test_params_sigma_space(self):
        r = bilateral_reduce(_noisy(), sigma_space=40.0)
        assert r.params.get("sigma_space") == pytest.approx(40.0)

    def test_noise_before_nonneg(self):
        assert bilateral_reduce(_noisy()).noise_estimate_before >= 0.0

    def test_noise_after_nonneg(self):
        assert bilateral_reduce(_noisy()).noise_estimate_after >= 0.0

    def test_gray_input(self):
        assert bilateral_reduce(_gray()).filtered.ndim == 2

    def test_bgr_input(self):
        assert bilateral_reduce(_bgr()).filtered.ndim == 3


# ─── auto_reduce ──────────────────────────────────────────────────────────────

class TestAutoReduce:
    def test_returns_result(self):
        assert isinstance(auto_reduce(_noisy()), NoiseReductionResult)

    def test_method_is_auto(self):
        assert auto_reduce(_noisy()).method == "auto"

    def test_chosen_filter_in_params(self):
        r = auto_reduce(_noisy())
        assert "chosen_filter" in r.params

    def test_noise_before_stored(self):
        assert auto_reduce(_noisy()).noise_estimate_before >= 0.0

    def test_noise_after_stored(self):
        assert auto_reduce(_noisy()).noise_estimate_after >= 0.0

    def test_thresholds_in_params(self):
        r = auto_reduce(_noisy(), low_thresh=3.0, high_thresh=15.0)
        assert r.params.get("low_thresh") == pytest.approx(3.0)
        assert r.params.get("high_thresh") == pytest.approx(15.0)

    def test_clean_image_uses_trivial(self):
        # σ ≈ 0 для постоянного изображения → gaussian_trivial
        r = auto_reduce(_smooth(), low_thresh=5.0)
        assert r.params.get("chosen_filter") == "gaussian_trivial"

    def test_noisy_image_uses_median(self):
        # Высокий шум → median
        r = auto_reduce(_noisy(), low_thresh=1.0, high_thresh=2.0)
        assert r.params.get("chosen_filter") == "median"

    def test_same_shape(self):
        r = auto_reduce(_noisy(48, 64))
        assert r.filtered.shape == (48, 64)

    def test_dtype_uint8(self):
        assert auto_reduce(_noisy()).filtered.dtype == np.uint8

    def test_gray_input(self):
        r = auto_reduce(_gray())
        assert r.filtered.ndim == 2

    def test_bgr_input(self):
        r = auto_reduce(_bgr())
        assert r.filtered.ndim == 3

    def test_sigma_before_in_params(self):
        r = auto_reduce(_noisy())
        assert "sigma_before" in r.params
        assert r.params["sigma_before"] >= 0.0


# ─── batch_reduce ─────────────────────────────────────────────────────────────

class TestBatchReduce:
    def test_returns_list(self):
        r = batch_reduce([_noisy(), _smooth()])
        assert isinstance(r, list)
        assert len(r) == 2

    def test_same_length(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        r    = batch_reduce(imgs)
        assert len(r) == len(imgs)

    def test_each_is_result(self):
        for r in batch_reduce([_noisy(), _smooth()]):
            assert isinstance(r, NoiseReductionResult)

    def test_empty_list(self):
        assert batch_reduce([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_reduce([_noisy()], method="magic_filter_xyz")

    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral", "auto"])
    def test_all_methods(self, method):
        r = batch_reduce([_noisy(), _smooth()], method=method)
        assert len(r) == 2
        for result in r:
            assert isinstance(result, NoiseReductionResult)
            assert result.method == method

    def test_ksize_forwarded(self):
        r = batch_reduce([_noisy()], method="gaussian", ksize=3)
        assert r[0].params.get("ksize") == 3

    def test_bgr_input(self):
        r = batch_reduce([_bgr()], method="gaussian")
        assert r[0].filtered.ndim == 3

    def test_shapes_preserved(self):
        imgs = [_noisy(32, 48), _noisy(64, 80)]
        r    = batch_reduce(imgs)
        assert r[0].filtered.shape == (32, 48)
        assert r[1].filtered.shape == (64, 80)
