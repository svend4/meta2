"""Extra tests for puzzle_reconstruction.preprocessing.noise_filter."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_filter import (
    NoiseFilterParams,
    apply_noise_filter,
    average_filter,
    batch_noise_filter,
    bilateral_filter,
    gaussian_filter,
    median_filter,
    nlm_filter,
)


def _gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _constant(h=16, w=16, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── TestNoiseFilterParamsExtra ──────────────────────────────────────────────

class TestNoiseFilterParamsExtra:
    def test_kernel_3_ok(self):
        assert NoiseFilterParams(kernel_size=3).kernel_size == 3

    def test_kernel_51_ok(self):
        assert NoiseFilterParams(kernel_size=51).kernel_size == 51

    def test_h_positive(self):
        assert NoiseFilterParams(h=0.1).h == pytest.approx(0.1)

    def test_sigma_color_positive(self):
        assert NoiseFilterParams(sigma_color=50.0).sigma_color == pytest.approx(50.0)

    def test_sigma_space_positive(self):
        assert NoiseFilterParams(sigma_space=100.0).sigma_space == pytest.approx(100.0)

    def test_template_window_3_ok(self):
        assert NoiseFilterParams(template_window=3).template_window == 3

    def test_search_window_3_ok(self):
        assert NoiseFilterParams(search_window=3).search_window == 3

    def test_search_window_too_small_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(search_window=1)

    def test_h_negative_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(h=-1.0)

    @pytest.mark.parametrize("method", ["average", "gaussian", "median", "bilateral", "nlm"])
    def test_all_methods_valid(self, method):
        assert NoiseFilterParams(method=method).method == method


# ─── TestAverageFilterExtra ──────────────────────────────────────────────────

class TestAverageFilterExtra:
    def test_smoothing_effect(self):
        img = _gray()
        out = average_filter(img, kernel_size=5)
        assert np.std(out.astype(float)) <= np.std(img.astype(float)) + 1.0

    def test_large_kernel(self):
        out = average_filter(_gray(), kernel_size=11)
        assert out.shape == (32, 32)

    def test_color_image_shape(self):
        img = _color()
        out = average_filter(img, kernel_size=3)
        assert out.shape == img.shape

    def test_constant_unchanged(self):
        img = _constant()
        np.testing.assert_array_equal(average_filter(img, kernel_size=3), img)

    def test_dtype_uint8(self):
        assert average_filter(_gray(), kernel_size=3).dtype == np.uint8

    def test_kernel_1_raises(self):
        with pytest.raises(ValueError):
            average_filter(_gray(), kernel_size=1)


# ─── TestGaussianFilterExtra ────────────────────────────────────────────────

class TestGaussianFilterExtra:
    def test_smoothing_effect(self):
        img = _gray()
        out = gaussian_filter(img, kernel_size=7)
        assert np.std(out.astype(float)) <= np.std(img.astype(float)) + 1.0

    def test_large_kernel(self):
        out = gaussian_filter(_gray(), kernel_size=11)
        assert out.shape == (32, 32)

    def test_color_shape(self):
        img = _color()
        out = gaussian_filter(img, kernel_size=5)
        assert out.shape == img.shape

    def test_constant_unchanged(self):
        img = _constant(h=32, w=32, val=200)
        np.testing.assert_array_equal(gaussian_filter(img, kernel_size=5), img)

    def test_dtype_uint8(self):
        assert gaussian_filter(_gray(), kernel_size=3).dtype == np.uint8

    def test_odd_kernel_required(self):
        with pytest.raises(ValueError):
            gaussian_filter(_gray(), kernel_size=4)


# ─── TestMedianFilterExtra ──────────────────────────────────────────────────

class TestMedianFilterExtra:
    def test_salt_pepper_removal(self):
        img = _constant(32, 32, 128)
        img[5, 5] = 255
        img[10, 10] = 0
        out = median_filter(img, kernel_size=3)
        assert out[5, 5] < 255
        assert out[10, 10] > 0

    def test_constant_unchanged(self):
        img = _constant()
        np.testing.assert_array_equal(median_filter(img, kernel_size=3), img)

    def test_shape_preserved(self):
        img = _gray(24, 48)
        assert median_filter(img, kernel_size=5).shape == (24, 48)

    def test_dtype_uint8(self):
        assert median_filter(_gray(), kernel_size=3).dtype == np.uint8

    def test_large_kernel(self):
        out = median_filter(_gray(), kernel_size=9)
        assert out.shape == (32, 32)

    def test_kernel_2_raises(self):
        with pytest.raises(ValueError):
            median_filter(_gray(), kernel_size=2)


# ─── TestBilateralFilterExtra ────────────────────────────────────────────────

class TestBilateralFilterExtra:
    def test_constant_unchanged(self):
        img = _constant(val=100)
        np.testing.assert_array_equal(
            bilateral_filter(img, kernel_size=5), img)

    def test_shape_preserved(self):
        img = _gray(24, 48)
        assert bilateral_filter(img, kernel_size=5).shape == (24, 48)

    def test_dtype_uint8(self):
        assert bilateral_filter(_gray(), kernel_size=5).dtype == np.uint8

    def test_color_image(self):
        img = _color()
        out = bilateral_filter(img, kernel_size=5)
        assert out.shape == img.shape

    def test_sigma_color_large(self):
        out = bilateral_filter(_gray(), kernel_size=5, sigma_color=200.0)
        assert out.dtype == np.uint8

    def test_sigma_space_large(self):
        out = bilateral_filter(_gray(), kernel_size=5, sigma_space=200.0)
        assert out.dtype == np.uint8

    def test_kernel_3_ok(self):
        out = bilateral_filter(_gray(), kernel_size=3)
        assert out.shape == (32, 32)


# ─── TestNlmFilterExtra ─────────────────────────────────────────────────────

class TestNlmFilterExtra:
    def test_gray_shape(self):
        img = _gray()
        out = nlm_filter(img, h=10.0, template_window=7, search_window=21)
        assert out.shape == img.shape

    def test_color_shape(self):
        img = _color()
        out = nlm_filter(img, h=10.0, template_window=7, search_window=21)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        out = nlm_filter(_gray(), h=10.0, template_window=7, search_window=21)
        assert out.dtype == np.uint8

    def test_h_large(self):
        out = nlm_filter(_gray(), h=50.0, template_window=7, search_window=21)
        assert out.dtype == np.uint8

    def test_template_window_3(self):
        out = nlm_filter(_gray(), h=10.0, template_window=3, search_window=21)
        assert out.shape == (32, 32)

    def test_template_window_even_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(_gray(), template_window=4)

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(_gray(), h=0.0)


# ─── TestApplyNoiseFilterExtra ───────────────────────────────────────────────

class TestApplyNoiseFilterExtra:
    @pytest.mark.parametrize("method", ["average", "gaussian", "median"])
    def test_basic_methods_shape(self, method):
        params = NoiseFilterParams(method=method, kernel_size=3)
        out = apply_noise_filter(_gray(), params)
        assert out.shape == (32, 32)

    def test_bilateral_shape(self):
        params = NoiseFilterParams(method="bilateral", kernel_size=5)
        out = apply_noise_filter(_gray(), params)
        assert out.shape == (32, 32)

    def test_nlm_shape(self):
        params = NoiseFilterParams(method="nlm", h=10.0,
                                   template_window=7, search_window=21)
        out = apply_noise_filter(_gray(), params)
        assert out.shape == (32, 32)

    def test_all_return_uint8(self):
        for method in ("average", "gaussian", "median", "bilateral"):
            params = NoiseFilterParams(method=method, kernel_size=5)
            out = apply_noise_filter(_gray(), params)
            assert out.dtype == np.uint8

    def test_color_image(self):
        params = NoiseFilterParams(method="gaussian", kernel_size=3)
        out = apply_noise_filter(_color(), params)
        assert out.shape == (32, 32, 3)


# ─── TestBatchNoiseFilterExtra ───────────────────────────────────────────────

class TestBatchNoiseFilterExtra:
    def test_empty(self):
        params = NoiseFilterParams(method="gaussian", kernel_size=3)
        assert batch_noise_filter([], params) == []

    def test_single(self):
        params = NoiseFilterParams(method="median", kernel_size=3)
        result = batch_noise_filter([_gray()], params)
        assert len(result) == 1

    def test_multiple(self):
        imgs = [_gray(seed=i) for i in range(5)]
        params = NoiseFilterParams(method="average", kernel_size=3)
        result = batch_noise_filter(imgs, params)
        assert len(result) == 5

    def test_all_uint8(self):
        imgs = [_gray(seed=i) for i in range(3)]
        params = NoiseFilterParams(method="gaussian", kernel_size=5)
        for out in batch_noise_filter(imgs, params):
            assert out.dtype == np.uint8

    def test_shapes_preserved(self):
        imgs = [_gray(16, 24), _gray(16, 24, seed=1)]
        params = NoiseFilterParams(method="median", kernel_size=3)
        for out in batch_noise_filter(imgs, params):
            assert out.shape == (16, 24)

    def test_returns_list(self):
        params = NoiseFilterParams(method="average", kernel_size=3)
        assert isinstance(batch_noise_filter([_gray()], params), list)
