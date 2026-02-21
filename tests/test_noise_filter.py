"""Тесты для puzzle_reconstruction.preprocessing.noise_filter."""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.noise_filter import (
    NoiseFilterParams,
    average_filter,
    gaussian_filter,
    median_filter,
    bilateral_filter,
    nlm_filter,
    apply_noise_filter,
    batch_noise_filter,
)


# ─── TestNoiseFilterParams ────────────────────────────────────────────────────

class TestNoiseFilterParams:
    def test_default_values(self):
        p = NoiseFilterParams()
        assert p.method == "gaussian"
        assert p.kernel_size == 5
        assert p.sigma_color == 75.0
        assert p.sigma_space == 75.0
        assert p.h == 10.0
        assert p.template_window == 7
        assert p.search_window == 21

    def test_valid_methods(self):
        for m in ("average", "gaussian", "median", "bilateral", "nlm"):
            p = NoiseFilterParams(method=m)
            assert p.method == m

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(method="unknown")

    def test_kernel_size_too_small_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(kernel_size=2)

    def test_kernel_size_even_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(kernel_size=4)

    def test_sigma_color_nonpositive_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(sigma_color=0.0)

    def test_sigma_space_nonpositive_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(sigma_space=-1.0)

    def test_h_nonpositive_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(h=0.0)

    def test_template_window_even_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(template_window=6)

    def test_template_window_too_small_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(template_window=1)

    def test_search_window_even_raises(self):
        with pytest.raises(ValueError):
            NoiseFilterParams(search_window=8)


# ─── TestAverageFilter ────────────────────────────────────────────────────────

class TestAverageFilter:
    def _gray(self):
        return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_returns_uint8(self):
        out = average_filter(self._gray(), kernel_size=3)
        assert out.dtype == np.uint8

    def test_same_shape_gray(self):
        img = self._gray()
        out = average_filter(img, kernel_size=3)
        assert out.shape == img.shape

    def test_same_shape_color(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = average_filter(img, kernel_size=3)
        assert out.shape == img.shape

    def test_constant_image_unchanged(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        out = average_filter(img, kernel_size=5)
        np.testing.assert_array_equal(out, img)

    def test_kernel_too_small_raises(self):
        with pytest.raises(ValueError):
            average_filter(self._gray(), kernel_size=2)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            average_filter(self._gray(), kernel_size=4)

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            average_filter(np.ones((4, 4, 4, 4), dtype=np.uint8), kernel_size=3)


# ─── TestGaussianFilter ───────────────────────────────────────────────────────

class TestGaussianFilter:
    def _gray(self):
        return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_returns_uint8(self):
        out = gaussian_filter(self._gray(), kernel_size=5)
        assert out.dtype == np.uint8

    def test_same_shape(self):
        img = self._gray()
        out = gaussian_filter(img, kernel_size=5)
        assert out.shape == img.shape

    def test_constant_image_unchanged(self):
        img = np.full((20, 20), 200, dtype=np.uint8)
        out = gaussian_filter(img, kernel_size=5)
        np.testing.assert_array_equal(out, img)

    def test_kernel_too_small_raises(self):
        with pytest.raises(ValueError):
            gaussian_filter(self._gray(), kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            gaussian_filter(self._gray(), kernel_size=6)

    def test_color_image(self):
        img = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
        out = gaussian_filter(img, kernel_size=3)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            gaussian_filter(np.ones((3,), dtype=np.uint8), kernel_size=3)


# ─── TestMedianFilter ─────────────────────────────────────────────────────────

class TestMedianFilter:
    def _gray(self):
        return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_returns_uint8(self):
        out = median_filter(self._gray(), kernel_size=3)
        assert out.dtype == np.uint8

    def test_same_shape(self):
        img = self._gray()
        out = median_filter(img, kernel_size=3)
        assert out.shape == img.shape

    def test_removes_salt_pepper(self):
        img = np.full((16, 16), 128, dtype=np.uint8)
        img[8, 8] = 255  # соль
        img[4, 4] = 0    # перец
        out = median_filter(img, kernel_size=3)
        assert out[8, 8] < 255
        assert out[4, 4] > 0

    def test_kernel_too_small_raises(self):
        with pytest.raises(ValueError):
            median_filter(self._gray(), kernel_size=2)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            median_filter(self._gray(), kernel_size=4)

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            median_filter(np.ones((2, 2, 2, 2), dtype=np.uint8), kernel_size=3)


# ─── TestBilateralFilter ──────────────────────────────────────────────────────

class TestBilateralFilter:
    def _gray(self):
        return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_returns_uint8(self):
        out = bilateral_filter(self._gray(), kernel_size=9)
        assert out.dtype == np.uint8

    def test_same_shape(self):
        img = self._gray()
        out = bilateral_filter(img, kernel_size=9)
        assert out.shape == img.shape

    def test_constant_image_unchanged(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        out = bilateral_filter(img, kernel_size=5)
        np.testing.assert_array_equal(out, img)

    def test_kernel_too_small_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(self._gray(), kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(self._gray(), kernel_size=8)

    def test_sigma_color_nonpositive_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(self._gray(), kernel_size=5, sigma_color=0.0)

    def test_sigma_space_nonpositive_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(self._gray(), kernel_size=5, sigma_space=-1.0)

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            bilateral_filter(np.ones((2,), dtype=np.uint8), kernel_size=3)


# ─── TestNlmFilter ────────────────────────────────────────────────────────────

class TestNlmFilter:
    def _gray(self):
        return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_returns_uint8_gray(self):
        out = nlm_filter(self._gray(), h=10.0, template_window=7, search_window=21)
        assert out.dtype == np.uint8

    def test_same_shape_gray(self):
        img = self._gray()
        out = nlm_filter(img, h=10.0, template_window=7, search_window=21)
        assert out.shape == img.shape

    def test_same_shape_color(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = nlm_filter(img, h=10.0, template_window=7, search_window=21)
        assert out.shape == img.shape

    def test_h_nonpositive_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(self._gray(), h=0.0)

    def test_template_window_even_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(self._gray(), template_window=6)

    def test_search_window_even_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(self._gray(), search_window=4)

    def test_search_window_too_small_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(self._gray(), search_window=1)

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            nlm_filter(np.ones((4,), dtype=np.uint8))


# ─── TestApplyNoiseFilter ─────────────────────────────────────────────────────

class TestApplyNoiseFilter:
    def _gray(self):
        return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

    def test_average(self):
        params = NoiseFilterParams(method="average", kernel_size=3)
        out = apply_noise_filter(self._gray(), params)
        assert out.dtype == np.uint8

    def test_gaussian(self):
        params = NoiseFilterParams(method="gaussian", kernel_size=5)
        out = apply_noise_filter(self._gray(), params)
        assert out.dtype == np.uint8

    def test_median(self):
        params = NoiseFilterParams(method="median", kernel_size=3)
        out = apply_noise_filter(self._gray(), params)
        assert out.dtype == np.uint8

    def test_bilateral(self):
        params = NoiseFilterParams(method="bilateral", kernel_size=9)
        out = apply_noise_filter(self._gray(), params)
        assert out.dtype == np.uint8

    def test_nlm(self):
        params = NoiseFilterParams(method="nlm", h=10.0,
                                   template_window=7, search_window=21)
        out = apply_noise_filter(self._gray(), params)
        assert out.dtype == np.uint8


# ─── TestBatchNoiseFilter ─────────────────────────────────────────────────────

class TestBatchNoiseFilter:
    def _gray(self, n=3):
        return [np.random.randint(0, 256, (16, 16), dtype=np.uint8) for _ in range(n)]

    def test_returns_list(self):
        params = NoiseFilterParams(method="gaussian", kernel_size=3)
        out = batch_noise_filter(self._gray(), params)
        assert isinstance(out, list)

    def test_correct_length(self):
        imgs = self._gray(5)
        params = NoiseFilterParams(method="median", kernel_size=3)
        out = batch_noise_filter(imgs, params)
        assert len(out) == 5

    def test_empty_list(self):
        params = NoiseFilterParams()
        out = batch_noise_filter([], params)
        assert out == []

    def test_each_uint8(self):
        params = NoiseFilterParams(method="average", kernel_size=3)
        for img_out in batch_noise_filter(self._gray(3), params):
            assert img_out.dtype == np.uint8
