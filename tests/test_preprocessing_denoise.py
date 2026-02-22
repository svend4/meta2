"""Тесты для puzzle_reconstruction.preprocessing.denoise."""
import pytest
import numpy as np
from puzzle_reconstruction.preprocessing.denoise import (
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlmeans_denoise,
    auto_denoise,
    denoise_batch,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=32, w=32, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=32, w=32, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestGaussianDenoise ──────────────────────────────────────────────────────

class TestGaussianDenoise:
    def test_returns_ndarray(self):
        out = gaussian_denoise(_rand_gray())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved_gray(self):
        img = _rand_gray(24, 32)
        out = gaussian_denoise(img)
        assert out.shape == (24, 32)

    def test_shape_preserved_rgb(self):
        img = _rand_rgb(24, 32)
        out = gaussian_denoise(img)
        assert out.shape == (24, 32, 3)

    def test_sigma_zero_returns_same(self):
        img = _rand_gray()
        out = gaussian_denoise(img, sigma=0)
        np.testing.assert_array_equal(out, img)

    def test_sigma_negative_returns_same(self):
        img = _rand_gray()
        out = gaussian_denoise(img, sigma=-1.0)
        np.testing.assert_array_equal(out, img)

    def test_small_sigma_smoother(self):
        img = _rand_gray(32, 32, seed=42)
        out = gaussian_denoise(img, sigma=2.0)
        assert out.std() <= img.std() + 5

    def test_rgb_ok(self):
        out = gaussian_denoise(_rand_rgb(), sigma=1.0)
        assert out.shape == (32, 32, 3)

    def test_output_dtype_uint8(self):
        out = gaussian_denoise(_rand_gray())
        assert out.dtype == np.uint8

    def test_constant_image_unchanged(self):
        img = _gray()
        out = gaussian_denoise(img, sigma=2.0)
        np.testing.assert_array_almost_equal(out, img, decimal=0)


# ─── TestMedianDenoise ────────────────────────────────────────────────────────

class TestMedianDenoise:
    def test_returns_ndarray(self):
        out = median_denoise(_rand_gray())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved_gray(self):
        img = _rand_gray(24, 32)
        out = median_denoise(img)
        assert out.shape == (24, 32)

    def test_shape_preserved_rgb(self):
        img = _rand_rgb(24, 32)
        out = median_denoise(img)
        assert out.shape == (24, 32, 3)

    def test_output_dtype_uint8(self):
        out = median_denoise(_rand_gray())
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        out = median_denoise(_rand_gray())
        assert out.min() >= 0
        assert out.max() <= 255

    def test_removes_salt_pepper(self):
        # Create image with salt-and-pepper noise
        img = np.full((32, 32), 128, dtype=np.uint8)
        rng = np.random.default_rng(0)
        noise_mask = rng.random((32, 32)) < 0.1
        img[noise_mask] = 0
        img[~noise_mask & (rng.random((32, 32)) < 0.1)] = 255
        out = median_denoise(img, ksize=3)
        # After median filter, fewer extreme pixels
        assert (out == 0).sum() <= (img == 0).sum()

    def test_ksize_1_becomes_3(self):
        # ksize is forced to be odd and >= 3
        out = median_denoise(_rand_gray(), ksize=1)
        assert out.shape == (32, 32)


# ─── TestBilateralDenoise ─────────────────────────────────────────────────────

class TestBilateralDenoise:
    def test_returns_ndarray(self):
        out = bilateral_denoise(_rand_gray())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved_gray(self):
        img = _rand_gray(24, 32)
        out = bilateral_denoise(img)
        assert out.shape == (24, 32)

    def test_shape_preserved_rgb(self):
        img = _rand_rgb(24, 32)
        out = bilateral_denoise(img)
        assert out.shape == (24, 32, 3)

    def test_output_dtype_uint8(self):
        out = bilateral_denoise(_rand_gray())
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        out = bilateral_denoise(_rand_rgb())
        assert out.min() >= 0
        assert out.max() <= 255

    def test_custom_params(self):
        out = bilateral_denoise(_rand_gray(), d=5, sigma_color=50.0, sigma_space=50.0)
        assert out.shape == (32, 32)


# ─── TestNlmeansDenoise ───────────────────────────────────────────────────────

class TestNlmeansDenoise:
    def test_returns_ndarray_gray(self):
        out = nlmeans_denoise(_rand_gray())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved_gray(self):
        out = nlmeans_denoise(_rand_gray(16, 16))
        assert out.shape == (16, 16)

    def test_shape_preserved_rgb(self):
        out = nlmeans_denoise(_rand_rgb(16, 16))
        assert out.shape == (16, 16, 3)

    def test_output_dtype_uint8(self):
        out = nlmeans_denoise(_rand_gray(16, 16))
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        out = nlmeans_denoise(_rand_gray(16, 16))
        assert out.min() >= 0
        assert out.max() <= 255


# ─── TestAutoDenoise ──────────────────────────────────────────────────────────

class TestAutoDenoise:
    def test_returns_ndarray(self):
        out = auto_denoise(_rand_rgb())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved_rgb(self):
        img = _rand_rgb(24, 32)
        out = auto_denoise(img)
        assert out.shape == (24, 32, 3)

    def test_low_noise_returns_input_or_bilateral(self):
        # Constant image has low noise level → bilateral or return original
        img = _gray(val=200)
        img_rgb = np.stack([img, img, img], axis=2)
        out = auto_denoise(img_rgb)
        assert out.shape == (32, 32, 3)

    def test_aggressive_uses_nlmeans(self):
        img = _rand_rgb(16, 16)
        out = auto_denoise(img, aggressive=True)
        assert out.shape == (16, 16, 3)

    def test_empty_image_returns_same(self):
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        out = auto_denoise(img)
        assert out is img

    def test_output_dtype_uint8(self):
        out = auto_denoise(_rand_rgb(16, 16))
        assert out.dtype == np.uint8


# ─── TestDenoiseBatch ─────────────────────────────────────────────────────────

class TestDenoiseBatch:
    def test_returns_list(self):
        imgs = [_rand_rgb(seed=i) for i in range(3)]
        result = denoise_batch(imgs, method="gaussian")
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_rand_rgb(seed=i) for i in range(4)]
        result = denoise_batch(imgs, method="median")
        assert len(result) == 4

    def test_empty_list(self):
        assert denoise_batch([], method="gaussian") == []

    def test_invalid_method_raises(self):
        imgs = [_rand_rgb()]
        with pytest.raises(ValueError):
            denoise_batch(imgs, method="wavelet")

    def test_method_gaussian(self):
        imgs = [_rand_rgb(seed=i) for i in range(2)]
        result = denoise_batch(imgs, method="gaussian", sigma=1.0)
        assert all(r.shape == imgs[0].shape for r in result)

    def test_method_median(self):
        imgs = [_rand_gray(seed=i) for i in range(2)]
        result = denoise_batch(imgs, method="median")
        assert len(result) == 2

    def test_method_bilateral(self):
        imgs = [_rand_gray(16, 16, seed=i) for i in range(2)]
        result = denoise_batch(imgs, method="bilateral")
        assert len(result) == 2

    def test_method_auto(self):
        imgs = [_rand_rgb(16, 16, seed=i) for i in range(2)]
        result = denoise_batch(imgs, method="auto")
        assert len(result) == 2

    def test_none_image_passthrough(self):
        imgs = [_rand_rgb(), None, _rand_rgb(seed=1)]
        result = denoise_batch(imgs, method="gaussian", sigma=1.0)
        assert result[1] is None

    def test_shapes_preserved(self):
        imgs = [_rand_rgb(24, 32, seed=i) for i in range(3)]
        result = denoise_batch(imgs, method="gaussian", sigma=1.0)
        for orig, out in zip(imgs, result):
            assert out.shape == orig.shape
