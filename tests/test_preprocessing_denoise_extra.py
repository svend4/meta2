"""Additional tests for puzzle_reconstruction.preprocessing.denoise."""
import numpy as np
import pytest
from puzzle_reconstruction.preprocessing.denoise import (
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlmeans_denoise,
    auto_denoise,
    denoise_batch,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestGaussianDenoiseExtra ─────────────────────────────────────────────────

class TestGaussianDenoiseExtra:
    def test_large_sigma_blurs(self):
        img = _rand_gray(32, 32, seed=1)
        out = gaussian_denoise(img, sigma=5.0)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_non_square_gray(self):
        img = _rand_gray(24, 48)
        out = gaussian_denoise(img, sigma=1.0)
        assert out.shape == (24, 48)

    def test_non_square_rgb(self):
        img = _rand_rgb(24, 48)
        out = gaussian_denoise(img, sigma=1.0)
        assert out.shape == (24, 48, 3)

    def test_64x64_no_error(self):
        img = _rand_gray(64, 64)
        out = gaussian_denoise(img, sigma=2.0)
        assert out.shape == (64, 64)

    def test_values_in_range(self):
        img = _rand_gray(32, 32, seed=5)
        out = gaussian_denoise(img, sigma=1.5)
        assert out.min() >= 0 and out.max() <= 255

    def test_large_sigma_constant_img_unchanged(self):
        img = _gray(32, 32, val=100)
        out = gaussian_denoise(img, sigma=10.0)
        np.testing.assert_array_almost_equal(out, img, decimal=0)

    def test_multiple_sigmas_consistent_shape(self):
        img = _rand_gray()
        for sigma in [0.5, 1.0, 2.0, 3.0]:
            assert gaussian_denoise(img, sigma=sigma).shape == img.shape

    def test_rgb_values_in_range(self):
        img = _rand_rgb()
        out = gaussian_denoise(img, sigma=1.0)
        assert out.min() >= 0 and out.max() <= 255


# ─── TestMedianDenoiseExtra ───────────────────────────────────────────────────

class TestMedianDenoiseExtra:
    def test_large_ksize(self):
        img = _rand_gray(32, 32)
        out = median_denoise(img, ksize=9)
        assert out.shape == (32, 32)

    def test_even_ksize_forced_odd(self):
        img = _rand_gray()
        out = median_denoise(img, ksize=4)
        assert out.shape == img.shape

    def test_non_square_gray(self):
        img = _rand_gray(24, 48)
        out = median_denoise(img)
        assert out.shape == (24, 48)

    def test_salt_pepper_rgb(self):
        img = _rand_rgb(32, 32, seed=3)
        out = median_denoise(img, ksize=3)
        assert out.shape == (32, 32, 3)

    def test_constant_image_preserved(self):
        img = _gray(32, 32, val=200)
        out = median_denoise(img, ksize=3)
        np.testing.assert_array_equal(out, img)

    def test_dtype_rgb_preserved(self):
        img = _rand_rgb()
        out = median_denoise(img)
        assert out.dtype == np.uint8

    def test_values_in_range_rgb(self):
        img = _rand_rgb()
        out = median_denoise(img, ksize=3)
        assert out.min() >= 0 and out.max() <= 255


# ─── TestBilateralDenoiseExtra ────────────────────────────────────────────────

class TestBilateralDenoiseExtra:
    def test_large_d_param(self):
        img = _rand_gray(32, 32)
        out = bilateral_denoise(img, d=9)
        assert out.shape == (32, 32)

    def test_non_square_gray(self):
        img = _rand_gray(24, 48)
        out = bilateral_denoise(img)
        assert out.shape == (24, 48)

    def test_constant_rgb_preserved(self):
        img = np.full((32, 32, 3), 150, dtype=np.uint8)
        out = bilateral_denoise(img)
        np.testing.assert_array_equal(out, img)

    def test_large_sigma_color(self):
        img = _rand_gray(32, 32)
        out = bilateral_denoise(img, sigma_color=200.0, sigma_space=200.0)
        assert out.shape == (32, 32)

    def test_values_in_range_gray(self):
        img = _rand_gray()
        out = bilateral_denoise(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_rgb_multiple_calls_consistent(self):
        img = _rand_rgb()
        out1 = bilateral_denoise(img)
        out2 = bilateral_denoise(img)
        np.testing.assert_array_equal(out1, out2)


# ─── TestNlmeansDenoiseExtra ──────────────────────────────────────────────────

class TestNlmeansDenoiseExtra:
    def test_large_gray(self):
        img = _rand_gray(64, 64)
        out = nlmeans_denoise(img)
        assert out.shape == (64, 64)

    def test_large_rgb(self):
        img = _rand_rgb(32, 32)
        out = nlmeans_denoise(img)
        assert out.shape == (32, 32, 3)

    def test_non_square_gray(self):
        img = _rand_gray(16, 32)
        out = nlmeans_denoise(img)
        assert out.shape == (16, 32)

    def test_values_in_range_gray(self):
        img = _rand_gray(16, 16)
        out = nlmeans_denoise(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_values_in_range_rgb(self):
        img = _rand_rgb(16, 16)
        out = nlmeans_denoise(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_constant_gray_preserved(self):
        img = _gray(16, 16, val=120)
        out = nlmeans_denoise(img)
        np.testing.assert_array_almost_equal(out, img, decimal=0)


# ─── TestAutoDenoiseExtra ─────────────────────────────────────────────────────

class TestAutoDenoiseExtra:
    def test_high_noise_still_returns_correct_shape(self):
        img = _rand_rgb(32, 32, seed=77)
        out = auto_denoise(img)
        assert out.shape == (32, 32, 3)

    def test_aggressive_false_default(self):
        img = _rand_rgb(16, 16)
        out = auto_denoise(img, aggressive=False)
        assert out.shape == img.shape

    def test_non_square_rgb(self):
        img = _rand_rgb(24, 48)
        out = auto_denoise(img)
        assert out.shape == (24, 48, 3)

    def test_all_white_returns_same(self):
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        out = auto_denoise(img)
        assert out.shape == (32, 32, 3)

    def test_output_values_in_range(self):
        img = _rand_rgb(32, 32, seed=9)
        out = auto_denoise(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_multiple_calls_deterministic(self):
        img = _rand_rgb(16, 16, seed=3)
        out1 = auto_denoise(img)
        out2 = auto_denoise(img)
        np.testing.assert_array_equal(out1, out2)


# ─── TestDenoiseBatchExtra ────────────────────────────────────────────────────

class TestDenoiseBatchExtra:
    def test_method_nlmeans(self):
        imgs = [_rand_rgb(16, 16, seed=i) for i in range(2)]
        result = denoise_batch(imgs, method="nlmeans")
        assert len(result) == 2
        assert all(r.shape == (16, 16, 3) for r in result)

    def test_mixed_gray_and_rgb(self):
        imgs = [_rand_gray(seed=0), _rand_rgb(seed=1)]
        result = denoise_batch(imgs, method="gaussian", sigma=1.0)
        assert result[0].shape == (32, 32)
        assert result[1].shape == (32, 32, 3)

    def test_sigma_kwarg_passed(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        result = denoise_batch(imgs, method="gaussian", sigma=3.0)
        assert len(result) == 3

    def test_all_none_returns_nones(self):
        result = denoise_batch([None, None], method="gaussian")
        assert result[0] is None and result[1] is None

    def test_large_batch(self):
        imgs = [_rand_gray(16, 16, seed=i) for i in range(10)]
        result = denoise_batch(imgs, method="median")
        assert len(result) == 10

    def test_shapes_preserved_mixed_sizes(self):
        imgs = [_rand_gray(16, 16), _rand_gray(32, 32), _rand_gray(8, 8)]
        result = denoise_batch(imgs, method="gaussian", sigma=1.0)
        for orig, out in zip(imgs, result):
            assert out.shape == orig.shape
