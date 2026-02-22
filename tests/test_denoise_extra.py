"""Additional tests for puzzle_reconstruction/preprocessing/denoise.py."""
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

def _clean_bgr(h=64, w=64):
    img = np.ones((h, w, 3), dtype=np.uint8) * 180
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = (100, 150, 200)
    return img


def _noisy_bgr(seed=42, h=64, w=64):
    rng = np.random.RandomState(seed)
    base = _clean_bgr(h, w)
    noise = rng.normal(0, 25, base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _gray(h=64, w=64, val=128):
    img = np.ones((h, w), dtype=np.uint8) * val
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 200
    return img


def _rand_gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestGaussianDenoiseExtra ─────────────────────────────────────────────────

class TestGaussianDenoiseExtra:
    def test_negative_sigma_returns_original(self):
        img = _rand_gray()
        out = gaussian_denoise(img, sigma=-2.0)
        np.testing.assert_array_equal(out, img)

    def test_sigma_1_5_values_in_range(self):
        img = _noisy_bgr()
        out = gaussian_denoise(img, sigma=1.5)
        assert out.min() >= 0 and out.max() <= 255

    def test_very_large_sigma_constant_bgr(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = gaussian_denoise(img, sigma=20.0)
        assert out.shape == (32, 32, 3)
        np.testing.assert_array_almost_equal(out.astype(float),
                                             img.astype(float), decimal=0)

    def test_kernel_size_5_runs(self):
        img = _noisy_bgr()
        out = gaussian_denoise(img, sigma=1.0, kernel_size=5)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_kernel_size_7_runs(self):
        img = _rand_rgb()
        out = gaussian_denoise(img, sigma=1.0, kernel_size=7)
        assert out.shape == img.shape

    def test_reduces_variance_noisy_to_clean(self):
        clean = _clean_bgr()
        noisy = _noisy_bgr()
        out = gaussian_denoise(noisy, sigma=3.0)
        std_before = float(np.std(noisy.astype(np.float32) - clean.astype(np.float32)))
        std_after = float(np.std(out.astype(np.float32) - clean.astype(np.float32)))
        assert std_after < std_before

    def test_grayscale_with_sigma_2(self):
        img = _gray()
        out = gaussian_denoise(img, sigma=2.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_non_square_bgr(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 64, 3), dtype=np.uint8)
        out = gaussian_denoise(img, sigma=1.0)
        assert out.shape == (32, 64, 3)


# ─── TestMedianDenoiseExtra ───────────────────────────────────────────────────

class TestMedianDenoiseExtra:
    def test_ksize_7(self):
        img = _noisy_bgr()
        out = median_denoise(img, ksize=7)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_ksize_9(self):
        img = _rand_gray()
        out = median_denoise(img, ksize=9)
        assert out.shape == img.shape

    def test_ksize_2_rounds_to_3(self):
        img = _noisy_bgr()
        out = median_denoise(img, ksize=2)
        assert out.shape == img.shape

    def test_ksize_6_rounds_to_7(self):
        img = _noisy_bgr()
        out = median_denoise(img, ksize=6)
        out7 = median_denoise(img, ksize=7)
        assert out.shape == out7.shape

    def test_constant_bgr_preserved(self):
        img = np.full((32, 32, 3), 150, dtype=np.uint8)
        out = median_denoise(img, ksize=3)
        np.testing.assert_array_equal(out, img)

    def test_single_pixel_outlier_gray(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        img[15, 15] = 0
        out = median_denoise(img, ksize=3)
        assert abs(int(out[15, 15]) - 128) < 30

    def test_values_0_255(self):
        img = _rand_rgb()
        out = median_denoise(img, ksize=3)
        assert out.min() >= 0 and out.max() <= 255


# ─── TestBilateralDenoiseExtra ────────────────────────────────────────────────

class TestBilateralDenoiseExtra:
    def test_sigma_color_200_no_crash(self):
        img = _rand_gray()
        out = bilateral_denoise(img, sigma_color=200.0, sigma_space=200.0)
        assert out.shape == img.shape

    def test_d_1(self):
        img = _rand_rgb()
        out = bilateral_denoise(img, d=1)
        assert out.shape == img.shape

    def test_constant_image_unchanged(self):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        out = bilateral_denoise(img)
        np.testing.assert_array_equal(out, img)

    def test_step_edge_preserved(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 200
        out = bilateral_denoise(img, d=9, sigma_color=75.0, sigma_space=75.0)
        assert float(out[:, :20].mean()) < 40
        assert float(out[:, 44:].mean()) > 160

    def test_bgr_values_in_range(self):
        img = _noisy_bgr()
        out = bilateral_denoise(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_deterministic(self):
        img = _rand_rgb(seed=3)
        out1 = bilateral_denoise(img)
        out2 = bilateral_denoise(img)
        np.testing.assert_array_equal(out1, out2)


# ─── TestNlmeansDenoiseExtra ──────────────────────────────────────────────────

class TestNlmeansDenoiseExtra:
    def test_custom_h_5(self):
        img = _noisy_bgr()
        out = nlmeans_denoise(img, h=5.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_custom_h_20(self):
        img = _noisy_bgr(h=32, w=32)
        out = nlmeans_denoise(img, h=20.0)
        assert out.shape == img.shape

    def test_gray_h_10(self):
        img = _gray()
        out = nlmeans_denoise(img, h=10.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_h_5_vs_h_20_different(self):
        img = _noisy_bgr(h=32, w=32)
        out5 = nlmeans_denoise(img, h=5.0)
        out20 = nlmeans_denoise(img, h=20.0)
        # Higher h → more smoothing → lower std
        assert out20.std() <= out5.std() + 10.0

    def test_values_in_range(self):
        img = _noisy_bgr()
        out = nlmeans_denoise(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_reduces_rmse(self):
        clean = _clean_bgr()
        noisy = _noisy_bgr()
        out = nlmeans_denoise(noisy, h=15.0)
        rmse_in = float(np.sqrt(np.mean((noisy.astype(float) - clean.astype(float)) ** 2)))
        rmse_out = float(np.sqrt(np.mean((out.astype(float) - clean.astype(float)) ** 2)))
        assert rmse_out < rmse_in


# ─── TestAutoDenoiseExtra ─────────────────────────────────────────────────────

class TestAutoDenoiseExtra:
    def test_very_noisy_processed(self):
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = auto_denoise(img)
        assert out.shape == (32, 32, 3)

    def test_aggressive_false_no_crash(self):
        img = _noisy_bgr()
        out = auto_denoise(img, aggressive=False)
        assert out.shape == img.shape

    def test_gray_rgb_shape_preserved(self):
        img = _rand_rgb(h=16, w=16, seed=5)
        out = auto_denoise(img)
        assert out.shape == (16, 16, 3)

    def test_output_uint8(self):
        img = _noisy_bgr()
        out = auto_denoise(img)
        assert out.dtype == np.uint8

    def test_multiple_calls_consistent(self):
        img = _noisy_bgr(seed=7)
        out1 = auto_denoise(img)
        out2 = auto_denoise(img)
        np.testing.assert_array_equal(out1, out2)

    def test_non_square(self):
        img = np.random.default_rng(1).integers(0, 256, (24, 48, 3), dtype=np.uint8)
        out = auto_denoise(img)
        assert out.shape == (24, 48, 3)


# ─── TestDenoiseBatchExtra ────────────────────────────────────────────────────

class TestDenoiseBatchExtra:
    def test_single_image_gaussian(self):
        img = _rand_rgb()
        result = denoise_batch([img], method="gaussian")
        assert len(result) == 1
        assert result[0].shape == img.shape

    def test_bilateral_method(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        result = denoise_batch(imgs, method="bilateral")
        assert len(result) == 3

    def test_sigma_forwarded(self):
        img = _rand_rgb()
        r1 = denoise_batch([img], method="gaussian", sigma=0.5)
        r2 = denoise_batch([img], method="gaussian", sigma=3.0)
        assert r1[0].shape == r2[0].shape

    def test_none_at_start(self):
        img = _rand_rgb()
        result = denoise_batch([None, img], method="gaussian")
        assert result[0] is None
        assert result[1].shape == img.shape

    def test_all_shapes_preserved_mixed(self):
        imgs = [_rand_gray(24, 32), _rand_rgb(16, 48)]
        result = denoise_batch(imgs, method="median")
        assert result[0].shape == (24, 32)
        assert result[1].shape == (16, 48, 3)

    def test_large_batch_returns_same_count(self):
        imgs = [_rand_gray(seed=i) for i in range(8)]
        result = denoise_batch(imgs, method="gaussian", sigma=1.0)
        assert len(result) == 8
