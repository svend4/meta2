"""Extra tests for puzzle_reconstruction/utils/image_stats.py"""
import pytest
import numpy as np
import cv2

from puzzle_reconstruction.utils.image_stats import (
    ImageStats,
    compute_entropy,
    compute_sharpness,
    compute_histogram_stats,
    compute_gradient_stats,
    compute_image_stats,
    compare_images,
    batch_stats,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const(val=100, h=32, w=32) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _gradient(h=64, w=64) -> np.ndarray:
    return np.tile(np.arange(w, dtype=np.uint8), (h, 1))


# ─── TestComputeEntropyExtra ──────────────────────────────────────────────────

class TestComputeEntropyExtra:
    def test_value_255_zero_entropy(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        e = compute_entropy(img)
        assert e == pytest.approx(0.0)

    def test_value_0_zero_entropy(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        e = compute_entropy(img)
        assert e == pytest.approx(0.0)

    def test_non_square_gray(self):
        img = _gray(h=16, w=48)
        e = compute_entropy(img)
        assert 0.0 <= e <= 8.0

    def test_three_value_image(self):
        img = np.zeros((90, 90), dtype=np.uint8)
        img[:30, :] = 0
        img[30:60, :] = 128
        img[60:, :] = 255
        e = compute_entropy(img)
        # Equal thirds → log2(3) ≈ 1.585
        assert abs(e - np.log2(3.0)) < 0.05

    def test_bgr_matches_gray_entropy(self):
        gray = _gray()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        e_gray = compute_entropy(gray)
        e_bgr = compute_entropy(bgr)
        assert e_gray == pytest.approx(e_bgr, abs=1e-9)


# ─── TestComputeSharpnessExtra ────────────────────────────────────────────────

class TestComputeSharpnessExtra:
    def test_blurred_less_sharp_than_original(self):
        img = _gray()
        blurred = cv2.GaussianBlur(img, (11, 11), 2.0)
        assert compute_sharpness(blurred) < compute_sharpness(img)

    def test_gradient_image_nonzero(self):
        img = _gradient()
        assert compute_sharpness(img) > 0.0

    def test_non_square_image(self):
        img = _gray(h=16, w=48)
        s = compute_sharpness(img)
        assert s >= 0.0

    def test_all_255_zero(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        assert compute_sharpness(img) == pytest.approx(0.0)

    def test_checkerboard_high_sharpness(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        s = compute_sharpness(img)
        assert s > compute_sharpness(_const(val=128))


# ─── TestComputeHistogramStatsExtra ──────────────────────────────────────────

class TestComputeHistogramStatsExtra:
    def test_all_black_zero_std(self):
        d = compute_histogram_stats(np.zeros((32, 32), dtype=np.uint8))
        assert d["std"] == pytest.approx(0.0)
        assert d["mean"] == pytest.approx(0.0)

    def test_all_white_zero_std(self):
        d = compute_histogram_stats(np.full((32, 32), 255, dtype=np.uint8))
        assert d["std"] == pytest.approx(0.0)
        assert d["mean"] == pytest.approx(255.0)

    def test_non_square(self):
        d = compute_histogram_stats(_gray(h=16, w=48))
        assert set(d.keys()) == {"mean", "std", "skewness", "kurtosis"}

    def test_mean_matches_numpy(self):
        img = _gray()
        d = compute_histogram_stats(img)
        assert abs(d["mean"] - float(img.astype(np.float64).mean())) < 0.01


# ─── TestComputeGradientStatsExtra ────────────────────────────────────────────

class TestComputeGradientStatsExtra:
    def test_non_square_image(self):
        d = compute_gradient_stats(_gray(h=16, w=48))
        assert "grad_mean" in d
        assert d["grad_mean"] >= 0.0

    def test_constant_image_near_zero_mean(self):
        d = compute_gradient_stats(_const(val=200))
        assert d["grad_mean"] < 2.0

    def test_gradient_max_ge_mean(self):
        d = compute_gradient_stats(_gradient())
        assert d["grad_max"] >= d["grad_mean"]

    def test_energy_nonneg(self):
        d = compute_gradient_stats(_gray())
        assert d["grad_energy"] >= 0.0

    def test_std_nonneg(self):
        d = compute_gradient_stats(_gray())
        assert d["grad_std"] >= 0.0


# ─── TestComputeImageStatsExtra ───────────────────────────────────────────────

class TestComputeImageStatsExtra:
    def test_non_square_image(self):
        img = _gray(h=16, w=48)
        s = compute_image_stats(img)
        assert s.n_pixels == 16 * 48

    def test_custom_percentiles(self):
        img = _gray()
        s = compute_image_stats(img, percentile_levels=(10, 90))
        assert 10 in s.percentiles
        assert 90 in s.percentiles
        assert len(s.percentiles) == 2

    def test_percentile_10_le_50_le_90(self):
        img = _gray(seed=9)
        s = compute_image_stats(img, percentile_levels=(10, 50, 90))
        assert s.percentiles[10] <= s.percentiles[50] <= s.percentiles[90]

    def test_constant_image_entropy_zero(self):
        img = _const(val=50)
        s = compute_image_stats(img)
        assert s.entropy == pytest.approx(0.0)

    def test_contrast_equals_std(self):
        img = _gray()
        s = compute_image_stats(img)
        assert s.contrast == pytest.approx(s.std)

    def test_repr_contains_mean(self):
        s = compute_image_stats(_gray())
        assert "mean=" in repr(s)

    def test_hist_bins_128(self):
        s = compute_image_stats(_gray(), hist_bins=128)
        assert s.histogram.shape == (128,)

    def test_skewness_in_extra(self):
        s = compute_image_stats(_gray())
        assert "skewness" in s.extra

    def test_kurtosis_in_extra(self):
        s = compute_image_stats(_gray())
        assert "kurtosis" in s.extra


# ─── TestCompareImagesExtra ───────────────────────────────────────────────────

class TestCompareImagesExtra:
    def test_self_comparison_zero_mean_diff(self):
        img = _gray()
        d = compare_images(img, img)
        assert d["mean_diff"] == pytest.approx(0.0)

    def test_self_comparison_std_ratio_1(self):
        img = _gray()
        d = compare_images(img, img)
        assert d["std_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_self_comparison_entropy_diff_zero(self):
        img = _gray()
        d = compare_images(img, img)
        assert d["entropy_diff"] == pytest.approx(0.0)

    def test_hist_bhatt_nonneg(self):
        img1, img2 = _gray(seed=3), _gray(seed=7)
        d = compare_images(img1, img2)
        assert d["hist_bhatt"] >= 0.0

    def test_hist_corr_range(self):
        img1, img2 = _gray(seed=0), _gray(seed=1)
        d = compare_images(img1, img2)
        assert -1.0 <= d["hist_corr"] <= 1.0

    def test_sharpness_ratio_positive(self):
        img1, img2 = _gray(seed=0), _gray(seed=2)
        d = compare_images(img1, img2)
        assert d["sharpness_ratio"] >= 0.0

    def test_constant_vs_noisy_entropy_diff(self):
        flat = _const(val=128)
        noisy = _gray()
        d = compare_images(flat, noisy)
        assert abs(d["entropy_diff"]) > 0.0


# ─── TestBatchStatsExtra ──────────────────────────────────────────────────────

class TestBatchStatsExtra:
    def test_single_image(self):
        result = batch_stats([_gray()])
        assert len(result) == 1
        assert isinstance(result[0], ImageStats)

    def test_bgr_images(self):
        result = batch_stats([_bgr(seed=i) for i in range(2)])
        assert len(result) == 2
        assert all(r.n_pixels == 64 * 64 for r in result)

    def test_with_gradient(self):
        result = batch_stats([_gray()], include_gradient=True)
        assert "grad_mean" in result[0].extra

    def test_hist_bins_16(self):
        result = batch_stats([_gray()], hist_bins=16)
        assert result[0].histogram.shape == (16,)

    def test_mean_in_range(self):
        result = batch_stats([_gray(seed=i) for i in range(4)])
        assert all(0.0 <= r.mean <= 255.0 for r in result)

    def test_non_square_images(self):
        result = batch_stats([_gray(h=20, w=60)])
        assert result[0].n_pixels == 20 * 60
