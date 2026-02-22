"""Tests for puzzle_reconstruction/utils/image_stats.py"""
import pytest
import numpy as np

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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, seed=0, val=None):
    if val is not None:
        return np.full((h, w), val, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_gradient_img(h=64, w=64):
    """Image with strong horizontal gradient."""
    arr = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    return arr


# ─── compute_entropy ─────────────────────────────────────────────────────────

class TestComputeEntropy:
    def test_uniform_image_max_entropy(self):
        """All 256 values equally likely → close to 8 bits."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        e = compute_entropy(img)
        assert e > 7.0  # close to 8

    def test_constant_image_zero_entropy(self):
        img = make_gray(val=128)
        e = compute_entropy(img)
        assert e == 0.0

    def test_two_value_image(self):
        """50/50 split → entropy = 1 bit."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 255
        e = compute_entropy(img)
        assert abs(e - 1.0) < 0.01

    def test_bgr_accepted(self):
        img = make_bgr()
        e = compute_entropy(img)
        assert e >= 0.0

    def test_result_in_range(self):
        for seed in range(5):
            img = make_gray(seed=seed)
            e = compute_entropy(img)
            assert 0.0 <= e <= 8.0


# ─── compute_sharpness ───────────────────────────────────────────────────────

class TestComputeSharpness:
    def test_constant_image_zero_sharpness(self):
        img = make_gray(val=128)
        s = compute_sharpness(img)
        assert s == 0.0

    def test_gradient_image_high_sharpness(self):
        img = make_gradient_img()
        s = compute_sharpness(img)
        assert s > 0.0

    def test_noisy_image_high_sharpness(self):
        img = make_gray(seed=42)
        s = compute_sharpness(img)
        assert s > 0.0

    def test_bgr_accepted(self):
        img = make_bgr()
        s = compute_sharpness(img)
        assert s >= 0.0

    def test_non_negative(self):
        for seed in range(5):
            img = make_gray(seed=seed)
            assert compute_sharpness(img) >= 0.0

    def test_blurry_vs_sharp(self):
        """Noisy image should have higher sharpness than constant."""
        sharp = make_gray(seed=99)
        flat = make_gray(val=100)
        assert compute_sharpness(sharp) >= compute_sharpness(flat)


# ─── compute_histogram_stats ─────────────────────────────────────────────────

class TestComputeHistogramStats:
    def test_returns_dict_with_keys(self):
        img = make_gray()
        d = compute_histogram_stats(img)
        assert set(d.keys()) == {"mean", "std", "skewness", "kurtosis"}

    def test_constant_image_zero_std(self):
        img = make_gray(val=128)
        d = compute_histogram_stats(img)
        assert d["std"] == 0.0
        assert d["skewness"] == 0.0
        assert d["kurtosis"] == 0.0

    def test_mean_in_range(self):
        img = make_gray()
        d = compute_histogram_stats(img)
        assert 0.0 <= d["mean"] <= 255.0

    def test_symmetric_distribution_skew_near_zero(self):
        """Uniform distribution → skewness ≈ 0."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        d = compute_histogram_stats(img)
        assert abs(d["skewness"]) < 0.5  # roughly symmetric

    def test_bgr_accepted(self):
        img = make_bgr()
        d = compute_histogram_stats(img)
        assert "mean" in d

    def test_std_non_negative(self):
        img = make_gray()
        d = compute_histogram_stats(img)
        assert d["std"] >= 0.0


# ─── compute_gradient_stats ──────────────────────────────────────────────────

class TestComputeGradientStats:
    def test_returns_dict_with_keys(self):
        img = make_gray()
        d = compute_gradient_stats(img)
        assert set(d.keys()) == {"grad_mean", "grad_std", "grad_max", "grad_energy"}

    def test_constant_image_zero_gradient(self):
        img = make_gray(val=128)
        d = compute_gradient_stats(img)
        # Edge effects may give small non-zero values, but mean should be small
        assert d["grad_mean"] < 1.0

    def test_gradient_image_high_gradient(self):
        img = make_gradient_img()
        d = compute_gradient_stats(img)
        assert d["grad_mean"] > 0.0

    def test_all_values_non_negative(self):
        img = make_gray()
        d = compute_gradient_stats(img)
        for v in d.values():
            assert v >= 0.0

    def test_grad_max_ge_mean(self):
        img = make_gray()
        d = compute_gradient_stats(img)
        assert d["grad_max"] >= d["grad_mean"]

    def test_bgr_accepted(self):
        img = make_bgr()
        d = compute_gradient_stats(img)
        assert "grad_mean" in d


# ─── compute_image_stats ─────────────────────────────────────────────────────

class TestComputeImageStats:
    def test_returns_image_stats(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert isinstance(s, ImageStats)

    def test_histogram_shape_default(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert s.histogram.shape == (256,)

    def test_histogram_shape_custom_bins(self):
        img = make_gray()
        s = compute_image_stats(img, hist_bins=64)
        assert s.histogram.shape == (64,)

    def test_histogram_sums_to_one(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert abs(s.histogram.sum() - 1.0) < 1e-5

    def test_n_pixels_correct(self):
        img = make_gray(64, 64)
        s = compute_image_stats(img)
        assert s.n_pixels == 64 * 64

    def test_percentiles_present(self):
        img = make_gray()
        s = compute_image_stats(img, percentile_levels=(5, 50, 95))
        assert 5 in s.percentiles
        assert 50 in s.percentiles
        assert 95 in s.percentiles

    def test_percentile_ordering(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert s.percentiles[5] <= s.percentiles[25] <= s.percentiles[50]
        assert s.percentiles[50] <= s.percentiles[75] <= s.percentiles[95]

    def test_mean_in_range(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert 0.0 <= s.mean <= 255.0

    def test_std_non_negative(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert s.std >= 0.0

    def test_entropy_in_range(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert 0.0 <= s.entropy <= 8.0

    def test_sharpness_non_negative(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert s.sharpness >= 0.0

    def test_contrast_equals_std(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert abs(s.contrast - s.std) < 1e-9

    def test_extra_has_gradient_by_default(self):
        img = make_gray()
        s = compute_image_stats(img, include_gradient=True)
        assert "grad_mean" in s.extra

    def test_extra_no_gradient_when_disabled(self):
        img = make_gray()
        s = compute_image_stats(img, include_gradient=False)
        assert "grad_mean" not in s.extra

    def test_extra_has_skewness_kurtosis(self):
        img = make_gray()
        s = compute_image_stats(img)
        assert "skewness" in s.extra
        assert "kurtosis" in s.extra

    def test_bgr_image(self):
        img = make_bgr()
        s = compute_image_stats(img)
        assert s.n_pixels == 64 * 64

    def test_repr_contains_info(self):
        img = make_gray()
        s = compute_image_stats(img)
        r = repr(s)
        assert "ImageStats" in r
        assert "mean=" in r


# ─── compare_images ───────────────────────────────────────────────────────────

class TestCompareImages:
    def test_identical_images(self):
        img = make_gray()
        result = compare_images(img, img)
        assert abs(result["mean_diff"]) < 1e-9
        assert abs(result["entropy_diff"]) < 1e-9

    def test_returns_dict_with_keys(self):
        img1 = make_gray(seed=0)
        img2 = make_gray(seed=1)
        d = compare_images(img1, img2)
        assert "mean_diff" in d
        assert "std_ratio" in d
        assert "entropy_diff" in d
        assert "sharpness_ratio" in d
        assert "hist_corr" in d
        assert "hist_bhatt" in d

    def test_hist_corr_identical(self):
        img = make_gray()
        d = compare_images(img, img)
        assert abs(d["hist_corr"] - 1.0) < 0.01

    def test_hist_bhatt_identical(self):
        img = make_gray()
        d = compare_images(img, img)
        assert d["hist_bhatt"] < 0.01

    def test_hist_corr_in_range(self):
        img1 = make_gray(seed=0)
        img2 = make_gray(seed=5)
        d = compare_images(img1, img2)
        assert -1.0 <= d["hist_corr"] <= 1.0

    def test_hist_bhatt_non_negative(self):
        img1 = make_gray(seed=0)
        img2 = make_gray(seed=5)
        d = compare_images(img1, img2)
        assert d["hist_bhatt"] >= 0.0

    def test_constant_vs_uniform(self):
        flat = make_gray(val=128)
        rand = make_gray(seed=42)
        d = compare_images(flat, rand)
        # Entropies should differ
        assert abs(d["entropy_diff"]) > 0.0

    def test_std_ratio_positive(self):
        img1 = make_gray(seed=1)
        img2 = make_gray(seed=2)
        d = compare_images(img1, img2)
        assert d["std_ratio"] > 0.0


# ─── batch_stats ─────────────────────────────────────────────────────────────

class TestBatchStats:
    def test_returns_list(self):
        images = [make_gray(seed=i) for i in range(3)]
        results = batch_stats(images)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_image_stats(self):
        images = [make_gray(seed=i) for i in range(4)]
        results = batch_stats(images)
        assert all(isinstance(r, ImageStats) for r in results)

    def test_empty_list(self):
        results = batch_stats([])
        assert results == []

    def test_no_gradient_by_default(self):
        images = [make_gray()]
        results = batch_stats(images, include_gradient=False)
        assert "grad_mean" not in results[0].extra

    def test_custom_bins(self):
        images = [make_gray()]
        results = batch_stats(images, hist_bins=32)
        assert results[0].histogram.shape == (32,)

    def test_n_pixels_consistent(self):
        images = [make_gray(50, 80)]
        results = batch_stats(images)
        assert results[0].n_pixels == 50 * 80
