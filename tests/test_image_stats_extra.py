"""Extra tests for puzzle_reconstruction/utils/image_stats.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.image_stats import (
    ImageStats,
    batch_stats,
    compare_images,
    compute_entropy,
    compute_gradient_stats,
    compute_histogram_stats,
    compute_image_stats,
    compute_sharpness,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h: int = 64, w: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 100
    img[:, :, 2] = 50
    return img


def _ramp(h: int = 64, w: int = 64) -> np.ndarray:
    col = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(col.astype(np.uint8), (h, 1))


# ─── compute_entropy (extra) ──────────────────────────────────────────────────

class TestComputeEntropyExtra:
    def test_constant_image_zero_entropy(self):
        assert compute_entropy(_gray()) == pytest.approx(0.0, abs=1e-6)

    def test_noisy_image_high_entropy(self):
        e = compute_entropy(_noisy())
        assert e > 4.0

    def test_returns_float(self):
        assert isinstance(compute_entropy(_gray()), float)

    def test_range_0_to_8(self):
        e = compute_entropy(_noisy())
        assert 0.0 <= e <= 8.0

    def test_bgr_input(self):
        e = compute_entropy(_bgr())
        assert isinstance(e, float)

    def test_two_value_image_entropy_1(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:32, :] = 128
        e = compute_entropy(img)
        assert e == pytest.approx(1.0, abs=0.1)

    def test_ramp_image_higher_than_constant(self):
        e_ramp = compute_entropy(_ramp())
        e_const = compute_entropy(_gray())
        assert e_ramp > e_const

    def test_nonneg(self):
        assert compute_entropy(_noisy()) >= 0.0

    def test_small_image(self):
        img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        e = compute_entropy(img)
        assert e >= 0.0


# ─── compute_sharpness (extra) ────────────────────────────────────────────────

class TestComputeSharpnessExtra:
    def test_constant_image_zero_sharpness(self):
        assert compute_sharpness(_gray()) == pytest.approx(0.0, abs=1e-4)

    def test_noisy_image_high_sharpness(self):
        s = compute_sharpness(_noisy())
        assert s > 100.0

    def test_returns_float(self):
        assert isinstance(compute_sharpness(_noisy()), float)

    def test_nonneg(self):
        assert compute_sharpness(_noisy()) >= 0.0

    def test_bgr_input(self):
        s = compute_sharpness(_bgr())
        assert isinstance(s, float)

    def test_ramp_has_some_sharpness(self):
        # Ramp has edges at borders
        s = compute_sharpness(_ramp())
        assert s >= 0.0

    def test_more_noise_more_sharpness(self):
        rng = np.random.default_rng(0)
        img_lo = (rng.integers(100, 156, (64, 64))).astype(np.uint8)
        img_hi = (rng.integers(0, 256, (64, 64))).astype(np.uint8)
        assert compute_sharpness(img_hi) >= compute_sharpness(img_lo)


# ─── compute_histogram_stats (extra) ──────────────────────────────────────────

class TestComputeHistogramStatsExtra:
    def test_returns_dict(self):
        d = compute_histogram_stats(_gray())
        assert isinstance(d, dict)

    def test_has_all_keys(self):
        d = compute_histogram_stats(_noisy())
        for k in ("mean", "std", "skewness", "kurtosis"):
            assert k in d

    def test_constant_image_zero_std(self):
        d = compute_histogram_stats(_gray(val=100))
        assert d["std"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_image_zero_skewness(self):
        d = compute_histogram_stats(_gray())
        assert d["skewness"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_image_zero_kurtosis(self):
        d = compute_histogram_stats(_gray())
        assert d["kurtosis"] == pytest.approx(0.0, abs=1e-6)

    def test_mean_in_range(self):
        d = compute_histogram_stats(_noisy())
        assert 0.0 <= d["mean"] <= 255.0

    def test_std_nonneg(self):
        d = compute_histogram_stats(_noisy())
        assert d["std"] >= 0.0

    def test_bgr_input(self):
        d = compute_histogram_stats(_bgr())
        assert "mean" in d

    def test_custom_bins(self):
        d = compute_histogram_stats(_noisy(), bins=64)
        assert "mean" in d

    def test_values_are_floats(self):
        d = compute_histogram_stats(_noisy())
        for v in d.values():
            assert isinstance(v, float)


# ─── compute_gradient_stats (extra) ───────────────────────────────────────────

class TestComputeGradientStatsExtra:
    def test_returns_dict(self):
        d = compute_gradient_stats(_gray())
        assert isinstance(d, dict)

    def test_has_all_keys(self):
        d = compute_gradient_stats(_noisy())
        for k in ("grad_mean", "grad_std", "grad_max", "grad_energy"):
            assert k in d

    def test_constant_image_zero_grad(self):
        d = compute_gradient_stats(_gray())
        assert d["grad_mean"] == pytest.approx(0.0, abs=1.0)

    def test_noisy_nonzero_gradient(self):
        d = compute_gradient_stats(_noisy())
        assert d["grad_mean"] > 0.0

    def test_grad_max_geq_mean(self):
        d = compute_gradient_stats(_noisy())
        assert d["grad_max"] >= d["grad_mean"]

    def test_all_nonneg(self):
        d = compute_gradient_stats(_noisy())
        for v in d.values():
            assert v >= 0.0

    def test_bgr_input(self):
        d = compute_gradient_stats(_bgr())
        assert "grad_mean" in d

    def test_values_are_floats(self):
        d = compute_gradient_stats(_noisy())
        for v in d.values():
            assert isinstance(v, float)


# ─── compute_image_stats (extra) ──────────────────────────────────────────────

class TestComputeImageStatsExtra:
    def test_returns_image_stats(self):
        s = compute_image_stats(_gray())
        assert isinstance(s, ImageStats)

    def test_mean_in_range(self):
        s = compute_image_stats(_gray(val=100))
        assert 0.0 <= s.mean <= 255.0

    def test_std_nonneg(self):
        s = compute_image_stats(_noisy())
        assert s.std >= 0.0

    def test_contrast_equals_std(self):
        s = compute_image_stats(_noisy())
        assert s.contrast == pytest.approx(s.std, abs=1e-6)

    def test_entropy_in_range(self):
        s = compute_image_stats(_noisy())
        assert 0.0 <= s.entropy <= 8.0

    def test_sharpness_nonneg(self):
        s = compute_image_stats(_noisy())
        assert s.sharpness >= 0.0

    def test_histogram_length(self):
        s = compute_image_stats(_gray(), hist_bins=256)
        assert len(s.histogram) == 256

    def test_histogram_sums_to_one(self):
        s = compute_image_stats(_noisy(), hist_bins=256)
        assert s.histogram.sum() == pytest.approx(1.0, abs=1e-4)

    def test_histogram_custom_bins(self):
        s = compute_image_stats(_gray(), hist_bins=64)
        assert len(s.histogram) == 64

    def test_percentiles_keys(self):
        s = compute_image_stats(_noisy())
        for p in (5, 25, 50, 75, 95):
            assert p in s.percentiles

    def test_percentiles_ordered(self):
        s = compute_image_stats(_ramp())
        assert s.percentiles[25] <= s.percentiles[50] <= s.percentiles[75]

    def test_n_pixels(self):
        s = compute_image_stats(_gray(32, 48))
        assert s.n_pixels == 32 * 48

    def test_include_gradient_true(self):
        s = compute_image_stats(_noisy(), include_gradient=True)
        assert "grad_mean" in s.extra

    def test_include_gradient_false(self):
        s = compute_image_stats(_noisy(), include_gradient=False)
        assert "grad_mean" not in s.extra

    def test_extra_has_skewness_kurtosis(self):
        s = compute_image_stats(_noisy())
        assert "skewness" in s.extra
        assert "kurtosis" in s.extra

    def test_bgr_input(self):
        s = compute_image_stats(_bgr())
        assert isinstance(s, ImageStats)

    def test_constant_image_zero_entropy(self):
        s = compute_image_stats(_gray())
        assert s.entropy == pytest.approx(0.0, abs=1e-5)


# ─── compare_images (extra) ───────────────────────────────────────────────────

class TestCompareImagesExtra:
    def test_returns_dict(self):
        d = compare_images(_gray(), _noisy())
        assert isinstance(d, dict)

    def test_has_all_keys(self):
        d = compare_images(_gray(), _noisy())
        for k in ("mean_diff", "std_ratio", "entropy_diff",
                  "sharpness_ratio", "hist_corr", "hist_bhatt"):
            assert k in d

    def test_identical_images_zero_mean_diff(self):
        img = _noisy()
        d = compare_images(img, img)
        assert d["mean_diff"] == pytest.approx(0.0, abs=1e-4)

    def test_identical_images_std_ratio_one(self):
        img = _noisy()
        d = compare_images(img, img)
        assert d["std_ratio"] == pytest.approx(1.0, abs=1e-4)

    def test_identical_images_hist_corr_one(self):
        img = _noisy()
        d = compare_images(img, img)
        assert d["hist_corr"] == pytest.approx(1.0, abs=0.01)

    def test_identical_images_bhatt_zero(self):
        img = _noisy()
        d = compare_images(img, img)
        assert d["hist_bhatt"] == pytest.approx(0.0, abs=0.01)

    def test_hist_corr_range(self):
        d = compare_images(_gray(), _noisy())
        assert -1.0 <= d["hist_corr"] <= 1.0

    def test_hist_bhatt_nonneg(self):
        d = compare_images(_gray(), _noisy())
        assert d["hist_bhatt"] >= 0.0

    def test_values_are_floats(self):
        d = compare_images(_gray(), _noisy())
        for v in d.values():
            assert isinstance(v, float)

    def test_bgr_inputs(self):
        d = compare_images(_bgr(), _bgr())
        assert isinstance(d, dict)


# ─── batch_stats (extra) ──────────────────────────────────────────────────────

class TestBatchStatsExtra:
    def test_empty_list_returns_empty(self):
        assert batch_stats([]) == []

    def test_single_image(self):
        results = batch_stats([_gray()])
        assert len(results) == 1
        assert isinstance(results[0], ImageStats)

    def test_multiple_images(self):
        imgs = [_gray(), _noisy(), _bgr()]
        results = batch_stats(imgs)
        assert len(results) == 3

    def test_all_are_image_stats(self):
        results = batch_stats([_gray(), _noisy()])
        for r in results:
            assert isinstance(r, ImageStats)

    def test_custom_hist_bins(self):
        results = batch_stats([_gray()], hist_bins=64)
        assert len(results[0].histogram) == 64

    def test_include_gradient_false(self):
        results = batch_stats([_noisy()], include_gradient=False)
        assert "grad_mean" not in results[0].extra

    def test_include_gradient_true(self):
        results = batch_stats([_noisy()], include_gradient=True)
        assert "grad_mean" in results[0].extra

    def test_n_pixels_correct(self):
        results = batch_stats([_gray(32, 48)])
        assert results[0].n_pixels == 32 * 48
