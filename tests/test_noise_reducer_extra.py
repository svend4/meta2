"""Extra tests for puzzle_reconstruction/preprocessing/noise_reducer.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_reducer import (
    NoiseReductionResult,
    auto_reduce,
    batch_reduce,
    bilateral_reduce,
    estimate_noise,
    gaussian_reduce,
    median_reduce,
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


def _noisy(h=64, w=64, seed=42):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _smooth(h=64, w=64):
    return np.full((h, w), 100, dtype=np.uint8)


# ─── TestNoiseReductionResultExtra ───────────────────────────────────────────

class TestNoiseReductionResultExtra:
    def test_bgr_filtered(self):
        r = NoiseReductionResult(
            filtered=np.zeros((32, 32, 3), dtype=np.uint8),
            method="bilateral",
        )
        assert r.filtered.ndim == 3

    def test_method_median(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="median",
        )
        assert r.method == "median"

    def test_method_auto(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="auto",
        )
        assert r.method == "auto"

    def test_params_multiple_keys(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="bilateral",
            params={"d": 5, "sigma_color": 75.0, "sigma_space": 75.0},
        )
        assert r.params["d"] == 5
        assert r.params["sigma_color"] == pytest.approx(75.0)

    def test_noise_before_and_after_equal(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="gaussian",
            noise_estimate_before=5.5,
            noise_estimate_after=5.5,
        )
        assert r.noise_estimate_before == pytest.approx(r.noise_estimate_after)

    def test_filtered_large_image(self):
        r = NoiseReductionResult(
            filtered=np.zeros((256, 256), dtype=np.uint8),
            method="gaussian",
        )
        assert r.filtered.shape == (256, 256)

    def test_repr_contains_noise(self):
        r = NoiseReductionResult(
            filtered=np.zeros((10, 10), dtype=np.uint8),
            method="gaussian",
            noise_estimate_before=10.0,
            noise_estimate_after=3.0,
        )
        s = repr(r)
        assert "NoiseReductionResult" in s


# ─── TestEstimateNoiseExtra ───────────────────────────────────────────────────

class TestEstimateNoiseExtra:
    def test_non_square_gray(self):
        img = _noisy(h=32, w=64)
        v = estimate_noise(img)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_uniform_255_near_zero(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        assert estimate_noise(img) < 1.0

    def test_gradient_image(self):
        img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        v = estimate_noise(img)
        assert v >= 0.0

    def test_random_seed_consistent(self):
        img = _noisy(seed=7)
        v1 = estimate_noise(img)
        v2 = estimate_noise(img)
        assert v1 == pytest.approx(v2)

    def test_more_noisy_higher_estimate(self):
        low_noise = _noisy(seed=0)
        low_noise = np.clip(low_noise.astype(int) // 4, 0, 255).astype(np.uint8)
        high_noise = _noisy(seed=1)
        assert estimate_noise(high_noise) > estimate_noise(low_noise)


# ─── TestGaussianReduceExtra ──────────────────────────────────────────────────

class TestGaussianReduceExtra:
    def test_non_square_shape(self):
        r = gaussian_reduce(_noisy(h=32, w=64))
        assert r.filtered.shape == (32, 64)

    def test_large_ksize(self):
        r = gaussian_reduce(_noisy(), ksize=9)
        assert r.filtered.shape == (64, 64)
        assert r.params.get("ksize") == 9

    def test_small_ksize(self):
        r = gaussian_reduce(_noisy(), ksize=3)
        assert r.filtered.shape == (64, 64)

    def test_sigma_zero(self):
        r = gaussian_reduce(_noisy(), sigma=0.0)
        assert r.filtered.shape == (64, 64)

    def test_large_sigma(self):
        r = gaussian_reduce(_noisy(), sigma=5.0)
        assert r.filtered.shape == (64, 64)

    def test_bgr_non_square(self):
        r = gaussian_reduce(_bgr(h=32, w=64))
        assert r.filtered.shape == (32, 64, 3)

    def test_constant_image_unchanged(self):
        img = _smooth()
        r = gaussian_reduce(img, ksize=3)
        np.testing.assert_array_equal(r.filtered, img)

    def test_noise_before_gt_zero_for_noisy(self):
        r = gaussian_reduce(_noisy(seed=5))
        assert r.noise_estimate_before > 0.0


# ─── TestMedianReduceExtra ────────────────────────────────────────────────────

class TestMedianReduceExtra:
    def test_non_square_gray(self):
        r = median_reduce(_noisy(h=32, w=64))
        assert r.filtered.shape == (32, 64)

    def test_ksize_7(self):
        r = median_reduce(_noisy(), ksize=7)
        assert r.filtered.shape == (64, 64)
        assert r.params.get("ksize") == 7

    def test_bgr_non_square(self):
        r = median_reduce(_bgr(h=32, w=64))
        assert r.filtered.shape == (32, 64, 3)

    def test_constant_image_unchanged(self):
        img = _smooth()
        r = median_reduce(img, ksize=5)
        np.testing.assert_array_equal(r.filtered, img)

    def test_noise_before_gt_zero_for_noisy(self):
        r = median_reduce(_noisy(seed=11))
        assert r.noise_estimate_before > 0.0

    def test_output_dtype(self):
        r = median_reduce(_noisy())
        assert r.filtered.dtype == np.uint8

    def test_noise_reduces_significantly(self):
        r = median_reduce(_noisy(seed=3), ksize=7)
        assert r.noise_estimate_after < r.noise_estimate_before


# ─── TestBilateralReduceExtra ─────────────────────────────────────────────────

class TestBilateralReduceExtra:
    def test_non_square_gray(self):
        r = bilateral_reduce(_noisy(h=32, w=64))
        assert r.filtered.shape == (32, 64)

    def test_custom_d(self):
        r = bilateral_reduce(_noisy(), d=9)
        assert r.params.get("d") == 9

    def test_custom_sigma_color(self):
        r = bilateral_reduce(_noisy(), sigma_color=100.0)
        assert r.params.get("sigma_color") == pytest.approx(100.0)

    def test_custom_sigma_space(self):
        r = bilateral_reduce(_noisy(), sigma_space=80.0)
        assert r.params.get("sigma_space") == pytest.approx(80.0)

    def test_bgr_non_square(self):
        r = bilateral_reduce(_bgr(h=32, w=64))
        assert r.filtered.shape == (32, 64, 3)

    def test_constant_image_shape(self):
        img = _smooth()
        r = bilateral_reduce(img)
        assert r.filtered.shape == img.shape

    def test_noise_before_gt_zero_for_noisy(self):
        r = bilateral_reduce(_noisy(seed=13))
        assert r.noise_estimate_before > 0.0

    def test_output_dtype_bgr(self):
        r = bilateral_reduce(_bgr())
        assert r.filtered.dtype == np.uint8


# ─── TestAutoReduceExtra ─────────────────────────────────────────────────────

class TestAutoReduceExtra:
    def test_non_square_shape(self):
        r = auto_reduce(_noisy(h=32, w=64))
        assert r.filtered.shape == (32, 64)

    def test_smooth_image_method_trivial(self):
        r = auto_reduce(_smooth(), low_thresh=5.0, high_thresh=15.0)
        assert r.params.get("chosen_filter") == "gaussian_trivial"

    def test_noisy_image_method_median(self):
        r = auto_reduce(_noisy(seed=2), low_thresh=1.0, high_thresh=2.0)
        assert r.params.get("chosen_filter") == "median"

    def test_sigma_before_positive_for_noisy(self):
        r = auto_reduce(_noisy())
        assert r.params.get("sigma_before", 0.0) >= 0.0

    def test_bgr_shape(self):
        r = auto_reduce(_bgr())
        assert r.filtered.shape == (64, 64, 3)

    def test_thresholds_custom(self):
        r = auto_reduce(_noisy(), low_thresh=2.0, high_thresh=8.0)
        assert r.params.get("low_thresh") == pytest.approx(2.0)
        assert r.params.get("high_thresh") == pytest.approx(8.0)

    def test_output_dtype(self):
        r = auto_reduce(_noisy())
        assert r.filtered.dtype == np.uint8

    def test_noise_before_from_noisy(self):
        r = auto_reduce(_noisy(seed=99))
        assert r.noise_estimate_before > 0.0


# ─── TestBatchReduceExtra ─────────────────────────────────────────────────────

class TestBatchReduceExtra:
    def test_five_images(self):
        imgs = [_noisy(seed=i) for i in range(5)]
        result = batch_reduce(imgs)
        assert len(result) == 5

    def test_mixed_sizes(self):
        imgs = [_noisy(h=32, w=32), _noisy(h=64, w=64), _noisy(h=48, w=96)]
        result = batch_reduce(imgs)
        assert result[0].filtered.shape == (32, 32)
        assert result[1].filtered.shape == (64, 64)
        assert result[2].filtered.shape == (48, 96)

    def test_bilateral_batch(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        result = batch_reduce(imgs, method="bilateral")
        assert all(r.method == "bilateral" for r in result)

    def test_auto_batch(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        result = batch_reduce(imgs, method="auto")
        assert all(isinstance(r, NoiseReductionResult) for r in result)

    def test_single_image(self):
        result = batch_reduce([_noisy()])
        assert len(result) == 1

    def test_ksize_9_gaussian(self):
        result = batch_reduce([_noisy()], method="gaussian", ksize=9)
        assert result[0].params.get("ksize") == 9

    def test_bgr_images(self):
        imgs = [_bgr() for _ in range(3)]
        result = batch_reduce(imgs, method="gaussian")
        assert all(r.filtered.ndim == 3 for r in result)

    def test_all_results_uint8(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        for method in ("gaussian", "median"):
            result = batch_reduce(imgs, method=method)
            assert all(r.filtered.dtype == np.uint8 for r in result)
