"""Extra tests for puzzle_reconstruction.preprocessing.noise_reducer."""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.noise_reducer import (
    NoiseReductionResult,
    estimate_noise,
    gaussian_reduce,
    median_reduce,
    bilateral_reduce,
    auto_reduce,
    batch_reduce,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_impulse_noisy(h=64, w=64, seed=0):
    base = np.full((h, w), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    n_noise = h * w // 2
    rows = rng.integers(0, h, n_noise)
    cols = rng.integers(0, w, n_noise)
    vals = rng.choice([0, 255], n_noise).astype(np.uint8)
    base[rows, cols] = vals
    return base


# ─── TestNoiseReductionResultExtra ───────────────────────────────────────────

class TestNoiseReductionResultExtra:
    def test_method_stored(self):
        r = NoiseReductionResult(filtered=make_gray(), method="bilateral")
        assert r.method == "bilateral"

    def test_default_noise_estimates(self):
        r = NoiseReductionResult(filtered=make_gray(), method="gaussian")
        assert r.noise_estimate_before == pytest.approx(0.0)
        assert r.noise_estimate_after == pytest.approx(0.0)

    def test_custom_params(self):
        r = NoiseReductionResult(filtered=make_gray(), method="median",
                                  params={"ksize": 3, "mode": "reflect"})
        assert r.params["ksize"] == 3
        assert r.params["mode"] == "reflect"

    def test_filtered_shape(self):
        img = make_gray(24, 36)
        r = NoiseReductionResult(filtered=img, method="gaussian")
        assert r.filtered.shape == (24, 36)

    def test_noise_estimates_custom(self):
        r = NoiseReductionResult(filtered=make_gray(), method="auto",
                                  noise_estimate_before=25.0,
                                  noise_estimate_after=8.0)
        assert r.noise_estimate_before == pytest.approx(25.0)
        assert r.noise_estimate_after == pytest.approx(8.0)


# ─── TestEstimateNoiseExtra ──────────────────────────────────────────────────

class TestEstimateNoiseExtra:
    def test_all_zeros(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        val = estimate_noise(img)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_all_255(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        val = estimate_noise(img)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_gradient_image(self):
        img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        val = estimate_noise(img)
        assert isinstance(val, float)

    def test_large_image(self):
        img = make_noisy(256, 256)
        val = estimate_noise(img)
        assert val > 0.0

    def test_small_image(self):
        img = make_noisy(8, 8)
        val = estimate_noise(img)
        assert val >= 0.0

    def test_bgr_noise_positive(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        val = estimate_noise(img)
        assert val > 0.0


# ─── TestGaussianReduceExtra ────────────────────────────────────────────────

class TestGaussianReduceExtra:
    def test_large_ksize(self):
        img = make_noisy()
        r = gaussian_reduce(img, ksize=15)
        assert r.filtered.shape == img.shape

    def test_ksize_3(self):
        img = make_noisy()
        r = gaussian_reduce(img, ksize=3)
        assert r.params["ksize"] == 3

    def test_bgr_shape_preserved(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3),
                                                  dtype=np.uint8)
        r = gaussian_reduce(img)
        assert r.filtered.shape == (32, 32, 3)

    def test_constant_image_unchanged(self):
        img = make_gray(h=64, w=64, fill=100)
        r = gaussian_reduce(img, ksize=5)
        np.testing.assert_array_equal(r.filtered, img)

    def test_noise_after_le_before(self):
        img = make_noisy(seed=42)
        r = gaussian_reduce(img, ksize=11)
        assert r.noise_estimate_after <= r.noise_estimate_before + 1.0

    def test_rectangular_image(self):
        img = make_noisy(h=32, w=96)
        r = gaussian_reduce(img)
        assert r.filtered.shape == (32, 96)


# ─── TestMedianReduceExtra ──────────────────────────────────────────────────

class TestMedianReduceExtra:
    def test_large_ksize(self):
        img = make_noisy()
        r = median_reduce(img, ksize=7)
        assert r.params["ksize"] == 7

    def test_bgr_accepted(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3),
                                                  dtype=np.uint8)
        r = median_reduce(img)
        assert r.filtered.shape == (32, 32, 3)

    def test_constant_image_unchanged(self):
        img = make_gray(h=64, w=64, fill=200)
        r = median_reduce(img, ksize=3)
        np.testing.assert_array_equal(r.filtered, img)

    def test_impulse_noise_well_filtered(self):
        img = make_impulse_noisy(seed=7)
        r = median_reduce(img, ksize=5)
        assert r.noise_estimate_after < r.noise_estimate_before

    def test_rectangular_image(self):
        img = make_noisy(h=24, w=80)
        r = median_reduce(img)
        assert r.filtered.shape == (24, 80)

    def test_dtype(self):
        r = median_reduce(make_noisy())
        assert r.filtered.dtype == np.uint8


# ─── TestBilateralReduceExtra ───────────────────────────────────────────────

class TestBilateralReduceExtra:
    def test_sigma_space(self):
        img = make_noisy()
        r = bilateral_reduce(img, sigma_space=100.0)
        assert r.params["sigma_space"] == pytest.approx(100.0)

    def test_d_param(self):
        img = make_noisy()
        r = bilateral_reduce(img, d=9)
        assert r.params["d"] == 9

    def test_bgr_accepted(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3),
                                                  dtype=np.uint8)
        r = bilateral_reduce(img)
        assert r.filtered.shape == (32, 32, 3)

    def test_rectangular_image(self):
        img = make_noisy(h=48, w=24)
        r = bilateral_reduce(img)
        assert r.filtered.shape == (48, 24)

    def test_constant_image(self):
        img = make_gray(fill=150)
        r = bilateral_reduce(img)
        np.testing.assert_array_equal(r.filtered, img)


# ─── TestAutoReduceExtra ────────────────────────────────────────────────────

class TestAutoReduceExtra:
    def test_moderate_noise(self):
        rng = np.random.default_rng(0)
        img = np.clip(make_gray(fill=128).astype(np.int16) +
                       rng.integers(-20, 21, (64, 64)), 0, 255).astype(np.uint8)
        r = auto_reduce(img, low_thresh=5.0, high_thresh=20.0)
        assert isinstance(r, NoiseReductionResult)

    def test_bgr_image(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        r = auto_reduce(img)
        assert r.filtered.shape == (64, 64, 3)

    def test_params_contain_noise_estimate(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert "chosen_filter" in r.params

    def test_rectangular_image(self):
        img = make_noisy(h=32, w=96)
        r = auto_reduce(img)
        assert r.filtered.shape == (32, 96)

    def test_noise_estimates_consistent(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert r.noise_estimate_before >= 0.0
        assert r.noise_estimate_after >= 0.0


# ─── TestBatchReduceExtra ───────────────────────────────────────────────────

class TestBatchReduceExtra:
    def test_single_image(self):
        result = batch_reduce([make_noisy()])
        assert len(result) == 1

    def test_method_auto(self):
        result = batch_reduce([make_noisy()], method="auto")
        assert result[0].method == "auto"

    def test_mixed_shapes(self):
        images = [make_noisy(32, 32), make_noisy(64, 48)]
        result = batch_reduce(images, method="gaussian")
        assert result[0].filtered.shape == (32, 32)
        assert result[1].filtered.shape == (64, 48)

    def test_five_images(self):
        images = [make_noisy(seed=i) for i in range(5)]
        result = batch_reduce(images)
        assert len(result) == 5

    def test_all_dtype_uint8(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = batch_reduce(images, method="median")
        for r in result:
            assert r.filtered.dtype == np.uint8

    def test_kwargs_ksize(self):
        result = batch_reduce([make_noisy()], method="gaussian", ksize=7)
        assert result[0].params.get("ksize") == 7
