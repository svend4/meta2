"""Тесты для puzzle_reconstruction/preprocessing/noise_reducer.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_impulse_noisy(h=64, w=64, seed=0):
    """Image with salt-and-pepper noise (high noise estimate)."""
    base = np.full((h, w), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    n_noise = h * w // 2
    rows = rng.integers(0, h, n_noise)
    cols = rng.integers(0, w, n_noise)
    vals = rng.choice([0, 255], n_noise).astype(np.uint8)
    base[rows, cols] = vals
    return base


# ─── NoiseReductionResult ─────────────────────────────────────────────────────

class TestNoiseReductionResult:
    def test_creation(self):
        img = make_gray()
        r = NoiseReductionResult(filtered=img, method="gaussian")
        assert r.method == "gaussian"
        assert r.params == {}
        assert r.noise_estimate_before == 0.0
        assert r.noise_estimate_after == 0.0

    def test_filtered_stored(self):
        img = make_gray()
        r = NoiseReductionResult(filtered=img, method="median")
        assert r.filtered is img

    def test_params_stored(self):
        img = make_gray()
        r = NoiseReductionResult(filtered=img, method="gaussian",
                                  params={"ksize": 5})
        assert r.params["ksize"] == 5

    def test_noise_estimates_stored(self):
        img = make_gray()
        r = NoiseReductionResult(filtered=img, method="gaussian",
                                  noise_estimate_before=10.0,
                                  noise_estimate_after=3.0)
        assert r.noise_estimate_before == pytest.approx(10.0)
        assert r.noise_estimate_after == pytest.approx(3.0)


# ─── estimate_noise ───────────────────────────────────────────────────────────

class TestEstimateNoise:
    def test_returns_nonnegative_float(self):
        img = make_noisy()
        val = estimate_noise(img)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_uniform_image_low_noise(self):
        img = make_gray(fill=128)
        val = estimate_noise(img)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_noisy_image_higher_than_uniform(self):
        flat = make_gray()
        noisy = make_noisy()
        assert estimate_noise(noisy) > estimate_noise(flat)

    def test_accepts_bgr(self):
        bgr = make_bgr()
        val = estimate_noise(bgr)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_accepts_grayscale(self):
        gray = make_gray()
        val = estimate_noise(gray)
        assert val >= 0.0

    def test_impulse_noise_high_estimate(self):
        img = make_impulse_noisy()
        val = estimate_noise(img)
        assert val > 10.0


# ─── gaussian_reduce ──────────────────────────────────────────────────────────

class TestGaussianReduce:
    def test_returns_noise_reduction_result(self):
        img = make_noisy()
        r = gaussian_reduce(img)
        assert isinstance(r, NoiseReductionResult)

    def test_method_is_gaussian(self):
        img = make_noisy()
        r = gaussian_reduce(img)
        assert r.method == "gaussian"

    def test_filtered_same_shape(self):
        img = make_noisy(h=32, w=48)
        r = gaussian_reduce(img)
        assert r.filtered.shape == img.shape

    def test_filtered_dtype_uint8(self):
        img = make_noisy()
        r = gaussian_reduce(img)
        assert r.filtered.dtype == np.uint8

    def test_noise_estimates_stored(self):
        img = make_noisy()
        r = gaussian_reduce(img)
        assert r.noise_estimate_before >= 0.0
        assert r.noise_estimate_after >= 0.0

    def test_filtering_reduces_noise(self):
        img = make_noisy()
        r = gaussian_reduce(img, ksize=9)
        assert r.noise_estimate_after <= r.noise_estimate_before + 1.0

    def test_accepts_bgr(self):
        img = make_bgr()
        r = gaussian_reduce(img)
        assert r.filtered.shape == img.shape

    def test_params_contain_ksize(self):
        img = make_noisy()
        r = gaussian_reduce(img, ksize=7)
        assert "ksize" in r.params

    def test_ksize_forced_odd(self):
        img = make_noisy()
        r = gaussian_reduce(img, ksize=4)  # even → forced to 5
        assert r.params["ksize"] % 2 == 1


# ─── median_reduce ────────────────────────────────────────────────────────────

class TestMedianReduce:
    def test_returns_noise_reduction_result(self):
        img = make_noisy()
        r = median_reduce(img)
        assert isinstance(r, NoiseReductionResult)

    def test_method_is_median(self):
        img = make_noisy()
        r = median_reduce(img)
        assert r.method == "median"

    def test_filtered_same_shape(self):
        img = make_noisy(h=32, w=32)
        r = median_reduce(img)
        assert r.filtered.shape == img.shape

    def test_filtered_dtype_uint8(self):
        img = make_noisy()
        r = median_reduce(img)
        assert r.filtered.dtype == np.uint8

    def test_params_contain_ksize(self):
        img = make_noisy()
        r = median_reduce(img, ksize=3)
        assert "ksize" in r.params

    def test_impulse_noise_reduced(self):
        img = make_impulse_noisy()
        r = median_reduce(img, ksize=5)
        assert r.noise_estimate_after < r.noise_estimate_before + 1.0

    def test_noise_estimates_nonneg(self):
        img = make_noisy()
        r = median_reduce(img)
        assert r.noise_estimate_before >= 0.0
        assert r.noise_estimate_after >= 0.0


# ─── bilateral_reduce ─────────────────────────────────────────────────────────

class TestBilateralReduce:
    def test_returns_noise_reduction_result(self):
        img = make_noisy()
        r = bilateral_reduce(img)
        assert isinstance(r, NoiseReductionResult)

    def test_method_is_bilateral(self):
        img = make_noisy()
        r = bilateral_reduce(img)
        assert r.method == "bilateral"

    def test_filtered_same_shape(self):
        img = make_noisy(h=32, w=32)
        r = bilateral_reduce(img)
        assert r.filtered.shape == img.shape

    def test_filtered_dtype_uint8(self):
        img = make_noisy()
        r = bilateral_reduce(img)
        assert r.filtered.dtype == np.uint8

    def test_params_contain_d(self):
        img = make_noisy()
        r = bilateral_reduce(img, d=5)
        assert "d" in r.params

    def test_params_contain_sigma_color(self):
        img = make_noisy()
        r = bilateral_reduce(img, sigma_color=50.0)
        assert r.params["sigma_color"] == pytest.approx(50.0)

    def test_noise_estimates_nonneg(self):
        img = make_noisy()
        r = bilateral_reduce(img)
        assert r.noise_estimate_before >= 0.0
        assert r.noise_estimate_after >= 0.0


# ─── auto_reduce ──────────────────────────────────────────────────────────────

class TestAutoReduce:
    def test_returns_noise_reduction_result(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert isinstance(r, NoiseReductionResult)

    def test_method_is_auto(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert r.method == "auto"

    def test_params_contain_chosen_filter(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert "chosen_filter" in r.params

    def test_uniform_image_chooses_gaussian_trivial(self):
        img = make_gray(fill=128)  # Very low noise
        r = auto_reduce(img, low_thresh=5.0)
        assert r.params["chosen_filter"] == "gaussian_trivial"

    def test_impulse_noise_chooses_median(self):
        img = make_impulse_noisy()  # Very high noise
        r = auto_reduce(img, low_thresh=5.0, high_thresh=10.0)
        # High noise → median
        assert r.params["chosen_filter"] == "median"

    def test_filtered_same_shape(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert r.filtered.shape == img.shape

    def test_params_contain_thresholds(self):
        img = make_noisy()
        r = auto_reduce(img, low_thresh=3.0, high_thresh=15.0)
        assert r.params["low_thresh"] == pytest.approx(3.0)
        assert r.params["high_thresh"] == pytest.approx(15.0)

    def test_noise_estimates_nonneg(self):
        img = make_noisy()
        r = auto_reduce(img)
        assert r.noise_estimate_before >= 0.0
        assert r.noise_estimate_after >= 0.0


# ─── batch_reduce ─────────────────────────────────────────────────────────────

class TestBatchReduce:
    def test_empty_list_returns_empty(self):
        result = batch_reduce([])
        assert result == []

    def test_length_matches_input(self):
        images = [make_noisy(seed=i) for i in range(4)]
        result = batch_reduce(images)
        assert len(result) == 4

    def test_returns_list_of_results(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = batch_reduce(images, method="gaussian")
        for r in result:
            assert isinstance(r, NoiseReductionResult)

    def test_method_gaussian(self):
        images = [make_noisy(seed=0)]
        result = batch_reduce(images, method="gaussian")
        assert result[0].method == "gaussian"

    def test_method_median(self):
        images = [make_noisy(seed=0)]
        result = batch_reduce(images, method="median")
        assert result[0].method == "median"

    def test_method_bilateral(self):
        images = [make_noisy(seed=0)]
        result = batch_reduce(images, method="bilateral")
        assert result[0].method == "bilateral"

    def test_method_auto(self):
        images = [make_noisy(seed=0)]
        result = batch_reduce(images, method="auto")
        assert result[0].method == "auto"

    def test_unknown_method_raises(self):
        images = [make_noisy()]
        with pytest.raises(ValueError, match="Unknown"):
            batch_reduce(images, method="wavelet")

    def test_kwargs_passed_through(self):
        images = [make_noisy(seed=0)]
        result = batch_reduce(images, method="gaussian", ksize=3)
        assert result[0].params.get("ksize") is not None
