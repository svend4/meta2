"""Tests for puzzle_reconstruction/algorithms/color_space.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.color_space import (
    ColorSpaceConfig,
    ColorHistogram,
    bgr_to_space,
    compute_channel_hist,
    compute_color_histogram,
    histogram_intersection,
    histogram_chi2,
    batch_compute_histograms,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_bgr(h=50, w=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_gray(h=50, w=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def make_histogram(n_bins=32, fragment_id=0, space="hsv"):
    hist = np.ones(n_bins * 3, dtype=np.float32) / (n_bins * 3)
    return ColorHistogram(fragment_id=fragment_id, space=space,
                          hist=hist, n_bins=n_bins)


# ─── ColorSpaceConfig ─────────────────────────────────────────────────────────

class TestColorSpaceConfig:
    def test_default_construction(self):
        cfg = ColorSpaceConfig()
        assert cfg.target_space == "hsv"
        assert cfg.n_bins == 32
        assert cfg.normalize is True

    def test_valid_spaces(self):
        for space in ("bgr", "hsv", "lab", "gray"):
            cfg = ColorSpaceConfig(target_space=space)
            assert cfg.target_space == space

    def test_invalid_space_raises(self):
        with pytest.raises(ValueError):
            ColorSpaceConfig(target_space="xyz")

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            ColorSpaceConfig(n_bins=3)

    def test_n_bins_minimum_ok(self):
        cfg = ColorSpaceConfig(n_bins=4)
        assert cfg.n_bins == 4

    def test_custom_config(self):
        cfg = ColorSpaceConfig(target_space="lab", n_bins=16, normalize=False)
        assert cfg.target_space == "lab"
        assert cfg.n_bins == 16
        assert cfg.normalize is False


# ─── ColorHistogram ───────────────────────────────────────────────────────────

class TestColorHistogram:
    def test_basic_creation(self):
        h = make_histogram(n_bins=16)
        assert h.fragment_id == 0
        assert h.n_bins == 16

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            ColorHistogram(fragment_id=-1, space="hsv",
                           hist=np.zeros(32, dtype=np.float32), n_bins=8)

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            ColorHistogram(fragment_id=0, space="hsv",
                           hist=np.zeros(32, dtype=np.float32), n_bins=3)

    def test_hist_converted_to_float32(self):
        hist = np.zeros(32, dtype=np.float64)
        h = ColorHistogram(fragment_id=0, space="gray", hist=hist, n_bins=8)
        assert h.hist.dtype == np.float32

    def test_hist_not_1d_raises(self):
        with pytest.raises(ValueError):
            ColorHistogram(fragment_id=0, space="gray",
                           hist=np.zeros((4, 4), dtype=np.float32), n_bins=4)

    def test_dim_property(self):
        h = make_histogram(n_bins=32)
        assert h.dim == 96  # 3 channels * 32 bins

    def test_dim_gray(self):
        hist = np.zeros(32, dtype=np.float32)
        h = ColorHistogram(fragment_id=0, space="gray", hist=hist, n_bins=32)
        assert h.dim == 32


# ─── bgr_to_space ─────────────────────────────────────────────────────────────

class TestBgrToSpace:
    def test_bgr_space_unchanged(self):
        img = make_bgr()
        result = bgr_to_space(img, "bgr")
        assert result.shape == img.shape

    def test_gray_space_2d(self):
        img = make_bgr()
        result = bgr_to_space(img, "gray")
        assert result.ndim == 2

    def test_hsv_space_3d(self):
        img = make_bgr()
        result = bgr_to_space(img, "hsv")
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_lab_space_3d(self):
        img = make_bgr()
        result = bgr_to_space(img, "lab")
        assert result.ndim == 3

    def test_grayscale_input_bgr(self):
        img = make_gray()
        result = bgr_to_space(img, "bgr")
        assert result.ndim == 3

    def test_grayscale_input_gray(self):
        img = make_gray()
        result = bgr_to_space(img, "gray")
        assert result.ndim == 2

    def test_unknown_space_raises(self):
        img = make_bgr()
        with pytest.raises(ValueError):
            bgr_to_space(img, "xyz")

    def test_wrong_ndim_raises(self):
        img = np.zeros((5, 5, 5, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            bgr_to_space(img, "gray")

    def test_output_uint8(self):
        img = make_bgr()
        for space in ("bgr", "hsv", "lab", "gray"):
            result = bgr_to_space(img, space)
            assert result.dtype == np.uint8


# ─── compute_channel_hist ─────────────────────────────────────────────────────

class TestComputeChannelHist:
    def test_shape(self):
        ch = make_gray()
        h = compute_channel_hist(ch, n_bins=32)
        assert h.shape == (32,)

    def test_dtype_float32(self):
        ch = make_gray()
        h = compute_channel_hist(ch)
        assert h.dtype == np.float32

    def test_normalized_sums_to_one(self):
        ch = make_gray()
        h = compute_channel_hist(ch, normalize=True)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_unnormalized_sums_to_pixel_count(self):
        ch = make_gray(h=50, w=50)
        h = compute_channel_hist(ch, normalize=False)
        assert abs(h.sum() - 2500.0) < 1e-3

    def test_n_bins_too_small_raises(self):
        ch = make_gray()
        with pytest.raises(ValueError):
            compute_channel_hist(ch, n_bins=3)

    def test_custom_bins(self):
        ch = make_gray()
        h = compute_channel_hist(ch, n_bins=64)
        assert h.shape == (64,)

    def test_constant_channel(self):
        ch = np.full((20, 20), 128, dtype=np.uint8)
        h = compute_channel_hist(ch, n_bins=32, normalize=True)
        # All pixels in one bin → one bin = 1.0
        assert abs(h.sum() - 1.0) < 1e-5
        assert (h > 0).sum() == 1


# ─── compute_color_histogram ──────────────────────────────────────────────────

class TestComputeColorHistogram:
    def test_returns_color_histogram(self):
        img = make_bgr()
        ch = compute_color_histogram(img)
        assert isinstance(ch, ColorHistogram)

    def test_hsv_dim(self):
        img = make_bgr()
        cfg = ColorSpaceConfig(target_space="hsv", n_bins=16)
        ch = compute_color_histogram(img, cfg)
        assert ch.dim == 16 * 3

    def test_gray_dim(self):
        img = make_bgr()
        cfg = ColorSpaceConfig(target_space="gray", n_bins=16)
        ch = compute_color_histogram(img, cfg)
        assert ch.dim == 16

    def test_fragment_id_set(self):
        img = make_bgr()
        ch = compute_color_histogram(img, fragment_id=42)
        assert ch.fragment_id == 42

    def test_space_stored(self):
        img = make_bgr()
        cfg = ColorSpaceConfig(target_space="lab")
        ch = compute_color_histogram(img, cfg)
        assert ch.space == "lab"

    def test_grayscale_input(self):
        img = make_gray()
        ch = compute_color_histogram(img)
        assert isinstance(ch, ColorHistogram)

    def test_default_config_normalized(self):
        img = make_bgr()
        ch = compute_color_histogram(img)
        # Normalized → sum per channel ≈ 1, total channels * 1
        # Actually concatenated, so total sum = n_channels
        assert ch.hist.sum() > 0


# ─── histogram_intersection ──────────────────────────────────────────────────

class TestHistogramIntersection:
    def test_identical_histograms_score_one(self):
        img = make_bgr()
        ch = compute_color_histogram(img)
        score = histogram_intersection(ch, ch)
        assert abs(score - 1.0) < 1e-5

    def test_in_range(self):
        img1 = make_bgr(seed=0)
        img2 = make_bgr(seed=1)
        h1 = compute_color_histogram(img1)
        h2 = compute_color_histogram(img2)
        score = histogram_intersection(h1, h2)
        assert 0.0 <= score <= 1.0

    def test_dim_mismatch_raises(self):
        cfg4 = ColorSpaceConfig(n_bins=4)
        cfg8 = ColorSpaceConfig(n_bins=8)
        h1 = compute_color_histogram(make_bgr(), cfg4)
        h2 = compute_color_histogram(make_bgr(), cfg8)
        with pytest.raises(ValueError):
            histogram_intersection(h1, h2)

    def test_noisy_vs_different(self):
        img = make_bgr(seed=10)
        h1 = compute_color_histogram(img)
        img2 = make_bgr(seed=99)
        h2 = compute_color_histogram(img2)
        # identical > different
        assert histogram_intersection(h1, h1) >= histogram_intersection(h1, h2)


# ─── histogram_chi2 ──────────────────────────────────────────────────────────

class TestHistogramChi2:
    def test_identical_histograms_score_near_one(self):
        img = make_bgr()
        ch = compute_color_histogram(img)
        score = histogram_chi2(ch, ch)
        assert abs(score - 1.0) < 0.001

    def test_in_range(self):
        img1 = make_bgr(seed=0)
        img2 = make_bgr(seed=5)
        h1 = compute_color_histogram(img1)
        h2 = compute_color_histogram(img2)
        score = histogram_chi2(h1, h2)
        assert 0.0 < score <= 1.0

    def test_dim_mismatch_raises(self):
        cfg4 = ColorSpaceConfig(n_bins=4)
        cfg8 = ColorSpaceConfig(n_bins=8)
        h1 = compute_color_histogram(make_bgr(), cfg4)
        h2 = compute_color_histogram(make_bgr(), cfg8)
        with pytest.raises(ValueError):
            histogram_chi2(h1, h2)

    def test_identical_higher_than_different(self):
        img = make_bgr(seed=7)
        h1 = compute_color_histogram(img)
        h2 = compute_color_histogram(make_bgr(seed=99))
        assert histogram_chi2(h1, h1) >= histogram_chi2(h1, h2)

    def test_positive(self):
        img1 = make_bgr(seed=0)
        img2 = make_bgr(seed=1)
        h1 = compute_color_histogram(img1)
        h2 = compute_color_histogram(img2)
        assert histogram_chi2(h1, h2) > 0.0


# ─── batch_compute_histograms ────────────────────────────────────────────────

class TestBatchComputeHistograms:
    def test_returns_list(self):
        images = [make_bgr(seed=i) for i in range(3)]
        results = batch_compute_histograms(images)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_fragment_ids_correct(self):
        images = [make_bgr(seed=i) for i in range(4)]
        results = batch_compute_histograms(images)
        for i, r in enumerate(results):
            assert r.fragment_id == i

    def test_all_color_histogram_instances(self):
        images = [make_bgr() for _ in range(3)]
        results = batch_compute_histograms(images)
        assert all(isinstance(r, ColorHistogram) for r in results)

    def test_empty_list(self):
        results = batch_compute_histograms([])
        assert results == []

    def test_custom_config(self):
        images = [make_bgr() for _ in range(2)]
        cfg = ColorSpaceConfig(target_space="gray", n_bins=8)
        results = batch_compute_histograms(images, cfg)
        for r in results:
            assert r.dim == 8
            assert r.space == "gray"
