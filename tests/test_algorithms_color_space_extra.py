"""Extra tests for puzzle_reconstruction.algorithms.color_space."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(h=50, w=50, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _solid_bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


# ─── TestColorSpaceConfigExtra ────────────────────────────────────────────────

class TestColorSpaceConfigExtra:
    def test_default_space_hsv(self):
        assert ColorSpaceConfig().target_space == "hsv"

    def test_default_n_bins_32(self):
        assert ColorSpaceConfig().n_bins == 32

    def test_default_normalize_true(self):
        assert ColorSpaceConfig().normalize is True

    @pytest.mark.parametrize("space", ["bgr", "hsv", "lab", "gray"])
    def test_valid_spaces(self, space):
        assert ColorSpaceConfig(target_space=space).target_space == space

    def test_invalid_space_raises(self):
        with pytest.raises(ValueError):
            ColorSpaceConfig(target_space="rgb")

    def test_n_bins_3_raises(self):
        with pytest.raises(ValueError):
            ColorSpaceConfig(n_bins=3)

    def test_n_bins_4_ok(self):
        assert ColorSpaceConfig(n_bins=4).n_bins == 4

    def test_n_bins_256(self):
        assert ColorSpaceConfig(n_bins=256).n_bins == 256

    def test_normalize_false(self):
        cfg = ColorSpaceConfig(normalize=False)
        assert cfg.normalize is False


# ─── TestColorHistogramExtra ──────────────────────────────────────────────────

class TestColorHistogramExtra:
    def test_fragment_id_stored(self):
        hist = np.ones(48, dtype=np.float32) / 48
        h = ColorHistogram(fragment_id=5, space="hsv", hist=hist, n_bins=16)
        assert h.fragment_id == 5

    def test_space_stored(self):
        hist = np.ones(48, dtype=np.float32) / 48
        h = ColorHistogram(fragment_id=0, space="lab", hist=hist, n_bins=16)
        assert h.space == "lab"

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            ColorHistogram(fragment_id=-1, space="hsv",
                           hist=np.zeros(32, dtype=np.float32), n_bins=8)

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            ColorHistogram(fragment_id=0, space="hsv",
                           hist=np.zeros(32, dtype=np.float32), n_bins=3)

    def test_hist_float64_converted(self):
        hist = np.ones(32, dtype=np.float64)
        h = ColorHistogram(fragment_id=0, space="gray", hist=hist, n_bins=8)
        assert h.hist.dtype == np.float32

    def test_hist_2d_raises(self):
        with pytest.raises(ValueError):
            ColorHistogram(fragment_id=0, space="gray",
                           hist=np.zeros((4, 4), dtype=np.float32), n_bins=4)

    def test_dim_hsv_32(self):
        hist = np.ones(96, dtype=np.float32) / 96
        h = ColorHistogram(fragment_id=0, space="hsv", hist=hist, n_bins=32)
        assert h.dim == 96

    def test_dim_gray_16(self):
        hist = np.ones(16, dtype=np.float32) / 16
        h = ColorHistogram(fragment_id=0, space="gray", hist=hist, n_bins=16)
        assert h.dim == 16


# ─── TestBgrToSpaceExtra ─────────────────────────────────────────────────────

class TestBgrToSpaceExtra:
    def test_bgr_unchanged_shape(self):
        img = _bgr()
        result = bgr_to_space(img, "bgr")
        assert result.shape == img.shape

    def test_gray_2d(self):
        result = bgr_to_space(_bgr(), "gray")
        assert result.ndim == 2

    def test_hsv_3ch(self):
        result = bgr_to_space(_bgr(), "hsv")
        assert result.ndim == 3 and result.shape[2] == 3

    def test_lab_3ch(self):
        result = bgr_to_space(_bgr(), "lab")
        assert result.ndim == 3 and result.shape[2] == 3

    def test_grayscale_input_to_bgr(self):
        result = bgr_to_space(_gray(), "bgr")
        assert result.ndim == 3

    def test_grayscale_input_to_gray(self):
        result = bgr_to_space(_gray(), "gray")
        assert result.ndim == 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            bgr_to_space(_bgr(), "yuv")

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            bgr_to_space(np.zeros((5, 5, 5, 3), dtype=np.uint8), "gray")

    @pytest.mark.parametrize("space", ["bgr", "hsv", "lab", "gray"])
    def test_output_uint8(self, space):
        result = bgr_to_space(_bgr(), space)
        assert result.dtype == np.uint8

    def test_solid_bgr_to_hsv_no_crash(self):
        result = bgr_to_space(_solid_bgr(), "hsv")
        assert result.shape[2] == 3


# ─── TestComputeChannelHistExtra ──────────────────────────────────────────────

class TestComputeChannelHistExtra:
    def test_shape_32(self):
        assert compute_channel_hist(_gray(), n_bins=32).shape == (32,)

    def test_shape_64(self):
        assert compute_channel_hist(_gray(), n_bins=64).shape == (64,)

    def test_dtype_float32(self):
        assert compute_channel_hist(_gray()).dtype == np.float32

    def test_normalized_sum_one(self):
        h = compute_channel_hist(_gray(), normalize=True)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_unnormalized_sum_pixels(self):
        h = compute_channel_hist(_gray(h=50, w=50), normalize=False)
        assert abs(h.sum() - 2500.0) < 1e-3

    def test_n_bins_3_raises(self):
        with pytest.raises(ValueError):
            compute_channel_hist(_gray(), n_bins=3)

    def test_constant_channel_one_bin(self):
        ch = np.full((20, 20), 100, dtype=np.uint8)
        h = compute_channel_hist(ch, n_bins=32, normalize=True)
        assert (h > 0).sum() == 1

    def test_all_nonneg(self):
        h = compute_channel_hist(_gray(seed=5))
        assert (h >= 0).all()


# ─── TestComputeColorHistogramExtra ──────────────────────────────────────────

class TestComputeColorHistogramExtra:
    def test_returns_color_histogram(self):
        assert isinstance(compute_color_histogram(_bgr()), ColorHistogram)

    def test_hsv_dim(self):
        cfg = ColorSpaceConfig(target_space="hsv", n_bins=16)
        ch = compute_color_histogram(_bgr(), cfg)
        assert ch.dim == 48

    def test_gray_dim(self):
        cfg = ColorSpaceConfig(target_space="gray", n_bins=16)
        ch = compute_color_histogram(_bgr(), cfg)
        assert ch.dim == 16

    def test_lab_dim(self):
        cfg = ColorSpaceConfig(target_space="lab", n_bins=8)
        ch = compute_color_histogram(_bgr(), cfg)
        assert ch.dim == 24

    def test_fragment_id(self):
        ch = compute_color_histogram(_bgr(), fragment_id=7)
        assert ch.fragment_id == 7

    def test_space_stored(self):
        cfg = ColorSpaceConfig(target_space="lab")
        ch = compute_color_histogram(_bgr(), cfg)
        assert ch.space == "lab"

    def test_grayscale_input(self):
        assert isinstance(compute_color_histogram(_gray()), ColorHistogram)

    def test_hist_nonneg(self):
        ch = compute_color_histogram(_bgr())
        assert (ch.hist >= 0).all()


# ─── TestHistogramIntersectionExtra ──────────────────────────────────────────

class TestHistogramIntersectionExtra:
    def test_identical_nonneg(self):
        ch = compute_color_histogram(_bgr())
        assert histogram_intersection(ch, ch) > 0.0

    def test_different_nonneg(self):
        h1 = compute_color_histogram(_bgr(seed=0))
        h2 = compute_color_histogram(_bgr(seed=1))
        assert histogram_intersection(h1, h2) >= 0.0

    def test_dim_mismatch_raises(self):
        h1 = compute_color_histogram(_bgr(), ColorSpaceConfig(n_bins=4))
        h2 = compute_color_histogram(_bgr(), ColorSpaceConfig(n_bins=8))
        with pytest.raises(ValueError):
            histogram_intersection(h1, h2)

    def test_self_geq_different(self):
        h1 = compute_color_histogram(_bgr(seed=10))
        h2 = compute_color_histogram(_bgr(seed=99))
        assert histogram_intersection(h1, h1) >= histogram_intersection(h1, h2)

    def test_returns_float(self):
        ch = compute_color_histogram(_bgr())
        assert isinstance(histogram_intersection(ch, ch), float)

    def test_symmetric(self):
        h1 = compute_color_histogram(_bgr(seed=0))
        h2 = compute_color_histogram(_bgr(seed=1))
        assert histogram_intersection(h1, h2) == pytest.approx(
            histogram_intersection(h2, h1), abs=1e-6)


# ─── TestHistogramChi2Extra ──────────────────────────────────────────────────

class TestHistogramChi2Extra:
    def test_identical_near_one(self):
        ch = compute_color_histogram(_bgr())
        assert histogram_chi2(ch, ch) == pytest.approx(1.0, abs=0.001)

    def test_in_range(self):
        h1 = compute_color_histogram(_bgr(seed=0))
        h2 = compute_color_histogram(_bgr(seed=5))
        assert 0.0 < histogram_chi2(h1, h2) <= 1.0

    def test_dim_mismatch_raises(self):
        h1 = compute_color_histogram(_bgr(), ColorSpaceConfig(n_bins=4))
        h2 = compute_color_histogram(_bgr(), ColorSpaceConfig(n_bins=8))
        with pytest.raises(ValueError):
            histogram_chi2(h1, h2)

    def test_self_geq_different(self):
        h1 = compute_color_histogram(_bgr(seed=7))
        h2 = compute_color_histogram(_bgr(seed=99))
        assert histogram_chi2(h1, h1) >= histogram_chi2(h1, h2)

    def test_positive(self):
        h1 = compute_color_histogram(_bgr(seed=0))
        h2 = compute_color_histogram(_bgr(seed=1))
        assert histogram_chi2(h1, h2) > 0.0

    def test_returns_float(self):
        ch = compute_color_histogram(_bgr())
        assert isinstance(histogram_chi2(ch, ch), float)


# ─── TestBatchComputeHistogramsExtra ─────────────────────────────────────────

class TestBatchComputeHistogramsExtra:
    def test_returns_list(self):
        assert isinstance(batch_compute_histograms([_bgr()]), list)

    def test_correct_length(self):
        imgs = [_bgr(seed=i) for i in range(5)]
        assert len(batch_compute_histograms(imgs)) == 5

    def test_fragment_ids(self):
        imgs = [_bgr(seed=i) for i in range(4)]
        for i, r in enumerate(batch_compute_histograms(imgs)):
            assert r.fragment_id == i

    def test_all_color_histograms(self):
        for r in batch_compute_histograms([_bgr(), _bgr(seed=1)]):
            assert isinstance(r, ColorHistogram)

    def test_empty_list(self):
        assert batch_compute_histograms([]) == []

    def test_custom_config_gray(self):
        cfg = ColorSpaceConfig(target_space="gray", n_bins=8)
        for r in batch_compute_histograms([_bgr()], cfg):
            assert r.dim == 8
            assert r.space == "gray"

    def test_custom_config_lab(self):
        cfg = ColorSpaceConfig(target_space="lab", n_bins=16)
        for r in batch_compute_histograms([_bgr()], cfg):
            assert r.dim == 48
            assert r.space == "lab"

    def test_grayscale_input(self):
        results = batch_compute_histograms([_gray()])
        assert isinstance(results[0], ColorHistogram)
