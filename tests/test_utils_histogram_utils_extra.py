"""Extra tests for puzzle_reconstruction/utils/histogram_utils.py"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.histogram_utils import (
    compute_1d_histogram,
    compute_2d_histogram,
    histogram_equalization,
    histogram_specification,
    earth_mover_distance,
    chi_squared_distance,
    histogram_intersection,
    backproject,
    joint_histogram,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=32, w=32, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=32, w=32, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _dirac(n=256, pos=0) -> np.ndarray:
    h = np.zeros(n, dtype=np.float64)
    h[pos] = 1.0
    return h


# ─── TestCompute1dHistogramExtra ─────────────────────────────────────────────

class TestCompute1dHistogramExtra:
    def test_non_normalized_sums_to_pixels(self):
        img = _rand_gray(8, 8)
        h = compute_1d_histogram(img, normalize=False)
        assert h.sum() == pytest.approx(64.0, abs=0.5)

    def test_custom_bins_8(self):
        h = compute_1d_histogram(_rand_gray(), n_bins=8, normalize=True)
        assert h.shape == (8,)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_channel_2_rgb(self):
        h = compute_1d_histogram(_rand_rgb(), channel=2)
        assert h.shape == (256,)

    def test_constant_val_0_bin_0(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        h = compute_1d_histogram(img, normalize=False)
        assert h[0] == pytest.approx(256.0, abs=0.5)

    def test_grayscale_no_channel_error(self):
        # channel=0 on grayscale should work fine
        h = compute_1d_histogram(_gray(), channel=0)
        assert h.shape == (256,)


# ─── TestCompute2dHistogramExtra ─────────────────────────────────────────────

class TestCompute2dHistogramExtra:
    def test_not_normalized_sums_to_pixels(self):
        img = _rand_rgb(16, 16)
        h = compute_2d_histogram(img, normalize=False)
        assert h.sum() == pytest.approx(256.0, abs=1.0)

    def test_channels_1_2(self):
        h = compute_2d_histogram(_rand_rgb(), channel1=1, channel2=2)
        assert h.ndim == 2

    def test_64_bins(self):
        h = compute_2d_histogram(_rand_rgb(), n_bins=64)
        assert h.shape == (64, 64)

    def test_dtype_float32(self):
        h = compute_2d_histogram(_rand_rgb(), normalize=False)
        assert h.dtype == np.float32


# ─── TestHistogramEqualizationExtra ──────────────────────────────────────────

class TestHistogramEqualizationExtra:
    def test_non_square_image(self):
        img = _rand_gray(16, 48)
        out = histogram_equalization(img)
        assert out.shape == (16, 48)

    def test_all_255_image(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        out = histogram_equalization(img)
        assert out.shape == (32, 32)
        assert out.dtype == np.uint8

    def test_output_range_0_255(self):
        img = _rand_gray()
        out = histogram_equalization(img)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_single_pixel(self):
        img = np.array([[128]], dtype=np.uint8)
        out = histogram_equalization(img)
        assert out.shape == (1, 1)


# ─── TestHistogramSpecificationExtra ─────────────────────────────────────────

class TestHistogramSpecificationExtra:
    def test_output_values_in_range(self):
        src = _rand_gray(16, 16, seed=2)
        ref = _rand_gray(16, 16, seed=3)
        out = histogram_specification(src, ref)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_output_uint8(self):
        out = histogram_specification(_rand_gray(seed=0), _rand_gray(seed=1))
        assert out.dtype == np.uint8

    def test_non_square_src(self):
        src = _rand_gray(10, 30, seed=4)
        ref = _rand_gray(20, 20, seed=5)
        out = histogram_specification(src, ref)
        assert out.shape == (10, 30)

    def test_uniform_ref_spreads_values(self):
        src = _gray(val=100)  # all same value
        ref = _rand_gray()   # varied reference
        out = histogram_specification(src, ref)
        assert out.shape == src.shape


# ─── TestEarthMoverDistanceExtra ──────────────────────────────────────────────

class TestEarthMoverDistanceExtra:
    def test_symmetric(self):
        h1 = np.array([0.2, 0.5, 0.3])
        h2 = np.array([0.1, 0.6, 0.3])
        d1 = earth_mover_distance(h1, h2)
        d2 = earth_mover_distance(h2, h1)
        assert d1 == pytest.approx(d2, abs=1e-9)

    def test_adjacent_diracs(self):
        h1 = _dirac(10, pos=4)
        h2 = _dirac(10, pos=5)
        d = earth_mover_distance(h1, h2)
        assert d > 0.0

    def test_single_element(self):
        h1 = np.array([1.0])
        h2 = np.array([1.0])
        d = earth_mover_distance(h1, h2)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_nonneg_all_cases(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            h1 = rng.random(32)
            h2 = rng.random(32)
            assert earth_mover_distance(h1, h2) >= 0.0


# ─── TestChiSquaredDistanceExtra ──────────────────────────────────────────────

class TestChiSquaredDistanceExtra:
    def test_single_element_identical(self):
        assert chi_squared_distance(np.array([1.0]), np.array([1.0])) == pytest.approx(0.0)

    def test_zero_both(self):
        h = np.zeros(256)
        d = chi_squared_distance(h, h)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_nonneg(self):
        rng = np.random.default_rng(11)
        h1 = rng.random(64)
        h2 = rng.random(64)
        assert chi_squared_distance(h1, h2) >= 0.0

    def test_returns_float(self):
        assert isinstance(chi_squared_distance(np.ones(4), np.ones(4)), float)


# ─── TestHistogramIntersectionExtra ──────────────────────────────────────────

class TestHistogramIntersectionExtra:
    def test_half_overlap(self):
        h1 = np.zeros(4)
        h1[:2] = 0.5
        h2 = np.zeros(4)
        h2[1:3] = 0.5
        s = histogram_intersection(h1, h2)
        assert 0.0 <= s <= 1.0

    def test_first_zero_hist_returns_zero(self):
        h2 = np.ones(4) / 4
        s = histogram_intersection(np.zeros(4), h2)
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_for_equal_histograms(self):
        h1 = np.array([0.25, 0.25, 0.25, 0.25])
        s = histogram_intersection(h1, h1)
        assert s == pytest.approx(1.0, abs=1e-5)

    def test_scaled_hist2(self):
        h1 = np.array([0.5, 0.5])
        h2 = np.array([1.0, 1.0])
        s = histogram_intersection(h1, h2)
        # min(0.5,1.0)+min(0.5,1.0) / (1.0+1.0) = 1.0/2.0 = 0.5
        assert s == pytest.approx(0.5, abs=1e-9)


# ─── TestBackprojectExtra ─────────────────────────────────────────────────────

class TestBackprojectExtra:
    def test_all_zero_hist_gives_zeros(self):
        img = _rand_gray()
        hist = np.zeros(256, dtype=np.float32)
        bp = backproject(img, hist)
        assert np.all(bp == 0.0)

    def test_uniform_hist_gives_constant(self):
        img = _rand_gray()
        hist = np.full(256, 1.0 / 256, dtype=np.float32)
        bp = backproject(img, hist)
        assert bp.min() == pytest.approx(bp.max(), abs=1e-6)

    def test_grayscale_shape(self):
        img = _rand_gray(8, 16)
        hist = np.ones(256, dtype=np.float32) / 256
        bp = backproject(img, hist)
        assert bp.shape == (8, 16)

    def test_rgb_channel_1(self):
        img = _rand_rgb()
        hist = compute_1d_histogram(img, channel=1)
        bp = backproject(img, hist, channel=1)
        assert bp.shape == (32, 32)
        assert bp.dtype == np.float32

    def test_n_bins_32(self):
        img = _rand_gray()
        hist = np.ones(32, dtype=np.float32) / 32
        bp = backproject(img, hist, n_bins=32)
        assert bp.shape == img.shape


# ─── TestJointHistogramExtra ─────────────────────────────────────────────────

class TestJointHistogramExtra:
    def test_not_normalized(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=1), normalize=False)
        assert h.sum() == pytest.approx(32 * 32, abs=1.0)

    def test_shape_16_bins(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=2), n_bins=16)
        assert h.shape == (16, 16)

    def test_identical_diagonal_dominant(self):
        img = _rand_gray()
        h = joint_histogram(img, img, n_bins=64, normalize=False)
        diag = np.trace(h)
        assert diag == pytest.approx(h.sum(), abs=1.0)

    def test_nonneg(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=3), normalize=True)
        assert np.all(h >= 0.0)

    def test_rgb_both_inputs(self):
        h = joint_histogram(_rand_rgb(), _rand_rgb(seed=4))
        assert h.ndim == 2
        assert h.dtype == np.float32
