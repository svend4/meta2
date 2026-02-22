"""Тесты для puzzle_reconstruction.utils.histogram_utils."""
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


def _uniform_hist(n=256) -> np.ndarray:
    h = np.ones(n, dtype=np.float64)
    return h / h.sum()


# ─── TestCompute1dHistogram ───────────────────────────────────────────────────

class TestCompute1dHistogram:
    def test_returns_ndarray(self):
        h = compute_1d_histogram(_rand_gray())
        assert isinstance(h, np.ndarray)

    def test_shape_default(self):
        h = compute_1d_histogram(_rand_gray())
        assert h.shape == (256,)

    def test_custom_n_bins(self):
        h = compute_1d_histogram(_rand_gray(), n_bins=64)
        assert h.shape == (64,)

    def test_dtype_float32(self):
        h = compute_1d_histogram(_rand_gray())
        assert h.dtype == np.float32

    def test_normalized_sums_to_one(self):
        h = compute_1d_histogram(_rand_gray(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_not_normalized_sums_to_n_pixels(self):
        img = _rand_gray(16, 16)
        h = compute_1d_histogram(img, normalize=False)
        assert h.sum() == pytest.approx(16 * 16, abs=1e-3)

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError):
            compute_1d_histogram(_rand_gray(), n_bins=0)

    def test_channel_out_of_range_raises(self):
        img = _rand_rgb()
        with pytest.raises(ValueError):
            compute_1d_histogram(img, channel=5)

    def test_grayscale_channel0(self):
        h = compute_1d_histogram(_rand_gray(), channel=0)
        assert h.shape == (256,)

    def test_rgb_channel1(self):
        h = compute_1d_histogram(_rand_rgb(), channel=1)
        assert h.shape == (256,)

    def test_constant_image_one_bin_full(self):
        img = _gray(val=100)
        h = compute_1d_histogram(img, normalize=False)
        assert h[100] == pytest.approx(32 * 32, abs=0.5)

    def test_4d_raises(self):
        img = np.zeros((4, 4, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_1d_histogram(img)


# ─── TestCompute2dHistogram ───────────────────────────────────────────────────

class TestCompute2dHistogram:
    def test_returns_ndarray(self):
        h = compute_2d_histogram(_rand_rgb())
        assert isinstance(h, np.ndarray)

    def test_shape_square(self):
        h = compute_2d_histogram(_rand_rgb(), n_bins=32)
        assert h.shape == (32, 32)

    def test_normalized_sums_to_one(self):
        h = compute_2d_histogram(_rand_rgb(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_single_channel_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_rand_gray())

    def test_n_bins_one_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_rand_rgb(), n_bins=1)

    def test_channel_out_of_range_raises(self):
        img = _rand_rgb()
        with pytest.raises(ValueError):
            compute_2d_histogram(img, channel1=0, channel2=5)

    def test_dtype_float32(self):
        h = compute_2d_histogram(_rand_rgb())
        assert h.dtype == np.float32


# ─── TestHistogramEqualization ────────────────────────────────────────────────

class TestHistogramEqualization:
    def test_returns_ndarray(self):
        out = histogram_equalization(_rand_gray())
        assert isinstance(out, np.ndarray)

    def test_output_shape_preserved(self):
        img = _rand_gray(24, 32)
        out = histogram_equalization(img)
        assert out.shape == (24, 32)

    def test_output_dtype_uint8(self):
        out = histogram_equalization(_rand_gray())
        assert out.dtype == np.uint8

    def test_rgb_raises(self):
        with pytest.raises(ValueError):
            histogram_equalization(_rand_rgb())

    def test_constant_image_ok(self):
        out = histogram_equalization(_gray())
        assert out.shape == (32, 32)

    def test_values_in_range(self):
        out = histogram_equalization(_rand_gray())
        assert out.min() >= 0
        assert out.max() <= 255


# ─── TestHistogramSpecification ───────────────────────────────────────────────

class TestHistogramSpecification:
    def test_returns_ndarray(self):
        src = _rand_gray(seed=0)
        ref = _rand_gray(seed=1)
        out = histogram_specification(src, ref)
        assert isinstance(out, np.ndarray)

    def test_output_shape_matches_src(self):
        src = _rand_gray(16, 24)
        ref = _rand_gray(32, 32)
        out = histogram_specification(src, ref)
        assert out.shape == src.shape

    def test_output_dtype_uint8(self):
        out = histogram_specification(_rand_gray(seed=0), _rand_gray(seed=1))
        assert out.dtype == np.uint8

    def test_rgb_src_raises(self):
        with pytest.raises(ValueError):
            histogram_specification(_rand_rgb(), _rand_gray())

    def test_rgb_ref_raises(self):
        with pytest.raises(ValueError):
            histogram_specification(_rand_gray(), _rand_rgb())

    def test_identical_src_ref(self):
        img = _rand_gray()
        out = histogram_specification(img, img)
        assert out.shape == img.shape


# ─── TestEarthMoverDistance ───────────────────────────────────────────────────

class TestEarthMoverDistance:
    def test_identical_histograms_zero(self):
        h = _uniform_hist()
        assert earth_mover_distance(h, h) == pytest.approx(0.0, abs=1e-9)

    def test_nonneg(self):
        h1 = _uniform_hist(64)
        h2 = np.zeros(64)
        h2[0] = 1.0
        assert earth_mover_distance(h1, h2) >= 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            earth_mover_distance(np.ones(10), np.ones(20))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            earth_mover_distance(np.array([]), np.array([]))

    def test_returns_float(self):
        h1 = _uniform_hist()
        h2 = _uniform_hist()
        assert isinstance(earth_mover_distance(h1, h2), float)

    def test_dirac_vs_dirac_far(self):
        h1 = np.zeros(256)
        h1[0] = 1.0
        h2 = np.zeros(256)
        h2[255] = 1.0
        d = earth_mover_distance(h1, h2)
        assert d > 0.0


# ─── TestChiSquaredDistance ───────────────────────────────────────────────────

class TestChiSquaredDistance:
    def test_identical_zero(self):
        h = _uniform_hist()
        assert chi_squared_distance(h, h) == pytest.approx(0.0, abs=1e-9)

    def test_nonneg(self):
        h1 = _uniform_hist(64)
        h2 = np.zeros(64)
        h2[5] = 1.0
        assert chi_squared_distance(h1, h2) >= 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            chi_squared_distance(np.ones(10), np.ones(20))

    def test_returns_float(self):
        assert isinstance(chi_squared_distance(_uniform_hist(), _uniform_hist()), float)

    def test_symmetric(self):
        h1 = _uniform_hist()
        h2 = _uniform_hist()
        h2[:10] = 0.0
        d1 = chi_squared_distance(h1, h2)
        d2 = chi_squared_distance(h2, h1)
        assert d1 == pytest.approx(d2, abs=1e-6)


# ─── TestHistogramIntersection ────────────────────────────────────────────────

class TestHistogramIntersection:
    def test_identical_returns_one(self):
        h = _uniform_hist()
        assert histogram_intersection(h, h) == pytest.approx(1.0, abs=1e-5)

    def test_zero_hist2_returns_zero(self):
        h = _uniform_hist()
        assert histogram_intersection(h, np.zeros(256)) == pytest.approx(0.0)

    def test_range_zero_to_one(self):
        h1 = _uniform_hist()
        h2 = np.zeros(256)
        h2[:128] = 1.0 / 128
        s = histogram_intersection(h1, h2)
        assert 0.0 <= s <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            histogram_intersection(np.ones(10), np.ones(20))

    def test_returns_float(self):
        assert isinstance(histogram_intersection(_uniform_hist(), _uniform_hist()), float)

    def test_disjoint_histograms_zero(self):
        h1 = np.zeros(256)
        h1[:128] = 1.0 / 128
        h2 = np.zeros(256)
        h2[128:] = 1.0 / 128
        s = histogram_intersection(h1, h2)
        assert s == pytest.approx(0.0, abs=1e-9)


# ─── TestBackproject ──────────────────────────────────────────────────────────

class TestBackproject:
    def test_returns_ndarray(self):
        img = _rand_gray()
        hist = compute_1d_histogram(img)
        bp = backproject(img, hist)
        assert isinstance(bp, np.ndarray)

    def test_output_shape(self):
        img = _rand_gray(16, 24)
        hist = compute_1d_histogram(img)
        bp = backproject(img, hist)
        assert bp.shape == (16, 24)

    def test_output_dtype_float32(self):
        img = _rand_gray()
        hist = compute_1d_histogram(img)
        bp = backproject(img, hist)
        assert bp.dtype == np.float32

    def test_wrong_hist_length_raises(self):
        img = _rand_gray()
        hist = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError):
            backproject(img, hist, n_bins=256)

    def test_rgb_channel(self):
        img = _rand_rgb()
        hist = compute_1d_histogram(img, channel=0)
        bp = backproject(img, hist, channel=0)
        assert bp.shape == (32, 32)


# ─── TestJointHistogram ───────────────────────────────────────────────────────

class TestJointHistogram:
    def test_returns_ndarray(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=1))
        assert isinstance(h, np.ndarray)

    def test_shape_square(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=1), n_bins=32)
        assert h.shape == (32, 32)

    def test_dtype_float32(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=1))
        assert h.dtype == np.float32

    def test_normalized_sums_to_one(self):
        h = joint_histogram(_rand_gray(), _rand_gray(seed=1), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_n_bins_one_raises(self):
        with pytest.raises(ValueError):
            joint_histogram(_rand_gray(), _rand_gray(), n_bins=1)

    def test_shape_mismatch_raises(self):
        img1 = _rand_gray(16, 16)
        img2 = _rand_gray(24, 24)
        with pytest.raises(ValueError):
            joint_histogram(img1, img2)

    def test_rgb_input_ok(self):
        h = joint_histogram(_rand_rgb(), _rand_rgb(seed=1))
        assert h.ndim == 2

    def test_identical_images_diagonal(self):
        img = _rand_gray()
        h = joint_histogram(img, img, normalize=False)
        # Diagonal should have all mass
        diag_sum = np.trace(h)
        assert diag_sum == pytest.approx(h.sum(), abs=1.0)
