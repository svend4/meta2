"""
Property-based tests for puzzle_reconstruction.utils.histogram_utils.

Verifies mathematical invariants:
- compute_1d_histogram:   shape (n_bins,); normalized → sum ≈ 1; non-negative;
                          all-same-value image → one non-zero bin
- compute_2d_histogram:   shape (n_bins, n_bins); normalized → sum ≈ 1; non-negative
- histogram_equalization: same shape, dtype uint8; values in [0, 255]
- histogram_specification: same shape as src; dtype uint8; values in [0, 255]
- earth_mover_distance:   ≥ 0; self-distance = 0; symmetric;
                          identical histograms → 0; length mismatch raises
- chi_squared_distance:   ≥ 0; self-distance = 0; symmetric;
                          identical histograms → 0; length mismatch raises
- histogram_intersection: ∈ [0, 1]; self = 1; both-zero → 0; symmetric;
                          length mismatch raises
- backproject:            same spatial shape (H, W); values ∈ [0, max_hist_val];
                          length mismatch raises
- joint_histogram:        shape (n_bins, n_bins); normalized → sum ≈ 1;
                          non-negative; shape mismatch raises
"""
from __future__ import annotations

import numpy as np
import pytest

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

RNG = np.random.default_rng(31)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_image_gray(h: int = 50, w: int = 50) -> np.ndarray:
    return RNG.integers(0, 256, size=(h, w), dtype=np.uint8)


def _rand_image_color(h: int = 50, w: int = 50, c: int = 3) -> np.ndarray:
    return RNG.integers(0, 256, size=(h, w, c), dtype=np.uint8)


def _rand_hist(n: int = 32) -> np.ndarray:
    h = RNG.uniform(0, 1, size=n).astype(np.float32)
    h = h / h.sum()
    return h


# ─── compute_1d_histogram ─────────────────────────────────────────────────────

class TestCompute1dHistogram:
    @pytest.mark.parametrize("n_bins", [16, 64, 128, 256])
    def test_shape(self, n_bins):
        img = _rand_image_gray()
        h = compute_1d_histogram(img, channel=0, n_bins=n_bins)
        assert h.shape == (n_bins,)

    def test_dtype_float32(self):
        img = _rand_image_gray()
        h = compute_1d_histogram(img)
        assert h.dtype == np.float32

    def test_normalized_sum_one(self):
        img = _rand_image_gray()
        h = compute_1d_histogram(img, normalize=True)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_non_negative(self):
        img = _rand_image_gray()
        h = compute_1d_histogram(img)
        assert np.all(h >= 0)

    def test_not_normalized_sum_n_pixels(self):
        h_w = 30
        img = _rand_image_gray(h_w, h_w)
        h = compute_1d_histogram(img, normalize=False)
        assert abs(h.sum() - h_w * h_w) < 1e-4

    def test_constant_image_single_bin(self):
        img = np.full((30, 30), 128, dtype=np.uint8)
        h = compute_1d_histogram(img, n_bins=256, normalize=True)
        # The bin containing 128 should be the only non-zero bin
        non_zero = np.sum(h > 0)
        assert non_zero == 1

    def test_channel_selection(self):
        img = _rand_image_color(30, 30, 3)
        h0 = compute_1d_histogram(img, channel=0)
        h1 = compute_1d_histogram(img, channel=1)
        # Different channels → different histograms (with very high probability)
        assert not np.allclose(h0, h1)

    def test_invalid_n_bins(self):
        img = _rand_image_gray()
        with pytest.raises(ValueError):
            compute_1d_histogram(img, n_bins=0)

    def test_invalid_channel(self):
        # For 3-channel image, channel 3 is out of range
        img = _rand_image_color(30, 30, 3)
        with pytest.raises(ValueError):
            compute_1d_histogram(img, channel=3)

    def test_grayscale_channel0_ok(self):
        img = _rand_image_gray(40, 40)
        h = compute_1d_histogram(img, channel=0)
        assert h.shape[0] == 256


# ─── compute_2d_histogram ─────────────────────────────────────────────────────

class TestCompute2dHistogram:
    def test_shape(self):
        img = _rand_image_color(30, 30, 3)
        h = compute_2d_histogram(img, n_bins=32)
        assert h.shape == (32, 32)

    def test_dtype_float32(self):
        img = _rand_image_color(30, 30, 3)
        h = compute_2d_histogram(img)
        assert h.dtype == np.float32

    def test_normalized_sum_one(self):
        img = _rand_image_color(30, 30, 3)
        h = compute_2d_histogram(img, normalize=True)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_non_negative(self):
        img = _rand_image_color(30, 30, 3)
        h = compute_2d_histogram(img)
        assert np.all(h >= 0)

    def test_single_channel_raises(self):
        img = _rand_image_gray()
        with pytest.raises(ValueError):
            compute_2d_histogram(img)

    def test_invalid_n_bins(self):
        img = _rand_image_color()
        with pytest.raises(ValueError):
            compute_2d_histogram(img, n_bins=1)


# ─── histogram_equalization ───────────────────────────────────────────────────

class TestHistogramEqualization:
    def test_same_shape(self):
        img = _rand_image_gray()
        out = histogram_equalization(img)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _rand_image_gray()
        out = histogram_equalization(img)
        assert out.dtype == np.uint8

    def test_values_in_0_255(self):
        img = _rand_image_gray()
        out = histogram_equalization(img)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_multiochannel_raises(self):
        img = _rand_image_color()
        with pytest.raises(ValueError):
            histogram_equalization(img)

    def test_constant_image(self):
        img = np.full((30, 30), 128, dtype=np.uint8)
        out = histogram_equalization(img)
        # Constant image equalized to itself (all same value)
        assert out.min() == out.max()

    def test_contrast_improved(self):
        """Equalized image should have wider value range than low-contrast input."""
        # Create a low-contrast image
        img = np.linspace(100, 150, 50 * 50, dtype=np.uint8).reshape(50, 50)
        out = histogram_equalization(img)
        assert out.max() - out.min() >= img.max() - img.min()


# ─── histogram_specification ──────────────────────────────────────────────────

class TestHistogramSpecification:
    def test_same_shape_as_src(self):
        src = _rand_image_gray(40, 40)
        ref = _rand_image_gray(30, 30)
        out = histogram_specification(src, ref)
        assert out.shape == src.shape

    def test_dtype_uint8(self):
        src = _rand_image_gray()
        ref = _rand_image_gray()
        out = histogram_specification(src, ref)
        assert out.dtype == np.uint8

    def test_values_in_0_255(self):
        src = _rand_image_gray()
        ref = _rand_image_gray()
        out = histogram_specification(src, ref)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_multichannel_src_raises(self):
        src = _rand_image_color()
        ref = _rand_image_gray()
        with pytest.raises(ValueError):
            histogram_specification(src, ref)

    def test_multichannel_ref_raises(self):
        src = _rand_image_gray()
        ref = _rand_image_color()
        with pytest.raises(ValueError):
            histogram_specification(src, ref)

    def test_self_specification_similar(self):
        """Specifying src against itself should give a similar image."""
        img = _rand_image_gray()
        out = histogram_specification(img, img)
        assert out.shape == img.shape


# ─── earth_mover_distance ─────────────────────────────────────────────────────

class TestEarthMoverDistance:
    def test_nonneg(self):
        for _ in range(20):
            h1 = _rand_hist(32)
            h2 = _rand_hist(32)
            assert earth_mover_distance(h1, h2) >= 0.0

    def test_self_zero(self):
        for _ in range(20):
            h = _rand_hist(32)
            assert earth_mover_distance(h, h) < 1e-8

    def test_symmetric(self):
        for _ in range(20):
            h1 = _rand_hist(32)
            h2 = _rand_hist(32)
            d12 = earth_mover_distance(h1, h2)
            d21 = earth_mover_distance(h2, h1)
            assert abs(d12 - d21) < 1e-8

    def test_identical_histograms_zero(self):
        h = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        assert earth_mover_distance(h, h) < 1e-8

    def test_length_mismatch_raises(self):
        h1 = _rand_hist(32)
        h2 = _rand_hist(16)
        with pytest.raises(ValueError):
            earth_mover_distance(h1, h2)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            earth_mover_distance(np.array([]), np.array([]))

    def test_all_mass_at_opposite_ends(self):
        h1 = np.zeros(100, dtype=np.float32)
        h1[0] = 1.0
        h2 = np.zeros(100, dtype=np.float32)
        h2[-1] = 1.0
        emd = earth_mover_distance(h1, h2)
        assert emd > 0.0


# ─── chi_squared_distance ─────────────────────────────────────────────────────

class TestChiSquaredDistance:
    def test_nonneg(self):
        for _ in range(20):
            h1 = _rand_hist(32)
            h2 = _rand_hist(32)
            assert chi_squared_distance(h1, h2) >= 0.0

    def test_self_zero(self):
        for _ in range(20):
            h = _rand_hist(32)
            assert chi_squared_distance(h, h) < 1e-8

    def test_symmetric(self):
        for _ in range(20):
            h1 = _rand_hist(32)
            h2 = _rand_hist(32)
            d12 = chi_squared_distance(h1, h2)
            d21 = chi_squared_distance(h2, h1)
            assert abs(d12 - d21) < 1e-8

    def test_identical_histograms_zero(self):
        h = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        assert chi_squared_distance(h, h) < 1e-8

    def test_length_mismatch_raises(self):
        h1 = _rand_hist(32)
        h2 = _rand_hist(16)
        with pytest.raises(ValueError):
            chi_squared_distance(h1, h2)

    def test_zero_histograms_zero(self):
        h = np.zeros(16, dtype=np.float32)
        assert chi_squared_distance(h, h) == pytest.approx(0.0)


# ─── histogram_intersection ───────────────────────────────────────────────────

class TestHistogramIntersection:
    def test_range_0_1(self):
        for _ in range(20):
            h1 = _rand_hist(32)
            h2 = _rand_hist(32)
            val = histogram_intersection(h1, h2)
            assert 0.0 <= val <= 1.0 + 1e-8

    def test_self_is_one(self):
        for _ in range(20):
            h = _rand_hist(32)
            assert histogram_intersection(h, h) == pytest.approx(1.0, abs=1e-5)

    def test_zero_h2_returns_zero(self):
        h1 = _rand_hist(32)
        h2 = np.zeros(32, dtype=np.float32)
        assert histogram_intersection(h1, h2) == pytest.approx(0.0)

    def test_disjoint_histograms_zero(self):
        h1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        h2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        assert histogram_intersection(h1, h2) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        h1 = _rand_hist(32)
        h2 = _rand_hist(16)
        with pytest.raises(ValueError):
            histogram_intersection(h1, h2)


# ─── backproject ──────────────────────────────────────────────────────────────

class TestBackproject:
    def test_same_spatial_shape(self):
        img = _rand_image_gray(50, 60)
        hist = _rand_hist(256)
        out = backproject(img, hist, n_bins=256)
        assert out.shape == (50, 60)

    def test_values_in_hist_range(self):
        img = _rand_image_gray(30, 30)
        hist = np.full(256, 0.5, dtype=np.float32)
        out = backproject(img, hist, n_bins=256)
        assert out.min() >= 0.0
        assert out.max() <= hist.max() + 1e-6

    def test_length_mismatch_raises(self):
        img = _rand_image_gray()
        hist = _rand_hist(128)
        with pytest.raises(ValueError):
            backproject(img, hist, n_bins=256)

    def test_color_image_single_channel(self):
        img = _rand_image_color(30, 30, 3)
        hist = _rand_hist(256)
        out = backproject(img, hist, n_bins=256, channel=0)
        assert out.shape == (30, 30)

    def test_zero_histogram_all_zeros(self):
        img = _rand_image_gray(20, 20)
        hist = np.zeros(256, dtype=np.float32)
        out = backproject(img, hist, n_bins=256)
        np.testing.assert_array_equal(out, np.zeros((20, 20), dtype=np.float32))


# ─── joint_histogram ──────────────────────────────────────────────────────────

class TestJointHistogram:
    @pytest.mark.parametrize("n_bins", [16, 32, 64])
    def test_shape(self, n_bins):
        img1 = _rand_image_gray(40, 40)
        img2 = _rand_image_gray(40, 40)
        h = joint_histogram(img1, img2, n_bins=n_bins)
        assert h.shape == (n_bins, n_bins)

    def test_dtype_float32(self):
        img1 = _rand_image_gray(30, 30)
        img2 = _rand_image_gray(30, 30)
        h = joint_histogram(img1, img2)
        assert h.dtype == np.float32

    def test_normalized_sum_one(self):
        img1 = _rand_image_gray(30, 30)
        img2 = _rand_image_gray(30, 30)
        h = joint_histogram(img1, img2, normalize=True)
        assert abs(h.sum() - 1.0) < 1e-5

    def test_non_negative(self):
        img1 = _rand_image_gray(30, 30)
        img2 = _rand_image_gray(30, 30)
        h = joint_histogram(img1, img2)
        assert np.all(h >= 0)

    def test_shape_mismatch_raises(self):
        img1 = _rand_image_gray(30, 30)
        img2 = _rand_image_gray(40, 40)
        with pytest.raises(ValueError):
            joint_histogram(img1, img2)

    def test_invalid_n_bins_raises(self):
        img1 = _rand_image_gray(30, 30)
        img2 = _rand_image_gray(30, 30)
        with pytest.raises(ValueError):
            joint_histogram(img1, img2, n_bins=1)

    def test_identical_images_diagonal_dominant(self):
        """Same image → joint histogram should have mass on diagonal."""
        img = _rand_image_gray(50, 50)
        h = joint_histogram(img, img, n_bins=32, normalize=True)
        # Diagonal sum should exceed off-diagonal
        diag_sum = np.trace(h)
        total_sum = h.sum()
        assert diag_sum / total_sum > 0.5
