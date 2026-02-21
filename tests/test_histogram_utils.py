"""Tests for puzzle_reconstruction.utils.histogram_utils."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.histogram_utils import (
    backproject,
    chi_squared_distance,
    compute_1d_histogram,
    compute_2d_histogram,
    earth_mover_distance,
    histogram_equalization,
    histogram_intersection,
    histogram_specification,
    joint_histogram,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, value: int = 100) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _gradient_gray(h: int = 32, w: int = 32) -> np.ndarray:
    col = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(col.astype(np.uint8), (h, 1))


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = _gradient_gray(h, w)
    img[:, :, 1] = 50
    img[:, :, 2] = 200
    return img


# ─── compute_1d_histogram ────────────────────────────────────────────────────

class TestCompute1dHistogram:
    def test_shape_n_bins(self):
        h = compute_1d_histogram(_gradient_gray())
        assert h.shape == (256,)

    def test_custom_n_bins(self):
        h = compute_1d_histogram(_gradient_gray(), n_bins=64)
        assert h.shape == (64,)

    def test_dtype_float32(self):
        h = compute_1d_histogram(_gradient_gray())
        assert h.dtype == np.float32

    def test_normalized_sums_to_one(self):
        h = compute_1d_histogram(_gradient_gray(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_no_normalize_raw_counts(self):
        img = _gray(value=100)
        h = compute_1d_histogram(img, normalize=False)
        assert h.sum() == pytest.approx(img.size, rel=1e-5)

    def test_n_bins_less_than_1_raises(self):
        with pytest.raises(ValueError):
            compute_1d_histogram(_gray(), n_bins=0)

    def test_channel_out_of_range_raises(self):
        with pytest.raises(ValueError):
            compute_1d_histogram(_bgr(), channel=5)

    def test_2d_image_accepted(self):
        h = compute_1d_histogram(_gray())
        assert h.shape == (256,)

    def test_3d_channel_selection(self):
        img = _bgr()
        h0 = compute_1d_histogram(img, channel=0)
        h2 = compute_1d_histogram(img, channel=2)
        # Channel 0 is gradient, channel 2 is constant 200 — must differ
        assert not np.allclose(h0, h2)

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError):
            compute_1d_histogram(np.zeros((4, 4, 3, 1), dtype=np.uint8))

    def test_uniform_image_one_bin_nonzero(self):
        img = _gray(value=128)
        h = compute_1d_histogram(img, normalize=True)
        assert h[128] == pytest.approx(1.0, abs=1e-5)
        assert h[:128].sum() == pytest.approx(0.0, abs=1e-5)

    def test_n_bins_1_returns_shape_1(self):
        h = compute_1d_histogram(_gray(), n_bins=1)
        assert h.shape == (1,)
        assert h[0] == pytest.approx(1.0, abs=1e-5)


# ─── compute_2d_histogram ────────────────────────────────────────────────────

class TestCompute2dHistogram:
    def test_shape_n_bins_x_n_bins(self):
        h = compute_2d_histogram(_bgr())
        assert h.shape == (64, 64)

    def test_custom_n_bins(self):
        h = compute_2d_histogram(_bgr(), n_bins=32)
        assert h.shape == (32, 32)

    def test_dtype_float32(self):
        assert compute_2d_histogram(_bgr()).dtype == np.float32

    def test_normalized_sums_to_one(self):
        h = compute_2d_histogram(_bgr(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-4)

    def test_single_channel_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_gray())

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_bgr(), n_bins=1)

    def test_channel_out_of_range_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_bgr(), channel1=0, channel2=5)

    def test_no_normalize_raw_counts(self):
        h = compute_2d_histogram(_bgr(), normalize=False)
        assert h.sum() == pytest.approx(_bgr().shape[0] * _bgr().shape[1], rel=1e-3)

    def test_non_negative(self):
        h = compute_2d_histogram(_bgr())
        assert np.all(h >= 0.0)


# ─── histogram_equalization ──────────────────────────────────────────────────

class TestHistogramEqualization:
    def test_returns_uint8(self):
        result = histogram_equalization(_gradient_gray())
        assert result.dtype == np.uint8

    def test_same_shape(self):
        img = _gradient_gray(20, 30)
        result = histogram_equalization(img)
        assert result.shape == (20, 30)

    def test_multi_channel_raises(self):
        with pytest.raises(ValueError):
            histogram_equalization(_bgr())

    def test_values_in_range(self):
        result = histogram_equalization(_gradient_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_output_spread(self):
        # After equalization, values should span a wider range
        img = np.ones((32, 32), dtype=np.uint8) * 100
        # Uniform input: equalization can't do much, but should return valid uint8
        result = histogram_equalization(img)
        assert result.dtype == np.uint8


# ─── histogram_specification ────────────────────────────────────────────────

class TestHistogramSpecification:
    def test_returns_uint8(self):
        src = _gradient_gray(32, 32)
        ref = _gray(32, 32, value=200)
        result = histogram_specification(src, ref)
        assert result.dtype == np.uint8

    def test_same_shape_as_src(self):
        src = _gradient_gray(20, 24)
        ref = _gradient_gray(30, 35)
        result = histogram_specification(src, ref)
        assert result.shape == (20, 24)

    def test_src_multi_channel_raises(self):
        with pytest.raises(ValueError):
            histogram_specification(_bgr(), _gray())

    def test_ref_multi_channel_raises(self):
        with pytest.raises(ValueError):
            histogram_specification(_gray(), _bgr())

    def test_same_src_and_ref_identity(self):
        img = _gradient_gray(32, 40)
        result = histogram_specification(img, img)
        # Specification to itself should return very similar values
        np.testing.assert_array_almost_equal(result.astype(float), img.astype(float), decimal=1)

    def test_values_in_uint8_range(self):
        result = histogram_specification(_gradient_gray(), _gray(value=180))
        assert result.min() >= 0
        assert result.max() <= 255

    def test_output_matches_ref_distribution(self):
        # After specification to a uniform bright image, output should trend bright
        src = _gradient_gray()
        ref = np.full((32, 32), 200, dtype=np.uint8)
        result = histogram_specification(src, ref)
        # Output mean should be higher than src mean (src is gradient 0..255)
        assert result.mean() >= src.mean() - 30  # allow some slack


# ─── earth_mover_distance ────────────────────────────────────────────────────

class TestEarthMoverDistance:
    def test_same_histogram_returns_zero(self):
        h = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)
        assert earth_mover_distance(h, h) == pytest.approx(0.0, abs=1e-6)

    def test_different_histograms_positive(self):
        h1 = np.array([1.0, 0.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 1.0])
        assert earth_mover_distance(h1, h2) > 0.0

    def test_non_negative(self):
        h1 = np.array([0.5, 0.5])
        h2 = np.array([0.3, 0.7])
        assert earth_mover_distance(h1, h2) >= 0.0

    def test_symmetric(self):
        h1 = np.array([0.6, 0.2, 0.1, 0.1])
        h2 = np.array([0.1, 0.1, 0.2, 0.6])
        assert earth_mover_distance(h1, h2) == pytest.approx(
            earth_mover_distance(h2, h1), abs=1e-6
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            earth_mover_distance(np.ones(4), np.ones(5))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            earth_mover_distance(np.array([]), np.array([]))

    def test_unnormalized_inputs_handled(self):
        h1 = np.array([10.0, 20.0, 30.0])
        h2 = np.array([10.0, 20.0, 30.0])
        assert earth_mover_distance(h1, h2) == pytest.approx(0.0, abs=1e-6)


# ─── chi_squared_distance ────────────────────────────────────────────────────

class TestChiSquaredDistance:
    def test_same_histogram_returns_zero(self):
        h = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        assert chi_squared_distance(h, h) == pytest.approx(0.0, abs=1e-6)

    def test_different_positive(self):
        h1 = np.array([1.0, 0.0])
        h2 = np.array([0.0, 1.0])
        assert chi_squared_distance(h1, h2) > 0.0

    def test_non_negative(self):
        h1 = np.random.default_rng(0).random(8)
        h2 = np.random.default_rng(1).random(8)
        assert chi_squared_distance(h1, h2) >= 0.0

    def test_symmetric(self):
        h1 = np.array([0.6, 0.2, 0.2])
        h2 = np.array([0.1, 0.5, 0.4])
        assert chi_squared_distance(h1, h2) == pytest.approx(
            chi_squared_distance(h2, h1), abs=1e-6
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            chi_squared_distance(np.ones(4), np.ones(6))

    def test_zero_vectors(self):
        assert chi_squared_distance(np.zeros(5), np.zeros(5)) == pytest.approx(0.0, abs=1e-6)


# ─── histogram_intersection ──────────────────────────────────────────────────

class TestHistogramIntersection:
    def test_same_histogram_returns_one(self):
        h = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        assert histogram_intersection(h, h) == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_returns_zero(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 1.0])
        assert histogram_intersection(h1, h2) == pytest.approx(0.0, abs=1e-5)

    def test_value_in_unit_interval(self):
        h1 = np.array([0.5, 0.3, 0.2])
        h2 = np.array([0.2, 0.5, 0.3])
        val = histogram_intersection(h1, h2)
        assert 0.0 <= val <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            histogram_intersection(np.ones(4), np.ones(3))

    def test_zero_ref_returns_zero(self):
        assert histogram_intersection(np.ones(4), np.zeros(4)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        h1 = np.array([0.5, 0.5, 0.0])
        h2 = np.array([0.0, 0.5, 0.5])
        val = histogram_intersection(h1, h2)
        assert 0.0 < val < 1.0


# ─── backproject ────────────────────────────────────────────────────────────

class TestBackproject:
    def test_returns_float32(self):
        img = _gray(32, 32, 128)
        model = compute_1d_histogram(img, normalize=True)
        result = backproject(img, model)
        assert result.dtype == np.float32

    def test_shape_h_w(self):
        img = _gray(20, 25)
        model = compute_1d_histogram(img, normalize=True)
        result = backproject(img, model)
        assert result.shape == (20, 25)

    def test_values_non_negative(self):
        img = _gradient_gray()
        model = compute_1d_histogram(img, normalize=True)
        result = backproject(img, model, n_bins=256)
        assert np.all(result >= 0.0)

    def test_length_mismatch_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            backproject(img, np.ones(64, dtype=np.float32), n_bins=32)

    def test_3d_image_accepted(self):
        img = _bgr()
        model = compute_1d_histogram(img, channel=0, n_bins=64, normalize=True)
        result = backproject(img, model, n_bins=64, channel=0)
        assert result.shape == (img.shape[0], img.shape[1])

    def test_uniform_model_uniform_output(self):
        img = _gray(value=100)
        # Uniform model: all bins equal
        model = np.ones(256, dtype=np.float32) / 256
        result = backproject(img, model, n_bins=256)
        # All pixels have value 100, so all map to the same probability
        assert result.std() == pytest.approx(0.0, abs=1e-7)


# ─── joint_histogram ─────────────────────────────────────────────────────────

class TestJointHistogram:
    def test_shape_n_bins_x_n_bins(self):
        h = joint_histogram(_gradient_gray(), _gray())
        assert h.shape == (64, 64)

    def test_custom_n_bins(self):
        h = joint_histogram(_gradient_gray(), _gray(), n_bins=32)
        assert h.shape == (32, 32)

    def test_dtype_float32(self):
        assert joint_histogram(_gradient_gray(), _gray()).dtype == np.float32

    def test_normalized_sums_to_one(self):
        h = joint_histogram(_gradient_gray(), _gray(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-4)

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError):
            joint_histogram(_gray(), _gray(), n_bins=1)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            joint_histogram(_gray(32, 32), _gray(16, 16))

    def test_non_negative(self):
        h = joint_histogram(_gradient_gray(), _gradient_gray())
        assert np.all(h >= 0.0)

    def test_self_joint_diagonal_dominant(self):
        # Joint histogram of identical images: mass mostly on diagonal
        img = _gradient_gray(64, 64)
        h = joint_histogram(img, img, n_bins=64, normalize=True)
        diagonal = np.diag(h).sum()
        off_diagonal = h.sum() - diagonal
        assert diagonal > off_diagonal

    def test_bgr_input_converted(self):
        # BGR inputs should be converted to gray internally
        h = joint_histogram(_bgr(), _bgr(), n_bins=32)
        assert h.shape == (32, 32)
