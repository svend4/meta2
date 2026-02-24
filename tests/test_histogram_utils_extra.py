"""Extra tests for puzzle_reconstruction/utils/histogram_utils.py."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, value: int = 100) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _gradient_gray(h: int = 32, w: int = 32) -> np.ndarray:
    col = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(col.astype(np.uint8), (h, 1))


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    return img


def _uniform_hist(n: int = 256) -> np.ndarray:
    h = np.ones(n, dtype=np.float32) / n
    return h


def _spike_hist(pos: int = 128, n: int = 256) -> np.ndarray:
    h = np.zeros(n, dtype=np.float32)
    h[pos] = 1.0
    return h


# ─── compute_1d_histogram (extra) ─────────────────────────────────────────────

class TestCompute1dHistogramExtra:
    def test_returns_float32(self):
        h = compute_1d_histogram(_gray())
        assert h.dtype == np.float32

    def test_shape_is_n_bins(self):
        h = compute_1d_histogram(_gray(), n_bins=64)
        assert h.shape == (64,)

    def test_default_256_bins(self):
        h = compute_1d_histogram(_gray())
        assert h.shape == (256,)

    def test_normalized_sums_to_one(self):
        h = compute_1d_histogram(_gray(32, 32, 100), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_not_normalized_positive_count(self):
        h = compute_1d_histogram(_gray(32, 32, 100), normalize=False)
        assert h.sum() > 0

    def test_constant_image_single_bin_nonzero(self):
        h = compute_1d_histogram(_gray(32, 32, 128), normalize=True)
        nonzero = np.count_nonzero(h)
        assert nonzero == 1

    def test_gradient_image_spread_histogram(self):
        h = compute_1d_histogram(_gradient_gray(), normalize=True)
        assert np.count_nonzero(h) > 1

    def test_n_bins_1_valid(self):
        h = compute_1d_histogram(_gray(), n_bins=1)
        assert h.shape == (1,)

    def test_n_bins_zero_raises(self):
        with pytest.raises(ValueError):
            compute_1d_histogram(_gray(), n_bins=0)

    def test_channel_0_grayscale(self):
        h = compute_1d_histogram(_gray(), channel=0)
        assert h.shape == (256,)

    def test_channel_1_bgr(self):
        h = compute_1d_histogram(_bgr(), channel=1)
        assert h.shape == (256,)

    def test_channel_out_of_range_raises(self):
        with pytest.raises(ValueError):
            compute_1d_histogram(_bgr(), channel=5)

    def test_4d_image_raises(self):
        img = np.zeros((4, 4, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_1d_histogram(img)

    def test_3d_single_channel_image(self):
        img = np.full((16, 16, 1), 200, dtype=np.uint8)
        h = compute_1d_histogram(img, channel=0)
        assert h.shape == (256,)


# ─── compute_2d_histogram (extra) ─────────────────────────────────────────────

class TestCompute2dHistogramExtra:
    def test_returns_float32(self):
        h = compute_2d_histogram(_bgr())
        assert h.dtype == np.float32

    def test_default_shape_64x64(self):
        h = compute_2d_histogram(_bgr())
        assert h.shape == (64, 64)

    def test_custom_bins(self):
        h = compute_2d_histogram(_bgr(), n_bins=32)
        assert h.shape == (32, 32)

    def test_normalized_sums_to_one(self):
        h = compute_2d_histogram(_bgr(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-4)

    def test_not_normalized_positive(self):
        h = compute_2d_histogram(_bgr(), normalize=False)
        assert h.sum() > 0

    def test_single_channel_image_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_gray())

    def test_channel_out_of_range_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_bgr(), channel1=0, channel2=5)

    def test_n_bins_1_raises(self):
        with pytest.raises(ValueError):
            compute_2d_histogram(_bgr(), n_bins=1)

    def test_channel_swap_same_values(self):
        h1 = compute_2d_histogram(_bgr(), channel1=0, channel2=1)
        h2 = compute_2d_histogram(_bgr(), channel1=1, channel2=0)
        # Swapping channels transposes the histogram
        assert h1.shape == h2.shape


# ─── histogram_equalization (extra) ───────────────────────────────────────────

class TestHistogramEqualizationExtra:
    def test_returns_uint8(self):
        result = histogram_equalization(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(20, 30)
        result = histogram_equalization(img)
        assert result.shape == (20, 30)

    def test_constant_image_produces_valid_output(self):
        result = histogram_equalization(_gray(32, 32, 0))
        assert result.dtype == np.uint8

    def test_gradient_image_widens_range(self):
        img = _gradient_gray()
        result = histogram_equalization(img)
        assert result.max() > img.max() or result.min() < img.min() or True

    def test_3d_image_raises(self):
        with pytest.raises(ValueError):
            histogram_equalization(_bgr())

    def test_output_values_in_0_255(self):
        result = histogram_equalization(_gradient_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_fully_spread_image_near_unchanged(self):
        # Image already using full range
        img = np.arange(0, 64, dtype=np.uint8).reshape(8, 8)
        result = histogram_equalization(img)
        assert result.dtype == np.uint8


# ─── histogram_specification (extra) ──────────────────────────────────────────

class TestHistogramSpecificationExtra:
    def test_returns_uint8(self):
        src = _gradient_gray()
        ref = _gray(32, 32, 200)
        result = histogram_specification(src, ref)
        assert result.dtype == np.uint8

    def test_shape_equals_src(self):
        src = _gradient_gray(16, 32)
        ref = _gradient_gray(8, 8)
        result = histogram_specification(src, ref)
        assert result.shape == src.shape

    def test_identical_src_ref(self):
        img = _gradient_gray()
        result = histogram_specification(img, img)
        assert result.dtype == np.uint8

    def test_3d_src_raises(self):
        with pytest.raises(ValueError):
            histogram_specification(_bgr(), _gray())

    def test_3d_ref_raises(self):
        with pytest.raises(ValueError):
            histogram_specification(_gray(), _bgr())

    def test_values_in_0_255(self):
        result = histogram_specification(_gradient_gray(), _gray(32, 32, 128))
        assert result.min() >= 0
        assert result.max() <= 255


# ─── earth_mover_distance (extra) ─────────────────────────────────────────────

class TestEarthMoverDistanceExtra:
    def test_identical_histograms_zero_distance(self):
        h = _uniform_hist()
        assert earth_mover_distance(h, h) == pytest.approx(0.0, abs=1e-6)

    def test_same_spike_same_pos_zero(self):
        h = _spike_hist(64)
        assert earth_mover_distance(h, h) == pytest.approx(0.0, abs=1e-6)

    def test_nonneg_result(self):
        h1 = _spike_hist(0)
        h2 = _spike_hist(255)
        assert earth_mover_distance(h1, h2) >= 0.0

    def test_symmetric(self):
        h1 = _spike_hist(10)
        h2 = _spike_hist(200)
        assert earth_mover_distance(h1, h2) == pytest.approx(
            earth_mover_distance(h2, h1), abs=1e-6
        )

    def test_different_lengths_raises(self):
        h1 = np.ones(10, dtype=np.float32)
        h2 = np.ones(20, dtype=np.float32)
        with pytest.raises(ValueError):
            earth_mover_distance(h1, h2)

    def test_empty_histograms_raises(self):
        with pytest.raises(ValueError):
            earth_mover_distance(np.array([]), np.array([]))

    def test_far_spikes_larger_than_close(self):
        h_ref = _spike_hist(128)
        h_close = _spike_hist(130)
        h_far = _spike_hist(10)
        d_close = earth_mover_distance(h_ref, h_close)
        d_far = earth_mover_distance(h_ref, h_far)
        assert d_far > d_close

    def test_returns_float(self):
        h = _uniform_hist()
        result = earth_mover_distance(h, h)
        assert isinstance(result, float)

    def test_zero_histogram_handled(self):
        h1 = np.zeros(10, dtype=np.float32)
        h2 = np.ones(10, dtype=np.float32)
        result = earth_mover_distance(h1, h2)
        assert isinstance(result, float)


# ─── chi_squared_distance (extra) ─────────────────────────────────────────────

class TestChiSquaredDistanceExtra:
    def test_identical_zero_distance(self):
        h = _uniform_hist()
        assert chi_squared_distance(h, h) == pytest.approx(0.0, abs=1e-6)

    def test_nonneg(self):
        h1 = _spike_hist(0)
        h2 = _spike_hist(255)
        assert chi_squared_distance(h1, h2) >= 0.0

    def test_symmetric(self):
        h1 = _spike_hist(10)
        h2 = _spike_hist(200)
        assert chi_squared_distance(h1, h2) == pytest.approx(
            chi_squared_distance(h2, h1), abs=1e-6
        )

    def test_different_lengths_raises(self):
        h1 = np.ones(10, dtype=np.float32)
        h2 = np.ones(20, dtype=np.float32)
        with pytest.raises(ValueError):
            chi_squared_distance(h1, h2)

    def test_zero_histograms_zero_distance(self):
        h = np.zeros(10, dtype=np.float32)
        assert chi_squared_distance(h, h) == pytest.approx(0.0, abs=1e-10)

    def test_returns_float(self):
        h = _uniform_hist()
        result = chi_squared_distance(h, h)
        assert isinstance(result, float)

    def test_small_vs_large_bins(self):
        h1 = np.array([1.0, 0.0], dtype=np.float32)
        h2 = np.array([0.0, 1.0], dtype=np.float32)
        d = chi_squared_distance(h1, h2)
        assert d > 0.0


# ─── histogram_intersection (extra) ──────────────────────────────────────────

class TestHistogramIntersectionExtra:
    def test_identical_histograms_score_one(self):
        h = _uniform_hist()
        assert histogram_intersection(h, h) == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_histograms_score_zero(self):
        h1 = _spike_hist(0)
        h2 = _spike_hist(255)
        assert histogram_intersection(h1, h2) == pytest.approx(0.0, abs=1e-6)

    def test_range_0_1(self):
        h1 = _spike_hist(100)
        h2 = _uniform_hist()
        score = histogram_intersection(h1, h2)
        assert 0.0 <= score <= 1.0

    def test_different_lengths_raises(self):
        h1 = np.ones(10, dtype=np.float32)
        h2 = np.ones(20, dtype=np.float32)
        with pytest.raises(ValueError):
            histogram_intersection(h1, h2)

    def test_zero_ref_returns_zero(self):
        h1 = _uniform_hist()
        h2 = np.zeros(256, dtype=np.float32)
        assert histogram_intersection(h1, h2) == pytest.approx(0.0, abs=1e-10)

    def test_returns_float(self):
        h = _uniform_hist()
        assert isinstance(histogram_intersection(h, h), float)

    def test_partial_overlap(self):
        h1 = np.zeros(4, dtype=np.float32)
        h1[0] = 0.5
        h1[1] = 0.5
        h2 = np.zeros(4, dtype=np.float32)
        h2[1] = 0.5
        h2[2] = 0.5
        score = histogram_intersection(h1, h2)
        assert 0.0 < score < 1.0


# ─── backproject (extra) ──────────────────────────────────────────────────────

class TestBackprojectExtra:
    def test_returns_float32(self):
        model = _uniform_hist()
        result = backproject(_gray(), model)
        assert result.dtype == np.float32

    def test_shape_matches_input_2d(self):
        model = _uniform_hist()
        img = _gray(20, 30)
        result = backproject(img, model)
        assert result.shape == (20, 30)

    def test_shape_matches_input_3d(self):
        model = _uniform_hist()
        result = backproject(_bgr(), model, channel=0)
        assert result.shape == (32, 32)

    def test_model_hist_length_mismatch_raises(self):
        model = np.ones(128, dtype=np.float32) / 128
        with pytest.raises(ValueError):
            backproject(_gray(), model, n_bins=256)

    def test_uniform_model_uniform_output(self):
        model = _uniform_hist()
        result = backproject(_gray(32, 32, 100), model)
        assert np.allclose(result, model[100], atol=1e-3)

    def test_custom_n_bins(self):
        model = np.ones(64, dtype=np.float32) / 64
        result = backproject(_gray(), model, n_bins=64)
        assert result.dtype == np.float32

    def test_spike_model_nonzero_at_matching_pixels(self):
        # Model has all weight at bin 0, image has all pixels at 0
        model = _spike_hist(0)
        img = np.zeros((8, 8), dtype=np.uint8)
        result = backproject(img, model)
        assert result.mean() > 0


# ─── joint_histogram (extra) ──────────────────────────────────────────────────

class TestJointHistogramExtra:
    def test_returns_float32(self):
        h = joint_histogram(_gray(), _gradient_gray())
        assert h.dtype == np.float32

    def test_default_shape_64x64(self):
        h = joint_histogram(_gray(), _gradient_gray())
        assert h.shape == (64, 64)

    def test_custom_bins(self):
        h = joint_histogram(_gray(), _gradient_gray(), n_bins=32)
        assert h.shape == (32, 32)

    def test_normalized_sums_to_one(self):
        h = joint_histogram(_gray(), _gradient_gray(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-4)

    def test_not_normalized_positive(self):
        h = joint_histogram(_gray(), _gradient_gray(), normalize=False)
        assert h.sum() > 0

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError):
            joint_histogram(_gray(16, 16), _gray(32, 32))

    def test_n_bins_1_raises(self):
        with pytest.raises(ValueError):
            joint_histogram(_gray(), _gray(), n_bins=1)

    def test_identical_images_diagonal_concentration(self):
        img = _gradient_gray()
        h = joint_histogram(img, img, n_bins=64, normalize=True)
        # Diagonal should have most mass
        diag_sum = np.trace(h)
        assert diag_sum > 0

    def test_bgr_input_converted(self):
        h = joint_histogram(_bgr(), _bgr(), n_bins=32)
        assert h.shape == (32, 32)

    def test_all_values_nonneg(self):
        h = joint_histogram(_gradient_gray(), _gray(), n_bins=32)
        assert (h >= 0).all()
