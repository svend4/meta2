"""Tests for puzzle_reconstruction.algorithms.gradient_flow."""
from __future__ import annotations

import math

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.gradient_flow import (
    GradientField,
    GradientStats,
    batch_gradient_fields,
    compare_gradient_fields,
    compute_curl,
    compute_divergence,
    compute_gradient,
    compute_gradient_stats,
    compute_magnitude,
    compute_orientation,
    flow_along_boundary,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _ramp_x(h: int = 32, w: int = 32) -> np.ndarray:
    """Horizontal intensity ramp: left=0, right=255."""
    col = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(col.astype(np.uint8), (h, 1))


def _ramp_y(h: int = 32, w: int = 32) -> np.ndarray:
    """Vertical intensity ramp: top=0, bottom=255."""
    row = np.linspace(0, 255, h, dtype=np.float32)
    return np.repeat(row.astype(np.uint8).reshape(-1, 1), w, axis=1)


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = _ramp_x(h, w)
    return img


def _square_contour(cx: int = 16, cy: int = 16, size: int = 10) -> np.ndarray:
    half = size // 2
    return np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.float32)


def _field_from_arrays(gx: np.ndarray, gy: np.ndarray) -> GradientField:
    return GradientField(gx=gx.astype(np.float32), gy=gy.astype(np.float32))


# ─── GradientField ───────────────────────────────────────────────────────────

class TestGradientField:
    def test_shape_property(self):
        gx = np.zeros((20, 30), dtype=np.float32)
        gy = np.zeros((20, 30), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert f.shape == (20, 30)

    def test_params_stored(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy, params={"ksize": 3})
        assert f.params["ksize"] == 3

    def test_shape_mismatch_raises(self):
        gx = np.zeros((10, 10), dtype=np.float32)
        gy = np.zeros((10, 12), dtype=np.float32)
        with pytest.raises(ValueError):
            GradientField(gx=gx, gy=gy)

    def test_gx_gy_stored(self):
        gx = np.ones((4, 4), dtype=np.float32) * 3.0
        gy = np.ones((4, 4), dtype=np.float32) * -1.0
        f = GradientField(gx=gx, gy=gy)
        np.testing.assert_array_equal(f.gx, gx)
        np.testing.assert_array_equal(f.gy, gy)


# ─── GradientStats ───────────────────────────────────────────────────────────

class TestGradientStats:
    def test_fields_accessible(self):
        stats = GradientStats(
            mean_magnitude=10.0,
            std_magnitude=2.0,
            mean_orientation=0.5,
            dominant_angle=0.1,
            edge_density=0.3,
        )
        assert stats.mean_magnitude == pytest.approx(10.0)
        assert stats.std_magnitude == pytest.approx(2.0)
        assert stats.mean_orientation == pytest.approx(0.5)
        assert stats.dominant_angle == pytest.approx(0.1)
        assert stats.edge_density == pytest.approx(0.3)
        assert stats.params == {}


# ─── compute_gradient ────────────────────────────────────────────────────────

class TestComputeGradient:
    def test_returns_gradient_field(self):
        f = compute_gradient(_ramp_x())
        assert isinstance(f, GradientField)

    def test_gx_gy_dtype_float32(self):
        f = compute_gradient(_ramp_x())
        assert f.gx.dtype == np.float32
        assert f.gy.dtype == np.float32

    def test_shape_preserved(self):
        img = _ramp_x(20, 24)
        f = compute_gradient(img)
        assert f.shape == (20, 24)

    def test_invalid_ksize_raises(self):
        for bad in (0, 2, 4, 6, 8):
            with pytest.raises(ValueError):
                compute_gradient(_gray(), ksize=bad)

    def test_valid_ksizes(self):
        for ks in (1, 3, 5, 7):
            f = compute_gradient(_ramp_x(), ksize=ks)
            assert isinstance(f, GradientField)

    def test_ramp_x_has_positive_gx(self):
        # Horizontal ramp: gradient mostly in x-direction
        f = compute_gradient(_ramp_x())
        assert f.gx.mean() > 0.0

    def test_ramp_y_has_positive_gy(self):
        f = compute_gradient(_ramp_y())
        assert f.gy.mean() > 0.0

    def test_uniform_image_near_zero(self):
        f = compute_gradient(_gray(value=100))
        # Interior pixels should be near zero; sum over interior
        assert np.abs(f.gx[2:-2, 2:-2]).mean() < 1.0
        assert np.abs(f.gy[2:-2, 2:-2]).mean() < 1.0

    def test_normalize_flag(self):
        f = compute_gradient(_ramp_x(), normalize=True)
        mag = np.sqrt(f.gx ** 2 + f.gy ** 2)
        assert mag.max() <= 1.0 + 1e-5

    def test_bgr_accepted(self):
        f = compute_gradient(_bgr())
        assert isinstance(f, GradientField)
        assert f.shape == (32, 32)

    def test_params_contain_ksize(self):
        f = compute_gradient(_ramp_x(), ksize=5)
        assert f.params["ksize"] == 5


# ─── compute_magnitude ───────────────────────────────────────────────────────

class TestComputeMagnitude:
    def test_dtype_float32(self):
        f = compute_gradient(_ramp_x())
        mag = compute_magnitude(f)
        assert mag.dtype == np.float32

    def test_shape_preserved(self):
        f = compute_gradient(_ramp_x(20, 24))
        assert compute_magnitude(f).shape == (20, 24)

    def test_non_negative(self):
        f = compute_gradient(_ramp_x())
        assert compute_magnitude(f).min() >= 0.0

    def test_zero_field_magnitude_zero(self):
        gx = np.zeros((10, 10), dtype=np.float32)
        gy = np.zeros((10, 10), dtype=np.float32)
        mag = compute_magnitude(_field_from_arrays(gx, gy))
        np.testing.assert_array_equal(mag, np.zeros((10, 10), dtype=np.float32))

    def test_known_values(self):
        gx = np.array([[3.0, 0.0]], dtype=np.float32)
        gy = np.array([[4.0, 0.0]], dtype=np.float32)
        mag = compute_magnitude(_field_from_arrays(gx, gy))
        assert mag[0, 0] == pytest.approx(5.0, abs=1e-4)
        assert mag[0, 1] == pytest.approx(0.0, abs=1e-4)


# ─── compute_orientation ─────────────────────────────────────────────────────

class TestComputeOrientation:
    def test_dtype_float32(self):
        f = compute_gradient(_ramp_x())
        assert compute_orientation(f).dtype == np.float32

    def test_shape_preserved(self):
        f = compute_gradient(_ramp_x(18, 22))
        assert compute_orientation(f).shape == (18, 22)

    def test_range_minus_pi_to_pi(self):
        f = compute_gradient(_ramp_x())
        orient = compute_orientation(f)
        assert orient.min() >= -math.pi - 1e-5
        assert orient.max() <= math.pi + 1e-5

    def test_rightward_gradient_near_zero_angle(self):
        # ramp_x: gradient mostly pointing right → angle ≈ 0
        f = compute_gradient(_ramp_x(), ksize=3)
        orient = compute_orientation(f)
        # Mean orientation in interior should be close to 0 (right-pointing)
        interior = orient[4:-4, 4:-4]
        assert abs(interior.mean()) < 1.0  # within ±1 radian of 0


# ─── compute_divergence ──────────────────────────────────────────────────────

class TestComputeDivergence:
    def test_dtype_float32(self):
        f = compute_gradient(_ramp_x())
        assert compute_divergence(f).dtype == np.float32

    def test_shape_preserved(self):
        f = compute_gradient(_ramp_x(24, 28))
        assert compute_divergence(f).shape == (24, 28)

    def test_uniform_zero_field_returns_zeros(self):
        gx = np.zeros((12, 12), dtype=np.float32)
        gy = np.zeros((12, 12), dtype=np.float32)
        div = compute_divergence(_field_from_arrays(gx, gy))
        np.testing.assert_array_almost_equal(div, np.zeros((12, 12)))

    def test_constant_gradient_near_zero_div(self):
        # Constant gx (no spatial variation) → ∂gx/∂x ≈ 0
        gx = np.ones((20, 20), dtype=np.float32) * 5.0
        gy = np.zeros((20, 20), dtype=np.float32)
        div = compute_divergence(_field_from_arrays(gx, gy))
        np.testing.assert_array_almost_equal(div[2:-2, 2:-2], 0.0, decimal=4)


# ─── compute_curl ────────────────────────────────────────────────────────────

class TestComputeCurl:
    def test_dtype_float32(self):
        f = compute_gradient(_ramp_x())
        assert compute_curl(f).dtype == np.float32

    def test_shape_preserved(self):
        f = compute_gradient(_ramp_x(22, 26))
        assert compute_curl(f).shape == (22, 26)

    def test_uniform_zero_field_returns_zeros(self):
        gx = np.zeros((12, 12), dtype=np.float32)
        gy = np.zeros((12, 12), dtype=np.float32)
        curl = compute_curl(_field_from_arrays(gx, gy))
        np.testing.assert_array_almost_equal(curl, np.zeros((12, 12)))

    def test_irrotational_field_near_zero_curl(self):
        # Gradient of a scalar field is irrotational: curl ≈ 0
        f = compute_gradient(_ramp_x(), ksize=3)
        curl = compute_curl(f)
        assert np.abs(curl[3:-3, 3:-3]).mean() < 5.0  # small interior values


# ─── flow_along_boundary ─────────────────────────────────────────────────────

class TestFlowAlongBoundary:
    def test_empty_contour_returns_empty(self):
        f = compute_gradient(_ramp_x())
        result = flow_along_boundary(f, np.empty((0, 2), dtype=np.float32))
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_window_less_than_1_raises(self):
        f = compute_gradient(_ramp_x())
        contour = _square_contour()
        with pytest.raises(ValueError):
            flow_along_boundary(f, contour, window=0)

    def test_returns_float32(self):
        f = compute_gradient(_ramp_x())
        contour = _square_contour()
        result = flow_along_boundary(f, contour)
        assert result.dtype == np.float32

    def test_shape_n_points(self):
        f = compute_gradient(_ramp_x())
        contour = _square_contour()
        result = flow_along_boundary(f, contour)
        assert result.shape == (len(contour),)

    def test_cv2_format_contour(self):
        # (N, 1, 2) shape
        f = compute_gradient(_ramp_x())
        contour = _square_contour().reshape(-1, 1, 2)
        result = flow_along_boundary(f, contour)
        assert result.shape == (4,)

    def test_single_point_contour(self):
        f = compute_gradient(_ramp_x())
        contour = np.array([[16.0, 16.0]], dtype=np.float32)
        result = flow_along_boundary(f, contour)
        assert result.shape == (1,)


# ─── compare_gradient_fields ─────────────────────────────────────────────────

class TestCompareGradientFields:
    def test_same_field_returns_near_one(self):
        f = compute_gradient(_ramp_x())
        sim = compare_gradient_fields(f, f)
        assert sim >= 0.9

    def test_value_in_minus_one_to_one(self):
        f1 = compute_gradient(_ramp_x())
        f2 = compute_gradient(_ramp_y())
        sim = compare_gradient_fields(f1, f2)
        assert -1.0 <= sim <= 1.0

    def test_shape_mismatch_raises(self):
        f1 = compute_gradient(_ramp_x(20, 20))
        f2 = compute_gradient(_ramp_x(20, 24))
        with pytest.raises(ValueError):
            compare_gradient_fields(f1, f2)

    def test_with_mask_all_ones(self):
        f = compute_gradient(_ramp_x())
        mask = np.ones(f.shape, dtype=np.uint8)
        sim_no_mask = compare_gradient_fields(f, f)
        sim_with_mask = compare_gradient_fields(f, f, mask=mask)
        assert sim_with_mask == pytest.approx(sim_no_mask, abs=1e-4)

    def test_with_empty_mask_returns_zero(self):
        f = compute_gradient(_ramp_x())
        mask = np.zeros(f.shape, dtype=np.uint8)
        sim = compare_gradient_fields(f, f, mask=mask)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_opposite_field_negative(self):
        gx = np.ones((16, 16), dtype=np.float32)
        gy = np.zeros((16, 16), dtype=np.float32)
        f1 = _field_from_arrays(gx, gy)
        f2 = _field_from_arrays(-gx, gy)
        sim = compare_gradient_fields(f1, f2)
        assert sim < 0.0


# ─── batch_gradient_fields ───────────────────────────────────────────────────

class TestBatchGradientFields:
    def test_empty_returns_empty(self):
        assert batch_gradient_fields([]) == []

    def test_length_preserved(self):
        imgs = [_ramp_x()] * 4
        result = batch_gradient_fields(imgs)
        assert len(result) == 4

    def test_all_gradient_field_instances(self):
        imgs = [_ramp_x(), _ramp_y(), _gray()]
        result = batch_gradient_fields(imgs)
        assert all(isinstance(f, GradientField) for f in result)

    def test_shapes_match_input(self):
        imgs = [_ramp_x(20, 24), _ramp_x(30, 36)]
        result = batch_gradient_fields(imgs)
        assert result[0].shape == (20, 24)
        assert result[1].shape == (30, 36)

    def test_normalize_flag_passed(self):
        imgs = [_ramp_x()]
        result = batch_gradient_fields(imgs, normalize=True)
        mag = compute_magnitude(result[0])
        assert mag.max() <= 1.0 + 1e-5


# ─── compute_gradient_stats ──────────────────────────────────────────────────

class TestComputeGradientStats:
    def test_returns_gradient_stats(self):
        f = compute_gradient(_ramp_x())
        stats = compute_gradient_stats(f)
        assert isinstance(stats, GradientStats)

    def test_mean_magnitude_non_negative(self):
        stats = compute_gradient_stats(compute_gradient(_ramp_x()))
        assert stats.mean_magnitude >= 0.0

    def test_std_magnitude_non_negative(self):
        stats = compute_gradient_stats(compute_gradient(_ramp_x()))
        assert stats.std_magnitude >= 0.0

    def test_edge_density_in_unit_interval(self):
        stats = compute_gradient_stats(compute_gradient(_ramp_x()), threshold=10.0)
        assert 0.0 <= stats.edge_density <= 1.0

    def test_dominant_angle_in_range(self):
        stats = compute_gradient_stats(compute_gradient(_ramp_x()))
        assert -math.pi <= stats.dominant_angle <= math.pi

    def test_threshold_negative_raises(self):
        f = compute_gradient(_ramp_x())
        with pytest.raises(ValueError):
            compute_gradient_stats(f, threshold=-1.0)

    def test_n_orientation_bins_less_than_1_raises(self):
        f = compute_gradient(_ramp_x())
        with pytest.raises(ValueError):
            compute_gradient_stats(f, n_orientation_bins=0)

    def test_uniform_image_edge_density_zero(self):
        # Uniform image has zero gradient → edge_density should be 0
        f = compute_gradient(_gray(value=100))
        stats = compute_gradient_stats(f, threshold=1.0)
        assert stats.edge_density == pytest.approx(0.0, abs=0.01)

    def test_params_stored(self):
        f = compute_gradient(_ramp_x())
        stats = compute_gradient_stats(f, threshold=5.0, n_orientation_bins=18)
        assert stats.params["threshold"] == pytest.approx(5.0)
        assert stats.params["n_orientation_bins"] == 18

    def test_ramp_x_dominant_angle_near_zero(self):
        # Horizontal ramp → gradient points right → dominant angle ≈ 0
        f = compute_gradient(_ramp_x(64, 64), ksize=3)
        stats = compute_gradient_stats(f, threshold=0.5, n_orientation_bins=72)
        assert abs(stats.dominant_angle) < 0.5  # within ±0.5 rad of 0
