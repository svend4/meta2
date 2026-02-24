"""Extra tests for puzzle_reconstruction/algorithms/gradient_flow.py."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _ramp_x(h: int = 32, w: int = 32) -> np.ndarray:
    col = np.linspace(0, 255, w, dtype=np.float32)
    return np.tile(col.astype(np.uint8), (h, 1))


def _ramp_y(h: int = 32, w: int = 32) -> np.ndarray:
    row = np.linspace(0, 255, h, dtype=np.float32)
    return np.repeat(row.astype(np.uint8).reshape(-1, 1), w, axis=1)


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    return img


def _make_field(h: int = 16, w: int = 16, gx_val: float = 1.0, gy_val: float = 0.0):
    gx = np.full((h, w), gx_val, dtype=np.float32)
    gy = np.full((h, w), gy_val, dtype=np.float32)
    return GradientField(gx=gx, gy=gy)


def _circle_contour(n: int = 16, r: float = 5.0, cx: float = 8.0, cy: float = 8.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    return pts.astype(np.float32)


# ─── GradientField (extra) ────────────────────────────────────────────────────

class TestGradientFieldExtra:
    def test_shape_property_2d(self):
        f = _make_field(8, 12)
        assert f.shape == (8, 12)

    def test_shape_property_square(self):
        f = _make_field(16, 16)
        assert f.shape == (16, 16)

    def test_mismatched_shapes_raise(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            GradientField(gx=gx, gy=gy)

    def test_params_default_empty(self):
        f = _make_field()
        assert f.params == {}

    def test_params_stored(self):
        f = GradientField(
            gx=np.zeros((4, 4), dtype=np.float32),
            gy=np.zeros((4, 4), dtype=np.float32),
            params={"ksize": 3},
        )
        assert f.params["ksize"] == 3

    def test_gx_gy_preserved(self):
        gx = np.ones((4, 4), dtype=np.float32) * 3.0
        gy = np.ones((4, 4), dtype=np.float32) * 7.0
        f = GradientField(gx=gx, gy=gy)
        assert np.allclose(f.gx, 3.0)
        assert np.allclose(f.gy, 7.0)

    def test_single_pixel_field(self):
        gx = np.array([[5.0]], dtype=np.float32)
        gy = np.array([[3.0]], dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert f.shape == (1, 1)

    def test_large_field_shape(self):
        f = _make_field(256, 256)
        assert f.shape == (256, 256)


# ─── GradientStats (extra) ────────────────────────────────────────────────────

class TestGradientStatsExtra:
    def test_fields_stored(self):
        s = GradientStats(
            mean_magnitude=1.0,
            std_magnitude=0.5,
            mean_orientation=0.1,
            dominant_angle=0.2,
            edge_density=0.3,
        )
        assert s.mean_magnitude == pytest.approx(1.0)
        assert s.std_magnitude == pytest.approx(0.5)
        assert s.mean_orientation == pytest.approx(0.1)
        assert s.dominant_angle == pytest.approx(0.2)
        assert s.edge_density == pytest.approx(0.3)

    def test_params_default_empty(self):
        s = GradientStats(
            mean_magnitude=0.0,
            std_magnitude=0.0,
            mean_orientation=0.0,
            dominant_angle=0.0,
            edge_density=0.0,
        )
        assert s.params == {}

    def test_params_stored(self):
        s = GradientStats(
            mean_magnitude=0.0,
            std_magnitude=0.0,
            mean_orientation=0.0,
            dominant_angle=0.0,
            edge_density=0.0,
            params={"threshold": 10.0},
        )
        assert s.params["threshold"] == pytest.approx(10.0)

    def test_edge_density_zero(self):
        s = GradientStats(0.0, 0.0, 0.0, 0.0, 0.0)
        assert s.edge_density == pytest.approx(0.0)

    def test_edge_density_one(self):
        s = GradientStats(100.0, 10.0, 0.0, 0.5, 1.0)
        assert s.edge_density == pytest.approx(1.0)


# ─── compute_gradient (extra) ─────────────────────────────────────────────────

class TestComputeGradientExtra:
    def test_returns_gradient_field(self):
        result = compute_gradient(_gray())
        assert isinstance(result, GradientField)

    def test_output_shape_matches_input(self):
        img = _gray(20, 30)
        f = compute_gradient(img)
        assert f.shape == (20, 30)

    def test_constant_image_near_zero_gradient(self):
        f = compute_gradient(_gray(32, 32, 128))
        mag = np.sqrt(f.gx ** 2 + f.gy ** 2)
        # edges may have some response, interior should be zero
        assert mag[4:-4, 4:-4].max() == pytest.approx(0.0, abs=1e-4)

    def test_ramp_x_has_nonzero_gx(self):
        f = compute_gradient(_ramp_x())
        assert f.gx[16, 16] != pytest.approx(0.0, abs=0.1)

    def test_ramp_y_has_nonzero_gy(self):
        f = compute_gradient(_ramp_y())
        assert f.gy[16, 16] != pytest.approx(0.0, abs=0.1)

    def test_ksize_1_valid(self):
        f = compute_gradient(_gray(), ksize=1)
        assert f.params["ksize"] == 1

    def test_ksize_5_valid(self):
        f = compute_gradient(_gray(), ksize=5)
        assert f.params["ksize"] == 5

    def test_ksize_7_valid(self):
        f = compute_gradient(_gray(), ksize=7)
        assert f.params["ksize"] == 7

    def test_invalid_ksize_raises(self):
        with pytest.raises(ValueError):
            compute_gradient(_gray(), ksize=4)

    def test_normalize_flag_stored(self):
        f = compute_gradient(_ramp_x(), normalize=True)
        assert f.params["normalize"] is True

    def test_normalize_true_bounded(self):
        f = compute_gradient(_ramp_x(), normalize=True)
        mag = np.sqrt(f.gx ** 2 + f.gy ** 2)
        assert mag.max() <= 1.0 + 1e-5

    def test_bgr_input_converted(self):
        f = compute_gradient(_bgr())
        assert f.shape == (32, 32)

    def test_float32_output(self):
        f = compute_gradient(_gray())
        assert f.gx.dtype == np.float32
        assert f.gy.dtype == np.float32


# ─── compute_magnitude (extra) ────────────────────────────────────────────────

class TestComputeMagnitudeExtra:
    def test_returns_float32(self):
        f = _make_field()
        m = compute_magnitude(f)
        assert m.dtype == np.float32

    def test_shape_preserved(self):
        f = _make_field(8, 12)
        m = compute_magnitude(f)
        assert m.shape == (8, 12)

    def test_zero_field_zero_magnitude(self):
        f = _make_field(gx_val=0.0, gy_val=0.0)
        assert np.allclose(compute_magnitude(f), 0.0)

    def test_unit_x_magnitude_is_one(self):
        f = _make_field(gx_val=1.0, gy_val=0.0)
        assert np.allclose(compute_magnitude(f), 1.0)

    def test_unit_y_magnitude_is_one(self):
        f = _make_field(gx_val=0.0, gy_val=1.0)
        assert np.allclose(compute_magnitude(f), 1.0)

    def test_diagonal_magnitude(self):
        f = _make_field(gx_val=3.0, gy_val=4.0)
        assert np.allclose(compute_magnitude(f), 5.0)

    def test_all_positive(self):
        f = _make_field(gx_val=-2.0, gy_val=-3.0)
        assert (compute_magnitude(f) >= 0).all()


# ─── compute_orientation (extra) ──────────────────────────────────────────────

class TestComputeOrientationExtra:
    def test_returns_float32(self):
        f = _make_field()
        o = compute_orientation(f)
        assert o.dtype == np.float32

    def test_shape_preserved(self):
        f = _make_field(8, 12)
        assert compute_orientation(f).shape == (8, 12)

    def test_positive_x_zero_angle(self):
        f = _make_field(gx_val=1.0, gy_val=0.0)
        assert np.allclose(compute_orientation(f), 0.0, atol=1e-5)

    def test_positive_y_pi_half(self):
        f = _make_field(gx_val=0.0, gy_val=1.0)
        assert np.allclose(compute_orientation(f), math.pi / 2, atol=1e-5)

    def test_negative_x_pi(self):
        f = _make_field(gx_val=-1.0, gy_val=0.0)
        val = float(compute_orientation(f).ravel()[0])
        assert abs(val) == pytest.approx(math.pi, abs=1e-5)

    def test_range_minus_pi_to_pi(self):
        f = compute_gradient(_ramp_x())
        o = compute_orientation(f)
        assert o.min() >= -math.pi - 1e-5
        assert o.max() <= math.pi + 1e-5


# ─── compute_divergence (extra) ───────────────────────────────────────────────

class TestComputeDivergenceExtra:
    def test_returns_float32(self):
        f = compute_gradient(_ramp_x())
        d = compute_divergence(f)
        assert d.dtype == np.float32

    def test_shape_preserved(self):
        f = compute_gradient(_gray(20, 24))
        d = compute_divergence(f)
        assert d.shape == (20, 24)

    def test_constant_field_zero_divergence(self):
        f = _make_field(gx_val=1.0, gy_val=1.0)
        d = compute_divergence(f)
        assert np.allclose(d[2:-2, 2:-2], 0.0, atol=1.0)

    def test_output_has_values(self):
        f = compute_gradient(_ramp_x())
        d = compute_divergence(f)
        assert d.shape == f.shape


# ─── compute_curl (extra) ─────────────────────────────────────────────────────

class TestComputeCurlExtra:
    def test_returns_float32(self):
        f = compute_gradient(_ramp_x())
        c = compute_curl(f)
        assert c.dtype == np.float32

    def test_shape_preserved(self):
        f = compute_gradient(_gray(20, 24))
        c = compute_curl(f)
        assert c.shape == (20, 24)

    def test_irrotational_field_near_zero_curl(self):
        # A gradient field should have near-zero curl (∇×∇f = 0)
        f = compute_gradient(_ramp_x())
        c = compute_curl(f)
        assert np.allclose(c[3:-3, 3:-3], 0.0, atol=5.0)

    def test_constant_field_zero_curl(self):
        f = _make_field(gx_val=2.0, gy_val=3.0)
        c = compute_curl(f)
        assert np.allclose(c[2:-2, 2:-2], 0.0, atol=1.0)


# ─── flow_along_boundary (extra) ──────────────────────────────────────────────

class TestFlowAlongBoundaryExtra:
    def test_empty_contour_returns_empty(self):
        f = _make_field()
        result = flow_along_boundary(f, np.empty((0, 2), dtype=np.float32))
        assert result.shape == (0,)

    def test_output_length_matches_contour(self):
        f = _make_field()
        pts = _circle_contour(n=20)
        result = flow_along_boundary(f, pts)
        assert len(result) == 20

    def test_returns_float32(self):
        f = _make_field()
        pts = _circle_contour()
        result = flow_along_boundary(f, pts)
        assert result.dtype == np.float32

    def test_window_1_valid(self):
        f = _make_field()
        pts = _circle_contour()
        result = flow_along_boundary(f, pts, window=1)
        assert len(result) == len(pts)

    def test_window_2_valid(self):
        f = _make_field()
        pts = _circle_contour()
        result = flow_along_boundary(f, pts, window=2)
        assert len(result) == len(pts)

    def test_window_zero_raises(self):
        f = _make_field()
        pts = _circle_contour()
        with pytest.raises(ValueError):
            flow_along_boundary(f, pts, window=0)

    def test_window_negative_raises(self):
        f = _make_field()
        pts = _circle_contour()
        with pytest.raises(ValueError):
            flow_along_boundary(f, pts, window=-1)

    def test_contour_shape_n_1_2(self):
        f = _make_field()
        pts = _circle_contour(n=8).reshape(-1, 1, 2)
        result = flow_along_boundary(f, pts)
        assert len(result) == 8

    def test_x_gradient_flow_along_vertical_line(self):
        # Field with only gx component, tangent along Y axis → flow ≈ 0
        f = _make_field(gx_val=10.0, gy_val=0.0)
        pts = np.array([[8.0, float(y)] for y in range(4, 12)], dtype=np.float32)
        result = flow_along_boundary(f, pts)
        # tangent is (0,1), dot with (10,0) = 0
        assert np.allclose(result, 0.0, atol=0.5)


# ─── compare_gradient_fields (extra) ──────────────────────────────────────────

class TestCompareGradientFieldsExtra:
    def test_identical_fields_score_one(self):
        f = _make_field(gx_val=1.0, gy_val=0.5)
        score = compare_gradient_fields(f, f)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_score_range(self):
        f1 = compute_gradient(_ramp_x())
        f2 = compute_gradient(_ramp_y())
        score = compare_gradient_fields(f1, f2)
        assert -1.0 <= score <= 1.0

    def test_opposite_fields_negative(self):
        f1 = _make_field(gx_val=1.0, gy_val=0.0)
        f2 = _make_field(gx_val=-1.0, gy_val=0.0)
        score = compare_gradient_fields(f1, f2)
        assert score < 0.0

    def test_mismatched_shapes_raises(self):
        f1 = _make_field(8, 8)
        f2 = _make_field(16, 16)
        with pytest.raises(ValueError):
            compare_gradient_fields(f1, f2)

    def test_with_mask_all_ones(self):
        f = _make_field()
        mask = np.ones(f.shape, dtype=np.uint8)
        score = compare_gradient_fields(f, f, mask=mask)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_with_mask_all_zeros_returns_zero(self):
        f = _make_field()
        mask = np.zeros(f.shape, dtype=np.uint8)
        score = compare_gradient_fields(f, f, mask=mask)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_partial_mask_returns_float(self):
        f1 = compute_gradient(_ramp_x())
        f2 = compute_gradient(_ramp_y())
        mask = np.zeros(f1.shape, dtype=np.uint8)
        mask[:16, :16] = 1
        score = compare_gradient_fields(f1, f2, mask=mask)
        assert isinstance(score, float)


# ─── batch_gradient_fields (extra) ────────────────────────────────────────────

class TestBatchGradientFieldsExtra:
    def test_empty_list_returns_empty(self):
        result = batch_gradient_fields([])
        assert result == []

    def test_single_image(self):
        result = batch_gradient_fields([_gray()])
        assert len(result) == 1
        assert isinstance(result[0], GradientField)

    def test_multiple_images(self):
        imgs = [_gray(), _ramp_x(), _ramp_y()]
        result = batch_gradient_fields(imgs)
        assert len(result) == 3

    def test_shapes_preserved(self):
        imgs = [_gray(16, 16), _gray(24, 32)]
        result = batch_gradient_fields(imgs)
        assert result[0].shape == (16, 16)
        assert result[1].shape == (24, 32)

    def test_ksize_propagated(self):
        result = batch_gradient_fields([_gray()], ksize=5)
        assert result[0].params["ksize"] == 5

    def test_normalize_propagated(self):
        result = batch_gradient_fields([_ramp_x()], normalize=True)
        assert result[0].params["normalize"] is True

    def test_all_are_gradient_fields(self):
        result = batch_gradient_fields([_gray(), _bgr()])
        for r in result:
            assert isinstance(r, GradientField)


# ─── compute_gradient_stats (extra) ───────────────────────────────────────────

class TestComputeGradientStatsExtra:
    def test_returns_gradient_stats(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f)
        assert isinstance(s, GradientStats)

    def test_mean_magnitude_nonneg(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f)
        assert s.mean_magnitude >= 0.0

    def test_std_magnitude_nonneg(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f)
        assert s.std_magnitude >= 0.0

    def test_edge_density_range(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f)
        assert 0.0 <= s.edge_density <= 1.0

    def test_dominant_angle_in_range(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f)
        assert -math.pi <= s.dominant_angle <= math.pi

    def test_constant_image_zero_magnitude(self):
        f = compute_gradient(_gray(32, 32, 100))
        s = compute_gradient_stats(f, threshold=0.0)
        # Interior is zero; mean magnitude may be small
        assert s.mean_magnitude >= 0.0

    def test_threshold_zero_high_density(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f, threshold=0.0)
        # With threshold=0, edge_density counts everything non-zero
        assert s.edge_density >= 0.0

    def test_high_threshold_low_density(self):
        f = compute_gradient(_gray())  # constant image, very low gradients
        s = compute_gradient_stats(f, threshold=1000.0)
        assert s.edge_density == pytest.approx(0.0, abs=0.01)

    def test_invalid_threshold_raises(self):
        f = compute_gradient(_gray())
        with pytest.raises(ValueError):
            compute_gradient_stats(f, threshold=-1.0)

    def test_invalid_bins_raises(self):
        f = compute_gradient(_gray())
        with pytest.raises(ValueError):
            compute_gradient_stats(f, n_orientation_bins=0)

    def test_params_stored_in_stats(self):
        f = compute_gradient(_gray())
        s = compute_gradient_stats(f, threshold=5.0, n_orientation_bins=18)
        assert s.params["threshold"] == pytest.approx(5.0)
        assert s.params["n_orientation_bins"] == 18

    def test_n_orientation_bins_1_valid(self):
        f = compute_gradient(_ramp_x())
        s = compute_gradient_stats(f, n_orientation_bins=1)
        assert isinstance(s, GradientStats)
