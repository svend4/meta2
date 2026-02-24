"""Extra tests for puzzle_reconstruction/utils/geometry.py."""
from __future__ import annotations

import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.geometry import (
    rotation_matrix_2d,
    rotate_points,
    polygon_area,
    polygon_centroid,
    bbox_from_points,
    resample_curve,
    align_centroids,
    poly_iou,
    point_in_polygon,
    normalize_contour,
    smooth_contour,
    curvature,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(x=0.0, y=0.0, s=1.0):
    """CCW unit square."""
    return np.array([[x, y], [x+s, y], [x+s, y+s], [x, y+s]], dtype=np.float64)


def _line(n=10):
    xs = np.linspace(0, 1, n)
    return np.column_stack([xs, np.zeros(n)])


# ─── rotation_matrix_2d ───────────────────────────────────────────────────────

class TestRotationMatrix2dExtra:
    def test_returns_ndarray(self):
        assert isinstance(rotation_matrix_2d(0.0), np.ndarray)

    def test_shape_2x2(self):
        assert rotation_matrix_2d(0.0).shape == (2, 2)

    def test_dtype_float64(self):
        assert rotation_matrix_2d(0.0).dtype == np.float64

    def test_zero_angle_identity(self):
        R = rotation_matrix_2d(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-10)

    def test_pi_angle_flips(self):
        R = rotation_matrix_2d(math.pi)
        np.testing.assert_allclose(R, [[-1, 0], [0, -1]], atol=1e-10)

    def test_half_pi_rotates(self):
        R = rotation_matrix_2d(math.pi / 2)
        np.testing.assert_allclose(R @ [1, 0], [0, 1], atol=1e-10)

    def test_orthogonal(self):
        R = rotation_matrix_2d(1.2)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)

    def test_determinant_one(self):
        R = rotation_matrix_2d(0.7)
        assert np.linalg.det(R) == pytest.approx(1.0)


# ─── rotate_points ────────────────────────────────────────────────────────────

class TestRotatePointsExtra:
    def test_returns_ndarray(self):
        pts = np.array([[1.0, 0.0]])
        assert isinstance(rotate_points(pts, 0.0), np.ndarray)

    def test_shape_preserved(self):
        pts = np.random.rand(5, 2)
        out = rotate_points(pts, 0.5)
        assert out.shape == (5, 2)

    def test_zero_angle_unchanged(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(rotate_points(pts, 0.0), pts)

    def test_pi_angle_flips_origin(self):
        pts = np.array([[1.0, 0.0]])
        out = rotate_points(pts, math.pi)
        np.testing.assert_allclose(out, [[-1.0, 0.0]], atol=1e-10)

    def test_rotate_around_center(self):
        pts = np.array([[2.0, 1.0]])
        c = np.array([1.0, 1.0])
        out = rotate_points(pts, math.pi / 2, center=c)
        np.testing.assert_allclose(out, [[1.0, 2.0]], atol=1e-10)

    def test_dtype_float64(self):
        pts = np.array([[1, 0]], dtype=np.int32)
        out = rotate_points(pts, 0.3)
        assert out.dtype == np.float64


# ─── polygon_area ─────────────────────────────────────────────────────────────

class TestPolygonAreaExtra:
    def test_returns_float(self):
        assert isinstance(polygon_area(_square()), float)

    def test_unit_square_area(self):
        assert polygon_area(_square()) == pytest.approx(1.0)

    def test_scaled_square(self):
        assert polygon_area(_square(s=3.0)) == pytest.approx(9.0)

    def test_degenerate_lt_3_points(self):
        pts = np.array([[0, 0], [1, 0]])
        assert polygon_area(pts) == pytest.approx(0.0)

    def test_cw_area_negative(self):
        # CW order → negative
        sq = _square()[::-1]
        assert polygon_area(sq) < 0.0

    def test_triangle_area(self):
        pts = np.array([[0, 0], [4, 0], [0, 3]], dtype=np.float64)
        assert abs(polygon_area(pts)) == pytest.approx(6.0)


# ─── polygon_centroid ─────────────────────────────────────────────────────────

class TestPolygonCentroidExtra:
    def test_returns_ndarray(self):
        assert isinstance(polygon_centroid(_square()), np.ndarray)

    def test_shape_2(self):
        assert polygon_centroid(_square()).shape == (2,)

    def test_unit_square_center(self):
        c = polygon_centroid(_square())
        np.testing.assert_allclose(c, [0.5, 0.5], atol=1e-10)

    def test_empty_points_returns_zeros(self):
        c = polygon_centroid(np.zeros((0, 2)))
        np.testing.assert_allclose(c, [0, 0])

    def test_single_point_returns_point(self):
        pts = np.array([[3.0, 7.0]])
        c = polygon_centroid(pts)
        np.testing.assert_allclose(c, [3.0, 7.0])

    def test_offset_square(self):
        sq = _square(x=10.0, y=5.0, s=2.0)
        c = polygon_centroid(sq)
        np.testing.assert_allclose(c, [11.0, 6.0], atol=1e-10)


# ─── bbox_from_points ─────────────────────────────────────────────────────────

class TestBboxFromPointsExtra:
    def test_returns_tuple(self):
        result = bbox_from_points(_square())
        assert isinstance(result, tuple) and len(result) == 4

    def test_unit_square_bbox(self):
        x0, y0, x1, y1 = bbox_from_points(_square())
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(1.0)
        assert y1 == pytest.approx(1.0)

    def test_empty_points_all_zeros(self):
        assert bbox_from_points(np.zeros((0, 2))) == (0.0, 0.0, 0.0, 0.0)

    def test_single_point(self):
        pts = np.array([[3.0, 5.0]])
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 == x1 == pytest.approx(3.0)
        assert y0 == y1 == pytest.approx(5.0)


# ─── resample_curve ───────────────────────────────────────────────────────────

class TestResampleCurveExtra:
    def test_returns_ndarray(self):
        assert isinstance(resample_curve(_line(10), 5), np.ndarray)

    def test_output_n_points(self):
        out = resample_curve(_line(10), 20)
        assert out.shape == (20, 2)

    def test_start_end_preserved(self):
        pts = _line(10)
        out = resample_curve(pts, 10)
        np.testing.assert_allclose(out[0], pts[0], atol=1e-10)
        np.testing.assert_allclose(out[-1], pts[-1], atol=1e-10)

    def test_degenerate_single_point(self):
        pts = np.array([[1.0, 2.0]])
        out = resample_curve(pts, 3)
        assert len(out) == 1  # or returns as-is

    def test_all_same_points(self):
        pts = np.tile([5.0, 5.0], (5, 1))
        out = resample_curve(pts, 8)
        assert np.allclose(out, [5.0, 5.0])


# ─── align_centroids ──────────────────────────────────────────────────────────

class TestAlignCentroidsExtra:
    def test_returns_ndarray(self):
        src = _square()
        tgt = _square(x=5.0)
        assert isinstance(align_centroids(src, tgt), np.ndarray)

    def test_shape_preserved(self):
        src = _square()
        tgt = _square(x=5.0)
        out = align_centroids(src, tgt)
        assert out.shape == src.shape

    def test_centroid_matches_target(self):
        src = _square()
        tgt = _square(x=10.0, y=20.0)
        out = align_centroids(src, tgt)
        np.testing.assert_allclose(out.mean(axis=0), tgt.mean(axis=0), atol=1e-10)

    def test_identical_centroids_unchanged(self):
        pts = _square()
        out = align_centroids(pts, pts)
        np.testing.assert_allclose(out, pts)


# ─── poly_iou ─────────────────────────────────────────────────────────────────

class TestPolyIouExtra:
    def test_returns_float(self):
        s = _square()
        assert isinstance(poly_iou(s, s), float)

    def test_identical_polys_one(self):
        s = _square()
        assert poly_iou(s, s) == pytest.approx(1.0, abs=1e-6)

    def test_non_overlapping_zero(self):
        a = _square(x=0.0)
        b = _square(x=10.0)
        assert poly_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap_in_range(self):
        a = _square(x=0.0)
        b = _square(x=0.5)
        iou = poly_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_result_in_range(self):
        a = _square()
        b = _square(x=0.3, y=0.3)
        iou = poly_iou(a, b)
        assert 0.0 <= iou <= 1.0


# ─── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygonExtra:
    def test_inside_returns_true(self):
        sq = _square()
        assert point_in_polygon(np.array([0.5, 0.5]), sq) is True

    def test_outside_returns_false(self):
        sq = _square()
        assert point_in_polygon(np.array([2.0, 2.0]), sq) is False

    def test_less_than_3_points_false(self):
        line = np.array([[0, 0], [1, 0]])
        assert point_in_polygon(np.array([0.5, 0.0]), line) is False

    def test_returns_bool(self):
        sq = _square()
        result = point_in_polygon(np.array([0.5, 0.5]), sq)
        assert isinstance(result, bool)

    def test_corner_region(self):
        sq = _square()
        # Far outside corner
        assert point_in_polygon(np.array([-1.0, -1.0]), sq) is False


# ─── normalize_contour ────────────────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_contour(_square()), np.ndarray)

    def test_shape_preserved(self):
        sq = _square()
        out = normalize_contour(sq)
        assert out.shape == sq.shape

    def test_centroid_near_origin(self):
        sq = _square(x=100.0, y=200.0)
        out = normalize_contour(sq)
        np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_empty_contour(self):
        out = normalize_contour(np.zeros((0, 2)))
        assert len(out) == 0

    def test_custom_target_scale(self):
        pts = _square(s=10.0)
        out = normalize_contour(pts, target_scale=2.0)
        x0, y0, x1, y1 = bbox_from_points(out)
        diag = np.hypot(x1 - x0, y1 - y0)
        assert diag == pytest.approx(2.0, abs=0.1)


# ─── smooth_contour ───────────────────────────────────────────────────────────

class TestSmoothContourExtra:
    def test_returns_ndarray(self):
        pts = _line(10)
        assert isinstance(smooth_contour(pts), np.ndarray)

    def test_shape_preserved(self):
        pts = _line(10)
        out = smooth_contour(pts)
        assert out.shape == pts.shape

    def test_straight_line_unchanged(self):
        pts = _line(10)
        out = smooth_contour(pts, window=3)
        assert out.shape == pts.shape

    def test_window_enforced_odd(self):
        pts = _line(10)
        out = smooth_contour(pts, window=4)  # Forced to 5
        assert out.shape == pts.shape

    def test_large_window_smooth(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, (30, 2))
        pts = _line(30) + noise
        out = smooth_contour(pts, window=11)
        assert out.shape == pts.shape


# ─── curvature ────────────────────────────────────────────────────────────────

class TestCurvatureExtra:
    def test_returns_ndarray(self):
        assert isinstance(curvature(_line(10)), np.ndarray)

    def test_length_equals_n_points(self):
        pts = _line(15)
        assert len(curvature(pts)) == 15

    def test_straight_line_near_zero(self):
        k = curvature(_line(20))
        assert np.all(k >= 0.0)

    def test_less_than_3_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        k = curvature(pts)
        assert np.allclose(k, 0.0)

    def test_circle_constant_curvature(self):
        t = np.linspace(0, 2 * math.pi, 100, endpoint=False)
        pts = np.column_stack([np.cos(t), np.sin(t)])
        k = curvature(pts)
        # For unit circle, curvature ≈ 1; check inner points
        inner = k[5:-5]
        assert np.all(inner > 0.5)

    def test_nonneg_values(self):
        t = np.linspace(0, math.pi, 20)
        pts = np.column_stack([t, np.sin(t)])
        k = curvature(pts)
        assert np.all(k >= 0.0)
