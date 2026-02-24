"""Extra tests for puzzle_reconstruction/utils/geometry.py."""
from __future__ import annotations

import math
import numpy as np
import pytest

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

def _square_ccw():
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def _square_cw():
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)


def _circle_pts(n=60, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)


# ─── rotation_matrix_2d (extra) ───────────────────────────────────────────────

class TestRotationMatrix2dExtra:
    def test_shape_is_2x2(self):
        R = rotation_matrix_2d(0.5)
        assert R.shape == (2, 2)

    def test_identity_at_zero(self):
        R = rotation_matrix_2d(0.0)
        assert np.allclose(R, np.eye(2), atol=1e-12)

    def test_det_one_various_angles(self):
        for angle in [0.1, 0.5, math.pi / 4, math.pi, 2 * math.pi, -1.3]:
            assert math.isclose(np.linalg.det(rotation_matrix_2d(angle)), 1.0, abs_tol=1e-10)

    def test_orthogonal(self):
        R = rotation_matrix_2d(math.pi / 6)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)

    def test_pi_over_2_rotates_x_to_y(self):
        R = rotation_matrix_2d(math.pi / 2)
        out = R @ np.array([1.0, 0.0])
        assert np.allclose(out, [0.0, 1.0], atol=1e-10)

    def test_pi_flips_x(self):
        R = rotation_matrix_2d(math.pi)
        out = R @ np.array([1.0, 0.0])
        assert np.allclose(out, [-1.0, 0.0], atol=1e-10)

    def test_negative_angle(self):
        R_pos = rotation_matrix_2d(math.pi / 4)
        R_neg = rotation_matrix_2d(-math.pi / 4)
        assert np.allclose(R_pos @ R_neg, np.eye(2), atol=1e-10)

    def test_full_rotation_is_identity(self):
        R = rotation_matrix_2d(2 * math.pi)
        assert np.allclose(R, np.eye(2), atol=1e-10)

    def test_composition(self):
        R1 = rotation_matrix_2d(math.pi / 4)
        R2 = rotation_matrix_2d(math.pi / 4)
        R_half = rotation_matrix_2d(math.pi / 2)
        assert np.allclose(R1 @ R2, R_half, atol=1e-10)


# ─── rotate_points (extra) ────────────────────────────────────────────────────

class TestRotatePointsExtra:
    def test_output_shape_preserved(self):
        pts = np.random.rand(15, 2)
        out = rotate_points(pts, 0.7)
        assert out.shape == (15, 2)

    def test_zero_angle_unchanged(self):
        pts = np.random.rand(10, 2)
        assert np.allclose(rotate_points(pts, 0.0), pts)

    def test_rotate_around_origin_pi_over_2(self):
        pts = np.array([[1.0, 0.0]])
        out = rotate_points(pts, math.pi / 2)
        assert np.allclose(out, [[0.0, 1.0]], atol=1e-10)

    def test_rotate_around_custom_center(self):
        pts = np.array([[2.0, 1.0]])
        center = np.array([1.0, 1.0])
        out = rotate_points(pts, math.pi / 2, center=center)
        assert np.allclose(out, [[1.0, 2.0]], atol=1e-10)

    def test_full_rotation_returns_original(self):
        pts = _circle_pts(20)
        out = rotate_points(pts, 2 * math.pi)
        assert np.allclose(out, pts, atol=1e-10)

    def test_single_point(self):
        pts = np.array([[3.0, 4.0]])
        out = rotate_points(pts, math.pi)
        assert out.shape == (1, 2)

    def test_180_deg_negates_from_origin(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = rotate_points(pts, math.pi)
        assert np.allclose(out, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-10)

    def test_center_zero_same_as_none(self):
        pts = np.random.rand(5, 2)
        out_none = rotate_points(pts, 0.3)
        out_zero = rotate_points(pts, 0.3, center=np.array([0.0, 0.0]))
        assert np.allclose(out_none, out_zero, atol=1e-10)


# ─── polygon_area (extra) ─────────────────────────────────────────────────────

class TestPolygonAreaExtra:
    def test_unit_square_abs_one(self):
        assert math.isclose(abs(polygon_area(_square_ccw())), 1.0, abs_tol=1e-10)

    def test_ccw_positive(self):
        assert polygon_area(_square_ccw()) > 0

    def test_cw_negative(self):
        assert polygon_area(_square_cw()) < 0

    def test_triangle_area_6(self):
        tri = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)
        assert math.isclose(abs(polygon_area(tri)), 6.0, abs_tol=1e-10)

    def test_zero_for_less_than_3_points(self):
        assert polygon_area(np.array([[0, 0], [1, 1]], dtype=float)) == 0.0

    def test_zero_for_single_point(self):
        assert polygon_area(np.array([[5.0, 5.0]])) == 0.0

    def test_degenerate_collinear(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        assert math.isclose(polygon_area(pts), 0.0, abs_tol=1e-10)

    def test_scaled_square_area(self):
        pts = _square_ccw() * 3
        assert math.isclose(abs(polygon_area(pts)), 9.0, abs_tol=1e-10)

    def test_circle_approx_pi(self):
        pts = _circle_pts(n=1000)
        assert math.isclose(abs(polygon_area(pts)), math.pi, rel_tol=1e-3)

    def test_rectangle_area(self):
        pts = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float)
        assert math.isclose(abs(polygon_area(pts)), 12.0, abs_tol=1e-10)


# ─── polygon_centroid (extra) ─────────────────────────────────────────────────

class TestPolygonCentroidExtra:
    def test_unit_square_centroid(self):
        c = polygon_centroid(_square_ccw())
        assert np.allclose(c, [0.5, 0.5], atol=1e-10)

    def test_rectangle_centroid(self):
        pts = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)
        c = polygon_centroid(pts)
        assert np.allclose(c, [2.0, 1.0], atol=1e-8)

    def test_triangle_centroid(self):
        tri = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
        c = polygon_centroid(tri)
        assert np.allclose(c, [2.0, 2.0], atol=1e-8)

    def test_output_shape(self):
        c = polygon_centroid(_square_ccw())
        assert c.shape == (2,)

    def test_empty_returns_shape_2(self):
        c = polygon_centroid(np.zeros((0, 2)))
        assert c.shape == (2,)

    def test_two_points_mean(self):
        pts = np.array([[0.0, 0.0], [4.0, 2.0]])
        c = polygon_centroid(pts)
        assert np.allclose(c, [2.0, 1.0], atol=1e-10)

    def test_symmetric_polygon(self):
        pts = _circle_pts(100, r=3.0)
        c = polygon_centroid(pts)
        assert np.allclose(c, [0.0, 0.0], atol=1e-3)

    def test_shifted_polygon(self):
        pts = _square_ccw() + np.array([10.0, 5.0])
        c = polygon_centroid(pts)
        assert np.allclose(c, [10.5, 5.5], atol=1e-8)


# ─── bbox_from_points (extra) ─────────────────────────────────────────────────

class TestBboxFromPointsExtra:
    def test_known_bbox(self):
        pts = np.array([[1, 2], [3, 4], [5, 0]], dtype=float)
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 == 1.0 and y0 == 0.0 and x1 == 5.0 and y1 == 4.0

    def test_single_point(self):
        x0, y0, x1, y1 = bbox_from_points(np.array([[3.0, 7.0]]))
        assert x0 == x1 == 3.0 and y0 == y1 == 7.0

    def test_returns_4_values(self):
        bbox = bbox_from_points(_square_ccw())
        assert len(bbox) == 4

    def test_x0_le_x1(self):
        pts = np.random.rand(20, 2) * 100
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 <= x1 and y0 <= y1

    def test_empty_returns_4_values(self):
        bbox = bbox_from_points(np.zeros((0, 2)))
        assert len(bbox) == 4

    def test_negative_coords(self):
        pts = np.array([[-3, -5], [-1, -2]], dtype=float)
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 == -3.0 and y0 == -5.0 and x1 == -1.0 and y1 == -2.0

    def test_large_set(self):
        pts = np.random.rand(100, 2) * 1000
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 <= pts[:, 0].min() + 1e-9
        assert x1 >= pts[:, 0].max() - 1e-9


# ─── resample_curve (extra) ───────────────────────────────────────────────────

class TestResampleCurveExtra:
    def test_output_length(self):
        pts = _circle_pts(30)
        assert len(resample_curve(pts, 50)) == 50

    def test_n_equals_input_length(self):
        pts = _circle_pts(20)
        out = resample_curve(pts, 20)
        assert len(out) == 20

    def test_n_2_gives_two_points(self):
        pts = _circle_pts(20)
        out = resample_curve(pts, 2)
        assert len(out) == 2

    def test_straight_line_uniform_spacing(self):
        pts = np.column_stack([np.linspace(0, 10, 11), np.zeros(11)])
        out = resample_curve(pts, 11)
        dists = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert np.allclose(dists, dists[0], rtol=1e-5)

    def test_constant_points_no_crash(self):
        pts = np.tile([3.0, 4.0], (10, 1))
        out = resample_curve(pts, 5)
        assert out.shape[0] == 5

    def test_output_shape_2d(self):
        pts = _circle_pts(20)
        out = resample_curve(pts, 15)
        assert out.ndim == 2 and out.shape[1] == 2

    def test_upsample_larger_n(self):
        pts = _circle_pts(10)
        out = resample_curve(pts, 100)
        assert len(out) == 100

    def test_downsample_smaller_n(self):
        pts = _circle_pts(100)
        out = resample_curve(pts, 10)
        assert len(out) == 10


# ─── align_centroids (extra) ──────────────────────────────────────────────────

class TestAlignCentroidsExtra:
    def test_centroid_matches_target(self):
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        tgt = np.array([[5, 5], [6, 5], [5, 6]], dtype=float)
        out = align_centroids(src, tgt)
        assert np.allclose(out.mean(axis=0), tgt.mean(axis=0), atol=1e-10)

    def test_shape_preserved(self):
        src = _circle_pts(30)
        tgt = _circle_pts(20)
        out = align_centroids(src, tgt)
        assert out.shape == src.shape

    def test_already_aligned_unchanged(self):
        pts = _circle_pts(20)
        out = align_centroids(pts, pts)
        assert np.allclose(out, pts, atol=1e-10)

    def test_translation_only(self):
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        shift = np.array([10.0, -5.0])
        tgt = src + shift
        out = align_centroids(src, tgt)
        assert np.allclose(out, tgt, atol=1e-10)

    def test_relative_shape_preserved(self):
        src = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        tgt = np.array([[5, 5], [6, 5], [5, 6]], dtype=float)
        out = align_centroids(src, tgt)
        diffs_src = src - src.mean(axis=0)
        diffs_out = out - out.mean(axis=0)
        assert np.allclose(diffs_src, diffs_out, atol=1e-10)


# ─── poly_iou (extra) ─────────────────────────────────────────────────────────

class TestPolyIouExtra:
    def test_identical_is_one(self):
        sq = _square_ccw()
        assert math.isclose(poly_iou(sq, sq), 1.0, abs_tol=1e-6)

    def test_no_overlap_is_zero(self):
        sq1 = _square_ccw()
        sq2 = sq1 + np.array([5.0, 0.0])
        assert math.isclose(poly_iou(sq1, sq2), 0.0, abs_tol=1e-6)

    def test_partial_overlap_in_range(self):
        sq1 = _square_ccw()
        sq2 = sq1 + np.array([0.5, 0.0])
        iou = poly_iou(sq1, sq2)
        assert 0.0 < iou < 1.0

    def test_containment(self):
        outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        inner = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=float)
        iou = poly_iou(inner, outer)
        assert math.isclose(iou, 1 / 16, rel_tol=1e-3)

    def test_symmetric(self):
        a = _square_ccw()
        b = a + np.array([0.3, 0.3])
        assert math.isclose(poly_iou(a, b), poly_iou(b, a), abs_tol=1e-8)

    def test_iou_in_range(self):
        a = _square_ccw()
        b = a + np.array([0.4, 0.4])
        iou = poly_iou(a, b)
        assert 0.0 <= iou <= 1.0

    def test_large_polygons(self):
        big = _square_ccw() * 10
        small = _square_ccw() * 2
        iou = poly_iou(big, small)
        assert math.isclose(iou, 4 / 100, rel_tol=1e-3)


# ─── point_in_polygon (extra) ─────────────────────────────────────────────────

class TestPointInPolygonExtra:
    def test_center_inside_square(self):
        assert point_in_polygon([0.5, 0.5], _square_ccw())

    def test_outside_square(self):
        assert not point_in_polygon([2.0, 2.0], _square_ccw())

    def test_origin_inside_centered_square(self):
        sq = _square_ccw() - 0.5
        assert point_in_polygon([0.0, 0.0], sq)

    def test_less_than_3_pts_returns_false(self):
        assert not point_in_polygon([0.5, 0.5],
                                    np.array([[0, 0], [1, 1]], dtype=float))

    def test_inside_circle(self):
        circ = _circle_pts(100, r=5.0)
        assert point_in_polygon([0.0, 0.0], circ)

    def test_outside_circle(self):
        circ = _circle_pts(100, r=1.0)
        assert not point_in_polygon([3.0, 3.0], circ)

    def test_far_outside(self):
        assert not point_in_polygon([100.0, 100.0], _square_ccw())

    def test_negative_coords_inside(self):
        pts = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], dtype=float)
        assert point_in_polygon([0.0, 0.0], pts)


# ─── normalize_contour (extra) ────────────────────────────────────────────────

class TestNormalizeContourExtra:
    def test_centroid_at_origin(self):
        pts = _square_ccw() * 5 + np.array([10.0, 20.0])
        out = normalize_contour(pts)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-10)

    def test_scale_one_diag(self):
        pts = np.random.RandomState(42).rand(30, 2) * 100 + 50
        out = normalize_contour(pts, target_scale=1.0)
        x0, y0, x1, y1 = bbox_from_points(out)
        diag = math.hypot(x1 - x0, y1 - y0)
        assert math.isclose(diag, 1.0, rel_tol=1e-5)

    def test_custom_scale(self):
        pts = _square_ccw() * 3
        out = normalize_contour(pts, target_scale=2.0)
        x0, y0, x1, y1 = bbox_from_points(out)
        diag = math.hypot(x1 - x0, y1 - y0)
        assert math.isclose(diag, 2.0, rel_tol=1e-5)

    def test_empty_returns_empty(self):
        out = normalize_contour(np.zeros((0, 2)))
        assert len(out) == 0

    def test_constant_pts_no_crash(self):
        pts = np.tile([3.0, 4.0], (10, 1))
        out = normalize_contour(pts)
        assert out.shape == pts.shape

    def test_shape_preserved(self):
        pts = _circle_pts(40)
        out = normalize_contour(pts, target_scale=1.0)
        assert out.shape == pts.shape


# ─── smooth_contour (extra) ───────────────────────────────────────────────────

class TestSmoothContourExtra:
    def test_output_shape(self):
        pts = _circle_pts(50)
        out = smooth_contour(pts, window=5)
        assert out.shape == pts.shape

    def test_constant_curve_unchanged(self):
        pts = np.tile([5.0, 3.0], (20, 1))
        out = smooth_contour(pts, window=5)
        assert np.allclose(out, pts, atol=1e-10)

    def test_spike_reduced(self):
        pts = _circle_pts(40)
        pts_noisy = pts.copy()
        pts_noisy[10] += np.array([10.0, 10.0])
        out = smooth_contour(pts_noisy, window=7)
        before = np.linalg.norm(pts_noisy[10] - pts[10])
        after = np.linalg.norm(out[10] - pts[10])
        assert after < before

    def test_even_window_no_crash(self):
        pts = _circle_pts(20)
        out = smooth_contour(pts, window=4)
        assert out.shape == pts.shape

    def test_window_one_no_crash(self):
        pts = _circle_pts(20)
        out = smooth_contour(pts, window=1)
        assert out.shape == pts.shape

    def test_small_window_less_smooth(self):
        pts = _circle_pts(40)
        pts_noisy = pts.copy()
        pts_noisy[15] += np.array([5.0, 5.0])
        out3 = smooth_contour(pts_noisy, window=3)
        out11 = smooth_contour(pts_noisy, window=11)
        diff3 = np.linalg.norm(out3[15] - pts[15])
        diff11 = np.linalg.norm(out11[15] - pts[15])
        assert diff11 <= diff3


# ─── curvature (extra) ────────────────────────────────────────────────────────

class TestCurvatureExtra:
    def test_output_shape_matches_input(self):
        pts = _circle_pts(50)
        out = curvature(pts)
        assert out.shape == (50,)

    def test_nonneg_values(self):
        pts = _circle_pts(40)
        out = curvature(pts)
        assert np.all(out >= 0)

    def test_straight_line_near_zero(self):
        pts = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
        out = curvature(pts)
        assert float(out[5:-5].mean()) < 1e-3

    def test_circle_near_constant(self):
        pts = _circle_pts(200, r=5.0)
        out = curvature(pts)
        inner = out[10:-10]
        cv = inner.std() / (inner.mean() + 1e-12)
        assert cv < 0.5

    def test_short_curve_no_crash(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = curvature(pts)
        assert out.shape == (2,)

    def test_single_point_no_crash(self):
        pts = np.array([[0.0, 0.0]])
        out = curvature(pts)
        assert out.shape[0] >= 1 or out.shape[0] == 0

    def test_circle_curvature_inverse_radius(self):
        r = 5.0
        pts = _circle_pts(300, r=r)
        out = curvature(pts)
        inner = out[20:-20]
        expected = 1.0 / r
        assert math.isclose(float(inner.mean()), expected, rel_tol=0.2)

    def test_larger_radius_smaller_curvature(self):
        out5 = curvature(_circle_pts(200, r=5.0))[10:-10]
        out10 = curvature(_circle_pts(200, r=10.0))[10:-10]
        assert out5.mean() > out10.mean()
