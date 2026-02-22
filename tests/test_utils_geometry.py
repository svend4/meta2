"""Тесты для puzzle_reconstruction.utils.geometry."""
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

def _square(cx=0.0, cy=0.0, r=1.0) -> np.ndarray:
    """CCW square centred at (cx,cy) with half-side r."""
    return np.array([
        [cx - r, cy - r],
        [cx + r, cy - r],
        [cx + r, cy + r],
        [cx - r, cy + r],
    ], dtype=np.float64)


def _circle(n=32, r=1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


# ─── TestRotationMatrix2d ─────────────────────────────────────────────────────

class TestRotationMatrix2d:
    def test_shape(self):
        R = rotation_matrix_2d(0.0)
        assert R.shape == (2, 2)

    def test_dtype(self):
        R = rotation_matrix_2d(math.pi / 4)
        assert R.dtype == np.float64

    def test_zero_angle_identity(self):
        R = rotation_matrix_2d(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-12)

    def test_pi_over_two(self):
        R = rotation_matrix_2d(math.pi / 2)
        # [[0,-1],[1,0]]
        np.testing.assert_allclose(R[0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(R[0, 1], -1.0, atol=1e-12)
        np.testing.assert_allclose(R[1, 0], 1.0, atol=1e-12)

    def test_determinant_one(self):
        for angle in [0.0, 0.5, math.pi, -math.pi / 3]:
            R = rotation_matrix_2d(angle)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal(self):
        R = rotation_matrix_2d(1.2)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)


# ─── TestRotatePoints ─────────────────────────────────────────────────────────

class TestRotatePoints:
    def test_returns_ndarray(self):
        pts = _square()
        out = rotate_points(pts, 0.0)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        pts = _square()
        out = rotate_points(pts, math.pi / 4)
        assert out.shape == pts.shape

    def test_zero_rotation_unchanged(self):
        pts = _square()
        out = rotate_points(pts, 0.0)
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_360_rotation_same(self):
        pts = _square()
        out = rotate_points(pts, 2 * math.pi)
        np.testing.assert_allclose(out, pts, atol=1e-10)

    def test_distances_preserved(self):
        pts = _square()
        out = rotate_points(pts, 1.23)
        dists_in = np.linalg.norm(pts, axis=1)
        dists_out = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(dists_out, dists_in, atol=1e-10)

    def test_center_rotation(self):
        pts = np.array([[1.0, 0.0]])
        center = np.array([1.0, 0.0])
        out = rotate_points(pts, math.pi, center)
        np.testing.assert_allclose(out, pts, atol=1e-10)

    def test_dtype_float64(self):
        pts = _square()
        out = rotate_points(pts, 0.3)
        assert out.dtype == np.float64


# ─── TestPolygonArea ──────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_unit_square_ccw(self):
        # CCW unit square: area = 4 * r^2 = 4*1^2 = 4
        pts = _square()
        area = polygon_area(pts)
        assert area == pytest.approx(4.0, abs=1e-10)

    def test_unit_square_cw_negative(self):
        pts = _square()[::-1]  # CW → negative area
        assert polygon_area(pts) < 0.0

    def test_triangle(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        assert polygon_area(pts) == pytest.approx(2.0, abs=1e-10)

    def test_degenerate_line_zero(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert polygon_area(pts) == pytest.approx(0.0)

    def test_degenerate_single_point(self):
        pts = np.array([[5.0, 3.0]])
        assert polygon_area(pts) == pytest.approx(0.0)

    def test_empty_zero(self):
        assert polygon_area(np.zeros((0, 2))) == pytest.approx(0.0)


# ─── TestPolygonCentroid ──────────────────────────────────────────────────────

class TestPolygonCentroid:
    def test_square_centroid_at_origin(self):
        pts = _square()
        c = polygon_centroid(pts)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=1e-10)

    def test_centroid_shifted_square(self):
        pts = _square(cx=5.0, cy=-3.0)
        c = polygon_centroid(pts)
        np.testing.assert_allclose(c, [5.0, -3.0], atol=1e-10)

    def test_returns_shape_2(self):
        c = polygon_centroid(_square())
        assert c.shape == (2,)

    def test_empty_zero(self):
        c = polygon_centroid(np.zeros((0, 2)))
        np.testing.assert_array_equal(c, [0.0, 0.0])

    def test_two_points_midpoint(self):
        pts = np.array([[0.0, 0.0], [4.0, 0.0]])
        c = polygon_centroid(pts)
        np.testing.assert_allclose(c, [2.0, 0.0], atol=1e-10)


# ─── TestBboxFromPoints ───────────────────────────────────────────────────────

class TestBboxFromPoints:
    def test_square_bbox(self):
        pts = _square()
        xmin, ymin, xmax, ymax = bbox_from_points(pts)
        assert xmin == pytest.approx(-1.0)
        assert ymin == pytest.approx(-1.0)
        assert xmax == pytest.approx(1.0)
        assert ymax == pytest.approx(1.0)

    def test_empty_zeros(self):
        bb = bbox_from_points(np.zeros((0, 2)))
        assert bb == (0.0, 0.0, 0.0, 0.0)

    def test_single_point(self):
        bb = bbox_from_points(np.array([[3.0, 4.0]]))
        assert bb[0] == pytest.approx(3.0)
        assert bb[2] == pytest.approx(3.0)

    def test_returns_tuple_of_floats(self):
        bb = bbox_from_points(_square())
        assert len(bb) == 4
        for v in bb:
            assert isinstance(v, float)

    def test_xmin_leq_xmax(self):
        xmin, ymin, xmax, ymax = bbox_from_points(_circle())
        assert xmin <= xmax
        assert ymin <= ymax


# ─── TestResampleCurve ────────────────────────────────────────────────────────

class TestResampleCurve:
    def test_returns_ndarray(self):
        pts = _circle(16)
        out = resample_curve(pts, 32)
        assert isinstance(out, np.ndarray)

    def test_output_n_rows(self):
        out = resample_curve(_circle(16), 32)
        assert len(out) == 32

    def test_start_and_end_preserved(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        out = resample_curve(pts, 5)
        np.testing.assert_allclose(out[0], pts[0], atol=1e-10)
        np.testing.assert_allclose(out[-1], pts[-1], atol=1e-10)

    def test_single_point_degenerate(self):
        pts = np.array([[5.0, 5.0]])
        out = resample_curve(pts, 4)
        # Returns available points (≤ n)
        assert len(out) <= 4

    def test_zero_length_curve(self):
        pts = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        out = resample_curve(pts, 5)
        assert len(out) == 5


# ─── TestAlignCentroids ───────────────────────────────────────────────────────

class TestAlignCentroids:
    def test_returns_ndarray(self):
        out = align_centroids(_square(), _square(cx=3.0))
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        src = _square()
        tgt = _square(cx=5.0)
        out = align_centroids(src, tgt)
        assert out.shape == src.shape

    def test_centroids_match(self):
        src = _square()
        tgt = _square(cx=4.0, cy=-2.0)
        out = align_centroids(src, tgt)
        np.testing.assert_allclose(out.mean(axis=0), tgt.mean(axis=0), atol=1e-10)

    def test_zero_shift(self):
        pts = _square()
        out = align_centroids(pts, pts)
        np.testing.assert_allclose(out, pts, atol=1e-10)


# ─── TestPolyIou ──────────────────────────────────────────────────────────────

class TestPolyIou:
    def test_identical_polygons_one(self):
        sq = _square()
        assert poly_iou(sq, sq) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap_zero(self):
        sq1 = _square(cx=-5.0)
        sq2 = _square(cx=5.0)
        assert poly_iou(sq1, sq2) == pytest.approx(0.0, abs=1e-6)

    def test_range_zero_to_one(self):
        sq1 = _square()
        sq2 = _square(cx=0.5)
        iou = poly_iou(sq1, sq2)
        assert 0.0 <= iou <= 1.0

    def test_fully_contained(self):
        big = _square(r=2.0)
        small = _square(r=1.0)
        iou = poly_iou(big, small)
        # small inside big → IoU = area_small / area_big = 4/16 = 0.25
        assert iou == pytest.approx(0.25, abs=1e-4)

    def test_symmetric(self):
        sq1 = _square()
        sq2 = _square(cx=0.5)
        assert poly_iou(sq1, sq2) == pytest.approx(poly_iou(sq2, sq1), abs=1e-6)


# ─── TestPointInPolygon ───────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_center_inside(self):
        sq = _square()
        assert point_in_polygon(np.array([0.0, 0.0]), sq) is True

    def test_far_point_outside(self):
        sq = _square()
        assert point_in_polygon(np.array([10.0, 10.0]), sq) is False

    def test_returns_bool(self):
        sq = _square()
        result = point_in_polygon(np.array([0.0, 0.0]), sq)
        assert isinstance(result, bool)

    def test_degenerate_polygon_false(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert point_in_polygon(np.array([0.5, 0.0]), pts) is False

    def test_corner_case(self):
        sq = _square()
        # Point clearly inside
        assert point_in_polygon(np.array([0.5, 0.5]), sq) is True


# ─── TestNormalizeContour ─────────────────────────────────────────────────────

class TestNormalizeContour:
    def test_returns_ndarray(self):
        out = normalize_contour(_square())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        pts = _circle(32)
        out = normalize_contour(pts)
        assert out.shape == pts.shape

    def test_centroid_near_origin(self):
        pts = _square(cx=10.0, cy=-5.0)
        out = normalize_contour(pts)
        np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_target_scale_one(self):
        pts = _circle(32, r=5.0)
        out = normalize_contour(pts, target_scale=1.0)
        xmin, ymin, xmax, ymax = bbox_from_points(out)
        diag = math.hypot(xmax - xmin, ymax - ymin)
        assert diag == pytest.approx(1.0, abs=1e-5)

    def test_empty_points(self):
        out = normalize_contour(np.zeros((0, 2)))
        assert len(out) == 0

    def test_degenerate_single_point_no_crash(self):
        pts = np.array([[3.0, 4.0]])
        out = normalize_contour(pts)
        assert out.shape == (1, 2)


# ─── TestSmoothContour ────────────────────────────────────────────────────────

class TestSmoothContour:
    def test_returns_ndarray(self):
        out = smooth_contour(_circle(32))
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        pts = _circle(32)
        out = smooth_contour(pts)
        assert out.shape == pts.shape

    def test_dtype_float64(self):
        out = smooth_contour(_circle(32))
        assert out.dtype == np.float64

    def test_constant_curve_unchanged(self):
        pts = np.tile([5.0, 3.0], (20, 1))
        out = smooth_contour(pts)
        np.testing.assert_allclose(out, pts, atol=1e-10)

    def test_larger_window_smoother(self):
        pts = _circle(32) + np.random.default_rng(0).uniform(-0.5, 0.5, (32, 2))
        out3 = smooth_contour(pts, window=3)
        out9 = smooth_contour(pts, window=9)
        # Larger window → lower variance
        assert out9.std() <= out3.std() + 1e-6


# ─── TestCurvature ────────────────────────────────────────────────────────────

class TestCurvature:
    def test_returns_ndarray(self):
        out = curvature(_circle(32))
        assert isinstance(out, np.ndarray)

    def test_shape_matches_input(self):
        pts = _circle(32)
        out = curvature(pts)
        assert out.shape == (32,)

    def test_nonneg(self):
        out = curvature(_circle(32))
        assert (out >= 0.0).all()

    def test_circle_uniform_curvature(self):
        # Circle of radius r has constant curvature 1/r
        pts = _circle(200, r=5.0)
        out = curvature(pts)
        # Interior points (not boundary) should be ~1/r = 0.2
        interior = out[1:-1]
        assert interior.mean() == pytest.approx(1.0 / 5.0, rel=0.3)

    def test_straight_line_low_curvature(self):
        pts = np.stack([np.linspace(0, 10, 50), np.zeros(50)], axis=1)
        out = curvature(pts)
        # Interior points should be near 0
        assert out[5:-5].mean() < 0.01

    def test_too_few_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = curvature(pts)
        assert len(out) == 2
        assert (out == 0.0).all()

    def test_single_point_zero(self):
        out = curvature(np.array([[1.0, 2.0]]))
        assert out.shape == (1,)
        assert out[0] == 0.0
