"""
Тесты для puzzle_reconstruction/utils/geometry.py

Покрытие:
    rotation_matrix_2d  — det=1, ортогональность, angle=0 → I, angle=π/2
    rotate_points       — форма, center=None, center задан
    polygon_area        — единичный квадрат, CCW/CW знак, < 3 точек
    polygon_centroid    — прямоугольник, треугольник, < 3 точек
    bbox_from_points    — пустой, известный bbox
    resample_curve      — длина результата, равноудалённость, прямая линия
    align_centroids     — совмещение центроидов
    poly_iou            — тождественный полигон → 1.0, непересекающиеся → 0.0,
                          частичное перекрытие ∈ (0, 1)
    point_in_polygon    — внутри, снаружи, < 3 точек
    normalize_contour   — нулевой центроид, диагональ ≈ target_scale
    smooth_contour      — форма, окно сглаживает выброс
    curvature           — форма, прямая → ≈0, окружность → постоянная
"""
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


# ─── Вспомогательные полигоны ─────────────────────────────────────────────────

def unit_square_ccw():
    return np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)

def unit_square_cw():
    return np.array([[0,0],[0,1],[1,1],[1,0]], dtype=float)

def circle_pts(n=60, r=1.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([r*np.cos(t), r*np.sin(t)], axis=-1)


# ─── rotation_matrix_2d ───────────────────────────────────────────────────────

class TestRotationMatrix2d:
    def test_identity_at_zero(self):
        R = rotation_matrix_2d(0.0)
        assert np.allclose(R, np.eye(2))

    def test_shape(self):
        R = rotation_matrix_2d(1.0)
        assert R.shape == (2, 2)

    def test_det_plus_one(self):
        for angle in [0.0, 0.5, math.pi, 2*math.pi, -1.0]:
            R = rotation_matrix_2d(angle)
            assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-10)

    def test_orthogonal(self):
        R = rotation_matrix_2d(math.pi / 3)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)

    def test_pi_over_2(self):
        R  = rotation_matrix_2d(math.pi / 2)
        pt = np.array([1.0, 0.0])
        assert np.allclose(R @ pt, [0.0, 1.0], atol=1e-10)

    def test_pi(self):
        R  = rotation_matrix_2d(math.pi)
        pt = np.array([1.0, 0.0])
        assert np.allclose(R @ pt, [-1.0, 0.0], atol=1e-10)


# ─── rotate_points ────────────────────────────────────────────────────────────

class TestRotatePoints:
    def test_output_shape(self):
        pts = np.random.rand(20, 2)
        out = rotate_points(pts, 1.0)
        assert out.shape == (20, 2)

    def test_zero_angle_unchanged(self):
        pts = np.random.rand(10, 2)
        out = rotate_points(pts, 0.0)
        assert np.allclose(out, pts)

    def test_center_none_rotates_around_origin(self):
        pts = np.array([[1.0, 0.0]])
        out = rotate_points(pts, math.pi / 2)
        assert np.allclose(out, [[0.0, 1.0]], atol=1e-10)

    def test_center_specified(self):
        pts    = np.array([[2.0, 1.0]])
        center = np.array([1.0, 1.0])
        out    = rotate_points(pts, math.pi / 2, center=center)
        # Поворот [2,1] вокруг [1,1] на 90° → [1,2]
        assert np.allclose(out, [[1.0, 2.0]], atol=1e-10)

    def test_full_rotation_identity(self):
        pts = circle_pts(20)
        out = rotate_points(pts, 2 * math.pi)
        assert np.allclose(out, pts, atol=1e-10)


# ─── polygon_area ─────────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_unit_square(self):
        area = polygon_area(unit_square_ccw())
        assert math.isclose(abs(area), 1.0, abs_tol=1e-10)

    def test_ccw_positive(self):
        area = polygon_area(unit_square_ccw())
        assert area > 0

    def test_cw_negative(self):
        area = polygon_area(unit_square_cw())
        assert area < 0

    def test_triangle(self):
        tri  = np.array([[0,0],[3,0],[0,4]], dtype=float)
        area = polygon_area(tri)
        assert math.isclose(abs(area), 6.0, abs_tol=1e-10)

    def test_degenerate_line(self):
        pts = np.array([[0,0],[1,0],[2,0]], dtype=float)
        assert math.isclose(polygon_area(pts), 0.0, abs_tol=1e-10)

    def test_less_than_3_points(self):
        assert polygon_area(np.array([[0,0],[1,1]], dtype=float)) == 0.0

    def test_circle_approx(self):
        # N=1000 точек → площадь ≈ π
        pts  = circle_pts(n=1000)
        area = abs(polygon_area(pts))
        assert math.isclose(area, math.pi, rel_tol=1e-3)


# ─── polygon_centroid ─────────────────────────────────────────────────────────

class TestPolygonCentroid:
    def test_unit_square(self):
        c = polygon_centroid(unit_square_ccw())
        assert np.allclose(c, [0.5, 0.5], atol=1e-10)

    def test_rectangle(self):
        pts = np.array([[0,0],[4,0],[4,2],[0,2]], dtype=float)
        c   = polygon_centroid(pts)
        assert np.allclose(c, [2.0, 1.0], atol=1e-8)

    def test_triangle(self):
        tri = np.array([[0,0],[6,0],[0,6]], dtype=float)
        c   = polygon_centroid(tri)
        assert np.allclose(c, [2.0, 2.0], atol=1e-8)

    def test_empty(self):
        c = polygon_centroid(np.zeros((0, 2)))
        assert c.shape == (2,)

    def test_two_points_fallback(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0]])
        c   = polygon_centroid(pts)
        assert np.allclose(c, [1.0, 0.0], atol=1e-10)


# ─── bbox_from_points ─────────────────────────────────────────────────────────

class TestBboxFromPoints:
    def test_known_bbox(self):
        pts = np.array([[1,2],[3,4],[5,0]], dtype=float)
        x0, y0, x1, y1 = bbox_from_points(pts)
        assert x0 == 1.0 and y0 == 0.0 and x1 == 5.0 and y1 == 4.0

    def test_single_point(self):
        x0, y0, x1, y1 = bbox_from_points(np.array([[3.0, 7.0]]))
        assert x0 == x1 == 3.0 and y0 == y1 == 7.0

    def test_empty(self):
        bbox = bbox_from_points(np.zeros((0, 2)))
        assert len(bbox) == 4


# ─── resample_curve ───────────────────────────────────────────────────────────

class TestResampleCurve:
    def test_output_length(self):
        pts = circle_pts(20)
        out = resample_curve(pts, 50)
        assert len(out) == 50

    def test_preserves_start(self):
        pts = circle_pts(20)
        out = resample_curve(pts, 20)
        assert np.allclose(out[0], pts[0], atol=1e-10)

    def test_straight_line(self):
        """Равномерная передискретизация прямой → одинаковые шаги."""
        pts = np.array([[i, 0.0] for i in range(11)], dtype=float)
        out = resample_curve(pts, 11)
        dists = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert np.allclose(dists, dists[0], rtol=1e-5)

    def test_n_2(self):
        """n=2 → только начало и конец."""
        pts = circle_pts(20)
        out = resample_curve(pts, 2)
        assert len(out) == 2

    def test_single_unique_point(self):
        """Все точки совпадают → результат тот же."""
        pts = np.tile([3.0, 4.0], (10, 1))
        out = resample_curve(pts, 5)
        assert np.allclose(out, pts[0], atol=1e-8)


# ─── align_centroids ──────────────────────────────────────────────────────────

class TestAlignCentroids:
    def test_centroid_matches(self):
        src = np.array([[0,0],[1,0],[0,1]], dtype=float)
        tgt = np.array([[5,5],[6,5],[5,6]], dtype=float)
        out = align_centroids(src, tgt)
        assert np.allclose(out.mean(axis=0), tgt.mean(axis=0), atol=1e-10)

    def test_shape_preserved(self):
        src = circle_pts(30)
        tgt = circle_pts(20)
        out = align_centroids(src, tgt)
        assert out.shape == src.shape

    def test_identical_centroids_no_change(self):
        pts = circle_pts(20)
        out = align_centroids(pts, pts)
        assert np.allclose(out, pts, atol=1e-10)


# ─── poly_iou ─────────────────────────────────────────────────────────────────

class TestPolyIou:
    def test_identical(self):
        sq  = unit_square_ccw()
        iou = poly_iou(sq, sq)
        assert math.isclose(iou, 1.0, abs_tol=1e-6)

    def test_no_overlap(self):
        sq1 = unit_square_ccw()
        sq2 = sq1 + np.array([5.0, 0.0])
        iou = poly_iou(sq1, sq2)
        assert math.isclose(iou, 0.0, abs_tol=1e-6)

    def test_half_overlap(self):
        sq1 = unit_square_ccw()
        sq2 = sq1 + np.array([0.5, 0.0])
        iou = poly_iou(sq1, sq2)
        # Пересечение 0.5, объединение 1.5 → IoU = 1/3
        assert 0.0 < iou < 1.0

    def test_containment(self):
        outer = np.array([[0,0],[4,0],[4,4],[0,4]], dtype=float)
        inner = np.array([[1,1],[2,1],[2,2],[1,2]], dtype=float)
        iou   = poly_iou(inner, outer)
        # inner ⊂ outer → IoU = area_inner / area_outer = 1/16
        assert math.isclose(iou, 1/16, rel_tol=1e-3)

    def test_symmetry(self):
        a = unit_square_ccw()
        b = a + np.array([0.3, 0.3])
        assert math.isclose(poly_iou(a, b), poly_iou(b, a), abs_tol=1e-8)


# ─── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_center_inside(self):
        assert point_in_polygon([0.5, 0.5], unit_square_ccw())

    def test_outside(self):
        assert not point_in_polygon([2.0, 2.0], unit_square_ccw())

    def test_negative_coords(self):
        sq = unit_square_ccw() - 0.5
        assert point_in_polygon([0.0, 0.0], sq)

    def test_less_than_3_points(self):
        assert not point_in_polygon([0.5, 0.5],
                                     np.array([[0,0],[1,1]], dtype=float))

    def test_circle_inside(self):
        circ = circle_pts(100, r=2.0)
        assert point_in_polygon([0.0, 0.0], circ)

    def test_outside_circle(self):
        circ = circle_pts(100, r=1.0)
        assert not point_in_polygon([2.0, 2.0], circ)


# ─── normalize_contour ────────────────────────────────────────────────────────

class TestNormalizeContour:
    def test_centroid_at_origin(self):
        pts = unit_square_ccw() * 5 + np.array([10.0, 20.0])
        out = normalize_contour(pts)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-10)

    def test_scale_is_one(self):
        pts = np.random.RandomState(0).rand(30, 2) * 100 + 50
        out = normalize_contour(pts, target_scale=1.0)
        x0, y0, x1, y1 = bbox_from_points(out)
        diag = math.hypot(x1 - x0, y1 - y0)
        assert math.isclose(diag, 1.0, rel_tol=1e-5)

    def test_custom_scale(self):
        pts = unit_square_ccw()
        out = normalize_contour(pts, target_scale=3.0)
        x0, y0, x1, y1 = bbox_from_points(out)
        diag = math.hypot(x1 - x0, y1 - y0)
        assert math.isclose(diag, 3.0, rel_tol=1e-5)

    def test_empty_input(self):
        out = normalize_contour(np.zeros((0, 2)))
        assert len(out) == 0

    def test_constant_input(self):
        """Все точки одинаковы — дисперсия 0, не должно быть ошибки."""
        pts = np.tile([3.0, 4.0], (10, 1))
        out = normalize_contour(pts)
        assert out.shape == pts.shape


# ─── smooth_contour ───────────────────────────────────────────────────────────

class TestSmoothContour:
    def test_output_shape(self):
        pts = circle_pts(50)
        out = smooth_contour(pts, window=5)
        assert out.shape == pts.shape

    def test_reduces_spike(self):
        """Одиночный выброс должен быть сглажен."""
        pts = circle_pts(40)
        pts_noisy = pts.copy()
        pts_noisy[10] += np.array([10.0, 10.0])  # Одиночный выброс
        out = smooth_contour(pts_noisy, window=7)
        spike_before = np.linalg.norm(pts_noisy[10] - pts[10])
        spike_after  = np.linalg.norm(out[10] - pts[10])
        assert spike_after < spike_before

    def test_even_window_rounded(self):
        """Чётное окно → нечётное, нет ошибки."""
        pts = circle_pts(20)
        out = smooth_contour(pts, window=4)
        assert out.shape == pts.shape

    def test_constant_curve_unchanged(self):
        """Однородная кривая не изменяется при сглаживании."""
        pts = np.tile([5.0, 3.0], (20, 1))
        out = smooth_contour(pts, window=5)
        assert np.allclose(out, pts, atol=1e-10)


# ─── curvature ────────────────────────────────────────────────────────────────

class TestCurvature:
    def test_output_shape(self):
        pts = circle_pts(50)
        out = curvature(pts)
        assert out.shape == (50,)

    def test_straight_line_low_curvature(self):
        """Прямая → кривизна ≈ 0."""
        pts = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
        out = curvature(pts)
        # Внутренние точки должны иметь малую кривизну
        assert float(out[5:-5].mean()) < 1e-3

    def test_circle_constant(self):
        """Окружность → кривизна примерно постоянная."""
        pts = circle_pts(200, r=5.0)
        out = curvature(pts)
        # Для окружности κ = 1/r = 0.2
        inner = out[10:-10]
        cv    = inner.std() / (inner.mean() + 1e-12)
        assert cv < 0.5, f"Variation coefficient = {cv:.3f}"

    def test_nonneg(self):
        pts = circle_pts(40)
        out = curvature(pts)
        assert np.all(out >= 0)

    def test_short_curve(self):
        """Кривая из 2 точек → нет ошибки."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = curvature(pts)
        assert out.shape == (2,)
