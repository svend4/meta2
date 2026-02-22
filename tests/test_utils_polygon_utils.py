"""Tests for puzzle_reconstruction/utils/polygon_utils.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.polygon_utils import (
    polygon_area,
    polygon_perimeter,
    polygon_centroid,
    point_in_polygon,
    convex_hull,
    polygon_bounding_box,
    polygon_aspect_ratio,
    translate_polygon,
    scale_polygon,
    rotate_polygon,
    polygon_similarity,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def unit_square():
    return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def rectangle(w, h):
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def equilateral_triangle(side=1.0):
    h = side * math.sqrt(3) / 2
    return [(0.0, 0.0), (side, 0.0), (side / 2, h)]


# ─── polygon_area ─────────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_unit_square(self):
        assert abs(polygon_area(unit_square()) - 1.0) < 1e-9

    def test_rectangle(self):
        assert abs(polygon_area(rectangle(3, 4)) - 12.0) < 1e-9

    def test_triangle(self):
        tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
        assert abs(polygon_area(tri) - 6.0) < 1e-9

    def test_less_than_3_vertices_raises(self):
        with pytest.raises(ValueError):
            polygon_area([(0.0, 0.0), (1.0, 0.0)])

    def test_non_negative(self):
        # Clockwise orientation
        poly = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        assert polygon_area(poly) >= 0.0

    def test_equilateral_triangle(self):
        tri = equilateral_triangle(2.0)
        expected = math.sqrt(3)  # area of equilateral triangle with side 2
        assert abs(polygon_area(tri) - expected) < 1e-6

    def test_larger_polygon(self):
        n = 100
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        R = 5.0
        poly = [(R * math.cos(a), R * math.sin(a)) for a in angles]
        expected = math.pi * R ** 2
        # Approximate area should be close to circle area
        assert abs(polygon_area(poly) - expected) < 0.5


# ─── polygon_perimeter ───────────────────────────────────────────────────────

class TestPolygonPerimeter:
    def test_unit_square(self):
        assert abs(polygon_perimeter(unit_square()) - 4.0) < 1e-9

    def test_rectangle(self):
        assert abs(polygon_perimeter(rectangle(3, 4)) - 14.0) < 1e-9

    def test_less_than_2_vertices_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([(0.0, 0.0)])

    def test_two_points_perimeter(self):
        # Two points: length is 2 * distance
        p = [(0.0, 0.0), (3.0, 4.0)]
        # distance = 5, perimeter = 5 + 5 = 10
        assert abs(polygon_perimeter(p) - 10.0) < 1e-9

    def test_non_negative(self):
        assert polygon_perimeter(unit_square()) >= 0.0

    def test_equilateral_triangle(self):
        side = 3.0
        tri = equilateral_triangle(side)
        assert abs(polygon_perimeter(tri) - 3 * side) < 1e-6


# ─── polygon_centroid ─────────────────────────────────────────────────────────

class TestPolygonCentroid:
    def test_unit_square(self):
        cx, cy = polygon_centroid(unit_square())
        assert abs(cx - 0.5) < 1e-9
        assert abs(cy - 0.5) < 1e-9

    def test_rectangle(self):
        cx, cy = polygon_centroid(rectangle(6, 4))
        assert abs(cx - 3.0) < 1e-9
        assert abs(cy - 2.0) < 1e-9

    def test_equilateral_triangle_centroid(self):
        side = 2.0
        tri = equilateral_triangle(side)
        cx, cy = polygon_centroid(tri)
        expected_cx = side / 2
        expected_cy = math.sqrt(3) * side / 6
        assert abs(cx - expected_cx) < 1e-6
        assert abs(cy - expected_cy) < 1e-6

    def test_less_than_3_vertices_raises(self):
        with pytest.raises(ValueError):
            polygon_centroid([(0.0, 0.0), (1.0, 0.0)])

    def test_returns_tuple_of_two(self):
        cx, cy = polygon_centroid(unit_square())
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_degenerate_polygon(self):
        """Collinear points → falls back to simple mean."""
        poly = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        cx, cy = polygon_centroid(poly)
        assert abs(cx - 1.0) < 1e-9
        assert abs(cy - 0.0) < 1e-9


# ─── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_inside(self):
        sq = unit_square()
        assert point_in_polygon((0.5, 0.5), sq) is True

    def test_outside(self):
        sq = unit_square()
        assert point_in_polygon((2.0, 2.0), sq) is False

    def test_origin_inside_square(self):
        sq = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        assert point_in_polygon((0.0, 0.0), sq) is True

    def test_outside_square(self):
        sq = unit_square()
        assert point_in_polygon((1.5, 0.5), sq) is False

    def test_less_than_3_vertices_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)])

    def test_triangle_inside(self):
        tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        assert point_in_polygon((5.0, 3.0), tri) is True

    def test_triangle_outside(self):
        tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        assert point_in_polygon((0.0, 10.0), tri) is False

    def test_far_outside(self):
        sq = unit_square()
        assert point_in_polygon((100.0, 100.0), sq) is False


# ─── convex_hull ──────────────────────────────────────────────────────────────

class TestConvexHull:
    def test_square_points_hull(self):
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
                  (0.5, 0.5)]  # interior point
        hull = convex_hull(points)
        assert len(hull) <= 4
        assert len(hull) >= 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            convex_hull([])

    def test_single_point(self):
        hull = convex_hull([(1.0, 1.0)])
        assert len(hull) == 1

    def test_two_points(self):
        hull = convex_hull([(0.0, 0.0), (1.0, 1.0)])
        assert len(hull) == 2

    def test_collinear_points(self):
        points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        hull = convex_hull(points)
        # Collinear → should keep endpoints at minimum
        xs = [p[0] for p in hull]
        assert min(xs) == 0.0
        assert max(xs) == 3.0

    def test_all_same_point(self):
        points = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
        hull = convex_hull(points)
        assert len(hull) == 1

    def test_returns_list_of_tuples(self):
        hull = convex_hull([(0, 0), (1, 0), (0.5, 1)])
        for p in hull:
            assert len(p) == 2


# ─── polygon_bounding_box ─────────────────────────────────────────────────────

class TestPolygonBoundingBox:
    def test_unit_square(self):
        sq = unit_square()
        x_min, y_min, x_max, y_max = polygon_bounding_box(sq)
        assert x_min == 0.0 and y_min == 0.0
        assert x_max == 1.0 and y_max == 1.0

    def test_shifted_polygon(self):
        poly = [(5.0, 3.0), (8.0, 3.0), (8.0, 7.0), (5.0, 7.0)]
        x_min, y_min, x_max, y_max = polygon_bounding_box(poly)
        assert x_min == 5.0 and x_max == 8.0
        assert y_min == 3.0 and y_max == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_bounding_box([])

    def test_single_point(self):
        x_min, y_min, x_max, y_max = polygon_bounding_box([(3.0, 4.0)])
        assert x_min == x_max == 3.0
        assert y_min == y_max == 4.0

    def test_returns_four_floats(self):
        result = polygon_bounding_box(unit_square())
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)


# ─── polygon_aspect_ratio ─────────────────────────────────────────────────────

class TestPolygonAspectRatio:
    def test_unit_square_ar(self):
        assert abs(polygon_aspect_ratio(unit_square()) - 1.0) < 1e-9

    def test_wide_rectangle(self):
        poly = rectangle(4, 2)
        assert abs(polygon_aspect_ratio(poly) - 2.0) < 1e-9

    def test_tall_rectangle(self):
        poly = rectangle(1, 4)
        assert abs(polygon_aspect_ratio(poly) - 0.25) < 1e-9

    def test_flat_polygon_zero(self):
        """Degenerate polygon with zero height → 0."""
        poly = [(0.0, 0.0), (5.0, 0.0), (3.0, 0.0)]
        ar = polygon_aspect_ratio(poly)
        assert ar == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_aspect_ratio([])


# ─── translate_polygon ────────────────────────────────────────────────────────

class TestTranslatePolygon:
    def test_basic_translate(self):
        sq = unit_square()
        moved = translate_polygon(sq, dx=5.0, dy=3.0)
        assert abs(moved[0][0] - 5.0) < 1e-9
        assert abs(moved[0][1] - 3.0) < 1e-9

    def test_zero_translate_unchanged(self):
        sq = unit_square()
        moved = translate_polygon(sq, dx=0.0, dy=0.0)
        for (x1, y1), (x2, y2) in zip(sq, moved):
            assert abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9

    def test_length_preserved(self):
        sq = unit_square()
        moved = translate_polygon(sq, dx=10.0, dy=20.0)
        assert len(moved) == len(sq)

    def test_area_preserved(self):
        sq = unit_square()
        moved = translate_polygon(sq, dx=100.0, dy=-50.0)
        assert abs(polygon_area(moved) - polygon_area(sq)) < 1e-9

    def test_negative_translate(self):
        sq = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
        moved = translate_polygon(sq, dx=-10.0, dy=-10.0)
        assert abs(moved[0][0] - 0.0) < 1e-9
        assert abs(moved[0][1] - 0.0) < 1e-9


# ─── scale_polygon ────────────────────────────────────────────────────────────

class TestScalePolygon:
    def test_scale_doubles_dimensions(self):
        sq = unit_square()
        cx, cy = polygon_centroid(sq)
        scaled = scale_polygon(sq, scale=2.0, center=(cx, cy))
        x_min, y_min, x_max, y_max = polygon_bounding_box(scaled)
        assert abs(x_max - x_min - 2.0) < 1e-9

    def test_scale_one_unchanged(self):
        sq = unit_square()
        scaled = scale_polygon(sq, scale=1.0)
        for (x1, y1), (x2, y2) in zip(sq, scaled):
            assert abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9

    def test_scale_zero_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(unit_square(), scale=0.0)

    def test_scale_negative_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(unit_square(), scale=-1.0)

    def test_area_scales_by_square(self):
        sq = unit_square()
        scaled = scale_polygon(sq, scale=3.0)
        assert abs(polygon_area(scaled) - 9.0) < 1e-9

    def test_none_center_uses_centroid(self):
        sq = unit_square()
        scaled_auto = scale_polygon(sq, scale=2.0, center=None)
        cx, cy = polygon_centroid(sq)
        scaled_explicit = scale_polygon(sq, scale=2.0, center=(cx, cy))
        for (x1, y1), (x2, y2) in zip(scaled_auto, scaled_explicit):
            assert abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9


# ─── rotate_polygon ───────────────────────────────────────────────────────────

class TestRotatePolygon:
    def test_zero_rotation_unchanged(self):
        sq = unit_square()
        rotated = rotate_polygon(sq, angle_deg=0.0)
        for (x1, y1), (x2, y2) in zip(sq, rotated):
            assert abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9

    def test_360_rotation_unchanged(self):
        sq = unit_square()
        rotated = rotate_polygon(sq, angle_deg=360.0)
        for (x1, y1), (x2, y2) in zip(sq, rotated):
            assert abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6

    def test_area_preserved(self):
        sq = unit_square()
        rotated = rotate_polygon(sq, angle_deg=45.0)
        assert abs(polygon_area(rotated) - polygon_area(sq)) < 1e-9

    def test_perimeter_preserved(self):
        sq = unit_square()
        rotated = rotate_polygon(sq, angle_deg=30.0)
        assert abs(polygon_perimeter(rotated) - polygon_perimeter(sq)) < 1e-9

    def test_90_degree_rotation(self):
        """Rotating (1,0) by 90° around origin → (0,1)."""
        poly = [(1.0, 0.0), (2.0, 0.0), (1.5, 1.0)]
        rotated = rotate_polygon(poly, angle_deg=90.0, center=(0.0, 0.0))
        # (1,0) → (0,1), (2,0) → (0,2)
        assert abs(rotated[0][0] - 0.0) < 1e-9
        assert abs(rotated[0][1] - 1.0) < 1e-9

    def test_length_preserved(self):
        sq = unit_square()
        rotated = rotate_polygon(sq, angle_deg=45.0)
        assert len(rotated) == len(sq)

    def test_none_center_uses_centroid(self):
        """Both None center and explicit centroid should give same result."""
        sq = unit_square()
        cx, cy = polygon_centroid(sq)
        r1 = rotate_polygon(sq, angle_deg=45.0, center=None)
        r2 = rotate_polygon(sq, angle_deg=45.0, center=(cx, cy))
        for (x1, y1), (x2, y2) in zip(r1, r2):
            assert abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9


# ─── polygon_similarity ──────────────────────────────────────────────────────

class TestPolygonSimilarity:
    def test_identical_polygons(self):
        sq = unit_square()
        assert abs(polygon_similarity(sq, sq) - 1.0) < 1e-9

    def test_different_polygons(self):
        sq = unit_square()
        large_sq = rectangle(10, 10)
        sim = polygon_similarity(sq, large_sq)
        assert 0.0 < sim < 1.0

    def test_result_in_range(self):
        sq = unit_square()
        rect = rectangle(3, 4)
        sim = polygon_similarity(sq, rect)
        assert 0.0 <= sim <= 1.0

    def test_symmetric(self):
        sq = unit_square()
        rect = rectangle(2, 3)
        assert abs(polygon_similarity(sq, rect) - polygon_similarity(rect, sq)) < 1e-9

    def test_similar_shapes_high_score(self):
        sq1 = unit_square()
        sq2 = rectangle(1.0, 1.0)  # same shape
        assert polygon_similarity(sq1, sq2) > 0.99

    def test_very_different_shapes_low_score(self):
        sq = unit_square()
        huge_rect = rectangle(1000, 1000)
        sim = polygon_similarity(sq, huge_rect)
        assert sim < 0.1
