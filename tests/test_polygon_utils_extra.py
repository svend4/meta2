"""Extra tests for puzzle_reconstruction/utils/polygon_utils.py."""
from __future__ import annotations

import math

import pytest

from puzzle_reconstruction.utils.polygon_utils import (
    convex_hull,
    point_in_polygon,
    polygon_area,
    polygon_aspect_ratio,
    polygon_bounding_box,
    polygon_centroid,
    polygon_perimeter,
    polygon_similarity,
    rotate_polygon,
    scale_polygon,
    translate_polygon,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(x: float = 0.0, y: float = 0.0, s: float = 1.0):
    """Unit square (CCW)."""
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def _triangle():
    """Simple right triangle."""
    return [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]


def _rect(w: float = 4.0, h: float = 2.0):
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


# ─── polygon_area (extra) ─────────────────────────────────────────────────────

class TestPolygonAreaExtra:
    def test_returns_float(self):
        assert isinstance(polygon_area(_square()), float)

    def test_unit_square_area(self):
        assert polygon_area(_square()) == pytest.approx(1.0)

    def test_triangle_area(self):
        # Area = 0.5 * base * height = 0.5 * 4 * 3 = 6
        assert polygon_area(_triangle()) == pytest.approx(6.0)

    def test_rectangle_area(self):
        assert polygon_area(_rect(4, 3)) == pytest.approx(12.0)

    def test_large_square(self):
        assert polygon_area(_square(s=10.0)) == pytest.approx(100.0)

    def test_area_non_negative(self):
        assert polygon_area(_square()) >= 0.0

    def test_two_points_raises(self):
        with pytest.raises(ValueError):
            polygon_area([(0, 0), (1, 1)])

    def test_one_point_raises(self):
        with pytest.raises(ValueError):
            polygon_area([(0, 0)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_area([])

    def test_translated_same_area(self):
        sq = _square()
        sq2 = translate_polygon(sq, 100, 100)
        assert polygon_area(sq2) == pytest.approx(polygon_area(sq))


# ─── polygon_perimeter (extra) ────────────────────────────────────────────────

class TestPolygonPerimeterExtra:
    def test_returns_float(self):
        assert isinstance(polygon_perimeter(_square()), float)

    def test_unit_square_perimeter(self):
        assert polygon_perimeter(_square()) == pytest.approx(4.0)

    def test_rect_perimeter(self):
        assert polygon_perimeter(_rect(4, 3)) == pytest.approx(14.0)

    def test_triangle_perimeter(self):
        # 4 + 3 + 5 = 12
        assert polygon_perimeter(_triangle()) == pytest.approx(12.0)

    def test_perimeter_non_negative(self):
        assert polygon_perimeter(_square()) >= 0.0

    def test_one_point_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([(0, 0)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([])

    def test_two_points_perimeter(self):
        # Degenerate: 2 points gives distance * 2
        p = [(0.0, 0.0), (3.0, 4.0)]
        assert polygon_perimeter(p) == pytest.approx(10.0)

    def test_larger_square_perimeter(self):
        assert polygon_perimeter(_square(s=5.0)) == pytest.approx(20.0)


# ─── polygon_centroid (extra) ─────────────────────────────────────────────────

class TestPolygonCentroidExtra:
    def test_returns_tuple(self):
        result = polygon_centroid(_square())
        assert isinstance(result, tuple) and len(result) == 2

    def test_unit_square_centroid(self):
        cx, cy = polygon_centroid(_square())
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_symmetric_triangle_centroid(self):
        # Centroid of triangle (0,0),(6,0),(3,6): cx=3, cy=2
        cx, cy = polygon_centroid([(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)])
        assert cx == pytest.approx(3.0)
        assert cy == pytest.approx(2.0)

    def test_rectangle_centroid(self):
        cx, cy = polygon_centroid(_rect(4.0, 2.0))
        assert cx == pytest.approx(2.0)
        assert cy == pytest.approx(1.0)

    def test_two_points_raises(self):
        with pytest.raises(ValueError):
            polygon_centroid([(0, 0), (1, 1)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_centroid([])

    def test_translated_centroid_shifted(self):
        sq = _square()
        cx1, cy1 = polygon_centroid(sq)
        sq2 = translate_polygon(sq, 10.0, 5.0)
        cx2, cy2 = polygon_centroid(sq2)
        assert cx2 == pytest.approx(cx1 + 10.0)
        assert cy2 == pytest.approx(cy1 + 5.0)


# ─── point_in_polygon (extra) ─────────────────────────────────────────────────

class TestPointInPolygonExtra:
    def test_inside_square(self):
        assert point_in_polygon((0.5, 0.5), _square()) is True

    def test_outside_square(self):
        assert point_in_polygon((2.0, 2.0), _square()) is False

    def test_outside_left(self):
        assert point_in_polygon((-1.0, 0.5), _square()) is False

    def test_outside_below(self):
        assert point_in_polygon((0.5, -1.0), _square()) is False

    def test_inside_rect(self):
        assert point_in_polygon((2.0, 1.0), _rect(4, 2)) is True

    def test_outside_rect(self):
        assert point_in_polygon((5.0, 1.0), _rect(4, 2)) is False

    def test_inside_triangle(self):
        # Triangle: (0,0),(4,0),(0,3) — point (1,1) inside
        assert point_in_polygon((1.0, 1.0), _triangle()) is True

    def test_outside_triangle(self):
        assert point_in_polygon((3.0, 3.0), _triangle()) is False

    def test_two_points_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0, 0), [(0, 0), (1, 1)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0, 0), [])


# ─── convex_hull (extra) ──────────────────────────────────────────────────────

class TestConvexHullExtra:
    def test_returns_list(self):
        result = convex_hull([(0, 0), (1, 0), (0, 1)])
        assert isinstance(result, list)

    def test_single_point(self):
        result = convex_hull([(3, 4)])
        assert len(result) == 1

    def test_two_distinct_points(self):
        result = convex_hull([(0, 0), (1, 1)])
        assert len(result) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            convex_hull([])

    def test_collinear_points_hull(self):
        pts = [(0, 0), (1, 0), (2, 0), (3, 0)]
        result = convex_hull(pts)
        # Collinear — expect at least 2 points
        assert len(result) >= 2

    def test_square_hull_has_4_points(self):
        pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = convex_hull(pts)
        assert len(result) == 4

    def test_inner_point_excluded(self):
        # Square + interior point
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]
        result = convex_hull(pts)
        # Interior (2,2) should not be in hull
        assert (2.0, 2.0) not in result


# ─── polygon_bounding_box (extra) ─────────────────────────────────────────────

class TestPolygonBoundingBoxExtra:
    def test_returns_four_values(self):
        bb = polygon_bounding_box(_square())
        assert len(bb) == 4

    def test_unit_square_bbox(self):
        x0, y0, x1, y1 = polygon_bounding_box(_square())
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(1.0)
        assert y1 == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_bounding_box([])

    def test_single_point(self):
        x0, y0, x1, y1 = polygon_bounding_box([(3.0, 5.0)])
        assert x0 == pytest.approx(3.0)
        assert x1 == pytest.approx(3.0)

    def test_translated_bbox(self):
        sq = translate_polygon(_square(), 10.0, 20.0)
        x0, y0, x1, y1 = polygon_bounding_box(sq)
        assert x0 == pytest.approx(10.0)
        assert y0 == pytest.approx(20.0)

    def test_xmax_gt_xmin(self):
        x0, y0, x1, y1 = polygon_bounding_box(_square(s=5.0))
        assert x1 > x0

    def test_ymax_gt_ymin(self):
        x0, y0, x1, y1 = polygon_bounding_box(_square(s=5.0))
        assert y1 > y0

    def test_triangle_bbox(self):
        x0, y0, x1, y1 = polygon_bounding_box(_triangle())
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(4.0)
        assert y1 == pytest.approx(3.0)


# ─── polygon_aspect_ratio (extra) ─────────────────────────────────────────────

class TestPolygonAspectRatioExtra:
    def test_returns_float(self):
        assert isinstance(polygon_aspect_ratio(_square()), float)

    def test_unit_square_ratio_is_one(self):
        assert polygon_aspect_ratio(_square()) == pytest.approx(1.0)

    def test_wide_rect_ratio_gt_1(self):
        assert polygon_aspect_ratio(_rect(4.0, 1.0)) == pytest.approx(4.0)

    def test_tall_rect_ratio_lt_1(self):
        assert polygon_aspect_ratio(_rect(1.0, 4.0)) == pytest.approx(0.25)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_aspect_ratio([])

    def test_flat_polygon_zero_height(self):
        pts = [(0, 0), (5, 0), (3, 0)]  # y all same
        ratio = polygon_aspect_ratio(pts)
        assert ratio == pytest.approx(0.0)

    def test_non_negative(self):
        assert polygon_aspect_ratio(_square()) >= 0.0


# ─── translate_polygon (extra) ────────────────────────────────────────────────

class TestTranslatePolygonExtra:
    def test_returns_list(self):
        assert isinstance(translate_polygon(_square(), 0, 0), list)

    def test_zero_translation_identity(self):
        sq = _square()
        result = translate_polygon(sq, 0.0, 0.0)
        for (x1, y1), (x2, y2) in zip(sq, result):
            assert x1 == pytest.approx(x2)
            assert y1 == pytest.approx(y2)

    def test_translation_applied(self):
        sq = _square()
        result = translate_polygon(sq, 5.0, 3.0)
        for (x1, y1), (x2, y2) in zip(sq, result):
            assert x2 == pytest.approx(x1 + 5.0)
            assert y2 == pytest.approx(y1 + 3.0)

    def test_negative_translation(self):
        sq = _square(x=10.0, y=10.0)
        result = translate_polygon(sq, -10.0, -10.0)
        x0, y0, _, _ = polygon_bounding_box(result)
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)

    def test_preserves_length(self):
        sq = _square()
        result = translate_polygon(sq, 1, 1)
        assert len(result) == len(sq)

    def test_area_preserved(self):
        sq = _square()
        result = translate_polygon(sq, 7.0, 3.0)
        assert polygon_area(result) == pytest.approx(polygon_area(sq))


# ─── scale_polygon (extra) ────────────────────────────────────────────────────

class TestScalePolygonExtra:
    def test_returns_list(self):
        assert isinstance(scale_polygon(_square(), 1.0), list)

    def test_scale_1_identity(self):
        sq = _square()
        result = scale_polygon(sq, 1.0)
        for (x1, y1), (x2, y2) in zip(sq, result):
            assert x1 == pytest.approx(x2)
            assert y1 == pytest.approx(y2)

    def test_scale_2_doubles_area(self):
        sq = _square()
        result = scale_polygon(sq, 2.0)
        assert polygon_area(result) == pytest.approx(polygon_area(sq) * 4.0)

    def test_scale_0_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(_square(), 0.0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(_square(), -1.0)

    def test_custom_center(self):
        sq = _square()
        result = scale_polygon(sq, 2.0, center=(0.0, 0.0))
        assert len(result) == len(sq)

    def test_preserves_length(self):
        sq = _square()
        assert len(scale_polygon(sq, 3.0)) == len(sq)


# ─── rotate_polygon (extra) ───────────────────────────────────────────────────

class TestRotatePolygonExtra:
    def test_returns_list(self):
        assert isinstance(rotate_polygon(_square(), 0.0), list)

    def test_zero_rotation_identity(self):
        sq = _square()
        result = rotate_polygon(sq, 0.0)
        for (x1, y1), (x2, y2) in zip(sq, result):
            assert x1 == pytest.approx(x2, abs=1e-9)
            assert y1 == pytest.approx(y2, abs=1e-9)

    def test_360_rotation_identity(self):
        sq = _square()
        result = rotate_polygon(sq, 360.0)
        for (x1, y1), (x2, y2) in zip(sq, result):
            assert x1 == pytest.approx(x2, abs=1e-9)
            assert y1 == pytest.approx(y2, abs=1e-9)

    def test_area_preserved_after_rotation(self):
        sq = _square()
        result = rotate_polygon(sq, 45.0)
        assert polygon_area(result) == pytest.approx(polygon_area(sq), rel=1e-6)

    def test_perimeter_preserved_after_rotation(self):
        sq = _square()
        result = rotate_polygon(sq, 90.0)
        assert polygon_perimeter(result) == pytest.approx(polygon_perimeter(sq), rel=1e-6)

    def test_90_rotation_correct(self):
        # Square (0,0),(1,0),(1,1),(0,1) rotated 90 degrees around (0,0)
        sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = rotate_polygon(sq, 90.0, center=(0.0, 0.0))
        # (1,0) -> (0,1) after 90 CCW around origin
        assert result[1][0] == pytest.approx(0.0, abs=1e-9)
        assert result[1][1] == pytest.approx(1.0, abs=1e-9)

    def test_custom_center_used(self):
        sq = _square()
        r1 = rotate_polygon(sq, 45.0, center=(0.5, 0.5))
        r2 = rotate_polygon(sq, 45.0, center=(0.0, 0.0))
        # Different centers → different results
        assert r1 != r2

    def test_preserves_length(self):
        sq = _square()
        assert len(rotate_polygon(sq, 30.0)) == len(sq)


# ─── polygon_similarity (extra) ───────────────────────────────────────────────

class TestPolygonSimilarityExtra:
    def test_identical_polygons_similarity_one(self):
        sq = _square()
        assert polygon_similarity(sq, sq) == pytest.approx(1.0)

    def test_result_in_0_1(self):
        result = polygon_similarity(_square(), _rect(2, 1))
        assert 0.0 <= result <= 1.0

    def test_very_different_polygons_low_similarity(self):
        # Tiny vs huge
        small = _square(s=0.01)
        big = _square(s=100.0)
        assert polygon_similarity(small, big) < 0.5

    def test_similar_polygons_high_similarity(self):
        sq1 = _square(s=1.0)
        sq2 = _square(s=1.01)  # Nearly identical
        assert polygon_similarity(sq1, sq2) > 0.9

    def test_symmetric(self):
        sq = _square()
        tr = _triangle()
        assert polygon_similarity(sq, tr) == pytest.approx(polygon_similarity(tr, sq))

    def test_returns_float(self):
        assert isinstance(polygon_similarity(_square(), _triangle()), float)
