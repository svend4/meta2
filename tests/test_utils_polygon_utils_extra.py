"""Extra tests for puzzle_reconstruction/utils/polygon_utils.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(x=0.0, y=0.0, s=1.0):
    return [(x, y), (x+s, y), (x+s, y+s), (x, y+s)]


def _triangle():
    return [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]


# ─── polygon_area ─────────────────────────────────────────────────────────────

class TestPolygonAreaExtra:
    def test_returns_float(self):
        assert isinstance(polygon_area(_square()), float)

    def test_unit_square(self):
        assert polygon_area(_square()) == pytest.approx(1.0)

    def test_scaled_square(self):
        assert polygon_area(_square(s=3.0)) == pytest.approx(9.0)

    def test_triangle(self):
        assert abs(polygon_area(_triangle())) == pytest.approx(6.0)

    def test_less_than_3_raises(self):
        with pytest.raises(ValueError):
            polygon_area([(0.0, 0.0), (1.0, 0.0)])

    def test_nonneg_for_ccw(self):
        assert polygon_area(_square()) >= 0.0

    def test_offset_square(self):
        # Translating doesn't change area
        assert polygon_area(_square(x=100.0, y=100.0)) == pytest.approx(1.0)


# ─── polygon_perimeter ────────────────────────────────────────────────────────

class TestPolygonPerimeterExtra:
    def test_returns_float(self):
        assert isinstance(polygon_perimeter(_square()), float)

    def test_unit_square(self):
        assert polygon_perimeter(_square()) == pytest.approx(4.0)

    def test_scaled_square(self):
        assert polygon_perimeter(_square(s=2.0)) == pytest.approx(8.0)

    def test_triangle(self):
        perim = polygon_perimeter(_triangle())
        expected = 4.0 + 3.0 + 5.0  # 3-4-5 right triangle
        assert perim == pytest.approx(expected)

    def test_less_than_2_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([(0.0, 0.0)])

    def test_positive_value(self):
        assert polygon_perimeter(_square()) > 0.0


# ─── polygon_centroid ─────────────────────────────────────────────────────────

class TestPolygonCentroidExtra:
    def test_returns_tuple(self):
        result = polygon_centroid(_square())
        assert isinstance(result, tuple) and len(result) == 2

    def test_unit_square_center(self):
        cx, cy = polygon_centroid(_square())
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)

    def test_offset_square(self):
        cx, cy = polygon_centroid(_square(x=10.0, y=5.0))
        assert cx == pytest.approx(10.5)
        assert cy == pytest.approx(5.5)

    def test_less_than_3_raises(self):
        with pytest.raises(ValueError):
            polygon_centroid([(0.0, 0.0), (1.0, 0.0)])


# ─── point_in_polygon ─────────────────────────────────────────────────────────

class TestPointInPolygonExtra:
    def test_inside_returns_true(self):
        assert point_in_polygon((0.5, 0.5), _square()) is True

    def test_outside_returns_false(self):
        assert point_in_polygon((2.0, 2.0), _square()) is False

    def test_returns_bool(self):
        assert isinstance(point_in_polygon((0.5, 0.5), _square()), bool)

    def test_far_outside_false(self):
        assert point_in_polygon((-10.0, -10.0), _square()) is False

    def test_less_than_3_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 0.0)])

    def test_triangle_inside(self):
        assert point_in_polygon((1.0, 0.5), _triangle()) is True

    def test_triangle_outside(self):
        assert point_in_polygon((3.0, 3.0), _triangle()) is False


# ─── convex_hull ──────────────────────────────────────────────────────────────

class TestConvexHullExtra:
    def test_returns_list(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        assert isinstance(convex_hull(pts), list)

    def test_square_hull(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
        hull = convex_hull(pts)
        assert len(hull) >= 4

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            convex_hull([])

    def test_collinear_points(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        hull = convex_hull(pts)
        assert len(hull) >= 2

    def test_hull_area_le_original_bbox_area(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
        hull = convex_hull(pts)
        assert polygon_area(hull) <= 1.0 + 1e-9


# ─── polygon_bounding_box ─────────────────────────────────────────────────────

class TestPolygonBoundingBoxExtra:
    def test_returns_tuple_4(self):
        result = polygon_bounding_box(_square())
        assert isinstance(result, tuple) and len(result) == 4

    def test_unit_square_bbox(self):
        x0, y0, x1, y1 = polygon_bounding_box(_square())
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(1.0)
        assert y1 == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_bounding_box([])

    def test_offset_polygon(self):
        x0, y0, x1, y1 = polygon_bounding_box(_square(x=5.0, y=3.0))
        assert x0 == pytest.approx(5.0)
        assert y0 == pytest.approx(3.0)

    def test_triangle_bbox(self):
        x0, y0, x1, y1 = polygon_bounding_box(_triangle())
        assert x0 == pytest.approx(0.0)
        assert y0 == pytest.approx(0.0)
        assert x1 == pytest.approx(4.0)
        assert y1 == pytest.approx(3.0)


# ─── polygon_aspect_ratio ─────────────────────────────────────────────────────

class TestPolygonAspectRatioExtra:
    def test_returns_float(self):
        assert isinstance(polygon_aspect_ratio(_square()), float)

    def test_square_is_one(self):
        assert polygon_aspect_ratio(_square()) == pytest.approx(1.0)

    def test_wide_rectangle(self):
        rect = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0)]
        assert polygon_aspect_ratio(rect) == pytest.approx(4.0)

    def test_tall_rectangle(self):
        rect = [(0.0, 0.0), (1.0, 0.0), (1.0, 4.0), (0.0, 4.0)]
        assert polygon_aspect_ratio(rect) == pytest.approx(0.25)

    def test_nonneg(self):
        assert polygon_aspect_ratio(_square()) >= 0.0


# ─── translate_polygon ────────────────────────────────────────────────────────

class TestTranslatePolygonExtra:
    def test_returns_list(self):
        assert isinstance(translate_polygon(_square(), 1.0, 0.0), list)

    def test_length_preserved(self):
        sq = _square()
        out = translate_polygon(sq, 3.0, 2.0)
        assert len(out) == len(sq)

    def test_zero_translation_unchanged(self):
        sq = _square()
        out = translate_polygon(sq, 0.0, 0.0)
        for (ox, oy), (px, py) in zip(out, sq):
            assert ox == pytest.approx(px)
            assert oy == pytest.approx(py)

    def test_translation_applied(self):
        sq = _square()
        out = translate_polygon(sq, 5.0, 3.0)
        for (ox, oy), (px, py) in zip(out, sq):
            assert ox == pytest.approx(px + 5.0)
            assert oy == pytest.approx(py + 3.0)

    def test_area_unchanged(self):
        sq = _square()
        out = translate_polygon(sq, 10.0, -5.0)
        assert polygon_area(out) == pytest.approx(polygon_area(sq))


# ─── scale_polygon ────────────────────────────────────────────────────────────

class TestScalePolygonExtra:
    def test_returns_list(self):
        assert isinstance(scale_polygon(_square(), 2.0), list)

    def test_scale_one_unchanged(self):
        sq = _square()
        out = scale_polygon(sq, 1.0)
        for (ox, oy), (px, py) in zip(out, sq):
            assert ox == pytest.approx(px, abs=1e-9)
            assert oy == pytest.approx(py, abs=1e-9)

    def test_scale_doubles_area(self):
        sq = _square()
        out = scale_polygon(sq, 2.0)
        assert polygon_area(out) == pytest.approx(polygon_area(sq) * 4.0)

    def test_scale_zero_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(_square(), 0.0)

    def test_scale_negative_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(_square(), -1.0)

    def test_custom_center(self):
        sq = _square()
        out = scale_polygon(sq, 2.0, center=(0.5, 0.5))
        assert polygon_area(out) == pytest.approx(4.0)


# ─── rotate_polygon ───────────────────────────────────────────────────────────

class TestRotatePolygonExtra:
    def test_returns_list(self):
        assert isinstance(rotate_polygon(_square(), 0.0), list)

    def test_length_preserved(self):
        sq = _square()
        out = rotate_polygon(sq, 45.0)
        assert len(out) == len(sq)

    def test_zero_rotation_unchanged(self):
        sq = _square()
        out = rotate_polygon(sq, 0.0)
        for (ox, oy), (px, py) in zip(out, sq):
            assert ox == pytest.approx(px, abs=1e-9)
            assert oy == pytest.approx(py, abs=1e-9)

    def test_360_rotation_identity(self):
        sq = _square()
        out = rotate_polygon(sq, 360.0)
        for (ox, oy), (px, py) in zip(out, sq):
            assert ox == pytest.approx(px, abs=1e-9)
            assert oy == pytest.approx(py, abs=1e-9)

    def test_area_preserved(self):
        sq = _square()
        out = rotate_polygon(sq, 45.0)
        assert abs(polygon_area(out)) == pytest.approx(abs(polygon_area(sq)), abs=1e-9)


# ─── polygon_similarity ───────────────────────────────────────────────────────

class TestPolygonSimilarityExtra:
    def test_returns_float(self):
        sq = _square()
        assert isinstance(polygon_similarity(sq, sq), float)

    def test_identical_polygons_one(self):
        sq = _square()
        assert polygon_similarity(sq, sq) == pytest.approx(1.0)

    def test_result_in_range(self):
        sq = _square()
        big = _square(s=10.0)
        sim = polygon_similarity(sq, big)
        assert 0.0 <= sim <= 1.0

    def test_symmetric(self):
        a = _square()
        b = _square(s=2.0)
        assert polygon_similarity(a, b) == pytest.approx(polygon_similarity(b, a))

    def test_very_different_polygons(self):
        small = _square()
        large = _square(s=100.0)
        assert polygon_similarity(small, large) < 1.0
