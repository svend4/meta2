"""Тесты для puzzle_reconstruction.utils.polygon_utils."""
import math
import pytest
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

def _square(side=10.0, ox=0.0, oy=0.0):
    """Квадрат с нижним левым углом (ox, oy)."""
    return [(ox, oy), (ox + side, oy),
            (ox + side, oy + side), (ox, oy + side)]


def _triangle():
    """Прямоугольный треугольник 3-4-5."""
    return [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]


def _regular_hex(r=1.0):
    """Правильный шестиугольник с радиусом r."""
    return [(r * math.cos(math.pi / 3 * i),
             r * math.sin(math.pi / 3 * i)) for i in range(6)]


# ─── TestPolygonArea ──────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_unit_square(self):
        assert polygon_area(_square(1.0)) == pytest.approx(1.0)

    def test_10x10_square(self):
        assert polygon_area(_square(10.0)) == pytest.approx(100.0)

    def test_triangle_3_4(self):
        # Area = 0.5 * 4 * 3
        assert polygon_area(_triangle()) == pytest.approx(6.0)

    def test_regular_hex(self):
        # Area = 3*sqrt(3)/2 * r^2 ≈ 2.598 for r=1
        area = polygon_area(_regular_hex(1.0))
        assert area == pytest.approx(3 * math.sqrt(3) / 2, rel=1e-4)

    def test_translated_same_area(self):
        sq = _square(5.0)
        shifted = _square(5.0, 100.0, 100.0)
        assert polygon_area(sq) == pytest.approx(polygon_area(shifted))

    def test_degenerate_line_zero_area(self):
        line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        assert polygon_area(line) == pytest.approx(0.0)

    def test_two_vertices_raises(self):
        with pytest.raises(ValueError):
            polygon_area([(0.0, 0.0), (1.0, 0.0)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_area([])

    def test_cw_same_as_ccw(self):
        ccw = _square(5.0)
        cw = list(reversed(ccw))
        assert polygon_area(ccw) == pytest.approx(polygon_area(cw))

    def test_large_polygon(self):
        n = 360
        r = 100.0
        pts = [(r * math.cos(2 * math.pi * i / n),
                r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        area = polygon_area(pts)
        assert area == pytest.approx(math.pi * r ** 2, rel=1e-3)


# ─── TestPolygonPerimeter ─────────────────────────────────────────────────────

class TestPolygonPerimeter:
    def test_unit_square(self):
        assert polygon_perimeter(_square(1.0)) == pytest.approx(4.0)

    def test_10x10_square(self):
        assert polygon_perimeter(_square(10.0)) == pytest.approx(40.0)

    def test_triangle_3_4_5(self):
        # Периметр = 3 + 4 + 5 = 12
        assert polygon_perimeter(_triangle()) == pytest.approx(3 + 4 + 5)

    def test_single_vertex_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([(0.0, 0.0)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_perimeter([])

    def test_two_vertices_closed(self):
        # Два вершины: идёт туда и обратно → 2 * расстояние
        p = polygon_perimeter([(0.0, 0.0), (5.0, 0.0)])
        assert p == pytest.approx(10.0)

    def test_translated_same_perimeter(self):
        sq = _square(5.0)
        shifted = _square(5.0, 50.0, 50.0)
        assert polygon_perimeter(sq) == pytest.approx(polygon_perimeter(shifted))

    def test_large_polygon_approaches_circle(self):
        n = 1000
        r = 10.0
        pts = [(r * math.cos(2 * math.pi * i / n),
                r * math.sin(2 * math.pi * i / n)) for i in range(n)]
        p = polygon_perimeter(pts)
        assert p == pytest.approx(2 * math.pi * r, rel=1e-3)


# ─── TestPolygonCentroid ──────────────────────────────────────────────────────

class TestPolygonCentroid:
    def test_unit_square_center(self):
        cx, cy = polygon_centroid(_square(2.0))
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(1.0)

    def test_translated_square(self):
        cx, cy = polygon_centroid(_square(4.0, 5.0, 3.0))
        assert cx == pytest.approx(7.0)
        assert cy == pytest.approx(5.0)

    def test_triangle_centroid(self):
        tri = [(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]
        cx, cy = polygon_centroid(tri)
        assert cx == pytest.approx(3.0)
        assert cy == pytest.approx(2.0)

    def test_regular_hex_origin(self):
        cx, cy = polygon_centroid(_regular_hex(5.0))
        assert cx == pytest.approx(0.0, abs=1e-9)
        assert cy == pytest.approx(0.0, abs=1e-9)

    def test_two_vertices_raises(self):
        with pytest.raises(ValueError):
            polygon_centroid([(0.0, 0.0), (1.0, 0.0)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_centroid([])

    def test_centroid_inside_convex(self):
        sq = _square(10.0)
        cx, cy = polygon_centroid(sq)
        assert 0.0 < cx < 10.0
        assert 0.0 < cy < 10.0


# ─── TestPointInPolygon ───────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_center_inside(self):
        sq = _square(10.0)
        assert point_in_polygon((5.0, 5.0), sq) is True

    def test_outside(self):
        sq = _square(10.0)
        assert point_in_polygon((15.0, 5.0), sq) is False

    def test_inside_triangle(self):
        tri = [(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]
        assert point_in_polygon((3.0, 1.0), tri) is True

    def test_outside_triangle(self):
        tri = [(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)]
        assert point_in_polygon((6.0, 6.0), tri) is False

    def test_far_outside(self):
        sq = _square(1.0)
        assert point_in_polygon((100.0, 100.0), sq) is False

    def test_two_vertices_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            point_in_polygon((0.0, 0.0), [])

    def test_returns_bool(self):
        result = point_in_polygon((5.0, 5.0), _square(10.0))
        assert isinstance(result, bool)


# ─── TestConvexHull ───────────────────────────────────────────────────────────

class TestConvexHull:
    def test_square_hull_has_4_points(self):
        hull = convex_hull(_square(4.0))
        assert len(hull) == 4

    def test_interior_point_excluded(self):
        pts = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 2.0)]
        hull = convex_hull(pts)
        assert (2.0, 2.0) not in hull
        assert len(hull) == 4

    def test_single_point(self):
        hull = convex_hull([(3.0, 5.0)])
        assert len(hull) == 1

    def test_two_points(self):
        hull = convex_hull([(0.0, 0.0), (5.0, 0.0)])
        assert len(hull) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            convex_hull([])

    def test_collinear_points(self):
        pts = [(i * 1.0, 0.0) for i in range(5)]
        hull = convex_hull(pts)
        assert len(hull) >= 2

    def test_returns_list_of_tuples(self):
        hull = convex_hull([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        for pt in hull:
            assert isinstance(pt, tuple)
            assert len(pt) == 2

    def test_duplicate_points(self):
        pts = [(0.0, 0.0)] * 5 + [(1.0, 0.0), (0.0, 1.0)]
        hull = convex_hull(pts)
        assert len(hull) >= 1

    def test_hull_encloses_points(self):
        import random
        rng = random.Random(42)
        pts = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(20)]
        hull = convex_hull(pts)
        assert len(hull) >= 3


# ─── TestPolygonBoundingBox ───────────────────────────────────────────────────

class TestPolygonBoundingBox:
    def test_unit_square(self):
        bbox = polygon_bounding_box(_square(1.0))
        assert bbox == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_translated(self):
        bbox = polygon_bounding_box(_square(5.0, 3.0, 7.0))
        assert bbox == pytest.approx((3.0, 7.0, 8.0, 12.0))

    def test_single_point(self):
        bbox = polygon_bounding_box([(4.0, 6.0)])
        assert bbox == pytest.approx((4.0, 6.0, 4.0, 6.0))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            polygon_bounding_box([])

    def test_returns_4_tuple(self):
        bbox = polygon_bounding_box(_square(3.0))
        assert len(bbox) == 4

    def test_x_min_less_x_max(self):
        x_min, y_min, x_max, y_max = polygon_bounding_box(_square(5.0))
        assert x_min <= x_max
        assert y_min <= y_max


# ─── TestPolygonAspectRatio ───────────────────────────────────────────────────

class TestPolygonAspectRatio:
    def test_square_ratio_one(self):
        assert polygon_aspect_ratio(_square(5.0)) == pytest.approx(1.0)

    def test_wide_rectangle(self):
        rect = [(0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0)]
        assert polygon_aspect_ratio(rect) == pytest.approx(5.0)

    def test_tall_rectangle(self):
        rect = [(0.0, 0.0), (2.0, 0.0), (2.0, 10.0), (0.0, 10.0)]
        assert polygon_aspect_ratio(rect) == pytest.approx(0.2)

    def test_flat_line_zero(self):
        # Высота → 0, ratio → 0
        line = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        assert polygon_aspect_ratio(line) == pytest.approx(0.0)

    def test_single_point(self):
        # w = 0, h = 0 → ratio = 0
        assert polygon_aspect_ratio([(3.0, 3.0)]) == pytest.approx(0.0)


# ─── TestTranslatePolygon ─────────────────────────────────────────────────────

class TestTranslatePolygon:
    def test_basic(self):
        sq = _square(5.0)
        shifted = translate_polygon(sq, 3.0, 4.0)
        assert shifted[0] == pytest.approx((3.0, 4.0))
        assert shifted[1] == pytest.approx((8.0, 4.0))

    def test_zero_shift(self):
        sq = _square(5.0)
        shifted = translate_polygon(sq, 0.0, 0.0)
        for a, b in zip(sq, shifted):
            assert a == pytest.approx(b)

    def test_negative_shift(self):
        sq = _square(5.0, 10.0, 10.0)
        shifted = translate_polygon(sq, -10.0, -10.0)
        assert shifted[0] == pytest.approx((0.0, 0.0))

    def test_does_not_mutate(self):
        sq = _square(5.0)
        original = list(sq)
        translate_polygon(sq, 10.0, 10.0)
        assert sq == original

    def test_preserves_length(self):
        sq = _square(5.0)
        shifted = translate_polygon(sq, 1.0, 1.0)
        assert len(shifted) == len(sq)

    def test_area_preserved(self):
        sq = _square(5.0)
        shifted = translate_polygon(sq, 7.0, 3.0)
        assert polygon_area(sq) == pytest.approx(polygon_area(shifted))


# ─── TestScalePolygon ─────────────────────────────────────────────────────────

class TestScalePolygon:
    def test_scale_2_doubles_side(self):
        sq = _square(5.0, 0.0, 0.0)
        scaled = scale_polygon(sq, 2.0)
        area_before = polygon_area(sq)
        area_after = polygon_area(scaled)
        assert area_after == pytest.approx(area_before * 4.0)

    def test_scale_one_identity(self):
        sq = _square(5.0)
        scaled = scale_polygon(sq, 1.0)
        for a, b in zip(sq, scaled):
            assert a == pytest.approx(b)

    def test_scale_half_reduces(self):
        sq = _square(4.0)
        scaled = scale_polygon(sq, 0.5)
        assert polygon_area(scaled) == pytest.approx(polygon_area(sq) * 0.25)

    def test_scale_zero_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(_square(5.0), 0.0)

    def test_scale_neg_raises(self):
        with pytest.raises(ValueError):
            scale_polygon(_square(5.0), -1.0)

    def test_does_not_mutate(self):
        sq = _square(5.0)
        original = list(sq)
        scale_polygon(sq, 2.0)
        assert sq == original

    def test_custom_center(self):
        sq = _square(4.0)
        scaled = scale_polygon(sq, 2.0, center=(0.0, 0.0))
        assert scaled[0] == pytest.approx((0.0, 0.0))

    def test_preserves_length(self):
        sq = _square(5.0)
        assert len(scale_polygon(sq, 1.5)) == len(sq)


# ─── TestRotatePolygon ────────────────────────────────────────────────────────

class TestRotatePolygon:
    def test_rotate_360_identity(self):
        sq = _square(5.0)
        rotated = rotate_polygon(sq, 360.0)
        for a, b in zip(sq, rotated):
            assert a == pytest.approx(b, abs=1e-8)

    def test_rotate_0_identity(self):
        sq = _square(5.0)
        rotated = rotate_polygon(sq, 0.0)
        for a, b in zip(sq, rotated):
            assert a == pytest.approx(b, abs=1e-8)

    def test_rotate_90_square(self):
        sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        rotated = rotate_polygon(sq, 90.0, center=(2.0, 2.0))
        # (4, 0) → (4, 4)
        assert rotated[1] == pytest.approx((4.0, 4.0), abs=1e-8)

    def test_area_preserved(self):
        sq = _square(5.0)
        rotated = rotate_polygon(sq, 45.0)
        assert polygon_area(rotated) == pytest.approx(polygon_area(sq), rel=1e-6)

    def test_perimeter_preserved(self):
        sq = _square(5.0)
        rotated = rotate_polygon(sq, 30.0)
        assert polygon_perimeter(rotated) == pytest.approx(polygon_perimeter(sq), rel=1e-6)

    def test_does_not_mutate(self):
        sq = _square(5.0)
        original = list(sq)
        rotate_polygon(sq, 45.0)
        assert sq == original

    def test_preserves_length(self):
        sq = _square(5.0)
        assert len(rotate_polygon(sq, 45.0)) == len(sq)

    def test_custom_center(self):
        # Rotate (1, 0) 90° around origin
        pt = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]
        rotated = rotate_polygon(pt, 90.0, center=(0.0, 0.0))
        assert rotated[0] == pytest.approx((0.0, 1.0), abs=1e-8)


# ─── TestPolygonSimilarity ────────────────────────────────────────────────────

class TestPolygonSimilarity:
    def test_identical_polygons(self):
        sq = _square(5.0)
        assert polygon_similarity(sq, sq) == pytest.approx(1.0)

    def test_same_shape_different_size(self):
        sq1 = _square(5.0)
        sq2 = _square(10.0)
        sim = polygon_similarity(sq1, sq2)
        assert 0.0 < sim < 1.0

    def test_completely_different(self):
        # Маленький треугольник vs огромный квадрат
        tri = [(0.0, 0.0), (0.001, 0.0), (0.0, 0.001)]
        sq = _square(1000.0)
        sim = polygon_similarity(tri, sq)
        assert sim < 0.5

    def test_in_range(self):
        sim = polygon_similarity(_square(5.0), _triangle())
        assert 0.0 <= sim <= 1.0

    def test_symmetric(self):
        a = _square(5.0)
        b = _square(8.0)
        assert polygon_similarity(a, b) == pytest.approx(polygon_similarity(b, a))

    def test_similar_shapes_high_score(self):
        sq1 = _square(5.0)
        sq2 = _square(5.1)
        assert polygon_similarity(sq1, sq2) > 0.8

    def test_returns_float(self):
        result = polygon_similarity(_square(1.0), _triangle())
        assert isinstance(result, float)
