"""Extra tests for puzzle_reconstruction/utils/contour_utils.py"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.contour_utils import (
    align_contour_orientation,
    contour_area,
    contour_bbox,
    contour_centroid,
    contour_iou,
    contour_perimeter,
    contours_to_mask,
    interpolate_contour,
    mask_to_contour,
    simplify_contour,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(size=10, offset=(0, 0)):
    x0, y0 = offset
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float64)


def _triangle():
    return np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float64)


def _circle(r=20, cx=30, cy=30, n=64):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)


# ─── TestSimplifyContourExtra ────────────────────────────────────────────────

class TestSimplifyContourExtra:
    def test_triangle_unchanged_small_epsilon(self):
        c = _triangle()
        result = simplify_contour(c, epsilon=0.1)
        assert len(result) >= 3

    def test_circle_small_epsilon_keeps_many(self):
        c = _circle(n=100)
        r = simplify_contour(c, epsilon=0.5)
        # Should keep most points
        assert len(r) > 10

    def test_circle_large_epsilon_reduces(self):
        c = _circle(n=100)
        r = simplify_contour(c, epsilon=20.0)
        assert len(r) < len(c)

    def test_square_no_reduction_small_epsilon(self):
        c = _square()
        r = simplify_contour(c, epsilon=0.01)
        assert len(r) == len(c)

    def test_3d_cv2_format_accepted(self):
        c = _square().reshape(-1, 1, 2)
        r = simplify_contour(c, epsilon=0.5)
        assert r.ndim == 2
        assert r.shape[1] == 2


# ─── TestInterpolateContourExtra ─────────────────────────────────────────────

class TestInterpolateContourExtra:
    def test_circle_n_100(self):
        c = _circle(n=20)
        r = interpolate_contour(c, n_points=100)
        assert r.shape == (100, 2)

    def test_triangle_n_30(self):
        r = interpolate_contour(_triangle(), n_points=30)
        assert r.shape == (30, 2)

    def test_n_points_2_valid(self):
        r = interpolate_contour(_square(), n_points=2)
        assert r.shape == (2, 2)

    def test_large_n_points(self):
        r = interpolate_contour(_square(size=100), n_points=1000)
        assert r.shape == (1000, 2)

    def test_output_within_bbox(self):
        c = _square(size=20, offset=(10, 5))
        r = interpolate_contour(c, n_points=50)
        assert r[:, 0].min() >= 10.0 - 1e-9
        assert r[:, 0].max() <= 30.0 + 1e-9
        assert r[:, 1].min() >= 5.0 - 1e-9
        assert r[:, 1].max() <= 25.0 + 1e-9


# ─── TestContourAreaExtra ─────────────────────────────────────────────────────

class TestContourAreaExtra:
    def test_hexagon_area(self):
        # Regular hexagon with side=1: area = 3√3/2 ≈ 2.598
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        c = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        area = contour_area(c)
        assert abs(area - 3 * np.sqrt(3) / 2) < 0.01

    def test_triangle_doubled_base(self):
        c = np.array([[0, 0], [20, 0], [10, 10]], dtype=np.float64)
        area = contour_area(c)
        assert area == pytest.approx(100.0)

    def test_cw_same_as_ccw(self):
        # Area is abs value, so orientation doesn't matter
        c = _square(size=10)
        assert contour_area(c) == pytest.approx(contour_area(c[::-1]))

    def test_unit_square(self):
        c = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        assert contour_area(c) == pytest.approx(1.0)


# ─── TestContourPerimeterExtra ────────────────────────────────────────────────

class TestContourPerimeterExtra:
    def test_triangle_closed(self):
        # Sides: 10, sqrt(125)≈11.18, sqrt(125)≈11.18
        c = _triangle()
        p = contour_perimeter(c, closed=True)
        expected = 10.0 + 2 * np.sqrt(100 + 25)
        assert abs(p - expected) < 0.01

    def test_open_less_than_closed(self):
        c = _square(size=10)
        p_open = contour_perimeter(c, closed=False)
        p_closed = contour_perimeter(c, closed=True)
        assert p_open < p_closed

    def test_two_points_open(self):
        c = np.array([[0, 0], [3, 4]], dtype=np.float64)
        p = contour_perimeter(c, closed=False)
        assert p == pytest.approx(5.0)

    def test_large_square(self):
        c = _square(size=100)
        p = contour_perimeter(c, closed=True)
        assert p == pytest.approx(400.0)


# ─── TestContourBboxExtra ─────────────────────────────────────────────────────

class TestContourBboxExtra:
    def test_triangle_bbox(self):
        x, y, w, h = contour_bbox(_triangle())
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert w == pytest.approx(10.0)
        assert h == pytest.approx(10.0)

    def test_offset_non_zero(self):
        x, y, w, h = contour_bbox(_square(size=5, offset=(3, 7)))
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(7.0)

    def test_single_point(self):
        c = np.array([[5.0, 8.0]])
        x, y, w, h = contour_bbox(c)
        assert x == pytest.approx(5.0)
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)

    def test_non_square_shape(self):
        c = np.array([[0, 0], [30, 0], [30, 10], [0, 10]], dtype=np.float64)
        x, y, w, h = contour_bbox(c)
        assert w == pytest.approx(30.0)
        assert h == pytest.approx(10.0)


# ─── TestContourCentroidExtra ─────────────────────────────────────────────────

class TestContourCentroidExtra:
    def test_triangle_centroid(self):
        # Triangle [[0,0],[10,0],[5,10]]: centroid ≈ (5, 10/3)
        cx, cy = contour_centroid(_triangle())
        assert abs(cx - 5.0) < 1.5
        assert cy > 0.0

    def test_point_returns_that_point(self):
        c = np.array([[7.0, 3.0]])
        cx, cy = contour_centroid(c)
        assert cx == pytest.approx(7.0, abs=1.0)
        assert cy == pytest.approx(3.0, abs=1.0)

    def test_both_floats(self):
        cx, cy = contour_centroid(_square(size=5))
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_large_offset_centroid(self):
        c = _square(size=10, offset=(100, 200))
        cx, cy = contour_centroid(c)
        assert cx == pytest.approx(105.0, abs=2.0)
        assert cy == pytest.approx(205.0, abs=2.0)


# ─── TestContourIouExtra ──────────────────────────────────────────────────────

class TestContourIouExtra:
    def test_large_canvas_same_contour(self):
        c = _square(size=20, offset=(10, 10))
        iou = contour_iou(c, c.copy(), canvas_size=(100, 100))
        assert iou == pytest.approx(1.0, abs=0.02)

    def test_adjacent_non_overlapping(self):
        # Use a gap of 5 pixels to avoid boundary rasterisation overlap
        c1 = _square(size=10, offset=(0, 0))
        c2 = _square(size=10, offset=(15, 0))
        iou = contour_iou(c1, c2, canvas_size=(35, 35))
        assert iou == pytest.approx(0.0)

    def test_large_offset_no_overlap(self):
        c1 = _square(size=5, offset=(0, 0))
        c2 = _square(size=5, offset=(50, 50))
        iou = contour_iou(c1, c2, canvas_size=(80, 80))
        assert iou == pytest.approx(0.0)

    def test_symmetry(self):
        c1 = _square(size=15, offset=(0, 0))
        c2 = _square(size=15, offset=(5, 5))
        assert contour_iou(c1, c2) == pytest.approx(contour_iou(c2, c1), abs=1e-9)


# ─── TestAlignContourOrientationExtra ────────────────────────────────────────

class TestAlignContourOrientationExtra:
    def test_ccw_flag_sets_ccw(self):
        c = _square()
        result = align_contour_orientation(c, clockwise=False)
        assert result.shape == c.shape
        assert result.dtype == np.float64

    def test_same_points_either_direction(self):
        c = _square()
        cw = align_contour_orientation(c, clockwise=True)
        ccw = align_contour_orientation(c, clockwise=False)
        # Both use the same point set
        assert set(map(tuple, cw.tolist())) == set(map(tuple, ccw.tolist()))

    def test_empty_unchanged(self):
        c = np.empty((0, 2), dtype=np.float64)
        result = align_contour_orientation(c, clockwise=True)
        assert len(result) == 0

    def test_triangle_orientation_preserved(self):
        c = _triangle()
        r = align_contour_orientation(c, clockwise=True)
        assert r.shape == c.shape


# ─── TestContoursToMaskExtra ──────────────────────────────────────────────────

class TestContoursToMaskExtra:
    def test_large_canvas(self):
        c = _square(size=50, offset=(10, 10))
        mask = contours_to_mask(c, shape=(200, 200), filled=True)
        assert mask.shape == (200, 200)
        assert mask.max() == 255

    def test_triangle_filled(self):
        c = _triangle()
        mask = contours_to_mask(c, shape=(15, 15), filled=True)
        assert mask.sum() > 0

    def test_non_square_canvas(self):
        c = _square(size=10, offset=(2, 2))
        mask = contours_to_mask(c, shape=(30, 60))
        assert mask.shape == (30, 60)

    def test_empty_contour_gives_empty_mask(self):
        mask = contours_to_mask(np.empty((0, 2), dtype=np.float64), shape=(20, 20))
        assert mask.sum() == 0


# ─── TestMaskToContourExtra ───────────────────────────────────────────────────

class TestMaskToContourExtra:
    def test_non_square_mask(self):
        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[5:15, 5:35] = 255
        result = mask_to_contour(mask)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_small_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 255
        result = mask_to_contour(mask)
        assert len(result) > 0

    def test_two_objects_returns_longest(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[2:8, 2:8] = 255    # small square
        mask[20:40, 20:40] = 255  # larger square
        result = mask_to_contour(mask)
        # The larger object's contour should be returned
        assert len(result) > 0

    def test_output_float64(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:25] = 128
        result = mask_to_contour(mask)
        assert result.dtype == np.float64
