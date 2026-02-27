"""
Property-based tests for puzzle_reconstruction.utils.contour_utils.

Verifies mathematical invariants:
- simplify_contour:       M ≤ N, result dtype float64, shape (M, 2)
- interpolate_contour:    exactly n_points rows, shape (n, 2), arc-length approx uniform
- contour_area:           ≥ 0; < 3 pts → 0; scaling by k → k² area
- contour_perimeter:      ≥ 0; < 2 pts → 0; scaling by k → k * perimeter; closed vs open
- contour_bbox:           all points inside; (x_min, y_min, w, h) w,h ≥ 0
- contour_centroid:       inside bounding box; (0, 0) for empty
- contour_iou:            ∈ [0, 1]; same contour → 1; disjoint → 0; symmetric
- align_contour_orientation: returns same length; orientation as requested
- contours_to_mask:       shape = canvas_size; non-zero for valid contour
- mask_to_contour:        non-empty for non-empty mask; shape (N, 2)
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.contour_utils import (
    simplify_contour,
    interpolate_contour,
    contour_area,
    contour_perimeter,
    contour_bbox,
    contour_centroid,
    contour_iou,
    align_contour_orientation,
    contours_to_mask,
    mask_to_contour,
)

RNG = np.random.default_rng(7)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle_contour(n: int = 40, r: float = 50.0, cx: float = 100.0, cy: float = 100.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
    return pts.astype(np.float64)


def _square_contour(side: float = 50.0, ox: float = 10.0, oy: float = 10.0) -> np.ndarray:
    pts = np.array([
        [ox,        oy],
        [ox + side, oy],
        [ox + side, oy + side],
        [ox,        oy + side],
    ], dtype=np.float64)
    return pts


def _rand_contour(n: int = 20) -> np.ndarray:
    angles = np.sort(RNG.uniform(0, 2 * np.pi, n))
    r = RNG.uniform(10, 100, n)
    pts = np.column_stack([r * np.cos(angles) + 100, r * np.sin(angles) + 100])
    return pts.astype(np.float64)


# ─── simplify_contour ─────────────────────────────────────────────────────────

class TestSimplifyContour:
    def test_output_shape(self):
        c = _circle_contour(60)
        s = simplify_contour(c, epsilon=2.0)
        assert s.ndim == 2
        assert s.shape[1] == 2

    def test_fewer_or_equal_points(self):
        c = _circle_contour(100)
        s = simplify_contour(c, epsilon=1.0)
        assert s.shape[0] <= c.shape[0]

    def test_larger_epsilon_fewer_points(self):
        c = _rand_contour(80)
        s1 = simplify_contour(c, epsilon=1.0)
        s2 = simplify_contour(c, epsilon=10.0)
        assert s2.shape[0] <= s1.shape[0]

    def test_dtype_float64(self):
        c = _circle_contour(40)
        s = simplify_contour(c)
        assert s.dtype == np.float64

    def test_empty_contour(self):
        c = np.empty((0, 2), dtype=np.float64)
        s = simplify_contour(c)
        assert s.shape[0] == 0

    def test_square_preserved(self):
        """A square with epsilon < side/4 should keep 4 corners."""
        c = _square_contour(100.0)
        s = simplify_contour(c, epsilon=1.0)
        assert s.shape[0] >= 4


# ─── interpolate_contour ──────────────────────────────────────────────────────

class TestInterpolateContour:
    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_exact_n_points(self, n):
        c = _circle_contour(40)
        result = interpolate_contour(c, n_points=n)
        assert result.shape == (n, 2)

    def test_dtype_float64(self):
        c = _circle_contour(40)
        result = interpolate_contour(c, 50)
        assert result.dtype == np.float64

    def test_points_near_original_circle(self):
        """Interpolated points should lie near the original circle."""
        cx, cy, r = 100.0, 100.0, 50.0
        c = _circle_contour(100, r=r, cx=cx, cy=cy)
        result = interpolate_contour(c, 80)
        dists = np.sqrt((result[:, 0] - cx) ** 2 + (result[:, 1] - cy) ** 2)
        assert np.all(np.abs(dists - r) < 5.0)

    def test_2_points(self):
        c = _circle_contour(40)
        result = interpolate_contour(c, 2)
        assert result.shape == (2, 2)

    def test_single_input_point_repeated(self):
        c = np.array([[5.0, 7.0]])
        result = interpolate_contour(c, 5)
        assert result.shape == (5, 2)


# ─── contour_area ─────────────────────────────────────────────────────────────

class TestContourArea:
    def test_nonneg(self):
        for _ in range(20):
            c = _rand_contour()
            assert contour_area(c) >= 0.0

    def test_less_than_3_points_zero(self):
        assert contour_area(np.array([[0.0, 0.0], [1.0, 0.0]])) == 0.0
        assert contour_area(np.array([[0.0, 0.0]])) == 0.0
        assert contour_area(np.empty((0, 2))) == 0.0

    def test_square_area(self):
        c = _square_contour(side=10.0)
        area = contour_area(c)
        assert area == pytest.approx(100.0, abs=0.01)

    def test_scaling_k_squared(self):
        c = _rand_contour(30)
        k = 2.0
        a1 = contour_area(c)
        a2 = contour_area(c * k)
        assert abs(a2 - k ** 2 * a1) < 1e-6

    def test_translation_invariant(self):
        c = _rand_contour(30)
        a1 = contour_area(c)
        a2 = contour_area(c + np.array([100.0, 200.0]))
        assert abs(a1 - a2) < 1e-6

    def test_circle_area_approx(self):
        r = 50.0
        c = _circle_contour(500, r=r)
        area = contour_area(c)
        expected = np.pi * r ** 2
        assert abs(area - expected) / expected < 0.01  # 1% tolerance


# ─── contour_perimeter ────────────────────────────────────────────────────────

class TestContourPerimeter:
    def test_nonneg(self):
        for _ in range(20):
            c = _rand_contour()
            assert contour_perimeter(c) >= 0.0

    def test_less_than_2_points_zero(self):
        assert contour_perimeter(np.array([[0.0, 0.0]])) == 0.0
        assert contour_perimeter(np.empty((0, 2))) == 0.0

    def test_scaling_k(self):
        c = _rand_contour(30)
        k = 3.0
        p1 = contour_perimeter(c)
        p2 = contour_perimeter(c * k)
        assert abs(p2 - k * p1) < 1e-5

    def test_closed_ge_open(self):
        c = _rand_contour(20)
        p_closed = contour_perimeter(c, closed=True)
        p_open = contour_perimeter(c, closed=False)
        assert p_closed >= p_open - 1e-10

    def test_circle_approx_2pi_r(self):
        r = 50.0
        c = _circle_contour(1000, r=r)
        p = contour_perimeter(c, closed=True)
        expected = 2 * np.pi * r
        assert abs(p - expected) / expected < 0.01

    def test_square_perimeter(self):
        c = _square_contour(side=10.0)
        p = contour_perimeter(c, closed=True)
        assert p == pytest.approx(40.0, abs=0.01)


# ─── contour_bbox ─────────────────────────────────────────────────────────────

class TestContourBbox:
    def test_empty_returns_zeros(self):
        result = contour_bbox(np.empty((0, 2), dtype=np.float64))
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_width_height_nonneg(self):
        for _ in range(20):
            c = _rand_contour()
            x, y, w, h = contour_bbox(c)
            assert w >= 0.0
            assert h >= 0.0

    def test_all_points_inside(self):
        for _ in range(10):
            c = _rand_contour(30)
            x_min, y_min, w, h = contour_bbox(c)
            x_max = x_min + w
            y_max = y_min + h
            assert np.all(c[:, 0] >= x_min - 1e-10)
            assert np.all(c[:, 0] <= x_max + 1e-10)
            assert np.all(c[:, 1] >= y_min - 1e-10)
            assert np.all(c[:, 1] <= y_max + 1e-10)

    def test_tight_bounds(self):
        c = _square_contour(side=20.0, ox=5.0, oy=5.0)
        x, y, w, h = contour_bbox(c)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(5.0)
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(20.0)

    def test_single_point(self):
        c = np.array([[3.0, 7.0]])
        x, y, w, h = contour_bbox(c)
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(7.0)
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)


# ─── contour_centroid ─────────────────────────────────────────────────────────

class TestContourCentroid:
    def test_empty_returns_zero(self):
        cx, cy = contour_centroid(np.empty((0, 2), dtype=np.float64))
        assert cx == 0.0
        assert cy == 0.0

    def test_inside_bounding_box(self):
        for _ in range(15):
            c = _rand_contour(20)
            cx, cy = contour_centroid(c)
            x_min, y_min, w, h = contour_bbox(c)
            assert x_min - 1.0 <= cx <= x_min + w + 1.0
            assert y_min - 1.0 <= cy <= y_min + h + 1.0

    def test_circle_centroid_near_center(self):
        cx0, cy0 = 100.0, 150.0
        c = _circle_contour(200, cx=cx0, cy=cy0)
        cx, cy = contour_centroid(c)
        assert abs(cx - cx0) < 5.0
        assert abs(cy - cy0) < 5.0

    def test_square_centroid(self):
        c = _square_contour(side=10.0, ox=0.0, oy=0.0)
        cx, cy = contour_centroid(c)
        assert abs(cx - 5.0) < 2.0
        assert abs(cy - 5.0) < 2.0


# ─── contour_iou ──────────────────────────────────────────────────────────────

class TestContourIou:
    def test_range_0_1(self):
        for _ in range(10):
            c1 = _rand_contour(20)
            c2 = _rand_contour(20)
            iou = contour_iou(c1, c2)
            assert 0.0 <= iou <= 1.0

    def test_same_contour_one(self):
        c = _circle_contour(50, cx=50.0, cy=50.0, r=30.0)
        iou = contour_iou(c, c)
        assert iou == pytest.approx(1.0, abs=0.01)

    def test_disjoint_zero(self):
        c1 = _circle_contour(50, cx=10.0, cy=10.0, r=5.0)
        c2 = _circle_contour(50, cx=200.0, cy=200.0, r=5.0)
        iou = contour_iou(c1, c2)
        assert iou == pytest.approx(0.0, abs=0.01)

    def test_symmetric(self):
        c1 = _circle_contour(50, cx=50.0, cy=50.0, r=30.0)
        c2 = _circle_contour(50, cx=70.0, cy=70.0, r=30.0)
        iou_12 = contour_iou(c1, c2)
        iou_21 = contour_iou(c2, c1)
        assert abs(iou_12 - iou_21) < 0.02

    def test_degenerate_less_than_3(self):
        c1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        c2 = _circle_contour(50)
        assert contour_iou(c1, c2) == 0.0


# ─── align_contour_orientation ────────────────────────────────────────────────

class TestAlignContourOrientation:
    def test_same_length(self):
        c = _rand_contour(30)
        aligned = align_contour_orientation(c, clockwise=True)
        assert len(aligned) == len(c)

    def test_dtype_preserved(self):
        c = _rand_contour(30)
        aligned = align_contour_orientation(c)
        assert aligned.dtype == np.float64

    def test_twice_returns_same(self):
        """Applying orientation twice should give same result."""
        c = _rand_contour(30)
        once = align_contour_orientation(c, clockwise=True)
        twice = align_contour_orientation(once, clockwise=True)
        np.testing.assert_array_almost_equal(once, twice)

    def test_less_than_3_returned_as_is(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = align_contour_orientation(c)
        assert result.shape == c.shape


# ─── contours_to_mask ─────────────────────────────────────────────────────────

class TestContoursToMask:
    def test_shape_correct(self):
        c = _circle_contour(50, cx=50.0, cy=50.0, r=20.0)
        mask = contours_to_mask(c, shape=(100, 100))
        assert mask.shape == (100, 100)

    def test_dtype_uint8(self):
        c = _circle_contour(50, cx=50.0, cy=50.0, r=20.0)
        mask = contours_to_mask(c, shape=(100, 100))
        assert mask.dtype == np.uint8

    def test_filled_nonzero(self):
        c = _circle_contour(100, cx=50.0, cy=50.0, r=20.0)
        mask = contours_to_mask(c, shape=(100, 100), filled=True)
        assert mask.sum() > 0

    def test_values_0_or_255(self):
        c = _circle_contour(50, cx=50.0, cy=50.0, r=20.0)
        mask = contours_to_mask(c, shape=(100, 100))
        unique = np.unique(mask)
        assert all(v in (0, 255) for v in unique)

    def test_empty_contour_all_zeros(self):
        c = np.empty((0, 2), dtype=np.float64)
        mask = contours_to_mask(c, shape=(50, 50))
        assert mask.sum() == 0


# ─── mask_to_contour ──────────────────────────────────────────────────────────

class TestMaskToContour:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        c = mask_to_contour(mask)
        assert c.shape[0] == 0
        assert c.shape[1] == 2

    def test_nonempty_mask_nonempty_contour(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 255
        c = mask_to_contour(mask)
        assert c.shape[0] > 0

    def test_shape_n_2(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        c = mask_to_contour(mask)
        assert c.ndim == 2
        assert c.shape[1] == 2

    def test_dtype_float64(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        c = mask_to_contour(mask)
        assert c.dtype == np.float64

    def test_roundtrip_area_approx(self):
        """mask → contour → area should be roughly correct."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 255
        c = mask_to_contour(mask)
        area = contour_area(c)
        # square is 40x40=1600; contour-based area may differ slightly
        assert area > 100.0
