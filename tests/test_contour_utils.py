"""Tests for puzzle_reconstruction.utils.contour_utils."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _square(size=10, offset=(0, 0)):
    """Axis-aligned square contour (4 points), float64."""
    x0, y0 = offset
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float64)


def _triangle():
    return np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float64)


def _collinear():
    """Degenerate 3-point 'contour' — all on the same line."""
    return np.array([[0, 0], [5, 0], [10, 0]], dtype=np.float64)


# ─── simplify_contour ────────────────────────────────────────────────────────

class TestSimplifyContour:
    def test_returns_2d_float64(self):
        c = _square()
        result = simplify_contour(c, epsilon=0.5)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert result.dtype == np.float64

    def test_result_not_more_points(self):
        c = _square()
        result = simplify_contour(c, epsilon=0.5)
        assert len(result) <= len(c)

    def test_large_epsilon_reduces_points(self):
        # Dense circle-like polygon; large epsilon should reduce it heavily
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        c = np.stack([np.cos(angles) * 50 + 100, np.sin(angles) * 50 + 100], axis=1)
        simplified = simplify_contour(c, epsilon=10.0)
        assert len(simplified) < len(c)

    def test_empty_contour_returns_empty(self):
        c = np.empty((0, 2), dtype=np.float64)
        result = simplify_contour(c, epsilon=1.0)
        assert len(result) == 0

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            simplify_contour(np.ones((5, 3)), epsilon=1.0)


# ─── interpolate_contour ─────────────────────────────────────────────────────

class TestInterpolateContour:
    def test_output_shape(self):
        c = _square()
        result = interpolate_contour(c, n_points=20)
        assert result.shape == (20, 2)

    def test_dtype_float64(self):
        result = interpolate_contour(_square(), n_points=10)
        assert result.dtype == np.float64

    def test_n_points_less_than_2_raises(self):
        with pytest.raises(ValueError):
            interpolate_contour(_square(), n_points=1)

    def test_empty_contour_raises(self):
        with pytest.raises(ValueError):
            interpolate_contour(np.empty((0, 2)), n_points=10)

    def test_degenerate_single_point_returns_filled(self):
        # All same point → degenerate contour
        c = np.ones((4, 2), dtype=np.float64) * 5.0
        result = interpolate_contour(c, n_points=8)
        assert result.shape == (8, 2)
        np.testing.assert_allclose(result, 5.0)

    def test_custom_n_points(self):
        for n in [4, 50, 100]:
            result = interpolate_contour(_square(size=100), n_points=n)
            assert len(result) == n

    def test_points_within_bounding_box(self):
        c = _square(size=20, offset=(5, 5))
        result = interpolate_contour(c, n_points=40)
        assert result[:, 0].min() >= 5.0 - 1e-9
        assert result[:, 0].max() <= 25.0 + 1e-9
        assert result[:, 1].min() >= 5.0 - 1e-9
        assert result[:, 1].max() <= 25.0 + 1e-9


# ─── contour_area ────────────────────────────────────────────────────────────

class TestContourArea:
    def test_less_than_3_points_zero(self):
        assert contour_area(np.array([[0, 0], [1, 0]], dtype=np.float64)) == 0.0

    def test_square_area(self):
        area = contour_area(_square(size=10))
        assert area == pytest.approx(100.0)

    def test_triangle_area(self):
        # Base=10, height=10 → area=50
        area = contour_area(_triangle())
        assert area == pytest.approx(50.0)

    def test_non_negative(self):
        # CW winding should still return positive area
        c = _square()[::-1]
        assert contour_area(c) >= 0.0

    def test_large_square(self):
        area = contour_area(_square(size=100))
        assert area == pytest.approx(10000.0)


# ─── contour_perimeter ───────────────────────────────────────────────────────

class TestContourPerimeter:
    def test_less_than_2_points_zero(self):
        assert contour_perimeter(np.array([[1, 1]], dtype=np.float64)) == 0.0

    def test_square_closed(self):
        # 4 sides of length 10
        p = contour_perimeter(_square(size=10), closed=True)
        assert p == pytest.approx(40.0)

    def test_square_open(self):
        # 3 sides of length 10 (no closing edge)
        p = contour_perimeter(_square(size=10), closed=False)
        assert p == pytest.approx(30.0)

    def test_two_points_closed(self):
        c = np.array([[0, 0], [10, 0]], dtype=np.float64)
        p = contour_perimeter(c, closed=True)
        assert p == pytest.approx(20.0)

    def test_non_negative(self):
        assert contour_perimeter(_triangle()) >= 0.0


# ─── contour_bbox ────────────────────────────────────────────────────────────

class TestContourBbox:
    def test_empty_returns_zeros(self):
        result = contour_bbox(np.empty((0, 2), dtype=np.float64))
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_square_bbox(self):
        x, y, w, h = contour_bbox(_square(size=10, offset=(5, 3)))
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(3.0)
        assert w == pytest.approx(10.0)
        assert h == pytest.approx(10.0)

    def test_returns_tuple_of_4(self):
        result = contour_bbox(_triangle())
        assert len(result) == 4

    def test_non_negative_width_height(self):
        _, _, w, h = contour_bbox(_triangle())
        assert w >= 0.0
        assert h >= 0.0


# ─── contour_centroid ────────────────────────────────────────────────────────

class TestContourCentroid:
    def test_empty_returns_origin(self):
        cx, cy = contour_centroid(np.empty((0, 2), dtype=np.float64))
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_square_centroid(self):
        # Square from (0,0) to (10,10) → centroid ≈ (5,5)
        cx, cy = contour_centroid(_square(size=10))
        assert cx == pytest.approx(5.0, abs=1.0)
        assert cy == pytest.approx(5.0, abs=1.0)

    def test_returns_two_floats(self):
        result = contour_centroid(_triangle())
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_offset_square(self):
        cx, cy = contour_centroid(_square(size=10, offset=(20, 30)))
        assert cx == pytest.approx(25.0, abs=1.5)
        assert cy == pytest.approx(35.0, abs=1.5)


# ─── contour_iou ─────────────────────────────────────────────────────────────

class TestContourIou:
    def test_less_than_3_points_returns_zero(self):
        c = np.array([[0, 0], [10, 0]], dtype=np.float64)
        assert contour_iou(c, _square()) == pytest.approx(0.0)

    def test_same_contour_returns_one(self):
        c = _square(size=20, offset=(5, 5))
        iou = contour_iou(c, c.copy(), canvas_size=(60, 60))
        assert iou == pytest.approx(1.0, abs=0.02)

    def test_no_overlap_returns_zero(self):
        c1 = _square(size=10, offset=(0, 0))
        c2 = _square(size=10, offset=(50, 50))
        iou = contour_iou(c1, c2, canvas_size=(80, 80))
        assert iou == pytest.approx(0.0)

    def test_partial_overlap_between_0_and_1(self):
        c1 = _square(size=20, offset=(0, 0))
        c2 = _square(size=20, offset=(10, 0))  # 50% overlap
        iou = contour_iou(c1, c2, canvas_size=(50, 50))
        assert 0.0 < iou < 1.0

    def test_result_in_unit_interval(self):
        c1 = _square(size=15, offset=(0, 0))
        c2 = _square(size=15, offset=(5, 5))
        iou = contour_iou(c1, c2, canvas_size=(40, 40))
        assert 0.0 <= iou <= 1.0

    def test_auto_canvas_size(self):
        c1 = _square(size=10, offset=(0, 0))
        c2 = _square(size=10, offset=(5, 0))
        iou = contour_iou(c1, c2)  # no explicit canvas_size
        assert 0.0 < iou < 1.0


# ─── align_contour_orientation ───────────────────────────────────────────────

class TestAlignContourOrientation:
    def test_less_than_3_points_unchanged(self):
        c = np.array([[0, 0], [1, 1]], dtype=np.float64)
        result = align_contour_orientation(c, clockwise=True)
        np.testing.assert_array_equal(result, c)

    def test_shape_preserved(self):
        c = _square()
        result = align_contour_orientation(c, clockwise=True)
        assert result.shape == c.shape

    def test_dtype_float64(self):
        result = align_contour_orientation(_square(), clockwise=True)
        assert result.dtype == np.float64

    def test_idempotent_when_already_correct_orientation(self):
        c = _square()
        r1 = align_contour_orientation(c, clockwise=True)
        r2 = align_contour_orientation(r1, clockwise=True)
        np.testing.assert_array_equal(r1, r2)

    def test_cw_and_ccw_are_reverses(self):
        c = _square()
        cw = align_contour_orientation(c, clockwise=True)
        ccw = align_contour_orientation(c, clockwise=False)
        # One should be the reverse of the other
        assert (np.array_equal(cw, ccw[::-1]) or np.array_equal(cw, ccw))


# ─── contours_to_mask ────────────────────────────────────────────────────────

class TestContoursToMask:
    def test_output_shape(self):
        c = _square(size=10, offset=(5, 5))
        mask = contours_to_mask(c, shape=(30, 30))
        assert mask.shape == (30, 30)

    def test_dtype_uint8(self):
        mask = contours_to_mask(_square(), shape=(20, 20))
        assert mask.dtype == np.uint8

    def test_filled_nonzero_inside(self):
        c = _square(size=10, offset=(2, 2))
        mask = contours_to_mask(c, shape=(20, 20), filled=True)
        # Interior pixel
        assert mask[7, 7] > 0

    def test_unfilled_zero_inside(self):
        c = _square(size=18, offset=(1, 1))
        mask = contours_to_mask(c, shape=(22, 22), filled=False)
        # Strictly interior pixel
        assert mask[10, 10] == 0

    def test_less_than_2_points_empty_mask(self):
        c = np.array([[5, 5]], dtype=np.float64)
        mask = contours_to_mask(c, shape=(20, 20))
        assert mask.sum() == 0

    def test_values_0_or_255(self):
        mask = contours_to_mask(_square(size=8, offset=(2, 2)), shape=(15, 15))
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})


# ─── mask_to_contour ─────────────────────────────────────────────────────────

class TestMaskToContour:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        result = mask_to_contour(mask)
        assert result.shape == (0, 2)

    def test_returns_2d_float64(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:20, 5:20] = 255
        result = mask_to_contour(mask)
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert result.dtype == np.float64

    def test_contour_from_square_mask_nonempty(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:25, 5:25] = 255
        result = mask_to_contour(mask)
        assert len(result) > 0

    def test_contour_points_within_mask_bounds(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        result = mask_to_contour(mask)
        assert result[:, 0].min() >= 9
        assert result[:, 0].max() <= 31
        assert result[:, 1].min() >= 9
        assert result[:, 1].max() <= 31

    def test_roundtrip_mask_contour_mask(self):
        """contours_to_mask → mask_to_contour → contours_to_mask: IoU ≈ 1."""
        c = _square(size=20, offset=(5, 5))
        shape = (40, 40)
        mask1 = contours_to_mask(c, shape=shape, filled=True)
        recovered = mask_to_contour(mask1)
        if len(recovered) >= 3:
            mask2 = contours_to_mask(recovered, shape=shape, filled=True)
            inter = np.count_nonzero((mask1 > 0) & (mask2 > 0))
            union = np.count_nonzero((mask1 > 0) | (mask2 > 0))
            iou = inter / union if union > 0 else 0.0
            assert iou > 0.85
