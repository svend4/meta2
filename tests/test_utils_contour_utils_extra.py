"""Extra tests for puzzle_reconstruction/utils/contour_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(n=8) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t) * 10 + 32, np.sin(t) * 10 + 32])


def _rect() -> np.ndarray:
    """10×20 rectangle with known area=200, perimeter≈60."""
    return np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float64)


# ─── simplify_contour ─────────────────────────────────────────────────────────

class TestSimplifyContourExtra:
    def test_returns_ndarray(self):
        assert isinstance(simplify_contour(_rect()), np.ndarray)

    def test_output_2_columns(self):
        out = simplify_contour(_rect())
        assert out.ndim == 2 and out.shape[1] == 2

    def test_fewer_or_equal_points(self):
        c = _square(64)
        out = simplify_contour(c, epsilon=1.0)
        assert len(out) <= len(c)

    def test_dtype_float64(self):
        assert simplify_contour(_rect()).dtype == np.float64


# ─── interpolate_contour ──────────────────────────────────────────────────────

class TestInterpolateContourExtra:
    def test_returns_ndarray(self):
        assert isinstance(interpolate_contour(_rect()), np.ndarray)

    def test_output_shape(self):
        out = interpolate_contour(_rect(), n_points=32)
        assert out.shape == (32, 2)

    def test_n_points_lt_2_raises(self):
        with pytest.raises(ValueError):
            interpolate_contour(_rect(), n_points=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            interpolate_contour(np.zeros((0, 2)), n_points=8)

    def test_dtype_float64(self):
        assert interpolate_contour(_rect(), n_points=16).dtype == np.float64


# ─── contour_area ─────────────────────────────────────────────────────────────

class TestContourAreaExtra:
    def test_returns_float(self):
        assert isinstance(contour_area(_rect()), float)

    def test_rectangle_area(self):
        area = contour_area(_rect())
        assert abs(area - 200.0) < 1.0

    def test_nonneg(self):
        assert contour_area(_rect()) >= 0.0

    def test_small_contour(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        assert contour_area(pts) >= 0.0


# ─── contour_perimeter ────────────────────────────────────────────────────────

class TestContourPerimeterExtra:
    def test_returns_float(self):
        assert isinstance(contour_perimeter(_rect()), float)

    def test_rectangle_perimeter(self):
        peri = contour_perimeter(_rect(), closed=True)
        assert abs(peri - 60.0) < 1.0

    def test_nonneg(self):
        assert contour_perimeter(_rect()) >= 0.0


# ─── contour_bbox ─────────────────────────────────────────────────────────────

class TestContourBboxExtra:
    def test_returns_tuple_4(self):
        result = contour_bbox(_rect())
        assert isinstance(result, tuple) and len(result) == 4

    def test_rect_bbox(self):
        x, y, w, h = contour_bbox(_rect())
        assert abs(x - 0.0) < 1.0 and abs(y - 0.0) < 1.0
        assert abs(w - 20.0) < 1.0 and abs(h - 10.0) < 1.0

    def test_width_height_positive(self):
        x, y, w, h = contour_bbox(_square())
        assert w > 0 and h > 0


# ─── contour_centroid ─────────────────────────────────────────────────────────

class TestContourCentroidExtra:
    def test_returns_tuple_2(self):
        result = contour_centroid(_rect())
        assert isinstance(result, tuple) and len(result) == 2

    def test_rect_centroid(self):
        cx, cy = contour_centroid(_rect())
        assert abs(cx - 10.0) < 2.0 and abs(cy - 5.0) < 2.0

    def test_centroid_is_float(self):
        cx, cy = contour_centroid(_rect())
        assert isinstance(cx, float) and isinstance(cy, float)


# ─── contour_iou ──────────────────────────────────────────────────────────────

class TestContourIoUExtra:
    def test_returns_float(self):
        c = _square()
        result = contour_iou(c, c, canvas_size=(64, 64))
        assert isinstance(result, float)

    def test_identical_is_one(self):
        c = _square()
        iou = contour_iou(c, c, canvas_size=(64, 64))
        assert iou == pytest.approx(1.0, abs=0.05)

    def test_in_range(self):
        c1 = _square()
        c2 = _square() + np.array([40, 40])
        iou = contour_iou(c1, c2, canvas_size=(128, 128))
        assert 0.0 <= iou <= 1.0


# ─── align_contour_orientation ────────────────────────────────────────────────

class TestAlignContourOrientationExtra:
    def test_returns_ndarray(self):
        assert isinstance(align_contour_orientation(_rect()), np.ndarray)

    def test_same_shape(self):
        c = _rect()
        out = align_contour_orientation(c)
        assert out.shape == c.shape

    def test_dtype_float64(self):
        assert align_contour_orientation(_rect()).dtype == np.float64


# ─── contours_to_mask ─────────────────────────────────────────────────────────

class TestContoursToMaskExtra:
    def test_returns_ndarray(self):
        c = _square()
        out = contours_to_mask(c, shape=(64, 64))
        assert isinstance(out, np.ndarray)

    def test_shape_correct(self):
        c = _square()
        out = contours_to_mask(c, shape=(64, 64))
        assert out.shape == (64, 64)

    def test_dtype_uint8(self):
        c = _square()
        out = contours_to_mask(c, shape=(64, 64))
        assert out.dtype == np.uint8

    def test_nonzero_pixels_inside(self):
        c = _square()
        out = contours_to_mask(c, shape=(64, 64))
        assert out.max() > 0


# ─── mask_to_contour ──────────────────────────────────────────────────────────

class TestMaskToContourExtra:
    def _circle_mask(self):
        import cv2
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 15, 255, -1)
        return mask

    def test_returns_ndarray(self):
        mask = self._circle_mask()
        out = mask_to_contour(mask)
        assert isinstance(out, np.ndarray)

    def test_shape_n_by_2(self):
        mask = self._circle_mask()
        out = mask_to_contour(mask)
        assert out.ndim == 2 and out.shape[1] == 2

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        out = mask_to_contour(mask)
        assert out.ndim == 2
