"""Tests for puzzle_reconstruction.utils.contour_utils."""
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

np.random.seed(99)


# ── Helpers ────────────────────────────────────────────────────────────────────

def square(side=10.0, offset=(0, 0)):
    x, y = offset
    return np.array([
        [x, y], [x+side, y], [x+side, y+side], [x, y+side]
    ], dtype=np.float64)


def circle_contour(n=32, r=10.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([r*np.cos(t) + 50, r*np.sin(t) + 50], axis=1)


# ── simplify_contour ───────────────────────────────────────────────────────────

def test_simplify_returns_float64():
    c = circle_contour(64)
    s = simplify_contour(c, epsilon=2.0)
    assert s.dtype == np.float64


def test_simplify_shape():
    c = circle_contour(64)
    s = simplify_contour(c, epsilon=2.0)
    assert s.ndim == 2 and s.shape[1] == 2


def test_simplify_reduces_points():
    c = circle_contour(128)
    s = simplify_contour(c, epsilon=3.0)
    assert len(s) <= len(c)


def test_simplify_empty_contour():
    c = np.empty((0, 2), dtype=np.float64)
    s = simplify_contour(c, epsilon=1.0)
    assert s.shape == (0, 2)


def test_simplify_invalid_shape():
    with pytest.raises(ValueError):
        simplify_contour(np.ones((5, 3)), epsilon=1.0)


# ── interpolate_contour ───────────────────────────────────────────────────────

def test_interpolate_shape():
    c = square()
    result = interpolate_contour(c, n_points=50)
    assert result.shape == (50, 2)


def test_interpolate_dtype():
    c = square()
    result = interpolate_contour(c, n_points=20)
    assert result.dtype == np.float64


def test_interpolate_invalid_n():
    with pytest.raises(ValueError):
        interpolate_contour(square(), n_points=1)


def test_interpolate_empty():
    with pytest.raises(ValueError):
        interpolate_contour(np.empty((0, 2), dtype=np.float64), n_points=10)


def test_interpolate_degenerate_contour():
    c = np.ones((5, 2)) * 5.0  # all same point
    result = interpolate_contour(c, n_points=10)
    assert result.shape == (10, 2)
    assert np.allclose(result, 5.0)


# ── contour_area ──────────────────────────────────────────────────────────────

def test_contour_area_square():
    c = square(10.0)
    area = contour_area(c)
    assert abs(area - 100.0) < 1e-6


def test_contour_area_triangle():
    c = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float64)
    area = contour_area(c)
    assert abs(area - 50.0) < 1e-6


def test_contour_area_nonnegative():
    c = circle_contour(64)
    assert contour_area(c) >= 0.0


def test_contour_area_too_few_points():
    c = np.array([[0, 0], [1, 1]], dtype=np.float64)
    assert contour_area(c) == 0.0


# ── contour_perimeter ─────────────────────────────────────────────────────────

def test_contour_perimeter_square():
    c = square(10.0)
    p = contour_perimeter(c, closed=True)
    assert abs(p - 40.0) < 1e-6


def test_contour_perimeter_open():
    c = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float64)
    p_open = contour_perimeter(c, closed=False)
    p_closed = contour_perimeter(c, closed=True)
    assert p_closed > p_open


def test_contour_perimeter_single():
    c = np.array([[5, 5]], dtype=np.float64)
    assert contour_perimeter(c) == 0.0


# ── contour_bbox ──────────────────────────────────────────────────────────────

def test_contour_bbox_square():
    c = square(10.0, offset=(5, 3))
    x, y, w, h = contour_bbox(c)
    assert abs(x - 5.0) < 1e-6
    assert abs(y - 3.0) < 1e-6
    assert abs(w - 10.0) < 1e-6
    assert abs(h - 10.0) < 1e-6


def test_contour_bbox_empty():
    assert contour_bbox(np.empty((0, 2))) == (0.0, 0.0, 0.0, 0.0)


def test_contour_bbox_returns_tuple():
    c = square(5.0)
    result = contour_bbox(c)
    assert isinstance(result, tuple) and len(result) == 4


# ── contour_centroid ──────────────────────────────────────────────────────────

def test_contour_centroid_square():
    c = square(10.0, offset=(0, 0))
    cx, cy = contour_centroid(c)
    assert abs(cx - 5.0) < 2.0   # centroid near center
    assert abs(cy - 5.0) < 2.0


def test_contour_centroid_empty():
    assert contour_centroid(np.empty((0, 2))) == (0.0, 0.0)


def test_contour_centroid_returns_floats():
    c = square()
    cx, cy = contour_centroid(c)
    assert isinstance(cx, float) and isinstance(cy, float)


# ── contour_iou ───────────────────────────────────────────────────────────────

def test_contour_iou_identical():
    c = square(10.0, offset=(20, 20))
    iou = contour_iou(c, c)
    assert iou > 0.9


def test_contour_iou_non_overlapping():
    c1 = square(5.0, offset=(0, 0))
    c2 = square(5.0, offset=(100, 100))
    iou = contour_iou(c1, c2, canvas_size=(200, 200))
    assert iou == 0.0


def test_contour_iou_in_01():
    c1 = square(10.0, offset=(0, 0))
    c2 = square(10.0, offset=(5, 5))
    iou = contour_iou(c1, c2, canvas_size=(30, 30))
    assert 0.0 <= iou <= 1.0


def test_contour_iou_too_few_points():
    c = np.array([[0, 0], [1, 1]], dtype=np.float64)
    assert contour_iou(c, c) == 0.0


# ── align_contour_orientation ─────────────────────────────────────────────────

def test_align_orientation_returns_same_shape():
    c = circle_contour(32)
    result = align_contour_orientation(c, clockwise=True)
    assert result.shape == c.shape


def test_align_orientation_does_not_mutate():
    c = square()
    original = c.copy()
    align_contour_orientation(c, clockwise=True)
    assert np.allclose(c, original)


def test_align_orientation_too_few():
    c = np.array([[0, 0], [1, 0]], dtype=np.float64)
    result = align_contour_orientation(c)
    assert result.shape == c.shape


# ── contours_to_mask ──────────────────────────────────────────────────────────

def test_contours_to_mask_shape():
    c = square(10.0, offset=(5, 5))
    mask = contours_to_mask(c, shape=(50, 50))
    assert mask.shape == (50, 50)


def test_contours_to_mask_dtype():
    c = square(10.0, offset=(5, 5))
    mask = contours_to_mask(c, shape=(50, 50))
    assert mask.dtype == np.uint8


def test_contours_to_mask_filled():
    c = square(20.0, offset=(5, 5))
    mask = contours_to_mask(c, shape=(50, 50), filled=True)
    assert mask.max() == 255
    assert mask.sum() > 0


def test_contours_to_mask_empty_contour():
    c = np.empty((0, 2), dtype=np.float64)
    mask = contours_to_mask(c, shape=(20, 20))
    assert mask.sum() == 0


# ── mask_to_contour ───────────────────────────────────────────────────────────

def test_mask_to_contour_basic():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    contour = mask_to_contour(mask)
    assert contour.shape[1] == 2
    assert len(contour) > 0


def test_mask_to_contour_empty_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    contour = mask_to_contour(mask)
    assert contour.shape == (0, 2)


def test_mask_to_contour_dtype():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    contour = mask_to_contour(mask)
    assert contour.dtype == np.float64
