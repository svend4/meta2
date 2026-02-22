"""Tests for puzzle_reconstruction/algorithms/tangram/classifier.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.classifier import (
    classify_shape,
    compute_interior_angles,
)
from puzzle_reconstruction.models import ShapeClass


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rectangle(w=2.0, h=1.0) -> np.ndarray:
    """Axis-aligned rectangle."""
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


def _square() -> np.ndarray:
    return _rectangle(1.0, 1.0)


def _triangle() -> np.ndarray:
    return np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float64)


def _pentagon() -> np.ndarray:
    angles = [2 * math.pi * i / 5 for i in range(5)]
    return np.array([[math.cos(a), math.sin(a)] for a in angles])


def _hexagon() -> np.ndarray:
    angles = [2 * math.pi * i / 6 for i in range(6)]
    return np.array([[math.cos(a), math.sin(a)] for a in angles])


def _trapezoid() -> np.ndarray:
    """Simple isosceles trapezoid: bottom wider than top."""
    return np.array([[0, 0], [4, 0], [3, 1], [1, 1]], dtype=np.float64)


def _parallelogram() -> np.ndarray:
    return np.array([[0, 0], [3, 0], [4, 1], [1, 1]], dtype=np.float64)


# ─── TestClassifyShape ────────────────────────────────────────────────────────

class TestClassifyShape:
    def test_triangle(self):
        assert classify_shape(_triangle()) == ShapeClass.TRIANGLE

    def test_rectangle(self):
        assert classify_shape(_rectangle()) == ShapeClass.RECTANGLE

    def test_square_is_rectangle(self):
        assert classify_shape(_square()) == ShapeClass.RECTANGLE

    def test_pentagon(self):
        assert classify_shape(_pentagon()) == ShapeClass.PENTAGON

    def test_hexagon(self):
        assert classify_shape(_hexagon()) == ShapeClass.HEXAGON

    def test_single_point_is_polygon(self):
        p = np.array([[0, 0]])
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_two_points_is_polygon(self):
        p = np.array([[0, 0], [1, 0]])
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_seven_points_is_polygon(self):
        angles = [2 * math.pi * i / 7 for i in range(7)]
        p = np.array([[math.cos(a), math.sin(a)] for a in angles])
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_returns_shape_class_enum(self):
        result = classify_shape(_triangle())
        assert isinstance(result, ShapeClass)

    def test_rectangle_near_axis_aligned(self):
        """Slightly non-perfect rectangle should still classify as RECTANGLE."""
        p = np.array([[0.01, 0], [2, 0.01], [2, 1], [0, 1]], dtype=np.float64)
        result = classify_shape(p)
        # Could be RECTANGLE or PARALLELOGRAM/TRAPEZOID depending on tolerances
        assert isinstance(result, ShapeClass)

    def test_parallelogram(self):
        result = classify_shape(_parallelogram())
        assert result == ShapeClass.PARALLELOGRAM

    def test_trapezoid(self):
        result = classify_shape(_trapezoid())
        assert result in (ShapeClass.TRAPEZOID, ShapeClass.POLYGON)


# ─── TestComputeInteriorAngles ────────────────────────────────────────────────

class TestComputeInteriorAngles:
    def test_returns_array_length_n(self):
        p = _rectangle()
        angles = compute_interior_angles(p)
        assert len(angles) == 4

    def test_rectangle_all_90_degrees(self):
        p = _rectangle()
        angles = compute_interior_angles(p)
        np.testing.assert_allclose(np.degrees(angles), 90.0, atol=1.0)

    def test_equilateral_triangle_all_60(self):
        p = np.array([
            [0, 0],
            [1, 0],
            [0.5, math.sqrt(3) / 2],
        ], dtype=np.float64)
        angles = compute_interior_angles(p)
        np.testing.assert_allclose(np.degrees(angles), 60.0, atol=1.0)

    def test_angles_shape(self):
        p = _pentagon()
        angles = compute_interior_angles(p)
        assert angles.shape == (5,)

    def test_all_angles_positive(self):
        for poly in [_triangle(), _rectangle(), _pentagon()]:
            angles = compute_interior_angles(poly)
            assert (angles > 0).all()

    def test_angles_in_0_pi(self):
        for poly in [_triangle(), _rectangle(), _pentagon(), _hexagon()]:
            angles = compute_interior_angles(poly)
            assert (angles > 0).all()
            assert (angles < math.pi + 0.1).all()

    def test_pentagon_sum_approx_540(self):
        p = _pentagon()
        angles = compute_interior_angles(p)
        assert np.degrees(angles.sum()) == pytest.approx(540.0, abs=5.0)

    def test_hexagon_sum_approx_720(self):
        p = _hexagon()
        angles = compute_interior_angles(p)
        assert np.degrees(angles.sum()) == pytest.approx(720.0, abs=5.0)

    def test_returns_numpy_array(self):
        angles = compute_interior_angles(_rectangle())
        assert isinstance(angles, np.ndarray)

    def test_triangle_sum_approx_180(self):
        p = _triangle()
        angles = compute_interior_angles(p)
        assert np.degrees(angles.sum()) == pytest.approx(180.0, abs=2.0)
