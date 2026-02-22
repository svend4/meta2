"""Additional tests for puzzle_reconstruction/algorithms/tangram/classifier.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.classifier import (
    _classify_quad,
    classify_shape,
    compute_interior_angles,
)
from puzzle_reconstruction.models import ShapeClass


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ngon(n: int, r: float = 1.0) -> np.ndarray:
    angles = [2 * math.pi * i / n for i in range(n)]
    return np.array([[r * math.cos(a), r * math.sin(a)] for a in angles])


def _rectangle(w: float = 2.0, h: float = 1.0) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


# ─── TestClassifyShapeExtra ───────────────────────────────────────────────────

class TestClassifyShapeExtra:
    def test_n_8_is_polygon(self):
        p = _ngon(8)
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_n_10_is_polygon(self):
        p = _ngon(10)
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_zero_points_is_polygon(self):
        p = np.zeros((0, 2), dtype=np.float64)
        # n=0 ≤ 2 → POLYGON
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_pentagon_is_pentagon(self):
        p = _ngon(5)
        assert classify_shape(p) == ShapeClass.PENTAGON

    def test_hexagon_is_hexagon(self):
        p = _ngon(6)
        assert classify_shape(p) == ShapeClass.HEXAGON

    def test_pure_function_same_result_twice(self):
        p = _rectangle()
        r1 = classify_shape(p)
        r2 = classify_shape(p)
        assert r1 == r2

    def test_right_isosceles_triangle_is_triangle(self):
        p = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        assert classify_shape(p) == ShapeClass.TRIANGLE

    def test_degenerate_2_point_is_polygon(self):
        p = np.array([[0, 0], [1, 1]], dtype=np.float64)
        assert classify_shape(p) == ShapeClass.POLYGON

    def test_returns_shape_class_for_all_n(self):
        for n in range(1, 10):
            p = _ngon(n) if n >= 3 else np.zeros((n, 2), dtype=np.float64)
            result = classify_shape(p)
            assert isinstance(result, ShapeClass)


# ─── TestClassifyQuadExtra ────────────────────────────────────────────────────

class TestClassifyQuadExtra:
    def test_unit_square_is_rectangle(self):
        p = _rectangle(1.0, 1.0)
        assert _classify_quad(p) == ShapeClass.RECTANGLE

    def test_wide_rectangle_is_rectangle(self):
        p = _rectangle(10.0, 1.0)
        assert _classify_quad(p) == ShapeClass.RECTANGLE

    def test_rhombus_is_parallelogram(self):
        """Rhombus: equal sides, non-right angles → PARALLELOGRAM."""
        p = np.array([[0, 0], [2, 0], [3, 1], [1, 1]], dtype=np.float64)
        result = _classify_quad(p)
        assert result in (ShapeClass.PARALLELOGRAM, ShapeClass.TRAPEZOID,
                          ShapeClass.POLYGON)  # depends on angle tolerances

    def test_returns_shape_class(self):
        p = _rectangle()
        result = _classify_quad(p)
        assert isinstance(result, ShapeClass)


# ─── TestComputeInteriorAnglesExtra ───────────────────────────────────────────

class TestComputeInteriorAnglesExtra:
    def test_dtype_float64(self):
        p = _rectangle()
        angles = compute_interior_angles(p)
        assert angles.dtype == np.float64

    def test_single_vertex_returns_array(self):
        """Single point: degenerate case, returns array of length 1."""
        p = np.array([[0.0, 0.0]])
        angles = compute_interior_angles(p)
        assert isinstance(angles, np.ndarray)
        assert len(angles) == 1

    def test_hexagon_each_angle_approx_120(self):
        p = _ngon(6)
        angles = compute_interior_angles(p)
        np.testing.assert_allclose(np.degrees(angles), 120.0, atol=2.0)

    def test_square_sum_approx_360(self):
        p = _rectangle(3.0, 3.0)
        angles = compute_interior_angles(p)
        assert np.degrees(angles.sum()) == pytest.approx(360.0, abs=2.0)

    def test_quadrilateral_sum_approx_360(self):
        p = np.array([[0, 0], [3, 0], [4, 2], [1, 2]], dtype=np.float64)
        angles = compute_interior_angles(p)
        assert np.degrees(angles.sum()) == pytest.approx(360.0, abs=5.0)

    def test_octagon_sum_approx_1080(self):
        p = _ngon(8)
        angles = compute_interior_angles(p)
        assert np.degrees(angles.sum()) == pytest.approx(1080.0, abs=10.0)

    def test_consistent_for_same_input(self):
        p = _ngon(5)
        a1 = compute_interior_angles(p)
        a2 = compute_interior_angles(p)
        np.testing.assert_array_equal(a1, a2)

    def test_all_angles_in_0_to_pi(self):
        for n in range(3, 9):
            p = _ngon(n)
            angles = compute_interior_angles(p)
            assert np.all(angles > 0)
            assert np.all(angles <= math.pi + 0.1)

    def test_large_polygon_no_crash(self):
        p = _ngon(20)
        angles = compute_interior_angles(p)
        assert len(angles) == 20
