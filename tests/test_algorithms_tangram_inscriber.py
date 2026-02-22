"""Tests for puzzle_reconstruction/algorithms/tangram/inscriber.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.inscriber import (
    fit_tangram,
    extract_tangram_edge,
    _inset_polygon,
    _polygon_area,
)
from puzzle_reconstruction.models import TangramSignature, ShapeClass


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rect_contour(w=40, h=30) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def _triangle_contour() -> np.ndarray:
    return np.array([[0, 0], [20, 0], [10, 20]], dtype=np.float32)


def _circle_contour(r=20, n=40) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles) + r, r * np.sin(angles) + r],
                    axis=1).astype(np.float32)


# ─── TestFitTangram ───────────────────────────────────────────────────────────

class TestFitTangram:
    def test_returns_tangram_signature(self):
        result = fit_tangram(_rect_contour())
        assert isinstance(result, TangramSignature)

    def test_polygon_is_ndarray(self):
        result = fit_tangram(_rect_contour())
        assert isinstance(result.polygon, np.ndarray)

    def test_polygon_is_2d(self):
        result = fit_tangram(_rect_contour())
        assert result.polygon.ndim == 2
        assert result.polygon.shape[1] == 2

    def test_shape_class_is_enum(self):
        result = fit_tangram(_rect_contour())
        assert isinstance(result.shape_class, ShapeClass)

    def test_area_nonneg(self):
        result = fit_tangram(_rect_contour())
        assert result.area >= 0.0

    def test_scale_positive(self):
        result = fit_tangram(_rect_contour())
        assert result.scale > 0.0

    def test_centroid_shape(self):
        result = fit_tangram(_rect_contour())
        assert result.centroid.shape == (2,)

    def test_rectangle_classifies_as_rectangle(self):
        result = fit_tangram(_rect_contour())
        assert result.shape_class == ShapeClass.RECTANGLE

    def test_triangle_classifies_as_triangle(self):
        result = fit_tangram(_triangle_contour())
        assert result.shape_class == ShapeClass.TRIANGLE

    def test_circle_contour_runs_ok(self):
        result = fit_tangram(_circle_contour())
        assert isinstance(result, TangramSignature)

    def test_custom_epsilon_ratio(self):
        result = fit_tangram(_rect_contour(), epsilon_ratio=0.01)
        assert isinstance(result, TangramSignature)

    def test_custom_inset_ratio(self):
        result = fit_tangram(_rect_contour(), inset_ratio=0.1)
        assert isinstance(result, TangramSignature)

    def test_zero_inset_ratio(self):
        result = fit_tangram(_rect_contour(), inset_ratio=0.0)
        assert isinstance(result, TangramSignature)

    def test_polygon_vertices_at_least_3(self):
        result = fit_tangram(_rect_contour())
        assert result.polygon.shape[0] >= 3


# ─── TestExtractTangramEdge ───────────────────────────────────────────────────

class TestExtractTangramEdge:
    def _tangram(self):
        return fit_tangram(_rect_contour())

    def test_returns_ndarray(self):
        t = self._tangram()
        result = extract_tangram_edge(t, 0)
        assert isinstance(result, np.ndarray)

    def test_shape_n_by_2(self):
        t = self._tangram()
        result = extract_tangram_edge(t, 0, n_points=64)
        assert result.shape == (64, 2)

    def test_default_n_points(self):
        t = self._tangram()
        result = extract_tangram_edge(t, 0)
        assert result.shape == (128, 2)

    def test_wraps_around_edge_index(self):
        t = self._tangram()
        n = t.polygon.shape[0]
        result = extract_tangram_edge(t, n)  # Should wrap: index n == index 0
        result_0 = extract_tangram_edge(t, 0)
        np.testing.assert_allclose(result, result_0, atol=1e-10)

    def test_first_point_on_polygon_vertex(self):
        t = self._tangram()
        curve = extract_tangram_edge(t, 0, n_points=10)
        np.testing.assert_allclose(curve[0], t.polygon[0], atol=1e-6)

    def test_last_point_on_next_vertex(self):
        t = self._tangram()
        curve = extract_tangram_edge(t, 0, n_points=10)
        np.testing.assert_allclose(curve[-1], t.polygon[1], atol=1e-6)

    def test_all_edges_run_ok(self):
        t = self._tangram()
        n = t.polygon.shape[0]
        for i in range(n):
            c = extract_tangram_edge(t, i)
            assert c.shape[0] == 128


# ─── TestInsetPolygon ─────────────────────────────────────────────────────────

class TestInsetPolygon:
    def test_returns_ndarray(self):
        p = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
        result = _inset_polygon(p, 0.1)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        p = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
        result = _inset_polygon(p, 0.2)
        assert result.shape == p.shape

    def test_zero_ratio_unchanged(self):
        p = np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=np.float64)
        result = _inset_polygon(p, 0.0)
        np.testing.assert_allclose(result, p, atol=1e-10)

    def test_full_ratio_collapses_to_centroid(self):
        p = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
        result = _inset_polygon(p, 1.0)
        centroid = p.mean(axis=0)
        np.testing.assert_allclose(result, np.tile(centroid, (4, 1)), atol=1e-10)

    def test_inset_smaller_than_original(self):
        p = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        centroid = p.mean(axis=0)
        result = _inset_polygon(p, 0.3)
        # Resulting points should be closer to centroid
        orig_dists = np.linalg.norm(p - centroid, axis=1)
        new_dists = np.linalg.norm(result - centroid, axis=1)
        assert (new_dists < orig_dists + 1e-10).all()


# ─── TestPolygonArea ──────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_unit_square(self):
        p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        assert _polygon_area(p) == pytest.approx(1.0)

    def test_rectangle(self):
        p = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float64)
        assert _polygon_area(p) == pytest.approx(12.0)

    def test_nonneg(self):
        for n in range(3, 10):
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
            p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            assert _polygon_area(p) >= 0.0

    def test_unit_circle_approx(self):
        n = 1000
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        assert _polygon_area(p) == pytest.approx(math.pi, rel=0.01)

    def test_returns_float(self):
        p = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float64)
        assert isinstance(_polygon_area(p), float)
