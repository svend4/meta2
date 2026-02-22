"""Additional tests for puzzle_reconstruction/algorithms/tangram/inscriber.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.inscriber import (
    _inset_polygon,
    _polygon_area,
    extract_tangram_edge,
    fit_tangram,
)
from puzzle_reconstruction.models import ShapeClass, TangramSignature


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rect(w: float = 40.0, h: float = 30.0) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def _triangle(base: float = 20.0, height: float = 20.0) -> np.ndarray:
    return np.array([[0, 0], [base, 0], [base / 2, height]], dtype=np.float32)


def _circle_contour(r: float = 20.0, n: int = 40) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles) + r, r * np.sin(angles) + r],
                    axis=1).astype(np.float32)


# ─── TestFitTangramExtra ──────────────────────────────────────────────────────

class TestFitTangramExtra:
    def test_large_epsilon_still_valid(self):
        result = fit_tangram(_rect(), epsilon_ratio=0.5)
        assert isinstance(result, TangramSignature)
        # High epsilon can reduce polygon to as few as 1 vertex — just check type
        assert result.polygon.ndim == 2

    def test_large_inset_ratio_still_valid(self):
        result = fit_tangram(_rect(), inset_ratio=0.9)
        assert isinstance(result, TangramSignature)
        assert result.area >= 0.0

    def test_circle_contour_large_n(self):
        result = fit_tangram(_circle_contour(n=60))
        assert isinstance(result, TangramSignature)
        assert result.polygon.ndim == 2

    def test_angle_is_float(self):
        result = fit_tangram(_rect())
        assert isinstance(result.angle, float)

    def test_polygon_finite_values(self):
        result = fit_tangram(_rect())
        assert np.all(np.isfinite(result.polygon))

    def test_centroid_is_2d(self):
        result = fit_tangram(_rect())
        assert result.centroid.shape == (2,)

    def test_area_is_float(self):
        result = fit_tangram(_rect())
        assert isinstance(result.area, float)

    def test_different_sizes_consistent_shape_class(self):
        r1 = fit_tangram(_rect(40, 30))
        r2 = fit_tangram(_rect(80, 60))  # doubled size
        assert r1.shape_class == r2.shape_class

    def test_triangle_contour_area_positive(self):
        result = fit_tangram(_triangle())
        assert result.area > 0.0

    def test_returns_tangram_signature_type(self):
        result = fit_tangram(_rect())
        assert isinstance(result, TangramSignature)


# ─── TestExtractTangramEdgeExtra ──────────────────────────────────────────────

class TestExtractTangramEdgeExtra:
    def _t(self):
        return fit_tangram(_rect())

    def test_n_points_1_returns_1_point(self):
        t = self._t()
        curve = extract_tangram_edge(t, 0, n_points=1)
        assert curve.shape == (1, 2)

    def test_n_points_custom_sizes(self):
        t = self._t()
        for n in [10, 32, 100]:
            c = extract_tangram_edge(t, 0, n_points=n)
            assert c.shape == (n, 2)

    def test_different_edges_different_curves(self):
        t = self._t()
        n = t.polygon.shape[0]
        if n >= 2:
            c0 = extract_tangram_edge(t, 0, n_points=10)
            c1 = extract_tangram_edge(t, 1, n_points=10)
            assert not np.allclose(c0, c1)

    def test_edge_is_linear_interpolation(self):
        """Midpoint of extracted edge should be midpoint of the two vertices."""
        t = self._t()
        n_pts = 11  # odd → midpoint at index 5
        curve = extract_tangram_edge(t, 0, n_points=n_pts)
        p0 = t.polygon[0]
        p1 = t.polygon[1]
        midpoint_expected = (p0 + p1) / 2.0
        np.testing.assert_allclose(curve[5], midpoint_expected, atol=1e-5)

    def test_all_edges_produce_finite_curves(self):
        t = self._t()
        n = t.polygon.shape[0]
        for i in range(n):
            c = extract_tangram_edge(t, i, n_points=16)
            assert np.all(np.isfinite(c))

    def test_negative_edge_index_wraps(self):
        t = self._t()
        c_neg = extract_tangram_edge(t, -1, n_points=8)
        c_wrap = extract_tangram_edge(t, len(t.polygon) - 1, n_points=8)
        np.testing.assert_allclose(c_neg, c_wrap, atol=1e-10)


# ─── TestInsetPolygonExtra ────────────────────────────────────────────────────

class TestInsetPolygonExtra:
    def test_inset_reduces_distance_to_centroid(self):
        p = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
        centroid = p.mean(axis=0)
        inset = _inset_polygon(p, 0.2)
        orig_dists = np.linalg.norm(p - centroid, axis=1)
        inset_dists = np.linalg.norm(inset - centroid, axis=1)
        np.testing.assert_array_less(inset_dists, orig_dists + 1e-10)

    def test_half_inset_halfway_to_centroid(self):
        p = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
        centroid = p.mean(axis=0)
        inset = _inset_polygon(p, 0.5)
        expected = centroid + 0.5 * (p - centroid)
        np.testing.assert_allclose(inset, expected, atol=1e-10)

    def test_triangle_shape_preserved(self):
        p = np.array([[0, 0], [3, 0], [1.5, 3]], dtype=np.float64)
        inset = _inset_polygon(p, 0.1)
        assert inset.shape == p.shape

    def test_nonnegative_inset_ratio(self):
        p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        # No crash for ratio=0
        result = _inset_polygon(p, 0.0)
        np.testing.assert_allclose(result, p, atol=1e-10)


# ─── TestPolygonAreaExtra ─────────────────────────────────────────────────────

class TestPolygonAreaExtra:
    def test_triangle_area(self):
        """Area of right triangle with legs a, b = 0.5*a*b."""
        a, b = 3.0, 4.0
        p = np.array([[0, 0], [a, 0], [0, b]], dtype=np.float64)
        assert _polygon_area(p) == pytest.approx(0.5 * a * b)

    def test_area_scale_invariant_property(self):
        """Scaling polygon by k scales area by k^2."""
        p = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        k = 5.0
        p_scaled = p * k
        assert _polygon_area(p_scaled) == pytest.approx(_polygon_area(p) * k ** 2)

    def test_degenerate_triangle_zero_area(self):
        """Collinear points → area = 0."""
        p = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        assert _polygon_area(p) == pytest.approx(0.0, abs=1e-10)

    def test_pentagon_area_positive(self):
        angles = np.linspace(0, 2 * math.pi, 5, endpoint=False)
        p = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        assert _polygon_area(p) > 0.0

    def test_area_independent_of_translation(self):
        p = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float64)
        p_shifted = p + np.array([100.0, -50.0])
        assert _polygon_area(p) == pytest.approx(_polygon_area(p_shifted), rel=1e-9)
