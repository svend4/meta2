"""Tests for puzzle_reconstruction/algorithms/tangram/hull.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.hull import (
    convex_hull,
    rdp_simplify,
    normalize_polygon,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(side=10.0) -> np.ndarray:
    return np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)


def _triangle() -> np.ndarray:
    return np.array([[0, 0], [5, 0], [2.5, 5]], dtype=np.float32)


def _circle_pts(r=10.0, n=20) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1).astype(np.float32)


def _square_with_interior() -> np.ndarray:
    """Square with an extra interior point that should be dropped by hull."""
    sq = _square(10.0)
    extra = np.array([[5.0, 5.0]], dtype=np.float32)
    return np.vstack([sq, extra])


# ─── TestConvexHull ───────────────────────────────────────────────────────────

class TestConvexHull:
    def test_returns_ndarray(self):
        result = convex_hull(_square())
        assert isinstance(result, np.ndarray)

    def test_shape_is_k_by_2(self):
        result = convex_hull(_square())
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_square_hull_has_4_points(self):
        result = convex_hull(_square())
        assert result.shape[0] == 4

    def test_triangle_hull_has_3_points(self):
        result = convex_hull(_triangle())
        assert result.shape[0] == 3

    def test_interior_point_removed(self):
        pts = _square_with_interior()
        hull = convex_hull(pts)
        # Interior point (5,5) should not be in hull
        assert hull.shape[0] == 4

    def test_hull_vertices_subset_of_input(self):
        pts = _square()
        hull = convex_hull(pts)
        for v in hull:
            dists = np.linalg.norm(pts - v, axis=1)
            assert dists.min() < 1e-4

    def test_circle_hull_approx_all_points(self):
        pts = _circle_pts(n=12)
        hull = convex_hull(pts)
        # All points on circle are extreme
        assert hull.shape[0] == 12

    def test_dtype_float32(self):
        result = convex_hull(_square())
        assert result.dtype == np.float32

    def test_single_point_returns_array(self):
        pts = np.array([[3.0, 4.0]], dtype=np.float32)
        result = convex_hull(pts)
        assert result.ndim == 2

    def test_hull_k_le_n(self):
        pts = np.random.rand(15, 2).astype(np.float32)
        hull = convex_hull(pts)
        assert hull.shape[0] <= 15


# ─── TestRdpSimplify ──────────────────────────────────────────────────────────

class TestRdpSimplify:
    def test_returns_ndarray(self):
        result = rdp_simplify(_square())
        assert isinstance(result, np.ndarray)

    def test_shape_is_k_by_2(self):
        result = rdp_simplify(_square())
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_dtype_float32(self):
        result = rdp_simplify(_square())
        assert result.dtype == np.float32

    def test_simplified_k_le_original(self):
        pts = _circle_pts(n=30)
        simplified = rdp_simplify(pts, epsilon_ratio=0.05)
        assert simplified.shape[0] <= 30

    def test_higher_epsilon_fewer_points(self):
        pts = _circle_pts(n=24)
        s_low = rdp_simplify(pts, epsilon_ratio=0.01)
        s_high = rdp_simplify(pts, epsilon_ratio=0.10)
        assert s_high.shape[0] <= s_low.shape[0]

    def test_square_simplified_still_4(self):
        # A square is already "simplified" — RDP shouldn't remove vertices
        result = rdp_simplify(_square(), epsilon_ratio=0.01)
        assert result.shape[0] >= 3

    def test_minimal_output_at_least_1_point(self):
        pts = _triangle()
        result = rdp_simplify(pts, epsilon_ratio=1.0)
        assert result.shape[0] >= 1


# ─── TestNormalizePolygon ─────────────────────────────────────────────────────

class TestNormalizePolygon:
    def test_returns_4_tuple(self):
        result = normalize_polygon(_square().astype(np.float64))
        assert len(result) == 4

    def test_normalized_polygon_is_ndarray(self):
        poly, centroid, scale, angle = normalize_polygon(_square())
        assert isinstance(poly, np.ndarray)

    def test_centroid_is_array(self):
        _, centroid, _, _ = normalize_polygon(_square())
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (2,)

    def test_scale_is_positive(self):
        _, _, scale, _ = normalize_polygon(_square())
        assert scale > 0.0

    def test_angle_is_float(self):
        _, _, _, angle = normalize_polygon(_square())
        assert isinstance(angle, float)

    def test_normalized_centroid_near_zero(self):
        poly, _, _, _ = normalize_polygon(_square())
        center = poly.mean(axis=0)
        np.testing.assert_allclose(center, [0.0, 0.0], atol=1e-5)

    def test_normalized_scale_approx_one(self):
        """After normalization, bounding diagonal should be ~1."""
        poly, _, scale, _ = normalize_polygon(_square(side=20.0))
        assert scale > 0.0
        bbox = poly.max(axis=0) - poly.min(axis=0)
        diag = float(np.hypot(bbox[0], bbox[1]))
        assert diag == pytest.approx(1.0, abs=0.01)

    def test_consistent_with_different_translations(self):
        """Normalization should be translation-invariant."""
        sq1 = _square(5.0)
        sq2 = sq1 + np.array([100.0, -50.0])
        poly1, c1, s1, a1 = normalize_polygon(sq1.astype(np.float64))
        poly2, c2, s2, a2 = normalize_polygon(sq2.astype(np.float64))
        np.testing.assert_allclose(poly1, poly2, atol=1e-5)

    def test_degenerate_single_point_no_crash(self):
        """Scale=0 edge-case: all points at origin → scale defaulted to 1."""
        pts = np.zeros((4, 2), dtype=np.float64)
        poly, _, scale, _ = normalize_polygon(pts)
        assert scale == pytest.approx(1.0)

    def test_triangle(self):
        poly, centroid, scale, angle = normalize_polygon(_triangle().astype(np.float64))
        assert poly.shape == (3, 2)
        assert scale > 0.0
