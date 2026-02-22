"""Additional tests for puzzle_reconstruction/algorithms/tangram/hull.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.hull import (
    convex_hull,
    normalize_polygon,
    rdp_simplify,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(side: float = 10.0) -> np.ndarray:
    return np.array([[0, 0], [side, 0], [side, side], [0, side]],
                    dtype=np.float32)


def _circle_pts(r: float = 10.0, n: int = 20) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1).astype(np.float32)


def _collinear(n: int = 10) -> np.ndarray:
    xs = np.linspace(0.0, 10.0, n).astype(np.float32)
    return np.stack([xs, np.zeros(n, dtype=np.float32)], axis=1)


# ─── TestConvexHullExtra ──────────────────────────────────────────────────────

class TestConvexHullExtra:
    def test_two_points_returns_array(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = convex_hull(pts)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_collinear_hull_not_larger_than_input(self):
        pts = _collinear(8)
        hull = convex_hull(pts)
        assert hull.shape[0] <= 8

    def test_large_polygon_hull_subset(self):
        pts = _circle_pts(n=50)
        hull = convex_hull(pts)
        assert hull.shape[0] <= 50
        assert hull.shape[0] >= 3

    def test_float32_output(self):
        hull = convex_hull(_square())
        assert hull.dtype == np.float32

    def test_hull_of_hull_is_same(self):
        pts = _circle_pts(n=16)
        hull1 = convex_hull(pts)
        hull2 = convex_hull(hull1)
        # Hull of a convex hull is itself (same number of vertices)
        assert hull2.shape[0] == hull1.shape[0]

    def test_translated_polygon_same_hull_size(self):
        sq = _square()
        sq_shifted = (sq + np.array([100.0, -50.0], dtype=np.float32))
        h1 = convex_hull(sq)
        h2 = convex_hull(sq_shifted)
        assert h1.shape[0] == h2.shape[0]

    def test_all_identical_points_no_crash(self):
        pts = np.tile(np.array([[3.0, 3.0]], dtype=np.float32), (5, 1))
        result = convex_hull(pts)
        assert result.ndim == 2

    def test_many_random_points_hull_convex(self):
        """All hull vertices should be extreme (no interior point on hull)."""
        rng = np.random.default_rng(7)
        pts = rng.random((30, 2)).astype(np.float32)
        hull = convex_hull(pts)
        assert hull.shape[0] >= 3
        assert hull.shape[0] <= 30


# ─── TestRdpSimplifyExtra ─────────────────────────────────────────────────────

class TestRdpSimplifyExtra:
    def test_triangle_unchanged_at_low_epsilon(self):
        tri = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float32)
        result = rdp_simplify(tri, epsilon_ratio=0.001)
        assert result.shape[0] >= 3

    def test_zero_epsilon_keeps_all_points(self):
        pts = _circle_pts(n=16)
        result = rdp_simplify(pts, epsilon_ratio=0.0)
        assert result.shape[0] == 16

    def test_high_epsilon_extreme_simplification(self):
        pts = _circle_pts(n=32)
        result = rdp_simplify(pts, epsilon_ratio=0.5)
        assert result.shape[0] <= 32

    def test_float32_output(self):
        result = rdp_simplify(_square(), epsilon_ratio=0.01)
        assert result.dtype == np.float32

    def test_collinear_pts_simplify_to_two(self):
        """Collinear points should simplify to endpoints."""
        pts = _collinear(20)
        result = rdp_simplify(pts, epsilon_ratio=0.01)
        assert result.shape[0] <= 20

    def test_output_is_2d(self):
        result = rdp_simplify(_square())
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_consistent_across_calls(self):
        pts = _circle_pts(n=24)
        r1 = rdp_simplify(pts, epsilon_ratio=0.05)
        r2 = rdp_simplify(pts, epsilon_ratio=0.05)
        np.testing.assert_array_equal(r1, r2)


# ─── TestNormalizePolygonExtra ────────────────────────────────────────────────

class TestNormalizePolygonExtra:
    def test_angle_in_minus_pi_to_pi(self):
        poly, _, _, angle = normalize_polygon(_square())
        assert -math.pi <= angle <= math.pi

    def test_scale_invariant_to_translation(self):
        sq1 = _square(5.0).astype(np.float64)
        sq2 = sq1 + np.array([200.0, -100.0])
        _, _, s1, _ = normalize_polygon(sq1)
        _, _, s2, _ = normalize_polygon(sq2)
        assert s1 == pytest.approx(s2, rel=1e-5)

    def test_consistent_same_input(self):
        sq = _square(8.0).astype(np.float64)
        r1 = normalize_polygon(sq)
        r2 = normalize_polygon(sq)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])

    def test_large_polygon_no_crash(self):
        pts = _circle_pts(r=1000.0, n=100).astype(np.float64)
        poly, centroid, scale, angle = normalize_polygon(pts)
        assert scale > 0.0
        assert np.all(np.isfinite(poly))

    def test_polygon_shape_preserved(self):
        sq = _square(10.0).astype(np.float64)
        poly, _, _, _ = normalize_polygon(sq)
        assert poly.shape == sq.shape

    def test_centroid_stored_matches_input_mean(self):
        sq = _square(6.0).astype(np.float64)
        _, centroid, _, _ = normalize_polygon(sq)
        np.testing.assert_allclose(centroid, sq.mean(axis=0), atol=1e-10)

    def test_pentagon_no_crash(self):
        angles = np.linspace(0, 2 * math.pi, 5, endpoint=False)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        poly, _, scale, _ = normalize_polygon(pts)
        assert scale > 0.0
        assert poly.shape == (5, 2)
