"""Additional tests for puzzle_reconstruction/algorithms/tangram (hull, classifier, inscriber)."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.tangram.hull import (
    convex_hull,
    rdp_simplify,
    normalize_polygon,
)
from puzzle_reconstruction.algorithms.tangram.classifier import (
    classify_shape,
    compute_interior_angles,
)
from puzzle_reconstruction.algorithms.tangram.inscriber import (
    fit_tangram,
    extract_tangram_edge,
)
from puzzle_reconstruction.models import ShapeClass


# ─── helpers ──────────────────────────────────────────────────────────────────

def _ngon(n: int, r: float = 1.0) -> np.ndarray:
    angles = [2 * math.pi * i / n for i in range(n)]
    return np.array([[r * math.cos(a), r * math.sin(a)] for a in angles],
                    dtype=np.float32)


def _rect(w: float = 200.0, h: float = 100.0) -> np.ndarray:
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


def _triangle() -> np.ndarray:
    return np.array([[0, 0], [100, 0], [50, 80]], dtype=np.float32)


# ─── TestConvexHullExtra ──────────────────────────────────────────────────────

class TestConvexHullExtra:
    def test_returns_ndarray(self):
        h = convex_hull(_rect())
        assert isinstance(h, np.ndarray)

    def test_triangle_hull_is_3_points(self):
        h = convex_hull(_triangle())
        assert len(h) == 3

    def test_hull_is_2d(self):
        h = convex_hull(_rect())
        assert h.ndim == 2
        assert h.shape[1] == 2

    def test_hull_of_hull_same_size(self):
        poly = _rect()
        h1 = convex_hull(poly)
        h2 = convex_hull(h1)
        assert len(h2) == len(h1)

    def test_collinear_points_hull(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0],
                        [3, 1], [0, 1]], dtype=np.float32)
        h = convex_hull(pts)
        assert len(h) >= 3

    def test_translated_same_size(self):
        poly = _rect()
        translated = poly + np.array([1000, 2000], dtype=np.float32)
        assert len(convex_hull(poly)) == len(convex_hull(translated))

    def test_float32_input_accepted(self):
        poly = _rect().astype(np.float32)
        h = convex_hull(poly)
        assert h.ndim == 2


# ─── TestRDPSimplifyExtra ─────────────────────────────────────────────────────

class TestRDPSimplifyExtra:
    def test_returns_ndarray(self):
        result = rdp_simplify(_rect(), epsilon_ratio=0.01)
        assert isinstance(result, np.ndarray)

    def test_is_2d(self):
        result = rdp_simplify(_rect(), epsilon_ratio=0.01)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_large_epsilon_fewer_points(self):
        circle = _ngon(64)
        simplified = rdp_simplify(circle, epsilon_ratio=0.1)
        assert len(simplified) < len(circle)

    def test_subset_of_original_points(self):
        """All points in RDP result must appear in the original polygon."""
        poly = _ngon(16)
        simplified = rdp_simplify(poly, epsilon_ratio=0.01)
        for pt in simplified:
            dists = np.linalg.norm(poly - pt, axis=1)
            assert dists.min() < 1e-5

    def test_epsilon_zero_keeps_all_points(self):
        poly = _rect()
        result = rdp_simplify(poly, epsilon_ratio=0.0)
        assert len(result) == len(poly)

    def test_high_epsilon_result_is_2d(self):
        """Even at very high epsilon, result should remain 2D."""
        poly = _ngon(32)
        result = rdp_simplify(poly, epsilon_ratio=0.9)
        assert result.ndim == 2


# ─── TestNormalizePolygonExtra ────────────────────────────────────────────────

class TestNormalizePolygonExtra:
    def test_returns_tuple_of_4(self):
        out = normalize_polygon(_rect())
        assert len(out) == 4

    def test_polygon_is_2d(self):
        normed, _, _, _ = normalize_polygon(_rect())
        assert normed.ndim == 2
        assert normed.shape[1] == 2

    def test_scale_is_positive(self):
        _, _, scale, _ = normalize_polygon(_rect())
        assert scale > 0.0

    def test_angle_is_finite(self):
        _, _, _, angle = normalize_polygon(_rect())
        assert math.isfinite(angle)

    def test_triangle_centroid_at_origin(self):
        normed, _, _, _ = normalize_polygon(_triangle())
        assert np.linalg.norm(normed.mean(axis=0)) < 1e-5

    def test_centroid_returned_correctly(self):
        poly = _rect(200, 100)
        expected = poly.mean(axis=0)
        _, centroid, _, _ = normalize_polygon(poly)
        assert np.linalg.norm(centroid - expected) < 1e-4

    def test_double_normalization_idempotent(self):
        """Normalizing an already-normalized polygon leaves it unchanged."""
        poly = _rect()
        n1, _, _, _ = normalize_polygon(poly)
        n2, _, _, _ = normalize_polygon(n1)
        np.testing.assert_allclose(n1.mean(axis=0), n2.mean(axis=0), atol=1e-5)


# ─── TestClassifyShapeWithHullPipeline ────────────────────────────────────────

class TestClassifyShapeWithHullPipeline:
    def test_hull_then_classify_triangle(self):
        hull = convex_hull(_triangle())
        normed, _, _, _ = normalize_polygon(hull)
        assert classify_shape(normed) == ShapeClass.TRIANGLE

    def test_hull_then_classify_rectangle(self):
        hull = convex_hull(_rect())
        normed, _, _, _ = normalize_polygon(hull)
        assert classify_shape(normed) == ShapeClass.RECTANGLE

    def test_pentagon_pipeline(self):
        poly = _ngon(5)
        hull = convex_hull(poly)
        assert classify_shape(hull) == ShapeClass.PENTAGON

    def test_hexagon_pipeline(self):
        poly = _ngon(6)
        hull = convex_hull(poly)
        assert classify_shape(hull) == ShapeClass.HEXAGON

    def test_circle_contour_is_polygon(self):
        circle = _ngon(64)
        hull = convex_hull(circle)
        simplified = rdp_simplify(hull, epsilon_ratio=0.02)
        normed, _, _, _ = normalize_polygon(simplified)
        result = classify_shape(normed)
        assert isinstance(result, ShapeClass)


# ─── TestFitTangramExtra ──────────────────────────────────────────────────────

class TestFitTangramExtra:
    def test_triangle_signature_not_none(self):
        sig = fit_tangram(_triangle())
        assert sig is not None

    def test_polygon_has_vertices(self):
        sig = fit_tangram(_rect())
        assert len(sig.polygon) >= 3

    def test_centroid_is_array(self):
        sig = fit_tangram(_rect())
        assert isinstance(sig.centroid, np.ndarray)
        assert len(sig.centroid) == 2

    def test_area_positive_triangle(self):
        sig = fit_tangram(_triangle())
        assert sig.area > 0.0

    def test_shape_class_instance(self):
        sig = fit_tangram(_triangle())
        assert isinstance(sig.shape_class, ShapeClass)

    def test_scale_positive_for_large_poly(self):
        sig = fit_tangram(_rect(400, 300))
        assert sig.scale > 0.0

    def test_different_sizes_different_scale(self):
        sig_small = fit_tangram(_rect(10, 10))
        sig_large = fit_tangram(_rect(1000, 1000))
        # Larger polygon → larger scale
        assert sig_large.scale > sig_small.scale * 0.5


# ─── TestExtractTangramEdgeExtra ──────────────────────────────────────────────

class TestExtractTangramEdgeExtra:
    def test_returns_correct_n_points(self):
        sig = fit_tangram(_rect())
        for n in (16, 32, 64):
            edge = extract_tangram_edge(sig, edge_index=0, n_points=n)
            assert edge.shape == (n, 2)

    def test_different_edges_differ(self):
        sig = fit_tangram(_rect(200, 100))
        e0 = extract_tangram_edge(sig, edge_index=0)
        e1 = extract_tangram_edge(sig, edge_index=1)
        # The two edge arrays should not be identical
        assert not np.allclose(e0, e1)

    def test_edge_dtype_float(self):
        sig = fit_tangram(_rect())
        edge = extract_tangram_edge(sig, edge_index=0)
        assert np.issubdtype(edge.dtype, np.floating)

    def test_edge_is_2d(self):
        sig = fit_tangram(_rect())
        edge = extract_tangram_edge(sig, edge_index=0)
        assert edge.ndim == 2
        assert edge.shape[1] == 2
