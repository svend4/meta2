"""
Property-based tests for fractal and tangram descriptor algorithms using Hypothesis.

Verifies mathematical invariants with randomly generated inputs:
- Box-counting: FD ∈ [1.0, 2.0] for any contour, determinism
- CSS: scale invariance, feature vector unit norm, similarity range
- Tangram hull: convex hull vertices are a subset of input points
- Tangram normalize_polygon: centroid shifts to origin
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st, assume
from hypothesis.extra.numpy import arrays

from puzzle_reconstruction.algorithms.fractal.box_counting import box_counting_fd
from puzzle_reconstruction.algorithms.fractal.css import (
    curvature_scale_space,
    css_to_feature_vector,
    css_similarity,
)
from puzzle_reconstruction.algorithms.tangram.hull import (
    convex_hull,
    normalize_polygon,
)


# ── Helper contour generators ─────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square(n: int = 64) -> np.ndarray:
    side = max(n // 4, 2)
    top    = np.column_stack([np.linspace(0, 1, side), np.ones(side)])
    right  = np.column_stack([np.ones(side), np.linspace(1, 0, side)])
    bottom = np.column_stack([np.linspace(1, 0, side), np.zeros(side)])
    left   = np.column_stack([np.zeros(side), np.linspace(0, 1, side)])
    return np.vstack([top, right, bottom, left])


def _random_contour(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random 2-D contour with n points."""
    if rng is None:
        rng = np.random.default_rng(42)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 2.0, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


# ── Box-counting FD properties ────────────────────────────────────────────────

class TestBoxCountingFDProperties:
    """Property tests for box_counting_fd: output range and determinism."""

    @given(st.integers(min_value=10, max_value=500))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_fd_always_in_range_circle(self, n_pts: int):
        """FD ∈ [1.0, 2.0] for circle contours of any size."""
        contour = _circle(n_pts)
        fd = box_counting_fd(contour)
        assert 1.0 <= fd <= 2.0, f"FD out of range for n={n_pts}: {fd}"

    @given(st.integers(min_value=8, max_value=300))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_fd_always_in_range_square(self, n_pts: int):
        """FD ∈ [1.0, 2.0] for square contours of any size."""
        n = max(n_pts, 8)
        contour = _square(n)
        fd = box_counting_fd(contour)
        assert 1.0 <= fd <= 2.0, f"FD out of range for n={n}: {fd}"

    @given(st.integers(min_value=16, max_value=256))
    @settings(max_examples=30, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_fd_deterministic(self, n_pts: int):
        """box_counting_fd is deterministic for any contour size."""
        contour = _random_contour(n_pts)
        fd1 = box_counting_fd(contour)
        fd2 = box_counting_fd(contour)
        assert fd1 == fd2, f"Non-deterministic FD: {fd1} != {fd2}"

    @given(st.integers(min_value=4, max_value=12))
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_fd_range_across_n_scales(self, n_scales: int):
        """FD stays in [1.0, 2.0] for any valid number of scales."""
        contour = _circle(64)
        fd = box_counting_fd(contour, n_scales=n_scales)
        assert 1.0 <= fd <= 2.0, f"FD out of range with n_scales={n_scales}: {fd}"

    def test_fd_straight_line_near_1(self):
        """A straight line has FD close to 1.0."""
        line = np.column_stack([np.linspace(0, 1, 128), np.zeros(128)])
        fd = box_counting_fd(line)
        assert fd < 1.5, f"Straight line FD should be near 1, got {fd:.3f}"

    def test_fd_complex_zigzag_above_1(self):
        """A complex zigzag has FD > 1.0."""
        x = np.linspace(0, 10, 200)
        y = np.abs(np.sin(x * 5)) + 0.1 * np.sin(x * 20)
        contour = np.column_stack([x, y])
        fd = box_counting_fd(contour)
        assert fd >= 1.0, f"Zigzag FD should be >= 1.0, got {fd}"


# ── CSS feature vector properties ────────────────────────────────────────────

class TestCSSFeatureVectorProperties:
    """Property tests for css_to_feature_vector: unit norm, non-negativity."""

    @given(st.integers(min_value=5, max_value=10))
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.differing_executors])
    def test_feature_vector_unit_norm(self, n_sigmas: int):
        """||css_to_feature_vector(css(C))|| ≈ 1 for any sigma count."""
        contour = _circle(128)
        css = curvature_scale_space(contour, n_sigmas=n_sigmas)
        vec = css_to_feature_vector(css)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-9 or norm == 0.0, \
            f"Feature vector not unit norm (n_sigmas={n_sigmas}): ||v||={norm}"

    @given(st.integers(min_value=16, max_value=64))
    @settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.differing_executors])
    def test_feature_vector_non_negative(self, n_bins: int):
        """All entries of css feature vector are >= 0 (histogram-based)."""
        contour = _square(128)
        css = curvature_scale_space(contour)
        vec = css_to_feature_vector(css, n_bins=n_bins)
        assert np.all(vec >= 0), f"Feature vector has negative entries (n_bins={n_bins})"

    @given(st.floats(min_value=0.5, max_value=10.0))
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.differing_executors])
    def test_css_scale_invariant(self, scale: float):
        """css_similarity(vec(scale*C), vec(C)) > 0.9 for any positive scale."""
        assume(scale > 0.1)
        c = _circle(64, r=1.0)
        c_scaled = _circle(64, r=scale)
        vec_a = css_to_feature_vector(curvature_scale_space(c))
        vec_b = css_to_feature_vector(curvature_scale_space(c_scaled))
        sim = css_similarity(vec_a, vec_b)
        assert sim > 0.9, f"Scale invariance violated (scale={scale:.2f}): sim={sim:.3f}"


# ── CSS similarity properties ─────────────────────────────────────────────────

class TestCSSSimilarityProperties:
    """Property tests for css_similarity: range, symmetry, reflexivity."""

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.differing_executors])
    def test_similarity_range_0_1(self, n_pts: int):
        """css_similarity always returns a value in [0.0, 1.0]."""
        c1 = _circle(n_pts)
        c2 = _square(n_pts)
        v1 = css_to_feature_vector(curvature_scale_space(c1))
        v2 = css_to_feature_vector(curvature_scale_space(c2))
        sim = css_similarity(v1, v2)
        assert 0.0 <= sim <= 1.0, f"Similarity out of [0,1]: {sim}"

    @given(st.integers(min_value=32, max_value=128))
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.differing_executors])
    def test_similarity_symmetric(self, n_pts: int):
        """css_similarity(a, b) == css_similarity(b, a)."""
        c1 = _circle(n_pts)
        c2 = _square(n_pts)
        v1 = css_to_feature_vector(curvature_scale_space(c1))
        v2 = css_to_feature_vector(curvature_scale_space(c2))
        assert css_similarity(v1, v2) == pytest.approx(css_similarity(v2, v1), abs=1e-9), \
            "CSS similarity is not symmetric"

    @given(st.integers(min_value=32, max_value=200))
    @settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.differing_executors])
    def test_self_similarity_is_one(self, n_pts: int):
        """css_similarity(v, v) == 1.0 for any non-zero feature vector."""
        contour = _circle(n_pts)
        vec = css_to_feature_vector(curvature_scale_space(contour))
        if np.linalg.norm(vec) > 0:
            sim = css_similarity(vec, vec)
            assert sim == pytest.approx(1.0, abs=1e-9), \
                f"Self-similarity is not 1 for n={n_pts}: {sim}"


# ── Tangram convex hull properties ────────────────────────────────────────────

class TestConvexHullProperties:
    """Property tests for convex_hull: hull vertices are subset of input."""

    @given(st.integers(min_value=3, max_value=50))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_convex_hull_vertices_subset_of_input(self, n_pts: int):
        """All convex hull vertices must come from input point set."""
        rng = np.random.default_rng(n_pts)
        points = rng.uniform(0, 100, (n_pts, 2)).astype(np.float32)
        hull = convex_hull(points)
        # Each hull vertex should be approximately in the input set
        for hv in hull:
            dists = np.linalg.norm(points - hv, axis=1)
            assert dists.min() < 1e-3, \
                f"Hull vertex {hv} not found in input points (min dist={dists.min():.6f})"

    @given(st.integers(min_value=3, max_value=50))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_convex_hull_size_leq_input(self, n_pts: int):
        """Convex hull cannot have more vertices than input."""
        rng = np.random.default_rng(n_pts + 100)
        points = rng.uniform(0, 100, (n_pts, 2)).astype(np.float32)
        hull = convex_hull(points)
        assert len(hull) <= n_pts, \
            f"Hull has more vertices ({len(hull)}) than input ({n_pts})"

    @given(st.integers(min_value=3, max_value=50))
    @settings(max_examples=40, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_convex_hull_at_least_3_vertices(self, n_pts: int):
        """For non-collinear points, hull must have at least 3 vertices."""
        rng = np.random.default_rng(n_pts + 200)
        # Use points on circle to guarantee non-collinear spread
        angles = rng.uniform(0, 2 * np.pi, n_pts)
        x = 50 + 40 * np.cos(angles)
        y = 50 + 40 * np.sin(angles)
        points = np.column_stack([x, y]).astype(np.float32)
        hull = convex_hull(points)
        assert len(hull) >= 3, f"Hull has fewer than 3 vertices: {len(hull)}"


# ── Tangram normalize_polygon properties ──────────────────────────────────────

class TestNormalizePolygonProperties:
    """Property tests for normalize_polygon: centroid at origin, scale invariance."""

    @given(st.integers(min_value=4, max_value=100))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_centroid_near_zero_after_normalize(self, n_pts: int):
        """After normalize_polygon, centroid of normalized polygon ≈ (0, 0)."""
        rng = np.random.default_rng(n_pts)
        polygon = rng.uniform(-10, 10, (n_pts, 2)).astype(np.float64)
        normalized, centroid, scale, angle = normalize_polygon(polygon)
        new_centroid = normalized.mean(axis=0)
        assert np.allclose(new_centroid, 0.0, atol=1e-9), \
            f"Centroid not at origin after normalize: {new_centroid}"

    @given(st.floats(min_value=0.1, max_value=100.0))
    @settings(max_examples=40, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_normalize_returns_4_values(self, offset: float):
        """normalize_polygon always returns a 4-tuple."""
        polygon = _circle(32) + offset
        result = normalize_polygon(polygon.astype(np.float64))
        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"

    @given(st.integers(min_value=4, max_value=60))
    @settings(max_examples=40, deadline=5000, suppress_health_check=[HealthCheck.differing_executors])
    def test_normalized_polygon_same_size(self, n_pts: int):
        """normalize_polygon preserves number of vertices."""
        rng = np.random.default_rng(n_pts + 300)
        polygon = rng.uniform(-5, 5, (n_pts, 2)).astype(np.float64)
        normalized, _, _, _ = normalize_polygon(polygon)
        assert len(normalized) == n_pts, \
            f"Normalized polygon has {len(normalized)} pts, expected {n_pts}"

    def test_normalize_circle_centroid_at_origin(self):
        """Circle polygon normalizes to centroid at origin."""
        polygon = _circle(64).astype(np.float64)
        normalized, centroid, scale, angle = normalize_polygon(polygon)
        assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-9)

    def test_normalize_scale_positive(self):
        """Scale returned by normalize_polygon is always positive."""
        polygon = _square(64).astype(np.float64)
        _, _, scale, _ = normalize_polygon(polygon)
        assert scale > 0, f"Scale should be positive, got {scale}"
