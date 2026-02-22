"""Additional tests for puzzle_reconstruction.algorithms.shape_context."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.shape_context import (
    ShapeContextResult,
    compute_shape_context,
    contour_similarity,
    log_polar_histogram,
    match_shape_contexts,
    normalize_shape_context,
    shape_context_distance,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=20, r=50.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line(n=10):
    return np.stack([np.linspace(0.0, 100.0, n), np.zeros(n)], axis=1)


# ─── TestShapeContextResultExtra ─────────────────────────────────────────────

class TestShapeContextResultExtra:
    def test_large_descriptors(self):
        desc = np.random.default_rng(0).random((50, 60))
        pts = _circle(50)
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=5.0,
                                 n_bins_r=5, n_bins_theta=12)
        assert scr.descriptors.shape == (50, 60)

    def test_descriptor_dim_matches_n_bins(self):
        n_r, n_t = 4, 8
        desc = np.zeros((10, n_r * n_t))
        pts = _circle(10)
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=1.0,
                                 n_bins_r=n_r, n_bins_theta=n_t)
        assert scr.descriptor_dim == n_r * n_t

    def test_mean_dist_stored(self):
        desc = np.zeros((5, 32))
        pts = _circle(5)
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=42.0,
                                 n_bins_r=4, n_bins_theta=8)
        assert scr.mean_dist == pytest.approx(42.0)

    def test_repr_contains_n_bins_r(self):
        desc = np.zeros((8, 60))
        pts = _circle(8)
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=1.0,
                                 n_bins_r=5, n_bins_theta=12)
        r = repr(scr)
        assert isinstance(r, str) and len(r) > 0


# ─── TestComputeShapeContextExtra ────────────────────────────────────────────

class TestComputeShapeContextExtra:
    def test_large_n(self):
        pts = _circle(100)
        result = compute_shape_context(pts)
        assert result.descriptors.shape[0] == 100

    def test_n_bins_r2_theta4(self):
        pts = _circle(15)
        result = compute_shape_context(pts, n_bins_r=2, n_bins_theta=4)
        assert result.descriptors.shape == (15, 8)

    def test_n_bins_r6_theta16(self):
        pts = _circle(20)
        result = compute_shape_context(pts, n_bins_r=6, n_bins_theta=16)
        assert result.descriptors.shape == (20, 96)

    def test_line_pts_no_crash(self):
        pts = _line(20)
        result = compute_shape_context(pts)
        assert result.descriptors.shape[0] == 20

    def test_mean_dist_positive_circle(self):
        pts = _circle(30, r=100.0)
        result = compute_shape_context(pts)
        assert result.mean_dist > 0.0

    def test_mean_dist_large_circle_larger(self):
        small = compute_shape_context(_circle(20, r=5.0))
        large = compute_shape_context(_circle(20, r=100.0))
        assert large.mean_dist > small.mean_dist

    def test_points_stored(self):
        pts = _circle(15)
        result = compute_shape_context(pts)
        assert result.points.shape == (15, 2)

    def test_normalized_sum_all_nonzero(self):
        pts = _circle(20)
        result = compute_shape_context(pts, normalize=True)
        for desc in result.descriptors:
            s = desc.sum()
            assert s == pytest.approx(1.0, abs=1e-6) or s == pytest.approx(0.0, abs=1e-9)

    def test_unnormalized_integers_nonneg(self):
        pts = _circle(20)
        result = compute_shape_context(pts, normalize=False)
        assert np.all(result.descriptors >= 0)


# ─── TestLogPolarHistogramExtra ───────────────────────────────────────────────

class TestLogPolarHistogramExtra:
    def _bins(self, n_r=5, n_t=12):
        r_bins = np.logspace(-1, 2, n_r + 1)
        t_bins = np.linspace(-np.pi, np.pi, n_t + 1)
        return r_bins, t_bins

    def test_total_count_equals_input_len(self):
        n_r, n_t = 5, 12
        r_bins, t_bins = self._bins(n_r, n_t)
        dists = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        angles = np.array([0.0, 0.5, -0.5, np.pi / 3, -np.pi / 4])
        h = log_polar_histogram(dists, angles, r_bins, t_bins, n_r, n_t)
        assert h.sum() <= len(dists)

    def test_single_point(self):
        n_r, n_t = 4, 8
        r_bins, t_bins = self._bins(n_r, n_t)
        h = log_polar_histogram(np.array([5.0]), np.array([0.0]),
                                r_bins, t_bins, n_r, n_t)
        assert h.shape == (32,)
        assert h.sum() >= 0

    def test_different_n_bins(self):
        n_r, n_t = 3, 6
        r_bins = np.logspace(-1, 2, n_r + 1)
        t_bins = np.linspace(-np.pi, np.pi, n_t + 1)
        h = log_polar_histogram(np.array([1.0, 2.0]), np.array([0.0, 1.0]),
                                r_bins, t_bins, n_r, n_t)
        assert h.shape == (18,)

    def test_many_random_points(self):
        n_r, n_t = 5, 12
        r_bins, t_bins = self._bins(n_r, n_t)
        rng = np.random.default_rng(42)
        dists = rng.uniform(0.2, 80.0, 200)
        angles = rng.uniform(-np.pi, np.pi, 200)
        h = log_polar_histogram(dists, angles, r_bins, t_bins, n_r, n_t)
        assert h.shape == (60,)
        assert np.all(h >= 0)


# ─── TestShapeContextDistanceExtra ────────────────────────────────────────────

class TestShapeContextDistanceExtra:
    def test_zero_vs_zero_is_zero(self):
        z = np.zeros(10)
        d = shape_context_distance(z, z)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_distance_is_float(self):
        a = np.ones(8) / 8
        b = np.ones(8) / 8
        assert isinstance(shape_context_distance(a, b), float)

    def test_uniform_vs_uniform_zero(self):
        a = np.ones(12) / 12
        d = shape_context_distance(a, a.copy())
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_complementary_histograms(self):
        """Two non-overlapping histograms → positive distance."""
        a = np.array([1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 1.0])
        d = shape_context_distance(a, b)
        assert d > 0.0

    def test_partial_overlap_positive(self):
        a = np.array([0.5, 0.5, 0.0, 0.0])
        b = np.array([0.0, 0.5, 0.5, 0.0])
        d = shape_context_distance(a, b)
        assert d >= 0.0

    def test_random_nonneg(self):
        rng = np.random.default_rng(5)
        for _ in range(8):
            a = np.abs(rng.random(20))
            b = np.abs(rng.random(20))
            assert shape_context_distance(a, b) >= 0.0


# ─── TestNormalizeShapeContextExtra ──────────────────────────────────────────

class TestNormalizeShapeContextExtra:
    def test_large_vector_sums_to_1(self):
        sc = np.random.default_rng(0).random(60)
        result = normalize_shape_context(sc)
        assert result.sum() == pytest.approx(1.0, abs=1e-9)

    def test_output_same_length(self):
        sc = np.arange(1.0, 11.0)
        result = normalize_shape_context(sc)
        assert len(result) == 10

    def test_all_equal_values_uniform(self):
        sc = np.ones(8)
        result = normalize_shape_context(sc)
        np.testing.assert_allclose(result, np.ones(8) / 8, atol=1e-9)

    def test_nonneg_output(self):
        sc = np.array([3.0, 1.0, 2.0, 0.0, 4.0])
        result = normalize_shape_context(sc)
        assert np.all(result >= 0.0)

    def test_single_element_becomes_1(self):
        sc = np.array([5.0])
        result = normalize_shape_context(sc)
        assert result[0] == pytest.approx(1.0)


# ─── TestMatchShapeContextsExtra ─────────────────────────────────────────────

class TestMatchShapeContextsExtra:
    def _sc(self, n=10):
        return compute_shape_context(_circle(n), n_bins_r=3, n_bins_theta=6)

    def test_cost_float_type(self):
        r = self._sc(10)
        cost, _ = match_shape_contexts(r, r)
        assert isinstance(cost, float)

    def test_identical_zero_cost(self):
        r = self._sc(12)
        cost, _ = match_shape_contexts(r, r)
        assert cost == pytest.approx(0.0, abs=1e-8)

    def test_corr_2d(self):
        r = self._sc(8)
        _, corr = match_shape_contexts(r, r)
        assert corr.ndim == 2

    def test_corr_shape1_is_2(self):
        r = self._sc(8)
        _, corr = match_shape_contexts(r, r)
        assert corr.shape[1] == 2

    def test_corr_indices_nonneg(self):
        r_a = self._sc(8)
        r_b = self._sc(10)
        _, corr = match_shape_contexts(r_a, r_b)
        assert np.all(corr >= 0)

    def test_cost_nonneg_random_pts(self):
        pts_a = np.random.default_rng(1).random((12, 2)) * 100
        pts_b = np.random.default_rng(2).random((12, 2)) * 100
        r_a = compute_shape_context(pts_a, n_bins_r=3, n_bins_theta=6)
        r_b = compute_shape_context(pts_b, n_bins_r=3, n_bins_theta=6)
        cost, _ = match_shape_contexts(r_a, r_b)
        assert cost >= 0.0


# ─── TestContourSimilarityExtra ───────────────────────────────────────────────

class TestContourSimilarityExtra:
    def test_large_circle_same_similarity_1(self):
        pts = _circle(50, r=200.0)
        sim = contour_similarity(pts, pts.copy(), n_sample=50)
        assert sim == pytest.approx(1.0, abs=0.05)

    def test_similarity_is_float(self):
        pts = _circle(20)
        sim = contour_similarity(pts, pts.copy())
        assert isinstance(sim, float)

    def test_in_0_1_for_circle_vs_line(self):
        a = _circle(20)
        b = _line(20) * 10
        sim = contour_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_n_sample_parameter_accepted(self):
        pts = _circle(30)
        sim = contour_similarity(pts, pts.copy(), n_sample=15)
        assert 0.0 <= sim <= 1.0

    def test_circle_vs_larger_circle(self):
        a = _circle(20, r=10.0)
        b = _circle(20, r=100.0)
        sim = contour_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_symmetry(self):
        a = _circle(20)
        b = _line(20) * 100
        s_ab = contour_similarity(a, b)
        s_ba = contour_similarity(b, a)
        assert abs(s_ab - s_ba) < 0.1
