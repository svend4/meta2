"""Tests for puzzle_reconstruction.algorithms.shape_context."""
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

def _circle_pts(n=20, r=50.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _line_pts(n=10):
    xs = np.linspace(0.0, 100.0, n)
    return np.stack([xs, np.zeros(n)], axis=1)


# ─── TestShapeContextResult ───────────────────────────────────────────────────

class TestShapeContextResult:
    def test_basic_creation(self):
        desc = np.zeros((5, 60), dtype=np.float64)
        pts = np.zeros((5, 2))
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=10.0,
                                 n_bins_r=5, n_bins_theta=12)
        assert scr.n_bins_r == 5
        assert scr.n_bins_theta == 12
        assert scr.mean_dist == pytest.approx(10.0)

    def test_descriptor_dim_property(self):
        desc = np.zeros((5, 60))
        pts = np.zeros((5, 2))
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=1.0,
                                 n_bins_r=5, n_bins_theta=12)
        assert scr.descriptor_dim == 60

    def test_repr_contains_n(self):
        desc = np.zeros((8, 60))
        pts = np.zeros((8, 2))
        scr = ShapeContextResult(descriptors=desc, points=pts, mean_dist=5.0,
                                 n_bins_r=5, n_bins_theta=12)
        assert "N=8" in repr(scr)


# ─── TestComputeShapeContext ──────────────────────────────────────────────────

class TestComputeShapeContext:
    def test_returns_shape_context_result(self):
        pts = _circle_pts(20)
        result = compute_shape_context(pts)
        assert isinstance(result, ShapeContextResult)

    def test_descriptor_shape(self):
        pts = _circle_pts(20)
        result = compute_shape_context(pts, n_bins_r=4, n_bins_theta=8)
        n = len(pts)
        assert result.descriptors.shape == (n, 4 * 8)

    def test_single_point_degenerate(self):
        pts = np.array([[0.0, 0.0]])
        result = compute_shape_context(pts)
        assert result.descriptors.shape[0] == 1
        assert result.mean_dist == pytest.approx(0.0)

    def test_two_points_no_crash(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        result = compute_shape_context(pts)
        assert result.descriptors.shape[0] == 2

    def test_not_2d_raises(self):
        pts = np.zeros((5, 3))
        with pytest.raises(ValueError):
            compute_shape_context(pts)

    def test_normalized_descriptors_sum_to_1(self):
        pts = _circle_pts(20)
        result = compute_shape_context(pts, normalize=True)
        for desc in result.descriptors:
            if desc.sum() > 0:
                assert desc.sum() == pytest.approx(1.0, abs=1e-6)

    def test_unnormalized_mode(self):
        pts = _circle_pts(20)
        result = compute_shape_context(pts, normalize=False)
        # Counts should be non-negative integers
        assert (result.descriptors >= 0).all()

    def test_mean_dist_positive_for_spread_points(self):
        pts = _circle_pts(20, r=50.0)
        result = compute_shape_context(pts)
        assert result.mean_dist > 0.0


# ─── TestLogPolarHistogram ────────────────────────────────────────────────────

class TestLogPolarHistogram:
    def _make_bins(self, n_r=5, n_theta=12):
        r_bins = np.logspace(-1, 2, n_r + 1)
        theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
        return r_bins, theta_bins

    def test_output_length(self):
        r_bins, theta_bins = self._make_bins(5, 12)
        dists = np.array([1.0, 5.0, 10.0])
        angles = np.array([0.0, np.pi / 4, -np.pi / 2])
        h = log_polar_histogram(dists, angles, r_bins, theta_bins, 5, 12)
        assert h.shape == (60,)

    def test_empty_inputs(self):
        r_bins, theta_bins = self._make_bins(4, 8)
        h = log_polar_histogram(np.array([]), np.array([]),
                                r_bins, theta_bins, 4, 8)
        assert h.shape == (32,)
        assert h.sum() == 0

    def test_counts_nonnegative(self):
        r_bins, theta_bins = self._make_bins(5, 12)
        rng = np.random.default_rng(0)
        dists = rng.uniform(0.5, 50.0, 30)
        angles = rng.uniform(-np.pi, np.pi, 30)
        h = log_polar_histogram(dists, angles, r_bins, theta_bins, 5, 12)
        assert (h >= 0).all()


# ─── TestShapeContextDistance ─────────────────────────────────────────────────

class TestShapeContextDistance:
    def test_identical_descriptors_zero_distance(self):
        sc = np.array([0.2, 0.3, 0.5])
        assert shape_context_distance(sc, sc) == pytest.approx(0.0, abs=1e-8)

    def test_shape_mismatch_raises(self):
        a = np.zeros(10)
        b = np.zeros(12)
        with pytest.raises(ValueError):
            shape_context_distance(a, b)

    def test_returns_float(self):
        a = np.array([0.3, 0.3, 0.4])
        b = np.array([0.4, 0.4, 0.2])
        d = shape_context_distance(a, b)
        assert isinstance(d, float)

    def test_distance_nonnegative(self):
        rng = np.random.default_rng(0)
        a = np.abs(rng.random(20))
        b = np.abs(rng.random(20))
        assert shape_context_distance(a, b) >= 0.0

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        a = np.abs(rng.random(15))
        b = np.abs(rng.random(15))
        d_ab = shape_context_distance(a, b)
        d_ba = shape_context_distance(b, a)
        assert d_ab == pytest.approx(d_ba, abs=1e-10)

    def test_distance_upper_bound(self):
        """Chi2 distance of L1-normalized histograms is in [0, 1]."""
        sc1 = np.array([1.0, 0.0, 0.0])
        sc2 = np.array([0.0, 1.0, 0.0])
        d = shape_context_distance(sc1, sc2)
        assert d <= 1.0 + 1e-9


# ─── TestNormalizeShapeContext ────────────────────────────────────────────────

class TestNormalizeShapeContext:
    def test_sums_to_1(self):
        sc = np.array([2.0, 3.0, 5.0])
        result = normalize_shape_context(sc)
        assert result.sum() == pytest.approx(1.0)

    def test_zero_vector_unchanged(self):
        sc = np.zeros(5)
        result = normalize_shape_context(sc)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_already_normalized_unchanged(self):
        sc = np.array([0.2, 0.3, 0.5])
        result = normalize_shape_context(sc)
        np.testing.assert_array_almost_equal(result, sc)


# ─── TestMatchShapeContexts ───────────────────────────────────────────────────

class TestMatchShapeContexts:
    def _make_result(self, n=10, n_bins_r=3, n_bins_theta=6):
        pts = _circle_pts(n)
        return compute_shape_context(pts, n_bins_r=n_bins_r,
                                     n_bins_theta=n_bins_theta)

    def test_returns_cost_and_correspondence(self):
        r_a = self._make_result(8)
        r_b = self._make_result(8)
        cost, corr = match_shape_contexts(r_a, r_b)
        assert isinstance(cost, float)
        assert corr.ndim == 2 and corr.shape[1] == 2

    def test_cost_nonnegative(self):
        r_a = self._make_result(10)
        r_b = self._make_result(10)
        cost, _ = match_shape_contexts(r_a, r_b)
        assert cost >= 0.0

    def test_identical_descriptors_zero_cost(self):
        r = self._make_result(10)
        cost, _ = match_shape_contexts(r, r)
        assert cost == pytest.approx(0.0, abs=1e-8)

    def test_correspondence_indices_in_range(self):
        n_a, n_b = 8, 6
        r_a = self._make_result(n_a)
        r_b = self._make_result(n_b)
        _, corr = match_shape_contexts(r_a, r_b)
        assert (corr[:, 0] < n_a).all()
        assert (corr[:, 1] < n_b).all()

    def test_correspondence_length(self):
        n_a, n_b = 8, 10
        r_a = self._make_result(n_a)
        r_b = self._make_result(n_b)
        _, corr = match_shape_contexts(r_a, r_b)
        assert len(corr) == min(n_a, n_b)


# ─── TestContourSimilarity ────────────────────────────────────────────────────

class TestContourSimilarity:
    def test_identical_contours_near_1(self):
        pts = _circle_pts(20)
        sim = contour_similarity(pts, pts.copy(), n_sample=20)
        assert sim == pytest.approx(1.0, abs=0.05)

    def test_returns_float_in_0_1(self):
        a = _circle_pts(20)
        b = _line_pts(20)
        sim = contour_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_different_shapes_lower_similarity(self):
        same = _circle_pts(20)
        diff = _line_pts(20) * 200
        sim_same = contour_similarity(same, same.copy())
        sim_diff = contour_similarity(same, diff)
        assert sim_same >= sim_diff

    def test_accepts_n1x2_shape(self):
        pts_3d = _circle_pts(10).reshape(-1, 1, 2)
        sim = contour_similarity(pts_3d, pts_3d.copy())
        assert 0.0 <= sim <= 1.0
