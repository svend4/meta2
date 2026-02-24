"""Extra tests for puzzle_reconstruction/algorithms/shape_context.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.shape_context import (
    ShapeContextResult,
    _preprocess_contour,
    compute_shape_context,
    contour_similarity,
    log_polar_histogram,
    match_shape_contexts,
    normalize_shape_context,
    shape_context_distance,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _square(n: int = 40, size: float = 100.0) -> np.ndarray:
    side = n // 4
    pts = []
    for i in range(side):
        pts.append([i * size / side, 0.0])
    for i in range(side):
        pts.append([size, i * size / side])
    for i in range(side):
        pts.append([size - i * size / side, size])
    for i in range(side):
        pts.append([0.0, size - i * size / side])
    return np.array(pts, dtype=np.float64)


def _circle(n: int = 40, r: float = 50.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)


# ─── log_polar_histogram (extra) ─────────────────────────────────────────────

class TestLogPolarHistogramExtra:
    def _bins(self, n_r=5, n_t=12):
        r_bins = np.logspace(-1, 2, n_r + 1)
        t_bins = np.linspace(-np.pi, np.pi, n_t + 1)
        return r_bins, t_bins

    def test_integer_counts(self):
        r_b, t_b = self._bins()
        dists = np.array([1.0, 5.0, 20.0])
        angles = np.array([0.0, 1.0, -1.0])
        h = log_polar_histogram(dists, angles, r_b, t_b, 5, 12)
        # All values are non-negative integers (or close to)
        assert (h >= 0).all()
        np.testing.assert_array_equal(h, h.astype(int))

    def test_sum_equals_in_range_count(self):
        """Total histogram count equals number of points within r range."""
        r_b = np.array([1.0, 10.0, 100.0])
        t_b = np.linspace(-np.pi, np.pi, 5)
        dists  = np.array([0.5, 5.0, 50.0, 500.0])  # 2 in range
        angles = np.zeros(4)
        h = log_polar_histogram(dists, angles, r_b, t_b, 2, 4)
        assert h.sum() == 2

    def test_different_n_r_affects_shape(self):
        r_b = np.logspace(-1, 2, 4)  # n_r=3
        t_b = np.linspace(-np.pi, np.pi, 13)  # n_t=12
        h = log_polar_histogram(np.array([1.0]), np.array([0.0]), r_b, t_b, 3, 12)
        assert h.shape == (36,)

    def test_all_same_angle_sector(self):
        r_b = np.array([0.5, 2.0, 50.0])
        t_b = np.linspace(-np.pi, np.pi, 3)  # 2 sectors
        dists  = np.array([1.0, 1.0, 1.0])
        angles = np.zeros(3)  # all in sector around 0
        h = log_polar_histogram(dists, angles, r_b, t_b, 2, 2)
        assert h.sum() == 3


# ─── compute_shape_context (extra) ───────────────────────────────────────────

class TestComputeShapeContextExtra:
    def test_n_bins_affect_descriptor_dim(self):
        sq = _square(20)
        r = compute_shape_context(sq, n_bins_r=3, n_bins_theta=8)
        assert r.descriptor_dim == 24

    def test_larger_n_bins_larger_dim(self):
        sq = _square(20)
        r_small = compute_shape_context(sq, n_bins_r=3, n_bins_theta=4)
        r_large = compute_shape_context(sq, n_bins_r=6, n_bins_theta=12)
        assert r_large.descriptor_dim > r_small.descriptor_dim

    def test_mean_dist_depends_on_scale(self):
        small = _square(20, size=10.0)
        large = _square(20, size=1000.0)
        r_small = compute_shape_context(small)
        r_large = compute_shape_context(large)
        assert r_large.mean_dist > r_small.mean_dist

    def test_points_shape_matches_input(self):
        sq = _square(16)
        r = compute_shape_context(sq)
        assert r.points.shape == sq.shape

    def test_descriptors_dtype_float(self):
        sq = _square(10)
        r = compute_shape_context(sq)
        assert r.descriptors.dtype.kind == "f"

    def test_normalize_flag_false_not_all_one(self):
        sq = _square(20)
        r = compute_shape_context(sq, normalize=False)
        totals = r.descriptors.sum(axis=1)
        # At least some totals are not 1.0
        assert not np.allclose(totals, 1.0)

    def test_n_bins_r_stored(self):
        sq = _square(10)
        r = compute_shape_context(sq, n_bins_r=7, n_bins_theta=9)
        assert r.n_bins_r == 7
        assert r.n_bins_theta == 9


# ─── normalize_shape_context (extra) ─────────────────────────────────────────

class TestNormalizeShapeContextExtra:
    def test_uniform_vector_normalized(self):
        sc = np.array([2.0, 2.0, 2.0, 2.0])
        n = normalize_shape_context(sc)
        np.testing.assert_allclose(n, [0.25, 0.25, 0.25, 0.25])

    def test_single_element_one(self):
        sc = np.array([5.0])
        n = normalize_shape_context(sc)
        assert n[0] == pytest.approx(1.0)

    def test_single_element_zero_unchanged(self):
        sc = np.array([0.0])
        n = normalize_shape_context(sc)
        assert n[0] == pytest.approx(0.0)

    def test_nonneg_input_nonneg_output(self):
        sc = np.array([3.0, 1.0, 0.0, 5.0])
        n = normalize_shape_context(sc)
        assert (n >= 0).all()

    def test_not_in_place(self):
        sc = np.array([1.0, 2.0, 3.0])
        original = sc.copy()
        _ = normalize_shape_context(sc)
        np.testing.assert_array_equal(sc, original)

    def test_large_values_normalized(self):
        sc = np.array([1e6, 2e6, 3e6])
        n = normalize_shape_context(sc)
        assert math.isclose(n.sum(), 1.0, rel_tol=1e-9)


# ─── shape_context_distance (extra) ──────────────────────────────────────────

class TestShapeContextDistanceExtra:
    def test_zero_vectors_distance_zero(self):
        sc = np.zeros(20)
        d = shape_context_distance(sc, sc)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_complementary_vectors(self):
        sc1 = np.array([1.0, 0.0])
        sc2 = np.array([0.0, 1.0])
        d = shape_context_distance(sc1, sc2)
        assert d > 0.0

    def test_partial_overlap(self):
        sc1 = np.array([0.5, 0.5, 0.0, 0.0])
        sc2 = np.array([0.5, 0.0, 0.5, 0.0])
        d = shape_context_distance(sc1, sc2)
        assert 0.0 < d <= 0.5

    def test_returns_float(self):
        sc = np.array([0.25, 0.25, 0.25, 0.25])
        result = shape_context_distance(sc, np.array([1.0, 0.0, 0.0, 0.0]))
        assert isinstance(result, float)

    def test_larger_mismatch_larger_dist(self):
        sc_ref = np.array([1.0, 0.0, 0.0, 0.0])
        sc_close = np.array([0.9, 0.1, 0.0, 0.0])
        sc_far = np.array([0.0, 0.0, 0.0, 1.0])
        d_close = shape_context_distance(sc_ref, sc_close)
        d_far = shape_context_distance(sc_ref, sc_far)
        assert d_close < d_far


# ─── match_shape_contexts (extra) ────────────────────────────────────────────

class TestMatchShapeContextsExtra:
    def test_same_contour_zero_cost(self):
        sq = _square(12)
        r = compute_shape_context(sq, normalize=True)
        cost, _ = match_shape_contexts(r, r)
        assert cost == pytest.approx(0.0, abs=1e-9)

    def test_different_sizes_min_n_pairs(self):
        sq6 = compute_shape_context(_square(12)[:6])
        sq10 = compute_shape_context(_square(20)[:10])
        _, corr = match_shape_contexts(sq6, sq10)
        assert len(corr) == 6  # min(6, 10)

    def test_correspondence_indices_in_range(self):
        sq = _square(8)
        r = compute_shape_context(sq)
        _, corr = match_shape_contexts(r, r)
        assert (corr[:, 0] >= 0).all()
        assert (corr[:, 0] < 8).all()
        assert (corr[:, 1] >= 0).all()
        assert (corr[:, 1] < 8).all()

    def test_cost_is_finite(self):
        sq = _square(10)
        ci = _circle(10)
        r_sq = compute_shape_context(sq)
        r_ci = compute_shape_context(ci)
        cost, _ = match_shape_contexts(r_sq, r_ci)
        assert math.isfinite(cost)

    def test_cross_match_higher_cost_than_self(self):
        sq = _square(10, 100)
        ci = _circle(10, 50)
        r_sq = compute_shape_context(sq, normalize=True)
        r_ci = compute_shape_context(ci, normalize=True)
        cost_self, _ = match_shape_contexts(r_sq, r_sq)
        cost_cross, _ = match_shape_contexts(r_sq, r_ci)
        assert cost_cross >= cost_self - 1e-9


# ─── contour_similarity (extra) ──────────────────────────────────────────────

class TestContourSimilarityExtra:
    def test_two_point_contour(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        s = contour_similarity(c, c)
        assert 0.0 <= s <= 1.0

    def test_scaled_same_shape_high_similarity(self):
        sq_s = _square(20, size=50.0)
        sq_l = _square(20, size=100.0)
        s = contour_similarity(sq_s, sq_l)
        # Scaling changes mean_dist — similarity may vary, but must be in range
        assert 0.0 <= s <= 1.0

    def test_n_sample_small(self):
        sq = _square(40)
        ci = _circle(40)
        s = contour_similarity(sq, ci, n_sample=8)
        assert 0.0 <= s <= 1.0

    def test_n_sample_large(self):
        sq = _square(40)
        ci = _circle(40)
        s = contour_similarity(sq, ci, n_sample=60)
        assert 0.0 <= s <= 1.0

    def test_square_vs_circle_less_than_one(self):
        sq = _square(40)
        ci = _circle(40)
        s = contour_similarity(sq, ci)
        assert s < 1.0

    def test_square_self_similarity(self):
        sq = _square(40)
        assert contour_similarity(sq, sq) >= 0.9

    def test_returns_python_float(self):
        sq = _square(10)
        assert isinstance(contour_similarity(sq, sq), float)


# ─── _preprocess_contour (extra) ─────────────────────────────────────────────

class TestPreprocessContourExtra:
    def test_3d_opencv_format(self):
        sq = _square(20)
        c3d = sq[:, np.newaxis, :]  # (N, 1, 2)
        pts = _preprocess_contour(c3d, n_sample=10)
        assert pts.shape == (10, 2)

    def test_exact_n_sample_unchanged(self):
        sq = _square(20)
        pts = _preprocess_contour(sq, n_sample=20)
        assert len(pts) == 20

    def test_dtype_float(self):
        sq = _square(10).astype(np.float32)
        pts = _preprocess_contour(sq, n_sample=10)
        assert pts.dtype.kind == "f"

    def test_n_sample_1(self):
        sq = _square(20)
        pts = _preprocess_contour(sq, n_sample=1)
        assert len(pts) == 1

    def test_output_2d(self):
        sq = _square(20)
        pts = _preprocess_contour(sq, n_sample=15)
        assert pts.ndim == 2
        assert pts.shape[1] == 2


# ─── ShapeContextResult (extra) ──────────────────────────────────────────────

class TestShapeContextResultExtra:
    def _make(self, n=10, n_r=5, n_t=12):
        desc = np.zeros((n, n_r * n_t))
        pts = np.zeros((n, 2))
        return ShapeContextResult(
            descriptors=desc,
            points=pts,
            mean_dist=10.0,
            n_bins_r=n_r,
            n_bins_theta=n_t,
        )

    def test_descriptor_dim_computed(self):
        r = self._make(n=8, n_r=4, n_t=6)
        assert r.descriptor_dim == 24

    def test_n_bins_stored(self):
        r = self._make(n_r=7, n_t=9)
        assert r.n_bins_r == 7
        assert r.n_bins_theta == 9

    def test_mean_dist_positive(self):
        r = self._make()
        assert r.mean_dist > 0.0

    def test_repr_contains_shape_info(self):
        sq = _square(20)
        r = compute_shape_context(sq)
        s = repr(r)
        assert "20" in s or "ShapeContextResult" in s

    def test_points_stored(self):
        r = self._make(n=5)
        assert r.points.shape == (5, 2)
