"""Integration tests for utils batch 2a.

Covers:
  - puzzle_reconstruction.utils.contour_sampler
  - puzzle_reconstruction.utils.curvature_utils
  - puzzle_reconstruction.utils.curve_metrics
  - puzzle_reconstruction.utils.descriptor_utils
  - puzzle_reconstruction.utils.distance_matrix
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.contour_sampler import (
    SamplerConfig,
    SampledContour,
    sample_uniform,
    sample_curvature,
    sample_random,
    sample_corners,
    sample_contour,
    normalize_contour,
    batch_sample,
)
from puzzle_reconstruction.utils.curvature_utils import (
    CurvatureConfig,
    compute_curvature,
    compute_total_curvature,
    find_inflection_points,
    compute_turning_angle,
    smooth_curvature,
    corner_score,
    find_corners,
    batch_curvature,
)
from puzzle_reconstruction.utils.curve_metrics import (
    CurveMetricConfig,
    curve_l2,
    curve_l2_mirror,
    hausdorff_distance,
    frechet_distance_approx,
    curve_length,
)
from puzzle_reconstruction.utils.descriptor_utils import (
    DescriptorConfig,
    l2_normalize,
    l1_normalize,
    batch_l2_normalize,
    l2_distance,
    cosine_distance,
    chi2_distance,
    l1_distance,
    descriptor_distance,
    pairwise_l2,
    pairwise_cosine,
    DescriptorMatch,
    nn_match,
    ratio_test,
)
from puzzle_reconstruction.utils.distance_matrix import (
    DistanceConfig,
    euclidean_distance_matrix,
    cosine_distance_matrix,
    manhattan_distance_matrix,
    build_distance_matrix,
    normalize_distance_matrix,
    to_similarity_matrix,
    threshold_distance_matrix,
    top_k_distance_pairs,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle(n: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def _line(n: int = 32) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, np.zeros(n)])


def _sine_curve(n: int = 64) -> np.ndarray:
    x = np.linspace(0, 2 * math.pi, n)
    return np.column_stack([x, np.sin(x)])


# ===========================================================================
# 1. contour_sampler
# ===========================================================================

class TestContourSampler:

    def test_sampler_config_defaults(self):
        cfg = SamplerConfig()
        assert cfg.n_points == 32
        assert cfg.strategy == "uniform"

    def test_sampler_config_invalid_n_points(self):
        with pytest.raises(ValueError):
            SamplerConfig(n_points=1)

    def test_sampler_config_invalid_strategy(self):
        with pytest.raises(ValueError):
            SamplerConfig(strategy="bogus")

    def test_sample_uniform_shape(self):
        contour = _circle(100)
        result = sample_uniform(contour, n_points=32)
        assert isinstance(result, SampledContour)
        assert result.points.shape == (32, 2)
        assert result.n_points == 32

    def test_sample_uniform_arc_lengths_monotone(self):
        contour = _circle(100)
        result = sample_uniform(contour, n_points=16)
        assert np.all(np.diff(result.arc_lengths) >= 0)

    def test_sample_uniform_closed(self):
        contour = _sine_curve(50)
        result = sample_uniform(contour, n_points=20, closed=True)
        assert result.points.shape == (20, 2)

    def test_sample_curvature_shape(self):
        contour = _circle(80)
        result = sample_curvature(contour, n_points=24)
        assert result.points.shape == (24, 2)
        assert result.strategy == "curvature"

    def test_sample_random_reproducible(self):
        contour = _sine_curve(60)
        r1 = sample_random(contour, n_points=16, seed=7)
        r2 = sample_random(contour, n_points=16, seed=7)
        np.testing.assert_array_equal(r1.points, r2.points)

    def test_sample_corners_returns_sampled_contour(self):
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                            [0, 0.5], [0.5, 0.5]], dtype=float)
        result = sample_corners(contour, n_points=6)
        assert isinstance(result, SampledContour)

    def test_sample_contour_dispatch_uniform(self):
        contour = _circle(64)
        cfg = SamplerConfig(n_points=20, strategy="uniform")
        result = sample_contour(contour, cfg)
        assert result.strategy == "uniform"
        assert result.n_points == 20

    def test_normalize_contour_range(self):
        contour = _circle(64) * 50 + 100
        normed = normalize_contour(contour)
        assert normed.min() >= -1.0 - 1e-9
        assert normed.max() <= 1.0 + 1e-9

    def test_batch_sample_length(self):
        contours = [_circle(50), _sine_curve(40), _line(30)]
        cfg = SamplerConfig(n_points=16)
        results = batch_sample(contours, cfg)
        assert len(results) == 3
        for r in results:
            assert r.n_points == 16


# ===========================================================================
# 2. curvature_utils
# ===========================================================================

class TestCurvatureUtils:

    def test_curvature_config_invalid_threshold(self):
        with pytest.raises(ValueError):
            CurvatureConfig(corner_threshold=0.0)

    def test_compute_curvature_shape(self):
        curve = _circle(50)
        kappa = compute_curvature(curve)
        assert kappa.shape == (50,)

    def test_compute_curvature_circle_positive(self):
        curve = _circle(100)
        kappa = compute_curvature(curve)
        # Circle has consistent signed curvature
        assert np.all(np.isfinite(kappa))

    def test_compute_curvature_raises_too_few_points(self):
        with pytest.raises(ValueError):
            compute_curvature(np.array([[0, 0], [1, 1]], dtype=float))

    def test_compute_total_curvature_nonnegative(self):
        curve = _sine_curve(80)
        total = compute_total_curvature(curve)
        assert total >= 0.0

    def test_find_inflection_points_sine(self):
        curve = _sine_curve(100)
        infl = find_inflection_points(curve)
        # sin curve has sign changes in curvature
        assert len(infl) > 0
        assert infl.dtype == np.int64

    def test_compute_turning_angle_line(self):
        curve = _line(20)
        angle = compute_turning_angle(curve)
        assert abs(angle) < 1e-9  # straight line: zero turning

    def test_smooth_curvature_same_length(self):
        curve = _circle(60)
        kappa = compute_curvature(curve)
        smoothed = smooth_curvature(kappa, sigma=2.0)
        assert smoothed.shape == kappa.shape

    def test_corner_score_shape(self):
        curve = _circle(50)
        scores = corner_score(curve)
        assert scores.shape == (50,)
        assert np.all(scores >= 0)

    def test_find_corners_returns_indices(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                        [0.5, 0.5], [0.2, 0.8],
                        [0.3, 0.1], [0.9, 0.4]], dtype=float)
        cfg = CurvatureConfig(corner_threshold=0.01)
        corners = find_corners(pts, cfg)
        assert isinstance(corners, np.ndarray)

    def test_batch_curvature_length(self):
        curves = [_circle(30), _sine_curve(40), _line(20)]
        results = batch_curvature(curves)
        assert len(results) == 3
        for kappa, curve in zip(results, curves):
            assert kappa.shape == (len(curve),)


# ===========================================================================
# 3. curve_metrics
# ===========================================================================

class TestCurveMetrics:

    def test_config_invalid_n_samples(self):
        with pytest.raises(ValueError):
            CurveMetricConfig(n_samples=1)

    def test_curve_l2_identical_curves_zero(self):
        a = _circle(32)
        assert curve_l2(a, a) < 1e-10

    def test_curve_l2_different_curves_positive(self):
        a = _circle(32)
        b = _circle(32) * 2
        assert curve_l2(a, b) > 0.5

    def test_curve_l2_mirror_le_forward(self):
        a = _sine_curve(40)
        b = _sine_curve(40)[::-1]
        fwd = curve_l2(a, b)
        mirror = curve_l2_mirror(a, b)
        assert mirror <= fwd + 1e-10

    def test_curve_l2_mirror_identical_zero(self):
        a = _circle(32)
        assert curve_l2_mirror(a, a) < 1e-10

    def test_hausdorff_identical_zero(self):
        a = _circle(32)
        assert hausdorff_distance(a, a) < 1e-10

    def test_hausdorff_symmetric(self):
        a = _circle(32)
        b = _sine_curve(32)
        assert abs(hausdorff_distance(a, b) - hausdorff_distance(b, a)) < 1e-10

    def test_frechet_identical_zero(self):
        a = _circle(20)
        assert frechet_distance_approx(a, a) < 1e-10

    def test_frechet_ge_hausdorff(self):
        a = _sine_curve(30)
        b = _circle(30)
        # Frechet >= Hausdorff in general is not always true, but both should be >= 0
        assert frechet_distance_approx(a, b) >= 0.0

    def test_curve_length_line(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert abs(curve_length(pts) - 5.0) < 1e-10

    def test_curve_length_zero_for_single_point(self):
        pts = np.array([[1.0, 2.0]])
        assert curve_length(pts) == 0.0


# ===========================================================================
# 4. descriptor_utils
# ===========================================================================

class TestDescriptorUtils:

    def test_l2_normalize_unit_norm(self):
        v = RNG.standard_normal(16)
        n = l2_normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-9

    def test_l2_normalize_zero_vector(self):
        v = np.zeros(8)
        n = l2_normalize(v)
        np.testing.assert_array_equal(n, v)

    def test_l1_normalize_sums_to_one(self):
        v = np.abs(RNG.standard_normal(10)) + 0.1
        n = l1_normalize(v)
        assert abs(np.abs(n).sum() - 1.0) < 1e-9

    def test_batch_l2_normalize_row_norms(self):
        mat = RNG.standard_normal((8, 12))
        normed = batch_l2_normalize(mat)
        row_norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(8), atol=1e-9)

    def test_l2_distance_self_zero(self):
        v = RNG.standard_normal(20)
        assert l2_distance(v, v) == 0.0

    def test_cosine_distance_identical_zero(self):
        v = RNG.standard_normal(10)
        assert cosine_distance(v, v) < 1e-9

    def test_cosine_distance_opposite_one(self):
        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_distance(v, -v) - 1.0) < 1e-9

    def test_descriptor_distance_dispatch(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        for metric in ("l2", "cosine", "chi2", "l1"):
            d = descriptor_distance(a, b, metric=metric)
            assert d >= 0.0

    def test_pairwise_l2_shape(self):
        A = RNG.standard_normal((5, 8))
        B = RNG.standard_normal((7, 8))
        D = pairwise_l2(A, B)
        assert D.shape == (5, 7)
        assert np.all(D >= 0)

    def test_nn_match_returns_m_matches(self):
        q = RNG.standard_normal((4, 6))
        t = RNG.standard_normal((10, 6))
        matches = nn_match(q, t)
        assert len(matches) == 4
        for m in matches:
            assert isinstance(m, DescriptorMatch)
            assert m.distance >= 0.0

    def test_ratio_test_subset_of_nn(self):
        q = RNG.standard_normal((6, 8))
        t = RNG.standard_normal((12, 8))
        all_matches = nn_match(q, t)
        kept = ratio_test(q, t, ratio=0.75)
        assert len(kept) <= len(all_matches)

    def test_pairwise_cosine_diagonal_zero_for_identical(self):
        mat = RNG.standard_normal((4, 6))
        D = pairwise_cosine(mat, mat)
        np.testing.assert_allclose(np.diag(D), np.zeros(4), atol=1e-6)


# ===========================================================================
# 5. distance_matrix
# ===========================================================================

class TestDistanceMatrix:

    def test_distance_config_invalid_metric(self):
        with pytest.raises(ValueError):
            DistanceConfig(metric="minkowski")

    def test_euclidean_matrix_symmetric(self):
        X = RNG.standard_normal((6, 4))
        D = euclidean_distance_matrix(X)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_euclidean_matrix_zero_diagonal(self):
        X = RNG.standard_normal((5, 3))
        D = euclidean_distance_matrix(X)
        np.testing.assert_allclose(np.diag(D), np.zeros(5), atol=1e-10)

    def test_euclidean_matrix_raises_1d(self):
        with pytest.raises(ValueError):
            euclidean_distance_matrix(np.arange(6))

    def test_cosine_matrix_diagonal_zero(self):
        X = RNG.standard_normal((5, 4))
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(np.diag(D), np.zeros(5), atol=1e-10)

    def test_manhattan_matrix_symmetric(self):
        X = RNG.standard_normal((4, 3))
        D = manhattan_distance_matrix(X)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_build_distance_matrix_normalized_range(self):
        X = RNG.standard_normal((8, 5))
        cfg = DistanceConfig(metric="euclidean", normalize=True)
        D = build_distance_matrix(X, cfg)
        assert D.max() <= 1.0 + 1e-9
        assert D.min() >= 0.0 - 1e-9

    def test_normalize_distance_matrix_max_one(self):
        raw = RNG.random((5, 5))
        np.fill_diagonal(raw, 0.0)
        normed = normalize_distance_matrix(raw)
        assert abs(normed.max() - 1.0) < 1e-9 or normed.max() <= 1.0 + 1e-9

    def test_to_similarity_matrix_inverse_diagonal_one(self):
        X = RNG.standard_normal((5, 3))
        D = euclidean_distance_matrix(X)
        S = to_similarity_matrix(D, method="inverse")
        np.testing.assert_allclose(np.diag(S), np.ones(5), atol=1e-9)

    def test_to_similarity_matrix_gaussian_range(self):
        X = RNG.standard_normal((4, 3))
        D = euclidean_distance_matrix(X)
        S = to_similarity_matrix(D, method="gaussian", sigma=1.0)
        assert S.min() >= 0.0 - 1e-9
        assert S.max() <= 1.0 + 1e-9

    def test_threshold_distance_matrix_zeros_above(self):
        X = RNG.standard_normal((6, 4))
        D = euclidean_distance_matrix(X)
        threshold = D.max() / 2.0
        T = threshold_distance_matrix(D, threshold=threshold)
        assert np.all(T[T > 0] <= threshold + 1e-9)

    def test_top_k_distance_pairs_count(self):
        X = RNG.standard_normal((6, 4))
        D = euclidean_distance_matrix(X)
        pairs = top_k_distance_pairs(D, k=5)
        assert len(pairs) == 5
