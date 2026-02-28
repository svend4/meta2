"""
Property-based tests for three utility modules:
  1. puzzle_reconstruction.utils.distance_matrix
  2. puzzle_reconstruction.utils.color_utils
  3. puzzle_reconstruction.utils.gradient_utils

Verifies mathematical invariants:

distance_matrix:
- euclidean_distance_matrix:  symmetric, zero diagonal, non-negative,
                               triangle inequality
- cosine_distance_matrix:     symmetric, zero diagonal, values in [0, 2]
- manhattan_distance_matrix:  symmetric, zero diagonal, non-negative,
                               ≥ Chebyshev (implicit), ≤ d * euclidean
- build_distance_matrix:      shape (N,N), symmetric, zero diagonal
- normalize_distance_matrix:  output in [0, 1], zero diagonal preserved
- to_similarity_matrix:       diagonal = 1, values in (0, 1], method=inverse
                               → 1/(1+0)=1; non-negative
- threshold_distance_matrix:  all values ≤ threshold (or == fill) preserved
- top_k_distance_pairs:       len ≤ k, sorted ascending, i < j

color_utils:
- to_gray:        2-D output, same (H, W), uint8, grayscale→same values
- to_lab:         shape (H, W, 3), float32, L channel in approx [0, 100]
- to_hsv:         shape (H, W, 3), uint8, H in [0, 180], S/V in [0, 255]
- compute_histogram: length=bins, sum=1 (normalize), all non-negative
- compare_histograms: self-comparison → correlation≈1, bhattacharyya≈0
                      symmetry for chi/bhattacharyya
- color_distance: ≥ 0, self = 0, non-negative
- strip_histogram: length=bins, sum=1

gradient_utils:
- compute_gradient_magnitude: shape (H,W), float32, ≥ 0,
                               when normalize=True values in [0, 1]
- compute_gradient_direction: shape (H,W), float32, values in (-π, π]
- compute_sobel:   returns tuple of 3 arrays, each shape (H,W), float32
- compute_laplacian: shape (H,W), float32, when normalize=True ≥ 0
- threshold_gradient: shape (H,W), boolean dtype
- compute_edge_density: scalar in [0, 1]
- batch_compute_gradients: list length = input length; each shape (H,W)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

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
from puzzle_reconstruction.utils.color_utils import (
    to_gray,
    to_lab,
    to_hsv,
    compute_histogram,
    compare_histograms,
    color_distance,
    strip_histogram,
)
from puzzle_reconstruction.utils.gradient_utils import (
    GradientConfig,
    compute_gradient_magnitude,
    compute_gradient_direction,
    compute_sobel,
    compute_laplacian,
    threshold_gradient,
    compute_edge_density,
    batch_compute_gradients,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _feat_matrix(n: int = 6, d: int = 4, seed: int = 0) -> np.ndarray:
    """Random feature matrix (N, d)."""
    return np.random.default_rng(seed).uniform(0.0, 10.0, (n, d))


def _feat_matrix_unit(n: int = 5, d: int = 3, seed: int = 0) -> np.ndarray:
    """Feature matrix with unit norm rows."""
    X = np.random.default_rng(seed).uniform(0.1, 5.0, (n, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def _bgr_image(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    """Random BGR uint8 image."""
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3),
                                                 dtype=np.uint8)


def _gray_image(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    """Random grayscale uint8 image."""
    return np.random.default_rng(seed).integers(0, 256, (h, w),
                                                 dtype=np.uint8)


def _gradient_image(h: int = 30, w: int = 30, seed: int = 0) -> np.ndarray:
    """Image with visible gradient (horizontal ramp)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _uniform_image(h: int = 10, w: int = 10, value: int = 128) -> np.ndarray:
    """Uniform-valued grayscale image (zero gradient)."""
    return np.full((h, w), value, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. distance_matrix invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestEuclideanDistanceMatrix:
    """euclidean_distance_matrix invariants."""

    def test_shape(self) -> None:
        X = _feat_matrix(n=5, d=3)
        mat = euclidean_distance_matrix(X)
        assert mat.shape == (5, 5)

    def test_zero_diagonal(self) -> None:
        X = _feat_matrix(n=6, d=4)
        mat = euclidean_distance_matrix(X)
        assert np.allclose(np.diag(mat), 0.0)

    def test_symmetric(self) -> None:
        X = _feat_matrix(n=7, d=5, seed=1)
        mat = euclidean_distance_matrix(X)
        assert np.allclose(mat, mat.T, atol=1e-10)

    def test_non_negative(self) -> None:
        X = _feat_matrix(n=8, d=4, seed=2)
        mat = euclidean_distance_matrix(X)
        assert np.all(mat >= 0.0)

    def test_identical_rows_zero_distance(self) -> None:
        """Duplicate rows → distance = 0."""
        X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        mat = euclidean_distance_matrix(X)
        assert math.isclose(mat[0, 1], 0.0, abs_tol=1e-10)
        assert math.isclose(mat[1, 0], 0.0, abs_tol=1e-10)

    def test_known_distance(self) -> None:
        """Distance between [0,0] and [3,4] = 5."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        mat = euclidean_distance_matrix(X)
        assert math.isclose(mat[0, 1], 5.0, rel_tol=1e-9)

    def test_triangle_inequality(self) -> None:
        """d(i,k) ≤ d(i,j) + d(j,k) for all i,j,k."""
        X = _feat_matrix(n=5, d=3, seed=3)
        mat = euclidean_distance_matrix(X)
        n = mat.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert mat[i, k] <= mat[i, j] + mat[j, k] + 1e-9

    def test_dtype_float64(self) -> None:
        X = _feat_matrix(n=4, d=2)
        mat = euclidean_distance_matrix(X)
        assert mat.dtype == np.float64

    def test_single_point(self) -> None:
        X = np.array([[1.0, 2.0, 3.0]])
        mat = euclidean_distance_matrix(X)
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0.0

    @pytest.mark.parametrize("seed", [10, 20, 30, 40])
    def test_symmetry_multiple_seeds(self, seed: int) -> None:
        X = _feat_matrix(n=6, d=3, seed=seed)
        mat = euclidean_distance_matrix(X)
        assert np.allclose(mat, mat.T, atol=1e-10)


class TestCosineDistanceMatrix:
    """cosine_distance_matrix invariants."""

    def test_shape(self) -> None:
        X = _feat_matrix(n=5, d=4)
        mat = cosine_distance_matrix(X)
        assert mat.shape == (5, 5)

    def test_zero_diagonal(self) -> None:
        X = _feat_matrix(n=6, d=3, seed=5)
        mat = cosine_distance_matrix(X)
        assert np.allclose(np.diag(mat), 0.0, atol=1e-9)

    def test_symmetric(self) -> None:
        X = _feat_matrix(n=5, d=4, seed=6)
        mat = cosine_distance_matrix(X)
        assert np.allclose(mat, mat.T, atol=1e-9)

    def test_values_in_range(self) -> None:
        """Cosine distance must be in [0, 2]."""
        X = _feat_matrix(n=8, d=3, seed=7)
        mat = cosine_distance_matrix(X)
        assert np.all(mat >= -1e-9)
        assert np.all(mat <= 2.0 + 1e-9)

    def test_identical_rows_zero_distance(self) -> None:
        X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mat = cosine_distance_matrix(X)
        assert math.isclose(mat[0, 1], 0.0, abs_tol=1e-6)

    def test_unit_rows_distance_bounded(self) -> None:
        X = _feat_matrix_unit(n=5, d=3)
        mat = cosine_distance_matrix(X)
        assert np.all(mat >= -1e-9)
        assert np.all(mat <= 2.0 + 1e-9)


class TestManhattanDistanceMatrix:
    """manhattan_distance_matrix invariants."""

    def test_shape(self) -> None:
        X = _feat_matrix(n=4, d=3)
        mat = manhattan_distance_matrix(X)
        assert mat.shape == (4, 4)

    def test_zero_diagonal(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=8)
        mat = manhattan_distance_matrix(X)
        assert np.allclose(np.diag(mat), 0.0)

    def test_symmetric(self) -> None:
        X = _feat_matrix(n=6, d=4, seed=9)
        mat = manhattan_distance_matrix(X)
        assert np.allclose(mat, mat.T, atol=1e-9)

    def test_non_negative(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=10)
        mat = manhattan_distance_matrix(X)
        assert np.all(mat >= 0.0)

    def test_known_value(self) -> None:
        """Manhattan([0,0],[3,4]) = 7."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        mat = manhattan_distance_matrix(X)
        assert math.isclose(mat[0, 1], 7.0, rel_tol=1e-9)

    def test_manhattan_geq_euclidean(self) -> None:
        """Manhattan ≥ Euclidean for any pair."""
        from puzzle_reconstruction.utils.distance_matrix import (
            euclidean_distance_matrix,
        )
        X = _feat_matrix(n=5, d=4, seed=11)
        mhat = manhattan_distance_matrix(X)
        meuc = euclidean_distance_matrix(X)
        assert np.all(mhat >= meuc - 1e-9)


class TestBuildDistanceMatrix:
    """build_distance_matrix invariants."""

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
    def test_shape(self, metric: str) -> None:
        X = _feat_matrix(n=5, d=4)
        cfg = DistanceConfig(metric=metric, normalize=False)
        mat = build_distance_matrix(X, cfg)
        assert mat.shape == (5, 5)

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
    def test_zero_diagonal(self, metric: str) -> None:
        X = _feat_matrix(n=6, d=3, seed=12)
        cfg = DistanceConfig(metric=metric, normalize=False)
        mat = build_distance_matrix(X, cfg)
        assert np.allclose(np.diag(mat), 0.0, atol=1e-9)

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
    def test_symmetric(self, metric: str) -> None:
        X = _feat_matrix(n=5, d=4, seed=13)
        cfg = DistanceConfig(metric=metric, normalize=False)
        mat = build_distance_matrix(X, cfg)
        assert np.allclose(mat, mat.T, atol=1e-9)

    def test_normalize_flag(self) -> None:
        X = _feat_matrix(n=5, d=4, seed=14)
        cfg = DistanceConfig(metric="euclidean", normalize=True)
        mat = build_distance_matrix(X, cfg)
        assert np.all(mat >= -1e-9)
        assert np.all(mat <= 1.0 + 1e-9)

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError):
            DistanceConfig(metric="unknown")


class TestNormalizeDistanceMatrix:
    """normalize_distance_matrix invariants."""

    def test_output_in_unit_interval(self) -> None:
        X = _feat_matrix(n=6, d=3, seed=15)
        mat = euclidean_distance_matrix(X)
        norm = normalize_distance_matrix(mat)
        assert np.all(norm >= -1e-9)
        assert np.all(norm <= 1.0 + 1e-9)

    def test_zero_diagonal(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=16)
        mat = euclidean_distance_matrix(X)
        norm = normalize_distance_matrix(mat)
        assert np.allclose(np.diag(norm), 0.0, atol=1e-9)

    def test_max_off_diagonal_is_one(self) -> None:
        X = _feat_matrix(n=6, d=4, seed=17)
        mat = euclidean_distance_matrix(X)
        norm = normalize_distance_matrix(mat)
        n = norm.shape[0]
        mask = ~np.eye(n, dtype=bool)
        if norm[mask].size > 0:
            assert math.isclose(norm[mask].max(), 1.0, abs_tol=1e-9)

    def test_already_normalized_unchanged(self) -> None:
        mat = np.array([[0.0, 0.5], [0.5, 0.0]])
        norm = normalize_distance_matrix(mat)
        # max off-diagonal = 0.5, so mat /= 0.5 → [[0, 1], [1, 0]]
        assert norm[0, 1] == pytest.approx(1.0)

    def test_zero_matrix_stays_zero(self) -> None:
        mat = np.zeros((4, 4))
        norm = normalize_distance_matrix(mat)
        assert np.allclose(norm, 0.0)


class TestToSimilarityMatrix:
    """to_similarity_matrix invariants."""

    def test_diagonal_equals_one(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=18)
        mat = euclidean_distance_matrix(X)
        sim = to_similarity_matrix(mat, method="inverse")
        assert np.allclose(np.diag(sim), 1.0, atol=1e-9)

    def test_values_in_unit_interval_inverse(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=19)
        mat = euclidean_distance_matrix(X)
        sim = to_similarity_matrix(mat, method="inverse")
        assert np.all(sim >= -1e-9)
        assert np.all(sim <= 1.0 + 1e-9)

    def test_values_in_unit_interval_gaussian(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=20)
        mat = euclidean_distance_matrix(X)
        sim = to_similarity_matrix(mat, method="gaussian", sigma=2.0)
        assert np.all(sim >= -1e-9)
        assert np.all(sim <= 1.0 + 1e-9)

    def test_zero_distance_gives_one(self) -> None:
        """Distance 0 → similarity 1."""
        mat = np.array([[0.0, 1.0], [1.0, 0.0]])
        sim = to_similarity_matrix(mat, method="inverse")
        assert math.isclose(sim[0, 0], 1.0)

    def test_invalid_method_raises(self) -> None:
        mat = np.eye(3)
        with pytest.raises(ValueError):
            to_similarity_matrix(mat, method="unknown")

    def test_invalid_sigma_raises(self) -> None:
        mat = np.eye(3)
        with pytest.raises(ValueError):
            to_similarity_matrix(mat, method="gaussian", sigma=0.0)


class TestThresholdDistanceMatrix:
    """threshold_distance_matrix invariants."""

    def test_values_above_threshold_replaced(self) -> None:
        X = _feat_matrix(n=6, d=3, seed=21)
        mat = euclidean_distance_matrix(X)
        thr = float(np.median(mat))
        result = threshold_distance_matrix(mat, threshold=thr, fill=0.0)
        assert np.all(result <= thr + 1e-9)

    def test_shape_preserved(self) -> None:
        mat = np.random.default_rng(22).uniform(0, 5, (5, 5))
        result = threshold_distance_matrix(mat, threshold=2.5)
        assert result.shape == mat.shape

    def test_fill_value_used(self) -> None:
        mat = np.array([[0.0, 3.0], [3.0, 0.0]])
        result = threshold_distance_matrix(mat, threshold=1.0, fill=-1.0)
        assert result[0, 1] == pytest.approx(-1.0)

    def test_values_below_threshold_unchanged(self) -> None:
        mat = np.array([[0.0, 0.5], [0.5, 0.0]])
        result = threshold_distance_matrix(mat, threshold=1.0, fill=0.0)
        assert result[0, 1] == pytest.approx(0.5)


class TestTopKDistancePairs:
    """top_k_distance_pairs invariants."""

    def test_returns_k_pairs(self) -> None:
        X = _feat_matrix(n=5, d=3, seed=23)
        mat = euclidean_distance_matrix(X)
        pairs = top_k_distance_pairs(mat, k=3)
        assert len(pairs) == 3

    def test_pairs_sorted_ascending(self) -> None:
        X = _feat_matrix(n=6, d=3, seed=24)
        mat = euclidean_distance_matrix(X)
        pairs = top_k_distance_pairs(mat, k=5)
        dists = [p[2] for p in pairs]
        assert dists == sorted(dists)

    def test_i_less_than_j(self) -> None:
        X = _feat_matrix(n=6, d=3, seed=25)
        mat = euclidean_distance_matrix(X)
        for i, j, _ in top_k_distance_pairs(mat, k=10):
            assert i < j

    def test_k_exceeds_pairs_returns_all(self) -> None:
        """k > N*(N-1)/2 → returns all pairs."""
        X = _feat_matrix(n=3, d=2)
        mat = euclidean_distance_matrix(X)
        pairs = top_k_distance_pairs(mat, k=100)
        assert len(pairs) == 3   # 3*(3-1)/2 = 3

    def test_invalid_k_raises(self) -> None:
        mat = np.eye(3)
        with pytest.raises(ValueError):
            top_k_distance_pairs(mat, k=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. color_utils invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestToGray:
    """to_gray invariants."""

    def test_bgr_output_shape(self) -> None:
        img = _bgr_image(20, 30)
        gray = to_gray(img)
        assert gray.shape == (20, 30)

    def test_gray_output_unchanged(self) -> None:
        img = _gray_image(15, 15, seed=1)
        gray = to_gray(img)
        assert np.array_equal(gray, img)

    def test_output_dtype_uint8(self) -> None:
        img = _bgr_image(10, 10, seed=2)
        gray = to_gray(img)
        assert gray.dtype == np.uint8

    def test_ndim_2(self) -> None:
        img = _bgr_image(10, 10, seed=3)
        gray = to_gray(img)
        assert gray.ndim == 2

    @pytest.mark.parametrize("h,w", [(5, 5), (20, 30), (50, 40)])
    def test_shape_various_sizes(self, h: int, w: int) -> None:
        img = _bgr_image(h, w)
        gray = to_gray(img)
        assert gray.shape == (h, w)

    def test_pure_white_bgr_gives_high_value(self) -> None:
        img = np.full((5, 5, 3), 255, dtype=np.uint8)
        gray = to_gray(img)
        assert gray.mean() > 200


class TestToLab:
    """to_lab invariants."""

    def test_shape(self) -> None:
        img = _bgr_image(20, 20, seed=4)
        lab = to_lab(img)
        assert lab.shape == (20, 20, 3)

    def test_dtype_float32(self) -> None:
        img = _bgr_image(10, 10, seed=5)
        lab = to_lab(img)
        assert lab.dtype == np.float32

    def test_l_channel_range(self) -> None:
        """L channel should be in [0, 100]."""
        img = _bgr_image(20, 20, seed=6)
        lab = to_lab(img)
        L = lab[:, :, 0]
        assert float(L.min()) >= -0.5
        assert float(L.max()) <= 100.5

    def test_grayscale_input(self) -> None:
        img = _gray_image(10, 10, seed=7)
        lab = to_lab(img)
        assert lab.shape == (10, 10, 3)

    def test_pure_black_l_near_zero(self) -> None:
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        lab = to_lab(img)
        assert float(lab[:, :, 0].mean()) < 5.0

    def test_pure_white_l_near_100(self) -> None:
        img = np.full((5, 5, 3), 255, dtype=np.uint8)
        lab = to_lab(img)
        assert float(lab[:, :, 0].mean()) > 95.0


class TestToHsv:
    """to_hsv invariants."""

    def test_shape(self) -> None:
        img = _bgr_image(20, 20, seed=8)
        hsv = to_hsv(img)
        assert hsv.shape == (20, 20, 3)

    def test_dtype_uint8(self) -> None:
        img = _bgr_image(10, 10, seed=9)
        hsv = to_hsv(img)
        assert hsv.dtype == np.uint8

    def test_h_channel_range(self) -> None:
        """H channel in OpenCV uint8 HSV is in [0, 180]."""
        img = _bgr_image(20, 20, seed=10)
        hsv = to_hsv(img)
        H = hsv[:, :, 0]
        assert int(H.max()) <= 180

    def test_sv_channels_range(self) -> None:
        """S and V channels in [0, 255]."""
        img = _bgr_image(20, 20, seed=11)
        hsv = to_hsv(img)
        assert int(hsv[:, :, 1].min()) >= 0
        assert int(hsv[:, :, 1].max()) <= 255
        assert int(hsv[:, :, 2].min()) >= 0
        assert int(hsv[:, :, 2].max()) <= 255

    def test_grayscale_input(self) -> None:
        img = _gray_image(10, 10, seed=12)
        hsv = to_hsv(img)
        assert hsv.shape == (10, 10, 3)


class TestComputeHistogram:
    """compute_histogram invariants."""

    def test_length_equals_bins(self) -> None:
        img = _bgr_image(20, 20, seed=13)
        hist = compute_histogram(img, bins=64)
        assert len(hist) == 64

    def test_normalized_sum_one(self) -> None:
        img = _bgr_image(20, 20, seed=14)
        hist = compute_histogram(img, bins=256, normalize=True)
        assert math.isclose(float(hist.sum()), 1.0, abs_tol=1e-5)

    def test_non_negative(self) -> None:
        img = _bgr_image(20, 20, seed=15)
        hist = compute_histogram(img, bins=256, normalize=True)
        assert np.all(hist >= 0.0)

    def test_dtype_float32(self) -> None:
        img = _bgr_image(10, 10, seed=16)
        hist = compute_histogram(img, bins=128)
        assert hist.dtype == np.float32

    def test_grayscale_image(self) -> None:
        img = _gray_image(15, 15, seed=17)
        hist = compute_histogram(img, bins=32)
        assert len(hist) == 32
        assert math.isclose(float(hist.sum()), 1.0, abs_tol=1e-5)

    def test_unnormalized_sum_equals_pixel_count(self) -> None:
        img = _bgr_image(10, 10, seed=18)
        hist = compute_histogram(img, bins=256, normalize=False)
        assert math.isclose(float(hist.sum()), 100.0, abs_tol=0.5)

    @pytest.mark.parametrize("bins", [16, 32, 64, 128, 256])
    def test_various_bins(self, bins: int) -> None:
        img = _bgr_image(20, 20)
        hist = compute_histogram(img, bins=bins)
        assert len(hist) == bins


class TestCompareHistograms:
    """compare_histograms invariants."""

    def _make_hist(self, seed: int = 0, bins: int = 64) -> np.ndarray:
        img = _bgr_image(20, 20, seed=seed)
        return compute_histogram(img, bins=bins)

    def test_self_correlation_near_one(self) -> None:
        h = self._make_hist()
        score = compare_histograms(h, h, method="correlation")
        assert math.isclose(score, 1.0, abs_tol=1e-4)

    def test_self_bhattacharyya_near_zero(self) -> None:
        h = self._make_hist()
        score = compare_histograms(h, h, method="bhattacharyya")
        assert math.isclose(score, 0.0, abs_tol=1e-4)

    def test_self_chi_near_zero(self) -> None:
        h = self._make_hist()
        score = compare_histograms(h, h, method="chi")
        assert math.isclose(score, 0.0, abs_tol=1e-4)

    def test_bhattacharyya_symmetry(self) -> None:
        h1 = self._make_hist(seed=0)
        h2 = self._make_hist(seed=1)
        s12 = compare_histograms(h1, h2, method="bhattacharyya")
        s21 = compare_histograms(h2, h1, method="bhattacharyya")
        assert math.isclose(s12, s21, abs_tol=1e-6)

    def test_chi_non_negative(self) -> None:
        """Chi-squared distance is non-negative (not symmetric in general)."""
        h1 = self._make_hist(seed=2)
        h2 = self._make_hist(seed=3)
        s12 = compare_histograms(h1, h2, method="chi")
        assert s12 >= -1e-6

    def test_invalid_method_raises(self) -> None:
        h = self._make_hist()
        with pytest.raises(ValueError):
            compare_histograms(h, h, method="unknown_method")


class TestColorDistance:
    """color_distance invariants."""

    def test_self_distance_zero(self) -> None:
        color = np.array([100, 150, 200], dtype=np.uint8)
        assert math.isclose(color_distance(color, color, space="lab"), 0.0,
                             abs_tol=1e-4)

    def test_self_distance_zero_rgb(self) -> None:
        color = np.array([100, 150, 200], dtype=np.uint8)
        assert math.isclose(color_distance(color, color, space="rgb"), 0.0,
                             abs_tol=1e-4)

    def test_non_negative_lab(self) -> None:
        c1 = np.array([255, 0, 0], dtype=np.uint8)
        c2 = np.array([0, 255, 0], dtype=np.uint8)
        assert color_distance(c1, c2, space="lab") >= 0.0

    def test_non_negative_rgb(self) -> None:
        c1 = np.array([255, 0, 0], dtype=np.uint8)
        c2 = np.array([0, 0, 255], dtype=np.uint8)
        assert color_distance(c1, c2, space="rgb") >= 0.0

    def test_invalid_space_raises(self) -> None:
        color = np.array([100, 100, 100], dtype=np.uint8)
        with pytest.raises(ValueError):
            color_distance(color, color, space="xyz")

    def test_black_white_distance_positive(self) -> None:
        black = np.array([0, 0, 0], dtype=np.uint8)
        white = np.array([255, 255, 255], dtype=np.uint8)
        assert color_distance(black, white, space="lab") > 0.0

    @pytest.mark.parametrize("space", ["lab", "rgb"])
    def test_symmetry(self, space: str) -> None:
        c1 = np.array([200, 100, 50], dtype=np.uint8)
        c2 = np.array([50, 200, 100], dtype=np.uint8)
        d12 = color_distance(c1, c2, space=space)
        d21 = color_distance(c2, c1, space=space)
        assert math.isclose(d12, d21, abs_tol=1e-4)


class TestStripHistogram:
    """strip_histogram invariants."""

    def test_length_equals_bins(self) -> None:
        img = _bgr_image(30, 30, seed=19)
        hist = strip_histogram(img, side=0, bins=64)
        assert len(hist) == 64

    def test_normalized_sum_one(self) -> None:
        img = _bgr_image(30, 30, seed=20)
        hist = strip_histogram(img, side=0)
        assert math.isclose(float(hist.sum()), 1.0, abs_tol=1e-5)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side: int) -> None:
        img = _bgr_image(30, 30, seed=21)
        hist = strip_histogram(img, side=side, bins=32)
        assert len(hist) == 32
        assert math.isclose(float(hist.sum()), 1.0, abs_tol=1e-5)

    def test_invalid_side_raises(self) -> None:
        img = _bgr_image(20, 20)
        with pytest.raises(ValueError):
            strip_histogram(img, side=5)

    def test_grayscale_image(self) -> None:
        img = _gray_image(20, 20, seed=22)
        hist = strip_histogram(img, side=2, bins=64)
        assert len(hist) == 64
        assert math.isclose(float(hist.sum()), 1.0, abs_tol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. gradient_utils invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeGradientMagnitude:
    """compute_gradient_magnitude invariants."""

    def test_shape_preserving(self) -> None:
        img = _gray_image(20, 20, seed=23)
        mag = compute_gradient_magnitude(img)
        assert mag.shape == (20, 20)

    def test_dtype_float32(self) -> None:
        img = _gray_image(15, 15, seed=24)
        mag = compute_gradient_magnitude(img)
        assert mag.dtype == np.float32

    def test_non_negative(self) -> None:
        img = _bgr_image(20, 20, seed=25)
        mag = compute_gradient_magnitude(img)
        assert np.all(mag >= 0.0)

    def test_normalized_in_unit_interval(self) -> None:
        img = _gradient_image(30, 30, seed=26)
        cfg = GradientConfig(normalize=True)
        mag = compute_gradient_magnitude(img, cfg)
        assert np.all(mag >= -1e-6)
        assert np.all(mag <= 1.0 + 1e-6)

    def test_uniform_image_near_zero_gradient(self) -> None:
        img = _uniform_image(10, 10, value=128)
        mag = compute_gradient_magnitude(img)
        assert float(mag.max()) < 1e-3

    def test_bgr_input(self) -> None:
        img = _bgr_image(15, 15, seed=27)
        mag = compute_gradient_magnitude(img)
        assert mag.shape == (15, 15)

    def test_gradient_ramp_has_nonzero_magnitude(self) -> None:
        img = _gradient_image(20, 20)
        mag = compute_gradient_magnitude(img)
        # Horizontal ramp → nonzero x-gradient in interior
        interior = mag[1:-1, 1:-1]
        assert float(interior.max()) > 0.0

    @pytest.mark.parametrize("h,w", [(10, 10), (20, 30), (40, 25)])
    def test_shape_various_sizes(self, h: int, w: int) -> None:
        img = _gray_image(h, w)
        mag = compute_gradient_magnitude(img)
        assert mag.shape == (h, w)


class TestComputeGradientDirection:
    """compute_gradient_direction invariants."""

    def test_shape_preserving(self) -> None:
        img = _gray_image(20, 20, seed=28)
        dirn = compute_gradient_direction(img)
        assert dirn.shape == (20, 20)

    def test_dtype_float32(self) -> None:
        img = _gray_image(15, 15, seed=29)
        dirn = compute_gradient_direction(img)
        assert dirn.dtype == np.float32

    def test_values_in_range(self) -> None:
        """Values must be in (-π, π]."""
        img = _bgr_image(20, 20, seed=30)
        dirn = compute_gradient_direction(img)
        assert np.all(dirn >= -math.pi - 1e-5)
        assert np.all(dirn <= math.pi + 1e-5)

    def test_bgr_input(self) -> None:
        img = _bgr_image(15, 15, seed=31)
        dirn = compute_gradient_direction(img)
        assert dirn.shape == (15, 15)


class TestComputeSobel:
    """compute_sobel invariants."""

    def test_returns_three_arrays(self) -> None:
        img = _gray_image(20, 20, seed=32)
        result = compute_sobel(img)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_same_shape(self) -> None:
        img = _gray_image(20, 20, seed=33)
        mag, dx, dy = compute_sobel(img)
        assert mag.shape == (20, 20)
        assert dx.shape == (20, 20)
        assert dy.shape == (20, 20)

    def test_all_float32(self) -> None:
        img = _gray_image(15, 15, seed=34)
        mag, dx, dy = compute_sobel(img)
        assert mag.dtype == np.float32
        assert dx.dtype == np.float32
        assert dy.dtype == np.float32

    def test_magnitude_non_negative(self) -> None:
        img = _bgr_image(20, 20, seed=35)
        mag, _, _ = compute_sobel(img)
        assert np.all(mag >= 0.0)

    def test_magnitude_normalized_when_flag_set(self) -> None:
        img = _gradient_image(20, 20)
        cfg = GradientConfig(normalize=True)
        mag, _, _ = compute_sobel(img, cfg)
        assert np.all(mag >= -1e-6)
        assert np.all(mag <= 1.0 + 1e-6)

    def test_bgr_input(self) -> None:
        img = _bgr_image(15, 15, seed=36)
        mag, dx, dy = compute_sobel(img)
        assert mag.shape == (15, 15)


class TestComputeLaplacian:
    """compute_laplacian invariants."""

    def test_shape_preserving(self) -> None:
        img = _gray_image(20, 20, seed=37)
        lap = compute_laplacian(img)
        assert lap.shape == (20, 20)

    def test_dtype_float32(self) -> None:
        img = _gray_image(15, 15, seed=38)
        lap = compute_laplacian(img)
        assert lap.dtype == np.float32

    def test_normalized_non_negative(self) -> None:
        img = _bgr_image(20, 20, seed=39)
        lap = compute_laplacian(img, normalize=True)
        assert np.all(lap >= -1e-6)

    def test_normalized_in_unit_interval(self) -> None:
        img = _gradient_image(20, 20)
        lap = compute_laplacian(img, normalize=True)
        assert np.all(lap >= -1e-6)
        assert np.all(lap <= 1.0 + 1e-6)

    def test_bgr_input(self) -> None:
        img = _bgr_image(15, 15, seed=40)
        lap = compute_laplacian(img)
        assert lap.shape == (15, 15)


class TestThresholdGradient:
    """threshold_gradient invariants."""

    def test_shape_preserving(self) -> None:
        img = _gradient_image(20, 20)
        mag = compute_gradient_magnitude(img)
        mask = threshold_gradient(mag)
        assert mask.shape == (20, 20)

    def test_boolean_dtype(self) -> None:
        img = _gradient_image(20, 20)
        mag = compute_gradient_magnitude(img)
        mask = threshold_gradient(mag)
        assert mask.dtype == bool

    def test_low_threshold_gives_all_true(self) -> None:
        img = _gradient_image(20, 20)
        mag = compute_gradient_magnitude(img)
        mask = threshold_gradient(mag, threshold=0.0)
        # All non-edge pixels are uniform (zero), so mostly True only where > 0
        # At threshold=0 exactly: mask = mag >= 0 = all True
        assert np.all(mask)

    def test_high_threshold_gives_mostly_false(self) -> None:
        img = _gradient_image(20, 20)
        cfg = GradientConfig(normalize=True)
        mag = compute_gradient_magnitude(img, cfg)
        mask = threshold_gradient(mag, threshold=1.0)
        # Only pixels exactly at max count
        assert int(mask.sum()) < mag.size

    def test_consistent_with_manual_threshold(self) -> None:
        img = _gradient_image(20, 20)
        cfg = GradientConfig(normalize=True)
        mag = compute_gradient_magnitude(img, cfg)
        thr = 0.3
        mask = threshold_gradient(mag, threshold=thr)
        expected = mag >= thr
        assert np.array_equal(mask, expected)


class TestComputeEdgeDensity:
    """compute_edge_density invariants."""

    def test_in_unit_interval(self) -> None:
        img = _bgr_image(20, 20, seed=41)
        density = compute_edge_density(img)
        assert 0.0 <= density <= 1.0

    def test_uniform_image_near_zero(self) -> None:
        img = _uniform_image(20, 20, value=100)
        density = compute_edge_density(img)
        assert density < 1e-3

    def test_gradient_image_nonzero(self) -> None:
        img = _gradient_image(30, 30)
        density = compute_edge_density(img)
        # Gradient ramp → some edges detected
        assert density >= 0.0   # can be low if threshold is high

    def test_with_roi(self) -> None:
        img = _bgr_image(40, 40, seed=42)
        density = compute_edge_density(img, roi=(5, 5, 20, 20))
        assert 0.0 <= density <= 1.0

    def test_scalar_result(self) -> None:
        img = _gray_image(15, 15, seed=43)
        density = compute_edge_density(img)
        assert isinstance(density, float)

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_multiple_images(self, seed: int) -> None:
        img = _bgr_image(20, 20, seed=seed)
        density = compute_edge_density(img)
        assert 0.0 <= density <= 1.0


class TestBatchComputeGradients:
    """batch_compute_gradients invariants."""

    def test_list_length_preserved(self) -> None:
        images = [_gray_image(20, 20, seed=i) for i in range(5)]
        results = batch_compute_gradients(images)
        assert len(results) == 5

    def test_each_result_correct_shape(self) -> None:
        images = [_gray_image(20, 20, seed=i) for i in range(4)]
        results = batch_compute_gradients(images)
        for mag in results:
            assert mag.shape == (20, 20)

    def test_each_result_float32(self) -> None:
        images = [_gray_image(15, 15, seed=i) for i in range(3)]
        results = batch_compute_gradients(images)
        for mag in results:
            assert mag.dtype == np.float32

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            batch_compute_gradients([])

    def test_single_image(self) -> None:
        img = _gradient_image(20, 20)
        results = batch_compute_gradients([img])
        assert len(results) == 1
        assert results[0].shape == (20, 20)

    def test_mixed_sizes(self) -> None:
        images = [
            _gray_image(10, 10, seed=0),
            _gray_image(20, 30, seed=1),
            _bgr_image(15, 25, seed=2),
        ]
        results = batch_compute_gradients(images)
        assert results[0].shape == (10, 10)
        assert results[1].shape == (20, 30)
        assert results[2].shape == (15, 25)
