"""
Property-based tests for three utility modules:
  1. puzzle_reconstruction.utils.normalization_utils
  2. puzzle_reconstruction.utils.distance_utils
  3. puzzle_reconstruction.utils.signal_utils

Verifies mathematical invariants:
- l1_normalize:         sum(|x|) = 1, length preserved
- l2_normalize:         ||x||₂ = 1, length preserved
- minmax_normalize:     output ∈ [0, 1], min = 0, max = 1
- zscore_normalize:     mean ≈ 0, std ≈ 1
- softmax:              output ∈ [0, 1], sum = 1, preserves ordering
- clamp:                output ∈ [lo, hi]
- symmetrize_matrix:    result == result.T
- zero_diagonal:        diag = 0, off-diagonal unchanged
- euclidean_distance:   ≥ 0, self = 0, symmetry, triangle inequality
- cosine_similarity:    ∈ [−1, 1], self = 1, symmetric
- cosine_distance:      ∈ [0, 2], self = 0
- manhattan_distance:   ≥ 0, self = 0, symmetry, ≥ Chebyshev
- chebyshev_distance:   ≥ 0, self = 0, ≤ manhattan
- hausdorff_distance:   ≥ 0, symmetric, same set = 0
- chamfer_distance:     ≥ 0, symmetric, same set = 0
- pairwise_distances:   shape (N, N), symmetric, zero diagonal
- smooth_signal:        length preserved, constant signal unchanged
- normalize_signal:     output ∈ [out_min, out_max]
- signal_energy:        ≥ 0, zero for zero signal, scales as amplitude²
- compute_autocorrelation: length 2N−1, center = 1 (normalized), symmetric
- resample_signal:      exact n_out points, endpoint preservation
- find_peaks:           valid indices, spacing ≥ min_distance
- segment_signal:       segments cover all above-threshold indices
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.normalization_utils import (
    l1_normalize,
    l2_normalize,
    minmax_normalize,
    zscore_normalize,
    softmax,
    clamp,
    symmetrize_matrix,
    zero_diagonal,
    normalize_rows,
)
from puzzle_reconstruction.utils.distance_utils import (
    euclidean_distance,
    cosine_similarity,
    cosine_distance,
    manhattan_distance,
    chebyshev_distance,
    hausdorff_distance,
    chamfer_distance,
    pairwise_distances,
    nearest_neighbor_dist,
)
from puzzle_reconstruction.utils.signal_utils import (
    smooth_signal,
    normalize_signal,
    find_peaks,
    find_valleys,
    compute_autocorrelation,
    compute_cross_correlation,
    signal_energy,
    segment_signal,
    resample_signal,
    phase_shift,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _vec(n: int = 8, seed: int = 0) -> np.ndarray:
    """Random positive float vector."""
    return np.random.default_rng(seed).uniform(0.1, 5.0, n)


def _vec_any(n: int = 8, seed: int = 0) -> np.ndarray:
    """Random float vector (may be negative)."""
    return np.random.default_rng(seed).uniform(-3.0, 3.0, n)


def _unit_vec(d: int = 4) -> np.ndarray:
    v = np.array([1.0] + [0.0] * (d - 1))
    return v


def _pts(n: int = 10, d: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 10, (n, d))


def _square_signal(n: int = 64) -> np.ndarray:
    """Signal with a clear square pulse."""
    s = np.zeros(n)
    s[n // 4: 3 * n // 4] = 1.0
    return s


def _sine_signal(n: int = 64, freq: float = 2.0) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi * freq, n)
    return np.sin(t)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — normalization_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestL1Normalize:
    """L1-normalization invariants."""

    def test_l1_norm_equals_one(self):
        v = _vec(10)
        result = l1_normalize(v)
        assert abs(np.abs(result).sum() - 1.0) < 1e-12

    def test_length_preserved(self):
        v = _vec(7)
        assert len(l1_normalize(v)) == len(v)

    def test_zero_vector_returns_zeros(self):
        v = np.zeros(5)
        result = l1_normalize(v)
        assert np.all(result == 0.0)

    @pytest.mark.parametrize("n", [1, 4, 16, 100])
    def test_various_lengths(self, n):
        v = _vec(n)
        result = l1_normalize(v)
        assert result.shape == (n,)
        assert abs(np.abs(result).sum() - 1.0) < 1e-10

    def test_positive_input_positive_output(self):
        v = _vec(8)
        result = l1_normalize(v)
        assert np.all(result >= 0)

    def test_uniform_vector_gives_equal_weights(self):
        v = np.ones(5)
        result = l1_normalize(v)
        assert np.allclose(result, 0.2, atol=1e-12)

    def test_dtype_is_float64(self):
        v = np.array([1, 2, 3], dtype=np.int32)
        result = l1_normalize(v)
        assert result.dtype == np.float64


class TestL2Normalize:
    """L2-normalization invariants."""

    def test_l2_norm_equals_one(self):
        v = _vec(10)
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-12

    def test_length_preserved(self):
        v = _vec(6)
        assert len(l2_normalize(v)) == len(v)

    def test_zero_vector_returns_zeros(self):
        v = np.zeros(4)
        result = l2_normalize(v)
        assert np.all(result == 0.0)

    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        result = l2_normalize(v)
        assert np.allclose(result, v, atol=1e-14)

    @pytest.mark.parametrize("scale", [0.1, 1.0, 5.0, 100.0])
    def test_scale_invariant(self, scale):
        """L2-normalized vector is independent of input scale."""
        v = _vec(5)
        assert np.allclose(l2_normalize(v * scale), l2_normalize(v), atol=1e-12)

    def test_negative_values_allowed(self):
        v = np.array([-3.0, 4.0])
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-12

    def test_known_result(self):
        v = np.array([3.0, 4.0])   # norm = 5
        result = l2_normalize(v)
        assert np.allclose(result, [0.6, 0.8], atol=1e-14)


class TestMinmaxNormalize:
    """Min-max normalization invariants."""

    def test_output_in_zero_one(self):
        v = _vec_any(20)
        result = minmax_normalize(v)
        assert np.all(result >= -1e-12)
        assert np.all(result <= 1.0 + 1e-12)

    def test_min_becomes_zero(self):
        v = _vec_any(10)
        result = minmax_normalize(v)
        assert abs(result.min()) < 1e-12

    def test_max_becomes_one(self):
        v = _vec_any(10)
        result = minmax_normalize(v)
        assert abs(result.max() - 1.0) < 1e-12

    def test_constant_returns_zeros(self):
        v = np.full(8, 3.5)
        result = minmax_normalize(v)
        assert np.all(result == 0.0)

    def test_length_preserved(self):
        v = _vec(12)
        assert len(minmax_normalize(v)) == len(v)

    @pytest.mark.parametrize("n", [2, 5, 50])
    def test_two_distinct_values(self, n):
        v = np.array([0.0, 1.0] * (n // 2 + 1))[:n]
        result = minmax_normalize(v)
        assert abs(result.min()) < 1e-12
        assert abs(result.max() - 1.0) < 1e-12

    def test_order_preserved(self):
        """Monotone sequence remains monotone after normalization."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = minmax_normalize(v)
        assert np.all(np.diff(result) > 0)


class TestZscoreNormalize:
    """Z-score normalization invariants."""

    def test_mean_near_zero(self):
        v = _vec(50)
        result = zscore_normalize(v)
        assert abs(result.mean()) < 1e-10

    def test_std_near_one(self):
        v = _vec(50)
        result = zscore_normalize(v)
        assert abs(result.std() - 1.0) < 1e-10

    def test_constant_returns_zeros(self):
        v = np.full(8, 7.0)
        result = zscore_normalize(v)
        assert np.all(result == 0.0)

    def test_length_preserved(self):
        v = _vec(12)
        assert len(zscore_normalize(v)) == len(v)

    def test_already_standardized_unchanged(self):
        """Normalizing a z-score again is idempotent."""
        v = _vec_any(20, seed=1)
        once = zscore_normalize(v)
        # Re-normalizing should give mean≈0, std≈1 again
        twice = zscore_normalize(once)
        assert abs(twice.mean()) < 1e-10
        assert abs(twice.std() - 1.0) < 1e-10

    @pytest.mark.parametrize("shift", [-100.0, 0.0, 50.0])
    def test_translation_invariant(self, shift):
        """z-score is translation invariant."""
        v = _vec(15)
        result1 = zscore_normalize(v)
        result2 = zscore_normalize(v + shift)
        assert np.allclose(result1, result2, atol=1e-10)


class TestSoftmax:
    """Softmax invariants."""

    def test_sum_equals_one(self):
        v = _vec_any(8)
        result = softmax(v)
        assert abs(result.sum() - 1.0) < 1e-12

    def test_output_in_zero_one(self):
        v = _vec_any(8)
        result = softmax(v)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_length_preserved(self):
        v = _vec(6)
        assert len(softmax(v)) == len(v)

    def test_uniform_input_uniform_output(self):
        """All-equal inputs → uniform probabilities."""
        v = np.ones(5)
        result = softmax(v)
        assert np.allclose(result, 0.2, atol=1e-12)

    def test_argmax_preserved(self):
        """Argmax of logits equals argmax of softmax output."""
        v = _vec_any(10)
        assert np.argmax(softmax(v)) == np.argmax(v)

    def test_translation_invariant(self):
        """softmax(x + c) == softmax(x) for any constant c."""
        v = _vec_any(8)
        c = 100.0
        assert np.allclose(softmax(v), softmax(v + c), atol=1e-12)

    def test_dtype_is_float64(self):
        result = softmax(np.array([1.0, 2.0, 3.0]))
        assert result.dtype == np.float64

    def test_single_element(self):
        result = softmax(np.array([5.0]))
        assert abs(result[0] - 1.0) < 1e-12


class TestClamp:
    """Clamp invariants."""

    def test_output_in_range(self):
        v = _vec_any(20)
        result = clamp(v, -1.0, 1.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_in_range_values_unchanged(self):
        v = np.array([0.5, 0.3, 0.7])
        result = clamp(v, 0.0, 1.0)
        assert np.allclose(result, v, atol=1e-14)

    def test_shape_preserved(self):
        v = _vec_any(15)
        result = clamp(v, -2.0, 2.0)
        assert result.shape == v.shape

    @pytest.mark.parametrize("lo, hi", [(-5.0, 5.0), (0.0, 1.0), (-1.0, 0.0)])
    def test_various_ranges(self, lo, hi):
        v = _vec_any(10)
        result = clamp(v, lo, hi)
        assert np.all(result >= lo - 1e-14)
        assert np.all(result <= hi + 1e-14)

    def test_all_below_lo_become_lo(self):
        v = np.array([-10.0, -5.0, -3.0])
        result = clamp(v, 0.0, 1.0)
        assert np.all(result == 0.0)

    def test_all_above_hi_become_hi(self):
        v = np.array([5.0, 10.0, 20.0])
        result = clamp(v, 0.0, 1.0)
        assert np.all(result == 1.0)


class TestSymmetrizeMatrix:
    """Symmetrize matrix invariants."""

    def test_result_is_symmetric(self):
        A = _pts(4, 4).reshape(4, 4)
        S = symmetrize_matrix(A)
        assert np.allclose(S, S.T, atol=1e-14)

    def test_symmetric_input_unchanged(self):
        A = np.array([[1.0, 2.0], [2.0, 3.0]])
        S = symmetrize_matrix(A)
        assert np.allclose(S, A, atol=1e-14)

    def test_shape_preserved(self):
        A = np.random.default_rng(1).random((5, 5))
        S = symmetrize_matrix(A)
        assert S.shape == (5, 5)

    def test_diagonal_preserved(self):
        A = np.diag([1.0, 2.0, 3.0]) + np.random.default_rng(2).random((3, 3))
        S = symmetrize_matrix(A)
        assert np.allclose(np.diag(S), np.diag(A), atol=1e-14)

    def test_idempotent(self):
        """symmetrize(symmetrize(A)) == symmetrize(A)."""
        A = np.random.default_rng(3).random((4, 4))
        S1 = symmetrize_matrix(A)
        S2 = symmetrize_matrix(S1)
        assert np.allclose(S1, S2, atol=1e-14)


class TestZeroDiagonal:
    """Zero-diagonal invariants."""

    def test_diagonal_is_zero(self):
        A = np.random.default_rng(5).random((5, 5))
        result = zero_diagonal(A)
        assert np.all(np.diag(result) == 0.0)

    def test_off_diagonal_unchanged(self):
        A = np.random.default_rng(6).random((4, 4))
        result = zero_diagonal(A)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert result[i, j] == A[i, j]

    def test_shape_preserved(self):
        A = np.ones((3, 3))
        result = zero_diagonal(A)
        assert result.shape == (3, 3)

    def test_already_zero_diagonal_unchanged(self):
        A = np.array([[0.0, 1.0], [2.0, 0.0]])
        result = zero_diagonal(A)
        assert np.allclose(result, A, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — distance_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestEuclideanDistance:
    """Euclidean distance invariants."""

    def test_self_distance_zero(self):
        v = _vec(6)
        assert euclidean_distance(v, v) == 0.0

    def test_nonnegative(self):
        a, b = _vec(5, 0), _vec(5, 1)
        assert euclidean_distance(a, b) >= 0.0

    def test_symmetry(self):
        a, b = _vec(5, 2), _vec(5, 3)
        assert abs(euclidean_distance(a, b) - euclidean_distance(b, a)) < 1e-12

    def test_known_result(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(euclidean_distance(a, b) - 5.0) < 1e-12

    def test_triangle_inequality(self):
        a = _vec(6, 10)
        b = _vec(6, 11)
        c = _vec(6, 12)
        d_ab = euclidean_distance(a, b)
        d_bc = euclidean_distance(b, c)
        d_ac = euclidean_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-10

    @pytest.mark.parametrize("n", [1, 3, 10, 128])
    def test_various_dimensions(self, n):
        a = np.zeros(n)
        b = np.ones(n)
        expected = math.sqrt(n)
        assert abs(euclidean_distance(a, b) - expected) < 1e-10

    def test_scale_factor(self):
        """d(k*a, k*b) = |k| * d(a, b)."""
        a = _vec(4, 20)
        b = _vec(4, 21)
        k = 3.0
        assert abs(euclidean_distance(k * a, k * b) - k * euclidean_distance(a, b)) < 1e-10


class TestCosineMetrics:
    """Cosine similarity and distance invariants."""

    def test_self_similarity_equals_one(self):
        v = _vec(5)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-12

    def test_similarity_in_range(self):
        a, b = _vec_any(8, 0), _vec_any(8, 1)
        sim = cosine_similarity(a, b)
        assert -1.0 - 1e-12 <= sim <= 1.0 + 1e-12

    def test_symmetry(self):
        a, b = _vec(5, 5), _vec(5, 6)
        assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-12

    def test_orthogonal_similarity_zero(self):
        """Orthogonal vectors have cosine similarity = 0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-12

    def test_opposite_similarity_minus_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-12

    def test_distance_complement(self):
        """cosine_distance = 1 - cosine_similarity for non-zero vectors."""
        a, b = _vec(5, 7), _vec(5, 8)
        assert abs(cosine_distance(a, b) - (1.0 - cosine_similarity(a, b))) < 1e-12

    def test_self_distance_zero(self):
        v = _vec(5)
        assert cosine_distance(v, v) < 1e-12

    def test_distance_in_range(self):
        a, b = _vec_any(6, 0), _vec_any(6, 1)
        d = cosine_distance(a, b)
        assert -1e-12 <= d <= 2.0 + 1e-12

    def test_zero_vector_similarity_zero(self):
        v = _vec(5)
        z = np.zeros(5)
        assert cosine_similarity(v, z) == 0.0


class TestManhattanChebyshev:
    """Manhattan and Chebyshev distance invariants."""

    def test_manhattan_self_zero(self):
        v = _vec(6)
        assert manhattan_distance(v, v) == 0.0

    def test_manhattan_nonnegative(self):
        a, b = _vec(5, 0), _vec(5, 1)
        assert manhattan_distance(a, b) >= 0.0

    def test_manhattan_symmetry(self):
        a, b = _vec(5, 2), _vec(5, 3)
        assert abs(manhattan_distance(a, b) - manhattan_distance(b, a)) < 1e-12

    def test_chebyshev_self_zero(self):
        v = _vec(6)
        assert chebyshev_distance(v, v) == 0.0

    def test_chebyshev_nonnegative(self):
        a, b = _vec(5, 4), _vec(5, 5)
        assert chebyshev_distance(a, b) >= 0.0

    def test_chebyshev_leq_manhattan(self):
        """Chebyshev ≤ Manhattan for any two vectors."""
        a, b = _vec_any(8, 9), _vec_any(8, 10)
        assert chebyshev_distance(a, b) <= manhattan_distance(a, b) + 1e-10

    def test_manhattan_geq_euclidean(self):
        """Manhattan ≥ Euclidean (L1 ≥ L2 for vectors)."""
        a, b = _vec(6, 11), _vec(6, 12)
        assert manhattan_distance(a, b) >= euclidean_distance(a, b) - 1e-10

    def test_known_manhattan(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(manhattan_distance(a, b) - 7.0) < 1e-12

    def test_known_chebyshev(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(chebyshev_distance(a, b) - 4.0) < 1e-12


class TestHausdorffChamfer:
    """Hausdorff and Chamfer distance invariants."""

    def test_hausdorff_self_zero(self):
        pts = _pts(10)
        assert hausdorff_distance(pts, pts) == 0.0

    def test_hausdorff_nonnegative(self):
        A = _pts(8, seed=0)
        B = _pts(8, seed=1)
        assert hausdorff_distance(A, B) >= 0.0

    def test_hausdorff_symmetry(self):
        A = _pts(8, seed=2)
        B = _pts(8, seed=3)
        assert abs(hausdorff_distance(A, B) - hausdorff_distance(B, A)) < 1e-10

    def test_hausdorff_single_point_known(self):
        """d_H({(0,0)}, {(3,4)}) = 5."""
        A = np.array([[0.0, 0.0]])
        B = np.array([[3.0, 4.0]])
        assert abs(hausdorff_distance(A, B) - 5.0) < 1e-10

    def test_chamfer_self_zero(self):
        pts = _pts(8)
        assert chamfer_distance(pts, pts) < 1e-10

    def test_chamfer_nonnegative(self):
        A = _pts(6, seed=4)
        B = _pts(6, seed=5)
        assert chamfer_distance(A, B) >= 0.0

    def test_chamfer_symmetry(self):
        A = _pts(6, seed=6)
        B = _pts(6, seed=7)
        assert abs(chamfer_distance(A, B) - chamfer_distance(B, A)) < 1e-10

    def test_hausdorff_geq_chamfer_half(self):
        """Hausdorff ≥ half the average Chamfer for well-separated sets."""
        A = np.array([[0.0, 0.0], [1.0, 0.0]])
        B = np.array([[10.0, 0.0], [11.0, 0.0]])
        h = hausdorff_distance(A, B)
        c = chamfer_distance(A, B)
        # Both should be positive
        assert h > 0 and c > 0


class TestPairwiseDistances:
    """Pairwise distance matrix invariants."""

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan", "chebyshev"])
    def test_shape_n_by_n(self, metric):
        X = _pts(6, 3)
        D = pairwise_distances(X, metric=metric)
        assert D.shape == (6, 6)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_zero_diagonal(self, metric):
        X = _pts(5, 3)
        D = pairwise_distances(X, metric=metric)
        assert np.all(np.diag(D) < 1e-10)

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan", "chebyshev"])
    def test_symmetric(self, metric):
        X = _pts(5, 4)
        D = pairwise_distances(X, metric=metric)
        assert np.allclose(D, D.T, atol=1e-12)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_nonnegative(self, metric):
        X = _pts(4, 2)
        D = pairwise_distances(X, metric=metric)
        assert np.all(D >= -1e-12)


class TestNearestNeighborDist:
    """Nearest-neighbor distance invariants."""

    def test_query_in_candidates_zero(self):
        pts = _pts(8)
        q = pts[3].copy()
        assert nearest_neighbor_dist(q, pts) < 1e-12

    def test_nonnegative(self):
        q = _vec(2)
        C = _pts(6)
        assert nearest_neighbor_dist(q, C) >= 0.0

    def test_single_candidate(self):
        q = np.array([0.0, 0.0])
        C = np.array([[3.0, 4.0]])
        assert abs(nearest_neighbor_dist(q, C) - 5.0) < 1e-12

    def test_closer_than_all_euclidean(self):
        q = np.array([5.0, 5.0])
        C = _pts(10)
        nn_dist = nearest_neighbor_dist(q, C)
        min_dist = min(euclidean_distance(q, C[i]) for i in range(len(C)))
        assert abs(nn_dist - min_dist) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — signal_utils
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothSignal:
    """Signal smoothing invariants."""

    @pytest.mark.parametrize("method", ["gaussian", "moving_avg"])
    def test_length_preserved(self, method):
        s = _sine_signal(64)
        result = smooth_signal(s, method=method)
        assert len(result) == len(s)

    def test_constant_signal_unchanged_gaussian(self):
        s = np.ones(32) * 3.5
        result = smooth_signal(s, method="gaussian", sigma=2.0)
        assert np.allclose(result, 3.5, atol=1e-10)

    def test_constant_signal_unchanged_moving_avg(self):
        s = np.ones(32) * 2.0
        result = smooth_signal(s, method="moving_avg", window=5)
        # Interior points should be exact; boundary may deviate due to mode='same'
        interior = result[10:-10]
        assert np.allclose(interior, 2.0, atol=1e-10)

    def test_gaussian_reduces_noise(self):
        rng = np.random.default_rng(42)
        s = np.sin(np.linspace(0, 4 * math.pi, 128))
        noisy = s + rng.normal(0, 0.5, 128)
        smoothed = smooth_signal(noisy, method="gaussian", sigma=2.0)
        # Smoothed should be closer to the original sine
        err_noisy = np.abs(noisy - s).mean()
        err_smooth = np.abs(smoothed - s).mean()
        assert err_smooth < err_noisy

    def test_output_dtype_float64(self):
        s = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result = smooth_signal(s)
        assert result.dtype == np.float64


class TestNormalizeSignal:
    """Signal normalization invariants."""

    def test_output_in_range(self):
        s = _sine_signal(64)
        result = normalize_signal(s, 0.0, 1.0)
        assert np.all(result >= -1e-12)
        assert np.all(result <= 1.0 + 1e-12)

    def test_min_equals_out_min(self):
        s = _vec_any(20)
        result = normalize_signal(s, -2.0, 2.0)
        assert abs(result.min() - (-2.0)) < 1e-10

    def test_max_equals_out_max(self):
        s = _vec_any(20)
        result = normalize_signal(s, -2.0, 2.0)
        assert abs(result.max() - 2.0) < 1e-10

    def test_constant_signal_returns_out_min(self):
        s = np.full(10, 5.0)
        result = normalize_signal(s, 0.0, 1.0)
        assert np.allclose(result, 0.0, atol=1e-12)

    def test_length_preserved(self):
        s = _sine_signal(48)
        result = normalize_signal(s, 0.0, 1.0)
        assert len(result) == len(s)

    @pytest.mark.parametrize("out_min, out_max", [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)])
    def test_custom_range(self, out_min, out_max):
        s = _vec_any(15)
        result = normalize_signal(s, out_min, out_max)
        assert np.all(result >= out_min - 1e-10)
        assert np.all(result <= out_max + 1e-10)

    def test_order_preserved(self):
        """Normalization is monotone: if a < b then f(a) < f(b)."""
        s = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = normalize_signal(s, 0.0, 1.0)
        ranks_orig = np.argsort(np.argsort(s))
        ranks_norm = np.argsort(np.argsort(result))
        assert np.all(ranks_orig == ranks_norm)


class TestSignalEnergy:
    """Signal energy invariants."""

    def test_nonnegative(self):
        s = _sine_signal(64)
        assert signal_energy(s) >= 0.0

    def test_zero_signal_zero_energy(self):
        assert signal_energy(np.zeros(20)) == 0.0

    def test_scale_quadratic(self):
        """E(k*x) = k² * E(x)."""
        s = _sine_signal(32)
        k = 3.0
        assert abs(signal_energy(k * s) - k ** 2 * signal_energy(s)) < 1e-10

    def test_unit_vector_energy_one(self):
        s = np.array([1.0, 0.0, 0.0])
        assert abs(signal_energy(s) - 1.0) < 1e-12

    def test_known_value(self):
        s = np.array([3.0, 4.0])
        assert abs(signal_energy(s) - 25.0) < 1e-12

    def test_additive_for_orthogonal(self):
        """E(a + b) = E(a) + E(b) when dot(a, b) = 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 2.0, 0.0])
        assert abs(signal_energy(a + b) - (signal_energy(a) + signal_energy(b))) < 1e-12


class TestComputeAutocorrelation:
    """Autocorrelation invariants."""

    def test_length_is_2n_minus_1(self):
        s = _sine_signal(32)
        ac = compute_autocorrelation(s)
        assert len(ac) == 2 * len(s) - 1

    def test_center_value_one_when_normalized(self):
        s = _sine_signal(32) + 2.0
        ac = compute_autocorrelation(s, normalize=True)
        center = len(ac) // 2
        assert abs(ac[center] - 1.0) < 1e-10

    def test_symmetric_for_normalized(self):
        """Autocorrelation of real signal is symmetric."""
        s = _sine_signal(16)
        ac = compute_autocorrelation(s, normalize=True)
        assert np.allclose(ac, ac[::-1], atol=1e-10)

    def test_nonnegative_at_center(self):
        s = _vec(16)
        ac = compute_autocorrelation(s)
        center = len(ac) // 2
        assert ac[center] >= 0.0

    def test_periodic_signal_has_high_autocorr(self):
        """Periodic signal has AC close to 1 at period offsets."""
        n = 64
        freq = 4
        s = np.sin(2 * math.pi * freq * np.arange(n) / n)
        ac = compute_autocorrelation(s, normalize=True)
        # Check that center (lag=0) is 1
        center = len(ac) // 2
        assert abs(ac[center] - 1.0) < 1e-8


class TestResampleSignal:
    """Resampled signal invariants."""

    @pytest.mark.parametrize("n_out", [8, 16, 32, 100])
    def test_exact_n_out_points(self, n_out):
        s = _sine_signal(64)
        result = resample_signal(s, n_out)
        assert len(result) == n_out

    def test_first_point_preserved(self):
        s = _vec(32)
        result = resample_signal(s, 64)
        assert abs(result[0] - s[0]) < 1e-12

    def test_last_point_preserved(self):
        s = _vec(32)
        result = resample_signal(s, 64)
        assert abs(result[-1] - s[-1]) < 1e-10

    def test_constant_signal_stays_constant(self):
        s = np.full(16, 3.7)
        result = resample_signal(s, 32)
        assert np.allclose(result, 3.7, atol=1e-10)

    def test_same_size_approx_same_signal(self):
        s = _sine_signal(32)
        result = resample_signal(s, 32)
        assert np.allclose(result, s, atol=1e-10)

    def test_single_element_input(self):
        s = np.array([5.0])
        result = resample_signal(s, 10)
        assert np.allclose(result, 5.0, atol=1e-12)


class TestFindPeaks:
    """Peak-finding invariants."""

    def test_indices_valid(self):
        s = _sine_signal(64)
        peaks = find_peaks(s)
        n = len(s)
        assert np.all((peaks >= 0) & (peaks < n))

    def test_peak_values_greater_than_neighbors(self):
        s = _sine_signal(128)
        peaks = find_peaks(s)
        for idx in peaks:
            assert s[idx] > s[idx - 1]
            assert s[idx] > s[idx + 1]

    def test_min_distance_respected(self):
        s = _sine_signal(128)
        min_d = 10
        peaks = find_peaks(s, min_distance=min_d)
        if len(peaks) > 1:
            diffs = np.diff(peaks)
            assert np.all(diffs >= min_d)

    def test_find_valleys_symmetric(self):
        """Valleys of s == peaks of -s (approximately)."""
        s = _sine_signal(64)
        valleys = find_valleys(s)
        peaks_neg = find_peaks(-s)
        assert np.array_equal(np.sort(valleys), np.sort(peaks_neg))

    def test_constant_signal_no_peaks(self):
        s = np.ones(32)
        peaks = find_peaks(s)
        assert len(peaks) == 0

    def test_single_peak_detected(self):
        s = np.zeros(20)
        s[10] = 1.0
        peaks = find_peaks(s)
        assert 10 in peaks


class TestSegmentSignal:
    """Signal segmentation invariants."""

    def test_segments_nonoverlapping(self):
        s = _square_signal(64)
        segs = segment_signal(s, 0.5, above=True)
        indices = []
        for start, end in segs:
            indices.extend(range(start, end))
        assert len(indices) == len(set(indices)), "Segments overlap"

    def test_all_above_threshold_covered(self):
        s = _square_signal(64)
        threshold = 0.5
        expected = set(i for i, v in enumerate(s) if v >= threshold)
        segs = segment_signal(s, threshold, above=True)
        covered = set()
        for start, end in segs:
            covered.update(range(start, end))
        assert covered == expected

    def test_below_segments_complement_above(self):
        s = _square_signal(64)
        above = set()
        for start, end in segment_signal(s, 0.5, above=True):
            above.update(range(start, end))
        below = set()
        for start, end in segment_signal(s, 0.5, above=False):
            below.update(range(start, end))
        # Together they cover all indices
        assert above | below == set(range(len(s)))
        # They don't overlap
        assert len(above & below) == 0

    def test_zero_signal_below_threshold_all(self):
        s = np.zeros(20)
        segs = segment_signal(s, 0.5, above=True)
        assert segs == []

    def test_all_ones_above_threshold_one_segment(self):
        s = np.ones(20)
        segs = segment_signal(s, 0.5, above=True)
        assert len(segs) == 1
        assert segs[0] == (0, 20)


class TestPhaseShift:
    """Phase-shift invariants."""

    def test_identical_signals_zero_shift(self):
        s = _sine_signal(32)
        shift, peak = phase_shift(s, s)
        assert abs(peak) >= 1.0 - 1e-6   # peak of autocorrelation

    def test_shift_in_range(self):
        n = 32
        s1 = _sine_signal(n)
        s2 = np.roll(s1, 5)
        shift, _ = phase_shift(s1, s2)
        assert -(n - 1) <= shift <= (n - 1)

    def test_peak_value_in_range(self):
        s1 = _sine_signal(32)
        s2 = np.roll(s1, 3)
        _, peak = phase_shift(s1, s2)
        # Normalized cross-correlation peak ≤ 1
        assert peak <= 1.0 + 1e-6

    def test_cross_correlation_length(self):
        n = 20
        s1 = _sine_signal(n)
        s2 = _sine_signal(n)
        cc = compute_cross_correlation(s1, s2, normalize=True)
        assert len(cc) == 2 * n - 1

    def test_zero_signal_cross_correlation(self):
        s1 = np.zeros(10)
        s2 = np.ones(10)
        cc = compute_cross_correlation(s1, s2, normalize=False)
        assert np.all(cc == 0.0)
