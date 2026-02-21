"""Tests for puzzle_reconstruction.utils.distance_utils."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.distance_utils import (
    chamfer_distance,
    chebyshev_distance,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    hausdorff_distance,
    manhattan_distance,
    nearest_neighbor_dist,
    normalized_distance,
    pairwise_distances,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _v(lst):
    return np.array(lst, dtype=np.float64)


def _pts(*rows):
    return np.array(rows, dtype=np.float64)


# ─── euclidean_distance ───────────────────────────────────────────────────────

class TestEuclideanDistance:
    def test_known_value(self):
        assert euclidean_distance(_v([0, 0]), _v([3, 4])) == pytest.approx(5.0)

    def test_identical_zero(self):
        a = _v([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_nonnegative(self):
        assert euclidean_distance(_v([1, 2]), _v([4, 6])) >= 0.0

    def test_symmetric(self):
        a, b = _v([1, 2, 3]), _v([4, 5, 6])
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            euclidean_distance(_v([1, 2]), _v([1, 2, 3]))

    def test_1d_vectors(self):
        assert euclidean_distance(_v([0.0]), _v([5.0])) == pytest.approx(5.0)

    def test_returns_float(self):
        assert isinstance(euclidean_distance(_v([1, 2]), _v([3, 4])), float)


# ─── cosine_similarity ────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_one(self):
        a = _v([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_opposite_vectors_minus_one(self):
        a = _v([1.0, 0.0])
        b = _v([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_zero(self):
        assert cosine_similarity(_v([1.0, 0.0]), _v([0.0, 1.0])) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity(_v([0.0, 0.0]), _v([1.0, 2.0])) == pytest.approx(0.0)

    def test_range(self):
        a, b = _v([1, 2, 3]), _v([4, 5, 6])
        s = cosine_similarity(a, b)
        assert -1.0 <= s <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity(_v([1, 2]), _v([1, 2, 3]))


# ─── cosine_distance ──────────────────────────────────────────────────────────

class TestCosineDistance:
    def test_identical_zero(self):
        a = _v([1.0, 2.0])
        assert cosine_distance(a, a) == pytest.approx(0.0)

    def test_opposite_two(self):
        assert cosine_distance(_v([1.0, 0.0]), _v([-1.0, 0.0])) == pytest.approx(2.0)

    def test_range(self):
        a, b = _v([1, 2, 3]), _v([4, 5, 6])
        d = cosine_distance(a, b)
        assert 0.0 <= d <= 2.0

    def test_nonnegative(self):
        assert cosine_distance(_v([1, -1]), _v([1, 1])) >= 0.0

    def test_complementary_to_similarity(self):
        a, b = _v([1, 2, 3]), _v([4, 5, 6])
        assert cosine_distance(a, b) == pytest.approx(1.0 - cosine_similarity(a, b))


# ─── manhattan_distance ───────────────────────────────────────────────────────

class TestManhattanDistance:
    def test_known_value(self):
        assert manhattan_distance(_v([0, 0]), _v([3, 4])) == pytest.approx(7.0)

    def test_identical_zero(self):
        a = _v([1.0, 2.0])
        assert manhattan_distance(a, a) == pytest.approx(0.0)

    def test_nonnegative(self):
        assert manhattan_distance(_v([1, 2]), _v([4, 6])) >= 0.0

    def test_symmetric(self):
        a, b = _v([1, 2, 3]), _v([4, 5, 6])
        assert manhattan_distance(a, b) == pytest.approx(manhattan_distance(b, a))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            manhattan_distance(_v([1, 2]), _v([1, 2, 3]))

    def test_ge_euclidean(self):
        a, b = _v([1.0, 2.0]), _v([4.0, 6.0])
        assert manhattan_distance(a, b) >= euclidean_distance(a, b)


# ─── chebyshev_distance ───────────────────────────────────────────────────────

class TestChebyshevDistance:
    def test_known_value(self):
        assert chebyshev_distance(_v([0, 0]), _v([3, 4])) == pytest.approx(4.0)

    def test_identical_zero(self):
        a = _v([5.0, 5.0])
        assert chebyshev_distance(a, a) == pytest.approx(0.0)

    def test_nonnegative(self):
        assert chebyshev_distance(_v([1, 2]), _v([4, 6])) >= 0.0

    def test_le_manhattan(self):
        a, b = _v([1.0, 2.0, 3.0]), _v([4.0, 6.0, 8.0])
        assert chebyshev_distance(a, b) <= manhattan_distance(a, b)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            chebyshev_distance(_v([1, 2]), _v([1, 2, 3]))


# ─── hausdorff_distance ───────────────────────────────────────────────────────

class TestHausdorffDistance:
    def test_identical_sets_zero(self):
        A = _pts([0, 0], [1, 1])
        assert hausdorff_distance(A, A) == pytest.approx(0.0)

    def test_known_value(self):
        A = _pts([0.0, 0.0])
        B = _pts([3.0, 4.0])
        assert hausdorff_distance(A, B) == pytest.approx(5.0)

    def test_symmetric(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([5, 0])
        assert hausdorff_distance(A, B) == pytest.approx(hausdorff_distance(B, A))

    def test_nonnegative(self):
        A = _pts([0, 0], [1, 1])
        B = _pts([2, 2], [3, 3])
        assert hausdorff_distance(A, B) >= 0.0

    def test_empty_a_raises(self):
        with pytest.raises(ValueError):
            hausdorff_distance(np.zeros((0, 2)), _pts([1, 1]))

    def test_empty_b_raises(self):
        with pytest.raises(ValueError):
            hausdorff_distance(_pts([1, 1]), np.zeros((0, 2)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            hausdorff_distance(np.zeros((3,)), _pts([1, 1]))

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            hausdorff_distance(_pts([0, 0]), np.array([[1, 2, 3]]))


# ─── chamfer_distance ────────────────────────────────────────────────────────

class TestChamferDistance:
    def test_identical_sets_zero(self):
        A = _pts([0, 0], [1, 1])
        assert chamfer_distance(A, A) == pytest.approx(0.0)

    def test_nonnegative(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([2, 0], [3, 0])
        assert chamfer_distance(A, B) >= 0.0

    def test_symmetric(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([5, 0])
        assert chamfer_distance(A, B) == pytest.approx(chamfer_distance(B, A))

    def test_le_hausdorff(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([5, 0], [6, 0])
        assert chamfer_distance(A, B) <= hausdorff_distance(A, B) + 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            chamfer_distance(np.zeros((0, 2)), _pts([1, 1]))

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            chamfer_distance(_pts([0, 0]), np.array([[1, 2, 3]]))


# ─── normalized_distance ─────────────────────────────────────────────────────

class TestNormalizedDistance:
    def test_zero_maps_to_zero(self):
        assert normalized_distance(0.0, 10.0) == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        assert normalized_distance(10.0, 10.0) == pytest.approx(1.0)

    def test_half(self):
        assert normalized_distance(5.0, 10.0) == pytest.approx(0.5)

    def test_clamped_above_one(self):
        assert normalized_distance(15.0, 10.0) == pytest.approx(1.0)

    def test_negative_dist_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(-1.0, 10.0)

    def test_zero_max_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(5.0, 0.0)

    def test_negative_max_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(5.0, -1.0)


# ─── pairwise_distances ───────────────────────────────────────────────────────

class TestPairwiseDistances:
    def test_returns_float64(self):
        X = np.random.default_rng(0).random((4, 3))
        D = pairwise_distances(X)
        assert D.dtype == np.float64

    def test_shape_n_n(self):
        X = np.random.default_rng(0).random((5, 2))
        D = pairwise_distances(X)
        assert D.shape == (5, 5)

    def test_diagonal_zero(self):
        X = np.random.default_rng(0).random((4, 3))
        D = pairwise_distances(X)
        assert np.all(np.diag(D) == pytest.approx(0.0))

    def test_symmetric(self):
        X = np.random.default_rng(0).random((4, 3))
        D = pairwise_distances(X)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_cosine_metric(self):
        X = np.random.default_rng(1).random((3, 4))
        D = pairwise_distances(X, metric="cosine")
        assert D.shape == (3, 3)

    def test_manhattan_metric(self):
        X = np.ones((3, 2))
        D = pairwise_distances(X, metric="manhattan")
        assert np.all(D == pytest.approx(0.0))

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.ones((3, 2)), metric="minkowski")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.ones((3,)))


# ─── nearest_neighbor_dist ───────────────────────────────────────────────────

class TestNearestNeighborDist:
    def test_exact_match_zero(self):
        candidates = _pts([1.0, 2.0], [3.0, 4.0])
        assert nearest_neighbor_dist(_v([1.0, 2.0]), candidates) == pytest.approx(0.0)

    def test_known_value(self):
        candidates = _pts([3.0, 4.0])
        assert nearest_neighbor_dist(_v([0.0, 0.0]), candidates) == pytest.approx(5.0)

    def test_nonnegative(self):
        candidates = _pts([1.0, 0.0], [2.0, 0.0])
        assert nearest_neighbor_dist(_v([0.0, 0.0]), candidates) >= 0.0

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            nearest_neighbor_dist(_v([1.0, 2.0]), np.zeros((0, 2)))

    def test_non_2d_candidates_raises(self):
        with pytest.raises(ValueError):
            nearest_neighbor_dist(_v([1.0, 2.0]), np.zeros((3,)))

    def test_returns_float(self):
        candidates = _pts([1.0, 2.0])
        assert isinstance(nearest_neighbor_dist(_v([0.0, 0.0]), candidates), float)
