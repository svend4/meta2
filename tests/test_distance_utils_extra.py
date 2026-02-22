"""Extra tests for puzzle_reconstruction.utils.distance_utils."""
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


def _v(lst):
    return np.array(lst, dtype=np.float64)


def _pts(*rows):
    return np.array(rows, dtype=np.float64)


# ─── TestEuclideanDistanceExtra ───────────────────────────────────────────────

class TestEuclideanDistanceExtra:
    def test_known_3_4_5(self):
        assert euclidean_distance(_v([0, 0]), _v([3, 4])) == pytest.approx(5.0)

    def test_identical_zero(self):
        a = _v([7, 8, 9])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_nonneg(self):
        assert euclidean_distance(_v([1, 2]), _v([-3, -4])) >= 0.0

    def test_symmetric(self):
        a, b = _v([1, 2]), _v([5, 6])
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            euclidean_distance(_v([1]), _v([1, 2]))

    def test_returns_float(self):
        assert isinstance(euclidean_distance(_v([0, 0]), _v([1, 1])), float)

    def test_unit_vector(self):
        assert euclidean_distance(_v([0]), _v([1])) == pytest.approx(1.0)

    def test_high_dim(self):
        a = np.zeros(100)
        b = np.ones(100)
        assert euclidean_distance(a, b) == pytest.approx(10.0)


# ─── TestCosineSimilarityExtra ────────────────────────────────────────────────

class TestCosineSimilarityExtra:
    def test_identical_one(self):
        a = _v([3, 4])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_opposite_minus_one(self):
        assert cosine_similarity(_v([1, 0]), _v([-1, 0])) == pytest.approx(-1.0)

    def test_orthogonal_zero(self):
        assert cosine_similarity(_v([1, 0]), _v([0, 1])) == pytest.approx(0.0)

    def test_zero_vector_zero(self):
        assert cosine_similarity(_v([0, 0]), _v([1, 2])) == pytest.approx(0.0)

    def test_range(self):
        s = cosine_similarity(_v([1, 2, 3]), _v([4, -5, 6]))
        assert -1.0 <= s <= 1.0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            cosine_similarity(_v([1, 2]), _v([1, 2, 3]))

    def test_positive_vectors(self):
        s = cosine_similarity(_v([1, 2, 3]), _v([4, 5, 6]))
        assert s > 0.9  # nearly parallel


# ─── TestCosineDistanceExtra ──────────────────────────────────────────────────

class TestCosineDistanceExtra:
    def test_identical_zero(self):
        assert cosine_distance(_v([3, 4]), _v([3, 4])) == pytest.approx(0.0)

    def test_opposite_two(self):
        assert cosine_distance(_v([1, 0]), _v([-1, 0])) == pytest.approx(2.0)

    def test_range(self):
        d = cosine_distance(_v([1, 2]), _v([3, 4]))
        assert 0.0 <= d <= 2.0

    def test_nonneg(self):
        assert cosine_distance(_v([1, -1]), _v([1, 1])) >= 0.0

    def test_complement(self):
        a, b = _v([1, 2, 3]), _v([4, 5, 6])
        assert cosine_distance(a, b) == pytest.approx(1.0 - cosine_similarity(a, b))


# ─── TestManhattanDistanceExtra ───────────────────────────────────────────────

class TestManhattanDistanceExtra:
    def test_known(self):
        assert manhattan_distance(_v([0, 0]), _v([3, 4])) == pytest.approx(7.0)

    def test_identical_zero(self):
        assert manhattan_distance(_v([5, 5]), _v([5, 5])) == pytest.approx(0.0)

    def test_nonneg(self):
        assert manhattan_distance(_v([1, 2]), _v([-3, -4])) >= 0.0

    def test_symmetric(self):
        a, b = _v([1, 2, 3]), _v([7, 8, 9])
        assert manhattan_distance(a, b) == pytest.approx(manhattan_distance(b, a))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            manhattan_distance(_v([1, 2]), _v([1]))

    def test_ge_euclidean(self):
        a, b = _v([1, 2]), _v([4, 6])
        assert manhattan_distance(a, b) >= euclidean_distance(a, b)


# ─── TestChebyshevDistanceExtra ───────────────────────────────────────────────

class TestChebyshevDistanceExtra:
    def test_known(self):
        assert chebyshev_distance(_v([0, 0]), _v([3, 4])) == pytest.approx(4.0)

    def test_identical_zero(self):
        assert chebyshev_distance(_v([7, 7]), _v([7, 7])) == pytest.approx(0.0)

    def test_nonneg(self):
        assert chebyshev_distance(_v([0, 0]), _v([-5, 3])) >= 0.0

    def test_le_manhattan(self):
        a, b = _v([1, 2, 3]), _v([4, 6, 8])
        assert chebyshev_distance(a, b) <= manhattan_distance(a, b)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            chebyshev_distance(_v([1, 2]), _v([1]))


# ─── TestHausdorffDistanceExtra ───────────────────────────────────────────────

class TestHausdorffDistanceExtra:
    def test_identical_zero(self):
        A = _pts([0, 0], [1, 1])
        assert hausdorff_distance(A, A) == pytest.approx(0.0)

    def test_known(self):
        assert hausdorff_distance(_pts([0, 0]), _pts([3, 4])) == pytest.approx(5.0)

    def test_symmetric(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([5, 0])
        assert hausdorff_distance(A, B) == pytest.approx(hausdorff_distance(B, A))

    def test_nonneg(self):
        assert hausdorff_distance(_pts([0, 0]), _pts([1, 1])) >= 0.0

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


# ─── TestChamferDistanceExtra ─────────────────────────────────────────────────

class TestChamferDistanceExtra:
    def test_identical_zero(self):
        A = _pts([0, 0], [1, 1])
        assert chamfer_distance(A, A) == pytest.approx(0.0)

    def test_nonneg(self):
        assert chamfer_distance(_pts([0, 0]), _pts([5, 5])) >= 0.0

    def test_symmetric(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([5, 0])
        assert chamfer_distance(A, B) == pytest.approx(chamfer_distance(B, A))

    def test_chamfer_hausdorff_both_positive(self):
        A = _pts([0, 0], [1, 0])
        B = _pts([5, 0], [6, 0])
        assert chamfer_distance(A, B) > 0.0
        assert hausdorff_distance(A, B) > 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            chamfer_distance(np.zeros((0, 2)), _pts([1, 1]))


# ─── TestNormalizedDistanceExtra ──────────────────────────────────────────────

class TestNormalizedDistanceExtra:
    def test_zero_maps_zero(self):
        assert normalized_distance(0.0, 10.0) == pytest.approx(0.0)

    def test_max_maps_one(self):
        assert normalized_distance(10.0, 10.0) == pytest.approx(1.0)

    def test_half(self):
        assert normalized_distance(5.0, 10.0) == pytest.approx(0.5)

    def test_clamped(self):
        assert normalized_distance(20.0, 10.0) == pytest.approx(1.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(-1.0, 10.0)

    def test_zero_max_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(5.0, 0.0)

    def test_negative_max_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(5.0, -1.0)


# ─── TestPairwiseDistancesExtra ───────────────────────────────────────────────

class TestPairwiseDistancesExtra:
    def test_shape(self):
        X = np.random.default_rng(0).random((5, 3))
        assert pairwise_distances(X).shape == (5, 5)

    def test_diagonal_zero(self):
        X = np.random.default_rng(0).random((4, 3))
        D = pairwise_distances(X)
        assert np.all(np.diag(D) == pytest.approx(0.0))

    def test_symmetric(self):
        X = np.random.default_rng(0).random((4, 3))
        D = pairwise_distances(X)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_dtype(self):
        X = np.random.default_rng(0).random((3, 2))
        assert pairwise_distances(X).dtype == np.float64

    def test_cosine_metric(self):
        X = np.random.default_rng(0).random((3, 4))
        assert pairwise_distances(X, metric="cosine").shape == (3, 3)

    def test_manhattan_metric(self):
        X = np.ones((3, 2))
        D = pairwise_distances(X, metric="manhattan")
        assert np.all(D == pytest.approx(0.0))

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.ones((3, 2)), metric="minkowski")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.ones((3,)))


# ─── TestNearestNeighborDistExtra ─────────────────────────────────────────────

class TestNearestNeighborDistExtra:
    def test_exact_match_zero(self):
        cand = _pts([1, 2], [3, 4])
        assert nearest_neighbor_dist(_v([1, 2]), cand) == pytest.approx(0.0)

    def test_known(self):
        assert nearest_neighbor_dist(_v([0, 0]), _pts([3, 4])) == pytest.approx(5.0)

    def test_nonneg(self):
        assert nearest_neighbor_dist(_v([0, 0]), _pts([1, 0], [2, 0])) >= 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nearest_neighbor_dist(_v([1, 2]), np.zeros((0, 2)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            nearest_neighbor_dist(_v([1, 2]), np.zeros((3,)))

    def test_returns_float(self):
        assert isinstance(nearest_neighbor_dist(_v([0, 0]), _pts([1, 1])), float)

    def test_picks_nearest(self):
        cand = _pts([10, 0], [1, 0])
        assert nearest_neighbor_dist(_v([0, 0]), cand) == pytest.approx(1.0)
