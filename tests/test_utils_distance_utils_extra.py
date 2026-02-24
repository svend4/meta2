"""Extra tests for puzzle_reconstruction/utils/distance_utils.py"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.distance_utils import (
    euclidean_distance,
    cosine_similarity,
    cosine_distance,
    manhattan_distance,
    chebyshev_distance,
    hausdorff_distance,
    chamfer_distance,
    normalized_distance,
    pairwise_distances,
    nearest_neighbor_dist,
)


# ─── euclidean_distance ───────────────────────────────────────────────────────

class TestEuclideanDistance:
    def test_zero_distance(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_known_value_3_4_5(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_orthogonal_unit_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert euclidean_distance(a, b) == pytest.approx(np.sqrt(2.0))

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            euclidean_distance(a, b)

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert isinstance(euclidean_distance(a, b), float)

    def test_nonneg(self):
        a = np.array([5.0, -3.0])
        b = np.array([1.0, 2.0])
        assert euclidean_distance(a, b) >= 0.0


# ─── cosine_similarity ────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0])
        with pytest.raises(ValueError):
            cosine_similarity(a, b)

    def test_result_in_minus_one_one(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert isinstance(cosine_similarity(a, b), float)


# ─── cosine_distance ──────────────────────────────────────────────────────────

class TestCosineDistance:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0])
        assert cosine_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0)

    def test_nonneg(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert cosine_distance(a, b) >= 0.0

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert isinstance(cosine_distance(a, b), float)

    def test_is_one_minus_similarity(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        expected = 1.0 - cosine_similarity(a, b)
        assert cosine_distance(a, b) == pytest.approx(expected, abs=1e-9)


# ─── manhattan_distance ───────────────────────────────────────────────────────

class TestManhattanDistance:
    def test_zero_distance(self):
        a = np.array([1.0, 2.0])
        assert manhattan_distance(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert manhattan_distance(a, b) == pytest.approx(2.0)

    def test_3d_vector(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 9.0])
        assert manhattan_distance(a, b) == pytest.approx(3.0 + 4.0 + 6.0)

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0])
        with pytest.raises(ValueError):
            manhattan_distance(a, b)

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert isinstance(manhattan_distance(a, b), float)

    def test_nonneg(self):
        a = np.array([-1.0, -2.0])
        b = np.array([3.0, 4.0])
        assert manhattan_distance(a, b) >= 0.0


# ─── chebyshev_distance ───────────────────────────────────────────────────────

class TestChebyshevDistance:
    def test_zero_distance(self):
        a = np.array([1.0, 2.0])
        assert chebyshev_distance(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 3.0])
        assert chebyshev_distance(a, b) == pytest.approx(3.0)

    def test_max_of_abs_diffs(self):
        a = np.array([2.0, 5.0, 1.0])
        b = np.array([0.0, 2.0, 4.0])
        assert chebyshev_distance(a, b) == pytest.approx(3.0)

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            chebyshev_distance(a, b)

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert isinstance(chebyshev_distance(a, b), float)

    def test_nonneg(self):
        a = np.array([-5.0, 2.0])
        b = np.array([3.0, -1.0])
        assert chebyshev_distance(a, b) >= 0.0


# ─── hausdorff_distance ───────────────────────────────────────────────────────

class TestHausdorffDistance:
    def test_identical_sets(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert hausdorff_distance(pts, pts) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        assert hausdorff_distance(a, b) == pytest.approx(5.0)

    def test_nonneg(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([[0.0, 1.0], [1.0, 1.0]])
        assert hausdorff_distance(a, b) >= 0.0

    def test_empty_set_raises(self):
        a = np.zeros((0, 2))
        b = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            hausdorff_distance(a, b)

    def test_dimension_mismatch_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            hausdorff_distance(a, b)

    def test_returns_float(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        assert isinstance(hausdorff_distance(a, b), float)


# ─── chamfer_distance ─────────────────────────────────────────────────────────

class TestChamferDistance:
    def test_identical_sets(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert chamfer_distance(pts, pts) == pytest.approx(0.0, abs=1e-9)

    def test_nonneg(self):
        a = np.array([[0.0, 0.0], [2.0, 0.0]])
        b = np.array([[0.0, 1.0], [2.0, 1.0]])
        assert chamfer_distance(a, b) >= 0.0

    def test_known_single_point(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        # Both terms equal 5.0, chamfer = 5+5 = 10.0
        assert chamfer_distance(a, b) == pytest.approx(10.0)

    def test_empty_set_raises(self):
        a = np.zeros((0, 2))
        b = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            chamfer_distance(a, b)

    def test_dimension_mismatch_raises(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            chamfer_distance(a, b)

    def test_returns_float(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        assert isinstance(chamfer_distance(a, b), float)


# ─── normalized_distance ──────────────────────────────────────────────────────

class TestNormalizedDistance:
    def test_zero_distance(self):
        assert normalized_distance(0.0, 10.0) == pytest.approx(0.0)

    def test_max_distance(self):
        assert normalized_distance(10.0, 10.0) == pytest.approx(1.0)

    def test_half(self):
        assert normalized_distance(5.0, 10.0) == pytest.approx(0.5)

    def test_clamps_above_one(self):
        assert normalized_distance(20.0, 10.0) == pytest.approx(1.0)

    def test_negative_dist_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(-1.0, 10.0)

    def test_zero_max_dist_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(1.0, 0.0)

    def test_negative_max_dist_raises(self):
        with pytest.raises(ValueError):
            normalized_distance(1.0, -5.0)

    def test_returns_float(self):
        assert isinstance(normalized_distance(1.0, 10.0), float)


# ─── pairwise_distances ───────────────────────────────────────────────────────

class TestPairwiseDistances:
    def test_shape(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = pairwise_distances(X)
        assert D.shape == (3, 3)

    def test_diagonal_zeros(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        D = pairwise_distances(X)
        np.testing.assert_allclose(np.diag(D), np.zeros(2), atol=1e-9)

    def test_symmetry(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = pairwise_distances(X)
        np.testing.assert_allclose(D, D.T, atol=1e-9)

    def test_euclidean_metric(self):
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(X, metric="euclidean")
        assert D[0, 1] == pytest.approx(5.0)

    def test_manhattan_metric(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = pairwise_distances(X, metric="manhattan")
        assert D[0, 1] == pytest.approx(2.0)

    def test_unknown_metric_raises(self):
        X = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError):
            pairwise_distances(X, metric="unknown")

    def test_non_2d_raises(self):
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            pairwise_distances(X)


# ─── nearest_neighbor_dist ────────────────────────────────────────────────────

class TestNearestNeighborDist:
    def test_exact_match(self):
        query = np.array([1.0, 0.0])
        candidates = np.array([[1.0, 0.0], [2.0, 3.0]])
        assert nearest_neighbor_dist(query, candidates) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        query = np.array([0.0, 0.0])
        candidates = np.array([[3.0, 4.0], [1.0, 0.0]])
        assert nearest_neighbor_dist(query, candidates) == pytest.approx(1.0)

    def test_empty_candidates_raises(self):
        query = np.array([0.0, 0.0])
        candidates = np.zeros((0, 2))
        with pytest.raises(ValueError):
            nearest_neighbor_dist(query, candidates)

    def test_nonneg(self):
        query = np.array([0.5, 0.5])
        candidates = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert nearest_neighbor_dist(query, candidates) >= 0.0

    def test_returns_float(self):
        query = np.array([0.0, 0.0])
        candidates = np.array([[1.0, 0.0]])
        assert isinstance(nearest_neighbor_dist(query, candidates), float)

    def test_single_candidate(self):
        query = np.array([0.0, 0.0])
        candidates = np.array([[3.0, 4.0]])
        assert nearest_neighbor_dist(query, candidates) == pytest.approx(5.0)
