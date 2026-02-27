"""Tests for puzzle_reconstruction.utils.distance_utils."""
import numpy as np
import pytest

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

np.random.seed(77)


# ── euclidean_distance ────────────────────────────────────────────────────────

def test_euclidean_same_point():
    v = np.array([1.0, 2.0, 3.0])
    assert euclidean_distance(v, v) == pytest.approx(0.0)


def test_euclidean_known_value():
    a = np.zeros(2)
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


def test_euclidean_shape_mismatch():
    with pytest.raises(ValueError):
        euclidean_distance(np.ones(3), np.ones(4))


def test_euclidean_nonneg():
    a = np.random.randn(10)
    b = np.random.randn(10)
    assert euclidean_distance(a, b) >= 0.0


# ── cosine_similarity ────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_opposite():
    v = np.array([1.0, 0.0])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_zero_vector():
    a = np.zeros(3)
    b = np.ones(3)
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_shape_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones(4))


def test_cosine_similarity_in_range():
    a = np.random.randn(8)
    b = np.random.randn(8)
    s = cosine_similarity(a, b)
    assert -1.0 - 1e-6 <= s <= 1.0 + 1e-6


# ── cosine_distance ────────────────────────────────────────────────────────────

def test_cosine_distance_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_in_range():
    a = np.random.randn(5)
    b = np.random.randn(5)
    d = cosine_distance(a, b)
    assert 0.0 - 1e-6 <= d <= 2.0 + 1e-6


# ── manhattan_distance ────────────────────────────────────────────────────────

def test_manhattan_known():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert manhattan_distance(a, b) == pytest.approx(9.0)


def test_manhattan_same():
    v = np.random.randn(5)
    assert manhattan_distance(v, v) == pytest.approx(0.0)


def test_manhattan_shape_mismatch():
    with pytest.raises(ValueError):
        manhattan_distance(np.ones(3), np.ones(5))


# ── chebyshev_distance ────────────────────────────────────────────────────────

def test_chebyshev_known():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([4.0, 2.0, 3.0])
    # |1-4|=3, |5-2|=3, |3-3|=0 → max=3
    assert chebyshev_distance(a, b) == pytest.approx(3.0)


def test_chebyshev_same():
    v = np.random.randn(4)
    assert chebyshev_distance(v, v) == pytest.approx(0.0)


def test_chebyshev_shape_mismatch():
    with pytest.raises(ValueError):
        chebyshev_distance(np.ones(3), np.ones(4))


# ── hausdorff_distance ────────────────────────────────────────────────────────

def test_hausdorff_identical():
    pts = np.random.randn(10, 2)
    assert hausdorff_distance(pts, pts) == pytest.approx(0.0, abs=1e-9)


def test_hausdorff_positive():
    A = np.zeros((5, 2))
    B = np.ones((5, 2)) * 10.0
    assert hausdorff_distance(A, B) > 0.0


def test_hausdorff_empty_raises():
    with pytest.raises(ValueError):
        hausdorff_distance(np.empty((0, 2)), np.ones((5, 2)))


def test_hausdorff_dim_mismatch():
    with pytest.raises(ValueError):
        hausdorff_distance(np.ones((5, 2)), np.ones((5, 3)))


def test_hausdorff_non_negative():
    A = np.random.randn(8, 3)
    B = np.random.randn(6, 3)
    assert hausdorff_distance(A, B) >= 0.0


# ── chamfer_distance ──────────────────────────────────────────────────────────

def test_chamfer_identical():
    pts = np.random.randn(10, 2)
    assert chamfer_distance(pts, pts) == pytest.approx(0.0, abs=1e-9)


def test_chamfer_positive():
    A = np.zeros((5, 2))
    B = np.ones((5, 2)) * 5.0
    assert chamfer_distance(A, B) > 0.0


def test_chamfer_empty_raises():
    with pytest.raises(ValueError):
        chamfer_distance(np.empty((0, 2)), np.ones((5, 2)))


def test_chamfer_non_negative():
    A = np.random.randn(6, 2)
    B = np.random.randn(4, 2)
    assert chamfer_distance(A, B) >= 0.0


# ── normalized_distance ───────────────────────────────────────────────────────

def test_normalized_zero():
    assert normalized_distance(0.0, 10.0) == pytest.approx(0.0)


def test_normalized_at_max():
    assert normalized_distance(10.0, 10.0) == pytest.approx(1.0)


def test_normalized_clipped():
    # dist > max_dist should be clipped to 1.0
    assert normalized_distance(20.0, 10.0) == pytest.approx(1.0)


def test_normalized_invalid_max():
    with pytest.raises(ValueError):
        normalized_distance(1.0, -5.0)


def test_normalized_negative_dist():
    with pytest.raises(ValueError):
        normalized_distance(-1.0, 5.0)


# ── pairwise_distances ────────────────────────────────────────────────────────

def test_pairwise_euclidean_shape():
    X = np.random.randn(5, 4)
    D = pairwise_distances(X, metric="euclidean")
    assert D.shape == (5, 5)


def test_pairwise_euclidean_symmetric():
    X = np.random.randn(4, 3)
    D = pairwise_distances(X)
    assert np.allclose(D, D.T)


def test_pairwise_diagonal_zero():
    X = np.random.randn(4, 3)
    D = pairwise_distances(X)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)


def test_pairwise_invalid_metric():
    X = np.random.randn(3, 2)
    with pytest.raises(ValueError):
        pairwise_distances(X, metric="unknown")


def test_pairwise_not_2d():
    with pytest.raises(ValueError):
        pairwise_distances(np.ones(5))


# ── nearest_neighbor_dist ─────────────────────────────────────────────────────

def test_nn_dist_self():
    q = np.array([3.0, 4.0])
    cands = np.array([[3.0, 4.0], [10.0, 10.0]])
    assert nearest_neighbor_dist(q, cands) == pytest.approx(0.0)


def test_nn_dist_positive():
    q = np.zeros(2)
    cands = np.ones((4, 2)) * 5.0
    d = nearest_neighbor_dist(q, cands)
    assert d > 0.0


def test_nn_dist_empty_raises():
    with pytest.raises(ValueError):
        nearest_neighbor_dist(np.ones(2), np.empty((0, 2)))


def test_nn_dist_not_2d():
    with pytest.raises(ValueError):
        nearest_neighbor_dist(np.ones(2), np.ones(4))
