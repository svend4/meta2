"""Tests for puzzle_reconstruction.utils.clustering_utils."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.clustering_utils import (
    ClusterResult,
    assign_to_clusters,
    cluster_indices,
    compute_inertia,
    find_optimal_k,
    hierarchical_cluster,
    kmeans_cluster,
    merge_clusters,
    silhouette_score_approx,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _blobs(n_per_cluster: int = 10, n_clusters: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 10, (n_clusters, 2))
    pts = np.vstack([
        rng.normal(c, 0.3, (n_per_cluster, 2)) for c in centers
    ])
    return pts.astype(np.float64)


def _dist_mat(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.uniform(0, 1, (n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    return M


# ─── ClusterResult ───────────────────────────────────────────────────────────

class TestClusterResult:
    def test_fields_stored(self):
        r = ClusterResult(
            labels=np.array([0, 1, 0]),
            centers=np.zeros((2, 2)),
            n_clusters=2,
            inertia=0.5,
        )
        assert r.n_clusters == 2
        assert r.inertia == pytest.approx(0.5)

    def test_n_clusters_zero_raises(self):
        with pytest.raises(ValueError):
            ClusterResult(
                labels=np.array([0]),
                centers=np.zeros((1, 2)),
                n_clusters=0,
                inertia=0.0,
            )

    def test_negative_inertia_raises(self):
        with pytest.raises(ValueError):
            ClusterResult(
                labels=np.array([0]),
                centers=np.zeros((1, 2)),
                n_clusters=1,
                inertia=-0.1,
            )

    def test_len(self):
        r = ClusterResult(
            labels=np.array([0, 1, 2]),
            centers=np.zeros((3, 2)),
            n_clusters=3,
            inertia=0.0,
        )
        assert len(r) == 3

    def test_default_params_empty(self):
        r = ClusterResult(
            labels=np.array([0]),
            centers=np.zeros((1, 2)),
            n_clusters=1,
            inertia=0.0,
        )
        assert r.params == {}


# ─── kmeans_cluster ──────────────────────────────────────────────────────────

class TestKmeansCluster:
    def test_returns_cluster_result(self):
        X = _blobs(10, 3)
        result = kmeans_cluster(X, n_clusters=3)
        assert isinstance(result, ClusterResult)

    def test_labels_shape(self):
        X = _blobs(10, 3)
        result = kmeans_cluster(X, n_clusters=3)
        assert result.labels.shape == (30,)

    def test_labels_in_range(self):
        X = _blobs(10, 3)
        result = kmeans_cluster(X, n_clusters=3)
        assert result.labels.min() >= 0
        assert result.labels.max() < 3

    def test_n_clusters_correct(self):
        X = _blobs(10, 2)
        result = kmeans_cluster(X, n_clusters=2)
        assert result.n_clusters == 2

    def test_inertia_nonnegative(self):
        X = _blobs(10, 3)
        result = kmeans_cluster(X, n_clusters=3)
        assert result.inertia >= 0.0

    def test_invalid_n_clusters_zero_raises(self):
        X = _blobs(5, 2)
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=0)

    def test_max_iter_below_one_raises(self):
        X = _blobs(5, 2)
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=2, max_iter=0)

    def test_single_cluster(self):
        X = _blobs(5, 1)
        result = kmeans_cluster(X, n_clusters=1)
        assert result.n_clusters == 1
        assert np.all(result.labels == 0)

    def test_reproducible_with_same_seed(self):
        X = _blobs(10, 3)
        r1 = kmeans_cluster(X, n_clusters=3, random_state=42)
        r2 = kmeans_cluster(X, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_params_stored(self):
        X = _blobs(5, 2)
        result = kmeans_cluster(X, n_clusters=2)
        assert result.params.get("algorithm") == "kmeans"


# ─── assign_to_clusters ──────────────────────────────────────────────────────

class TestAssignToClusters:
    def test_returns_int64(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert labels.dtype == np.int64

    def test_shape_n(self):
        X = np.zeros((5, 3))
        centers = np.zeros((2, 3))
        labels = assign_to_clusters(X, centers)
        assert labels.shape == (5,)

    def test_nearest_center_assigned(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert labels[0] == 0
        assert labels[1] == 1

    def test_dim_mismatch_raises(self):
        X = np.zeros((5, 3))
        centers = np.zeros((2, 4))
        with pytest.raises(ValueError):
            assign_to_clusters(X, centers)

    def test_non_2d_x_raises(self):
        with pytest.raises(ValueError):
            assign_to_clusters(np.zeros((5,)), np.zeros((2, 2)))

    def test_non_2d_centers_raises(self):
        with pytest.raises(ValueError):
            assign_to_clusters(np.zeros((5, 2)), np.zeros((2,)))


# ─── compute_inertia ─────────────────────────────────────────────────────────

class TestComputeInertia:
    def test_zero_inertia_at_centers(self):
        X = np.array([[1.0, 0.0], [3.0, 0.0]])
        labels = np.array([0, 1])
        centers = np.array([[1.0, 0.0], [3.0, 0.0]])
        assert compute_inertia(X, labels, centers) == pytest.approx(0.0)

    def test_nonnegative(self):
        X = _blobs(10, 3)
        result = kmeans_cluster(X, n_clusters=3)
        inertia = compute_inertia(X, result.labels, result.centers)
        assert inertia >= 0.0

    def test_known_value(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0]])
        labels = np.array([0, 0])
        centers = np.array([[1.0, 0.0]])
        assert compute_inertia(X, labels, centers) == pytest.approx(2.0)

    def test_returns_float(self):
        X = _blobs(5, 2)
        labels = np.zeros(10, dtype=np.int64)
        centers = np.zeros((1, 2))
        assert isinstance(compute_inertia(X, labels, centers), float)


# ─── silhouette_score_approx ─────────────────────────────────────────────────

class TestSilhouetteScoreApprox:
    def test_range(self):
        X = _blobs(10, 3)
        result = kmeans_cluster(X, n_clusters=3)
        score = silhouette_score_approx(X, result.labels)
        assert -1.0 <= score <= 1.0

    def test_single_cluster_returns_zero(self):
        X = _blobs(10, 1)
        labels = np.zeros(10, dtype=np.int64)
        assert silhouette_score_approx(X, labels) == pytest.approx(0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            silhouette_score_approx(np.zeros((5,)), np.zeros(5, dtype=np.int64))

    def test_well_separated_clusters_positive(self):
        X = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 100])
        labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        score = silhouette_score_approx(X, labels)
        assert score > 0.0

    def test_returns_float(self):
        X = _blobs(5, 2)
        result = kmeans_cluster(X, n_clusters=2)
        assert isinstance(silhouette_score_approx(X, result.labels), float)


# ─── hierarchical_cluster ────────────────────────────────────────────────────

class TestHierarchicalCluster:
    def test_returns_int64(self):
        labels = hierarchical_cluster(_dist_mat(), n_clusters=2)
        assert labels.dtype == np.int64

    def test_shape_n(self):
        labels = hierarchical_cluster(_dist_mat(8), n_clusters=3)
        assert labels.shape == (8,)

    def test_n_unique_labels(self):
        labels = hierarchical_cluster(_dist_mat(6), n_clusters=2)
        assert len(np.unique(labels)) == 2

    def test_single_cluster(self):
        labels = hierarchical_cluster(_dist_mat(), n_clusters=1)
        assert len(np.unique(labels)) == 1

    def test_invalid_linkage_raises(self):
        with pytest.raises(ValueError):
            hierarchical_cluster(_dist_mat(), n_clusters=2, linkage="ward")

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            hierarchical_cluster(np.zeros((3, 4)), n_clusters=2)

    def test_complete_linkage(self):
        labels = hierarchical_cluster(_dist_mat(), n_clusters=2, linkage="complete")
        assert labels.shape == (5,)

    def test_average_linkage(self):
        labels = hierarchical_cluster(_dist_mat(), n_clusters=2, linkage="average")
        assert labels.shape == (5,)


# ─── find_optimal_k ──────────────────────────────────────────────────────────

class TestFindOptimalK:
    def test_returns_int(self):
        X = _blobs(10, 3)
        k = find_optimal_k(X, k_min=2, k_max=5)
        assert isinstance(k, int)

    def test_in_range(self):
        X = _blobs(10, 3)
        k = find_optimal_k(X, k_min=2, k_max=5)
        assert 2 <= k <= 5

    def test_k_min_above_k_max_raises(self):
        X = _blobs(10, 3)
        with pytest.raises(ValueError):
            find_optimal_k(X, k_min=5, k_max=2)

    def test_k_min_below_one_raises(self):
        X = _blobs(10, 3)
        with pytest.raises(ValueError):
            find_optimal_k(X, k_min=0)

    def test_single_k_returns_k_min(self):
        X = _blobs(5, 2)
        k = find_optimal_k(X, k_min=2, k_max=2)
        assert k == 2


# ─── cluster_indices ─────────────────────────────────────────────────────────

class TestClusterIndices:
    def test_returns_dict(self):
        labels = np.array([0, 1, 0, 2])
        result = cluster_indices(labels)
        assert isinstance(result, dict)

    def test_all_clusters_present(self):
        labels = np.array([0, 1, 0, 2])
        result = cluster_indices(labels)
        assert set(result.keys()) == {0, 1, 2}

    def test_indices_correct(self):
        labels = np.array([0, 1, 0])
        result = cluster_indices(labels)
        assert result[0] == [0, 2]
        assert result[1] == [1]

    def test_n_clusters_includes_empty_cluster(self):
        labels = np.array([0, 2])
        result = cluster_indices(labels, n_clusters=3)
        assert 1 in result
        assert result[1] == []

    def test_indices_sorted(self):
        labels = np.array([1, 0, 1, 0])
        result = cluster_indices(labels)
        assert result[0] == sorted(result[0])


# ─── merge_clusters ──────────────────────────────────────────────────────────

class TestMergeClusters:
    def test_returns_int64(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        result = merge_clusters(labels, [1, 2], target_id=0)
        assert result.dtype == np.int64

    def test_shape_preserved(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [1, 2], 0)
        assert result.shape == labels.shape

    def test_merge_correct(self):
        labels = np.array([0, 1, 2, 1])
        result = merge_clusters(labels, [1, 2], target_id=0)
        assert np.all(result == 0)

    def test_original_unchanged(self):
        labels = np.array([0, 1, 2])
        merge_clusters(labels, [1], target_id=0)
        assert labels[1] == 1

    def test_empty_ids_to_merge(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [], target_id=0)
        np.testing.assert_array_equal(result, labels)
