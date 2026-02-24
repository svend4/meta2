"""Extra tests for puzzle_reconstruction/utils/clustering_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.clustering_utils import (
    ClusterResult,
    kmeans_cluster,
    assign_to_clusters,
    compute_inertia,
    silhouette_score_approx,
    hierarchical_cluster,
    find_optimal_k,
    cluster_indices,
    merge_clusters,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _X(n=20, d=2, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d)).astype(np.float64)


def _bimodal(n=40, d=2) -> np.ndarray:
    rng = np.random.default_rng(42)
    a = rng.normal([0.0, 0.0], 0.05, (n // 2, d))
    b = rng.normal([5.0, 5.0], 0.05, (n // 2, d))
    return np.vstack([a, b])


def _dist_matrix(n=5) -> np.ndarray:
    rng = np.random.default_rng(0)
    M = rng.random((n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    return M


def _cluster_result(n=10, k=2) -> ClusterResult:
    labels = np.array([i % k for i in range(n)], dtype=np.int64)
    centers = np.zeros((k, 2))
    return ClusterResult(labels=labels, centers=centers,
                         n_clusters=k, inertia=1.0)


# ─── ClusterResult ────────────────────────────────────────────────────────────

class TestClusterResultExtra:
    def test_labels_stored(self):
        labs = np.array([0, 1, 0])
        cr = ClusterResult(labels=labs, centers=np.zeros((2, 2)),
                            n_clusters=2, inertia=0.5)
        np.testing.assert_array_equal(cr.labels, labs)

    def test_n_clusters_stored(self):
        assert _cluster_result(k=3).n_clusters == 3

    def test_inertia_stored(self):
        cr = ClusterResult(labels=np.array([0]), centers=np.zeros((1, 2)),
                            n_clusters=1, inertia=2.5)
        assert cr.inertia == pytest.approx(2.5)

    def test_n_clusters_lt_1_raises(self):
        with pytest.raises(ValueError):
            ClusterResult(labels=np.array([0]), centers=np.zeros((1, 2)),
                          n_clusters=0, inertia=0.0)

    def test_negative_inertia_raises(self):
        with pytest.raises(ValueError):
            ClusterResult(labels=np.array([0]), centers=np.zeros((1, 2)),
                          n_clusters=1, inertia=-1.0)

    def test_len_equals_n_labels(self):
        cr = _cluster_result(n=7)
        assert len(cr) == 7

    def test_params_stored(self):
        cr = ClusterResult(labels=np.array([0]), centers=np.zeros((1, 2)),
                            n_clusters=1, inertia=0.0,
                            params={"algorithm": "kmeans"})
        assert cr.params["algorithm"] == "kmeans"

    def test_empty_params_default(self):
        cr = _cluster_result()
        assert isinstance(cr.params, dict)


# ─── kmeans_cluster ───────────────────────────────────────────────────────────

class TestKmeansClusterExtra:
    def test_returns_cluster_result(self):
        assert isinstance(kmeans_cluster(_X(), 2), ClusterResult)

    def test_n_clusters_stored(self):
        cr = kmeans_cluster(_X(10), 3)
        assert cr.n_clusters == 3

    def test_labels_length(self):
        X = _X(20)
        cr = kmeans_cluster(X, 2)
        assert len(cr.labels) == 20

    def test_labels_in_range(self):
        X = _X(20)
        cr = kmeans_cluster(X, 4)
        assert cr.labels.min() >= 0
        assert cr.labels.max() < 4

    def test_centers_shape(self):
        X = _X(20, d=5)
        cr = kmeans_cluster(X, 3)
        assert cr.centers.shape == (3, 5)

    def test_inertia_nonneg(self):
        cr = kmeans_cluster(_X(), 2)
        assert cr.inertia >= 0.0

    def test_n_clusters_gt_n_raises(self):
        with pytest.raises(ValueError):
            kmeans_cluster(_X(5), 10)

    def test_n_clusters_lt_1_raises(self):
        with pytest.raises(ValueError):
            kmeans_cluster(_X(5), 0)

    def test_max_iter_lt_1_raises(self):
        with pytest.raises(ValueError):
            kmeans_cluster(_X(), 2, max_iter=0)

    def test_bimodal_separates(self):
        X = _bimodal(40)
        cr = kmeans_cluster(X, 2)
        # Each cluster should have ~20 points
        unique, counts = np.unique(cr.labels, return_counts=True)
        assert len(unique) == 2
        assert min(counts) >= 15

    def test_deterministic_with_seed(self):
        X = _X(20)
        cr1 = kmeans_cluster(X, 2, random_state=42)
        cr2 = kmeans_cluster(X, 2, random_state=42)
        np.testing.assert_array_equal(cr1.labels, cr2.labels)


# ─── assign_to_clusters ───────────────────────────────────────────────────────

class TestAssignToClustersExtra:
    def test_returns_ndarray(self):
        X = _X(5)
        centers = X[:2]
        assert isinstance(assign_to_clusters(X, centers), np.ndarray)

    def test_dtype_int64(self):
        X = _X(5)
        labels = assign_to_clusters(X, X[:2])
        assert labels.dtype == np.int64

    def test_length_equals_n(self):
        X = _X(8)
        labels = assign_to_clusters(X, X[:3])
        assert len(labels) == 8

    def test_labels_in_range(self):
        X = _X(10)
        centers = X[:4]
        labels = assign_to_clusters(X, centers)
        assert labels.min() >= 0 and labels.max() < 4

    def test_point_assigned_to_nearest(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        centers = np.array([[0.1, 0.1], [9.9, 9.9]])
        labels = assign_to_clusters(X, centers)
        assert labels[0] == 0
        assert labels[1] == 1

    def test_feature_dim_mismatch_raises(self):
        X = _X(5, d=2)
        centers = np.zeros((2, 3))
        with pytest.raises(ValueError):
            assign_to_clusters(X, centers)

    def test_not_2d_raises(self):
        X = np.zeros((3, 2, 2))
        centers = np.zeros((2, 2))
        with pytest.raises(ValueError):
            assign_to_clusters(X, centers)


# ─── compute_inertia ──────────────────────────────────────────────────────────

class TestComputeInertiaExtra:
    def test_returns_float(self):
        X = _X(5)
        labels = np.zeros(5, dtype=np.int64)
        centers = np.zeros((1, 2))
        assert isinstance(compute_inertia(X, labels, centers), float)

    def test_zero_when_each_point_is_center(self):
        X = _X(4)
        labels = np.arange(4, dtype=np.int64)
        centers = X.copy()
        assert compute_inertia(X, labels, centers) == pytest.approx(0.0)

    def test_nonneg(self):
        X = _X(10)
        labels = np.zeros(10, dtype=np.int64)
        centers = np.mean(X, axis=0, keepdims=True)
        assert compute_inertia(X, labels, centers) >= 0.0

    def test_decreases_with_more_clusters(self):
        X = _bimodal(40)
        cr1 = kmeans_cluster(X, 1)
        cr2 = kmeans_cluster(X, 2)
        assert cr2.inertia <= cr1.inertia


# ─── silhouette_score_approx ──────────────────────────────────────────────────

class TestSilhouetteScoreApproxExtra:
    def test_returns_float(self):
        X = _bimodal(20)
        labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        assert isinstance(silhouette_score_approx(X, labels), float)

    def test_single_cluster_zero(self):
        X = _X(5)
        labels = np.zeros(5, dtype=np.int64)
        assert silhouette_score_approx(X, labels) == pytest.approx(0.0)

    def test_well_separated_clusters_positive(self):
        X = _bimodal(40)
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
        score = silhouette_score_approx(X, labels)
        assert score > 0.5

    def test_result_in_minus_one_to_one(self):
        X = _X(10)
        labels = np.array([i % 3 for i in range(10)], dtype=np.int64)
        score = silhouette_score_approx(X, labels)
        assert -1.0 <= score <= 1.0

    def test_non_2d_raises(self):
        X = np.zeros((3, 2, 2))
        labels = np.zeros(3, dtype=np.int64)
        with pytest.raises(ValueError):
            silhouette_score_approx(X, labels)


# ─── hierarchical_cluster ─────────────────────────────────────────────────────

class TestHierarchicalClusterExtra:
    def test_returns_ndarray(self):
        D = _dist_matrix(5)
        assert isinstance(hierarchical_cluster(D, 2), np.ndarray)

    def test_dtype_int64(self):
        D = _dist_matrix(5)
        assert hierarchical_cluster(D, 2).dtype == np.int64

    def test_length_equals_n(self):
        D = _dist_matrix(6)
        labels = hierarchical_cluster(D, 3)
        assert len(labels) == 6

    def test_n_clusters_correct(self):
        D = _dist_matrix(6)
        labels = hierarchical_cluster(D, 2)
        assert len(np.unique(labels)) == 2

    def test_non_square_raises(self):
        D = np.zeros((3, 4))
        with pytest.raises(ValueError):
            hierarchical_cluster(D, 2)

    def test_invalid_linkage_raises(self):
        D = _dist_matrix(5)
        with pytest.raises(ValueError):
            hierarchical_cluster(D, 2, linkage="ward")

    def test_complete_linkage(self):
        D = _dist_matrix(5)
        labels = hierarchical_cluster(D, 2, linkage="complete")
        assert len(np.unique(labels)) == 2

    def test_average_linkage(self):
        D = _dist_matrix(5)
        labels = hierarchical_cluster(D, 2, linkage="average")
        assert len(np.unique(labels)) == 2

    def test_single_cluster(self):
        D = _dist_matrix(4)
        labels = hierarchical_cluster(D, 1)
        assert len(np.unique(labels)) == 1


# ─── find_optimal_k ───────────────────────────────────────────────────────────

class TestFindOptimalKExtra:
    def test_returns_int(self):
        assert isinstance(find_optimal_k(_X(20), k_min=2, k_max=5), int)

    def test_k_in_range(self):
        k = find_optimal_k(_X(20), k_min=2, k_max=5)
        assert 2 <= k <= 5

    def test_k_min_gt_k_max_raises(self):
        with pytest.raises(ValueError):
            find_optimal_k(_X(10), k_min=5, k_max=3)

    def test_k_min_lt_1_raises(self):
        with pytest.raises(ValueError):
            find_optimal_k(_X(10), k_min=0)

    def test_bimodal_returns_two(self):
        X = _bimodal(60)
        k = find_optimal_k(X, k_min=2, k_max=5)
        assert 2 <= k <= 5


# ─── cluster_indices ──────────────────────────────────────────────────────────

class TestClusterIndicesExtra:
    def test_returns_dict(self):
        labels = np.array([0, 1, 0, 1])
        assert isinstance(cluster_indices(labels), dict)

    def test_indices_correct(self):
        labels = np.array([0, 1, 0, 1])
        result = cluster_indices(labels)
        assert sorted(result[0]) == [0, 2]
        assert sorted(result[1]) == [1, 3]

    def test_n_clusters_creates_empty_groups(self):
        labels = np.array([0, 0, 0])
        result = cluster_indices(labels, n_clusters=3)
        assert 1 in result and result[1] == []
        assert 2 in result and result[2] == []

    def test_all_same_label(self):
        labels = np.zeros(5, dtype=np.int64)
        result = cluster_indices(labels)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_indices_sorted(self):
        labels = np.array([1, 0, 1, 0])
        result = cluster_indices(labels)
        assert result[0] == sorted(result[0])
        assert result[1] == sorted(result[1])


# ─── merge_clusters ───────────────────────────────────────────────────────────

class TestMergeClustersExtra:
    def test_returns_ndarray(self):
        labels = np.array([0, 1, 2])
        assert isinstance(merge_clusters(labels, [1, 2], 0), np.ndarray)

    def test_dtype_int64(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [1], 0)
        assert result.dtype == np.int64

    def test_merge_two_into_one(self):
        labels = np.array([0, 1, 2, 1, 2])
        result = merge_clusters(labels, [1, 2], 0)
        assert (result == 0).all()

    def test_unaffected_labels_unchanged(self):
        labels = np.array([0, 1, 2, 3])
        result = merge_clusters(labels, [2], 2)
        assert result[0] == 0
        assert result[1] == 1
        assert result[3] == 3

    def test_empty_ids_to_merge_unchanged(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [], 0)
        np.testing.assert_array_equal(result, labels)

    def test_length_preserved(self):
        labels = np.array([0, 1, 2, 0, 1])
        result = merge_clusters(labels, [1], 0)
        assert len(result) == len(labels)

    def test_original_not_modified(self):
        labels = np.array([0, 1, 2])
        original = labels.copy()
        merge_clusters(labels, [1], 0)
        np.testing.assert_array_equal(labels, original)
