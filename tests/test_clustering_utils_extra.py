"""Extra tests for puzzle_reconstruction/utils/clustering_utils.py"""
import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blobs(n_per=10, k=3, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 10, (k, 2))
    return np.vstack([
        rng.normal(c, 0.3, (n_per, 2)) for c in centers
    ]).astype(np.float64)


def _dist_mat(n=5, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.uniform(0, 1, (n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    return M


# ─── TestClusterResultExtra ───────────────────────────────────────────────────

class TestClusterResultExtra:
    def test_large_n_clusters_valid(self):
        r = ClusterResult(
            labels=np.zeros(5, dtype=np.int64),
            centers=np.zeros((1, 2)),
            n_clusters=1,
            inertia=0.0,
        )
        assert r.n_clusters == 1

    def test_zero_inertia_valid(self):
        r = ClusterResult(
            labels=np.array([0, 1]),
            centers=np.zeros((2, 2)),
            n_clusters=2,
            inertia=0.0,
        )
        assert r.inertia == pytest.approx(0.0)

    def test_len_multiple(self):
        r = ClusterResult(
            labels=np.zeros(10, dtype=np.int64),
            centers=np.zeros((1, 2)),
            n_clusters=1,
            inertia=0.0,
        )
        assert len(r) == 10

    def test_params_stored(self):
        r = ClusterResult(
            labels=np.array([0]),
            centers=np.zeros((1, 2)),
            n_clusters=1,
            inertia=0.0,
            params={"algorithm": "kmeans", "max_iter": 100},
        )
        assert r.params["algorithm"] == "kmeans"


# ─── TestKmeansClusterExtra ───────────────────────────────────────────────────

class TestKmeansClusterExtra:
    def test_4_clusters(self):
        X = _blobs(n_per=15, k=4)
        r = kmeans_cluster(X, n_clusters=4)
        assert r.n_clusters == 4
        assert r.labels.max() < 4

    def test_high_max_iter(self):
        X = _blobs(n_per=10, k=3)
        r = kmeans_cluster(X, n_clusters=3, max_iter=500)
        assert r.inertia >= 0.0

    def test_different_random_states(self):
        X = _blobs()
        r1 = kmeans_cluster(X, n_clusters=3, random_state=0)
        r2 = kmeans_cluster(X, n_clusters=3, random_state=99)
        # Same objective (inertia) may differ; both should be valid
        assert r1.inertia >= 0.0
        assert r2.inertia >= 0.0

    def test_centers_shape(self):
        X = _blobs(n_per=10, k=3)
        r = kmeans_cluster(X, n_clusters=3)
        assert r.centers.shape == (3, 2)

    def test_params_contains_tol(self):
        X = _blobs()
        r = kmeans_cluster(X, n_clusters=2)
        assert "tol" in r.params

    def test_n_clusters_equals_n_valid(self):
        X = np.arange(6).reshape(3, 2).astype(np.float64)
        r = kmeans_cluster(X, n_clusters=3)
        assert r.n_clusters == 3


# ─── TestAssignToClustersExtra ────────────────────────────────────────────────

class TestAssignToClustersExtra:
    def test_3d_features(self):
        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        labels = assign_to_clusters(X, centers)
        assert labels.shape == (3,)
        assert labels[0] == 0
        assert labels[1] == 1

    def test_single_center(self):
        X = np.random.default_rng(0).random((5, 2))
        labels = assign_to_clusters(X, np.zeros((1, 2)))
        assert np.all(labels == 0)

    def test_labels_in_range(self):
        X = _blobs()
        centers = _blobs(n_per=1, k=3)
        labels = assign_to_clusters(X, centers)
        assert labels.min() >= 0
        assert labels.max() < 3


# ─── TestComputeInertiaExtra ──────────────────────────────────────────────────

class TestComputeInertiaExtra:
    def test_single_point_at_center(self):
        X = np.array([[2.0, 3.0]])
        labels = np.array([0])
        centers = np.array([[2.0, 3.0]])
        assert compute_inertia(X, labels, centers) == pytest.approx(0.0)

    def test_large_data(self):
        X = _blobs(n_per=50, k=3)
        r = kmeans_cluster(X, n_clusters=3)
        inertia = compute_inertia(X, r.labels, r.centers)
        assert inertia >= 0.0

    def test_returns_float(self):
        X = _blobs(n_per=5, k=2)
        r = kmeans_cluster(X, n_clusters=2)
        result = compute_inertia(X, r.labels, r.centers)
        assert isinstance(result, float)

    def test_high_dim_features(self):
        rng = np.random.default_rng(3)
        X = rng.random((10, 8))
        labels = np.zeros(10, dtype=np.int64)
        centers = X.mean(axis=0, keepdims=True)
        inertia = compute_inertia(X, labels, centers)
        assert inertia >= 0.0


# ─── TestSilhouetteScoreApproxExtra ──────────────────────────────────────────

class TestSilhouetteScoreApproxExtra:
    def test_two_well_separated(self):
        X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 100])
        labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        score = silhouette_score_approx(X, labels)
        assert score > 0.9

    def test_all_same_label_zero(self):
        X = _blobs(n_per=8, k=1)
        labels = np.zeros(8, dtype=np.int64)
        assert silhouette_score_approx(X, labels) == pytest.approx(0.0)

    def test_range_minus_1_to_1(self):
        X = _blobs()
        r = kmeans_cluster(X, n_clusters=3, random_state=7)
        score = silhouette_score_approx(X, r.labels)
        assert -1.0 <= score <= 1.0

    def test_two_clusters_result_is_float(self):
        X = _blobs(n_per=5, k=2)
        r = kmeans_cluster(X, n_clusters=2)
        result = silhouette_score_approx(X, r.labels)
        assert isinstance(result, float)


# ─── TestHierarchicalClusterExtra ────────────────────────────────────────────

class TestHierarchicalClusterExtra:
    def test_single_element_matrix(self):
        mat = np.array([[0.0]])
        labels = hierarchical_cluster(mat, n_clusters=1)
        assert labels.shape == (1,)
        assert labels[0] == 0

    def test_3_clusters_from_6(self):
        labels = hierarchical_cluster(_dist_mat(6), n_clusters=3)
        assert len(np.unique(labels)) == 3

    def test_single_linkage(self):
        labels = hierarchical_cluster(_dist_mat(5), n_clusters=2, linkage="single")
        assert labels.shape == (5,)

    def test_all_same_cluster(self):
        labels = hierarchical_cluster(_dist_mat(4), n_clusters=1)
        assert len(np.unique(labels)) == 1

    def test_n_clusters_equals_n(self):
        n = 4
        labels = hierarchical_cluster(_dist_mat(n), n_clusters=n)
        assert labels.shape == (n,)

    def test_unknown_linkage_raises(self):
        with pytest.raises(ValueError):
            hierarchical_cluster(_dist_mat(), n_clusters=2, linkage="centroid")


# ─── TestFindOptimalKExtra ────────────────────────────────────────────────────

class TestFindOptimalKExtra:
    def test_uniform_data(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 100, (30, 2))
        k = find_optimal_k(X, k_min=2, k_max=6)
        assert 2 <= k <= 6

    def test_k_min_1_valid(self):
        X = _blobs(n_per=5, k=2)
        k = find_optimal_k(X, k_min=1, k_max=3)
        assert 1 <= k <= 3

    def test_large_k_range(self):
        X = _blobs(n_per=5, k=3)
        k = find_optimal_k(X, k_min=2, k_max=8)
        assert 2 <= k <= 8

    def test_k_min_below_1_raises(self):
        X = np.ones((5, 2))
        with pytest.raises(ValueError):
            find_optimal_k(X, k_min=0)


# ─── TestClusterIndicesExtra ──────────────────────────────────────────────────

class TestClusterIndicesExtra:
    def test_all_same_cluster(self):
        labels = np.zeros(5, dtype=np.int64)
        result = cluster_indices(labels)
        assert result[0] == [0, 1, 2, 3, 4]

    def test_n_clusters_fills_empty(self):
        labels = np.array([0, 2], dtype=np.int64)
        result = cluster_indices(labels, n_clusters=4)
        assert result[1] == []
        assert result[3] == []

    def test_large_cluster_count(self):
        labels = np.arange(10, dtype=np.int64)
        result = cluster_indices(labels)
        assert len(result) == 10

    def test_indices_sorted_order(self):
        labels = np.array([2, 0, 1, 0, 2], dtype=np.int64)
        result = cluster_indices(labels)
        assert result[0] == [1, 3]
        assert result[2] == [0, 4]


# ─── TestMergeClustersExtra ───────────────────────────────────────────────────

class TestMergeClustersExtra:
    def test_merge_2_and_3_into_0(self):
        labels = np.array([0, 1, 2, 3, 2, 3], dtype=np.int64)
        result = merge_clusters(labels, [2, 3], target_id=0)
        assert np.all(result[2:] == 0)
        assert result[1] == 1

    def test_merge_single(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        result = merge_clusters(labels, [2], target_id=1)
        assert result[2] == 1

    def test_target_not_in_labels_ok(self):
        labels = np.array([1, 2, 3], dtype=np.int64)
        result = merge_clusters(labels, [1], target_id=99)
        assert result[0] == 99

    def test_returns_int64(self):
        labels = np.array([0, 1], dtype=np.int32)
        result = merge_clusters(labels, [1], target_id=0)
        assert result.dtype == np.int64

    def test_original_not_modified(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        merge_clusters(labels, [1, 2], target_id=0)
        assert labels[1] == 1
        assert labels[2] == 2
