"""Тесты для puzzle_reconstruction/utils/clustering_utils.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_blobs(n_per=20, n_clusters=3, d=2, seed=0):
    """Create well-separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, (n_clusters, d)) * 5
    X = []
    for c in centers:
        X.append(rng.normal(c, 0.5, (n_per, d)))
    return np.vstack(X)


# ─── ClusterResult ────────────────────────────────────────────────────────────

class TestClusterResult:
    def test_creation(self):
        labels = np.array([0, 1, 0, 1])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        cr = ClusterResult(labels=labels, centers=centers, n_clusters=2, inertia=0.5)
        assert cr.n_clusters == 2
        assert cr.inertia == pytest.approx(0.5)

    def test_n_clusters_zero_raises(self):
        with pytest.raises(ValueError, match="n_clusters"):
            ClusterResult(
                labels=np.array([0]),
                centers=np.array([[0.0]]),
                n_clusters=0,
                inertia=0.0,
            )

    def test_negative_inertia_raises(self):
        with pytest.raises(ValueError, match="inertia"):
            ClusterResult(
                labels=np.array([0]),
                centers=np.array([[0.0]]),
                n_clusters=1,
                inertia=-1.0,
            )

    def test_len_returns_n_samples(self):
        labels = np.array([0, 1, 0, 1, 2])
        centers = np.array([[0.0], [1.0], [2.0]])
        cr = ClusterResult(labels=labels, centers=centers, n_clusters=3, inertia=0.0)
        assert len(cr) == 5

    def test_params_stored(self):
        labels = np.array([0])
        centers = np.array([[0.0]])
        cr = ClusterResult(labels=labels, centers=centers, n_clusters=1,
                           inertia=0.0, params={"algo": "kmeans"})
        assert cr.params["algo"] == "kmeans"


# ─── kmeans_cluster ───────────────────────────────────────────────────────────

class TestKmeansCluster:
    def test_returns_cluster_result(self):
        X = make_blobs()
        result = kmeans_cluster(X, n_clusters=3)
        assert isinstance(result, ClusterResult)

    def test_labels_shape(self):
        X = make_blobs()
        result = kmeans_cluster(X, n_clusters=3)
        assert result.labels.shape == (len(X),)

    def test_centers_shape(self):
        X = make_blobs(n_per=10, d=4)
        result = kmeans_cluster(X, n_clusters=3)
        assert result.centers.shape == (3, 4)

    def test_n_clusters_too_large_raises(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=3)

    def test_n_clusters_zero_raises(self):
        X = make_blobs()
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=0)

    def test_max_iter_zero_raises(self):
        X = make_blobs()
        with pytest.raises(ValueError, match="max_iter"):
            kmeans_cluster(X, n_clusters=2, max_iter=0)

    def test_inertia_nonnegative(self):
        X = make_blobs()
        result = kmeans_cluster(X, n_clusters=3)
        assert result.inertia >= 0.0

    def test_reproducible_with_seed(self):
        X = make_blobs()
        r1 = kmeans_cluster(X, n_clusters=3, random_state=42)
        r2 = kmeans_cluster(X, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_k_equals_1(self):
        X = make_blobs(n_per=10)
        result = kmeans_cluster(X, n_clusters=1)
        assert result.n_clusters == 1
        assert np.all(result.labels == 0)

    def test_labels_in_valid_range(self):
        X = make_blobs(n_per=15, n_clusters=4)
        result = kmeans_cluster(X, n_clusters=4)
        assert np.all(result.labels >= 0)
        assert np.all(result.labels < 4)

    def test_params_algo_key(self):
        X = make_blobs(n_per=10)
        result = kmeans_cluster(X, n_clusters=2)
        assert result.params.get("algorithm") == "kmeans"


# ─── assign_to_clusters ───────────────────────────────────────────────────────

class TestAssignToClusters:
    def test_returns_int64(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.1]])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = assign_to_clusters(X, centers)
        assert labels.dtype == np.int64

    def test_shape(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = assign_to_clusters(X, centers)
        assert labels.shape == (3,)

    def test_non_2d_X_raises(self):
        with pytest.raises(ValueError):
            assign_to_clusters(np.array([1.0, 2.0]), np.array([[1.0]]))

    def test_feature_dim_mismatch_raises(self):
        X = np.array([[1.0, 2.0]])
        centers = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="dimensions"):
            assign_to_clusters(X, centers)

    def test_nearest_center(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert labels[0] == 0
        assert labels[1] == 1

    def test_all_assigned_to_single_center(self):
        X = np.random.default_rng(0).standard_normal((20, 3))
        centers = np.array([[100.0, 100.0, 100.0]])
        labels = assign_to_clusters(X, centers)
        np.testing.assert_array_equal(labels, 0)


# ─── compute_inertia ──────────────────────────────────────────────────────────

class TestComputeInertia:
    def test_returns_float(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array([0, 1])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        val = compute_inertia(X, labels, centers)
        assert isinstance(val, float)

    def test_perfect_clustering_zero_inertia(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array([0, 1])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        val = compute_inertia(X, labels, centers)
        assert val == pytest.approx(0.0)

    def test_nonnegative(self):
        X = np.random.default_rng(0).standard_normal((30, 3))
        labels = np.zeros(30, dtype=int)
        centers = np.array([[0.0, 0.0, 0.0]])
        val = compute_inertia(X, labels, centers)
        assert val >= 0.0

    def test_known_value(self):
        X = np.array([[1.0, 0.0], [3.0, 0.0]])
        labels = np.array([0, 0])
        centers = np.array([[2.0, 0.0]])
        # Distances: 1.0^2 + 1.0^2 = 2.0
        val = compute_inertia(X, labels, centers)
        assert val == pytest.approx(2.0)


# ─── silhouette_score_approx ──────────────────────────────────────────────────

class TestSilhouetteScoreApprox:
    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            silhouette_score_approx(np.array([1.0, 2.0]), np.array([0, 1]))

    def test_single_cluster_returns_0(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        labels = np.zeros(10, dtype=int)
        assert silhouette_score_approx(X, labels) == pytest.approx(0.0)

    def test_result_in_neg1_1(self):
        X = make_blobs(n_per=10, n_clusters=3)
        result = kmeans_cluster(X, n_clusters=3)
        score = silhouette_score_approx(X, result.labels)
        assert -1.0 <= score <= 1.0

    def test_well_separated_clusters_high_score(self):
        # Very well-separated blobs
        X = np.vstack([
            np.random.default_rng(0).normal([0, 0], 0.01, (20, 2)),
            np.random.default_rng(1).normal([100, 100], 0.01, (20, 2)),
        ])
        labels = np.array([0] * 20 + [1] * 20)
        score = silhouette_score_approx(X, labels)
        assert score > 0.9


# ─── hierarchical_cluster ─────────────────────────────────────────────────────

class TestHierarchicalCluster:
    def _dist_matrix(self, n=5, seed=0):
        rng = np.random.default_rng(seed)
        M = rng.uniform(0, 1, (n, n))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0.0)
        return M

    def test_returns_int64(self):
        M = self._dist_matrix()
        labels = hierarchical_cluster(M, n_clusters=2)
        assert labels.dtype == np.int64

    def test_labels_shape(self):
        M = self._dist_matrix(n=6)
        labels = hierarchical_cluster(M, n_clusters=2)
        assert labels.shape == (6,)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            hierarchical_cluster(np.ones((3, 4)), n_clusters=2)

    def test_invalid_linkage_raises(self):
        M = self._dist_matrix()
        with pytest.raises(ValueError, match="linkage"):
            hierarchical_cluster(M, n_clusters=2, linkage="ward")

    def test_n_clusters_equals_n(self):
        M = self._dist_matrix(n=4)
        labels = hierarchical_cluster(M, n_clusters=4)
        assert len(set(labels.tolist())) == 4

    def test_n_clusters_1(self):
        M = self._dist_matrix(n=4)
        labels = hierarchical_cluster(M, n_clusters=1)
        assert len(set(labels.tolist())) == 1

    def test_single_linkage(self):
        M = self._dist_matrix()
        labels = hierarchical_cluster(M, n_clusters=2, linkage="single")
        assert labels.shape == (5,)

    def test_complete_linkage(self):
        M = self._dist_matrix()
        labels = hierarchical_cluster(M, n_clusters=2, linkage="complete")
        assert labels.shape == (5,)

    def test_average_linkage(self):
        M = self._dist_matrix()
        labels = hierarchical_cluster(M, n_clusters=2, linkage="average")
        assert labels.shape == (5,)

    def test_n_clusters_too_large_raises(self):
        M = self._dist_matrix(n=3)
        with pytest.raises(ValueError):
            hierarchical_cluster(M, n_clusters=4)


# ─── find_optimal_k ───────────────────────────────────────────────────────────

class TestFindOptimalK:
    def test_k_min_less_than_1_raises(self):
        X = make_blobs(n_per=10)
        with pytest.raises(ValueError, match="k_min"):
            find_optimal_k(X, k_min=0)

    def test_k_min_greater_than_k_max_raises(self):
        X = make_blobs(n_per=10)
        with pytest.raises(ValueError):
            find_optimal_k(X, k_min=5, k_max=3)

    def test_returns_int(self):
        X = make_blobs(n_per=10, n_clusters=3)
        k = find_optimal_k(X, k_min=2, k_max=5)
        assert isinstance(k, int)

    def test_returns_k_in_range(self):
        X = make_blobs(n_per=10, n_clusters=3)
        k = find_optimal_k(X, k_min=2, k_max=6)
        assert 2 <= k <= 6

    def test_single_k_returns_that_k(self):
        X = make_blobs(n_per=10)
        k = find_optimal_k(X, k_min=3, k_max=3)
        assert k == 3

    def test_well_separated_3_blobs(self):
        X = make_blobs(n_per=30, n_clusters=3)
        k = find_optimal_k(X, k_min=2, k_max=6)
        # Elbow should find k=3 or nearby
        assert 2 <= k <= 6


# ─── cluster_indices ──────────────────────────────────────────────────────────

class TestClusterIndices:
    def test_basic(self):
        labels = np.array([0, 1, 0, 1, 2])
        result = cluster_indices(labels)
        assert result[0] == [0, 2]
        assert result[1] == [1, 3]
        assert result[2] == [4]

    def test_returns_sorted_indices(self):
        labels = np.array([2, 0, 1, 0])
        result = cluster_indices(labels)
        for v in result.values():
            assert v == sorted(v)

    def test_n_clusters_includes_empty(self):
        labels = np.array([0, 0, 2])  # cluster 1 missing
        result = cluster_indices(labels, n_clusters=3)
        assert 1 in result
        assert result[1] == []

    def test_all_same_label(self):
        labels = np.array([0, 0, 0, 0])
        result = cluster_indices(labels)
        assert result[0] == [0, 1, 2, 3]

    def test_returns_dict(self):
        labels = np.array([0, 1])
        result = cluster_indices(labels)
        assert isinstance(result, dict)


# ─── merge_clusters ───────────────────────────────────────────────────────────

class TestMergeClusters:
    def test_returns_int64(self):
        labels = np.array([0, 1, 2, 1, 0])
        result = merge_clusters(labels, [1, 2], target_id=0)
        assert result.dtype == np.int64

    def test_merges_to_target(self):
        labels = np.array([0, 1, 2, 1, 0])
        result = merge_clusters(labels, [1, 2], target_id=0)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])

    def test_unmerged_labels_unchanged(self):
        labels = np.array([0, 1, 2, 3])
        result = merge_clusters(labels, [2], target_id=1)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 3

    def test_empty_ids_to_merge(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [], target_id=5)
        np.testing.assert_array_equal(result, labels)

    def test_does_not_modify_original(self):
        labels = np.array([0, 1, 2])
        original = labels.copy()
        merge_clusters(labels, [1], target_id=0)
        np.testing.assert_array_equal(labels, original)

    def test_returns_copy(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [1], target_id=0)
        labels[0] = 99
        assert result[0] != 99
