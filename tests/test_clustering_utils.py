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

def _two_blobs(n: int = 30, seed: int = 0) -> np.ndarray:
    """Two clearly separated 2-D blobs (first n/2 near 0, rest near 10)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    c1 = rng.normal(0.0, 0.3, (half, 2))
    c2 = rng.normal(10.0, 0.3, (n - half, 2))
    return np.vstack([c1, c2]).astype(np.float64)


def _three_blobs(n: int = 30, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    third = n // 3
    c1 = rng.normal(0.0, 0.3, (third, 2))
    c2 = rng.normal(10.0, 0.3, (third, 2))
    c3 = rng.normal(20.0, 0.3, (n - 2 * third, 2))
    return np.vstack([c1, c2, c3]).astype(np.float64)


def _dist_matrix(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


# ─── ClusterResult ────────────────────────────────────────────────────────────

class TestClusterResult:
    def test_fields_stored(self):
        labels = np.array([0, 1, 0])
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        cr = ClusterResult(labels=labels, centers=centers,
                           n_clusters=2, inertia=0.5)
        assert cr.n_clusters == 2
        assert cr.inertia == pytest.approx(0.5)

    def test_len_returns_n_points(self):
        labels = np.array([0, 1, 0, 1])
        centers = np.array([[0.0], [1.0]])
        cr = ClusterResult(labels=labels, centers=centers,
                           n_clusters=2, inertia=0.0)
        assert len(cr) == 4

    def test_n_clusters_lt_1_raises(self):
        with pytest.raises(ValueError):
            ClusterResult(labels=np.array([0]), centers=np.array([[0.0]]),
                          n_clusters=0, inertia=0.0)

    def test_negative_inertia_raises(self):
        with pytest.raises(ValueError):
            ClusterResult(labels=np.array([0]), centers=np.array([[0.0]]),
                          n_clusters=1, inertia=-0.1)

    def test_params_default_empty(self):
        cr = ClusterResult(labels=np.array([0]), centers=np.array([[0.0]]),
                           n_clusters=1, inertia=0.0)
        assert cr.params == {}

    def test_params_stored(self):
        cr = ClusterResult(labels=np.array([0]), centers=np.array([[0.0]]),
                           n_clusters=1, inertia=0.0,
                           params={"algorithm": "kmeans"})
        assert cr.params["algorithm"] == "kmeans"


# ─── kmeans_cluster ──────────────────────────────────────────────────────────

class TestKMeansCluster:
    def test_returns_cluster_result(self):
        X = _two_blobs()
        result = kmeans_cluster(X, n_clusters=2)
        assert isinstance(result, ClusterResult)

    def test_correct_n_clusters(self):
        X = _two_blobs()
        result = kmeans_cluster(X, n_clusters=2)
        assert result.n_clusters == 2

    def test_labels_shape(self):
        X = _two_blobs(20)
        result = kmeans_cluster(X, n_clusters=2)
        assert result.labels.shape == (20,)

    def test_labels_values_in_range(self):
        X = _two_blobs()
        result = kmeans_cluster(X, n_clusters=2)
        assert np.all(result.labels >= 0)
        assert np.all(result.labels < 2)

    def test_centers_shape(self):
        X = _two_blobs()
        result = kmeans_cluster(X, n_clusters=2)
        assert result.centers.shape == (2, 2)

    def test_inertia_nonnegative(self):
        X = _two_blobs()
        result = kmeans_cluster(X, n_clusters=2)
        assert result.inertia >= 0.0

    def test_two_clear_blobs_separated(self):
        X = _two_blobs(40, seed=42)
        result = kmeans_cluster(X, n_clusters=2, random_state=0)
        # First 20 should have same label, last 20 another
        assert len(set(result.labels[:20].tolist())) == 1
        assert len(set(result.labels[20:].tolist())) == 1
        assert result.labels[0] != result.labels[20]

    def test_n_clusters_too_large_raises(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=5)

    def test_n_clusters_zero_raises(self):
        X = _two_blobs(10)
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=0)

    def test_max_iter_lt_1_raises(self):
        X = _two_blobs(10)
        with pytest.raises(ValueError):
            kmeans_cluster(X, n_clusters=2, max_iter=0)

    def test_reproducible_with_seed(self):
        X = _two_blobs(30)
        r1 = kmeans_cluster(X, n_clusters=2, random_state=7)
        r2 = kmeans_cluster(X, n_clusters=2, random_state=7)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_single_cluster(self):
        X = _two_blobs(10)
        result = kmeans_cluster(X, n_clusters=1)
        assert result.n_clusters == 1
        assert np.all(result.labels == 0)


# ─── assign_to_clusters ───────────────────────────────────────────────────────

class TestAssignToClusters:
    def test_returns_int64_array(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert labels.dtype == np.int64

    def test_shape_is_n(self):
        X = _two_blobs(20)
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert labels.shape == (20,)

    def test_nearest_center_assigned(self):
        X = np.array([[0.1, 0.1], [9.9, 9.9], [0.2, 0.2]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert labels[0] == 0
        assert labels[1] == 1
        assert labels[2] == 0

    def test_feature_dim_mismatch_raises(self):
        X = np.ones((5, 3))
        centers = np.ones((2, 2))
        with pytest.raises(ValueError):
            assign_to_clusters(X, centers)

    def test_non_2d_X_raises(self):
        X = np.ones((5,))
        centers = np.ones((2, 1))
        with pytest.raises(ValueError):
            assign_to_clusters(X, centers)

    def test_all_labels_valid(self):
        X = _two_blobs()
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_to_clusters(X, centers)
        assert np.all((labels == 0) | (labels == 1))


# ─── compute_inertia ──────────────────────────────────────────────────────────

class TestComputeInertia:
    def test_returns_float(self):
        X = _two_blobs(10)
        labels = np.zeros(10, dtype=np.int64)
        centers = np.array([[0.0, 0.0]])
        result = compute_inertia(X, labels, centers)
        assert isinstance(result, float)

    def test_nonnegative(self):
        X = _two_blobs()
        result_obj = kmeans_cluster(X, n_clusters=2)
        inertia = compute_inertia(X, result_obj.labels, result_obj.centers)
        assert inertia >= 0.0

    def test_zero_inertia_when_centers_equal_points(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 1])
        centers = X.copy()
        assert compute_inertia(X, labels, centers) == pytest.approx(0.0)

    def test_more_clusters_lower_inertia(self):
        X = _three_blobs(60)
        r2 = kmeans_cluster(X, n_clusters=2)
        r3 = kmeans_cluster(X, n_clusters=3)
        assert r3.inertia <= r2.inertia + 1e-6

    def test_matches_result_inertia(self):
        X = _two_blobs(20)
        r = kmeans_cluster(X, n_clusters=2)
        manual = compute_inertia(X, r.labels, r.centers)
        assert manual == pytest.approx(r.inertia, rel=1e-5)


# ─── silhouette_score_approx ──────────────────────────────────────────────────

class TestSilhouetteScoreApprox:
    def test_returns_float(self):
        X = _two_blobs()
        labels = np.array([0] * 15 + [1] * 15)
        result = silhouette_score_approx(X, labels)
        assert isinstance(result, float)

    def test_in_range_minus1_to_1(self):
        X = _two_blobs(20)
        labels = np.array([0] * 10 + [1] * 10)
        score = silhouette_score_approx(X, labels)
        assert -1.0 <= score <= 1.0

    def test_good_clustering_positive_score(self):
        X = _two_blobs(40, seed=0)
        labels = np.array([0] * 20 + [1] * 20)
        score = silhouette_score_approx(X, labels)
        assert score > 0.5

    def test_single_cluster_returns_zero(self):
        X = _two_blobs(10)
        labels = np.zeros(10, dtype=np.int64)
        score = silhouette_score_approx(X, labels)
        assert score == pytest.approx(0.0)

    def test_not_2d_raises(self):
        X = np.ones((10,))
        labels = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError):
            silhouette_score_approx(X, labels)

    def test_well_separated_blobs_high_score(self):
        X = _two_blobs(60)
        r = kmeans_cluster(X, n_clusters=2, random_state=0)
        score = silhouette_score_approx(X, r.labels)
        assert score > 0.7


# ─── hierarchical_cluster ────────────────────────────────────────────────────

class TestHierarchicalCluster:
    def _simple_dist(self) -> np.ndarray:
        # 4 points: 0-1 close, 2-3 close, pairs far from each other
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        return _dist_matrix(X)

    def test_returns_int64_array(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=2)
        assert labels.dtype == np.int64

    def test_shape_is_n(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=2)
        assert labels.shape == (4,)

    def test_correct_grouping_single_linkage(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=2, linkage="single")
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_complete_linkage_accepted(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=2, linkage="complete")
        assert labels.shape == (4,)

    def test_average_linkage_accepted(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=2, linkage="average")
        assert labels.shape == (4,)

    def test_invalid_linkage_raises(self):
        D = self._simple_dist()
        with pytest.raises(ValueError):
            hierarchical_cluster(D, n_clusters=2, linkage="ward")

    def test_non_square_matrix_raises(self):
        D = np.ones((3, 4))
        with pytest.raises(ValueError):
            hierarchical_cluster(D, n_clusters=2)

    def test_n_clusters_too_large_raises(self):
        D = self._simple_dist()
        with pytest.raises(ValueError):
            hierarchical_cluster(D, n_clusters=10)

    def test_n_clusters_zero_raises(self):
        D = self._simple_dist()
        with pytest.raises(ValueError):
            hierarchical_cluster(D, n_clusters=0)

    def test_single_cluster(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=1)
        assert np.all(labels == labels[0])

    def test_n_clusters_equals_n(self):
        D = self._simple_dist()
        labels = hierarchical_cluster(D, n_clusters=4)
        assert len(np.unique(labels)) == 4


# ─── find_optimal_k ──────────────────────────────────────────────────────────

class TestFindOptimalK:
    def test_returns_int(self):
        X = _two_blobs(20)
        k = find_optimal_k(X, k_min=2, k_max=5)
        assert isinstance(k, int)

    def test_returns_value_in_range(self):
        X = _two_blobs(20)
        k = find_optimal_k(X, k_min=2, k_max=6)
        assert 2 <= k <= 6

    def test_two_blobs_finds_2(self):
        X = _two_blobs(40)
        k = find_optimal_k(X, k_min=2, k_max=5)
        assert k == 2

    def test_three_blobs_finds_3(self):
        X = _three_blobs(60)
        k = find_optimal_k(X, k_min=2, k_max=6)
        assert k == 3

    def test_k_min_gt_k_max_raises(self):
        X = _two_blobs(10)
        with pytest.raises(ValueError):
            find_optimal_k(X, k_min=5, k_max=2)

    def test_k_min_lt_1_raises(self):
        X = _two_blobs(10)
        with pytest.raises(ValueError):
            find_optimal_k(X, k_min=0, k_max=5)

    def test_k_max_clamped_to_n(self):
        X = _two_blobs(5)
        # k_max > n should not raise
        k = find_optimal_k(X, k_min=1, k_max=100)
        assert 1 <= k <= 5


# ─── cluster_indices ──────────────────────────────────────────────────────────

class TestClusterIndices:
    def test_returns_dict(self):
        labels = np.array([0, 1, 0, 1])
        result = cluster_indices(labels)
        assert isinstance(result, dict)

    def test_all_indices_covered(self):
        labels = np.array([0, 1, 0, 2])
        result = cluster_indices(labels)
        all_idx = sorted(idx for v in result.values() for idx in v)
        assert all_idx == [0, 1, 2, 3]

    def test_correct_grouping(self):
        labels = np.array([0, 1, 0, 1])
        result = cluster_indices(labels)
        assert sorted(result[0]) == [0, 2]
        assert sorted(result[1]) == [1, 3]

    def test_n_clusters_includes_empty(self):
        labels = np.array([0, 0, 2])
        result = cluster_indices(labels, n_clusters=3)
        assert 1 in result
        assert result[1] == []

    def test_single_cluster(self):
        labels = np.zeros(5, dtype=np.int64)
        result = cluster_indices(labels)
        assert sorted(result[0]) == [0, 1, 2, 3, 4]

    def test_each_point_once(self):
        labels = np.array([0, 1, 2, 0, 1])
        result = cluster_indices(labels)
        all_idx = [idx for v in result.values() for idx in v]
        assert len(all_idx) == 5
        assert len(set(all_idx)) == 5


# ─── merge_clusters ──────────────────────────────────────────────────────────

class TestMergeClusters:
    def test_returns_int64_array(self):
        labels = np.array([0, 1, 2, 0])
        result = merge_clusters(labels, [1, 2], target_id=0)
        assert result.dtype == np.int64

    def test_shape_preserved(self):
        labels = np.array([0, 1, 2, 3])
        result = merge_clusters(labels, [1, 2], target_id=0)
        assert result.shape == (4,)

    def test_merged_ids_become_target(self):
        labels = np.array([0, 1, 2, 0, 1])
        result = merge_clusters(labels, [1, 2], target_id=0)
        assert np.all(result == 0)

    def test_unmerged_ids_unchanged(self):
        labels = np.array([0, 1, 2, 3])
        result = merge_clusters(labels, [2], target_id=0)
        assert result[1] == 1
        assert result[3] == 3

    def test_empty_merge_list_unchanged(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [], target_id=0)
        np.testing.assert_array_equal(result, labels)

    def test_original_not_modified(self):
        labels = np.array([0, 1, 2])
        original = labels.copy()
        merge_clusters(labels, [1], target_id=0)
        np.testing.assert_array_equal(labels, original)

    def test_target_id_not_in_source(self):
        labels = np.array([0, 1, 2])
        result = merge_clusters(labels, [0, 1, 2], target_id=99)
        assert np.all(result == 99)
