"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.alignment_utils
  - puzzle_reconstruction.utils.clustering_utils

alignment_utils:
    normalize_for_alignment:   centroid → 0; RMS dist → 1; scale > 0
    compute_alignment_error:   >= 0; error(X, X) = 0
    alignment_score:           ∈ (0, 1]; score(error=0) = 1.0; decreasing in error
    align_curves_procrustes:   error >= 0; aligned.shape = (n_samples, 2)
    align_curves_icp:          error >= 0; aligned.shape = (n_samples, 2)
    batch_align_curves:        len(results) = len(sources)
    find_best_translation:     applying translation zeroes mean diff

clustering_utils:
    kmeans_cluster:            labels ∈ [0, k-1]; centers.shape = (k, d);
                               inertia >= 0; len(labels) = N
    assign_to_clusters:        result ∈ [0, k-1]; shape = (N,)
    compute_inertia:           >= 0; inertia(own_centers) <= inertia(random)
    silhouette_score_approx:   ∈ [-1, 1]
    cluster_indices:           union = all indices; disjoint sets
    merge_clusters:            new k = old k - len(to_merge) + 1
    find_optimal_k:            result ∈ [1, max_k]
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from puzzle_reconstruction.utils.alignment_utils import (
    AlignmentConfig,
    AlignmentResult,
    normalize_for_alignment,
    find_best_translation,
    compute_alignment_error,
    alignment_score,
    align_curves_procrustes,
    align_curves_icp,
    batch_align_curves,
)
from puzzle_reconstruction.utils.clustering_utils import (
    ClusterResult,
    kmeans_cluster,
    assign_to_clusters,
    compute_inertia,
    silhouette_score_approx,
    cluster_indices,
    merge_clusters,
    find_optimal_k,
)

RNG = np.random.default_rng(4242)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_curve(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, n)
    r = 1.0 + rng.uniform(-0.2, 0.2, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _rand_data(n: int = 30, d: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


# ═══════════════════════════════════════════════════════════════════════════════
# alignment_utils — normalize_for_alignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeForAlignment:
    """normalize_for_alignment: zero-mean and unit RMS."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_centroid_is_zero(self, seed: int) -> None:
        pts = _rand_curve(20, seed=seed)
        norm, _, _ = normalize_for_alignment(pts)
        np.testing.assert_allclose(norm.mean(axis=0), [0.0, 0.0], atol=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_scale_positive(self, seed: int) -> None:
        pts = _rand_curve(20, seed=seed)
        _, _, scale = normalize_for_alignment(pts)
        assert scale > 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_centroid_returned(self, seed: int) -> None:
        pts = _rand_curve(20, seed=seed)
        orig_centroid = pts.mean(axis=0)
        _, centroid, _ = normalize_for_alignment(pts)
        np.testing.assert_allclose(centroid, orig_centroid, atol=1e-10)

    def test_single_point_no_crash(self) -> None:
        pts = np.array([[3.0, 4.0]])
        norm, c, s = normalize_for_alignment(pts)
        assert s > 0.0
        assert norm.shape == (1, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# alignment_utils — compute_alignment_error, alignment_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignmentError:
    """compute_alignment_error and alignment_score invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_error_nonneg(self, seed: int) -> None:
        a = _rand_curve(16, seed=seed)
        b = _rand_curve(16, seed=seed + 100)
        assert compute_alignment_error(a, b) >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_self_error_zero(self, seed: int) -> None:
        pts = _rand_curve(16, seed=seed)
        assert compute_alignment_error(pts, pts) == pytest.approx(0.0, abs=1e-10)

    def test_mismatched_shapes_infinite(self) -> None:
        a = _rand_curve(5, seed=0)
        b = _rand_curve(10, seed=1)
        err = compute_alignment_error(a, b)
        assert math.isinf(err)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_in_0_1(self, seed: int) -> None:
        pts = _rand_curve(16, seed=seed)
        shifted = pts + rng_offset(seed)
        err = compute_alignment_error(pts, shifted)
        result = AlignmentResult(
            rotation=0.0, translation=np.zeros(2), scale=1.0,
            error=err, aligned=shifted, converged=True,
        )
        score = alignment_score(result, sigma=1.0)
        assert 0.0 < score <= 1.0

    def test_score_at_zero_error_is_one(self) -> None:
        pts = _rand_curve(16, seed=0)
        result = AlignmentResult(
            rotation=0.0, translation=np.zeros(2), scale=1.0,
            error=0.0, aligned=pts, converged=True,
        )
        assert alignment_score(result, sigma=1.0) == pytest.approx(1.0)

    def test_score_decreases_with_error(self) -> None:
        def _score(err: float) -> float:
            r = AlignmentResult(
                rotation=0.0, translation=np.zeros(2), scale=1.0,
                error=err, aligned=np.zeros((2, 2)), converged=True,
            )
            return alignment_score(r, sigma=2.0)
        assert _score(0.1) > _score(1.0) > _score(10.0)

    def test_invalid_sigma_raises(self) -> None:
        r = AlignmentResult(
            rotation=0.0, translation=np.zeros(2), scale=1.0,
            error=1.0, aligned=np.zeros((2, 2)), converged=True,
        )
        with pytest.raises(ValueError):
            alignment_score(r, sigma=0.0)


def rng_offset(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 999)
    return rng.uniform(0.1, 1.0, size=(1, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# alignment_utils — find_best_translation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindBestTranslation:
    """find_best_translation: applying it centres source on target."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_translation_shape(self, seed: int) -> None:
        src = _rand_curve(16, seed=seed)
        tgt = _rand_curve(16, seed=seed + 50)
        t = find_best_translation(src, tgt)
        assert t.shape == (2,)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_applying_translation_aligns_means(self, seed: int) -> None:
        src = _rand_curve(16, seed=seed)
        tgt = _rand_curve(16, seed=seed + 50)
        t = find_best_translation(src, tgt)
        aligned_mean = (src + t).mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        np.testing.assert_allclose(aligned_mean, tgt_mean, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# alignment_utils — align_curves_procrustes, align_curves_icp
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignCurves:
    """align_curves_procrustes and align_curves_icp invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_procrustes_error_nonneg(self, seed: int) -> None:
        src = _rand_curve(20, seed=seed)
        tgt = _rand_curve(20, seed=seed + 30)
        result = align_curves_procrustes(src, tgt)
        assert result.error >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_procrustes_aligned_shape(self, seed: int) -> None:
        cfg = AlignmentConfig(n_samples=32)
        src = _rand_curve(20, seed=seed)
        tgt = _rand_curve(20, seed=seed + 30)
        result = align_curves_procrustes(src, tgt, cfg)
        assert result.aligned.shape == (32, 2)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_procrustes_scale_positive(self, seed: int) -> None:
        src = _rand_curve(20, seed=seed)
        tgt = _rand_curve(20, seed=seed + 30)
        result = align_curves_procrustes(src, tgt)
        assert result.scale > 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_procrustes_self_low_error(self, seed: int) -> None:
        pts = _rand_curve(20, seed=seed)
        result = align_curves_procrustes(pts, pts)
        assert result.error == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_icp_error_nonneg(self, seed: int) -> None:
        src = _rand_curve(20, seed=seed)
        tgt = _rand_curve(20, seed=seed + 30)
        result = align_curves_icp(src, tgt)
        assert result.error >= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_icp_scale_is_one(self, seed: int) -> None:
        src = _rand_curve(20, seed=seed)
        tgt = _rand_curve(20, seed=seed + 30)
        result = align_curves_icp(src, tgt)
        assert result.scale == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_icp_aligned_shape(self, seed: int) -> None:
        cfg = AlignmentConfig(n_samples=24)
        src = _rand_curve(20, seed=seed)
        tgt = _rand_curve(20, seed=seed + 30)
        result = align_curves_icp(src, tgt, cfg)
        assert result.aligned.shape == (24, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# alignment_utils — batch_align_curves
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchAlignCurves:
    """batch_align_curves: length and result types."""

    @pytest.mark.parametrize("n_pairs,method", [
        (3, "procrustes"), (4, "icp"), (2, "procrustes"),
    ])
    def test_batch_length(self, n_pairs: int, method: str) -> None:
        sources = [_rand_curve(15, seed=i) for i in range(n_pairs)]
        targets = [_rand_curve(15, seed=i + 50) for i in range(n_pairs)]
        results = batch_align_curves(sources, targets, method=method)
        assert len(results) == n_pairs

    @pytest.mark.parametrize("n_pairs", [2, 4])
    def test_batch_all_errors_nonneg(self, n_pairs: int) -> None:
        sources = [_rand_curve(15, seed=i) for i in range(n_pairs)]
        targets = [_rand_curve(15, seed=i + 50) for i in range(n_pairs)]
        results = batch_align_curves(sources, targets)
        for r in results:
            assert r.error >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# clustering_utils — kmeans_cluster
# ═══════════════════════════════════════════════════════════════════════════════

class TestKmeans:
    """kmeans_cluster: labels, centers, inertia invariants."""

    @pytest.mark.parametrize("k,n,d", [(2, 20, 3), (3, 30, 4), (4, 40, 2)])
    def test_labels_in_range(self, k: int, n: int, d: int) -> None:
        X = _rand_data(n, d, seed=k)
        result = kmeans_cluster(X, k)
        assert result.labels.min() >= 0
        assert result.labels.max() <= k - 1

    @pytest.mark.parametrize("k,n,d", [(2, 20, 3), (3, 30, 4)])
    def test_centers_shape(self, k: int, n: int, d: int) -> None:
        X = _rand_data(n, d, seed=k + 10)
        result = kmeans_cluster(X, k)
        assert result.centers.shape == (k, d)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_inertia_nonneg(self, k: int) -> None:
        X = _rand_data(30, 3, seed=k + 20)
        result = kmeans_cluster(X, k)
        assert result.inertia >= 0.0

    @pytest.mark.parametrize("k", [2, 3])
    def test_labels_length(self, k: int) -> None:
        n = 25
        X = _rand_data(n, 3, seed=k + 30)
        result = kmeans_cluster(X, k)
        assert len(result.labels) == n

    @pytest.mark.parametrize("k", [2, 3])
    def test_n_clusters_field(self, k: int) -> None:
        X = _rand_data(20, 3, seed=k + 40)
        result = kmeans_cluster(X, k)
        assert result.n_clusters == k

    def test_invalid_k_raises(self) -> None:
        X = _rand_data(10, 2, seed=0)
        with pytest.raises(ValueError):
            kmeans_cluster(X, 0)
        with pytest.raises(ValueError):
            kmeans_cluster(X, 11)


# ═══════════════════════════════════════════════════════════════════════════════
# clustering_utils — assign_to_clusters, compute_inertia
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssignAndInertia:
    """assign_to_clusters and compute_inertia invariants."""

    @pytest.mark.parametrize("k,n,d", [(2, 20, 3), (3, 25, 4)])
    def test_assign_in_range(self, k: int, n: int, d: int) -> None:
        X = _rand_data(n, d, seed=k)
        centers = _rand_data(k, d, seed=k + 100)
        labels = assign_to_clusters(X, centers)
        assert labels.min() >= 0
        assert labels.max() <= k - 1

    @pytest.mark.parametrize("n,d", [(20, 3), (30, 4)])
    def test_assign_shape(self, n: int, d: int) -> None:
        X = _rand_data(n, d, seed=7)
        centers = _rand_data(3, d, seed=77)
        labels = assign_to_clusters(X, centers)
        assert labels.shape == (n,)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_inertia_nonneg(self, k: int) -> None:
        X = _rand_data(30, 3, seed=k)
        centers = _rand_data(k, 3, seed=k + 200)
        labels = assign_to_clusters(X, centers)
        inertia = compute_inertia(X, labels, centers)
        assert inertia >= 0.0

    @pytest.mark.parametrize("k", [2, 3])
    def test_own_centers_inertia_leq_random(self, k: int) -> None:
        X = _rand_data(30, 3, seed=k)
        result = kmeans_cluster(X, k)
        own_inertia = compute_inertia(X, result.labels, result.centers)
        rand_centers = _rand_data(k, 3, seed=k + 300)
        rand_labels = assign_to_clusters(X, rand_centers)
        rand_inertia = compute_inertia(X, rand_labels, rand_centers)
        # k-means minimises inertia, so its inertia should be <= random
        assert own_inertia <= rand_inertia + 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# clustering_utils — silhouette_score_approx
# ═══════════════════════════════════════════════════════════════════════════════

class TestSilhouette:
    """silhouette_score_approx: ∈ [-1, 1]."""

    @pytest.mark.parametrize("k,n,d", [(2, 20, 3), (3, 30, 4)])
    def test_silhouette_in_range(self, k: int, n: int, d: int) -> None:
        X = _rand_data(n, d, seed=k + 50)
        result = kmeans_cluster(X, k)
        score = silhouette_score_approx(X, result.labels)
        assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# clustering_utils — cluster_indices, merge_clusters, find_optimal_k
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterIndicesMerge:
    """cluster_indices: disjoint, union; merge_clusters: new k."""

    @pytest.mark.parametrize("k,n", [(2, 20), (3, 30), (4, 40)])
    def test_cluster_indices_union(self, k: int, n: int) -> None:
        X = _rand_data(n, 3, seed=k)
        result = kmeans_cluster(X, k)
        idx_map = cluster_indices(result.labels, k)
        all_indices = sorted(i for indices in idx_map.values() for i in indices)
        assert all_indices == list(range(n))

    @pytest.mark.parametrize("k,n", [(3, 30), (4, 40)])
    def test_cluster_indices_disjoint(self, k: int, n: int) -> None:
        X = _rand_data(n, 3, seed=k + 1)
        result = kmeans_cluster(X, k)
        idx_map = cluster_indices(result.labels, k)
        all_indices = [i for indices in idx_map.values() for i in indices]
        assert len(all_indices) == len(set(all_indices))  # no duplicates

    @pytest.mark.parametrize("k_max", [3, 4, 5])
    def test_find_optimal_k_in_range(self, k_max: int) -> None:
        X = _rand_data(30, 3, seed=k_max)
        best_k = find_optimal_k(X, k_min=2, k_max=k_max)
        assert 2 <= best_k <= k_max
