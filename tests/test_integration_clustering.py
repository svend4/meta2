"""
Integration tests for puzzle_reconstruction.clustering module.

~60 tests across 5 classes covering:
- ClusteringResult data class
- cluster_fragments with single fragment
- cluster_fragments with method="kmeans"
- cluster_fragments with method="gmm"
- split_by_cluster utility
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Fragment,
    FractalSignature,
    TangramSignature,
    ShapeClass,
)
from puzzle_reconstruction.clustering import (
    ClusteringResult,
    cluster_fragments,
    split_by_cluster,
)


# ─── Fragment factory ─────────────────────────────────────────────────────────

def make_fragment(fid: int, seed: int = 0) -> Fragment:
    """Create a minimal but valid Fragment with fractal and tangram signatures."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
    mask = np.ones((60, 60), dtype=np.uint8) * 255
    contour = np.array([[0, 0], [60, 0], [60, 60], [0, 60]], dtype=np.int32)

    fractal = FractalSignature(
        fd_box=1.3 + seed * 0.1,
        fd_divider=1.2 + seed * 0.05,
        ifs_coeffs=np.zeros(6),
        css_image=[],
        chain_code="",
        curve=np.zeros((4, 2)),
    )
    tangram = TangramSignature(
        polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=0.5 + seed * 0.01,
    )
    return Fragment(
        fragment_id=fid,
        image=img,
        mask=mask,
        contour=contour,
        edges=[],
        tangram=tangram,
        fractal=fractal,
    )


def make_distinct_fragments(n: int, n_clusters: int, base_seed: int = 0) -> list:
    """
    Create n fragments with features that naturally form n_clusters groups.
    Fragments in the same cluster share similar fractal fd_box values;
    clusters are separated by a large step in fd_box.
    """
    fragments = []
    for i in range(n):
        cluster_idx = i % n_clusters
        rng = np.random.default_rng(base_seed + i)
        img = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
        mask = np.ones((60, 60), dtype=np.uint8) * 255
        contour = np.array([[0, 0], [60, 0], [60, 60], [0, 60]], dtype=np.int32)

        # Large separation between clusters in the feature space
        fd_box = 1.0 + cluster_idx * 0.5 + rng.uniform(-0.05, 0.05)
        fd_divider = 1.0 + cluster_idx * 0.5 + rng.uniform(-0.05, 0.05)

        fractal = FractalSignature(
            fd_box=fd_box,
            fd_divider=fd_divider,
            ifs_coeffs=np.zeros(6),
            css_image=[],
            chain_code="",
            curve=np.zeros((4, 2)),
        )
        tangram = TangramSignature(
            polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
            shape_class=ShapeClass.RECTANGLE,
            centroid=np.array([0.5, 0.5]),
            angle=0.0,
            scale=1.0,
            area=0.5 + cluster_idx * 0.2,
        )
        fragments.append(Fragment(
            fragment_id=i,
            image=img,
            mask=mask,
            contour=contour,
            edges=[],
            tangram=tangram,
            fractal=fractal,
        ))
    return fragments


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestClusteringResult (~10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusteringResult:
    """Tests for the ClusteringResult dataclass and its summary() method."""

    def _make_result(self, n: int = 4, k: int = 2) -> ClusteringResult:
        labels = np.array([i % k for i in range(n)])
        confidence = np.ones(n) * 0.9
        groups: list = [[] for _ in range(k)]
        for i, lbl in enumerate(labels):
            groups[lbl].append(i)
        return ClusteringResult(
            labels=labels,
            n_clusters=k,
            silhouette=0.5,
            confidence=confidence,
            cluster_groups=groups,
        )

    def test_labels_stored(self):
        result = self._make_result()
        assert len(result.labels) == 4

    def test_n_clusters_stored(self):
        result = self._make_result(n=6, k=3)
        assert result.n_clusters == 3

    def test_silhouette_stored(self):
        result = self._make_result()
        assert result.silhouette == pytest.approx(0.5)

    def test_confidence_stored(self):
        result = self._make_result(n=4, k=2)
        assert len(result.confidence) == 4
        assert all(c == pytest.approx(0.9) for c in result.confidence)

    def test_cluster_groups_stored(self):
        result = self._make_result(n=4, k=2)
        assert len(result.cluster_groups) == 2

    def test_cluster_groups_cover_all_ids(self):
        result = self._make_result(n=4, k=2)
        all_ids = [fid for g in result.cluster_groups for fid in g]
        assert sorted(all_ids) == [0, 1, 2, 3]

    def test_summary_returns_string(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str)

    def test_summary_contains_n_clusters(self):
        result = self._make_result(n=4, k=2)
        s = result.summary()
        assert "2" in s

    def test_summary_contains_silhouette(self):
        result = self._make_result()
        s = result.summary()
        # Silhouette value 0.5 should appear somewhere in summary
        assert "0.5" in s or "0.500" in s or "+0.500" in s

    def test_summary_nonempty(self):
        result = self._make_result(n=6, k=3)
        s = result.summary()
        assert len(s) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestClusterFragmentsSingle (~10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterFragmentsSingle:
    """Tests for cluster_fragments with a single fragment."""

    def _single(self) -> ClusteringResult:
        frag = make_fragment(fid=0, seed=7)
        return cluster_fragments([frag])

    def test_n_clusters_is_one(self):
        result = self._single()
        assert result.n_clusters == 1

    def test_labels_has_single_element(self):
        result = self._single()
        assert len(result.labels) == 1

    def test_label_is_zero(self):
        result = self._single()
        assert result.labels[0] == 0

    def test_confidence_is_one(self):
        result = self._single()
        assert result.confidence[0] == pytest.approx(1.0)

    def test_cluster_groups_has_one_group(self):
        result = self._single()
        assert len(result.cluster_groups) == 1

    def test_cluster_group_contains_fragment_id(self):
        frag = make_fragment(fid=42, seed=3)
        result = cluster_fragments([frag])
        assert 42 in result.cluster_groups[0]

    def test_silhouette_is_float(self):
        result = self._single()
        assert isinstance(result.silhouette, float)

    def test_labels_is_ndarray(self):
        result = self._single()
        assert isinstance(result.labels, np.ndarray)

    def test_confidence_is_ndarray(self):
        result = self._single()
        assert isinstance(result.confidence, np.ndarray)

    def test_cluster_groups_is_list_of_lists(self):
        result = self._single()
        assert isinstance(result.cluster_groups, list)
        assert isinstance(result.cluster_groups[0], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestClusterFragmentsKMeans (~15 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterFragmentsKMeans:
    """Integration tests for cluster_fragments with method='kmeans'."""

    def _frags6(self) -> list:
        return make_distinct_fragments(n=6, n_clusters=2, base_seed=0)

    def test_n_clusters_equals_k(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert result.n_clusters == 2

    def test_labels_length_matches_fragments(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert len(result.labels) == len(frags)

    def test_labels_are_integers(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert result.labels.dtype in (np.int32, np.int64, int)

    def test_labels_in_valid_range(self):
        frags = self._frags6()
        k = 2
        result = cluster_fragments(frags, k=k, method="kmeans", seed=42)
        assert all(0 <= lbl < k for lbl in result.labels)

    def test_confidence_length_matches_fragments(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert len(result.confidence) == len(frags)

    def test_confidence_values_in_unit_interval(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert all(0.0 <= c <= 1.0 for c in result.confidence)

    def test_cluster_groups_count_equals_k(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert len(result.cluster_groups) == 2

    def test_cluster_groups_cover_all_fragment_ids(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        all_ids = sorted(fid for g in result.cluster_groups for fid in g)
        expected = sorted(f.fragment_id for f in frags)
        assert all_ids == expected

    def test_no_fragment_in_multiple_groups(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        all_ids = [fid for g in result.cluster_groups for fid in g]
        assert len(all_ids) == len(set(all_ids))

    def test_k3_returns_three_clusters(self):
        frags = make_distinct_fragments(n=9, n_clusters=3, base_seed=10)
        result = cluster_fragments(frags, k=3, method="kmeans", seed=42)
        assert result.n_clusters == 3

    def test_k3_labels_in_range(self):
        frags = make_distinct_fragments(n=9, n_clusters=3, base_seed=10)
        result = cluster_fragments(frags, k=3, method="kmeans", seed=42)
        assert all(0 <= lbl < 3 for lbl in result.labels)

    def test_empty_fragments_raises_value_error(self):
        with pytest.raises(ValueError):
            cluster_fragments([], method="kmeans")

    def test_returns_clustering_result_instance(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert isinstance(result, ClusteringResult)

    def test_silhouette_is_float(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        assert isinstance(result.silhouette, float)

    def test_summary_callable_on_result(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="kmeans", seed=42)
        s = result.summary()
        assert isinstance(s, str) and len(s) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestClusterFragmentsGMM (~15 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterFragmentsGMM:
    """Integration tests for cluster_fragments with method='gmm'."""

    def _frags6(self) -> list:
        return make_distinct_fragments(n=6, n_clusters=2, base_seed=20)

    def test_n_clusters_equals_k(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert result.n_clusters == 2

    def test_labels_length_matches_fragments(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert len(result.labels) == len(frags)

    def test_labels_are_integers(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert result.labels.dtype in (np.int32, np.int64, int)

    def test_labels_in_valid_range(self):
        frags = self._frags6()
        k = 2
        result = cluster_fragments(frags, k=k, method="gmm", seed=42)
        assert all(0 <= lbl < k for lbl in result.labels)

    def test_confidence_length_matches_fragments(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert len(result.confidence) == len(frags)

    def test_confidence_values_in_unit_interval(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert all(0.0 <= c <= 1.0 for c in result.confidence)

    def test_cluster_groups_count_equals_k(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert len(result.cluster_groups) == 2

    def test_cluster_groups_cover_all_fragment_ids(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        all_ids = sorted(fid for g in result.cluster_groups for fid in g)
        expected = sorted(f.fragment_id for f in frags)
        assert all_ids == expected

    def test_no_fragment_in_multiple_groups(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        all_ids = [fid for g in result.cluster_groups for fid in g]
        assert len(all_ids) == len(set(all_ids))

    def test_silhouette_in_valid_range(self):
        frags = self._frags6()
        result = cluster_fragments(frags, k=2, method="gmm", seed=42)
        assert -1.0 <= result.silhouette <= 1.0

    def test_k3_returns_three_clusters(self):
        frags = make_distinct_fragments(n=9, n_clusters=3, base_seed=30)
        result = cluster_fragments(frags, k=3, method="gmm", seed=42)
        assert result.n_clusters == 3

    def test_k3_cluster_groups_count(self):
        frags = make_distinct_fragments(n=9, n_clusters=3, base_seed=30)
        result = cluster_fragments(frags, k=3, method="gmm", seed=42)
        assert len(result.cluster_groups) == 3

    def test_k_none_auto_detect_returns_clustering_result(self):
        frags = make_distinct_fragments(n=6, n_clusters=2, base_seed=40)
        result = cluster_fragments(frags, k=None, k_max=4, method="gmm", seed=42)
        assert isinstance(result, ClusteringResult)

    def test_k_none_n_clusters_at_least_one(self):
        frags = make_distinct_fragments(n=6, n_clusters=2, base_seed=40)
        result = cluster_fragments(frags, k=None, k_max=4, method="gmm", seed=42)
        assert result.n_clusters >= 1

    def test_empty_fragments_raises_value_error(self):
        with pytest.raises(ValueError):
            cluster_fragments([], method="gmm")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestSplitByCluster (~10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitByCluster:
    """Tests for the split_by_cluster utility function."""

    def _setup(self, n: int = 6, k: int = 2) -> tuple:
        frags = make_distinct_fragments(n=n, n_clusters=k, base_seed=50)
        result = cluster_fragments(frags, k=k, method="kmeans", seed=42)
        return frags, result

    def test_returns_list(self):
        frags, result = self._setup()
        groups = split_by_cluster(frags, result)
        assert isinstance(groups, list)

    def test_returns_k_sublists(self):
        frags, result = self._setup(n=6, k=2)
        groups = split_by_cluster(frags, result)
        assert len(groups) == 2

    def test_sublists_contain_fragment_objects(self):
        frags, result = self._setup(n=6, k=2)
        groups = split_by_cluster(frags, result)
        for group in groups:
            for item in group:
                assert isinstance(item, Fragment)

    def test_union_of_sublists_covers_all_fragments(self):
        frags, result = self._setup(n=6, k=2)
        groups = split_by_cluster(frags, result)
        all_fids = sorted(f.fragment_id for group in groups for f in group)
        expected = sorted(f.fragment_id for f in frags)
        assert all_fids == expected

    def test_no_duplicate_fragments_across_groups(self):
        frags, result = self._setup(n=6, k=2)
        groups = split_by_cluster(frags, result)
        all_fids = [f.fragment_id for group in groups for f in group]
        assert len(all_fids) == len(set(all_fids))

    def test_k3_returns_three_sublists(self):
        frags = make_distinct_fragments(n=9, n_clusters=3, base_seed=60)
        result = cluster_fragments(frags, k=3, method="kmeans", seed=42)
        groups = split_by_cluster(frags, result)
        assert len(groups) == 3

    def test_total_fragment_count_preserved(self):
        frags, result = self._setup(n=6, k=2)
        groups = split_by_cluster(frags, result)
        total = sum(len(g) for g in groups)
        assert total == len(frags)

    def test_single_fragment_returns_one_group(self):
        frag = make_fragment(fid=0, seed=5)
        result = cluster_fragments([frag])
        groups = split_by_cluster([frag], result)
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0].fragment_id == 0

    def test_unknown_frag_id_in_cluster_groups_handled_gracefully(self):
        frags = [make_fragment(fid=i, seed=i) for i in range(4)]
        # Construct a result with an unknown frag_id in cluster_groups
        labels = np.array([0, 0, 1, 1])
        confidence = np.ones(4)
        cluster_groups = [[0, 1, 999], [2, 3]]  # 999 is not a real fragment
        result = ClusteringResult(
            labels=labels,
            n_clusters=2,
            silhouette=0.5,
            confidence=confidence,
            cluster_groups=cluster_groups,
        )
        groups = split_by_cluster(frags, result)
        # 999 should be silently ignored; real fragments should be present
        all_fids = [f.fragment_id for group in groups for f in group]
        assert 999 not in all_fids
        assert sorted(all_fids) == [0, 1, 2, 3]

    def test_empty_group_possible_when_cluster_group_is_empty(self):
        frags = [make_fragment(fid=i, seed=i) for i in range(2)]
        labels = np.array([0, 0])
        confidence = np.ones(2)
        cluster_groups = [[0, 1], []]  # second cluster is empty
        result = ClusteringResult(
            labels=labels,
            n_clusters=2,
            silhouette=0.0,
            confidence=confidence,
            cluster_groups=cluster_groups,
        )
        groups = split_by_cluster(frags, result)
        assert len(groups) == 2
        assert groups[1] == []
