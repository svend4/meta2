"""Extra tests for puzzle_reconstruction.clustering."""
import math
import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn не установлен")

from puzzle_reconstruction.clustering import (
    ClusteringResult,
    _build_feature_matrix,
    _fragment_features,
    _run_clustering,
    cluster_fragments,
    split_by_cluster,
)
from puzzle_reconstruction.models import (
    EdgeSide,
    EdgeSignature,
    Fragment,
    FractalSignature,
    ShapeClass,
    TangramSignature,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_fractal(fd=1.3):
    return FractalSignature(
        fd_box=fd,
        fd_divider=fd + 0.05,
        ifs_coeffs=np.zeros(8),
        css_image=[(1.0, [0.1, 0.3]), (2.0, [0.2]), (4.0, [])],
        chain_code="01234567",
        curve=np.zeros((20, 2)),
    )


def _make_tangram(area=0.5):
    return TangramSignature(
        polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=area,
    )


def _make_fragment(frag_id, fd=1.3, brightness=200.0, n_edges=1):
    h, w = 80, 60
    img = np.full((h, w, 3), int(brightness), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    frag = Fragment(fragment_id=frag_id, image=img, mask=mask, contour=contour)
    frag.fractal = _make_fractal(fd=fd)
    frag.tangram = _make_tangram(area=0.5)
    t = np.linspace(0, 1, 32)
    frag.edges = [
        EdgeSignature(
            edge_id=frag_id * 4 + i,
            side=EdgeSide.TOP,
            virtual_curve=np.column_stack([t, np.zeros(32)]),
            fd=fd,
            css_vec=np.zeros(32),
            ifs_coeffs=np.zeros(8),
            length=100.0,
        )
        for i in range(n_edges)
    ]
    return frag


def _mixed(n_per=4, n_docs=2):
    frags = []
    fid = 0
    for doc in range(n_docs):
        fd = 1.1 + doc * 0.4
        br = 150 + doc * 40
        for _ in range(n_per):
            frags.append(_make_fragment(fid, fd=fd, brightness=br))
            fid += 1
    return frags


# ─── _fragment_features extras ───────────────────────────────────────────────

class TestFragmentFeaturesExtra:
    def test_no_edges_still_works(self):
        frag = _make_fragment(0)
        frag.edges = []
        vec = _fragment_features(frag)
        assert vec.ndim == 1
        assert np.all(np.isfinite(vec))

    def test_multiple_edges(self):
        frag = _make_fragment(0, n_edges=3)
        vec = _fragment_features(frag)
        assert np.all(np.isfinite(vec))

    def test_different_brightness_different_vectors(self):
        f1 = _make_fragment(0, brightness=50.0)
        f2 = _make_fragment(1, brightness=200.0)
        v1 = _fragment_features(f1)
        v2 = _fragment_features(f2)
        assert not np.allclose(v1, v2)

    def test_length_same_for_10_fragments(self):
        lengths = set(len(_fragment_features(_make_fragment(i))) for i in range(10))
        assert len(lengths) == 1

    def test_all_finite_various_fd(self):
        for fd in (1.0, 1.5, 2.0):
            vec = _fragment_features(_make_fragment(0, fd=fd))
            assert np.all(np.isfinite(vec))

    def test_returns_float64(self):
        vec = _fragment_features(_make_fragment(0))
        assert vec.dtype == np.float64


# ─── _build_feature_matrix extras ────────────────────────────────────────────

class TestBuildFeatureMatrixExtra:
    def test_single_fragment(self):
        mat = _build_feature_matrix([_make_fragment(0)])
        assert mat.shape[0] == 1
        assert mat.shape[1] > 0

    def test_ten_fragments(self):
        frags = [_make_fragment(i) for i in range(10)]
        mat = _build_feature_matrix(frags)
        assert mat.shape == (10, mat.shape[1])

    def test_all_same_fd_all_rows_equal(self):
        frags = [_make_fragment(i, fd=1.4, brightness=150.0) for i in range(4)]
        mat = _build_feature_matrix(frags)
        # All rows should be nearly identical
        np.testing.assert_allclose(mat[0], mat[1], atol=1e-6)

    def test_dtype_float64(self):
        mat = _build_feature_matrix([_make_fragment(0)])
        assert mat.dtype == np.float64

    def test_no_nan_inf(self):
        frags = [_make_fragment(i) for i in range(5)]
        mat = _build_feature_matrix(frags)
        assert np.all(np.isfinite(mat))


# ─── _run_clustering extras ───────────────────────────────────────────────────

class TestRunClusteringExtra:
    def _X(self, n=12, k=2, seed=0):
        rng = np.random.RandomState(seed)
        parts = [rng.randn(n // k, 5) + [i * 10, 0, 0, 0, 0] for i in range(k)]
        return np.vstack(parts)

    def test_k_1_all_same_label(self):
        X = self._X(8, 2)
        labels, conf = _run_clustering(X, k=1, method="kmeans", seed=0)
        assert all(l == 0 for l in labels)

    def test_confidence_gmm_range(self):
        X = self._X(12, 3)
        _, conf = _run_clustering(X, k=3, method="gmm", seed=0)
        assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    def test_confidence_spectral_range(self):
        X = self._X(10, 2)
        _, conf = _run_clustering(X, k=2, method="spectral", seed=0)
        assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    def test_label_count_matches_rows(self):
        X = self._X(12, 3)
        for method in ("gmm", "kmeans", "spectral"):
            labels, _ = _run_clustering(X, k=3, method=method, seed=0)
            assert len(labels) == 12

    def test_k_2_two_distinct_labels(self):
        X = self._X(12, 2)
        labels, _ = _run_clustering(X, k=2, method="kmeans", seed=0)
        assert len(set(labels)) == 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _run_clustering(self._X(), k=2, method="bad", seed=0)


# ─── cluster_fragments extras ─────────────────────────────────────────────────

class TestClusterFragmentsExtra:
    def test_k_1_all_in_one_cluster(self):
        frags = [_make_fragment(i) for i in range(5)]
        result = cluster_fragments(frags, k=1)
        assert result.n_clusters == 1
        assert all(l == 0 for l in result.labels)

    def test_k_3_three_clusters(self):
        frags = _mixed(n_per=3, n_docs=3)
        result = cluster_fragments(frags, k=3, method="kmeans", seed=0)
        assert result.n_clusters == 3

    def test_result_has_silhouette(self):
        frags = _mixed(n_per=4, n_docs=2)
        result = cluster_fragments(frags, k=2)
        assert math.isfinite(result.silhouette)

    def test_auto_k_in_range(self):
        frags = _mixed(n_per=6, n_docs=2)
        result = cluster_fragments(frags, k=None, k_max=4, seed=0)
        assert 1 <= result.n_clusters <= 4

    def test_confidence_all_in_range(self):
        frags = _mixed(n_per=4, n_docs=2)
        result = cluster_fragments(frags, k=2)
        assert np.all(result.confidence >= 0.0)
        assert np.all(result.confidence <= 1.0)

    def test_cluster_groups_non_overlapping(self):
        frags = [_make_fragment(i) for i in range(6)]
        result = cluster_fragments(frags, k=2)
        all_ids = [fid for g in result.cluster_groups for fid in g]
        assert len(all_ids) == len(set(all_ids))

    def test_labels_integers(self):
        frags = [_make_fragment(i) for i in range(4)]
        result = cluster_fragments(frags, k=2)
        assert all(isinstance(l, (int, np.integer)) for l in result.labels)

    def test_all_methods_three_clusters(self):
        frags = _mixed(n_per=3, n_docs=3)
        for method in ("gmm", "kmeans", "spectral"):
            result = cluster_fragments(frags, k=3, method=method, seed=0)
            assert result.n_clusters == 3, f"method={method}"


# ─── ClusteringResult.summary extras ─────────────────────────────────────────

class TestClusteringResultSummaryExtra:
    def test_summary_is_string(self):
        frags = _mixed(n_per=4, n_docs=2)
        result = cluster_fragments(frags, k=2)
        assert isinstance(result.summary(), str)

    def test_summary_nonempty(self):
        frags = [_make_fragment(0)]
        result = cluster_fragments(frags, k=1)
        assert len(result.summary()) > 0

    def test_summary_contains_cluster_count(self):
        frags = _mixed(n_per=3, n_docs=3)
        result = cluster_fragments(frags, k=3)
        s = result.summary()
        assert "3" in s or "кластер" in s.lower()

    def test_summary_consistent(self):
        frags = _mixed(n_per=4, n_docs=2)
        result = cluster_fragments(frags, k=2)
        s1 = result.summary()
        s2 = result.summary()
        assert s1 == s2


# ─── split_by_cluster extras ─────────────────────────────────────────────────

class TestSplitByClusterExtra:
    def test_k_1_single_group(self):
        frags = [_make_fragment(i) for i in range(5)]
        result = cluster_fragments(frags, k=1)
        groups = split_by_cluster(frags, result)
        assert len(groups) == 1
        assert len(groups[0]) == 5

    def test_k_3_three_groups(self):
        frags = _mixed(n_per=3, n_docs=3)
        result = cluster_fragments(frags, k=3, method="kmeans", seed=0)
        groups = split_by_cluster(frags, result)
        assert len(groups) == 3

    def test_all_fragments_in_groups(self):
        frags = [_make_fragment(i) for i in range(9)]
        result = cluster_fragments(frags, k=3)
        groups = split_by_cluster(frags, result)
        found = sorted(f.fragment_id for g in groups for f in g)
        expected = sorted(f.fragment_id for f in frags)
        assert found == expected

    def test_no_duplicates_k3(self):
        frags = [_make_fragment(i) for i in range(9)]
        result = cluster_fragments(frags, k=3)
        groups = split_by_cluster(frags, result)
        all_ids = [f.fragment_id for g in groups for f in g]
        assert len(all_ids) == len(set(all_ids))

    def test_returns_list_of_lists(self):
        frags = [_make_fragment(i) for i in range(4)]
        result = cluster_fragments(frags, k=2)
        groups = split_by_cluster(frags, result)
        assert isinstance(groups, list)
        for g in groups:
            assert isinstance(g, list)
