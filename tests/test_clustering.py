"""
Юнит-тесты для модуля puzzle_reconstruction/clustering.py.

Тесты покрывают:
    - _build_feature_matrix()   — размерность и значения дескрипторов
    - _fragment_features()      — деградация при отсутствующих полях
    - _estimate_n_clusters()    — автопоиск K
    - _run_clustering()         — все три метода (gmm, kmeans, spectral)
    - cluster_fragments()       — main API, граничные случаи
    - split_by_cluster()        — разбивка на подсписки
    - ClusteringResult.summary() — строковый вывод
"""
import math
import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn не установлен")

# ConvergenceWarning ожидаема: часть тестов создаёт идентичные/вырожденные
# фрагменты (одинаковый fd/brightness) и запрашивает k > числа реальных кластеров.
# Это намеренно — тесты проверяют структуру результата, а не качество кластеризации.
pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)

from puzzle_reconstruction.clustering import (
    cluster_fragments,
    split_by_cluster,
    ClusteringResult,
    _build_feature_matrix,
    _fragment_features,
    _run_clustering,
)
from puzzle_reconstruction.models import (
    Fragment, FractalSignature, TangramSignature, EdgeSignature,
    ShapeClass, EdgeSide,
)


# ─── Фикстуры ────────────────────────────────────────────────────────────

def _make_fractal(fd: float = 1.3) -> FractalSignature:
    return FractalSignature(
        fd_box=fd,
        fd_divider=fd + 0.05,
        ifs_coeffs=np.zeros(8),
        css_image=[(1.0, [0.1, 0.3]), (2.0, [0.2]), (4.0, [])],
        chain_code="01234567",
        curve=np.zeros((20, 2)),
    )


def _make_tangram(area: float = 0.5) -> TangramSignature:
    return TangramSignature(
        polygon=np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0,
        scale=1.0,
        area=area,
    )


def _make_fragment(frag_id: int,
                    fd: float = 1.3,
                    brightness: float = 200.0) -> Fragment:
    """Создаёт фрагмент с заданной фрактальной размерностью и яркостью."""
    h, w = 80, 60
    img = np.full((h, w, 3), int(brightness), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    contour = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=float)

    frag = Fragment(fragment_id=frag_id, image=img, mask=mask, contour=contour)
    frag.fractal = _make_fractal(fd=fd)
    frag.tangram = _make_tangram(area=0.5)

    # Края
    t = np.linspace(0, 1, 32)
    edge = EdgeSignature(
        edge_id=frag_id * 4,
        side=EdgeSide.TOP,
        virtual_curve=np.column_stack([t, np.zeros(32)]),
        fd=fd,
        css_vec=np.zeros(32),
        ifs_coeffs=np.zeros(8),
        length=100.0,
    )
    frag.edges = [edge]
    return frag


def _make_mixed_fragments(n_per_doc: int = 4,
                           n_docs: int = 2) -> list[Fragment]:
    """
    Создаёт n_per_doc * n_docs фрагментов из n_docs «документов»,
    каждый с чётко различающейся FD (1.1, 1.5, 1.8...).
    """
    fragments = []
    fid = 0
    for doc_id in range(n_docs):
        fd_target = 1.1 + doc_id * 0.4       # Разные FD для разных документов
        brightness = 150 + doc_id * 40        # Разная яркость
        for _ in range(n_per_doc):
            frag = _make_fragment(fid, fd=fd_target, brightness=brightness)
            fragments.append(frag)
            fid += 1
    return fragments


# ─── _fragment_features ───────────────────────────────────────────────────

class TestFragmentFeatures:

    def test_returns_1d_array(self):
        frag = _make_fragment(0)
        vec  = _fragment_features(frag)
        assert vec.ndim == 1
        assert len(vec) > 0

    def test_all_finite(self):
        frag = _make_fragment(0)
        vec  = _fragment_features(frag)
        assert np.all(np.isfinite(vec))

    def test_no_fractal_still_works(self):
        frag = _make_fragment(0)
        frag.fractal = None
        vec = _fragment_features(frag)
        assert vec.ndim == 1
        assert np.all(np.isfinite(vec))

    def test_no_tangram_still_works(self):
        frag = _make_fragment(0)
        frag.tangram = None
        vec = _fragment_features(frag)
        assert np.all(np.isfinite(vec))

    def test_no_image_still_works(self):
        frag = _make_fragment(0)
        frag.image = None
        vec = _fragment_features(frag)
        assert np.all(np.isfinite(vec))

    def test_no_contour_still_works(self):
        frag = _make_fragment(0)
        frag.contour = None
        vec = _fragment_features(frag)
        assert np.all(np.isfinite(vec))

    def test_different_fd_gives_different_vectors(self):
        f1 = _make_fragment(0, fd=1.1)
        f2 = _make_fragment(1, fd=1.9)
        v1 = _fragment_features(f1)
        v2 = _fragment_features(f2)
        assert not np.allclose(v1, v2)

    def test_consistent_length(self):
        """Все фрагменты должны давать вектор одинаковой длины."""
        lengths = [len(_fragment_features(_make_fragment(i))) for i in range(5)]
        assert len(set(lengths)) == 1


# ─── _build_feature_matrix ────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_shape(self):
        frags = [_make_fragment(i) for i in range(6)]
        mat   = _build_feature_matrix(frags)
        assert mat.shape[0] == 6
        assert mat.shape[1] > 0

    def test_dtype_float64(self):
        frags = [_make_fragment(i) for i in range(3)]
        mat   = _build_feature_matrix(frags)
        assert mat.dtype == np.float64


# ─── _run_clustering ──────────────────────────────────────────────────────

class TestRunClustering:

    def _X(self, n: int = 12, k: int = 2) -> np.ndarray:
        """Синтетические данные: k хорошо разделённых кластеров."""
        rng = np.random.RandomState(0)
        parts = [rng.randn(n // k, 5) + [i * 10, 0, 0, 0, 0]
                 for i in range(k)]
        return np.vstack(parts)

    def test_gmm_returns_correct_n_labels(self):
        X      = self._X(12, 2)
        labels, conf = _run_clustering(X, k=2, method="gmm", seed=0)
        assert len(labels) == 12
        assert len(set(labels)) <= 2

    def test_kmeans_labels_in_range(self):
        X      = self._X(12, 3)
        labels, conf = _run_clustering(X, k=3, method="kmeans", seed=0)
        assert all(0 <= l < 3 for l in labels)

    def test_spectral_labels_in_range(self):
        X      = self._X(12, 2)
        labels, conf = _run_clustering(X, k=2, method="spectral", seed=0)
        assert all(0 <= l < 2 for l in labels)

    def test_confidence_in_range(self):
        X = self._X(12, 2)
        for method in ("gmm", "kmeans", "spectral"):
            _, conf = _run_clustering(X, k=2, method=method, seed=0)
            assert np.all(conf >= 0.0) and np.all(conf <= 1.0), method

    def test_unknown_method_raises(self):
        X = self._X(6, 2)
        with pytest.raises(ValueError, match="Неизвестный метод"):
            _run_clustering(X, k=2, method="unknown_method", seed=0)


# ─── cluster_fragments ────────────────────────────────────────────────────

class TestClusterFragments:

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            cluster_fragments([])

    def test_single_fragment(self):
        frag   = _make_fragment(0)
        result = cluster_fragments([frag], k=1)
        assert result.n_clusters == 1
        assert len(result.labels) == 1
        assert result.labels[0] == 0

    def test_k_fixed_two_docs(self):
        frags  = _make_mixed_fragments(n_per_doc=4, n_docs=2)
        result = cluster_fragments(frags, k=2, method="kmeans", seed=0)
        assert result.n_clusters == 2
        assert len(result.labels) == 8

    def test_labels_count_matches_fragments(self):
        frags  = _make_mixed_fragments(n_per_doc=3, n_docs=2)
        result = cluster_fragments(frags, k=2)
        assert len(result.labels) == len(frags)

    def test_confidence_in_range(self):
        frags  = _make_mixed_fragments(n_per_doc=3, n_docs=2)
        result = cluster_fragments(frags, k=2)
        assert np.all(result.confidence >= 0.0)
        assert np.all(result.confidence <= 1.0)

    def test_cluster_groups_partition_all_ids(self):
        frags  = [_make_fragment(i) for i in range(6)]
        result = cluster_fragments(frags, k=2)
        all_ids_in_groups = sorted(
            fid for group in result.cluster_groups for fid in group
        )
        all_frag_ids = sorted(f.fragment_id for f in frags)
        assert all_ids_in_groups == all_frag_ids

    def test_silhouette_finite(self):
        frags  = _make_mixed_fragments(n_per_doc=4, n_docs=2)
        result = cluster_fragments(frags, k=2)
        assert math.isfinite(result.silhouette)

    def test_all_methods_work(self):
        frags = _make_mixed_fragments(n_per_doc=4, n_docs=2)
        for method in ("gmm", "kmeans", "spectral"):
            result = cluster_fragments(frags, k=2, method=method, seed=0)
            assert result.n_clusters == 2, method

    def test_auto_k_returns_sensible_value(self):
        """Автоопределение K на чётко разделённых данных."""
        frags  = _make_mixed_fragments(n_per_doc=6, n_docs=2)
        result = cluster_fragments(frags, k=None, k_max=5, seed=0)
        # Не можем гарантировать точный K=2 на синтетических данных,
        # но K должен быть в разумных пределах
        assert 1 <= result.n_clusters <= 5


# ─── ClusteringResult.summary ────────────────────────────────────────────

class TestClusteringResultSummary:

    def test_summary_contains_n_clusters(self):
        frags  = _make_mixed_fragments(n_per_doc=3, n_docs=2)
        result = cluster_fragments(frags, k=2)
        s = result.summary()
        assert "2" in s or "кластер" in s.lower()

    def test_summary_is_string(self):
        frag   = _make_fragment(0)
        result = cluster_fragments([frag], k=1)
        assert isinstance(result.summary(), str)


# ─── split_by_cluster ────────────────────────────────────────────────────

class TestSplitByCluster:

    def test_returns_k_groups(self):
        frags  = [_make_fragment(i) for i in range(6)]
        result = cluster_fragments(frags, k=3)
        groups = split_by_cluster(frags, result)
        assert len(groups) == 3

    def test_all_fragments_present(self):
        frags  = [_make_fragment(i) for i in range(8)]
        result = cluster_fragments(frags, k=2)
        groups = split_by_cluster(frags, result)
        found  = sorted(f.fragment_id for g in groups for f in g)
        expected = sorted(f.fragment_id for f in frags)
        assert found == expected

    def test_no_duplicates(self):
        frags  = [_make_fragment(i) for i in range(6)]
        result = cluster_fragments(frags, k=2)
        groups = split_by_cluster(frags, result)
        all_ids = [f.fragment_id for g in groups for f in g]
        assert len(all_ids) == len(set(all_ids))
