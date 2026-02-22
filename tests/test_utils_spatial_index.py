"""Tests for puzzle_reconstruction/utils/spatial_index.py"""
import math
import pytest
import numpy as np

from puzzle_reconstruction.utils.spatial_index import (
    SpatialConfig,
    SpatialEntry,
    SpatialIndex,
    build_spatial_index,
    query_radius,
    query_knn,
    pairwise_distances,
    cluster_by_distance,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _grid_positions(n=4) -> np.ndarray:
    """Return n*n points on a regular integer grid."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def _entry(item_id=0, x=0.0, y=0.0, payload=None) -> SpatialEntry:
    return SpatialEntry(item_id=item_id, position=np.array([x, y]), payload=payload)


def _small_index(n=5) -> SpatialIndex:
    """SpatialIndex with n entries at (i*10, 0) for i in range(n)."""
    idx = SpatialIndex()
    for i in range(n):
        idx.insert(_entry(item_id=i, x=float(i * 10)))
    return idx


# ─── TestSpatialConfig ────────────────────────────────────────────────────────

class TestSpatialConfig:
    def test_defaults(self):
        cfg = SpatialConfig()
        assert cfg.cell_size == pytest.approx(50.0)
        assert cfg.metric == "euclidean"
        assert cfg.max_results == 0

    def test_custom_values(self):
        cfg = SpatialConfig(cell_size=10.0, metric="manhattan", max_results=5)
        assert cfg.cell_size == pytest.approx(10.0)
        assert cfg.metric == "manhattan"
        assert cfg.max_results == 5

    def test_cell_size_zero_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(cell_size=0.0)

    def test_cell_size_negative_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(cell_size=-1.0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(metric="minkowski")

    def test_negative_max_results_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(max_results=-1)

    def test_zero_max_results_ok(self):
        cfg = SpatialConfig(max_results=0)
        assert cfg.max_results == 0

    def test_chebyshev_metric_ok(self):
        cfg = SpatialConfig(metric="chebyshev")
        assert cfg.metric == "chebyshev"


# ─── TestSpatialEntry ─────────────────────────────────────────────────────────

class TestSpatialEntry:
    def test_stores_item_id(self):
        e = _entry(item_id=7)
        assert e.item_id == 7

    def test_position_is_float64(self):
        e = _entry(x=3.0, y=4.0)
        assert e.position.dtype == np.float64

    def test_position_shape(self):
        e = _entry(x=1.0, y=2.0)
        assert e.position.shape == (2,)

    def test_payload_stored(self):
        e = _entry(payload={"key": "value"})
        assert e.payload == {"key": "value"}

    def test_negative_item_id_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=-1, position=np.array([0.0, 0.0]))

    def test_wrong_position_shape_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=0, position=np.array([1.0, 2.0, 3.0]))

    def test_1d_position_wrong_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=0, position=np.array([1.0]))

    def test_item_id_zero_ok(self):
        e = SpatialEntry(item_id=0, position=np.array([0.0, 0.0]))
        assert e.item_id == 0


# ─── TestSpatialIndex ─────────────────────────────────────────────────────────

class TestSpatialIndex:
    def test_initial_size_zero(self):
        idx = SpatialIndex()
        assert idx.size == 0

    def test_len_zero(self):
        assert len(SpatialIndex()) == 0

    def test_insert_increases_size(self):
        idx = SpatialIndex()
        idx.insert(_entry(0))
        assert idx.size == 1

    def test_insert_multiple(self):
        idx = _small_index(5)
        assert idx.size == 5

    def test_contains_inserted(self):
        idx = SpatialIndex()
        idx.insert(_entry(item_id=3))
        assert 3 in idx

    def test_contains_missing(self):
        idx = SpatialIndex()
        assert 99 not in idx

    def test_get_all_returns_list(self):
        idx = _small_index(3)
        entries = idx.get_all()
        assert isinstance(entries, list)
        assert len(entries) == 3

    def test_clear_empties_index(self):
        idx = _small_index(4)
        idx.clear()
        assert idx.size == 0
        assert idx.get_all() == []

    def test_remove_existing(self):
        idx = _small_index(3)
        removed = idx.remove(1)
        assert removed is True
        assert idx.size == 2
        assert 1 not in idx

    def test_remove_missing_returns_false(self):
        idx = _small_index(3)
        removed = idx.remove(99)
        assert removed is False

    def test_config_property(self):
        cfg = SpatialConfig(cell_size=25.0)
        idx = SpatialIndex(cfg)
        assert idx.config.cell_size == pytest.approx(25.0)

    def test_default_config_if_none(self):
        idx = SpatialIndex(None)
        assert idx.config.cell_size == pytest.approx(50.0)


# ─── TestSpatialIndexQueryRadius ──────────────────────────────────────────────

class TestSpatialIndexQueryRadius:
    def test_negative_radius_raises(self):
        idx = _small_index(3)
        with pytest.raises(ValueError):
            idx.query_radius(np.array([0.0, 0.0]), -1.0)

    def test_returns_list(self):
        idx = _small_index(3)
        result = idx.query_radius(np.array([0.0, 0.0]), 100.0)
        assert isinstance(result, list)

    def test_radius_zero_finds_exact(self):
        idx = _small_index(3)
        result = idx.query_radius(np.array([0.0, 0.0]), 0.0)
        assert len(result) >= 1
        assert result[0][1].item_id == 0

    def test_large_radius_finds_all(self):
        idx = _small_index(5)
        result = idx.query_radius(np.array([20.0, 0.0]), 1000.0)
        assert len(result) == 5

    def test_results_sorted_by_distance(self):
        idx = _small_index(5)
        result = idx.query_radius(np.array([0.0, 0.0]), 1000.0)
        dists = [r[0] for r in result]
        assert dists == sorted(dists)

    def test_results_within_radius(self):
        idx = _small_index(5)
        center = np.array([0.0, 0.0])
        radius = 15.0
        result = idx.query_radius(center, radius)
        for d, entry in result:
            assert d <= radius + 1e-10

    def test_max_results_limits_output(self):
        cfg = SpatialConfig(max_results=2)
        idx = SpatialIndex(cfg)
        for i in range(10):
            idx.insert(_entry(i, x=float(i)))
        result = idx.query_radius(np.array([5.0, 0.0]), 1000.0)
        assert len(result) <= 2

    def test_empty_index_returns_empty(self):
        idx = SpatialIndex()
        result = idx.query_radius(np.array([0.0, 0.0]), 100.0)
        assert result == []


# ─── TestSpatialIndexQueryKnn ─────────────────────────────────────────────────

class TestSpatialIndexQueryKnn:
    def test_k_zero_raises(self):
        idx = _small_index(3)
        with pytest.raises(ValueError):
            idx.query_knn(np.array([0.0, 0.0]), 0)

    def test_k_negative_raises(self):
        idx = _small_index(3)
        with pytest.raises(ValueError):
            idx.query_knn(np.array([0.0, 0.0]), -1)

    def test_returns_list(self):
        idx = _small_index(5)
        result = idx.query_knn(np.array([0.0, 0.0]), 3)
        assert isinstance(result, list)

    def test_k_capped_at_size(self):
        idx = _small_index(3)
        result = idx.query_knn(np.array([0.0, 0.0]), 10)
        assert len(result) == 3

    def test_returns_k_results(self):
        idx = _small_index(5)
        result = idx.query_knn(np.array([0.0, 0.0]), 3)
        assert len(result) == 3

    def test_results_sorted(self):
        idx = _small_index(5)
        result = idx.query_knn(np.array([25.0, 0.0]), 5)
        dists = [r[0] for r in result]
        assert dists == sorted(dists)

    def test_nearest_is_closest(self):
        idx = _small_index(5)
        result = idx.query_knn(np.array([0.0, 0.0]), 1)
        assert result[0][1].item_id == 0

    def test_empty_index_returns_empty(self):
        idx = SpatialIndex()
        result = idx.query_knn(np.array([0.0, 0.0]), 3)
        assert result == []


# ─── TestBuildSpatialIndex ────────────────────────────────────────────────────

class TestBuildSpatialIndex:
    def test_returns_spatial_index(self):
        pts = np.array([[0.0, 0.0], [1.0, 2.0]])
        idx = build_spatial_index(pts)
        assert isinstance(idx, SpatialIndex)

    def test_size_matches_n(self):
        pts = _grid_positions(3)
        idx = build_spatial_index(pts)
        assert idx.size == 9

    def test_wrong_shape_raises(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            build_spatial_index(pts)

    def test_1d_raises(self):
        pts = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            build_spatial_index(pts)

    def test_payloads_stored(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        payloads = ["a", "b"]
        idx = build_spatial_index(pts, payloads=payloads)
        entries = sorted(idx.get_all(), key=lambda e: e.item_id)
        assert entries[0].payload == "a"
        assert entries[1].payload == "b"

    def test_payloads_length_mismatch_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            build_spatial_index(pts, payloads=["only_one"])

    def test_custom_cfg_applied(self):
        pts = np.array([[0.0, 0.0]])
        cfg = SpatialConfig(cell_size=10.0)
        idx = build_spatial_index(pts, cfg=cfg)
        assert idx.config.cell_size == pytest.approx(10.0)

    def test_item_ids_sequential(self):
        pts = _grid_positions(2)
        idx = build_spatial_index(pts)
        ids = sorted(e.item_id for e in idx.get_all())
        assert ids == list(range(4))


# ─── TestQueryRadiusWrapper ───────────────────────────────────────────────────

class TestQueryRadiusWrapper:
    def test_returns_same_as_method(self):
        pts = _grid_positions(3)
        idx = build_spatial_index(pts)
        center = np.array([1.0, 1.0])
        r = 1.5
        assert query_radius(idx, center, r) == idx.query_radius(center, r)


# ─── TestQueryKnnWrapper ──────────────────────────────────────────────────────

class TestQueryKnnWrapper:
    def test_returns_same_as_method(self):
        pts = _grid_positions(3)
        idx = build_spatial_index(pts)
        center = np.array([1.0, 1.0])
        assert query_knn(idx, center, 3) == idx.query_knn(center, 3)


# ─── TestPairwiseDistances ────────────────────────────────────────────────────

class TestPairwiseDistances:
    def test_returns_ndarray(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = pairwise_distances(pts)
        assert isinstance(D, np.ndarray)

    def test_shape_n_by_n(self):
        pts = _grid_positions(3)
        D = pairwise_distances(pts)
        assert D.shape == (9, 9)

    def test_diagonal_zero(self):
        pts = _grid_positions(3)
        D = pairwise_distances(pts)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_symmetric(self):
        pts = _grid_positions(3)
        D = pairwise_distances(pts)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_euclidean_known_value(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(pts, metric="euclidean")
        assert D[0, 1] == pytest.approx(5.0)

    def test_manhattan_known_value(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(pts, metric="manhattan")
        assert D[0, 1] == pytest.approx(7.0)

    def test_chebyshev_known_value(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(pts, metric="chebyshev")
        assert D[0, 1] == pytest.approx(4.0)

    def test_wrong_shape_raises(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            pairwise_distances(pts)

    def test_invalid_metric_raises(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            pairwise_distances(pts, metric="cosine")

    def test_dtype_float64(self):
        pts = _grid_positions(2)
        D = pairwise_distances(pts)
        assert D.dtype == np.float64

    def test_nonneg_all_values(self):
        pts = _grid_positions(4)
        D = pairwise_distances(pts)
        assert (D >= 0).all()


# ─── TestClusterByDistance ────────────────────────────────────────────────────

class TestClusterByDistance:
    def test_returns_list(self):
        pts = _grid_positions(2)
        result = cluster_by_distance(pts, threshold=1.5)
        assert isinstance(result, list)

    def test_empty_positions(self):
        pts = np.zeros((0, 2))
        result = cluster_by_distance(pts, threshold=1.0)
        assert result == []

    def test_negative_threshold_raises(self):
        pts = _grid_positions(2)
        with pytest.raises(ValueError):
            cluster_by_distance(pts, threshold=-1.0)

    def test_zero_threshold_all_separate(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        clusters = cluster_by_distance(pts, threshold=0.0)
        assert len(clusters) == 3

    def test_large_threshold_one_cluster(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        clusters = cluster_by_distance(pts, threshold=100.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_total_points_preserved(self):
        pts = _grid_positions(3)
        clusters = cluster_by_distance(pts, threshold=1.5)
        total = sum(len(c) for c in clusters)
        assert total == 9

    def test_each_point_in_exactly_one_cluster(self):
        pts = _grid_positions(3)
        clusters = cluster_by_distance(pts, threshold=1.5)
        all_ids = [idx for c in clusters for idx in c]
        assert sorted(all_ids) == list(range(9))

    def test_two_separated_clusters(self):
        pts = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 0.0], [10.5, 0.0]])
        clusters = cluster_by_distance(pts, threshold=1.0)
        assert len(clusters) == 2

    def test_manhattan_metric(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        clusters = cluster_by_distance(pts, threshold=2.0, metric="manhattan")
        assert len(clusters) == 1

    def test_chebyshev_metric(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        clusters = cluster_by_distance(pts, threshold=2.0, metric="chebyshev")
        assert len(clusters) == 1

    def test_clusters_sorted_by_first_element(self):
        pts = np.array([[5.0, 0.0], [0.0, 0.0]])
        clusters = cluster_by_distance(pts, threshold=0.0)
        firsts = [c[0] for c in clusters]
        assert firsts == sorted(firsts)


# ─── TestMetricVariants ───────────────────────────────────────────────────────

class TestMetricVariants:
    """Test SpatialIndex with different metric configurations."""

    def test_manhattan_query_radius(self):
        cfg = SpatialConfig(metric="manhattan", cell_size=10.0)
        idx = SpatialIndex(cfg)
        idx.insert(_entry(0, x=0.0, y=0.0))
        idx.insert(_entry(1, x=3.0, y=0.0))
        # Manhattan dist = 3.0, so radius=2 should miss, radius=4 should hit
        assert len(idx.query_radius(np.array([0.0, 0.0]), 2.0)) == 1
        assert len(idx.query_radius(np.array([0.0, 0.0]), 4.0)) == 2

    def test_chebyshev_query_radius(self):
        cfg = SpatialConfig(metric="chebyshev", cell_size=10.0)
        idx = SpatialIndex(cfg)
        idx.insert(_entry(0, x=0.0, y=0.0))
        idx.insert(_entry(1, x=3.0, y=2.0))
        # Chebyshev dist = max(3,2) = 3
        result = idx.query_radius(np.array([0.0, 0.0]), 3.0)
        ids = {r[1].item_id for r in result}
        assert 1 in ids
