"""Extra tests for puzzle_reconstruction/utils/spatial_index.py."""
from __future__ import annotations

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

def _pos(x, y) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


def _entry(item_id, x, y, payload=None) -> SpatialEntry:
    return SpatialEntry(item_id=item_id, position=_pos(x, y), payload=payload)


def _grid_positions(n=9) -> np.ndarray:
    k = int(np.sqrt(n))
    xs, ys = np.meshgrid(np.arange(k), np.arange(k))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


# ─── SpatialConfig ────────────────────────────────────────────────────────────

class TestSpatialConfigExtra:
    def test_default_cell_size(self):
        assert SpatialConfig().cell_size == pytest.approx(50.0)

    def test_default_metric(self):
        assert SpatialConfig().metric == "euclidean"

    def test_default_max_results(self):
        assert SpatialConfig().max_results == 0

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

    def test_valid_metrics(self):
        for m in ("euclidean", "manhattan", "chebyshev"):
            cfg = SpatialConfig(metric=m)
            assert cfg.metric == m

    def test_custom_cell_size(self):
        cfg = SpatialConfig(cell_size=25.0)
        assert cfg.cell_size == pytest.approx(25.0)


# ─── SpatialEntry ─────────────────────────────────────────────────────────────

class TestSpatialEntryExtra:
    def test_stores_item_id(self):
        e = _entry(5, 1.0, 2.0)
        assert e.item_id == 5

    def test_stores_position(self):
        e = _entry(0, 3.0, 4.0)
        np.testing.assert_array_equal(e.position, [3.0, 4.0])

    def test_position_dtype_float64(self):
        e = _entry(0, 1, 2)
        assert e.position.dtype == np.float64

    def test_negative_item_id_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=-1, position=_pos(0, 0))

    def test_wrong_position_shape_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=0, position=np.array([1.0, 2.0, 3.0]))

    def test_payload_stored(self):
        e = SpatialEntry(item_id=0, position=_pos(0, 0), payload="meta")
        assert e.payload == "meta"

    def test_default_payload_none(self):
        e = _entry(0, 0, 0)
        assert e.payload is None


# ─── SpatialIndex ─────────────────────────────────────────────────────────────

class TestSpatialIndexExtra:
    def test_empty_size_zero(self):
        idx = SpatialIndex()
        assert idx.size == 0

    def test_len_empty(self):
        idx = SpatialIndex()
        assert len(idx) == 0

    def test_insert_increases_size(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 0.0, 0.0))
        assert idx.size == 1

    def test_contains_after_insert(self):
        idx = SpatialIndex()
        idx.insert(_entry(7, 1.0, 1.0))
        assert 7 in idx

    def test_not_contains_missing(self):
        idx = SpatialIndex()
        assert 99 not in idx

    def test_remove_returns_true(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 0.0, 0.0))
        assert idx.remove(0) is True

    def test_remove_decreases_size(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 0.0, 0.0))
        idx.remove(0)
        assert idx.size == 0

    def test_remove_missing_returns_false(self):
        idx = SpatialIndex()
        assert idx.remove(42) is False

    def test_clear_empties(self):
        idx = SpatialIndex()
        for i in range(5):
            idx.insert(_entry(i, float(i), 0.0))
        idx.clear()
        assert idx.size == 0

    def test_get_all_returns_all(self):
        idx = SpatialIndex()
        for i in range(3):
            idx.insert(_entry(i, float(i), 0.0))
        assert len(idx.get_all()) == 3

    def test_config_accessible(self):
        cfg = SpatialConfig(cell_size=10.0)
        idx = SpatialIndex(cfg=cfg)
        assert idx.config.cell_size == pytest.approx(10.0)

    def test_query_radius_returns_list(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 0.0, 0.0))
        result = idx.query_radius(_pos(0.0, 0.0), 1.0)
        assert isinstance(result, list)

    def test_query_radius_finds_nearby(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 0.0, 0.0))
        idx.insert(_entry(1, 100.0, 100.0))
        result = idx.query_radius(_pos(0.0, 0.0), 1.0)
        ids = [e.item_id for _, e in result]
        assert 0 in ids
        assert 1 not in ids

    def test_query_radius_negative_raises(self):
        idx = SpatialIndex()
        with pytest.raises(ValueError):
            idx.query_radius(_pos(0.0, 0.0), -1.0)

    def test_query_radius_sorted(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 5.0, 0.0))
        idx.insert(_entry(1, 1.0, 0.0))
        result = idx.query_radius(_pos(0.0, 0.0), 10.0)
        dists = [d for d, _ in result]
        assert dists == sorted(dists)

    def test_query_knn_returns_list(self):
        idx = SpatialIndex()
        idx.insert(_entry(0, 0.0, 0.0))
        result = idx.query_knn(_pos(0.0, 0.0), 1)
        assert isinstance(result, list)

    def test_query_knn_k_lt_1_raises(self):
        idx = SpatialIndex()
        with pytest.raises(ValueError):
            idx.query_knn(_pos(0.0, 0.0), 0)

    def test_query_knn_returns_k_results(self):
        idx = SpatialIndex()
        for i in range(5):
            idx.insert(_entry(i, float(i), 0.0))
        result = idx.query_knn(_pos(0.0, 0.0), 3)
        assert len(result) == 3

    def test_query_knn_sorted(self):
        idx = SpatialIndex()
        for i in range(5):
            idx.insert(_entry(i, float(i * 2), 0.0))
        result = idx.query_knn(_pos(0.0, 0.0), 3)
        dists = [d for d, _ in result]
        assert dists == sorted(dists)

    def test_none_cfg_uses_defaults(self):
        idx = SpatialIndex(cfg=None)
        assert idx.config.cell_size == pytest.approx(50.0)


# ─── build_spatial_index ──────────────────────────────────────────────────────

class TestBuildSpatialIndexExtra:
    def test_returns_spatial_index(self):
        pos = _grid_positions(4)
        assert isinstance(build_spatial_index(pos), SpatialIndex)

    def test_size_equals_n(self):
        pos = _grid_positions(9)
        idx = build_spatial_index(pos)
        assert idx.size == 9

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            build_spatial_index(np.zeros((5, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            build_spatial_index(np.zeros(5))

    def test_payload_mismatch_raises(self):
        pos = _grid_positions(4)
        with pytest.raises(ValueError):
            build_spatial_index(pos, payloads=["a", "b"])  # only 2 payloads for 4 points

    def test_sequential_ids(self):
        pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        idx = build_spatial_index(pos)
        all_ids = sorted(e.item_id for e in idx.get_all())
        assert all_ids == [0, 1, 2]

    def test_with_payloads(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        idx = build_spatial_index(pos, payloads=["a", "b"])
        payloads = {e.item_id: e.payload for e in idx.get_all()}
        assert payloads[0] == "a"
        assert payloads[1] == "b"

    def test_custom_cfg(self):
        pos = _grid_positions(4)
        cfg = SpatialConfig(cell_size=1.0)
        idx = build_spatial_index(pos, cfg=cfg)
        assert idx.config.cell_size == pytest.approx(1.0)


# ─── module-level query wrappers ──────────────────────────────────────────────

class TestQueryWrappersExtra:
    def _make_index(self):
        pos = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 10.0]])
        return build_spatial_index(pos)

    def test_query_radius_returns_list(self):
        idx = self._make_index()
        result = query_radius(idx, _pos(0.0, 0.0), 5.0)
        assert isinstance(result, list)

    def test_query_radius_finds_close(self):
        idx = self._make_index()
        result = query_radius(idx, _pos(0.0, 0.0), 5.0)
        ids = [e.item_id for _, e in result]
        assert 0 in ids and 1 in ids
        assert 2 not in ids

    def test_query_knn_returns_list(self):
        idx = self._make_index()
        result = query_knn(idx, _pos(0.0, 0.0), 2)
        assert isinstance(result, list)

    def test_query_knn_length(self):
        idx = self._make_index()
        result = query_knn(idx, _pos(0.0, 0.0), 2)
        assert len(result) == 2


# ─── pairwise_distances ───────────────────────────────────────────────────────

class TestPairwiseDistancesExtra:
    def test_returns_ndarray(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert isinstance(pairwise_distances(pos), np.ndarray)

    def test_dtype_float64(self):
        pos = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert pairwise_distances(pos).dtype == np.float64

    def test_shape_n_n(self):
        pos = _grid_positions(4)
        out = pairwise_distances(pos)
        assert out.shape == (4, 4)

    def test_diagonal_zero(self):
        pos = _grid_positions(4)
        out = pairwise_distances(pos)
        assert np.all(np.diag(out) == 0.0)

    def test_symmetric(self):
        pos = _grid_positions(4)
        out = pairwise_distances(pos)
        np.testing.assert_allclose(out, out.T)

    def test_euclidean_known_value(self):
        pos = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = pairwise_distances(pos, metric="euclidean")
        assert out[0, 1] == pytest.approx(5.0)

    def test_manhattan_known_value(self):
        pos = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = pairwise_distances(pos, metric="manhattan")
        assert out[0, 1] == pytest.approx(7.0)

    def test_chebyshev_known_value(self):
        pos = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = pairwise_distances(pos, metric="chebyshev")
        assert out[0, 1] == pytest.approx(4.0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.zeros((5, 3)))

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(_grid_positions(4), metric="cosine")


# ─── cluster_by_distance ──────────────────────────────────────────────────────

class TestClusterByDistanceExtra:
    def test_returns_list(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert isinstance(cluster_by_distance(pos, 5.0), list)

    def test_empty_positions_returns_empty(self):
        result = cluster_by_distance(np.zeros((0, 2)), 1.0)
        assert result == []

    def test_negative_threshold_raises(self):
        pos = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            cluster_by_distance(pos, -1.0)

    def test_large_threshold_one_cluster(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = cluster_by_distance(pos, 100.0)
        assert len(result) == 1

    def test_zero_threshold_all_separate(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = cluster_by_distance(pos, 0.0)
        assert len(result) == 3

    def test_two_groups(self):
        pos = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        result = cluster_by_distance(pos, 1.0)
        assert len(result) == 2

    def test_all_indices_covered(self):
        pos = _grid_positions(9)
        result = cluster_by_distance(pos, 2.0)
        all_idx = sorted(i for cluster in result for i in cluster)
        assert all_idx == list(range(9))
