"""Тесты для puzzle_reconstruction.algorithms.path_planner."""
import math
import pytest
import numpy as np
from puzzle_reconstruction.algorithms.path_planner import (
    PathResult,
    dijkstra,
    shortest_path,
    all_pairs_shortest_paths,
    topological_sort,
    find_connected_components,
    minimum_spanning_tree,
    batch_dijkstra,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _chain(n=4) -> np.ndarray:
    """Chain graph: 0-1-2-...(n-1) with unit costs."""
    mat = np.zeros((n, n))
    for i in range(n - 1):
        mat[i, i + 1] = 1.0
        mat[i + 1, i] = 1.0
    return mat


def _complete(n=4, weight=1.0) -> np.ndarray:
    mat = np.full((n, n), weight)
    np.fill_diagonal(mat, 0.0)
    return mat


def _disconnected(n=4) -> np.ndarray:
    """Two isolated nodes + two connected."""
    mat = np.zeros((n, n))
    mat[0, 1] = mat[1, 0] = 1.0
    return mat


# ─── TestPathResult ───────────────────────────────────────────────────────────

class TestPathResult:
    def test_basic(self):
        r = PathResult(path=[0, 1, 2], cost=2.0)
        assert r.found is True
        assert r.cost == pytest.approx(2.0)

    def test_len(self):
        r = PathResult(path=[0, 1, 2], cost=2.0)
        assert len(r) == 3

    def test_empty_path(self):
        r = PathResult(path=[], cost=0.0, found=False)
        assert len(r) == 0
        assert r.found is False

    def test_cost_neg_raises(self):
        with pytest.raises(ValueError):
            PathResult(path=[0, 1], cost=-0.1)

    def test_cost_zero_ok(self):
        r = PathResult(path=[0], cost=0.0)
        assert r.cost == 0.0

    def test_found_default_true(self):
        r = PathResult(path=[0, 1], cost=1.0)
        assert r.found is True

    def test_params_stored(self):
        r = PathResult(path=[0], cost=0.0, params={"algorithm": "dijkstra"})
        assert r.params["algorithm"] == "dijkstra"


# ─── TestDijkstra ─────────────────────────────────────────────────────────────

class TestDijkstra:
    def test_returns_path_result(self):
        mat = _chain(4)
        r = dijkstra(mat, 0, 3)
        assert isinstance(r, PathResult)

    def test_direct_connection(self):
        mat = _chain(3)
        r = dijkstra(mat, 0, 1)
        assert r.found is True
        assert r.cost == pytest.approx(1.0)

    def test_multi_hop_path(self):
        mat = _chain(4)
        r = dijkstra(mat, 0, 3)
        assert r.found is True
        assert r.cost == pytest.approx(3.0)

    def test_start_equals_end(self):
        mat = _chain(4)
        r = dijkstra(mat, 2, 2)
        assert r.found is True
        assert r.cost == pytest.approx(0.0)
        assert r.path == [2]

    def test_no_path_found(self):
        mat = _disconnected(4)  # nodes 2,3 isolated
        r = dijkstra(mat, 0, 3)
        assert r.found is False
        assert r.path == []

    def test_path_starts_and_ends_correctly(self):
        mat = _chain(5)
        r = dijkstra(mat, 0, 4)
        assert r.path[0] == 0
        assert r.path[-1] == 4

    def test_non_square_raises(self):
        mat = np.ones((3, 4))
        with pytest.raises(ValueError):
            dijkstra(mat, 0, 2)

    def test_out_of_range_start_raises(self):
        mat = _chain(3)
        with pytest.raises(ValueError):
            dijkstra(mat, 5, 2)

    def test_out_of_range_end_raises(self):
        mat = _chain(3)
        with pytest.raises(ValueError):
            dijkstra(mat, 0, 10)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            dijkstra(np.ones(4), 0, 1)

    def test_shortest_among_alternatives(self):
        # Two paths: direct (cost=5) and indirect via node 1 (cost=2+2=4)
        mat = np.zeros((4, 4))
        mat[0, 3] = 5.0; mat[3, 0] = 5.0   # direct
        mat[0, 1] = 2.0; mat[1, 0] = 2.0
        mat[1, 3] = 2.0; mat[3, 1] = 2.0   # indirect: 0→1→3 costs 4
        r = dijkstra(mat, 0, 3)
        assert r.cost == pytest.approx(4.0)


# ─── TestShortestPath ─────────────────────────────────────────────────────────

class TestShortestPath:
    def test_returns_path_result(self):
        mat = _chain(4)
        r = shortest_path(mat, 0, 3)
        assert isinstance(r, PathResult)

    def test_found(self):
        mat = _chain(4)
        r = shortest_path(mat, 0, 3)
        assert r.found is True

    def test_no_path(self):
        mat = _disconnected(4)
        r = shortest_path(mat, 0, 3)
        assert r.found is False

    def test_cost_non_negative(self):
        mat = _chain(5)
        r = shortest_path(mat, 0, 4)
        assert r.cost >= 0.0


# ─── TestAllPairsShortestPaths ────────────────────────────────────────────────

class TestAllPairsShortestPaths:
    def test_returns_ndarray(self):
        mat = _chain(4)
        d = all_pairs_shortest_paths(mat)
        assert isinstance(d, np.ndarray)

    def test_shape_n_by_n(self):
        mat = _chain(4)
        d = all_pairs_shortest_paths(mat)
        assert d.shape == (4, 4)

    def test_diagonal_zero(self):
        mat = _chain(4)
        d = all_pairs_shortest_paths(mat)
        np.testing.assert_allclose(np.diag(d), 0.0)

    def test_connected_no_inf(self):
        mat = _complete(4)
        d = all_pairs_shortest_paths(mat)
        assert not np.any(np.isinf(d))

    def test_disconnected_has_inf(self):
        mat = _disconnected(4)
        d = all_pairs_shortest_paths(mat)
        assert np.any(np.isinf(d))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            all_pairs_shortest_paths(np.ones((3, 4)))

    def test_chain_distances(self):
        mat = _chain(3)
        d = all_pairs_shortest_paths(mat)
        # d[0,2] should be 2.0 (0→1→2)
        assert d[0, 2] == pytest.approx(2.0)


# ─── TestTopologicalSort ──────────────────────────────────────────────────────

class TestTopologicalSort:
    def _dag(self) -> np.ndarray:
        """DAG: 0→1, 0→2, 1→3, 2→3."""
        mat = np.zeros((4, 4))
        mat[0, 1] = mat[0, 2] = mat[1, 3] = mat[2, 3] = 1.0
        return mat

    def test_returns_list(self):
        r = topological_sort(self._dag())
        assert isinstance(r, list)

    def test_length_equals_n(self):
        r = topological_sort(self._dag())
        assert len(r) == 4

    def test_all_nodes_present(self):
        r = topological_sort(self._dag())
        assert sorted(r) == [0, 1, 2, 3]

    def test_precedence_respected(self):
        r = topological_sort(self._dag())
        # 0 must come before 1 and 2; 3 must be last
        assert r.index(0) < r.index(1)
        assert r.index(0) < r.index(2)
        assert r.index(1) < r.index(3)

    def test_cycle_raises(self):
        mat = np.zeros((3, 3))
        mat[0, 1] = mat[1, 2] = mat[2, 0] = 1.0  # cycle
        with pytest.raises(RuntimeError):
            topological_sort(mat)

    def test_empty_dag_ok(self):
        mat = np.zeros((0, 0))
        r = topological_sort(mat)
        assert r == []

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            topological_sort(np.ones((3, 4)))


# ─── TestFindConnectedComponents ──────────────────────────────────────────────

class TestFindConnectedComponents:
    def test_returns_list(self):
        r = find_connected_components(_chain(4))
        assert isinstance(r, list)

    def test_connected_graph_one_component(self):
        r = find_connected_components(_chain(4))
        assert len(r) == 1

    def test_disconnected_two_components(self):
        mat = _disconnected(4)
        r = find_connected_components(mat)
        assert len(r) == 3  # {0,1}, {2}, {3}

    def test_all_nodes_accounted(self):
        mat = _chain(5)
        comps = find_connected_components(mat)
        all_nodes = sorted(n for c in comps for n in c)
        assert all_nodes == [0, 1, 2, 3, 4]

    def test_isolated_nodes(self):
        mat = np.zeros((3, 3))  # no edges
        comps = find_connected_components(mat)
        assert len(comps) == 3

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            find_connected_components(np.ones((3, 4)))


# ─── TestMinimumSpanningTree ──────────────────────────────────────────────────

class TestMinimumSpanningTree:
    def test_returns_ndarray(self):
        mat = _complete(4)
        mst = minimum_spanning_tree(mat)
        assert isinstance(mst, np.ndarray)

    def test_shape(self):
        mat = _complete(4)
        mst = minimum_spanning_tree(mat)
        assert mst.shape == (4, 4)

    def test_symmetric(self):
        mat = _complete(4)
        mst = minimum_spanning_tree(mat)
        np.testing.assert_allclose(mst, mst.T)

    def test_n_edges_equals_n_minus_1(self):
        mat = _complete(5)
        mst = minimum_spanning_tree(mat)
        # Each edge counted twice (symmetric), n-1 edges → sum/weight
        n_edges = np.count_nonzero(mst) // 2
        assert n_edges == 4

    def test_empty_matrix_ok(self):
        mat = np.zeros((0, 0))
        mst = minimum_spanning_tree(mat)
        assert mst.shape == (0, 0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            minimum_spanning_tree(np.ones((3, 4)))

    def test_chain_mst_preserves_chain(self):
        mat = _chain(4)
        mst = minimum_spanning_tree(mat)
        # Chain is already an MST
        np.testing.assert_allclose(mst, mat)


# ─── TestBatchDijkstra ────────────────────────────────────────────────────────

class TestBatchDijkstra:
    def test_returns_list(self):
        mat = _chain(4)
        r = batch_dijkstra(mat, [0, 1], end=3)
        assert isinstance(r, list)

    def test_length_matches_starts(self):
        mat = _chain(5)
        r = batch_dijkstra(mat, [0, 1, 2], end=4)
        assert len(r) == 3

    def test_empty_starts(self):
        mat = _chain(4)
        assert batch_dijkstra(mat, [], end=3) == []

    def test_all_path_results(self):
        mat = _chain(4)
        for r in batch_dijkstra(mat, [0, 1], end=3):
            assert isinstance(r, PathResult)

    def test_costs_decrease_for_closer_starts(self):
        mat = _chain(5)
        results = batch_dijkstra(mat, [0, 2, 4], end=4)
        costs = [r.cost for r in results]
        assert costs[0] >= costs[1] >= costs[2]
