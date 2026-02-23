"""Extra tests for puzzle_reconstruction/algorithms/path_planner.py"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.path_planner import (
    PathResult,
    all_pairs_shortest_paths,
    batch_dijkstra,
    dijkstra,
    find_connected_components,
    minimum_spanning_tree,
    shortest_path,
    topological_sort,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _chain_cost(n: int, w: float = 1.0) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        mat[i, i + 1] = w
    return mat


def _undirected_chain(n: int, w: float = 1.0) -> np.ndarray:
    mat = _chain_cost(n, w)
    return mat + mat.T


def _disconnected(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def _complete_graph(n: int, w: float = 1.0) -> np.ndarray:
    mat = np.full((n, n), w, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    return mat


# ─── TestPathResultExtra ─────────────────────────────────────────────────────

class TestPathResultExtra:
    def test_params_stored(self):
        r = PathResult(path=[0, 1], cost=1.0, params={"algorithm": "dijkstra"})
        assert r.params["algorithm"] == "dijkstra"

    def test_single_node_path(self):
        r = PathResult(path=[0], cost=0.0)
        assert len(r) == 1
        assert r.cost == pytest.approx(0.0)

    def test_long_path(self):
        path = list(range(20))
        r = PathResult(path=path, cost=19.0)
        assert len(r) == 20

    def test_found_defaults_true(self):
        r = PathResult(path=[0, 1, 2], cost=2.0)
        assert r.found is True

    def test_not_found_path_empty(self):
        r = PathResult(path=[], cost=0.0, found=False)
        assert len(r) == 0
        assert r.found is False

    def test_cost_float(self):
        r = PathResult(path=[0], cost=3.14)
        assert isinstance(r.cost, float)


# ─── TestDijkstraExtra ───────────────────────────────────────────────────────

class TestDijkstraExtra:
    def test_single_node_graph(self):
        mat = np.zeros((1, 1), dtype=np.float64)
        r = dijkstra(mat, 0, 0)
        assert r.found is True
        assert r.path == [0]
        assert r.cost == pytest.approx(0.0)

    def test_two_nodes_direct(self):
        mat = np.array([[0, 2.0], [0, 0]], dtype=np.float64)
        r = dijkstra(mat, 0, 1)
        assert r.found is True
        assert r.cost == pytest.approx(2.0)
        assert r.path == [0, 1]

    def test_alternative_path_lower_cost(self):
        # 0->1 costs 10, 0->2->1 costs 1+1=2
        mat = np.zeros((3, 3), dtype=np.float64)
        mat[0, 1] = 10.0
        mat[0, 2] = 1.0
        mat[2, 1] = 1.0
        r = dijkstra(mat, 0, 1)
        assert r.cost == pytest.approx(2.0)
        assert 2 in r.path

    def test_five_node_chain_cost(self):
        mat = _chain_cost(5, w=3.0)
        r = dijkstra(mat, 0, 4)
        assert r.cost == pytest.approx(12.0)
        assert r.path == [0, 1, 2, 3, 4]

    def test_weighted_graph_picks_cheaper_route(self):
        # 0->1 costs 5, 0->2->1 costs 1+1=2 → should pick 0->2->1
        mat = np.zeros((3, 3), dtype=np.float64)
        mat[0, 1] = 5.0
        mat[0, 2] = 1.0
        mat[2, 1] = 1.0
        r = dijkstra(mat, 0, 1)
        assert r.cost == pytest.approx(2.0)

    def test_algorithm_param_dijkstra(self):
        mat = _chain_cost(3)
        r = dijkstra(mat, 0, 2)
        assert r.params.get("algorithm") == "dijkstra"

    def test_returns_path_result_type(self):
        mat = _chain_cost(3)
        r = dijkstra(mat, 0, 2)
        assert isinstance(r, PathResult)


# ─── TestShortestPathExtra ────────────────────────────────────────────────────

class TestShortestPathExtra:
    def _score_mat(self, n, s=0.9):
        mat = np.zeros((n, n))
        for i in range(n - 1):
            mat[i, i + 1] = s
        return mat

    def test_single_node(self):
        mat = np.zeros((1, 1))
        r = shortest_path(mat, 0, 0)
        assert r.found is True
        assert r.path == [0]

    def test_two_node_score_path(self):
        mat = np.zeros((2, 2))
        mat[0, 1] = 0.95
        r = shortest_path(mat, 0, 1)
        assert r.found is True
        assert r.path == [0, 1]

    def test_five_node_chain(self):
        mat = self._score_mat(5)
        r = shortest_path(mat, 0, 4)
        assert r.found is True
        assert r.path[0] == 0
        assert r.path[-1] == 4

    def test_disconnected_not_found(self):
        mat = _disconnected(4)
        r = shortest_path(mat, 0, 3)
        assert r.found is False

    def test_start_equals_end(self):
        mat = self._score_mat(3)
        r = shortest_path(mat, 2, 2)
        assert r.found is True
        assert r.path == [2]


# ─── TestAllPairsShortestPathsExtra ──────────────────────────────────────────

class TestAllPairsShortestPathsExtra:
    def test_single_node(self):
        mat = np.zeros((1, 1), dtype=np.float64)
        result = all_pairs_shortest_paths(mat)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0)

    def test_symmetric_for_undirected(self):
        mat = _undirected_chain(4)
        result = all_pairs_shortest_paths(mat)
        np.testing.assert_allclose(result, result.T)

    def test_complete_graph_all_one(self):
        mat = _complete_graph(4, w=1.0)
        result = all_pairs_shortest_paths(mat)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert result[i, j] == pytest.approx(1.0)

    def test_five_node_chain_distances(self):
        mat = _undirected_chain(5, w=1.0)
        result = all_pairs_shortest_paths(mat)
        assert result[0, 4] == pytest.approx(4.0)
        assert result[1, 3] == pytest.approx(2.0)

    def test_two_node_disconnected(self):
        mat = _disconnected(2)
        result = all_pairs_shortest_paths(mat)
        assert np.isinf(result[0, 1])
        assert np.isinf(result[1, 0])


# ─── TestTopologicalSortExtra ────────────────────────────────────────────────

class TestTopologicalSortExtra:
    def test_single_node(self):
        mat = np.zeros((1, 1), dtype=np.float64)
        order = topological_sort(mat)
        assert order == [0]

    def test_two_node_dag(self):
        mat = np.zeros((2, 2), dtype=np.float64)
        mat[0, 1] = 1.0
        order = topological_sort(mat)
        assert order.index(0) < order.index(1)

    def test_five_node_chain_order(self):
        mat = _chain_cost(5)
        order = topological_sort(mat)
        for i in range(4):
            assert order.index(i) < order.index(i + 1)

    def test_all_unique(self):
        mat = _chain_cost(6)
        order = topological_sort(mat)
        assert len(set(order)) == 6

    def test_disconnected_all_present(self):
        mat = _disconnected(4)
        order = topological_sort(mat)
        assert sorted(order) == [0, 1, 2, 3]


# ─── TestFindConnectedComponentsExtra ────────────────────────────────────────

class TestFindConnectedComponentsExtra:
    def test_single_node(self):
        mat = np.zeros((1, 1), dtype=np.float64)
        components = find_connected_components(mat)
        assert len(components) == 1
        assert components[0] == [0]

    def test_two_nodes_connected(self):
        mat = np.array([[0, 1], [1, 0]], dtype=np.float64)
        components = find_connected_components(mat)
        assert len(components) == 1

    def test_five_isolated_nodes(self):
        mat = _disconnected(5)
        components = find_connected_components(mat)
        assert len(components) == 5

    def test_three_cluster_graph(self):
        mat = np.zeros((6, 6), dtype=np.float64)
        mat[0, 1] = mat[1, 0] = 1.0
        mat[2, 3] = mat[3, 2] = 1.0
        mat[4, 5] = mat[5, 4] = 1.0
        components = find_connected_components(mat)
        assert len(components) == 3

    def test_no_duplicate_nodes(self):
        mat = _undirected_chain(6)
        components = find_connected_components(mat)
        all_nodes = [n for comp in components for n in comp]
        assert len(all_nodes) == len(set(all_nodes))


# ─── TestMinimumSpanningTreeExtra ────────────────────────────────────────────

class TestMinimumSpanningTreeExtra:
    def test_single_node(self):
        mat = np.zeros((1, 1), dtype=np.float64)
        mst = minimum_spanning_tree(mat)
        assert mst.shape == (1, 1)

    def test_complete_graph_n_minus_one_edges(self):
        mat = _complete_graph(5, w=1.0)
        mst = minimum_spanning_tree(mat)
        n_edges = int(np.sum(mst > 0)) // 2
        assert n_edges == 4

    def test_chain_total_weight(self):
        mat = _undirected_chain(4, w=2.0)
        mst = minimum_spanning_tree(mat)
        total_weight = np.sum(mst) / 2  # symmetric
        assert total_weight == pytest.approx(6.0)

    def test_mst_is_subgraph(self):
        mat = _undirected_chain(4)
        mst = minimum_spanning_tree(mat)
        # Every edge in MST must exist in original graph
        for i in range(4):
            for j in range(4):
                if mst[i, j] > 0:
                    assert mat[i, j] > 0 or mat[j, i] > 0

    def test_two_nodes(self):
        mat = np.array([[0, 3.0], [3.0, 0]], dtype=np.float64)
        mst = minimum_spanning_tree(mat)
        assert mst.shape == (2, 2)
        n_edges = int(np.sum(mst > 0)) // 2
        assert n_edges == 1


# ─── TestBatchDijkstraExtra ───────────────────────────────────────────────────

class TestBatchDijkstraExtra:
    def test_single_start(self):
        mat = _chain_cost(4)
        result = batch_dijkstra(mat, starts=[0], end=3)
        assert len(result) == 1
        assert result[0].found is True

    def test_all_starts_reachable(self):
        mat = _chain_cost(5)
        result = batch_dijkstra(mat, starts=[0, 1, 2], end=4)
        assert all(r.found for r in result)

    def test_no_path_from_later_start(self):
        mat = _chain_cost(4)
        # Nodes 5 wouldn't exist so use a case where start > end in directed graph
        result = batch_dijkstra(mat, starts=[3], end=0)
        # In directed chain, 3→0 has no path
        assert result[0].found is False

    def test_costs_decreasing_with_closer_start(self):
        mat = _chain_cost(5, w=1.0)
        result = batch_dijkstra(mat, starts=[0, 1, 2], end=4)
        costs = [r.cost for r in result if r.found]
        assert costs == sorted(costs, reverse=True)

    def test_result_indices_correct(self):
        mat = _chain_cost(4)
        result = batch_dijkstra(mat, starts=[0, 1], end=3)
        for r in result:
            if r.found:
                assert r.path[-1] == 3
