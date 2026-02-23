"""Extra tests for puzzle_reconstruction.algorithms.path_planner."""
from __future__ import annotations

import math
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

def _chain(n=4):
    mat = np.zeros((n, n))
    for i in range(n - 1):
        mat[i, i + 1] = mat[i + 1, i] = 1.0
    return mat


def _complete(n=4, weight=1.0):
    mat = np.full((n, n), weight)
    np.fill_diagonal(mat, 0.0)
    return mat


def _disconnected(n=4):
    mat = np.zeros((n, n))
    mat[0, 1] = mat[1, 0] = 1.0
    return mat


def _dag_linear(n=4):
    """0→1→2→...→n-1"""
    mat = np.zeros((n, n))
    for i in range(n - 1):
        mat[i, i + 1] = 1.0
    return mat


# ─── TestPathResultExtra ─────────────────────────────────────────────────────

class TestPathResultExtra:
    def test_single_node_path(self):
        r = PathResult(path=[5], cost=0.0)
        assert r.found is True
        assert len(r) == 1

    def test_long_path(self):
        r = PathResult(path=list(range(10)), cost=9.0)
        assert len(r) == 10

    def test_cost_zero(self):
        r = PathResult(path=[0], cost=0.0)
        assert r.cost == pytest.approx(0.0)

    def test_params_default_empty(self):
        r = PathResult(path=[0, 1], cost=1.0)
        assert r.params == {}

    def test_params_stored(self):
        r = PathResult(path=[0], cost=0.0, params={"k": "v"})
        assert r.params["k"] == "v"

    def test_found_false(self):
        r = PathResult(path=[], cost=0.0, found=False)
        assert not r.found


# ─── TestDijkstraExtra ───────────────────────────────────────────────────────

class TestDijkstraExtra:
    def test_chain_5_cost_4(self):
        r = dijkstra(_chain(5), 0, 4)
        assert r.cost == pytest.approx(4.0)

    def test_path_length_chain(self):
        r = dijkstra(_chain(5), 0, 4)
        assert len(r.path) == 5

    def test_equal_start_end(self):
        r = dijkstra(_chain(4), 1, 1)
        assert r.cost == pytest.approx(0.0)
        assert r.path == [1]

    def test_two_nodes(self):
        mat = np.array([[0, 1.5], [1.5, 0]])
        r = dijkstra(mat, 0, 1)
        assert r.cost == pytest.approx(1.5)

    def test_path_list(self):
        r = dijkstra(_chain(4), 0, 3)
        assert r.path[0] == 0 and r.path[-1] == 3

    def test_weighted_shortcut(self):
        mat = np.zeros((3, 3))
        mat[0, 1] = mat[1, 0] = 10.0
        mat[0, 2] = mat[2, 0] = 1.0
        mat[1, 2] = mat[2, 1] = 1.0
        r = dijkstra(mat, 0, 1)
        assert r.cost == pytest.approx(2.0)  # 0→2→1

    def test_isolated_node_not_found(self):
        mat = _disconnected(4)
        r = dijkstra(mat, 0, 2)
        assert not r.found


# ─── TestShortestPathExtra ───────────────────────────────────────────────────

def _chain_half(n=4):
    """Chain with 0.5 scores; cost=0.5 each edge (not 0, so edges are present)."""
    mat = np.zeros((n, n))
    for i in range(n - 1):
        mat[i, i + 1] = mat[i + 1, i] = 0.5
    return mat


class TestShortestPathExtra:
    def test_chain_4(self):
        r = shortest_path(_chain_half(4), 0, 3)
        assert r.found is True
        assert r.cost == pytest.approx(1.5)  # 3 × (1-0.5)

    def test_disconnected(self):
        r = shortest_path(_disconnected(4), 0, 2)
        assert not r.found

    def test_self_path(self):
        r = shortest_path(_chain_half(4), 2, 2)
        assert r.cost == pytest.approx(0.0)

    def test_result_type(self):
        r = shortest_path(_chain_half(3), 0, 2)
        assert isinstance(r, PathResult)

    def test_cost_nonneg(self):
        r = shortest_path(_chain_half(5), 0, 4)
        assert r.cost >= 0.0


# ─── TestAllPairsShortestPathsExtra ──────────────────────────────────────────

class TestAllPairsShortestPathsExtra:
    def test_chain_3_distances(self):
        d = all_pairs_shortest_paths(_chain(3))
        assert d[0, 1] == pytest.approx(1.0)
        assert d[0, 2] == pytest.approx(2.0)

    def test_complete_4_all_one(self):
        d = all_pairs_shortest_paths(_complete(4))
        offdiag = d[~np.eye(4, dtype=bool)]
        np.testing.assert_allclose(offdiag, 1.0, atol=1e-5)

    def test_symmetric(self):
        d = all_pairs_shortest_paths(_chain(5))
        np.testing.assert_allclose(d, d.T, atol=1e-5)

    def test_chain_5_corner(self):
        d = all_pairs_shortest_paths(_chain(5))
        assert d[0, 4] == pytest.approx(4.0)

    def test_nonneg(self):
        d = all_pairs_shortest_paths(_complete(4))
        assert d.min() >= 0.0

    def test_single_node(self):
        d = all_pairs_shortest_paths(np.zeros((1, 1)))
        assert d.shape == (1, 1)
        assert d[0, 0] == pytest.approx(0.0)


# ─── TestTopologicalSortExtra ────────────────────────────────────────────────

class TestTopologicalSortExtra:
    def test_linear_dag(self):
        r = topological_sort(_dag_linear(4))
        assert r == [0, 1, 2, 3]

    def test_single_node(self):
        r = topological_sort(np.zeros((1, 1)))
        assert r == [0]

    def test_two_nodes_ordered(self):
        mat = np.array([[0, 1], [0, 0]], dtype=float)
        r = topological_sort(mat)
        assert r.index(0) < r.index(1)

    def test_all_present(self):
        r = topological_sort(_dag_linear(5))
        assert sorted(r) == [0, 1, 2, 3, 4]

    def test_no_edges_any_order(self):
        r = topological_sort(np.zeros((3, 3)))
        assert sorted(r) == [0, 1, 2]


# ─── TestFindConnectedComponentsExtra ────────────────────────────────────────

class TestFindConnectedComponentsExtra:
    def test_complete_one_component(self):
        comps = find_connected_components(_complete(5))
        assert len(comps) == 1

    def test_three_isolated_nodes(self):
        comps = find_connected_components(np.zeros((3, 3)))
        assert len(comps) == 3

    def test_two_pairs_two_components(self):
        mat = np.zeros((4, 4))
        mat[0, 1] = mat[1, 0] = 1.0
        mat[2, 3] = mat[3, 2] = 1.0
        comps = find_connected_components(mat)
        assert len(comps) == 2

    def test_all_nodes_accounted(self):
        comps = find_connected_components(_chain(6))
        all_nodes = sorted(n for c in comps for n in c)
        assert all_nodes == list(range(6))

    def test_chain_one_component(self):
        comps = find_connected_components(_chain(6))
        assert len(comps) == 1


# ─── TestMinimumSpanningTreeExtra ────────────────────────────────────────────

class TestMinimumSpanningTreeExtra:
    def test_5_node_complete_4_edges(self):
        mst = minimum_spanning_tree(_complete(5))
        n_edges = np.count_nonzero(mst) // 2
        assert n_edges == 4

    def test_chain_mst_equals_input(self):
        chain = _chain(4)
        mst = minimum_spanning_tree(chain)
        np.testing.assert_allclose(mst, chain)

    def test_symmetric(self):
        mst = minimum_spanning_tree(_complete(4))
        np.testing.assert_allclose(mst, mst.T)

    def test_3_node_complete_2_edges(self):
        mst = minimum_spanning_tree(_complete(3))
        n_edges = np.count_nonzero(mst) // 2
        assert n_edges == 2

    def test_weighted_selects_lighter(self):
        mat = np.zeros((3, 3))
        mat[0, 1] = mat[1, 0] = 1.0
        mat[0, 2] = mat[2, 0] = 5.0
        mat[1, 2] = mat[2, 1] = 2.0
        mst = minimum_spanning_tree(mat)
        # MST should include edge (0,1) and (1,2) with total cost 3
        total = np.sum(mst) / 2
        assert total == pytest.approx(3.0)


# ─── TestBatchDijkstraExtra ──────────────────────────────────────────────────

class TestBatchDijkstraExtra:
    def test_single_start(self):
        result = batch_dijkstra(_chain(4), [0], end=3)
        assert len(result) == 1

    def test_all_path_results(self):
        for r in batch_dijkstra(_chain(5), [0, 1, 2], end=4):
            assert isinstance(r, PathResult)

    def test_costs_monotone_closer_to_end(self):
        mat = _chain(6)
        results = batch_dijkstra(mat, [0, 3, 5], end=5)
        assert results[0].cost >= results[1].cost >= results[2].cost

    def test_empty_starts_empty_result(self):
        assert batch_dijkstra(_chain(4), [], end=3) == []

    def test_start_equals_end_cost_zero(self):
        results = batch_dijkstra(_chain(4), [2], end=2)
        assert results[0].cost == pytest.approx(0.0)
