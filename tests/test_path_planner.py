"""Tests for puzzle_reconstruction.algorithms.path_planner."""
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
    """Chain graph 0→1→2→...→n-1 with weight w."""
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        mat[i, i + 1] = w
    return mat


def _undirected_chain(n: int, w: float = 1.0) -> np.ndarray:
    """Undirected chain graph."""
    mat = _chain_cost(n, w)
    return mat + mat.T


def _disconnected(n: int) -> np.ndarray:
    """Completely disconnected graph."""
    return np.zeros((n, n), dtype=np.float64)


# ─── PathResult ──────────────────────────────────────────────────────────────

class TestPathResult:
    def test_fields_stored(self):
        r = PathResult(path=[0, 1, 2], cost=2.0)
        assert r.path == [0, 1, 2]
        assert r.cost == pytest.approx(2.0)
        assert r.found is True

    def test_default_found_true(self):
        r = PathResult(path=[], cost=0.0)
        assert r.found is True

    def test_default_params_empty(self):
        r = PathResult(path=[], cost=0.0)
        assert r.params == {}

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError):
            PathResult(path=[0], cost=-0.1)

    def test_zero_cost_allowed(self):
        r = PathResult(path=[0], cost=0.0)
        assert r.cost == pytest.approx(0.0)

    def test_len(self):
        r = PathResult(path=[0, 1, 2], cost=2.0)
        assert len(r) == 3

    def test_len_zero(self):
        r = PathResult(path=[], cost=0.0)
        assert len(r) == 0

    def test_found_false(self):
        r = PathResult(path=[], cost=0.0, found=False)
        assert r.found is False


# ─── dijkstra ────────────────────────────────────────────────────────────────

class TestDijkstra:
    def test_returns_path_result(self):
        mat = _chain_cost(4)
        r = dijkstra(mat, 0, 3)
        assert isinstance(r, PathResult)

    def test_direct_path_found(self):
        mat = _chain_cost(4)
        r = dijkstra(mat, 0, 1)
        assert r.found is True
        assert r.path == [0, 1]

    def test_chained_path(self):
        mat = _chain_cost(4)
        r = dijkstra(mat, 0, 3)
        assert r.found is True
        assert r.path == [0, 1, 2, 3]

    def test_cost_correct(self):
        mat = _chain_cost(4, w=2.0)
        r = dijkstra(mat, 0, 3)
        assert r.cost == pytest.approx(6.0)

    def test_same_start_end(self):
        mat = _chain_cost(3)
        r = dijkstra(mat, 1, 1)
        assert r.found is True
        assert r.path == [1]
        assert r.cost == pytest.approx(0.0)

    def test_no_path_returns_not_found(self):
        mat = _disconnected(3)
        r = dijkstra(mat, 0, 2)
        assert r.found is False
        assert r.path == []

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            dijkstra(np.zeros((3, 4)), 0, 1)

    def test_start_out_of_range_raises(self):
        mat = _chain_cost(3)
        with pytest.raises(ValueError):
            dijkstra(mat, 5, 1)

    def test_end_out_of_range_raises(self):
        mat = _chain_cost(3)
        with pytest.raises(ValueError):
            dijkstra(mat, 0, 5)

    def test_algorithm_param_stored(self):
        mat = _chain_cost(3)
        r = dijkstra(mat, 0, 2)
        assert r.params.get("algorithm") == "dijkstra"

    def test_path_starts_at_start(self):
        mat = _chain_cost(4)
        r = dijkstra(mat, 0, 3)
        assert r.path[0] == 0

    def test_path_ends_at_end(self):
        mat = _chain_cost(4)
        r = dijkstra(mat, 0, 3)
        assert r.path[-1] == 3


# ─── shortest_path ───────────────────────────────────────────────────────────

class TestShortestPath:
    def _score_chain(self, n: int, s: float = 0.9) -> np.ndarray:
        mat = np.zeros((n, n))
        for i in range(n - 1):
            mat[i, i + 1] = s
        return mat

    def test_returns_path_result(self):
        r = shortest_path(self._score_chain(4), 0, 3)
        assert isinstance(r, PathResult)

    def test_path_found(self):
        r = shortest_path(self._score_chain(4), 0, 3)
        assert r.found is True

    def test_zero_scores_no_path(self):
        mat = np.zeros((3, 3))
        r = shortest_path(mat, 0, 2)
        assert r.found is False

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            shortest_path(np.zeros((3, 4)), 0, 1)

    def test_direct_connection(self):
        mat = np.zeros((3, 3))
        mat[0, 1] = 0.9
        r = shortest_path(mat, 0, 1)
        assert r.found is True
        assert r.path == [0, 1]


# ─── all_pairs_shortest_paths ────────────────────────────────────────────────

class TestAllPairsShortestPaths:
    def test_returns_square_matrix(self):
        mat = _undirected_chain(4)
        result = all_pairs_shortest_paths(mat)
        assert result.shape == (4, 4)

    def test_dtype_float64(self):
        mat = _undirected_chain(3)
        result = all_pairs_shortest_paths(mat)
        assert result.dtype == np.float64

    def test_diagonal_zero(self):
        mat = _undirected_chain(4)
        result = all_pairs_shortest_paths(mat)
        np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_disconnected_is_inf(self):
        mat = _disconnected(3)
        result = all_pairs_shortest_paths(mat)
        # Off-diagonal should be inf
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isinf(result[i, j])

    def test_chain_distances_correct(self):
        mat = _undirected_chain(3, w=1.0)
        result = all_pairs_shortest_paths(mat)
        assert result[0, 2] == pytest.approx(2.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            all_pairs_shortest_paths(np.zeros((3, 4)))


# ─── topological_sort ────────────────────────────────────────────────────────

class TestTopologicalSort:
    def test_chain_dag(self):
        mat = _chain_cost(4)
        order = topological_sort(mat)
        assert order == [0, 1, 2, 3]

    def test_disconnected_all_present(self):
        mat = _disconnected(3)
        order = topological_sort(mat)
        assert sorted(order) == [0, 1, 2]

    def test_cycle_raises(self):
        mat = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float64)
        with pytest.raises(RuntimeError):
            topological_sort(mat)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            topological_sort(np.zeros((3, 4)))

    def test_length_equals_n(self):
        mat = _chain_cost(5)
        order = topological_sort(mat)
        assert len(order) == 5

    def test_no_duplicates(self):
        mat = _chain_cost(5)
        order = topological_sort(mat)
        assert len(set(order)) == 5


# ─── find_connected_components ───────────────────────────────────────────────

class TestFindConnectedComponents:
    def test_fully_connected_one_component(self):
        mat = _undirected_chain(4)
        components = find_connected_components(mat)
        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2, 3]

    def test_disconnected_n_components(self):
        mat = _disconnected(3)
        components = find_connected_components(mat)
        assert len(components) == 3

    def test_two_clusters(self):
        mat = np.zeros((4, 4), dtype=np.float64)
        mat[0, 1] = mat[1, 0] = 1.0
        mat[2, 3] = mat[3, 2] = 1.0
        components = find_connected_components(mat)
        assert len(components) == 2
        sizes = sorted(len(c) for c in components)
        assert sizes == [2, 2]

    def test_returns_sorted_components(self):
        mat = _undirected_chain(4)
        components = find_connected_components(mat)
        for comp in components:
            assert comp == sorted(comp)

    def test_all_nodes_covered(self):
        mat = _undirected_chain(5)
        components = find_connected_components(mat)
        all_nodes = sorted(n for comp in components for n in comp)
        assert all_nodes == list(range(5))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            find_connected_components(np.zeros((3, 4)))


# ─── minimum_spanning_tree ───────────────────────────────────────────────────

class TestMinimumSpanningTree:
    def test_returns_square_matrix(self):
        mat = _undirected_chain(4)
        mst = minimum_spanning_tree(mat)
        assert mst.shape == (4, 4)

    def test_dtype_float64(self):
        mat = _undirected_chain(3)
        mst = minimum_spanning_tree(mat)
        assert mst.dtype == np.float64

    def test_symmetric(self):
        mat = _undirected_chain(4)
        mst = minimum_spanning_tree(mat)
        np.testing.assert_array_equal(mst, mst.T)

    def test_diagonal_zero(self):
        mat = _undirected_chain(4)
        mst = minimum_spanning_tree(mat)
        np.testing.assert_array_equal(np.diag(mst), np.zeros(4))

    def test_n_minus_one_edges(self):
        mat = _undirected_chain(4)
        mst = minimum_spanning_tree(mat)
        # count edges (upper triangle only, symmetric)
        n_edges = int(np.sum(mst > 0)) // 2
        assert n_edges == 3

    def test_empty_matrix_returns_empty(self):
        mat = np.zeros((0, 0), dtype=np.float64)
        mst = minimum_spanning_tree(mat)
        assert mst.shape == (0, 0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            minimum_spanning_tree(np.zeros((3, 4)))


# ─── batch_dijkstra ──────────────────────────────────────────────────────────

class TestBatchDijkstra:
    def test_returns_list(self):
        mat = _chain_cost(4)
        result = batch_dijkstra(mat, starts=[0, 1], end=3)
        assert isinstance(result, list)

    def test_length_matches_starts(self):
        mat = _chain_cost(5)
        result = batch_dijkstra(mat, starts=[0, 1, 2], end=4)
        assert len(result) == 3

    def test_empty_starts_returns_empty(self):
        mat = _chain_cost(3)
        result = batch_dijkstra(mat, starts=[], end=2)
        assert result == []

    def test_all_path_results(self):
        mat = _chain_cost(4)
        result = batch_dijkstra(mat, starts=[0, 1], end=3)
        assert all(isinstance(r, PathResult) for r in result)

    def test_paths_end_at_correct_node(self):
        mat = _chain_cost(4)
        result = batch_dijkstra(mat, starts=[0, 1], end=3)
        for r in result:
            if r.found:
                assert r.path[-1] == 3

    def test_start_equals_end(self):
        mat = _chain_cost(3)
        result = batch_dijkstra(mat, starts=[2], end=2)
        assert result[0].found is True
        assert result[0].path == [2]
