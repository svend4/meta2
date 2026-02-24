"""Extra tests for puzzle_reconstruction/utils/graph_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.graph_utils import (
    GraphEdge,
    FragmentGraph,
    build_graph,
    dijkstra,
    shortest_path,
    minimum_spanning_tree,
    connected_components,
    node_degrees,
    subgraph,
    batch_build_graphs,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _complete_matrix(n=4, val=1.0) -> np.ndarray:
    M = np.full((n, n), val, dtype=np.float64)
    np.fill_diagonal(M, 0.0)
    return M


def _simple_graph() -> FragmentGraph:
    """3-node fully connected graph with weight 1.0."""
    return build_graph(_complete_matrix(3))


# ─── GraphEdge ────────────────────────────────────────────────────────────────

class TestGraphEdgeExtra:
    def test_stores_src_dst(self):
        e = GraphEdge(src=0, dst=2, weight=0.5)
        assert e.src == 0 and e.dst == 2

    def test_negative_src_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=-1, dst=0, weight=1.0)

    def test_negative_dst_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=0, dst=-1, weight=1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=0, dst=1, weight=-0.5)

    def test_zero_weight_ok(self):
        e = GraphEdge(src=0, dst=1, weight=0.0)
        assert e.weight == pytest.approx(0.0)


# ─── FragmentGraph ────────────────────────────────────────────────────────────

class TestFragmentGraphExtra:
    def test_stores_n_nodes(self):
        g = _simple_graph()
        assert g.n_nodes == 3

    def test_len_equals_n_nodes(self):
        g = _simple_graph()
        assert len(g) == 3

    def test_zero_nodes_raises(self):
        with pytest.raises(ValueError):
            FragmentGraph(n_nodes=0, edges=[], adj={})


# ─── build_graph ──────────────────────────────────────────────────────────────

class TestBuildGraphExtra:
    def test_returns_fragment_graph(self):
        g = build_graph(_complete_matrix(3))
        assert isinstance(g, FragmentGraph)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            build_graph(np.array([1.0, 2.0]))

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_graph(np.ones((3, 4)))

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            build_graph(_complete_matrix(3), threshold=-0.1)

    def test_threshold_filters_edges(self):
        M = _complete_matrix(3, val=0.5)
        g = build_graph(M, threshold=0.8)
        assert len(g.edges) == 0

    def test_correct_n_edges_complete(self):
        g = _simple_graph()
        # 3-node complete: 3 edges
        assert len(g.edges) == 3


# ─── dijkstra ─────────────────────────────────────────────────────────────────

class TestDijkstraExtra:
    def test_returns_dist_prev_tuple(self):
        g = _simple_graph()
        result = dijkstra(g, 0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_source_dist_zero(self):
        g = _simple_graph()
        dist, _ = dijkstra(g, 0)
        assert dist[0] == pytest.approx(0.0)

    def test_invalid_source_raises(self):
        g = _simple_graph()
        with pytest.raises(ValueError):
            dijkstra(g, 99)

    def test_dist_shape(self):
        g = _simple_graph()
        dist, _ = dijkstra(g, 0)
        assert dist.shape == (3,)


# ─── shortest_path ────────────────────────────────────────────────────────────

class TestShortestPathExtra:
    def test_source_to_self(self):
        g = _simple_graph()
        path = shortest_path(g, 0, 0)
        assert path == [0]

    def test_returns_list(self):
        g = _simple_graph()
        assert isinstance(shortest_path(g, 0, 2), list)

    def test_path_exists(self):
        g = _simple_graph()
        path = shortest_path(g, 0, 2)
        assert len(path) >= 2 and path[0] == 0 and path[-1] == 2

    def test_invalid_target_raises(self):
        g = _simple_graph()
        with pytest.raises(ValueError):
            shortest_path(g, 0, 99)


# ─── minimum_spanning_tree ────────────────────────────────────────────────────

class TestMSTExtra:
    def test_returns_list(self):
        assert isinstance(minimum_spanning_tree(_simple_graph()), list)

    def test_mst_n_edges(self):
        g = _simple_graph()
        mst = minimum_spanning_tree(g)
        assert len(mst) == g.n_nodes - 1

    def test_each_is_graph_edge(self):
        for e in minimum_spanning_tree(_simple_graph()):
            assert isinstance(e, GraphEdge)


# ─── connected_components ─────────────────────────────────────────────────────

class TestConnectedComponentsExtra:
    def test_single_component_complete(self):
        g = _simple_graph()
        cc = connected_components(g)
        assert len(cc) == 1

    def test_isolated_nodes_separate(self):
        # Two isolated nodes: no edges
        M = np.zeros((2, 2))
        g = build_graph(M, threshold=0.1)
        cc = connected_components(g)
        assert len(cc) == 2

    def test_each_component_sorted(self):
        for comp in connected_components(_simple_graph()):
            assert comp == sorted(comp)


# ─── node_degrees ─────────────────────────────────────────────────────────────

class TestNodeDegreesExtra:
    def test_returns_array(self):
        g = _simple_graph()
        d = node_degrees(g)
        assert isinstance(d, np.ndarray)

    def test_shape_equals_n_nodes(self):
        g = _simple_graph()
        assert node_degrees(g).shape == (3,)

    def test_complete_graph_degrees(self):
        g = _simple_graph()
        # 3-node complete: each node has degree 2
        assert (node_degrees(g) == 2).all()


# ─── subgraph ─────────────────────────────────────────────────────────────────

class TestSubgraphExtra:
    def test_returns_fragment_graph(self):
        g = _simple_graph()
        sg = subgraph(g, [0, 1])
        assert isinstance(sg, FragmentGraph)

    def test_n_nodes_correct(self):
        g = _simple_graph()
        sg = subgraph(g, [0, 2])
        assert sg.n_nodes == 2

    def test_empty_nodes_raises(self):
        g = _simple_graph()
        with pytest.raises(ValueError):
            subgraph(g, [])

    def test_invalid_node_raises(self):
        g = _simple_graph()
        with pytest.raises(ValueError):
            subgraph(g, [99])


# ─── batch_build_graphs ───────────────────────────────────────────────────────

class TestBatchBuildGraphsExtra:
    def test_returns_list(self):
        result = batch_build_graphs([_complete_matrix(3)])
        assert isinstance(result, list)

    def test_length_matches(self):
        mats = [_complete_matrix(3), _complete_matrix(4)]
        assert len(batch_build_graphs(mats)) == 2

    def test_empty_input(self):
        assert batch_build_graphs([]) == []
