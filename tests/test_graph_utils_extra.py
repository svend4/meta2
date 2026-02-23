"""Extra tests for puzzle_reconstruction.utils.graph_utils."""
import numpy as np
import pytest

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _chain(n):
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
        M[i + 1, i] = 1.0
    return M


def _complete(n, w=2.0):
    M = np.full((n, n), w, dtype=np.float64)
    np.fill_diagonal(M, 0.0)
    return M


def _disconnected(n):
    return np.zeros((n, n), dtype=np.float64)


def _two_component(n=4):
    """Two 2-node components: {0,1} and {2,3}."""
    M = np.zeros((n, n), dtype=np.float64)
    M[0, 1] = M[1, 0] = 1.0
    M[2, 3] = M[3, 2] = 1.0
    return M


# ─── TestGraphEdgeExtra ──────────────────────────────────────────────────────

class TestGraphEdgeExtra:
    def test_src_stored(self):
        e = GraphEdge(src=3, dst=5, weight=0.7)
        assert e.src == 3

    def test_dst_stored(self):
        e = GraphEdge(src=3, dst=5, weight=0.7)
        assert e.dst == 5

    def test_weight_stored(self):
        e = GraphEdge(src=0, dst=1, weight=3.14)
        assert e.weight == pytest.approx(3.14)

    def test_zero_weight_ok(self):
        e = GraphEdge(src=0, dst=1, weight=0.0)
        assert e.weight == 0.0

    def test_large_weight_ok(self):
        e = GraphEdge(src=0, dst=1, weight=1e9)
        assert e.weight == pytest.approx(1e9)

    def test_negative_src_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=-1, dst=0, weight=1.0)

    def test_negative_dst_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=0, dst=-1, weight=1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=0, dst=1, weight=-0.01)

    def test_self_loop(self):
        e = GraphEdge(src=2, dst=2, weight=0.5)
        assert e.src == e.dst


# ─── TestFragmentGraphExtra ──────────────────────────────────────────────────

class TestFragmentGraphExtra:
    def test_len_equals_n_nodes(self):
        g = FragmentGraph(n_nodes=7, edges=[], adj={i: [] for i in range(7)})
        assert len(g) == 7

    def test_n_nodes_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentGraph(n_nodes=0, edges=[], adj={})

    def test_edges_stored(self):
        e = GraphEdge(src=0, dst=1, weight=1.0)
        g = FragmentGraph(n_nodes=2, edges=[e], adj={0: [(1, 1.0)], 1: [(0, 1.0)]})
        assert len(g.edges) == 1

    def test_adj_stored(self):
        g = FragmentGraph(n_nodes=2, edges=[],
                          adj={0: [(1, 0.5)], 1: [(0, 0.5)]})
        assert (1, 0.5) in g.adj[0]


# ─── TestBuildGraphExtra ─────────────────────────────────────────────────────

class TestBuildGraphExtra:
    def test_n_nodes_1(self):
        g = build_graph(np.array([[0.0]]))
        assert g.n_nodes == 1

    def test_no_edges_disconnected(self):
        g = build_graph(_disconnected(5), threshold=0.5)
        assert len(g.edges) == 0

    def test_threshold_0_all_edges(self):
        M = _complete(3, w=0.1)
        g = build_graph(M, threshold=0.0)
        assert len(g.edges) > 0

    def test_returns_fragment_graph(self):
        assert isinstance(build_graph(_chain(3)), FragmentGraph)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_graph(np.ones((3, 4)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            build_graph(np.ones(5))

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            build_graph(_chain(3), threshold=-0.1)

    def test_chain_6_edges(self):
        g = build_graph(_chain(4), threshold=0.5)
        assert len(g.edges) == 3


# ─── TestDijkstraExtra ───────────────────────────────────────────────────────

class TestDijkstraExtra:
    def test_source_0_dist_zero(self):
        g = build_graph(_chain(4))
        dist, _ = dijkstra(g, source=0)
        assert dist[0] == pytest.approx(0.0)

    def test_all_reachable_chain(self):
        g = build_graph(_chain(5))
        dist, _ = dijkstra(g, source=0)
        assert all(d < float("inf") for d in dist)

    def test_unreachable_is_inf(self):
        g = build_graph(_disconnected(3), threshold=0.5)
        dist, _ = dijkstra(g, source=0)
        assert dist[1] == float("inf")
        assert dist[2] == float("inf")

    def test_prev_of_source_is_neg1(self):
        g = build_graph(_chain(3))
        _, prev = dijkstra(g, source=0)
        assert prev[0] == -1

    def test_source_out_of_range_raises(self):
        g = build_graph(_chain(3))
        with pytest.raises(ValueError):
            dijkstra(g, source=10)

    def test_chain_distances_correct(self):
        g = build_graph(_chain(4), threshold=0.5)
        dist, _ = dijkstra(g, source=0)
        assert dist[1] == pytest.approx(1.0)
        assert dist[2] == pytest.approx(2.0)
        assert dist[3] == pytest.approx(3.0)

    def test_complete_graph_dist_2(self):
        g = build_graph(_complete(4, w=2.0))
        dist, _ = dijkstra(g, source=0)
        assert dist[1] == pytest.approx(2.0)


# ─── TestShortestPathExtra ───────────────────────────────────────────────────

class TestShortestPathExtra:
    def test_same_source_and_target(self):
        g = build_graph(_chain(3))
        assert shortest_path(g, source=0, target=0) == [0]

    def test_direct_edge(self):
        g = build_graph(_chain(3))
        assert shortest_path(g, source=0, target=1) == [0, 1]

    def test_multi_hop_4_nodes(self):
        g = build_graph(_chain(4), threshold=0.5)
        assert shortest_path(g, source=0, target=3) == [0, 1, 2, 3]

    def test_unreachable_empty_list(self):
        g = build_graph(_disconnected(3), threshold=0.5)
        assert shortest_path(g, source=0, target=2) == []

    def test_target_out_of_range_raises(self):
        g = build_graph(_chain(3))
        with pytest.raises(ValueError):
            shortest_path(g, source=0, target=99)

    def test_path_starts_at_source(self):
        g = build_graph(_chain(5))
        path = shortest_path(g, source=1, target=4)
        assert path[0] == 1

    def test_path_ends_at_target(self):
        g = build_graph(_chain(5))
        path = shortest_path(g, source=0, target=4)
        assert path[-1] == 4


# ─── TestMinimumSpanningTreeExtra ────────────────────────────────────────────

class TestMinimumSpanningTreeExtra:
    def test_complete_5_mst_4_edges(self):
        g = build_graph(_complete(5))
        mst = minimum_spanning_tree(g)
        assert len(mst) == 4

    def test_chain_mst_equals_chain(self):
        g = build_graph(_chain(4))
        mst = minimum_spanning_tree(g)
        assert len(mst) == 3

    def test_all_graph_edges_in_mst(self):
        g = build_graph(_complete(3))
        mst = minimum_spanning_tree(g)
        assert all(isinstance(e, GraphEdge) for e in mst)

    def test_disconnected_mst_shorter(self):
        g = build_graph(_disconnected(4), threshold=0.5)
        mst = minimum_spanning_tree(g)
        assert len(mst) < 3

    def test_weights_nonneg(self):
        g = build_graph(_complete(4))
        for e in minimum_spanning_tree(g):
            assert e.weight >= 0.0

    def test_two_component_mst_lt_3(self):
        g = build_graph(_two_component(4), threshold=0.5)
        mst = minimum_spanning_tree(g)
        assert len(mst) < 3


# ─── TestConnectedComponentsExtra ────────────────────────────────────────────

class TestConnectedComponentsExtra:
    def test_complete_one_component(self):
        g = build_graph(_complete(4))
        comps = connected_components(g)
        assert len(comps) == 1

    def test_chain_one_component(self):
        g = build_graph(_chain(5))
        comps = connected_components(g)
        assert len(comps) == 1

    def test_disconnected_each_isolated(self):
        g = build_graph(_disconnected(4), threshold=0.5)
        comps = connected_components(g)
        assert len(comps) == 4

    def test_two_components(self):
        g = build_graph(_two_component(4), threshold=0.5)
        comps = connected_components(g)
        assert len(comps) == 2

    def test_all_nodes_covered(self):
        g = build_graph(_chain(5))
        comps = connected_components(g)
        all_nodes = sorted(n for c in comps for n in c)
        assert all_nodes == [0, 1, 2, 3, 4]

    def test_returns_list(self):
        g = build_graph(_chain(3))
        assert isinstance(connected_components(g), list)


# ─── TestNodeDegreesExtra ────────────────────────────────────────────────────

class TestNodeDegreesExtra:
    def test_shape_equals_n_nodes(self):
        g = build_graph(_chain(5))
        assert node_degrees(g).shape == (5,)

    def test_isolated_all_zero(self):
        g = build_graph(_disconnected(3), threshold=0.5)
        assert (node_degrees(g) == 0).all()

    def test_chain_3_degrees(self):
        g = build_graph(_chain(3), threshold=0.5)
        degs = node_degrees(g)
        assert degs[0] == 1
        assert degs[1] == 2
        assert degs[2] == 1

    def test_complete_3_degrees_all_2(self):
        g = build_graph(_complete(3))
        degs = node_degrees(g)
        assert (degs == 2).all()

    def test_dtype_int64(self):
        g = build_graph(_chain(3))
        assert node_degrees(g).dtype == np.int64

    def test_sum_equals_twice_edges(self):
        g = build_graph(_chain(4))
        degs = node_degrees(g)
        assert degs.sum() == 2 * len(g.edges)


# ─── TestSubgraphExtra ───────────────────────────────────────────────────────

class TestSubgraphExtra:
    def test_n_nodes_correct(self):
        g = build_graph(_complete(5))
        sg = subgraph(g, nodes=[0, 1, 2])
        assert sg.n_nodes == 3

    def test_single_node_subgraph(self):
        g = build_graph(_chain(3))
        sg = subgraph(g, nodes=[1])
        assert sg.n_nodes == 1
        assert len(sg.edges) == 0

    def test_chain_subgraph_edges(self):
        g = build_graph(_chain(4), threshold=0.5)
        sg = subgraph(g, nodes=[0, 1, 2])
        assert len(sg.edges) == 2

    def test_empty_nodes_raises(self):
        g = build_graph(_chain(3))
        with pytest.raises(ValueError):
            subgraph(g, nodes=[])

    def test_out_of_range_raises(self):
        g = build_graph(_chain(3))
        with pytest.raises(ValueError):
            subgraph(g, nodes=[0, 99])

    def test_full_graph_same_edges(self):
        g = build_graph(_chain(4))
        sg = subgraph(g, nodes=[0, 1, 2, 3])
        assert len(sg.edges) == len(g.edges)


# ─── TestBatchBuildGraphsExtra ───────────────────────────────────────────────

class TestBatchBuildGraphsExtra:
    def test_empty_input(self):
        assert batch_build_graphs([]) == []

    def test_single_matrix(self):
        result = batch_build_graphs([_chain(3)])
        assert len(result) == 1

    def test_returns_fragment_graphs(self):
        for g in batch_build_graphs([_chain(3), _complete(4)]):
            assert isinstance(g, FragmentGraph)

    def test_n_nodes_per_graph(self):
        result = batch_build_graphs([_chain(3), _chain(5), _chain(2)])
        assert result[0].n_nodes == 3
        assert result[1].n_nodes == 5
        assert result[2].n_nodes == 2

    def test_five_matrices(self):
        matrices = [_chain(n + 2) for n in range(5)]
        result = batch_build_graphs(matrices)
        assert len(result) == 5

    def test_threshold_forwarded(self):
        M = np.array([[0.0, 0.5], [0.5, 0.0]])
        result = batch_build_graphs([M], threshold=1.0)
        assert len(result[0].edges) == 0
