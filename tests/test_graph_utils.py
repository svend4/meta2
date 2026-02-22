"""Тесты для puzzle_reconstruction.utils.graph_utils."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _chain_matrix(n: int) -> np.ndarray:
    """Матрица: граф-цепочка 0-1-2-...(n-1), вес = 1."""
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
        M[i + 1, i] = 1.0
    return M


def _complete_matrix(n: int, weight: float = 2.0) -> np.ndarray:
    M = np.full((n, n), weight, dtype=np.float64)
    np.fill_diagonal(M, 0.0)
    return M


def _disconnected_matrix(n: int) -> np.ndarray:
    """Матрица без рёбер (граф из изолированных вершин)."""
    return np.zeros((n, n), dtype=np.float64)


# ─── TestGraphEdge ────────────────────────────────────────────────────────────

class TestGraphEdge:
    def test_basic_creation(self):
        e = GraphEdge(src=0, dst=1, weight=1.5)
        assert e.weight == 1.5

    def test_negative_src_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=-1, dst=0, weight=1.0)

    def test_negative_dst_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=0, dst=-1, weight=1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            GraphEdge(src=0, dst=1, weight=-0.1)

    def test_zero_weight_valid(self):
        e = GraphEdge(src=0, dst=1, weight=0.0)
        assert e.weight == 0.0

    def test_self_loop_valid(self):
        # src == dst допустимо (граф может иметь петли)
        e = GraphEdge(src=3, dst=3, weight=0.5)
        assert e.src == e.dst


# ─── TestFragmentGraph ────────────────────────────────────────────────────────

class TestFragmentGraph:
    def test_basic_creation(self):
        g = FragmentGraph(n_nodes=3, edges=[], adj={0: [], 1: [], 2: []})
        assert len(g) == 3

    def test_n_nodes_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentGraph(n_nodes=0, edges=[], adj={})

    def test_len(self):
        g = FragmentGraph(n_nodes=5, edges=[], adj={i: [] for i in range(5)})
        assert len(g) == 5


# ─── TestBuildGraph ───────────────────────────────────────────────────────────

class TestBuildGraph:
    def test_returns_fragment_graph(self):
        g = build_graph(_chain_matrix(4))
        assert isinstance(g, FragmentGraph)

    def test_n_nodes_correct(self):
        g = build_graph(_chain_matrix(5))
        assert g.n_nodes == 5

    def test_edges_count_chain(self):
        g = build_graph(_chain_matrix(4), threshold=0.5)
        assert len(g.edges) == 3  # 0-1, 1-2, 2-3

    def test_adj_symmetric(self):
        g = build_graph(_chain_matrix(3), threshold=0.0)
        assert any(n == 1 for n, _ in g.adj[0])
        assert any(n == 0 for n, _ in g.adj[1])

    def test_threshold_filters_edges(self):
        M = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
        g = build_graph(M, threshold=1.0)  # 0.5 < 1.0 → удалено
        assert len(g.edges) == 0

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_graph(np.ones((3, 4)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            build_graph(np.ones(5))

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            build_graph(_chain_matrix(3), threshold=-0.1)

    def test_no_edges_disconnected(self):
        g = build_graph(_disconnected_matrix(4), threshold=0.5)
        assert len(g.edges) == 0


# ─── TestDijkstra ─────────────────────────────────────────────────────────────

class TestDijkstra:
    def test_source_distance_zero(self):
        g = build_graph(_chain_matrix(4))
        dist, _ = dijkstra(g, source=0)
        assert dist[0] == pytest.approx(0.0)

    def test_distances_correct_chain(self):
        g = build_graph(_chain_matrix(4))
        dist, _ = dijkstra(g, source=0)
        # 0→1 = 1, 0→2 = 2, 0→3 = 3
        assert dist[1] == pytest.approx(1.0)
        assert dist[2] == pytest.approx(2.0)
        assert dist[3] == pytest.approx(3.0)

    def test_unreachable_inf(self):
        g = build_graph(_disconnected_matrix(3), threshold=0.5)
        dist, _ = dijkstra(g, source=0)
        assert dist[1] == float("inf")
        assert dist[2] == float("inf")

    def test_source_out_of_range_raises(self):
        g = build_graph(_chain_matrix(3))
        with pytest.raises(ValueError):
            dijkstra(g, source=5)

    def test_prev_none_for_source(self):
        g = build_graph(_chain_matrix(3))
        _, prev = dijkstra(g, source=0)
        assert prev[0] == -1

    def test_dist_nonnegative(self):
        g = build_graph(_complete_matrix(4))
        dist, _ = dijkstra(g, source=0)
        assert (dist >= 0).all() or any(d == float("inf") for d in dist)


# ─── TestShortestPath ─────────────────────────────────────────────────────────

class TestShortestPath:
    def test_same_source_target(self):
        g = build_graph(_chain_matrix(3))
        path = shortest_path(g, source=0, target=0)
        assert path == [0]

    def test_direct_neighbor(self):
        g = build_graph(_chain_matrix(3))
        path = shortest_path(g, source=0, target=1)
        assert path == [0, 1]

    def test_multi_hop(self):
        g = build_graph(_chain_matrix(4))
        path = shortest_path(g, source=0, target=3)
        assert path == [0, 1, 2, 3]

    def test_unreachable_empty(self):
        g = build_graph(_disconnected_matrix(3), threshold=0.5)
        path = shortest_path(g, source=0, target=2)
        assert path == []

    def test_target_out_of_range_raises(self):
        g = build_graph(_chain_matrix(3))
        with pytest.raises(ValueError):
            shortest_path(g, source=0, target=10)


# ─── TestMinimumSpanningTree ──────────────────────────────────────────────────

class TestMinimumSpanningTree:
    def test_n_minus_one_edges(self):
        g = build_graph(_complete_matrix(5))
        mst = minimum_spanning_tree(g)
        assert len(mst) == 4

    def test_chain_mst_is_chain(self):
        g = build_graph(_chain_matrix(4))
        mst = minimum_spanning_tree(g)
        assert len(mst) == 3

    def test_disconnected_fewer_edges(self):
        # Граф из изолированных вершин → MST пустой (или меньше n-1)
        g = build_graph(_disconnected_matrix(4), threshold=0.5)
        mst = minimum_spanning_tree(g)
        assert len(mst) < 3

    def test_returns_list_of_graph_edges(self):
        g = build_graph(_complete_matrix(3))
        mst = minimum_spanning_tree(g)
        assert all(isinstance(e, GraphEdge) for e in mst)

    def test_weights_nonnegative(self):
        g = build_graph(_complete_matrix(4))
        mst = minimum_spanning_tree(g)
        assert all(e.weight >= 0 for e in mst)


# ─── TestConnectedComponents ──────────────────────────────────────────────────

class TestConnectedComponents:
    def test_single_component(self):
        g = build_graph(_complete_matrix(4))
        comps = connected_components(g)
        assert len(comps) == 1
        assert sorted(comps[0]) == [0, 1, 2, 3]

    def test_chain_one_component(self):
        g = build_graph(_chain_matrix(5))
        comps = connected_components(g)
        assert len(comps) == 1

    def test_isolated_nodes(self):
        g = build_graph(_disconnected_matrix(3), threshold=0.5)
        comps = connected_components(g)
        assert len(comps) == 3

    def test_two_components(self):
        M = np.zeros((4, 4), dtype=np.float64)
        M[0, 1] = M[1, 0] = 1.0
        M[2, 3] = M[3, 2] = 1.0
        g = build_graph(M, threshold=0.5)
        comps = connected_components(g)
        assert len(comps) == 2

    def test_components_cover_all_nodes(self):
        g = build_graph(_chain_matrix(5))
        comps = connected_components(g)
        all_nodes = sorted(n for comp in comps for n in comp)
        assert all_nodes == [0, 1, 2, 3, 4]


# ─── TestNodeDegrees ──────────────────────────────────────────────────────────

class TestNodeDegrees:
    def test_chain_degrees(self):
        g = build_graph(_chain_matrix(3))
        degs = node_degrees(g)
        # 0→1 рёбра: вершины 0 и 2 имеют степень 1, вершина 1 — степень 2
        assert degs[0] == 1
        assert degs[1] == 2
        assert degs[2] == 1

    def test_isolated_degrees_zero(self):
        g = build_graph(_disconnected_matrix(4), threshold=0.5)
        degs = node_degrees(g)
        assert (degs == 0).all()

    def test_dtype_int64(self):
        g = build_graph(_chain_matrix(3))
        degs = node_degrees(g)
        assert degs.dtype == np.int64

    def test_shape(self):
        g = build_graph(_chain_matrix(5))
        degs = node_degrees(g)
        assert degs.shape == (5,)


# ─── TestSubgraph ─────────────────────────────────────────────────────────────

class TestSubgraph:
    def test_basic_subgraph(self):
        g = build_graph(_complete_matrix(4))
        sg = subgraph(g, nodes=[0, 1, 2])
        assert sg.n_nodes == 3

    def test_edges_preserved(self):
        g = build_graph(_chain_matrix(4))
        sg = subgraph(g, nodes=[0, 1, 2])
        # В цепочке 0-1-2 должно быть 2 ребра
        assert len(sg.edges) == 2

    def test_indices_remapped(self):
        g = build_graph(_complete_matrix(3))
        sg = subgraph(g, nodes=[1, 2])
        # Вершины пронумерованы заново: 0 (было 1), 1 (было 2)
        assert sg.n_nodes == 2

    def test_empty_nodes_raises(self):
        g = build_graph(_chain_matrix(3))
        with pytest.raises(ValueError):
            subgraph(g, nodes=[])

    def test_out_of_range_node_raises(self):
        g = build_graph(_chain_matrix(3))
        with pytest.raises(ValueError):
            subgraph(g, nodes=[0, 99])


# ─── TestBatchBuildGraphs ─────────────────────────────────────────────────────

class TestBatchBuildGraphs:
    def test_returns_list(self):
        matrices = [_chain_matrix(3), _complete_matrix(4)]
        result = batch_build_graphs(matrices)
        assert isinstance(result, list)

    def test_correct_length(self):
        matrices = [_chain_matrix(3), _chain_matrix(4), _chain_matrix(5)]
        result = batch_build_graphs(matrices)
        assert len(result) == 3

    def test_empty_list(self):
        assert batch_build_graphs([]) == []

    def test_each_fragment_graph(self):
        matrices = [_chain_matrix(3), _complete_matrix(2)]
        result = batch_build_graphs(matrices)
        assert all(isinstance(g, FragmentGraph) for g in result)

    def test_n_nodes_per_graph(self):
        matrices = [_chain_matrix(3), _chain_matrix(5)]
        result = batch_build_graphs(matrices)
        assert result[0].n_nodes == 3
        assert result[1].n_nodes == 5
