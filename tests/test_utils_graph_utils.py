"""Tests for puzzle_reconstruction.utils.graph_utils."""
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

np.random.seed(42)


# ── GraphEdge ─────────────────────────────────────────────────────────────────

def test_graph_edge_valid():
    e = GraphEdge(src=0, dst=1, weight=0.5)
    assert e.src == 0
    assert e.dst == 1
    assert e.weight == 0.5


def test_graph_edge_invalid_src():
    with pytest.raises(ValueError):
        GraphEdge(src=-1, dst=0, weight=1.0)


def test_graph_edge_invalid_dst():
    with pytest.raises(ValueError):
        GraphEdge(src=0, dst=-1, weight=1.0)


def test_graph_edge_invalid_weight():
    with pytest.raises(ValueError):
        GraphEdge(src=0, dst=1, weight=-0.1)


# ── FragmentGraph ─────────────────────────────────────────────────────────────

def test_fragment_graph_len():
    g = build_graph(np.ones((5, 5)))
    assert len(g) == 5


def test_fragment_graph_invalid_n_nodes():
    with pytest.raises(ValueError):
        FragmentGraph(n_nodes=0, edges=[], adj={})


# ── build_graph ───────────────────────────────────────────────────────────────

def _sym_matrix(n, fill=1.0):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            M[i, j] = M[j, i] = fill
    return M


def test_build_graph_basic():
    M = _sym_matrix(4, 1.0)
    g = build_graph(M)
    assert g.n_nodes == 4
    assert len(g.edges) == 6  # C(4,2)


def test_build_graph_with_threshold():
    M = np.array([[0, 0.5, 0.1],
                  [0.5, 0, 0.9],
                  [0.1, 0.9, 0]], dtype=float)
    g = build_graph(M, threshold=0.4)
    # Only edges with weight > 0.4: (0,1)=0.5 and (1,2)=0.9
    assert len(g.edges) == 2


def test_build_graph_non_square_raises():
    with pytest.raises(ValueError):
        build_graph(np.ones((3, 4)))


def test_build_graph_negative_threshold():
    with pytest.raises(ValueError):
        build_graph(np.ones((3, 3)), threshold=-0.1)


def test_build_graph_adj_symmetry():
    M = _sym_matrix(3, 1.0)
    g = build_graph(M)
    for i in range(3):
        neighbors = [v for v, _ in g.adj[i]]
        assert i not in neighbors  # no self-loops


# ── dijkstra ──────────────────────────────────────────────────────────────────

def _triangle_graph():
    M = np.array([[0, 1, 2],
                  [1, 0, 1],
                  [2, 1, 0]], dtype=float)
    return build_graph(M)


def test_dijkstra_source_zero():
    g = _triangle_graph()
    dist, prev = dijkstra(g, source=0)
    assert dist[0] == 0.0


def test_dijkstra_distances():
    g = _triangle_graph()
    dist, prev = dijkstra(g, source=0)
    assert dist[1] == pytest.approx(1.0)
    assert dist[2] == pytest.approx(2.0)


def test_dijkstra_prev_array_shape():
    g = _triangle_graph()
    dist, prev = dijkstra(g, source=0)
    assert prev.shape == (3,)


def test_dijkstra_invalid_source():
    g = _triangle_graph()
    with pytest.raises(ValueError):
        dijkstra(g, source=10)


# ── shortest_path ─────────────────────────────────────────────────────────────

def test_shortest_path_trivial():
    g = _triangle_graph()
    path = shortest_path(g, source=0, target=0)
    assert path == [0]


def test_shortest_path_basic():
    g = _triangle_graph()
    path = shortest_path(g, source=0, target=2)
    assert path[0] == 0
    assert path[-1] == 2


def test_shortest_path_unreachable():
    # Disconnected graph: two components
    M = np.zeros((4, 4))
    M[0, 1] = M[1, 0] = 1.0
    M[2, 3] = M[3, 2] = 1.0
    g = build_graph(M)
    path = shortest_path(g, source=0, target=2)
    assert path == []


def test_shortest_path_invalid_target():
    g = _triangle_graph()
    with pytest.raises(ValueError):
        shortest_path(g, source=0, target=99)


# ── minimum_spanning_tree ─────────────────────────────────────────────────────

def test_mst_edge_count():
    M = _sym_matrix(5, 1.0)
    g = build_graph(M)
    mst = minimum_spanning_tree(g)
    assert len(mst) == 4  # n-1 for connected graph


def test_mst_edges_are_graph_edges():
    M = _sym_matrix(4, 1.0)
    g = build_graph(M)
    mst = minimum_spanning_tree(g)
    for e in mst:
        assert isinstance(e, GraphEdge)


def test_mst_disconnected():
    M = np.zeros((4, 4))
    M[0, 1] = M[1, 0] = 1.0
    g = build_graph(M)
    mst = minimum_spanning_tree(g)
    assert len(mst) < 3


# ── connected_components ──────────────────────────────────────────────────────

def test_connected_components_single():
    M = _sym_matrix(4, 1.0)
    g = build_graph(M)
    comps = connected_components(g)
    assert len(comps) == 1
    assert sorted(comps[0]) == [0, 1, 2, 3]


def test_connected_components_two():
    M = np.zeros((4, 4))
    M[0, 1] = M[1, 0] = 1.0
    M[2, 3] = M[3, 2] = 1.0
    g = build_graph(M)
    comps = connected_components(g)
    assert len(comps) == 2


def test_connected_components_isolated():
    M = np.zeros((3, 3))
    g = build_graph(M)
    comps = connected_components(g)
    assert len(comps) == 3


# ── node_degrees ──────────────────────────────────────────────────────────────

def test_node_degrees_complete():
    M = _sym_matrix(4, 1.0)
    g = build_graph(M)
    degrees = node_degrees(g)
    assert degrees.shape == (4,)
    assert np.all(degrees == 3)  # complete graph K4


def test_node_degrees_empty_graph():
    M = np.zeros((3, 3))
    g = build_graph(M)
    degrees = node_degrees(g)
    assert np.all(degrees == 0)


# ── subgraph ──────────────────────────────────────────────────────────────────

def test_subgraph_basic():
    M = _sym_matrix(5, 1.0)
    g = build_graph(M)
    sg = subgraph(g, [0, 1, 2])
    assert sg.n_nodes == 3


def test_subgraph_edges():
    M = _sym_matrix(4, 1.0)
    g = build_graph(M)
    sg = subgraph(g, [0, 1])
    assert len(sg.edges) == 1


def test_subgraph_empty_raises():
    M = _sym_matrix(3, 1.0)
    g = build_graph(M)
    with pytest.raises(ValueError):
        subgraph(g, [])


def test_subgraph_invalid_node():
    M = _sym_matrix(3, 1.0)
    g = build_graph(M)
    with pytest.raises(ValueError):
        subgraph(g, [0, 99])


# ── batch_build_graphs ────────────────────────────────────────────────────────

def test_batch_build_graphs_count():
    matrices = [_sym_matrix(3, 1.0) for _ in range(4)]
    graphs = batch_build_graphs(matrices)
    assert len(graphs) == 4


def test_batch_build_graphs_types():
    matrices = [_sym_matrix(3, 1.0)]
    graphs = batch_build_graphs(matrices)
    assert isinstance(graphs[0], FragmentGraph)


def test_batch_build_graphs_empty():
    assert batch_build_graphs([]) == []
