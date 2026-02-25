"""Тесты для puzzle_reconstruction.matching.graph_match."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.graph_match import (
    FragmentGraph,
    GraphMatchResult,
    analyze_graph,
    build_fragment_graph,
    degree_centrality,
    mst_ordering,
    random_walk_similarity,
    spectral_ordering,
)
from puzzle_reconstruction.models import CompatEntry, Fragment


def _fragment(fid: int) -> Fragment:
    """Минимальный Fragment для тестов."""
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    return Fragment(fragment_id=fid, image=img)


def _compat(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    """Compat entry: fid = edge_id // 10, поэтому edge_id = fid * 10."""
    from puzzle_reconstruction.models import EdgeSignature, EdgeSide
    e_i = EdgeSignature(edge_id=fid_i * 10, side=EdgeSide.RIGHT,
                        virtual_curve=np.zeros((4, 2)),
                        fd=1.0, css_vec=np.zeros(8),
                        ifs_coeffs=np.zeros(4), length=10.0)
    e_j = EdgeSignature(edge_id=fid_j * 10, side=EdgeSide.LEFT,
                        virtual_curve=np.zeros((4, 2)),
                        fd=1.0, css_vec=np.zeros(8),
                        ifs_coeffs=np.zeros(4), length=10.0)
    return CompatEntry(edge_i=e_i, edge_j=e_j, score=score)


def _build_graph(n=4) -> FragmentGraph:
    """Простой граф с n фрагментами по цепочке."""
    graph = FragmentGraph()
    for i in range(n):
        graph.add_node(_fragment(i))
    for i in range(n - 1):
        graph.add_edge(i, i + 1, 0.8)
    return graph


# ─── TestFragmentGraph ────────────────────────────────────────────────────────

class TestFragmentGraph:
    def test_empty_graph(self):
        g = FragmentGraph()
        assert g.n_nodes == 0
        assert g.n_edges == 0

    def test_add_node(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        assert g.n_nodes == 1
        assert 0 in g.nodes

    def test_add_edge(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        g.add_node(_fragment(1))
        g.add_edge(0, 1, 0.7)
        assert g.n_edges == 1

    def test_weight_returns_correct(self):
        g = _build_graph(2)
        assert g.weight(0, 1) == pytest.approx(0.8)

    def test_weight_missing_returns_zero(self):
        g = _build_graph(2)
        assert g.weight(0, 99) == pytest.approx(0.0)

    def test_add_edge_keeps_max(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        g.add_node(_fragment(1))
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 1, 0.9)
        assert g.weight(0, 1) == pytest.approx(0.9)

    def test_neighbors(self):
        g = _build_graph(3)
        neighbors = g.neighbors(1)
        cand_ids = [n for n, _ in neighbors]
        assert 0 in cand_ids
        assert 2 in cand_ids

    def test_sorted_nodes(self):
        g = _build_graph(4)
        nodes = g.sorted_nodes()
        assert nodes == sorted(nodes)

    def test_adjacency_matrix_shape(self):
        g = _build_graph(3)
        A, fids = g.adjacency_matrix()
        assert A.shape == (3, 3)
        assert len(fids) == 3

    def test_adjacency_matrix_symmetric(self):
        g = _build_graph(3)
        A, _ = g.adjacency_matrix()
        np.testing.assert_array_almost_equal(A, A.T)

    def test_laplacian_shape(self):
        g = _build_graph(3)
        L, fids = g.laplacian()
        assert L.shape == (3, 3)

    def test_repr(self):
        g = _build_graph(2)
        assert "FragmentGraph" in repr(g)


# ─── TestBuildFragmentGraph ───────────────────────────────────────────────────

class TestBuildFragmentGraph:
    def test_returns_fragment_graph(self):
        frags = [_fragment(i) for i in range(3)]
        entries = [_compat(0, 1, 0.8), _compat(1, 2, 0.7)]
        g = build_fragment_graph(frags, entries)
        assert isinstance(g, FragmentGraph)

    def test_all_fragments_added(self):
        frags = [_fragment(i) for i in range(4)]
        g = build_fragment_graph(frags, [])
        assert g.n_nodes == 4

    def test_threshold_filters_edges(self):
        frags = [_fragment(i) for i in range(2)]
        entries = [_compat(0, 1, 0.3)]
        g = build_fragment_graph(frags, entries, threshold=0.5)
        assert g.n_edges == 0

    def test_edge_added_above_threshold(self):
        frags = [_fragment(i) for i in range(2)]
        entries = [_compat(0, 1, 0.8)]
        g = build_fragment_graph(frags, entries, threshold=0.5)
        assert g.n_edges >= 1

    def test_empty_entries(self):
        frags = [_fragment(i) for i in range(3)]
        g = build_fragment_graph(frags, [])
        assert g.n_edges == 0


# ─── TestMstOrdering ──────────────────────────────────────────────────────────

class TestMstOrdering:
    def test_returns_list(self):
        g = _build_graph(4)
        order = mst_ordering(g)
        assert isinstance(order, list)

    def test_all_nodes_included(self):
        g = _build_graph(5)
        order = mst_ordering(g)
        assert set(order) == g.nodes

    def test_empty_graph(self):
        g = FragmentGraph()
        assert mst_ordering(g) == []

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        assert mst_ordering(g) == [0]

    def test_no_duplicates(self):
        g = _build_graph(4)
        order = mst_ordering(g)
        assert len(order) == len(set(order))


# ─── TestSpectralOrdering ─────────────────────────────────────────────────────

class TestSpectralOrdering:
    def test_returns_list(self):
        g = _build_graph(4)
        order = spectral_ordering(g)
        assert isinstance(order, list)

    def test_all_nodes_included(self):
        g = _build_graph(4)
        order = spectral_ordering(g)
        assert set(order) == g.nodes

    def test_empty_graph(self):
        g = FragmentGraph()
        assert spectral_ordering(g) == []

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        assert spectral_ordering(g) == [0]


# ─── TestRandomWalkSimilarity ─────────────────────────────────────────────────

class TestRandomWalkSimilarity:
    def test_returns_matrix(self):
        g = _build_graph(3)
        M = random_walk_similarity(g)
        assert M.ndim == 2

    def test_shape_n_by_n(self):
        g = _build_graph(4)
        M = random_walk_similarity(g)
        assert M.shape == (4, 4)

    def test_empty_graph(self):
        g = FragmentGraph()
        M = random_walk_similarity(g)
        assert M.shape == (0, 0)

    def test_values_non_negative(self):
        g = _build_graph(3)
        M = random_walk_similarity(g)
        assert (M >= 0).all()


# ─── TestDegreeCentrality ─────────────────────────────────────────────────────

class TestDegreeCentrality:
    def test_returns_dict(self):
        g = _build_graph(4)
        c = degree_centrality(g)
        assert isinstance(c, dict)

    def test_all_fragments_present(self):
        g = _build_graph(4)
        c = degree_centrality(g)
        assert set(c.keys()) == g.nodes

    def test_values_in_range(self):
        g = _build_graph(4)
        c = degree_centrality(g)
        for v in c.values():
            assert 0.0 <= v <= 1.0

    def test_empty_graph(self):
        g = FragmentGraph()
        assert degree_centrality(g) == {}

    def test_most_connected_has_highest(self):
        # Star topology: node 0 connects to 1, 2, 3
        g = FragmentGraph()
        for i in range(4):
            g.add_node(_fragment(i))
        g.add_edge(0, 1, 0.9)
        g.add_edge(0, 2, 0.9)
        g.add_edge(0, 3, 0.9)
        c = degree_centrality(g)
        assert c[0] >= max(c[1], c[2], c[3])


# ─── TestAnalyzeGraph ─────────────────────────────────────────────────────────

class TestAnalyzeGraph:
    def test_returns_graph_match_result(self):
        g = _build_graph(4)
        r = analyze_graph(g)
        assert isinstance(r, GraphMatchResult)

    def test_mst_order_contains_all(self):
        g = _build_graph(4)
        r = analyze_graph(g)
        assert set(r.mst_order) == g.nodes

    def test_centrality_has_all_nodes(self):
        g = _build_graph(4)
        r = analyze_graph(g)
        assert set(r.centrality.keys()) == g.nodes

    def test_summary_returns_string(self):
        g = _build_graph(3)
        r = analyze_graph(g)
        assert isinstance(r.summary(), str)
