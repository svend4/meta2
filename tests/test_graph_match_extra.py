"""Extra tests for puzzle_reconstruction/matching/graph_match.py."""
from __future__ import annotations

import numpy as np
import pytest

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
from puzzle_reconstruction.models import CompatEntry, Edge, Fragment


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_fragment(fid: int) -> Fragment:
    from puzzle_reconstruction.models import EdgeSide, EdgeSignature, FractalSignature, TangramSignature, ShapeClass
    import numpy as _np
    fs = FractalSignature(
        fd_box=1.0, fd_divider=1.0,
        ifs_coeffs=_np.zeros(4),
        css_image=[],
        chain_code="0",
        curve=_np.zeros((2, 2)),
    )
    ts = TangramSignature(
        polygon=_np.zeros((3, 2)),
        shape_class=ShapeClass.TRIANGLE,
        centroid=_np.zeros(2),
        angle=0.0,
        scale=1.0,
        area=0.5,
    )
    edge = Edge(
        edge_id=fid * 10,
        side=EdgeSide.TOP,
        fractal=fs,
        tangram=ts,
        signature=EdgeSignature(edge_id=fid * 10, side=EdgeSide.TOP, fractal=fs, tangram=ts),
    )
    return Fragment(
        fragment_id=fid,
        image=_np.zeros((16, 16, 3), dtype=_np.uint8),
        mask=_np.zeros((16, 16), dtype=_np.uint8),
        edges=[edge],
    )


def _make_compat(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    from puzzle_reconstruction.models import EdgeSide, EdgeSignature, FractalSignature, TangramSignature, ShapeClass
    import numpy as _np

    def _edge(fid):
        fs = FractalSignature(
            fd_box=1.0, fd_divider=1.0,
            ifs_coeffs=_np.zeros(4),
            css_image=[],
            chain_code="0",
            curve=_np.zeros((2, 2)),
        )
        ts = TangramSignature(
            polygon=_np.zeros((3, 2)),
            shape_class=ShapeClass.TRIANGLE,
            centroid=_np.zeros(2),
            angle=0.0,
            scale=1.0,
            area=0.5,
        )
        sig = EdgeSignature(edge_id=fid * 10, side=EdgeSide.TOP, fractal=fs, tangram=ts)
        return Edge(edge_id=fid * 10, side=EdgeSide.TOP, fractal=fs, tangram=ts, signature=sig)

    return CompatEntry(edge_i=_edge(fid_i), edge_j=_edge(fid_j), score=score)


def _triangle_graph() -> FragmentGraph:
    """Graph with 3 nodes and 3 edges forming a triangle."""
    g = FragmentGraph()
    for fid in (0, 1, 2):
        g.nodes.add(fid)
    g.add_edge(0, 1, 0.9)
    g.add_edge(1, 2, 0.7)
    g.add_edge(0, 2, 0.5)
    return g


def _line_graph(n: int = 4) -> FragmentGraph:
    """Linear chain graph: 0-1-2-..."""
    g = FragmentGraph()
    for fid in range(n):
        g.nodes.add(fid)
    for i in range(n - 1):
        g.add_edge(i, i + 1, 0.8)
    return g


# ─── FragmentGraph (extra) ────────────────────────────────────────────────────

class TestFragmentGraphExtra:
    def test_empty_graph_zero_nodes(self):
        g = FragmentGraph()
        assert g.n_nodes == 0

    def test_empty_graph_zero_edges(self):
        g = FragmentGraph()
        assert g.n_edges == 0

    def test_add_node_increases_count(self):
        g = FragmentGraph()
        frag = _make_fragment(0)
        g.add_node(frag)
        assert g.n_nodes == 1

    def test_add_node_twice_same_fid(self):
        g = FragmentGraph()
        frag = _make_fragment(0)
        g.add_node(frag)
        g.add_node(frag)
        assert g.n_nodes == 1

    def test_add_edge_increases_count(self):
        g = FragmentGraph()
        g.nodes.update({0, 1})
        g.add_edge(0, 1, 0.5)
        assert g.n_edges == 1

    def test_add_edge_max_weight(self):
        g = FragmentGraph()
        g.nodes.update({0, 1})
        g.add_edge(0, 1, 0.4)
        g.add_edge(0, 1, 0.9)
        assert g.weight(0, 1) == pytest.approx(0.9)

    def test_add_edge_max_weight_lower_second(self):
        g = FragmentGraph()
        g.nodes.update({0, 1})
        g.add_edge(0, 1, 0.9)
        g.add_edge(0, 1, 0.2)
        assert g.weight(0, 1) == pytest.approx(0.9)

    def test_weight_missing_edge_zero(self):
        g = FragmentGraph()
        assert g.weight(99, 100) == pytest.approx(0.0)

    def test_weight_symmetric(self):
        g = _triangle_graph()
        assert g.weight(0, 1) == g.weight(1, 0)

    def test_neighbors_empty_for_isolated_node(self):
        g = FragmentGraph()
        g.nodes.add(5)
        assert g.neighbors(5) == []

    def test_neighbors_returns_connected(self):
        g = _triangle_graph()
        nbrs = dict(g.neighbors(0))
        assert 1 in nbrs
        assert 2 in nbrs

    def test_sorted_nodes_order(self):
        g = FragmentGraph()
        for fid in (3, 1, 2, 0):
            g.nodes.add(fid)
        assert g.sorted_nodes() == [0, 1, 2, 3]

    def test_adjacency_matrix_shape(self):
        g = _triangle_graph()
        A, fids = g.adjacency_matrix()
        assert A.shape == (3, 3)
        assert len(fids) == 3

    def test_adjacency_matrix_symmetric(self):
        g = _triangle_graph()
        A, _ = g.adjacency_matrix()
        assert np.allclose(A, A.T)

    def test_adjacency_matrix_diagonal_zero(self):
        g = _triangle_graph()
        A, _ = g.adjacency_matrix()
        assert np.allclose(np.diag(A), 0.0)

    def test_laplacian_shape(self):
        g = _triangle_graph()
        L, fids = g.laplacian()
        assert L.shape == (3, 3)

    def test_laplacian_row_sum_zero(self):
        g = _triangle_graph()
        L, _ = g.laplacian()
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_repr_contains_counts(self):
        g = _triangle_graph()
        r = repr(g)
        assert "3" in r

    def test_adjacency_matrix_values(self):
        g = FragmentGraph()
        g.nodes.update({0, 1})
        g.add_edge(0, 1, 0.75)
        A, fids = g.adjacency_matrix()
        i = fids.index(0)
        j = fids.index(1)
        assert A[i, j] == pytest.approx(0.75)


# ─── GraphMatchResult (extra) ─────────────────────────────────────────────────

class TestGraphMatchResultExtra:
    def test_summary_returns_string(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        s = result.summary()
        assert isinstance(s, str)

    def test_summary_contains_node_count(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert "3" in result.summary()

    def test_graph_reference_preserved(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert result.graph is g

    def test_centrality_is_dict(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert isinstance(result.centrality, dict)

    def test_mst_order_list(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert isinstance(result.mst_order, list)

    def test_spectral_order_list(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert isinstance(result.spectral_order, list)


# ─── build_fragment_graph (extra) ─────────────────────────────────────────────

class TestBuildFragmentGraphExtra:
    def test_empty_fragments_empty_graph(self):
        g = build_fragment_graph([], [])
        assert g.n_nodes == 0
        assert g.n_edges == 0

    def test_nodes_added_for_all_fragments(self):
        frags = [_make_fragment(i) for i in range(4)]
        g = build_fragment_graph(frags, [])
        assert g.n_nodes == 4

    def test_edges_added_above_threshold(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entry = _make_compat(0, 1, 0.8)
        g = build_fragment_graph(frags, [entry], threshold=0.5)
        assert g.n_edges == 1

    def test_edges_below_threshold_excluded(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entry = _make_compat(0, 1, 0.3)
        g = build_fragment_graph(frags, [entry], threshold=0.5)
        assert g.n_edges == 0

    def test_self_edges_excluded(self):
        frags = [_make_fragment(0)]
        # edge_id // 10 == 0 for both → self-edge
        entry = _make_compat(0, 0, 0.9)
        g = build_fragment_graph(frags, [entry])
        assert g.n_edges == 0

    def test_multiple_entries_same_pair_max_weight(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        e1 = _make_compat(0, 1, 0.4)
        e2 = _make_compat(0, 1, 0.9)
        g = build_fragment_graph(frags, [e1, e2])
        assert g.weight(0, 1) == pytest.approx(0.9)


# ─── mst_ordering (extra) ─────────────────────────────────────────────────────

class TestMstOrderingExtra:
    def test_empty_graph_returns_empty(self):
        g = FragmentGraph()
        assert mst_ordering(g) == []

    def test_single_node_returns_single(self):
        g = FragmentGraph()
        g.nodes.add(7)
        assert mst_ordering(g) == [7]

    def test_two_nodes_returns_both(self):
        g = FragmentGraph()
        g.nodes.update({0, 1})
        g.add_edge(0, 1, 0.8)
        order = mst_ordering(g)
        assert sorted(order) == [0, 1]

    def test_all_nodes_covered(self):
        g = _line_graph(5)
        order = mst_ordering(g)
        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_no_duplicates(self):
        g = _triangle_graph()
        order = mst_ordering(g)
        assert len(order) == len(set(order))

    def test_disconnected_graph_all_covered(self):
        g = FragmentGraph()
        g.nodes.update({0, 1, 2, 3})
        g.add_edge(0, 1, 0.9)
        # nodes 2,3 isolated
        order = mst_ordering(g)
        assert sorted(order) == [0, 1, 2, 3]

    def test_returns_list(self):
        g = _triangle_graph()
        assert isinstance(mst_ordering(g), list)

    def test_length_equals_n_nodes(self):
        g = _line_graph(6)
        assert len(mst_ordering(g)) == 6


# ─── spectral_ordering (extra) ────────────────────────────────────────────────

class TestSpectralOrderingExtra:
    def test_empty_graph_returns_empty(self):
        g = FragmentGraph()
        assert spectral_ordering(g) == []

    def test_single_node_returns_single(self):
        g = FragmentGraph()
        g.nodes.add(3)
        assert spectral_ordering(g) == [3]

    def test_two_nodes_returns_both(self):
        g = FragmentGraph()
        g.nodes.update({0, 1})
        g.add_edge(0, 1, 0.5)
        order = spectral_ordering(g)
        assert sorted(order) == [0, 1]

    def test_length_equals_n_nodes(self):
        g = _triangle_graph()
        assert len(spectral_ordering(g)) == 3

    def test_no_duplicates(self):
        g = _line_graph(5)
        order = spectral_ordering(g)
        assert len(order) == len(set(order))

    def test_all_nodes_present(self):
        g = _line_graph(4)
        order = spectral_ordering(g)
        assert sorted(order) == [0, 1, 2, 3]

    def test_returns_list(self):
        g = _triangle_graph()
        assert isinstance(spectral_ordering(g), list)


# ─── random_walk_similarity (extra) ───────────────────────────────────────────

class TestRandomWalkSimilarityExtra:
    def test_empty_graph_returns_empty_matrix(self):
        g = FragmentGraph()
        R = random_walk_similarity(g)
        assert R.shape == (0, 0)

    def test_single_node_1x1(self):
        g = FragmentGraph()
        g.nodes.add(0)
        R = random_walk_similarity(g)
        assert R.shape == (1, 1)

    def test_shape_nxn(self):
        g = _triangle_graph()
        R = random_walk_similarity(g)
        assert R.shape == (3, 3)

    def test_values_nonneg(self):
        g = _triangle_graph()
        R = random_walk_similarity(g)
        assert (R >= 0).all()

    def test_rows_sum_to_one(self):
        g = _triangle_graph()
        R = random_walk_similarity(g)
        assert np.allclose(R.sum(axis=1), 1.0, atol=1e-6)

    def test_diagonal_nonneg(self):
        g = _line_graph(4)
        R = random_walk_similarity(g)
        assert (np.diag(R) >= 0).all()

    def test_returns_float64(self):
        g = _triangle_graph()
        R = random_walk_similarity(g)
        assert R.dtype == np.float64

    def test_different_alpha(self):
        g = _triangle_graph()
        R1 = random_walk_similarity(g, alpha=0.5)
        R2 = random_walk_similarity(g, alpha=0.9)
        # Should both be valid
        assert R1.shape == R2.shape


# ─── degree_centrality (extra) ────────────────────────────────────────────────

class TestDegreeCentralityExtra:
    def test_empty_graph_returns_empty_dict(self):
        g = FragmentGraph()
        assert degree_centrality(g) == {}

    def test_isolated_node_zero_centrality(self):
        g = FragmentGraph()
        g.nodes.add(5)
        c = degree_centrality(g)
        assert c[5] == pytest.approx(0.0)

    def test_all_values_in_0_1(self):
        g = _triangle_graph()
        c = degree_centrality(g)
        for v in c.values():
            assert 0.0 <= v <= 1.0 + 1e-10

    def test_max_value_is_one(self):
        g = _triangle_graph()
        c = degree_centrality(g)
        assert max(c.values()) == pytest.approx(1.0, abs=1e-10)

    def test_returns_dict_with_all_nodes(self):
        g = _line_graph(4)
        c = degree_centrality(g)
        assert set(c.keys()) == {0, 1, 2, 3}

    def test_hub_node_has_higher_centrality(self):
        # Star graph: node 0 connected to all others
        g = FragmentGraph()
        g.nodes.update({0, 1, 2, 3})
        for i in (1, 2, 3):
            g.add_edge(0, i, 0.9)
        c = degree_centrality(g)
        assert c[0] >= c[1]
        assert c[0] >= c[2]
        assert c[0] >= c[3]


# ─── analyze_graph (extra) ────────────────────────────────────────────────────

class TestAnalyzeGraphExtra:
    def test_returns_graph_match_result(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert isinstance(result, GraphMatchResult)

    def test_mst_order_subset_of_nodes(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert set(result.mst_order).issubset(g.nodes)

    def test_spectral_order_subset_of_nodes(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert set(result.spectral_order).issubset(g.nodes)

    def test_mst_edges_count_at_most_n_minus_1(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert len(result.mst_edges) <= g.n_nodes - 1

    def test_centrality_keys_equal_nodes(self):
        g = _triangle_graph()
        result = analyze_graph(g)
        assert set(result.centrality.keys()) == g.nodes

    def test_mst_edges_are_triples(self):
        g = _line_graph(4)
        result = analyze_graph(g)
        for e in result.mst_edges:
            assert len(e) == 3

    def test_empty_graph(self):
        g = FragmentGraph()
        result = analyze_graph(g)
        assert result.mst_order == []
        assert result.spectral_order == []
        assert result.centrality == {}

    def test_single_node_graph(self):
        g = FragmentGraph()
        g.nodes.add(0)
        result = analyze_graph(g)
        assert result.mst_order == [0]
        assert result.spectral_order == [0]
