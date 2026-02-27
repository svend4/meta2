"""Тесты для puzzle_reconstruction/matching/graph_match.py."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.graph_match import (
    FragmentGraph,
    GraphMatchResult,
    mst_ordering,
    spectral_ordering,
    random_walk_similarity,
    degree_centrality,
    analyze_graph,
)
from puzzle_reconstruction.models import Fragment


def _make_fragment(fid):
    return Fragment(fragment_id=fid, image=np.zeros((10, 10, 3), dtype=np.uint8))


def _build_test_graph():
    g = FragmentGraph()
    for i in range(4):
        g.add_node(_make_fragment(i))
    g.add_edge(0, 1, 0.9)
    g.add_edge(1, 2, 0.7)
    g.add_edge(2, 3, 0.8)
    g.add_edge(0, 3, 0.5)
    return g


class TestFragmentGraph:
    def test_empty_graph(self):
        g = FragmentGraph()
        assert g.n_nodes == 0 and g.n_edges == 0

    def test_add_node(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        assert g.n_nodes == 1

    def test_add_edge(self):
        g = FragmentGraph()
        for i in range(2):
            g.add_node(_make_fragment(i))
        g.add_edge(0, 1, 0.8)
        assert g.n_edges == 1
        assert g.weight(0, 1) == pytest.approx(0.8)

    def test_max_weight_on_duplicate_edge(self):
        g = FragmentGraph()
        for i in range(2):
            g.add_node(_make_fragment(i))
        g.add_edge(0, 1, 0.5)
        g.add_edge(0, 1, 0.9)
        assert g.weight(0, 1) == pytest.approx(0.9)

    def test_weight_missing_edge_is_zero(self):
        assert _build_test_graph().weight(0, 2) == 0.0

    def test_adjacency_matrix_shape(self):
        A, fids = _build_test_graph().adjacency_matrix()
        assert A.shape == (4, 4) and len(fids) == 4

    def test_adjacency_matrix_symmetric(self):
        A, _ = _build_test_graph().adjacency_matrix()
        assert np.allclose(A, A.T)


class TestMstOrdering:
    def test_empty_graph(self):
        assert mst_ordering(FragmentGraph()) == []

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        assert mst_ordering(g) == [0]

    def test_all_nodes_included(self):
        order = mst_ordering(_build_test_graph())
        assert set(order) == {0, 1, 2, 3}

    def test_no_duplicates(self):
        order = mst_ordering(_build_test_graph())
        assert len(order) == len(set(order))


class TestSpectralOrdering:
    def test_empty_graph(self):
        assert spectral_ordering(FragmentGraph()) == []

    def test_all_nodes_included(self):
        order = spectral_ordering(_build_test_graph())
        assert set(order) == {0, 1, 2, 3}

    def test_no_duplicates(self):
        order = spectral_ordering(_build_test_graph())
        assert len(order) == len(set(order))


class TestRandomWalkSimilarity:
    def test_empty_graph(self):
        assert random_walk_similarity(FragmentGraph()).shape == (0, 0)

    def test_shape_nxn(self):
        R = random_walk_similarity(_build_test_graph())
        assert R.shape == (4, 4)

    def test_nonnegative(self):
        R = random_walk_similarity(_build_test_graph())
        assert np.all(R >= -1e-9)


class TestDegreeCentrality:
    def test_empty_returns_empty(self):
        assert degree_centrality(FragmentGraph()) == {}

    def test_all_fragments_included(self):
        c = degree_centrality(_build_test_graph())
        assert set(c.keys()) == {0, 1, 2, 3}

    def test_centrality_in_range(self):
        for val in degree_centrality(_build_test_graph()).values():
            assert 0.0 <= val <= 1.0


class TestAnalyzeGraph:
    def test_returns_graph_match_result(self):
        assert isinstance(analyze_graph(_build_test_graph()), GraphMatchResult)

    def test_mst_edges_count(self):
        g = _build_test_graph()
        r = analyze_graph(g)
        assert len(r.mst_edges) <= g.n_nodes - 1

    def test_centrality_keys(self):
        r = analyze_graph(_build_test_graph())
        assert set(r.centrality.keys()) == {0, 1, 2, 3}
