"""Extra tests for puzzle_reconstruction/matching/graph_match.py."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fragment(fid: int) -> Fragment:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    contour = np.array([[0, 0], [9, 0], [9, 9], [0, 9]])
    return Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)


def _graph(n=3, connected=True):
    g = FragmentGraph()
    for i in range(n):
        g.add_node(_fragment(i))
    if connected:
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j, 0.5 + 0.1 * (i + j))
    return g


# ─── FragmentGraph ──────────────────────────────────────────────────────────

class TestFragmentGraphExtra:
    def test_empty(self):
        g = FragmentGraph()
        assert g.n_nodes == 0 and g.n_edges == 0

    def test_add_node(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        assert g.n_nodes == 1
        assert 0 in g.nodes

    def test_add_edge(self):
        g = _graph(2)
        assert g.n_edges == 1
        assert g.weight(0, 1) > 0

    def test_weight_missing_edge(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        g.add_node(_fragment(1))
        assert g.weight(0, 1) == pytest.approx(0.0)

    def test_add_edge_max_weight(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        g.add_node(_fragment(1))
        g.add_edge(0, 1, 0.3)
        g.add_edge(0, 1, 0.8)
        assert g.weight(0, 1) == pytest.approx(0.8)

    def test_neighbors(self):
        g = _graph(3)
        nbrs = g.neighbors(0)
        nbr_ids = {n[0] for n in nbrs}
        assert 1 in nbr_ids and 2 in nbr_ids

    def test_sorted_nodes(self):
        g = _graph(3)
        assert g.sorted_nodes() == [0, 1, 2]

    def test_adjacency_matrix(self):
        g = _graph(3)
        A, fids = g.adjacency_matrix()
        assert A.shape == (3, 3)
        assert len(fids) == 3
        # Symmetric
        assert np.allclose(A, A.T)

    def test_laplacian(self):
        g = _graph(3)
        L, fids = g.laplacian()
        assert L.shape == (3, 3)
        # Row sums of laplacian should be 0
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_repr(self):
        g = _graph(2)
        r = repr(g)
        assert "nodes=2" in r and "edges=1" in r


# ─── GraphMatchResult ───────────────────────────────────────────────────────

class TestGraphMatchResultExtra:
    def test_summary(self):
        g = _graph(3)
        result = GraphMatchResult(
            mst_edges=[(0, 1, 0.5)],
            mst_order=[0, 1, 2],
            spectral_order=[0, 1, 2],
            centrality={0: 1.0, 1: 0.8, 2: 0.6},
            graph=g,
        )
        s = result.summary()
        assert "nodes=3" in s


# ─── mst_ordering ───────────────────────────────────────────────────────────

class TestMstOrderingExtra:
    def test_empty_graph(self):
        g = FragmentGraph()
        assert mst_ordering(g) == []

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        assert mst_ordering(g) == [0]

    def test_connected(self):
        g = _graph(4)
        order = mst_ordering(g)
        assert set(order) == {0, 1, 2, 3}
        assert len(order) == 4

    def test_disconnected_includes_all(self):
        g = FragmentGraph()
        for i in range(3):
            g.add_node(_fragment(i))
        g.add_edge(0, 1, 0.8)
        # Node 2 is isolated
        order = mst_ordering(g)
        assert set(order) == {0, 1, 2}


# ─── spectral_ordering ─────────────────────────────────────────────────────

class TestSpectralOrderingExtra:
    def test_empty(self):
        g = FragmentGraph()
        assert spectral_ordering(g) == []

    def test_single(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        assert spectral_ordering(g) == [0]

    def test_two_nodes(self):
        g = _graph(2)
        order = spectral_ordering(g)
        assert set(order) == {0, 1}

    def test_connected(self):
        g = _graph(4)
        order = spectral_ordering(g)
        assert set(order) == {0, 1, 2, 3}
        assert len(order) == 4


# ─── random_walk_similarity ─────────────────────────────────────────────────

class TestRandomWalkExtra:
    def test_empty(self):
        g = FragmentGraph()
        R = random_walk_similarity(g)
        assert R.shape == (0, 0)

    def test_connected(self):
        g = _graph(3)
        R = random_walk_similarity(g)
        assert R.shape == (3, 3)
        # Row sums approximately 1
        assert np.allclose(R.sum(axis=1), 1.0, atol=0.1)

    def test_nonnegative(self):
        g = _graph(3)
        R = random_walk_similarity(g)
        assert np.all(R >= -1e-10)


# ─── degree_centrality ──────────────────────────────────────────────────────

class TestDegreeCentralityExtra:
    def test_empty(self):
        g = FragmentGraph()
        assert degree_centrality(g) == {}

    def test_connected(self):
        g = _graph(3)
        c = degree_centrality(g)
        assert len(c) == 3
        # All values in [0, 1]
        for v in c.values():
            assert 0.0 <= v <= 1.0

    def test_max_is_one(self):
        g = _graph(3)
        c = degree_centrality(g)
        assert max(c.values()) == pytest.approx(1.0)

    def test_isolated_node(self):
        g = FragmentGraph()
        g.add_node(_fragment(0))
        c = degree_centrality(g)
        assert c[0] == pytest.approx(0.0)


# ─── analyze_graph ──────────────────────────────────────────────────────────

class TestAnalyzeGraphExtra:
    def test_returns_result(self):
        g = _graph(3)
        r = analyze_graph(g)
        assert isinstance(r, GraphMatchResult)
        assert r.graph is g

    def test_mst_order_length(self):
        g = _graph(4)
        r = analyze_graph(g)
        assert len(r.mst_order) == 4

    def test_spectral_order_length(self):
        g = _graph(4)
        r = analyze_graph(g)
        assert len(r.spectral_order) == 4

    def test_centrality_keys(self):
        g = _graph(3)
        r = analyze_graph(g)
        assert set(r.centrality.keys()) == {0, 1, 2}
