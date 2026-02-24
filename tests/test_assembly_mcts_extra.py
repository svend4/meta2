"""Extra tests for puzzle_reconstruction/assembly/mcts.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import (
    EdgeSignature,
    EdgeSide,
    Fragment,
    CompatEntry,
    Assembly,
)
from puzzle_reconstruction.assembly.mcts import (
    MCTSNode,
    mcts_assembly,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int, side: EdgeSide = EdgeSide.TOP) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id, side=side,
        virtual_curve=np.zeros((10, 2)), fd=1.0,
        css_vec=np.zeros(8), ifs_coeffs=np.zeros(4), length=100.0,
    )


def _fragment(fid: int, n_edges: int = 2) -> Fragment:
    base = fid * 10
    edges = [_edge(base + i) for i in range(n_edges)]
    return Fragment(
        fragment_id=fid,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        mask=np.ones((20, 20), dtype=np.uint8) * 255,
        contour=np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
        edges=edges,
    )


def _compat(e1: EdgeSignature, e2: EdgeSignature, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=e1, edge_j=e2, score=score,
        dtw_dist=1.0 - score, css_sim=score,
        fd_diff=0.1, text_score=score,
    )


# ─── MCTSNode ───────────────────────────────────────────────────────────────

class TestMCTSNodeExtra:
    def test_terminal(self):
        node = MCTSNode(order=[0, 1], remaining=set())
        assert node.is_terminal is True

    def test_not_terminal(self):
        node = MCTSNode(order=[0], remaining={1})
        assert node.is_terminal is False

    def test_is_fully_expanded_no_remaining(self):
        node = MCTSNode(order=[0], remaining=set())
        assert node.is_fully_expanded is True

    def test_is_fully_expanded_with_children(self):
        node = MCTSNode(order=[], remaining={0, 1})
        child0 = MCTSNode(order=[0], remaining={1}, parent=node)
        child1 = MCTSNode(order=[1], remaining={0}, parent=node)
        node.children = {0: child0, 1: child1}
        assert node.is_fully_expanded is True

    def test_not_fully_expanded(self):
        node = MCTSNode(order=[], remaining={0, 1})
        node.children = {0: MCTSNode(order=[0], remaining={1}, parent=node)}
        assert node.is_fully_expanded is False

    def test_mean_score(self):
        node = MCTSNode(order=[], remaining=set(), visits=4, total_score=8.0)
        assert node.mean_score == pytest.approx(2.0)

    def test_mean_score_zero_visits(self):
        node = MCTSNode(order=[], remaining=set(), visits=0, total_score=0.0)
        assert node.mean_score == pytest.approx(0.0)

    def test_ucb1_unvisited(self):
        node = MCTSNode(order=[], remaining=set(), visits=0)
        assert node.ucb1(1.41) == float("inf")

    def test_ucb1_visited(self):
        parent = MCTSNode(order=[], remaining={0}, visits=10, total_score=5.0)
        child = MCTSNode(order=[0], remaining=set(), parent=parent,
                         visits=2, total_score=1.0)
        val = child.ucb1(1.41)
        assert isinstance(val, float)
        assert val > child.mean_score  # exploration bonus

    def test_best_child_empty(self):
        node = MCTSNode(order=[], remaining=set())
        assert node.best_child() is None

    def test_best_child(self):
        parent = MCTSNode(order=[], remaining={0, 1}, visits=10)
        c0 = MCTSNode(order=[0], remaining={1}, parent=parent,
                      visits=5, total_score=3.0)
        c1 = MCTSNode(order=[1], remaining={0}, parent=parent,
                      visits=5, total_score=4.0)
        parent.children = {0: c0, 1: c1}
        best = parent.best_child(exploration_c=0.0)
        assert best is c1


# ─── mcts_assembly ──────────────────────────────────────────────────────────

class TestMCTSAssemblyExtra:
    def test_empty(self):
        result = mcts_assembly([], [])
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 0

    def test_single_fragment(self):
        f = _fragment(0)
        result = mcts_assembly([f], [], n_simulations=5)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 1
        assert 0 in result.placements

    def test_two_fragments(self):
        f0, f1 = _fragment(0), _fragment(1)
        e = _compat(f0.edges[0], f1.edges[0], 0.9)
        result = mcts_assembly([f0, f1], [e], n_simulations=20, seed=42)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 2
        assert result.total_score >= 0.0

    def test_three_fragments(self):
        frags = [_fragment(i) for i in range(3)]
        entries = [
            _compat(frags[0].edges[0], frags[1].edges[0], 0.8),
            _compat(frags[1].edges[1], frags[2].edges[0], 0.7),
        ]
        result = mcts_assembly(frags, entries, n_simulations=20, seed=42)
        assert isinstance(result, Assembly)
        assert len(result.placements) == 3

    def test_deterministic(self):
        frags = [_fragment(i) for i in range(3)]
        entries = [_compat(frags[0].edges[0], frags[1].edges[0], 0.8)]
        r1 = mcts_assembly(frags, entries, n_simulations=10, seed=42)
        r2 = mcts_assembly(frags, entries, n_simulations=10, seed=42)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_all_placed(self):
        frags = [_fragment(i) for i in range(4)]
        result = mcts_assembly(frags, [], n_simulations=10, seed=0)
        for f in frags:
            assert f.fragment_id in result.placements
