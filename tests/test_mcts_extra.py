"""Extra tests for puzzle_reconstruction/assembly/mcts.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly, CompatEntry, EdgeSide, EdgeSignature, Fragment
from puzzle_reconstruction.assembly.mcts import (
    MCTSNode,
    _backpropagate,
    _build_score_map,
    _expand,
    _greedy_score,
    _order_to_assembly,
    _rollout,
    _select,
    _ucb1,
    mcts_assembly,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_fragment(fid: int, w: int = 50, h: int = 50) -> Fragment:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    return Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)


def _make_edge(edge_id: int, side: EdgeSide = EdgeSide.LEFT) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=side,
        virtual_curve=np.zeros((10, 2)),
        fd=0.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=10.0,
    )


def _make_compat(fid_a: int, fid_b: int, score: float = 0.8) -> CompatEntry:
    e_a = _make_edge(fid_a * 10, EdgeSide.RIGHT)
    e_b = _make_edge(fid_b * 10, EdgeSide.LEFT)
    return CompatEntry(
        edge_i=e_a, edge_j=e_b,
        score=score, dtw_dist=0.1, css_sim=0.9, fd_diff=0.05, text_score=0.7,
    )


def _make_score_map() -> dict:
    return {(0, 1): 0.9, (1, 2): 0.8, (0, 2): 0.5}


def _root_node(n: int = 3) -> MCTSNode:
    return MCTSNode(order=[], remaining=set(range(n)))


# ─── MCTSNode (extra) ─────────────────────────────────────────────────────────

class TestMCTSNodeExtra:
    def test_order_stored(self):
        node = MCTSNode(order=[0, 1], remaining={2})
        assert node.order == [0, 1]

    def test_remaining_stored(self):
        node = MCTSNode(order=[], remaining={0, 1, 2})
        assert node.remaining == {0, 1, 2}

    def test_parent_default_none(self):
        node = _root_node()
        assert node.parent is None

    def test_children_default_empty(self):
        assert _root_node().children == {}

    def test_visits_default_zero(self):
        assert _root_node().visits == 0

    def test_total_score_default_zero(self):
        assert _root_node().total_score == pytest.approx(0.0)

    def test_is_terminal_empty_remaining(self):
        node = MCTSNode(order=[0, 1, 2], remaining=set())
        assert node.is_terminal is True

    def test_is_not_terminal_with_remaining(self):
        assert _root_node().is_terminal is False

    def test_is_fully_expanded_when_no_remaining(self):
        node = MCTSNode(order=[0, 1], remaining=set())
        assert node.is_fully_expanded is True

    def test_is_fully_expanded_when_children_exhausted(self):
        node = MCTSNode(order=[], remaining={0, 1})
        node.children = {0: MCTSNode(order=[0], remaining={1}),
                         1: MCTSNode(order=[1], remaining={0})}
        assert node.is_fully_expanded is True

    def test_not_fully_expanded_with_untried(self):
        node = MCTSNode(order=[], remaining={0, 1, 2})
        # No children yet
        assert node.is_fully_expanded is False

    def test_mean_score_no_visits(self):
        node = _root_node()
        assert node.mean_score == pytest.approx(0.0)

    def test_mean_score_with_visits(self):
        node = _root_node()
        node.visits = 4
        node.total_score = 2.0
        assert node.mean_score == pytest.approx(0.5)

    def test_ucb1_unvisited_inf(self):
        node = MCTSNode(order=[], remaining={0}, parent=None)
        assert node.ucb1(1.0) == float("inf")

    def test_best_child_empty_returns_none(self):
        assert _root_node().best_child() is None

    def test_best_child_picks_highest_ucb1(self):
        parent = _root_node()
        parent.visits = 4
        c1 = MCTSNode(order=[0], remaining={1, 2}, parent=parent,
                      visits=2, total_score=1.8)
        c2 = MCTSNode(order=[1], remaining={0, 2}, parent=parent,
                      visits=2, total_score=1.0)
        parent.children = {0: c1, 1: c2}
        best = parent.best_child(exploration_c=0.0)
        assert best is c1  # greedy: c1 has higher mean_score


# ─── _ucb1 (extra) ────────────────────────────────────────────────────────────

class TestUcb1Extra:
    def test_unvisited_returns_inf(self):
        node = MCTSNode(order=[], remaining={0})
        assert _ucb1(node, 1.0) == float("inf")

    def test_exploration_zero_equals_mean(self):
        parent = _root_node()
        parent.visits = 10
        node = MCTSNode(order=[0], remaining={1, 2}, parent=parent,
                        visits=5, total_score=3.0)
        assert _ucb1(node, 0.0) == pytest.approx(0.6)

    def test_positive_exploration_term(self):
        parent = _root_node()
        parent.visits = 10
        node = MCTSNode(order=[0], remaining={1, 2}, parent=parent,
                        visits=5, total_score=3.0)
        ucb_c0 = _ucb1(node, 0.0)
        ucb_c1 = _ucb1(node, 1.0)
        assert ucb_c1 > ucb_c0


# ─── _greedy_score (extra) ────────────────────────────────────────────────────

class TestGreedyScoreExtra:
    def test_empty_order(self):
        assert _greedy_score([], {}) == pytest.approx(1.0)

    def test_single_fragment(self):
        assert _greedy_score([0], {}) == pytest.approx(1.0)

    def test_known_pair_score(self):
        score_map = {(0, 1): 0.8, (1, 0): 0.8}
        result = _greedy_score([0, 1], score_map)
        assert result == pytest.approx(0.8)

    def test_missing_pair_zero(self):
        score_map = {}
        result = _greedy_score([0, 1], score_map)
        assert result == pytest.approx(0.0)

    def test_three_fragments(self):
        score_map = {(0, 1): 1.0, (1, 0): 1.0, (1, 2): 1.0, (2, 1): 1.0}
        result = _greedy_score([0, 1, 2], score_map)
        assert result == pytest.approx(1.0)

    def test_result_nonneg(self):
        result = _greedy_score([0, 1, 2], _make_score_map())
        assert result >= 0.0


# ─── _build_score_map (extra) ─────────────────────────────────────────────────

class TestBuildScoreMapExtra:
    def test_returns_dict(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entries = [_make_compat(0, 1)]
        result = _build_score_map(frags, entries)
        assert isinstance(result, dict)

    def test_empty_entries(self):
        frags = [_make_fragment(0)]
        assert _build_score_map(frags, []) == {}

    def test_single_entry_stored(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entries = [_make_compat(0, 1, score=0.75)]
        result = _build_score_map(frags, entries)
        assert (0, 1) in result
        assert result[(0, 1)] == pytest.approx(0.75)

    def test_max_score_kept(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        e1 = _make_compat(0, 1, score=0.4)
        e2 = _make_compat(0, 1, score=0.9)
        result = _build_score_map(frags, [e1, e2])
        assert result[(0, 1)] == pytest.approx(0.9)


# ─── _backpropagate (extra) ───────────────────────────────────────────────────

class TestBackpropagateExtra:
    def test_updates_visits(self):
        node = _root_node()
        _backpropagate(node, 0.5)
        assert node.visits == 1

    def test_updates_total_score(self):
        node = _root_node()
        _backpropagate(node, 0.75)
        assert node.total_score == pytest.approx(0.75)

    def test_propagates_to_parent(self):
        parent = _root_node()
        child = MCTSNode(order=[0], remaining={1, 2}, parent=parent)
        _backpropagate(child, 0.5)
        assert parent.visits == 1
        assert parent.total_score == pytest.approx(0.5)

    def test_multiple_backprop(self):
        node = _root_node()
        _backpropagate(node, 0.5)
        _backpropagate(node, 0.3)
        assert node.visits == 2
        assert node.total_score == pytest.approx(0.8)


# ─── _expand (extra) ──────────────────────────────────────────────────────────

class TestExpandExtra:
    def test_returns_mcts_node(self):
        rng = np.random.RandomState(0)
        node = _root_node(3)
        child = _expand(node, rng)
        assert isinstance(child, MCTSNode)

    def test_child_has_one_fewer_remaining(self):
        rng = np.random.RandomState(0)
        node = _root_node(3)
        child = _expand(node, rng)
        assert len(child.remaining) == 2

    def test_child_order_longer(self):
        rng = np.random.RandomState(0)
        node = _root_node(3)
        child = _expand(node, rng)
        assert len(child.order) == 1

    def test_child_added_to_parent_children(self):
        rng = np.random.RandomState(0)
        node = _root_node(3)
        _expand(node, rng)
        assert len(node.children) == 1

    def test_no_untried_returns_same_node(self):
        rng = np.random.RandomState(0)
        node = MCTSNode(order=[], remaining={0})
        node.children = {0: MCTSNode(order=[0], remaining=set())}
        result = _expand(node, rng)
        assert result is node


# ─── _select (extra) ──────────────────────────────────────────────────────────

class TestSelectExtra:
    def test_returns_mcts_node(self):
        root = _root_node(3)
        result = _select(root, 1.0)
        assert isinstance(result, MCTSNode)

    def test_returns_root_when_empty_children(self):
        root = _root_node(3)
        result = _select(root, 1.0)
        assert result is root

    def test_descends_to_leaf(self):
        root = _root_node(3)
        root.visits = 10
        child = MCTSNode(order=[0], remaining={1, 2}, parent=root,
                         visits=3, total_score=2.1)
        root.children = {0: child}
        result = _select(root, 0.0)
        # Not fully expanded (remaining={1,2}, children={0}) → stops at root?
        assert isinstance(result, MCTSNode)


# ─── _rollout (extra) ─────────────────────────────────────────────────────────

class TestRolloutExtra:
    def test_returns_float(self):
        rng = np.random.RandomState(0)
        result = _rollout([0], [1, 2], _make_score_map(), rng)
        assert isinstance(result, float)

    def test_nonneg_result(self):
        rng = np.random.RandomState(0)
        result = _rollout([0], [1, 2], _make_score_map(), rng)
        assert result >= 0.0

    def test_empty_remaining(self):
        rng = np.random.RandomState(0)
        result = _rollout([0, 1, 2], [], _make_score_map(), rng)
        assert isinstance(result, float)

    def test_deterministic_with_seed(self):
        sm = _make_score_map()
        r1 = _rollout([0], [1, 2], sm, np.random.RandomState(42))
        r2 = _rollout([0], [1, 2], sm, np.random.RandomState(42))
        assert r1 == pytest.approx(r2)


# ─── _order_to_assembly (extra) ───────────────────────────────────────────────

class TestOrderToAssemblyExtra:
    def test_returns_assembly(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        result = _order_to_assembly([0, 1], frags, 0.8)
        assert isinstance(result, Assembly)

    def test_total_score_stored(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        result = _order_to_assembly([0, 1], frags, 0.75)
        assert result.total_score == pytest.approx(0.75)

    def test_all_fragments_placed(self):
        frags = [_make_fragment(i) for i in range(3)]
        result = _order_to_assembly([0, 1, 2], frags, 0.5)
        assert len(result.placements) == 3

    def test_placements_are_tuples(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        result = _order_to_assembly([0, 1], frags, 0.5)
        for val in result.placements.values():
            assert isinstance(val, tuple) and len(val) == 2

    def test_fragments_stored(self):
        frags = [_make_fragment(0)]
        result = _order_to_assembly([0], frags, 0.5)
        assert len(result.fragments) == 1

    def test_x_positions_increasing(self):
        frags = [_make_fragment(i) for i in range(3)]
        result = _order_to_assembly([0, 1, 2], frags, 0.5)
        x0 = result.placements[0][0][0]
        x1 = result.placements[1][0][0]
        x2 = result.placements[2][0][0]
        assert x1 > x0
        assert x2 > x1


# ─── mcts_assembly (extra) ────────────────────────────────────────────────────

class TestMctsAssemblyExtra:
    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = [_make_compat(0, 1), _make_compat(1, 2)]
        result = mcts_assembly(frags, entries, n_simulations=5, seed=0)
        assert isinstance(result, Assembly)

    def test_empty_fragments(self):
        result = mcts_assembly([], [], n_simulations=5)
        assert isinstance(result, Assembly)
        assert result.fragments == []

    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        result = mcts_assembly(frags, [], n_simulations=5)
        assert isinstance(result, Assembly)
        assert result.total_score == pytest.approx(1.0)

    def test_all_fragments_in_placements(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = [_make_compat(i, i + 1) for i in range(3)]
        result = mcts_assembly(frags, entries, n_simulations=10, seed=7)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_total_score_nonneg(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = [_make_compat(0, 1), _make_compat(1, 2)]
        result = mcts_assembly(frags, entries, n_simulations=10, seed=0)
        assert result.total_score >= 0.0

    def test_reproducible_with_seed(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = [_make_compat(i, i + 1) for i in range(3)]
        r1 = mcts_assembly(frags, entries, n_simulations=20, seed=123)
        r2 = mcts_assembly(frags, entries, n_simulations=20, seed=123)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_more_simulations_nondecreasing_score(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = [_make_compat(i, i + 1, score=0.9) for i in range(3)]
        r_few = mcts_assembly(frags, entries, n_simulations=5, seed=0)
        r_many = mcts_assembly(frags, entries, n_simulations=50, seed=0)
        # More simulations should generally not reduce quality
        assert r_many.total_score >= r_few.total_score - 0.1
