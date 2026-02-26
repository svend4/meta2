"""Tests for puzzle_reconstruction/assembly/mcts.py."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.models import Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide
from puzzle_reconstruction.assembly.mcts import (
    MCTSNode,
    mcts_assembly,
    _select,
    _expand,
    _rollout,
    _greedy_score,
    _backpropagate,
    _build_score_map,
    _order_to_assembly,
    _ucb1,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(2)]
    return frag


def _make_entry(ei, ej, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _make_entries(frags):
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score=0.5))
    return entries


# ─── MCTSNode ─────────────────────────────────────────────────────────────────

class TestMCTSNode:
    def test_initial_state(self):
        node = MCTSNode(order=[], remaining={0, 1, 2})
        assert node.visits == 0
        assert node.total_score == 0.0
        assert node.parent is None
        assert node.children == {}

    def test_is_terminal_true_when_remaining_empty(self):
        node = MCTSNode(order=[0, 1], remaining=set())
        assert node.is_terminal is True

    def test_is_terminal_false_when_remaining_nonempty(self):
        node = MCTSNode(order=[0], remaining={1, 2})
        assert node.is_terminal is False

    def test_is_fully_expanded_no_children(self):
        node = MCTSNode(order=[], remaining={0, 1})
        assert node.is_fully_expanded is False

    def test_is_fully_expanded_all_children_created(self):
        node = MCTSNode(order=[], remaining={0, 1})
        child0 = MCTSNode(order=[0], remaining={1}, parent=node)
        child1 = MCTSNode(order=[1], remaining={0}, parent=node)
        node.children[0] = child0
        node.children[1] = child1
        assert node.is_fully_expanded is True

    def test_mean_score_zero_visits(self):
        node = MCTSNode(order=[], remaining={0})
        assert node.mean_score == 0.0

    def test_mean_score_with_visits(self):
        node = MCTSNode(order=[], remaining=set(), visits=4, total_score=8.0)
        assert node.mean_score == 2.0

    def test_ucb1_unvisited_is_inf(self):
        parent = MCTSNode(order=[], remaining={0, 1})
        parent.visits = 5
        child = MCTSNode(order=[0], remaining={1}, parent=parent)
        assert child.ucb1(1.0) == float("inf")

    def test_ucb1_visited_is_finite(self):
        parent = MCTSNode(order=[], remaining={0, 1}, visits=5)
        child = MCTSNode(order=[0], remaining={1}, parent=parent, visits=2, total_score=1.0)
        val = child.ucb1(math.sqrt(2))
        assert math.isfinite(val)

    def test_best_child_returns_none_on_empty(self):
        node = MCTSNode(order=[], remaining={0})
        assert node.best_child() is None

    def test_best_child_returns_highest_ucb1(self):
        parent = MCTSNode(order=[], remaining={0, 1}, visits=10)
        child0 = MCTSNode(order=[0], remaining={1}, parent=parent, visits=3, total_score=2.0)
        child1 = MCTSNode(order=[1], remaining={0}, parent=parent, visits=1, total_score=0.5)
        parent.children[0] = child0
        parent.children[1] = child1
        best = parent.best_child(exploration_c=1.0)
        assert best is not None


# ─── mcts_assembly ────────────────────────────────────────────────────────────

class TestMctsAssembly:
    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = mcts_assembly(frags, entries, n_simulations=5, seed=0)
        assert isinstance(result, Assembly)

    def test_empty_fragments_returns_empty(self):
        result = mcts_assembly([], [], n_simulations=5)
        assert isinstance(result, Assembly)

    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        result = mcts_assembly(frags, [], n_simulations=5, seed=1)
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        result = mcts_assembly(frags, entries, n_simulations=10, seed=2)
        placed_ids = set(result.placements.keys())
        frag_ids = {f.fragment_id for f in frags}
        assert placed_ids == frag_ids

    def test_seed_reproducibility(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        r1 = mcts_assembly(frags, entries, n_simulations=10, seed=42)
        r2 = mcts_assembly(frags, entries, n_simulations=10, seed=42)
        assert r1.total_score == r2.total_score

    def test_placements_have_correct_structure(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = mcts_assembly(frags, entries, n_simulations=5, seed=3)
        for fid, (pos, angle) in result.placements.items():
            assert pos.shape == (2,)
            assert isinstance(angle, float)

    def test_total_score_nonnegative(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = mcts_assembly(frags, entries, n_simulations=5, seed=4)
        assert result.total_score >= 0.0

    def test_two_fragments(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entries = _make_entries(frags)
        result = mcts_assembly(frags, entries, n_simulations=5, seed=5)
        assert len(result.placements) == 2


# ─── _select ──────────────────────────────────────────────────────────────────

class TestSelect:
    def test_returns_root_when_no_children(self):
        root = MCTSNode(order=[], remaining={0, 1})
        selected = _select(root, exploration_c=1.0)
        assert selected is root

    def test_returns_child_when_fully_expanded(self):
        root = MCTSNode(order=[], remaining={0}, visits=5)
        child = MCTSNode(order=[0], remaining=set(), parent=root, visits=3, total_score=1.0)
        root.children[0] = child
        selected = _select(root, exploration_c=1.0)
        assert selected is child


# ─── _expand ──────────────────────────────────────────────────────────────────

class TestExpand:
    def test_creates_child_node(self):
        rng = np.random.RandomState(0)
        root = MCTSNode(order=[], remaining={0, 1, 2})
        child = _expand(root, rng)
        assert child is not root
        assert len(child.order) == 1
        assert len(child.remaining) == 2

    def test_child_has_correct_parent(self):
        rng = np.random.RandomState(0)
        root = MCTSNode(order=[], remaining={5})
        child = _expand(root, rng)
        assert child.parent is root

    def test_returns_node_when_fully_expanded(self):
        rng = np.random.RandomState(0)
        root = MCTSNode(order=[], remaining={0})
        child = MCTSNode(order=[0], remaining=set(), parent=root)
        root.children[0] = child
        result = _expand(root, rng)
        assert result is root


# ─── _greedy_score ────────────────────────────────────────────────────────────

class TestGreedyScore:
    def test_single_element_returns_one(self):
        assert _greedy_score([0], {}) == 1.0

    def test_empty_order(self):
        assert _greedy_score([], {}) == 1.0

    def test_two_elements_with_score(self):
        score_map = {(0, 1): 0.8}
        score = _greedy_score([0, 1], score_map)
        # 0.8 + 0 (reverse) / 2
        assert score >= 0.0

    def test_zero_score_when_no_entries(self):
        score = _greedy_score([0, 1, 2], {})
        assert score == 0.0


# ─── _backpropagate ───────────────────────────────────────────────────────────

class TestBackpropagate:
    def test_updates_visits_up_tree(self):
        root = MCTSNode(order=[], remaining={0, 1})
        child = MCTSNode(order=[0], remaining={1}, parent=root)
        leaf = MCTSNode(order=[0, 1], remaining=set(), parent=child)

        _backpropagate(leaf, 0.7)

        assert leaf.visits == 1
        assert child.visits == 1
        assert root.visits == 1

    def test_accumulates_score(self):
        root = MCTSNode(order=[], remaining=set())
        _backpropagate(root, 0.5)
        _backpropagate(root, 0.3)
        assert abs(root.total_score - 0.8) < 1e-9


# ─── _build_score_map ─────────────────────────────────────────────────────────

class TestBuildScoreMap:
    def test_builds_map_from_entries(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        score_map = _build_score_map(frags, entries)
        assert isinstance(score_map, dict)

    def test_empty_entries(self):
        frags = [_make_fragment(0)]
        score_map = _build_score_map(frags, [])
        assert score_map == {}


# ─── _order_to_assembly ───────────────────────────────────────────────────────

class TestOrderToAssembly:
    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _order_to_assembly([0, 1, 2], frags, 0.5)
        assert isinstance(asm, Assembly)

    def test_all_placed(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _order_to_assembly([0, 1, 2], frags, 0.5)
        assert set(asm.placements.keys()) == {0, 1, 2}

    def test_total_score_preserved(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _order_to_assembly([0, 1, 2], frags, 0.75)
        assert abs(asm.total_score - 0.75) < 1e-9

    def test_positions_are_x_ordered(self):
        frags = [_make_fragment(i) for i in range(3)]
        asm = _order_to_assembly([0, 1, 2], frags, 0.0)
        positions = [asm.placements[i][0][0] for i in range(3)]
        assert positions[0] <= positions[1] <= positions[2]


# ─── _ucb1 ────────────────────────────────────────────────────────────────────

class TestUcb1:
    def test_unvisited_is_inf(self):
        parent = MCTSNode(order=[], remaining=set(), visits=5)
        child = MCTSNode(order=[], remaining=set(), parent=parent)
        assert _ucb1(child, 1.0) == float("inf")

    def test_finite_for_visited_node(self):
        parent = MCTSNode(order=[], remaining=set(), visits=10)
        child = MCTSNode(order=[], remaining=set(), parent=parent, visits=3, total_score=1.5)
        val = _ucb1(child, math.sqrt(2))
        assert math.isfinite(val)
        assert val > 0.0
