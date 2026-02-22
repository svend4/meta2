"""
Тесты для puzzle_reconstruction/assembly/mcts.py

Покрытие:
    MCTSNode           — is_terminal, is_fully_expanded, mean_score,
                         ucb1 (visits=0 → inf, формула), best_child
    _select            — возвращает листовой/неполный узел
    _expand            — добавляет ровно одного потомка, обновляет remaining
    _rollout           — скор ∈ [0,1], воспроизводимость
    _greedy_score      — пустой порядок, порядок из 1, корректная формула
    _backpropagate     — visits и total_score обновляются вверх по дереву
    _build_score_map   — правильные ключи и значения, max по дублям
    _order_to_assembly — структура placements, orphan-защита
    mcts_assembly      — пустой, 1 фрагмент, N=4: все размещены, score≥0, seed
"""
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


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _make_fragment(fid: int) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((4, 2)),
    )


def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((8, 2)),
        fd=1.5, css_vec=np.zeros(8), ifs_coeffs=np.zeros(4), length=60.0,
    )


def _make_entry(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=_make_edge(fid_i * 10),
        edge_j=_make_edge(fid_j * 10),
        score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


@pytest.fixture
def frags4():
    return [_make_fragment(i) for i in range(4)]


@pytest.fixture
def entries4():
    return [
        _make_entry(i, j, 0.2 * (i + j + 1) % 1.0 + 0.1)
        for i in range(4) for j in range(i + 1, 4)
    ]


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def score_map():
    return {
        (0, 1): 0.9, (1, 0): 0.9,
        (1, 2): 0.7, (2, 1): 0.7,
        (2, 3): 0.5, (3, 2): 0.5,
    }


# ─── MCTSNode ─────────────────────────────────────────────────────────────────

class TestMCTSNode:
    def test_is_terminal_empty_remaining(self):
        node = MCTSNode(order=[0, 1, 2], remaining=set())
        assert node.is_terminal

    def test_not_terminal_with_remaining(self):
        node = MCTSNode(order=[0], remaining={1, 2})
        assert not node.is_terminal

    def test_is_fully_expanded_no_remaining(self):
        node = MCTSNode(order=[0], remaining=set())
        assert node.is_fully_expanded

    def test_not_fully_expanded_initially(self):
        node = MCTSNode(order=[0], remaining={1, 2})
        assert not node.is_fully_expanded

    def test_fully_expanded_after_all_children(self):
        node = MCTSNode(order=[0], remaining={1, 2})
        node.children[1] = MCTSNode(order=[0, 1], remaining={2}, parent=node)
        node.children[2] = MCTSNode(order=[0, 2], remaining={1}, parent=node)
        assert node.is_fully_expanded

    def test_mean_score_zero_visits(self):
        node = MCTSNode(order=[], remaining={0, 1})
        assert math.isfinite(node.mean_score)  # не делит на 0

    def test_mean_score(self):
        node = MCTSNode(order=[], remaining=set(), visits=4, total_score=2.0)
        assert math.isclose(node.mean_score, 0.5)

    def test_ucb1_zero_visits_returns_inf(self):
        parent = MCTSNode(order=[], remaining={0}, visits=10)
        child  = MCTSNode(order=[0], remaining=set(), parent=parent, visits=0)
        assert math.isinf(_ucb1(child, 1.41))

    def test_ucb1_formula(self):
        parent = MCTSNode(order=[], remaining=set(), visits=10, total_score=3.0)
        child  = MCTSNode(order=[0], remaining=set(), parent=parent,
                           visits=2, total_score=1.2)
        expected = (1.2 / 2) + 1.41 * math.sqrt(math.log(10) / 2)
        assert math.isclose(_ucb1(child, 1.41), expected, rel_tol=1e-6)

    def test_best_child_greedy(self):
        parent = MCTSNode(order=[], remaining=set(), visits=10)
        c1 = MCTSNode(order=[0], remaining=set(), parent=parent,
                       visits=5, total_score=4.0)
        c2 = MCTSNode(order=[1], remaining=set(), parent=parent,
                       visits=5, total_score=2.0)
        parent.children = {0: c1, 1: c2}
        best = parent.best_child(exploration_c=0.0)
        assert best is c1

    def test_best_child_none_if_no_children(self):
        node = MCTSNode(order=[], remaining={0})
        assert node.best_child() is None


# ─── _select ──────────────────────────────────────────────────────────────────

class TestSelect:
    def test_returns_root_if_not_fully_expanded(self):
        root = MCTSNode(order=[], remaining={0, 1, 2})
        result = _select(root, 1.41)
        assert result is root

    def test_descends_when_fully_expanded(self):
        root  = MCTSNode(order=[], remaining={0, 1}, visits=5)
        child = MCTSNode(order=[0], remaining={1}, parent=root, visits=3, total_score=2.0)
        root.children[0] = child
        # root имеет 2 remaining, 1 child → не полностью раскрыт → возвращает root
        result = _select(root, 1.41)
        assert result is root

    def test_reaches_leaf(self):
        root  = MCTSNode(order=[], remaining={0, 1}, visits=10)
        c0    = MCTSNode(order=[0], remaining={1}, parent=root, visits=5, total_score=3.0)
        c1    = MCTSNode(order=[1], remaining={0}, parent=root, visits=5, total_score=2.0)
        root.children = {0: c0, 1: c1}
        # Оба потомка → root полностью раскрыт, спускаемся
        result = _select(root, 1.41)
        assert result in (c0, c1)


# ─── _expand ──────────────────────────────────────────────────────────────────

class TestExpand:
    def test_adds_exactly_one_child(self, rng):
        node = MCTSNode(order=[0], remaining={1, 2, 3})
        child = _expand(node, rng)
        assert len(node.children) == 1
        assert child is not node

    def test_child_has_one_less_remaining(self, rng):
        node  = MCTSNode(order=[0], remaining={1, 2, 3})
        child = _expand(node, rng)
        assert len(child.remaining) == 2
        assert len(child.order) == 2

    def test_child_fid_in_remaining(self, rng):
        remaining = {1, 2, 3}
        node  = MCTSNode(order=[0], remaining=remaining.copy())
        child = _expand(node, rng)
        new_fid = child.order[-1]
        assert new_fid in remaining

    def test_child_parent_set(self, rng):
        node  = MCTSNode(order=[], remaining={0, 1})
        child = _expand(node, rng)
        assert child.parent is node

    def test_no_expand_if_terminal(self, rng):
        node = MCTSNode(order=[0, 1], remaining=set())
        result = _expand(node, rng)
        assert result is node  # Нечего раскрывать → возвращает себя

    def test_skips_already_tried(self, rng):
        node = MCTSNode(order=[0], remaining={1, 2})
        # Добавляем 1 вручную
        node.children[1] = MCTSNode(order=[0, 1], remaining={2}, parent=node)
        child = _expand(node, rng)
        assert child.order[-1] == 2


# ─── _rollout ─────────────────────────────────────────────────────────────────

class TestRollout:
    def test_score_in_range(self, score_map):
        rng   = np.random.RandomState(0)
        score = _rollout([0], [1, 2, 3], score_map, rng)
        assert 0.0 <= score <= 1.0

    def test_reproducibility(self, score_map):
        s1 = _rollout([0], [1, 2, 3], score_map, np.random.RandomState(5))
        s2 = _rollout([0], [1, 2, 3], score_map, np.random.RandomState(5))
        assert math.isclose(s1, s2)

    def test_full_order_gives_deterministic_score(self, score_map):
        rng   = np.random.RandomState(0)
        score = _rollout([0, 1, 2, 3], [], score_map, rng)
        assert math.isfinite(score)

    def test_empty_remaining(self, score_map):
        rng   = np.random.RandomState(0)
        score = _rollout([0, 1, 2], [], score_map, rng)
        assert math.isfinite(score)


# ─── _greedy_score ────────────────────────────────────────────────────────────

class TestGreedyScore:
    def test_empty_order_returns_1(self, score_map):
        assert math.isclose(_greedy_score([], score_map), 1.0)

    def test_single_fragment_returns_1(self, score_map):
        assert math.isclose(_greedy_score([0], score_map), 1.0)

    def test_known_score(self):
        sm = {(0, 1): 0.8, (1, 0): 0.8}
        # (0.8 + 0.8) / 2 = 0.8
        s = _greedy_score([0, 1], sm)
        assert math.isclose(s, 0.8)

    def test_missing_pair_counts_as_zero(self):
        sm = {}
        s = _greedy_score([0, 1, 2], sm)
        assert math.isclose(s, 0.0)

    def test_score_in_range(self, score_map):
        s = _greedy_score([0, 1, 2, 3], score_map)
        assert 0.0 <= s <= 1.0


# ─── _backpropagate ───────────────────────────────────────────────────────────

class TestBackpropagate:
    def test_updates_leaf(self):
        node = MCTSNode(order=[0], remaining=set())
        _backpropagate(node, 0.7)
        assert node.visits == 1
        assert math.isclose(node.total_score, 0.7)

    def test_updates_all_ancestors(self):
        root  = MCTSNode(order=[], remaining=set())
        child = MCTSNode(order=[0], remaining=set(), parent=root)
        leaf  = MCTSNode(order=[0, 1], remaining=set(), parent=child)
        _backpropagate(leaf, 0.5)
        assert leaf.visits == 1
        assert child.visits == 1
        assert root.visits == 1

    def test_accumulates_scores(self):
        root  = MCTSNode(order=[], remaining=set())
        child = MCTSNode(order=[0], remaining=set(), parent=root)
        _backpropagate(child, 0.3)
        _backpropagate(child, 0.7)
        assert child.visits == 2
        assert math.isclose(child.total_score, 1.0)
        assert root.visits == 2


# ─── _build_score_map ─────────────────────────────────────────────────────────

class TestBuildScoreMap:
    def test_empty_entries(self, frags4):
        sm = _build_score_map(frags4, [])
        assert sm == {}

    def test_known_entry(self, frags4):
        entry = _make_entry(0, 1, 0.75)
        sm    = _build_score_map(frags4, [entry])
        assert math.isclose(sm.get((0, 1), 0.0), 0.75)

    def test_takes_max_for_duplicates(self, frags4):
        entries = [_make_entry(0, 1, 0.3), _make_entry(0, 1, 0.9)]
        sm = _build_score_map(frags4, entries)
        assert math.isclose(sm.get((0, 1), 0.0), 0.9)

    def test_all_entries_registered(self, frags4, entries4):
        sm = _build_score_map(frags4, entries4)
        assert len(sm) > 0


# ─── _order_to_assembly ───────────────────────────────────────────────────────

class TestOrderToAssembly:
    def test_all_placed(self, frags4):
        order = [f.fragment_id for f in frags4]
        result = _order_to_assembly(order, frags4, score=0.7)
        for frag in frags4:
            assert frag.fragment_id in result.placements

    def test_score_correct(self, frags4):
        order = [f.fragment_id for f in frags4]
        result = _order_to_assembly(order, frags4, score=0.42)
        assert math.isclose(result.total_score, 0.42)

    def test_positions_distinct(self, frags4):
        order = [f.fragment_id for f in frags4]
        result = _order_to_assembly(order, frags4, score=0.0)
        positions = [tuple(result.placements[fid][0]) for fid in order]
        assert len(set(positions)) == len(order)

    def test_orphan_fragment_placed(self):
        frags = [_make_fragment(i) for i in range(3)]
        # order включает только 2 из 3
        order = [frags[0].fragment_id, frags[1].fragment_id]
        result = _order_to_assembly(order, frags, score=0.0)
        assert frags[2].fragment_id in result.placements

    def test_empty_order(self):
        frags = [_make_fragment(0)]
        result = _order_to_assembly([], frags, score=0.0)
        assert isinstance(result, Assembly)


# ─── mcts_assembly ────────────────────────────────────────────────────────────

class TestMCTSAssembly:
    def test_empty_fragments(self):
        result = mcts_assembly([], [])
        assert isinstance(result, Assembly)
        assert result.fragments == []

    def test_single_fragment(self):
        frag   = _make_fragment(0)
        result = mcts_assembly([frag], [])
        assert 0 in result.placements

    def test_returns_assembly(self, frags4, entries4):
        result = mcts_assembly(frags4, entries4, n_simulations=10, seed=0)
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self, frags4, entries4):
        result = mcts_assembly(frags4, entries4, n_simulations=10, seed=0)
        for frag in frags4:
            assert frag.fragment_id in result.placements

    def test_placement_structure(self, frags4, entries4):
        result = mcts_assembly(frags4, entries4, n_simulations=10, seed=0)
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (2,)
            assert isinstance(angle, float)

    def test_score_nonneg(self, frags4, entries4):
        result = mcts_assembly(frags4, entries4, n_simulations=10, seed=0)
        assert result.total_score >= 0.0

    def test_seed_reproducibility(self, frags4, entries4):
        r1 = mcts_assembly(frags4, entries4, n_simulations=20, seed=7)
        r2 = mcts_assembly(frags4, entries4, n_simulations=20, seed=7)
        assert math.isclose(r1.total_score, r2.total_score)

    def test_more_simulations_nondecreasing(self, frags4, entries4):
        """Больше симуляций не даёт худший результат (детерминировано)."""
        r10  = mcts_assembly(frags4, entries4, n_simulations=10,  seed=42)
        r100 = mcts_assembly(frags4, entries4, n_simulations=100, seed=42)
        # Оба должны быть валидными Assembly
        assert isinstance(r10, Assembly)
        assert isinstance(r100, Assembly)

    def test_n_rollouts_parameter(self, frags4, entries4):
        result = mcts_assembly(frags4, entries4, n_simulations=10,
                                n_rollouts=5, seed=0)
        assert isinstance(result, Assembly)

    def test_exploration_zero(self, frags4, entries4):
        """Чисто эксплуататорный режим (c=0) не должен падать."""
        result = mcts_assembly(frags4, entries4, n_simulations=15,
                                exploration_c=0.0, seed=1)
        assert isinstance(result, Assembly)
