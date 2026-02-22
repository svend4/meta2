"""
Тесты для puzzle_reconstruction.assembly.placement_optimizer.
"""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.placement_optimizer import (
    PlacementResult,
    score_placement,
    find_best_next,
    greedy_place,
    remove_worst_placed,
    iterative_place,
)
from puzzle_reconstruction.assembly.assembly_state import (
    AssemblyState,
    create_state,
    place_fragment,
    add_adjacency,
)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _uniform_matrix(n: int, value: float = 1.0) -> np.ndarray:
    """Матрица совместимости n×n с заданным значением (диагональ = 0)."""
    m = np.full((n, n), value, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _identity_matrix(n: int) -> np.ndarray:
    """Единичная матрица n×n."""
    return np.eye(n, dtype=np.float32)


def _chain_score_matrix(n: int, value: float = 1.0) -> np.ndarray:
    """Только соседние фрагменты имеют ненулевой балл."""
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        m[i, i + 1] = value
        m[i + 1, i] = value
    return m


# ─── PlacementResult ──────────────────────────────────────────────────────────

class TestPlacementResult:
    def test_fields_accessible(self):
        state = create_state(3)
        pr = PlacementResult(state=state, score=0.5, n_placed=0)
        assert pr.score == pytest.approx(0.5)
        assert pr.n_placed == 0
        assert pr.history == []
        assert pr.params == {}

    def test_repr_contains_n_placed(self):
        state = create_state(3)
        pr = PlacementResult(state=state, score=1.0, n_placed=3)
        assert "3" in repr(pr)


# ─── score_placement ──────────────────────────────────────────────────────────

class TestScorePlacement:
    def test_empty_state_zero_score(self):
        state = create_state(5)
        m = _uniform_matrix(5)
        assert score_placement(state, m) == pytest.approx(0.0)

    def test_single_fragment_zero_score(self):
        state = place_fragment(create_state(5), 0, (0.0, 0.0))
        m = _uniform_matrix(5)
        assert score_placement(state, m) == pytest.approx(0.0)

    def test_two_adjacent_fragments(self):
        state = place_fragment(create_state(5), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        state = add_adjacency(state, 0, 1)
        m = _uniform_matrix(5, value=2.0)
        # Одно ребро (0,1) с весом 2.0
        assert score_placement(state, m) == pytest.approx(2.0)

    def test_each_edge_counted_once(self):
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        state = place_fragment(state, 2, (2.0, 0.0))
        state = add_adjacency(state, 0, 1)
        state = add_adjacency(state, 1, 2)
        m = _uniform_matrix(3, value=3.0)
        # Два ребра (0,1) и (1,2), каждое по 3.0
        assert score_placement(state, m) == pytest.approx(6.0)

    def test_matrix_smaller_than_indices_ignored(self):
        state = place_fragment(create_state(10), 0, (0.0, 0.0))
        state = place_fragment(state, 8, (1.0, 0.0))
        state = add_adjacency(state, 0, 8)
        m = np.eye(4, dtype=np.float32)  # только 4×4
        # Индекс 8 >= 4, ребро не учитывается
        assert score_placement(state, m) == pytest.approx(0.0)


# ─── find_best_next ───────────────────────────────────────────────────────────

class TestFindBestNext:
    def test_no_candidates_returns_minus_one(self):
        state = create_state(3)
        for i in range(3):
            state = place_fragment(state, i, (float(i), 0.0))
        m = _uniform_matrix(3)
        idx, gain = find_best_next(state, m)
        assert idx == -1
        assert gain == pytest.approx(0.0)

    def test_selects_highest_gain(self):
        # 0 уже размещён; 1 имеет балл 5, 2 имеет балл 1
        m = np.array([[0., 5., 1.],
                      [5., 0., 1.],
                      [1., 1., 0.]], dtype=np.float32)
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        idx, gain = find_best_next(state, m)
        assert idx == 1
        assert gain == pytest.approx(5.0)

    def test_empty_placed_selects_first_candidate(self):
        state = create_state(3)
        m = _uniform_matrix(3, value=1.0)
        idx, gain = find_best_next(state, m)
        # Нет размещённых → gain = 0 для всех, возвращает первого
        assert idx in (0, 1, 2)
        assert gain == pytest.approx(0.0)

    def test_explicit_candidates(self):
        m = np.array([[0., 1., 9.],
                      [1., 0., 3.],
                      [9., 3., 0.]], dtype=np.float32)
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        # Кандидаты только [1]
        idx, gain = find_best_next(state, m, candidates=[1])
        assert idx == 1
        assert gain == pytest.approx(1.0)

    def test_already_placed_skipped(self):
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        m = _uniform_matrix(3, value=1.0)
        # Только 2 должен быть возвращён
        idx, _ = find_best_next(state, m)
        assert idx == 2


# ─── greedy_place ─────────────────────────────────────────────────────────────

class TestGreedyPlace:
    def test_returns_placement_result(self):
        m = _uniform_matrix(4)
        pr = greedy_place(4, m, root=0)
        assert isinstance(pr, PlacementResult)

    def test_all_fragments_placed(self):
        m = _uniform_matrix(5)
        pr = greedy_place(5, m)
        assert pr.n_placed == 5

    def test_n_fragments_1(self):
        m = np.array([[0.0]], dtype=np.float32)
        pr = greedy_place(1, m, root=0)
        assert pr.n_placed == 1
        assert pr.score == pytest.approx(0.0)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            greedy_place(0, np.zeros((0, 0), dtype=np.float32))

    def test_invalid_root_raises(self):
        with pytest.raises(ValueError):
            greedy_place(3, _uniform_matrix(3), root=5)

    def test_root_is_first_in_history(self):
        m = _uniform_matrix(4)
        pr = greedy_place(4, m, root=2)
        assert pr.history[0]["idx"] == 2

    def test_history_length_equals_n_placed(self):
        m = _uniform_matrix(6)
        pr = greedy_place(6, m)
        assert len(pr.history) == pr.n_placed

    def test_score_positive_for_connected(self):
        m = _uniform_matrix(4, value=2.0)
        pr = greedy_place(4, m)
        assert pr.score > 0.0

    def test_chain_matrix_places_all(self):
        m = _chain_score_matrix(5)
        pr = greedy_place(5, m)
        assert pr.n_placed == 5

    def test_params_method_greedy(self):
        m = _uniform_matrix(3)
        pr = greedy_place(3, m)
        assert pr.params["method"] == "greedy"


# ─── remove_worst_placed ──────────────────────────────────────────────────────

class TestRemoveWorstPlaced:
    def test_single_fragment_unchanged(self):
        m  = _uniform_matrix(3)
        pr = greedy_place(1, np.array([[0.0]], dtype=np.float32), root=0)
        result = remove_worst_placed(pr, m)
        assert result.n_placed == 1

    def test_reduces_n_placed_by_one(self):
        m  = _uniform_matrix(4, value=1.0)
        pr = greedy_place(4, m)
        result = remove_worst_placed(pr, m)
        assert result.n_placed == 3

    def test_root_not_removed(self):
        m  = _uniform_matrix(4, value=1.0)
        pr = greedy_place(4, m, root=0)
        result = remove_worst_placed(pr, m)
        assert 0 in result.state.placed

    def test_returns_new_placement_result(self):
        m  = _uniform_matrix(4)
        pr = greedy_place(4, m)
        result = remove_worst_placed(pr, m)
        assert isinstance(result, PlacementResult)

    def test_score_recalculated(self):
        m  = _uniform_matrix(4, value=1.0)
        pr = greedy_place(4, m)
        result = remove_worst_placed(pr, m)
        # Балл мог уменьшиться или измениться
        assert isinstance(result.score, float)

    def test_history_shrunken(self):
        m  = _uniform_matrix(5, value=1.0)
        pr = greedy_place(5, m)
        result = remove_worst_placed(pr, m)
        assert len(result.history) < len(pr.history)


# ─── iterative_place ──────────────────────────────────────────────────────────

class TestIterativePlacement:
    def test_returns_placement_result(self):
        m = _uniform_matrix(4)
        pr = iterative_place(4, m)
        assert isinstance(pr, PlacementResult)

    def test_all_placed(self):
        m = _uniform_matrix(5)
        pr = iterative_place(5, m)
        assert pr.n_placed == 5

    def test_score_not_worse_than_greedy(self):
        m      = _chain_score_matrix(6, value=2.0)
        greedy = greedy_place(6, m)
        itr    = iterative_place(6, m, max_iter=5)
        assert itr.score >= greedy.score - 1e-6

    def test_max_iter_zero_returns_greedy_result(self):
        m  = _uniform_matrix(4)
        pr = iterative_place(4, m, max_iter=0)
        assert pr.n_placed == 4

    def test_patience_stops_early(self):
        m = _uniform_matrix(3)
        # Должно завершиться без исключений
        pr = iterative_place(3, m, max_iter=100, patience=1)
        assert isinstance(pr, PlacementResult)

    def test_params_method_iterative_or_greedy(self):
        m  = _uniform_matrix(4)
        pr = iterative_place(4, m, max_iter=3)
        assert pr.params.get("method") in ("iterative", "greedy")
