"""Extra tests for puzzle_reconstruction.assembly.placement_optimizer."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.placement_optimizer import (
    PlacementResult,
    find_best_next,
    greedy_place,
    iterative_place,
    remove_worst_placed,
    score_placement,
)
from puzzle_reconstruction.assembly.assembly_state import (
    AssemblyState,
    create_state,
    place_fragment,
    add_adjacency,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _matrix(n=4, val=0.5):
    m = np.full((n, n), val, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _zeros(n=4):
    return np.zeros((n, n), dtype=np.float32)


def _state_with(n=4, placed=(0, 1)):
    state = create_state(n)
    for i, idx in enumerate(placed):
        state = place_fragment(state, idx, position=(float(i), 0.0))
    return state


def _result(n_placed=2, score=0.5, n=4):
    state = _state_with(n, list(range(n_placed)))
    return PlacementResult(state=state, score=score, n_placed=n_placed)


# ─── PlacementResult extras ───────────────────────────────────────────────────

class TestPlacementResultExtra:
    def test_repr_is_string(self):
        r = _result()
        assert isinstance(repr(r), str)

    def test_history_default_empty(self):
        state = _state_with()
        r = PlacementResult(state=state, score=0.0, n_placed=2)
        assert r.history == []

    def test_params_default_empty(self):
        state = _state_with()
        r = PlacementResult(state=state, score=0.0, n_placed=2)
        assert r.params == {}

    def test_score_zero_valid(self):
        r = _result(score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one_valid(self):
        r = _result(score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_n_placed_one_valid(self):
        r = _result(n_placed=1)
        assert r.n_placed == 1

    def test_params_stored(self):
        state = _state_with()
        r = PlacementResult(state=state, score=0.5, n_placed=2,
                            params={"method": "greedy", "root": 0})
        assert r.params["method"] == "greedy"

    def test_history_stored(self):
        state = _state_with()
        history = [{"step": 0, "idx": 0, "score_delta": 0.3}]
        r = PlacementResult(state=state, score=0.3, n_placed=1,
                            history=history)
        assert r.history[0]["idx"] == 0


# ─── score_placement extras ───────────────────────────────────────────────────

class TestScorePlacementExtra:
    def test_single_placed_no_adj_zero(self):
        state = _state_with(4, (0,))
        assert score_placement(state, _matrix()) == pytest.approx(0.0)

    def test_two_adjacent_symmetric_matrix(self):
        state = _state_with(4, (0, 1))
        state = add_adjacency(state, 0, 1)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 1] = m[1, 0] = 0.8
        s1 = score_placement(state, m)
        assert s1 == pytest.approx(0.8)

    def test_two_adjacencies(self):
        state = _state_with(4, (0, 1, 2))
        state = add_adjacency(state, 0, 1)
        state = add_adjacency(state, 1, 2)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 1] = m[1, 0] = 0.5
        m[1, 2] = m[2, 1] = 0.5
        s = score_placement(state, m)
        assert s == pytest.approx(1.0)

    def test_large_uniform_matrix(self):
        state = _state_with(6, (0, 1, 2))
        state = add_adjacency(state, 0, 1)
        matrix = _matrix(6, 1.0)
        s = score_placement(state, matrix)
        assert s >= 0.0

    def test_returns_float_type(self):
        state = _state_with()
        result = score_placement(state, _matrix())
        assert isinstance(result, float)


# ─── find_best_next extras ────────────────────────────────────────────────────

class TestFindBestNextExtra:
    def test_two_candidates_returns_best(self):
        state = _state_with(4, (0,))
        state = add_adjacency(state, 0, 1)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 1] = 0.9
        m[0, 2] = 0.3
        idx, gain = find_best_next(state, m, candidates=[1, 2])
        assert idx == 1

    def test_empty_candidates_list_behaves_gracefully(self):
        state = _state_with(4, (0,))
        idx, gain = find_best_next(state, _matrix(), candidates=[])
        assert idx == -1

    def test_gain_nonneg_for_nonneg_matrix(self):
        state = _state_with(4, (0,))
        _, gain = find_best_next(state, _matrix())
        assert gain >= 0.0

    def test_gain_is_float(self):
        state = _state_with(4, (0,))
        _, gain = find_best_next(state, _matrix())
        assert isinstance(gain, float)

    def test_single_candidate_returned(self):
        state = _state_with(4, (0,))
        idx, gain = find_best_next(state, _matrix(), candidates=[3])
        assert idx == 3

    def test_all_placed_returns_minus1(self):
        state = _state_with(3, (0, 1, 2))
        idx, gain = find_best_next(state, _matrix(3))
        assert idx == -1
        assert gain == pytest.approx(0.0)


# ─── greedy_place extras ──────────────────────────────────────────────────────

class TestGreedyPlaceExtra:
    def test_three_fragments_all_placed(self):
        result = greedy_place(3, _matrix(3))
        assert result.n_placed == 3

    def test_five_fragments_all_placed(self):
        result = greedy_place(5, _matrix(5))
        assert result.n_placed == 5

    def test_root_0_first_in_history(self):
        result = greedy_place(4, _matrix(), root=0)
        assert result.history[0]["idx"] == 0

    def test_root_3_first_in_history(self):
        result = greedy_place(4, _matrix(), root=3)
        assert result.history[0]["idx"] == 3

    def test_history_has_score_delta(self):
        result = greedy_place(4, _matrix())
        for entry in result.history:
            assert "score_delta" in entry

    def test_high_matrix_high_score(self):
        result = greedy_place(4, _matrix(4, 0.9))
        assert result.score >= 0.0

    def test_params_method_greedy(self):
        result = greedy_place(4, _matrix())
        assert result.params.get("method") == "greedy"

    def test_state_is_assembly_state(self):
        result = greedy_place(4, _matrix())
        assert isinstance(result.state, AssemblyState)

    def test_two_frags_root_1(self):
        result = greedy_place(2, _matrix(2), root=1)
        assert result.n_placed == 2
        assert result.history[0]["idx"] == 1


# ─── remove_worst_placed extras ───────────────────────────────────────────────

class TestRemoveWorstPlacedExtra:
    def test_three_to_two(self):
        base = greedy_place(3, _matrix(3))
        result = remove_worst_placed(base, _matrix(3))
        assert result.n_placed == 2

    def test_root_stays_zero_variant(self):
        base = greedy_place(4, _matrix(), root=2)
        result = remove_worst_placed(base, _matrix())
        assert 2 in result.state.placed

    def test_result_is_placement_result(self):
        base = greedy_place(4, _matrix())
        assert isinstance(remove_worst_placed(base, _matrix()), PlacementResult)

    def test_score_nonneg_after_remove(self):
        base = greedy_place(4, _matrix())
        result = remove_worst_placed(base, _matrix())
        assert result.score >= 0.0

    def test_five_to_four(self):
        base = greedy_place(5, _matrix(5))
        result = remove_worst_placed(base, _matrix(5))
        assert result.n_placed == 4


# ─── iterative_place extras ───────────────────────────────────────────────────

class TestIterativePlaceExtra:
    def test_three_fragments(self):
        result = iterative_place(3, _matrix(3))
        assert result.n_placed == 3

    def test_five_fragments(self):
        result = iterative_place(5, _matrix(5))
        assert result.n_placed == 5

    def test_score_nonneg(self):
        result = iterative_place(4, _matrix(), max_iter=3)
        assert result.score >= 0.0

    def test_returns_placement_result(self):
        result = iterative_place(4, _matrix())
        assert isinstance(result, PlacementResult)

    def test_max_iter_5(self):
        result = iterative_place(4, _matrix(), max_iter=5)
        assert result.n_placed == 4

    def test_root_choice(self):
        for root in range(4):
            result = iterative_place(4, _matrix(), root=root)
            assert result.n_placed == 4

    def test_zero_matrix(self):
        result = iterative_place(4, _zeros(4))
        assert result.n_placed == 4

    def test_patience_2(self):
        result = iterative_place(4, _matrix(), max_iter=10, patience=2)
        assert result.n_placed == 4
