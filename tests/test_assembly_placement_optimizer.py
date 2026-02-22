"""Расширенные тесты для puzzle_reconstruction/assembly/placement_optimizer.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _matrix(n: int = 4, val: float = 0.5) -> np.ndarray:
    """Uniform score matrix (off-diagonal = val, diagonal = 0)."""
    m = np.full((n, n), val, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _identity_matrix(n: int = 4) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float32)


def _state_with(n: int = 4, placed_idxs=(0, 1)) -> AssemblyState:
    state = create_state(n)
    for i, idx in enumerate(placed_idxs):
        state = place_fragment(state, idx, position=(float(i), 0.0))
    return state


# ─── TestPlacementResult ──────────────────────────────────────────────────────

class TestPlacementResult:
    def _make(self, n_placed=2, score=0.5):
        state = _state_with(4, list(range(n_placed)))
        return PlacementResult(
            state=state,
            score=score,
            n_placed=n_placed,
            history=[{"step": i, "idx": i, "score_delta": 0.1} for i in range(n_placed)],
            params={"method": "test"},
        )

    def test_stores_state(self):
        r = self._make()
        assert isinstance(r.state, AssemblyState)

    def test_stores_score(self):
        r = self._make(score=0.75)
        assert r.score == pytest.approx(0.75)

    def test_stores_n_placed(self):
        r = self._make(n_placed=3)
        assert r.n_placed == 3

    def test_default_history_list(self):
        state = _state_with()
        r = PlacementResult(state=state, score=0.0, n_placed=2)
        assert isinstance(r.history, list)

    def test_default_params_dict(self):
        state = _state_with()
        r = PlacementResult(state=state, score=0.0, n_placed=2)
        assert isinstance(r.params, dict)

    def test_repr_contains_n_placed(self):
        r = self._make(n_placed=3)
        assert "3" in repr(r)

    def test_repr_contains_score(self):
        r = self._make(score=0.12345)
        assert "0.12" in repr(r) or "0.1234" in repr(r)


# ─── TestScorePlacement ───────────────────────────────────────────────────────

class TestScorePlacement:
    def test_returns_float(self):
        state = _state_with()
        matrix = _matrix()
        result = score_placement(state, matrix)
        assert isinstance(result, float)

    def test_empty_placed_zero(self):
        state = create_state(4)
        matrix = _matrix()
        assert score_placement(state, matrix) == pytest.approx(0.0)

    def test_no_adjacency_zero(self):
        state = _state_with(4, (0, 1))
        matrix = _matrix(4, 0.5)
        # No adjacency added → score = 0
        assert score_placement(state, matrix) == pytest.approx(0.0)

    def test_with_adjacency_positive(self):
        state = _state_with(4, (0, 1))
        state = add_adjacency(state, 0, 1)
        matrix = _matrix(4, 0.5)
        result = score_placement(state, matrix)
        assert result > 0.0

    def test_nonneg_for_nonneg_matrix(self):
        state = _state_with(4, (0, 1, 2))
        state = add_adjacency(state, 0, 1)
        state = add_adjacency(state, 1, 2)
        matrix = _matrix(4, 0.3)
        assert score_placement(state, matrix) >= 0.0

    def test_zero_matrix_zero_score(self):
        state = _state_with(4, (0, 1))
        state = add_adjacency(state, 0, 1)
        matrix = np.zeros((4, 4), dtype=np.float32)
        assert score_placement(state, matrix) == pytest.approx(0.0)

    def test_counts_each_edge_once(self):
        # Only adjacency (0,1) with matrix[0,1]=matrix[1,0]=1.0
        state = _state_with(4, (0, 1))
        state = add_adjacency(state, 0, 1)
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 1] = matrix[1, 0] = 1.0
        result = score_placement(state, matrix)
        assert result == pytest.approx(1.0)


# ─── TestFindBestNext ─────────────────────────────────────────────────────────

class TestFindBestNext:
    def test_returns_tuple(self):
        state = _state_with(4, (0,))
        result = find_best_next(state, _matrix())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_no_candidates_returns_minus1(self):
        state = _state_with(4, (0, 1, 2, 3))  # all placed
        idx, gain = find_best_next(state, _matrix())
        assert idx == -1
        assert gain == pytest.approx(0.0)

    def test_idx_not_placed(self):
        state = _state_with(4, (0,))
        idx, gain = find_best_next(state, _matrix())
        assert idx not in state.placed

    def test_idx_in_valid_range(self):
        state = _state_with(4, (0,))
        idx, gain = find_best_next(state, _matrix())
        n = _matrix().shape[0]
        assert 0 <= idx < n

    def test_empty_state_zero_gain(self):
        state = create_state(4)
        idx, gain = find_best_next(state, _matrix())
        # No placed → all gains are 0
        assert gain == pytest.approx(0.0)

    def test_explicit_candidates_respected(self):
        state = _state_with(4, (0,))
        idx, gain = find_best_next(state, _matrix(), candidates=[2])
        assert idx == 2

    def test_candidates_already_placed_skipped(self):
        state = _state_with(4, (0, 1))
        # Candidate 1 is already placed; only [2] valid
        idx, gain = find_best_next(state, _matrix(), candidates=[1, 2])
        assert idx == 2

    def test_gain_is_float(self):
        state = _state_with(4, (0,))
        _, gain = find_best_next(state, _matrix())
        assert isinstance(gain, float)


# ─── TestGreedyPlace ──────────────────────────────────────────────────────────

class TestGreedyPlace:
    def test_returns_placement_result(self):
        result = greedy_place(4, _matrix())
        assert isinstance(result, PlacementResult)

    def test_n_fragments_lt1_raises(self):
        with pytest.raises(ValueError):
            greedy_place(0, np.zeros((0, 0)))

    def test_root_out_of_range_raises(self):
        with pytest.raises(ValueError):
            greedy_place(4, _matrix(), root=5)

    def test_root_negative_raises(self):
        with pytest.raises(ValueError):
            greedy_place(4, _matrix(), root=-1)

    def test_all_placed(self):
        result = greedy_place(4, _matrix())
        assert result.n_placed == 4

    def test_history_first_is_root(self):
        result = greedy_place(4, _matrix(), root=2)
        assert result.history[0]["idx"] == 2

    def test_score_nonneg(self):
        result = greedy_place(4, _matrix())
        assert result.score >= 0.0

    def test_params_stored(self):
        result = greedy_place(4, _matrix(), root=1)
        assert result.params.get("root") == 1
        assert result.params.get("method") == "greedy"

    def test_single_fragment(self):
        result = greedy_place(1, np.zeros((1, 1)))
        assert result.n_placed == 1

    def test_two_fragments(self):
        result = greedy_place(2, _matrix(2))
        assert result.n_placed == 2

    def test_zero_matrix_all_placed(self):
        result = greedy_place(4, _identity_matrix(4))
        assert result.n_placed == 4

    def test_history_length(self):
        result = greedy_place(4, _matrix())
        assert len(result.history) == 4

    def test_deterministic(self):
        m = _matrix()
        r1 = greedy_place(4, m)
        r2 = greedy_place(4, m)
        assert r1.score == pytest.approx(r2.score)


# ─── TestRemoveWorstPlaced ────────────────────────────────────────────────────

class TestRemoveWorstPlaced:
    def test_returns_placement_result(self):
        base = greedy_place(4, _matrix())
        result = remove_worst_placed(base, _matrix())
        assert isinstance(result, PlacementResult)

    def test_single_placed_unchanged(self):
        base = greedy_place(1, np.zeros((1, 1)))
        result = remove_worst_placed(base, np.zeros((1, 1)))
        assert result.n_placed == base.n_placed

    def test_n_placed_decreases(self):
        base = greedy_place(4, _matrix())
        result = remove_worst_placed(base, _matrix())
        assert result.n_placed == base.n_placed - 1

    def test_root_not_removed(self):
        base = greedy_place(4, _matrix(), root=0)
        result = remove_worst_placed(base, _matrix())
        # Root (0) should still be present
        assert 0 in result.state.placed

    def test_new_score_is_float(self):
        base = greedy_place(4, _matrix())
        result = remove_worst_placed(base, _matrix())
        assert isinstance(result.score, float)

    def test_two_placed_leaves_one(self):
        base = greedy_place(2, _matrix(2))
        result = remove_worst_placed(base, _matrix(2))
        assert result.n_placed == 1


# ─── TestIterativePlace ───────────────────────────────────────────────────────

class TestIterativePlace:
    def test_returns_placement_result(self):
        result = iterative_place(4, _matrix())
        assert isinstance(result, PlacementResult)

    def test_all_placed(self):
        result = iterative_place(4, _matrix())
        assert result.n_placed == 4

    def test_score_nonneg(self):
        result = iterative_place(4, _matrix())
        assert result.score >= 0.0

    def test_max_iter_zero_is_greedy(self):
        m = _matrix()
        greedy = greedy_place(4, m)
        iterative = iterative_place(4, m, max_iter=0)
        assert iterative.score == pytest.approx(greedy.score)

    def test_max_iter_1(self):
        result = iterative_place(4, _matrix(), max_iter=1)
        assert result.n_placed == 4

    def test_different_roots(self):
        m = _matrix()
        for root in range(4):
            result = iterative_place(4, m, root=root)
            assert result.n_placed == 4

    def test_patience_1(self):
        result = iterative_place(4, _matrix(), max_iter=5, patience=1)
        assert result.n_placed == 4

    def test_single_fragment(self):
        result = iterative_place(1, np.zeros((1, 1)))
        assert result.n_placed == 1
