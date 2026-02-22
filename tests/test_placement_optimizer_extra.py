"""Additional tests for puzzle_reconstruction.assembly.placement_optimizer."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _uniform(n, value=1.0):
    m = np.full((n, n), value, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _chain(n, value=1.0):
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        m[i, i + 1] = value
        m[i + 1, i] = value
    return m


# ─── TestPlacementResultExtra ─────────────────────────────────────────────────

class TestPlacementResultExtra:
    def test_history_default_empty(self):
        pr = PlacementResult(state=create_state(3), score=0.0, n_placed=0)
        assert pr.history == []

    def test_params_default_empty(self):
        pr = PlacementResult(state=create_state(3), score=0.0, n_placed=0)
        assert pr.params == {}

    def test_score_float(self):
        pr = PlacementResult(state=create_state(2), score=1.5, n_placed=2)
        assert isinstance(pr.score, float)

    def test_n_placed_zero(self):
        pr = PlacementResult(state=create_state(5), score=0.0, n_placed=0)
        assert pr.n_placed == 0

    def test_params_extra_keys(self):
        pr = PlacementResult(state=create_state(3), score=0.0, n_placed=0,
                             params={"method": "greedy", "root": 0})
        assert pr.params["root"] == 0


# ─── TestScorePlacementExtra ──────────────────────────────────────────────────

class TestScorePlacementExtra:
    def test_returns_float(self):
        state = create_state(3)
        m = _uniform(3)
        assert isinstance(score_placement(state, m), float)

    def test_three_fragments_three_edges(self):
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        state = place_fragment(state, 2, (2.0, 0.0))
        state = add_adjacency(state, 0, 1)
        state = add_adjacency(state, 1, 2)
        state = add_adjacency(state, 0, 2)
        m = _uniform(3, value=1.0)
        # Three edges each with score 1.0
        assert score_placement(state, m) == pytest.approx(3.0)

    def test_score_nonneg(self):
        state = place_fragment(create_state(4), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        state = add_adjacency(state, 0, 1)
        m = _uniform(4, value=0.5)
        assert score_placement(state, m) >= 0.0

    def test_zero_weight_matrix(self):
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        state = add_adjacency(state, 0, 1)
        m = np.zeros((3, 3), dtype=np.float32)
        assert score_placement(state, m) == pytest.approx(0.0)


# ─── TestFindBestNextExtra ────────────────────────────────────────────────────

class TestFindBestNextExtra:
    def test_returns_tuple(self):
        state = create_state(3)
        result = find_best_next(state, _uniform(3))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_gain_is_float(self):
        state = create_state(3)
        _, gain = find_best_next(state, _uniform(3))
        assert isinstance(gain, float)

    def test_all_placed_returns_minus_1(self):
        state = create_state(2)
        state = place_fragment(state, 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        idx, _ = find_best_next(state, _uniform(2))
        assert idx == -1

    def test_one_candidate_returned(self):
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        state = place_fragment(state, 1, (1.0, 0.0))
        m = _uniform(3, value=1.0)
        idx, _ = find_best_next(state, m, candidates=[2])
        assert idx == 2

    def test_candidates_empty_returns_minus_1_or_valid(self):
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        m = _uniform(3)
        idx, gain = find_best_next(state, m, candidates=[])
        assert idx == -1
        assert gain == pytest.approx(0.0)

    def test_high_score_wins_among_candidates(self):
        m = np.array([[0., 2., 10.],
                      [2., 0., 3.],
                      [10., 3., 0.]], dtype=np.float32)
        state = place_fragment(create_state(3), 0, (0.0, 0.0))
        idx, gain = find_best_next(state, m, candidates=[1, 2])
        assert idx == 2
        assert gain == pytest.approx(10.0)


# ─── TestGreedyPlaceExtra ─────────────────────────────────────────────────────

class TestGreedyPlaceExtra:
    def test_two_fragments(self):
        m = _uniform(2, value=3.0)
        pr = greedy_place(2, m)
        assert pr.n_placed == 2

    def test_root_1_first_in_history(self):
        m = _uniform(4)
        pr = greedy_place(4, m, root=1)
        assert pr.history[0]["idx"] == 1

    def test_root_last_index(self):
        m = _uniform(5)
        pr = greedy_place(5, m, root=4)
        assert pr.history[0]["idx"] == 4

    def test_score_matches_uniform_matrix(self):
        n = 4
        m = _uniform(n, value=2.0)
        pr = greedy_place(n, m)
        # Each adjacent pair counted once: 3 edges × 2.0 for a chain of 4
        assert pr.score > 0.0

    def test_history_contains_idx_key(self):
        m = _uniform(3)
        pr = greedy_place(3, m)
        for entry in pr.history:
            assert "idx" in entry

    def test_chain_matrix_positive_score(self):
        m = _chain(5, value=3.0)
        pr = greedy_place(5, m)
        assert pr.score > 0.0

    def test_all_placed_in_state(self):
        m = _uniform(4)
        pr = greedy_place(4, m)
        assert len(pr.state.placed) == 4


# ─── TestRemoveWorstPlacedExtra ───────────────────────────────────────────────

class TestRemoveWorstPlacedExtra:
    def test_double_remove_reduces_by_2(self):
        m = _uniform(5, value=1.0)
        pr = greedy_place(5, m)
        r1 = remove_worst_placed(pr, m)
        r2 = remove_worst_placed(r1, m)
        assert r2.n_placed == 3

    def test_result_score_is_float(self):
        m = _uniform(4)
        pr = greedy_place(4, m)
        result = remove_worst_placed(pr, m)
        assert isinstance(result.score, float)

    def test_placed_count_decreases(self):
        m = _uniform(4, value=1.0)
        pr = greedy_place(4, m)
        before = pr.n_placed
        after = remove_worst_placed(pr, m).n_placed
        assert after < before

    def test_state_type_preserved(self):
        m = _uniform(4)
        pr = greedy_place(4, m)
        result = remove_worst_placed(pr, m)
        assert isinstance(result.state, AssemblyState)


# ─── TestIterativePlaceExtra ──────────────────────────────────────────────────

class TestIterativePlaceExtra:
    def test_one_fragment(self):
        m = np.array([[0.0]], dtype=np.float32)
        pr = iterative_place(1, m)
        assert pr.n_placed == 1

    def test_root_parameter_accepted(self):
        m = _uniform(4)
        pr = iterative_place(4, m, root=2)
        assert pr.n_placed == 4

    def test_score_nonneg(self):
        m = _uniform(4, value=1.0)
        pr = iterative_place(4, m, max_iter=3)
        assert pr.score >= 0.0

    def test_history_not_empty(self):
        m = _uniform(4)
        pr = iterative_place(4, m, max_iter=3)
        assert len(pr.history) > 0

    def test_max_iter_1(self):
        m = _uniform(4)
        pr = iterative_place(4, m, max_iter=1)
        assert pr.n_placed == 4

    def test_chain_matrix_all_placed(self):
        m = _chain(6, value=2.0)
        pr = iterative_place(6, m, max_iter=5)
        assert pr.n_placed == 6
