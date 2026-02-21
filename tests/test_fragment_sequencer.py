"""Tests for puzzle_reconstruction.assembly.fragment_sequencer."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.fragment_sequencer import (
    SequenceResult,
    batch_sequence,
    compute_sequence_score,
    find_best_start,
    reverse_sequence,
    rotate_sequence,
    sequence_by_score,
    sequence_greedy,
    sequence_to_pairs,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _score_matrix_3x3() -> np.ndarray:
    """3×3 score matrix with clear best path 0→1→2."""
    mat = np.array([
        [0.0, 0.9, 0.1],
        [0.9, 0.0, 0.8],
        [0.1, 0.8, 0.0],
    ], dtype=np.float64)
    return mat


def _score_matrix_4x4() -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[0, 1] = mat[1, 0] = 0.9
    mat[1, 2] = mat[2, 1] = 0.8
    mat[2, 3] = mat[3, 2] = 0.7
    mat[0, 3] = mat[3, 0] = 0.2
    return mat


# ─── SequenceResult ──────────────────────────────────────────────────────────

class TestSequenceResult:
    def test_fields_stored(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.7)
        assert r.order == [0, 1, 2]
        assert r.total_score == pytest.approx(1.7)

    def test_default_params_empty(self):
        r = SequenceResult(order=[], total_score=0.0)
        assert r.params == {}

    def test_len(self):
        r = SequenceResult(order=[0, 2, 1], total_score=0.5)
        assert len(r) == 3

    def test_len_zero(self):
        r = SequenceResult(order=[], total_score=0.0)
        assert len(r) == 0

    def test_negative_total_score_raises(self):
        with pytest.raises(ValueError):
            SequenceResult(order=[], total_score=-0.1)

    def test_zero_total_score_allowed(self):
        r = SequenceResult(order=[], total_score=0.0)
        assert r.total_score == pytest.approx(0.0)

    def test_params_stored(self):
        r = SequenceResult(order=[1], total_score=0.0, params={"algorithm": "greedy"})
        assert r.params["algorithm"] == "greedy"


# ─── sequence_greedy ─────────────────────────────────────────────────────────

class TestSequenceGreedy:
    def test_returns_sequence_result(self):
        r = sequence_greedy(_score_matrix_3x3())
        assert isinstance(r, SequenceResult)

    def test_length_n(self):
        r = sequence_greedy(_score_matrix_3x3())
        assert len(r) == 3

    def test_all_indices_visited(self):
        r = sequence_greedy(_score_matrix_4x4())
        assert sorted(r.order) == [0, 1, 2, 3]

    def test_no_repeated_indices(self):
        r = sequence_greedy(_score_matrix_4x4())
        assert len(set(r.order)) == len(r.order)

    def test_empty_matrix_returns_empty(self):
        r = sequence_greedy(np.zeros((0, 0)))
        assert r.order == []
        assert r.total_score == pytest.approx(0.0)

    def test_single_element(self):
        r = sequence_greedy(np.zeros((1, 1)))
        assert r.order == [0]

    def test_start_parameter_respected(self):
        r = sequence_greedy(_score_matrix_3x3(), start=2)
        assert r.order[0] == 2

    def test_invalid_start_raises(self):
        with pytest.raises(ValueError):
            sequence_greedy(_score_matrix_3x3(), start=5)
        with pytest.raises(ValueError):
            sequence_greedy(_score_matrix_3x3(), start=-1)

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError):
            sequence_greedy(np.zeros((3, 4)))

    def test_total_score_positive(self):
        r = sequence_greedy(_score_matrix_3x3())
        assert r.total_score > 0.0

    def test_algorithm_param_stored(self):
        r = sequence_greedy(_score_matrix_3x3())
        assert r.params.get("algorithm") == "greedy"


# ─── sequence_by_score ───────────────────────────────────────────────────────

class TestSequenceByScore:
    def test_returns_sequence_result(self):
        r = sequence_by_score([0.5, 0.8, 0.3])
        assert isinstance(r, SequenceResult)

    def test_length_matches(self):
        r = sequence_by_score([0.5, 0.8, 0.3])
        assert len(r) == 3

    def test_descending_order(self):
        scores = [0.3, 0.9, 0.1, 0.7]
        r = sequence_by_score(scores, descending=True)
        # First element should be index of max score
        assert scores[r.order[0]] == max(scores)

    def test_ascending_order(self):
        scores = [0.3, 0.9, 0.1, 0.7]
        r = sequence_by_score(scores, descending=False)
        assert scores[r.order[0]] == min(scores)

    def test_total_score_is_sum(self):
        scores = [0.3, 0.5, 0.2]
        r = sequence_by_score(scores)
        assert r.total_score == pytest.approx(sum(scores))

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            sequence_by_score([0.5, -0.1, 0.3])

    def test_empty_list_ok(self):
        r = sequence_by_score([])
        assert r.order == []

    def test_all_indices_present(self):
        r = sequence_by_score([0.1, 0.5, 0.3, 0.9])
        assert sorted(r.order) == [0, 1, 2, 3]


# ─── compute_sequence_score ──────────────────────────────────────────────────

class TestComputeSequenceScore:
    def test_empty_order_zero(self):
        mat = _score_matrix_3x3()
        assert compute_sequence_score([], mat) == pytest.approx(0.0)

    def test_single_element_zero(self):
        mat = _score_matrix_3x3()
        assert compute_sequence_score([0], mat) == pytest.approx(0.0)

    def test_two_elements_correct(self):
        mat = _score_matrix_3x3()
        score = compute_sequence_score([0, 1], mat)
        assert score == pytest.approx(0.9)

    def test_three_elements_correct(self):
        mat = _score_matrix_3x3()
        score = compute_sequence_score([0, 1, 2], mat)
        assert score == pytest.approx(0.9 + 0.8)

    def test_out_of_range_index_raises(self):
        mat = _score_matrix_3x3()
        with pytest.raises(ValueError):
            compute_sequence_score([0, 5], mat)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_sequence_score([0, 1], np.zeros((3, 4)))


# ─── reverse_sequence ────────────────────────────────────────────────────────

class TestReverseSequence:
    def test_reversed_order(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        rev = reverse_sequence(r)
        assert rev.order == [2, 1, 0]

    def test_same_total_score(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.7)
        rev = reverse_sequence(r)
        assert rev.total_score == pytest.approx(1.7)

    def test_empty_order(self):
        r = SequenceResult(order=[], total_score=0.0)
        rev = reverse_sequence(r)
        assert rev.order == []

    def test_single_element_unchanged(self):
        r = SequenceResult(order=[3], total_score=0.0)
        rev = reverse_sequence(r)
        assert rev.order == [3]

    def test_double_reverse_identity(self):
        r = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        assert reverse_sequence(reverse_sequence(r)).order == r.order

    def test_returns_new_object(self):
        r = SequenceResult(order=[0, 1], total_score=0.5)
        rev = reverse_sequence(r)
        assert rev is not r


# ─── rotate_sequence ─────────────────────────────────────────────────────────

class TestRotateSequence:
    def test_rotate_to_middle(self):
        r = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        rot = rotate_sequence(r, start_idx=2)
        assert rot.order == [2, 3, 0, 1]

    def test_rotate_to_first_unchanged(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        rot = rotate_sequence(r, start_idx=0)
        assert rot.order == [0, 1, 2]

    def test_rotate_preserves_total_score(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.7)
        rot = rotate_sequence(r, start_idx=1)
        assert rot.total_score == pytest.approx(1.7)

    def test_invalid_start_raises(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        with pytest.raises(ValueError):
            rotate_sequence(r, start_idx=5)

    def test_all_elements_present(self):
        r = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        rot = rotate_sequence(r, start_idx=3)
        assert sorted(rot.order) == [0, 1, 2, 3]

    def test_returns_new_object(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        rot = rotate_sequence(r, start_idx=1)
        assert rot is not r


# ─── sequence_to_pairs ───────────────────────────────────────────────────────

class TestSequenceToPairs:
    def test_empty_order(self):
        r = SequenceResult(order=[], total_score=0.0)
        assert sequence_to_pairs(r) == []

    def test_single_element(self):
        r = SequenceResult(order=[0], total_score=0.0)
        assert sequence_to_pairs(r) == []

    def test_two_elements(self):
        r = SequenceResult(order=[0, 1], total_score=0.9)
        assert sequence_to_pairs(r) == [(0, 1)]

    def test_three_elements(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.7)
        assert sequence_to_pairs(r) == [(0, 1), (1, 2)]

    def test_n_minus_one_pairs(self):
        r = SequenceResult(order=list(range(5)), total_score=0.0)
        pairs = sequence_to_pairs(r)
        assert len(pairs) == 4

    def test_pairs_are_consecutive(self):
        r = SequenceResult(order=[2, 0, 3, 1], total_score=0.0)
        pairs = sequence_to_pairs(r)
        for i, (a, b) in enumerate(pairs):
            assert a == r.order[i]
            assert b == r.order[i + 1]


# ─── find_best_start ─────────────────────────────────────────────────────────

class TestFindBestStart:
    def test_returns_int(self):
        mat = _score_matrix_3x3()
        result = find_best_start([0, 1, 2], mat)
        assert isinstance(result, int)

    def test_result_in_order(self):
        mat = _score_matrix_3x3()
        order = [0, 1, 2]
        result = find_best_start(order, mat)
        assert result in order

    def test_empty_order_raises(self):
        with pytest.raises(ValueError):
            find_best_start([], _score_matrix_3x3())

    def test_single_element_returns_it(self):
        mat = _score_matrix_3x3()
        result = find_best_start([2], mat)
        assert result == 2

    def test_symmetric_matrix_consistent(self):
        mat = _score_matrix_4x4()
        order = [0, 1, 2, 3]
        result = find_best_start(order, mat)
        assert result in order


# ─── batch_sequence ──────────────────────────────────────────────────────────

class TestBatchSequence:
    def test_returns_list(self):
        mats = [_score_matrix_3x3(), _score_matrix_4x4()]
        result = batch_sequence(mats)
        assert isinstance(result, list)

    def test_length_matches_input(self):
        mats = [_score_matrix_3x3(), _score_matrix_3x3(), _score_matrix_4x4()]
        result = batch_sequence(mats)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert batch_sequence([]) == []

    def test_all_sequence_results(self):
        mats = [_score_matrix_3x3(), _score_matrix_4x4()]
        result = batch_sequence(mats)
        assert all(isinstance(r, SequenceResult) for r in result)

    def test_start_forwarded(self):
        mats = [_score_matrix_3x3()]
        result = batch_sequence(mats, start=1)
        assert result[0].order[0] == 1

    def test_each_result_correct_length(self):
        mats = [_score_matrix_3x3(), _score_matrix_4x4()]
        result = batch_sequence(mats)
        assert len(result[0]) == 3
        assert len(result[1]) == 4
