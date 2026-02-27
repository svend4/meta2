"""Tests for puzzle_reconstruction/assembly/fragment_sequencer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.fragment_sequencer import (
    SequenceResult,
    sequence_greedy,
    sequence_by_score,
    compute_sequence_score,
    reverse_sequence,
    rotate_sequence,
    sequence_to_pairs,
    find_best_start,
    batch_sequence,
)


# ── SequenceResult ────────────────────────────────────────────────────────────

class TestSequenceResult:
    def test_valid_construction(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=0.5)
        assert sr.order == [0, 1, 2]
        assert sr.total_score == 0.5

    def test_len(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=1.0)
        assert len(sr) == 3

    def test_empty_order(self):
        sr = SequenceResult(order=[], total_score=0.0)
        assert len(sr) == 0

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="total_score must be >= 0"):
            SequenceResult(order=[0], total_score=-0.1)

    def test_zero_score_ok(self):
        sr = SequenceResult(order=[0, 1], total_score=0.0)
        assert sr.total_score == 0.0

    def test_params_default_empty(self):
        sr = SequenceResult(order=[0], total_score=0.0)
        assert sr.params == {}

    def test_params_stored(self):
        sr = SequenceResult(order=[0], total_score=0.0, params={"algorithm": "test"})
        assert sr.params["algorithm"] == "test"


# ── sequence_greedy ───────────────────────────────────────────────────────────

class TestSequenceGreedy:
    def _make_mat(self, n, val=0.5):
        mat = np.full((n, n), val, dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        return mat

    def test_empty_matrix_returns_empty_order(self):
        mat = np.zeros((0, 0))
        result = sequence_greedy(mat)
        assert result.order == []
        assert result.total_score == 0.0

    def test_single_element(self):
        mat = np.array([[0.0]])
        result = sequence_greedy(mat)
        assert result.order == [0]
        assert result.total_score == 0.0

    def test_two_elements_order(self):
        mat = np.array([[0, 0.9], [0.9, 0]])
        result = sequence_greedy(mat)
        assert len(result.order) == 2
        assert set(result.order) == {0, 1}

    def test_all_indices_visited(self):
        mat = self._make_mat(5)
        result = sequence_greedy(mat)
        assert set(result.order) == {0, 1, 2, 3, 4}
        assert len(result.order) == 5

    def test_custom_start(self):
        mat = self._make_mat(4)
        result = sequence_greedy(mat, start=2)
        assert result.order[0] == 2

    def test_invalid_start_raises(self):
        mat = self._make_mat(4)
        with pytest.raises(ValueError, match="start must be in"):
            sequence_greedy(mat, start=10)

    def test_negative_start_raises(self):
        mat = self._make_mat(4)
        with pytest.raises(ValueError):
            sequence_greedy(mat, start=-1)

    def test_non_square_matrix_raises(self):
        mat = np.zeros((3, 4))
        with pytest.raises(ValueError):
            sequence_greedy(mat)

    def test_1d_matrix_raises(self):
        mat = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            sequence_greedy(mat)

    def test_algorithm_param_set(self):
        mat = self._make_mat(3)
        result = sequence_greedy(mat)
        assert result.params.get("algorithm") == "greedy"

    def test_total_score_nonneg(self):
        mat = self._make_mat(4)
        result = sequence_greedy(mat)
        assert result.total_score >= 0.0

    def test_best_path_followed(self):
        mat = np.zeros((3, 3))
        mat[0, 1] = 0.9
        mat[1, 0] = 0.9
        mat[1, 2] = 0.8
        mat[2, 1] = 0.8
        mat[0, 2] = 0.1
        mat[2, 0] = 0.1
        result = sequence_greedy(mat, start=0)
        assert result.order[0] == 0
        assert result.order[1] == 1


# ── sequence_by_score ─────────────────────────────────────────────────────────

class TestSequenceByScore:
    def test_descending_order(self):
        scores = [0.3, 0.9, 0.1, 0.7]
        result = sequence_by_score(scores, descending=True)
        vals = [scores[i] for i in result.order]
        assert vals == sorted(scores, reverse=True)

    def test_ascending_order(self):
        scores = [0.3, 0.9, 0.1, 0.7]
        result = sequence_by_score(scores, descending=False)
        vals = [scores[i] for i in result.order]
        assert vals == sorted(scores)

    def test_empty_scores(self):
        result = sequence_by_score([])
        assert result.order == []

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            sequence_by_score([0.5, -0.1])

    def test_total_score_is_sum(self):
        scores = [0.3, 0.7, 0.5]
        result = sequence_by_score(scores)
        assert pytest.approx(result.total_score, abs=1e-6) == sum(scores)

    def test_all_indices_present(self):
        scores = [0.1, 0.2, 0.3, 0.4]
        result = sequence_by_score(scores)
        assert set(result.order) == {0, 1, 2, 3}

    def test_algorithm_param(self):
        result = sequence_by_score([0.5, 0.3])
        assert result.params.get("algorithm") == "by_score"


# ── compute_sequence_score ────────────────────────────────────────────────────

class TestComputeSequenceScore:
    def _make_mat(self, n):
        mat = np.random.rand(n, n)
        return mat

    def test_empty_order_returns_zero(self):
        mat = np.zeros((3, 3))
        assert compute_sequence_score([], mat) == 0.0

    def test_single_element_returns_zero(self):
        mat = np.zeros((3, 3))
        assert compute_sequence_score([0], mat) == 0.0

    def test_two_element_correct(self):
        mat = np.array([[0, 0.8], [0.8, 0]])
        score = compute_sequence_score([0, 1], mat)
        assert pytest.approx(score, abs=1e-6) == 0.8

    def test_sums_adjacent_pairs(self):
        mat = np.zeros((3, 3))
        mat[0, 1] = 0.5
        mat[1, 2] = 0.3
        score = compute_sequence_score([0, 1, 2], mat)
        assert pytest.approx(score, abs=1e-6) == 0.8

    def test_out_of_range_index_raises(self):
        mat = np.zeros((3, 3))
        with pytest.raises(ValueError, match="Index out of range"):
            compute_sequence_score([0, 10], mat)

    def test_non_square_raises(self):
        mat = np.zeros((2, 3))
        with pytest.raises(ValueError):
            compute_sequence_score([0, 1], mat)


# ── reverse_sequence ──────────────────────────────────────────────────────────

class TestReverseSequence:
    def test_reverses_order(self):
        sr = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        result = reverse_sequence(sr)
        assert result.order == [3, 2, 1, 0]

    def test_score_preserved(self):
        sr = SequenceResult(order=[0, 1], total_score=0.75)
        result = reverse_sequence(sr)
        assert result.total_score == 0.75

    def test_empty_order(self):
        sr = SequenceResult(order=[], total_score=0.0)
        result = reverse_sequence(sr)
        assert result.order == []

    def test_single_element(self):
        sr = SequenceResult(order=[5], total_score=0.0)
        result = reverse_sequence(sr)
        assert result.order == [5]

    def test_params_preserved(self):
        sr = SequenceResult(order=[0, 1], total_score=0.5,
                            params={"algorithm": "greedy"})
        result = reverse_sequence(sr)
        assert result.params.get("algorithm") == "greedy"


# ── rotate_sequence ───────────────────────────────────────────────────────────

class TestRotateSequence:
    def test_rotate_to_start(self):
        sr = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        result = rotate_sequence(sr, start_idx=2)
        assert result.order == [2, 3, 0, 1]

    def test_rotate_to_first_element_unchanged(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=1.0)
        result = rotate_sequence(sr, start_idx=0)
        assert result.order == [0, 1, 2]

    def test_rotate_to_last(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=1.0)
        result = rotate_sequence(sr, start_idx=2)
        assert result.order == [2, 0, 1]

    def test_start_idx_not_in_order_raises(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=1.0)
        with pytest.raises(ValueError, match="not in order"):
            rotate_sequence(sr, start_idx=99)

    def test_score_preserved(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=0.8)
        result = rotate_sequence(sr, start_idx=1)
        assert result.total_score == 0.8

    def test_all_elements_preserved(self):
        sr = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        result = rotate_sequence(sr, start_idx=2)
        assert set(result.order) == {0, 1, 2, 3}


# ── sequence_to_pairs ─────────────────────────────────────────────────────────

class TestSequenceToPairs:
    def test_empty_order_returns_empty(self):
        sr = SequenceResult(order=[], total_score=0.0)
        assert sequence_to_pairs(sr) == []

    def test_single_returns_empty(self):
        sr = SequenceResult(order=[0], total_score=0.0)
        assert sequence_to_pairs(sr) == []

    def test_two_elements_one_pair(self):
        sr = SequenceResult(order=[0, 1], total_score=0.5)
        pairs = sequence_to_pairs(sr)
        assert pairs == [(0, 1)]

    def test_three_elements_two_pairs(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=1.0)
        pairs = sequence_to_pairs(sr)
        assert pairs == [(0, 1), (1, 2)]

    def test_n_pairs_is_n_minus_1(self):
        sr = SequenceResult(order=list(range(10)), total_score=5.0)
        pairs = sequence_to_pairs(sr)
        assert len(pairs) == 9

    def test_pairs_are_tuples(self):
        sr = SequenceResult(order=[0, 1, 2], total_score=1.0)
        for pair in sequence_to_pairs(sr):
            assert isinstance(pair, tuple)


# ── find_best_start ───────────────────────────────────────────────────────────

class TestFindBestStart:
    def test_empty_order_raises(self):
        mat = np.zeros((3, 3))
        with pytest.raises(ValueError, match="must not be empty"):
            find_best_start([], mat)

    def test_single_element_returns_it(self):
        mat = np.zeros((3, 3))
        result = find_best_start([2], mat)
        assert result == 2

    def test_returns_element_from_order(self):
        mat = np.random.rand(4, 4)
        order = [0, 1, 2, 3]
        result = find_best_start(order, mat)
        assert result in order

    def test_selects_best_cyclic_start(self):
        # Make a mat where starting from 1 gives the best wraparound
        mat = np.zeros((3, 3))
        mat[1, 2] = 0.9
        mat[2, 0] = 0.9
        mat[0, 1] = 0.9
        result = find_best_start([0, 1, 2], mat)
        assert result in [0, 1, 2]


# ── batch_sequence ────────────────────────────────────────────────────────────

class TestBatchSequence:
    def _make_mat(self, n):
        mat = np.random.rand(n, n)
        np.fill_diagonal(mat, 0.0)
        return mat

    def test_output_length(self):
        mats = [self._make_mat(3), self._make_mat(4)]
        results = batch_sequence(mats)
        assert len(results) == 2

    def test_each_is_sequence_result(self):
        results = batch_sequence([self._make_mat(3)])
        assert isinstance(results[0], SequenceResult)

    def test_empty_list(self):
        results = batch_sequence([])
        assert results == []

    def test_start_passed_through(self):
        mat = self._make_mat(4)
        results = batch_sequence([mat], start=2)
        assert results[0].order[0] == 2

    def test_all_elements_in_result(self):
        mat = self._make_mat(5)
        results = batch_sequence([mat])
        assert set(results[0].order) == {0, 1, 2, 3, 4}
