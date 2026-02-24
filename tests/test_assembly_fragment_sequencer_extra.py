"""Extra tests for puzzle_reconstruction/assembly/fragment_sequencer.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _score_matrix(n=4):
    """Create a simple symmetric score matrix."""
    rng = np.random.RandomState(42)
    m = rng.rand(n, n)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


# ─── SequenceResult ─────────────────────────────────────────────────────────

class TestSequenceResultExtra:
    def test_valid(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.5)
        assert len(r) == 3

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            SequenceResult(order=[0], total_score=-1.0)

    def test_repr(self):
        r = SequenceResult(order=[0, 1], total_score=0.5)
        s = repr(r)
        assert "0.5" in s

    def test_empty(self):
        r = SequenceResult(order=[], total_score=0.0)
        assert len(r) == 0


# ─── sequence_greedy ────────────────────────────────────────────────────────

class TestSequenceGreedyExtra:
    def test_empty_matrix(self):
        mat = np.zeros((0, 0))
        r = sequence_greedy(mat)
        assert r.order == []
        assert r.total_score == 0.0

    def test_single(self):
        mat = np.array([[0.0]])
        r = sequence_greedy(mat)
        assert len(r.order) == 1

    def test_visits_all(self):
        mat = _score_matrix(4)
        r = sequence_greedy(mat)
        assert len(r.order) == 4
        assert set(r.order) == {0, 1, 2, 3}

    def test_custom_start(self):
        mat = _score_matrix(4)
        r = sequence_greedy(mat, start=2)
        assert r.order[0] == 2

    def test_bad_start_raises(self):
        mat = _score_matrix(3)
        with pytest.raises(ValueError):
            sequence_greedy(mat, start=5)

    def test_non_square_raises(self):
        mat = np.zeros((2, 3))
        with pytest.raises(ValueError):
            sequence_greedy(mat)

    def test_total_score_nonneg(self):
        mat = _score_matrix(4)
        r = sequence_greedy(mat)
        assert r.total_score >= 0.0


# ─── sequence_by_score ──────────────────────────────────────────────────────

class TestSequenceByScoreExtra:
    def test_descending(self):
        r = sequence_by_score([0.1, 0.5, 0.3], descending=True)
        assert r.order[0] == 1  # highest score

    def test_ascending(self):
        r = sequence_by_score([0.1, 0.5, 0.3], descending=False)
        assert r.order[0] == 0  # lowest score

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            sequence_by_score([0.5, -0.1])

    def test_total_score(self):
        r = sequence_by_score([1.0, 2.0, 3.0])
        assert r.total_score == pytest.approx(6.0)

    def test_empty(self):
        r = sequence_by_score([])
        assert r.order == []


# ─── compute_sequence_score ─────────────────────────────────────────────────

class TestComputeSequenceScoreExtra:
    def test_single_element(self):
        mat = np.array([[0.0]])
        assert compute_sequence_score([0], mat) == 0.0

    def test_two_elements(self):
        mat = np.array([[0.0, 0.8], [0.8, 0.0]])
        score = compute_sequence_score([0, 1], mat)
        assert score == pytest.approx(0.8)

    def test_empty(self):
        mat = np.array([[0.0]])
        assert compute_sequence_score([], mat) == 0.0

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            compute_sequence_score([0], np.zeros((2, 3)))

    def test_out_of_range_raises(self):
        mat = np.array([[0.0, 0.5], [0.5, 0.0]])
        with pytest.raises(ValueError):
            compute_sequence_score([0, 5], mat)


# ─── reverse_sequence ───────────────────────────────────────────────────────

class TestReverseSequenceExtra:
    def test_reverses(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        rev = reverse_sequence(r)
        assert rev.order == [2, 1, 0]

    def test_preserves_score(self):
        r = SequenceResult(order=[0, 1], total_score=0.5)
        rev = reverse_sequence(r)
        assert rev.total_score == pytest.approx(0.5)


# ─── rotate_sequence ────────────────────────────────────────────────────────

class TestRotateSequenceExtra:
    def test_rotate(self):
        r = SequenceResult(order=[0, 1, 2, 3], total_score=1.0)
        rot = rotate_sequence(r, start_idx=2)
        assert rot.order == [2, 3, 0, 1]

    def test_missing_start_raises(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        with pytest.raises(ValueError):
            rotate_sequence(r, start_idx=5)

    def test_noop_rotation(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        rot = rotate_sequence(r, start_idx=0)
        assert rot.order == [0, 1, 2]


# ─── sequence_to_pairs ──────────────────────────────────────────────────────

class TestSequenceToPairsExtra:
    def test_pairs(self):
        r = SequenceResult(order=[0, 1, 2], total_score=1.0)
        pairs = sequence_to_pairs(r)
        assert pairs == [(0, 1), (1, 2)]

    def test_single(self):
        r = SequenceResult(order=[0], total_score=0.0)
        assert sequence_to_pairs(r) == []

    def test_empty(self):
        r = SequenceResult(order=[], total_score=0.0)
        assert sequence_to_pairs(r) == []


# ─── find_best_start ────────────────────────────────────────────────────────

class TestFindBestStartExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            find_best_start([], np.zeros((2, 2)))

    def test_single(self):
        assert find_best_start([0], np.array([[0.0]])) == 0

    def test_returns_valid_index(self):
        mat = _score_matrix(4)
        best = find_best_start([0, 1, 2, 3], mat)
        assert best in [0, 1, 2, 3]


# ─── batch_sequence ─────────────────────────────────────────────────────────

class TestBatchSequenceExtra:
    def test_empty(self):
        assert batch_sequence([]) == []

    def test_length(self):
        results = batch_sequence([_score_matrix(3), _score_matrix(3)])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_sequence([_score_matrix(3)])
        assert isinstance(results[0], SequenceResult)
