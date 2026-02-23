"""Extra tests for puzzle_reconstruction.assembly.fragment_sequencer."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mat3():
    m = np.array([
        [0.0, 0.9, 0.1],
        [0.9, 0.0, 0.8],
        [0.1, 0.8, 0.0],
    ], dtype=np.float64)
    return m


def _mat4():
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 1] = m[1, 0] = 0.9
    m[1, 2] = m[2, 1] = 0.8
    m[2, 3] = m[3, 2] = 0.7
    m[0, 3] = m[3, 0] = 0.2
    return m


def _seq(order, score=1.0, params=None):
    return SequenceResult(order=order, total_score=score,
                          params=params or {})


# ─── TestSequenceResultExtra ──────────────────────────────────────────────────

class TestSequenceResultExtra:
    def test_params_empty_is_dict(self):
        r = _seq([0, 1, 2])
        assert isinstance(r.params, dict)

    def test_total_score_large_ok(self):
        r = SequenceResult(order=[0, 1], total_score=999.0)
        assert r.total_score == pytest.approx(999.0)

    def test_order_stored_as_list(self):
        r = _seq([2, 0, 1])
        assert r.order == [2, 0, 1]

    def test_len_five(self):
        r = _seq(list(range(5)))
        assert len(r) == 5

    def test_params_overridden(self):
        r = SequenceResult(order=[], total_score=0.0,
                           params={"algo": "test", "k": 3})
        assert r.params["k"] == 3


# ─── TestSequenceGreedyExtra ──────────────────────────────────────────────────

class TestSequenceGreedyExtra:
    def test_result_visits_all(self):
        r = sequence_greedy(_mat4())
        assert sorted(r.order) == [0, 1, 2, 3]

    def test_start_0_first(self):
        r = sequence_greedy(_mat3(), start=0)
        assert r.order[0] == 0

    def test_start_1_first(self):
        r = sequence_greedy(_mat3(), start=1)
        assert r.order[0] == 1

    def test_no_duplicates_4x4(self):
        r = sequence_greedy(_mat4())
        assert len(set(r.order)) == 4

    def test_total_score_nonneg(self):
        r = sequence_greedy(_mat4())
        assert r.total_score >= 0.0

    def test_total_score_matches_compute(self):
        mat = _mat3()
        r = sequence_greedy(mat)
        expected = compute_sequence_score(r.order, mat)
        assert r.total_score == pytest.approx(expected, abs=1e-6)

    def test_2x2_matrix(self):
        mat = np.array([[0.0, 0.7], [0.7, 0.0]])
        r = sequence_greedy(mat)
        assert sorted(r.order) == [0, 1]

    def test_algorithm_param_greedy(self):
        r = sequence_greedy(_mat3())
        assert "algorithm" in r.params


# ─── TestSequenceByScoreExtra ─────────────────────────────────────────────────

class TestSequenceByScoreExtra:
    def test_single_element(self):
        r = sequence_by_score([0.5])
        assert r.order == [0]
        assert r.total_score == pytest.approx(0.5)

    def test_all_zeros_ok(self):
        r = sequence_by_score([0.0, 0.0, 0.0])
        assert sorted(r.order) == [0, 1, 2]

    def test_descending_order_full(self):
        scores = [0.1, 0.9, 0.5, 0.3]
        r = sequence_by_score(scores, descending=True)
        ordered_scores = [scores[i] for i in r.order]
        assert ordered_scores == sorted(ordered_scores, reverse=True)

    def test_ascending_order_full(self):
        scores = [0.7, 0.2, 0.5]
        r = sequence_by_score(scores, descending=False)
        ordered_scores = [scores[i] for i in r.order]
        assert ordered_scores == sorted(ordered_scores)

    def test_total_score_float(self):
        r = sequence_by_score([0.4, 0.6])
        assert isinstance(r.total_score, float)

    def test_params_is_dict(self):
        r = sequence_by_score([0.5, 0.3])
        assert isinstance(r.params, dict)


# ─── TestComputeSequenceScoreExtra ────────────────────────────────────────────

class TestComputeSequenceScoreExtra:
    def test_known_3_path(self):
        mat = _mat3()
        # 0→1: 0.9, 1→2: 0.8
        s = compute_sequence_score([0, 1, 2], mat)
        assert s == pytest.approx(1.7)

    def test_reversed_path_same_score_symmetric(self):
        mat = _mat3()
        s_fwd = compute_sequence_score([0, 1, 2], mat)
        s_rev = compute_sequence_score([2, 1, 0], mat)
        assert s_fwd == pytest.approx(s_rev)

    def test_order_with_repeated_allowed_if_matrix_ok(self):
        # no error expected even for repeated usage of edges (no dedup)
        mat = _mat3()
        s = compute_sequence_score([0, 1, 0], mat)
        assert s == pytest.approx(0.9 + 0.9)

    def test_four_element_chain(self):
        mat = _mat4()
        s = compute_sequence_score([0, 1, 2, 3], mat)
        assert s == pytest.approx(0.9 + 0.8 + 0.7)

    def test_returns_float(self):
        result = compute_sequence_score([0, 1], _mat3())
        assert isinstance(result, float)


# ─── TestReverseSequenceExtra ─────────────────────────────────────────────────

class TestReverseSequenceExtra:
    def test_preserves_params(self):
        r = _seq([0, 1, 2], params={"algo": "greedy"})
        rev = reverse_sequence(r)
        assert rev.params.get("algo") == "greedy"

    def test_four_elements_reversed(self):
        r = _seq([0, 1, 2, 3])
        rev = reverse_sequence(r)
        assert rev.order == [3, 2, 1, 0]

    def test_original_unmodified(self):
        r = _seq([0, 1, 2])
        _ = reverse_sequence(r)
        assert r.order == [0, 1, 2]

    def test_total_score_zero_preserved(self):
        r = SequenceResult(order=[0, 1], total_score=0.0)
        rev = reverse_sequence(r)
        assert rev.total_score == pytest.approx(0.0)


# ─── TestRotateSequenceExtra ──────────────────────────────────────────────────

class TestRotateSequenceExtra:
    def test_rotate_to_last_element(self):
        r = _seq([0, 1, 2, 3])
        rot = rotate_sequence(r, start_idx=3)
        assert rot.order[0] == 3

    def test_rotate_preserves_count(self):
        r = _seq([0, 1, 2, 3, 4])
        rot = rotate_sequence(r, start_idx=2)
        assert len(rot.order) == 5

    def test_original_unmodified(self):
        r = _seq([0, 1, 2, 3])
        _ = rotate_sequence(r, start_idx=2)
        assert r.order == [0, 1, 2, 3]

    def test_rotate_by_negative_raises(self):
        r = _seq([0, 1, 2])
        with pytest.raises(ValueError):
            rotate_sequence(r, start_idx=-1)

    def test_rotate_cycle_identity(self):
        r = _seq([0, 1, 2, 3])
        rot = rotate_sequence(r, start_idx=2)
        # rotating back to fragment value 0 should restore canonical order
        rot2 = rotate_sequence(rot, start_idx=0)
        assert rot2.order == [0, 1, 2, 3]


# ─── TestSequenceToPairsExtra ─────────────────────────────────────────────────

class TestSequenceToPairsExtra:
    def test_pair_types_are_tuples(self):
        r = _seq([0, 1, 2])
        for p in sequence_to_pairs(r):
            assert isinstance(p, tuple)

    def test_four_element_three_pairs(self):
        r = _seq([3, 1, 0, 2])
        pairs = sequence_to_pairs(r)
        assert len(pairs) == 3

    def test_pair_values_correct(self):
        r = _seq([3, 1, 0])
        pairs = sequence_to_pairs(r)
        assert pairs[0] == (3, 1)
        assert pairs[1] == (1, 0)

    def test_no_overlap_between_pairs(self):
        r = _seq(list(range(5)))
        pairs = sequence_to_pairs(r)
        for i in range(len(pairs) - 1):
            assert pairs[i][1] == pairs[i + 1][0]


# ─── TestFindBestStartExtra ───────────────────────────────────────────────────

class TestFindBestStartExtra:
    def test_best_start_produces_higher_total(self):
        mat = _mat4()
        order = [0, 1, 2, 3]
        best = find_best_start(order, mat)
        # Starting from best should give >= starting from worst
        best_r = sequence_greedy(mat, start=best)
        scores = [sequence_greedy(mat, start=i).total_score for i in order]
        assert best_r.total_score >= min(scores) - 1e-6

    def test_two_element_order(self):
        mat = _mat3()
        result = find_best_start([0, 2], mat)
        assert result in [0, 2]

    def test_returns_int_type(self):
        result = find_best_start([0, 1, 2], _mat3())
        assert isinstance(result, int)


# ─── TestBatchSequenceExtra ───────────────────────────────────────────────────

class TestBatchSequenceExtra:
    def test_scores_all_nonneg(self):
        mats = [_mat3(), _mat4()]
        results = batch_sequence(mats)
        for r in results:
            assert r.total_score >= 0.0

    def test_single_matrix(self):
        results = batch_sequence([_mat3()])
        assert len(results) == 1
        assert isinstance(results[0], SequenceResult)

    def test_start_none_uses_default(self):
        results = batch_sequence([_mat3()], start=None)
        assert len(results[0]) == 3

    def test_all_visit_all_nodes(self):
        mats = [_mat3(), _mat4()]
        results = batch_sequence(mats)
        assert sorted(results[0].order) == [0, 1, 2]
        assert sorted(results[1].order) == [0, 1, 2, 3]
