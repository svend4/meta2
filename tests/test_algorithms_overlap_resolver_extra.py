"""Additional tests for puzzle_reconstruction/algorithms/overlap_resolver.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.overlap_resolver import (
    OverlapConflict,
    compute_separation_vector,
    detect_overlap_conflicts,
    resolve_single_conflict,
    resolve_all_conflicts,
    conflict_score,
)
from puzzle_reconstruction.assembly.assembly_state import AssemblyState


# ─── helpers ──────────────────────────────────────────────────────────────────

def _empty_state(n: int = 4) -> AssemblyState:
    return AssemblyState(n_fragments=n)


def _rect(x: float, y: float, w: float = 20.0, h: float = 20.0) -> np.ndarray:
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float
    )


# ─── TestOverlapConflictExtra ─────────────────────────────────────────────────

class TestOverlapConflictExtra:
    def test_fields_accessible(self):
        oc = OverlapConflict(idx1=2, idx2=5, iou=0.4)
        assert oc.idx1 == 2
        assert oc.idx2 == 5

    def test_iou_zero_valid(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.0)
        assert oc.iou == pytest.approx(0.0)

    def test_iou_one_valid(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=1.0)
        assert oc.iou == pytest.approx(1.0)

    def test_shift_vector_default_is_zero_pair(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.5)
        dx, dy = oc.shift_vector
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_shift_vector_stored_correctly(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.5, shift_vector=(-2.5, 7.0))
        assert oc.shift_vector[0] == pytest.approx(-2.5)
        assert oc.shift_vector[1] == pytest.approx(7.0)

    def test_large_idx_values(self):
        oc = OverlapConflict(idx1=999, idx2=1000, iou=0.1)
        assert oc.idx1 == 999
        assert oc.idx2 == 1000

    def test_multiple_instances_independent(self):
        oc1 = OverlapConflict(idx1=0, idx2=1, iou=0.3)
        oc2 = OverlapConflict(idx1=2, idx2=3, iou=0.7)
        assert oc1.iou != oc2.iou


# ─── TestComputeSeparationVectorExtra ─────────────────────────────────────────

class TestComputeSeparationVectorExtra:
    def test_same_contour_diagonal_fallback(self):
        c = _rect(0, 0)
        dx, dy = compute_separation_vector(c, c)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(1.0)

    def test_far_apart_right_direction(self):
        c1 = _rect(0, 0)
        c2 = _rect(500, 0)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx > 0.0

    def test_far_apart_up_direction(self):
        c1 = _rect(0, 0)
        c2 = _rect(0, 500)
        dx, dy = compute_separation_vector(c1, c2)
        assert dy > 0.0

    def test_both_empty_returns_zero(self):
        empty = np.empty((0, 2), dtype=float)
        dx, dy = compute_separation_vector(empty, empty)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_result_type_is_tuple(self):
        result = compute_separation_vector(_rect(0, 0), _rect(50, 50))
        assert isinstance(result, tuple)

    def test_result_length_is_2(self):
        result = compute_separation_vector(_rect(0, 0), _rect(50, 50))
        assert len(result) == 2

    def test_antisymmetry(self):
        """swap c1 and c2 → dx, dy flip sign."""
        c1 = _rect(0, 0, 10, 10)
        c2 = _rect(100, 0, 10, 10)
        dx1, dy1 = compute_separation_vector(c1, c2)
        dx2, dy2 = compute_separation_vector(c2, c1)
        assert dx1 * dx2 < 0

    def test_diagonal_offset(self):
        c1 = _rect(0, 0)
        c2 = _rect(100, 100)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx > 0.0
        assert dy > 0.0


# ─── TestDetectOverlapConflictsExtra ──────────────────────────────────────────

class TestDetectOverlapConflictsExtra:
    def test_empty_state_1_fragment(self):
        state = _empty_state(1)
        result = detect_overlap_conflicts(state, [])
        assert result == []

    def test_empty_state_large_n(self):
        state = _empty_state(20)
        result = detect_overlap_conflicts(state, [])
        assert result == []

    def test_result_is_list_type(self):
        state = _empty_state(3)
        result = detect_overlap_conflicts(state, [])
        assert type(result) is list

    def test_custom_threshold_no_crash(self):
        state = _empty_state(3)
        result = detect_overlap_conflicts(state, [], threshold=0.0)
        assert isinstance(result, list)

    def test_high_threshold_no_crash(self):
        state = _empty_state(3)
        result = detect_overlap_conflicts(state, [], threshold=0.99)
        assert isinstance(result, list)


# ─── TestResolveAllConflictsExtra ─────────────────────────────────────────────

class TestResolveAllConflictsExtra:
    def test_returns_assembly_state(self):
        state = _empty_state()
        result = resolve_all_conflicts(state, [])
        assert isinstance(result, AssemblyState)

    def test_n_fragments_preserved(self):
        state = _empty_state(7)
        result = resolve_all_conflicts(state, [])
        assert result.n_fragments == 7

    def test_max_iter_0_no_crash(self):
        state = _empty_state()
        result = resolve_all_conflicts(state, [], max_iter=0)
        assert isinstance(result, AssemblyState)

    def test_max_iter_20_no_crash(self):
        state = _empty_state()
        result = resolve_all_conflicts(state, [], max_iter=20)
        assert isinstance(result, AssemblyState)

    def test_empty_placed_unchanged(self):
        state = _empty_state(5)
        result = resolve_all_conflicts(state, [], max_iter=3)
        assert result.placed == {}

    def test_custom_threshold(self):
        state = _empty_state()
        result = resolve_all_conflicts(state, [], threshold=0.01)
        assert isinstance(result, AssemblyState)


# ─── TestConflictScoreExtra ───────────────────────────────────────────────────

class TestConflictScoreExtra:
    def test_empty_state_n1(self):
        score = conflict_score(_empty_state(1), [])
        assert score == pytest.approx(0.0)

    def test_empty_state_large_n(self):
        score = conflict_score(_empty_state(50), [])
        assert score >= 0.0

    def test_returns_float_type(self):
        result = conflict_score(_empty_state(), [])
        assert isinstance(result, float)

    def test_custom_threshold_0(self):
        result = conflict_score(_empty_state(), [], threshold=0.0)
        assert result >= 0.0

    def test_custom_threshold_1(self):
        result = conflict_score(_empty_state(), [], threshold=1.0)
        assert result >= 0.0

    def test_is_finite(self):
        result = conflict_score(_empty_state(), [])
        assert np.isfinite(result)
