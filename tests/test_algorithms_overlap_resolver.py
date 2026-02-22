"""Tests for puzzle_reconstruction/algorithms/overlap_resolver.py

Note: detect_overlap_conflicts / resolve_single_conflict / resolve_all_conflicts
access PlacedFragment.fragment_idx, which differs from assembly_state.PlacedFragment.idx.
Tests are scoped to safe paths (empty state) and standalone pure functions.
"""
import pytest
import numpy as np
from types import SimpleNamespace

from puzzle_reconstruction.algorithms.overlap_resolver import (
    OverlapConflict,
    compute_separation_vector,
    detect_overlap_conflicts,
    resolve_single_conflict,
    resolve_all_conflicts,
    conflict_score,
)
from puzzle_reconstruction.assembly.assembly_state import AssemblyState


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_empty_state(n=3):
    """AssemblyState with no placed fragments."""
    return AssemblyState(n_fragments=n)


def make_rect(x, y, w=20, h=20):
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=float,
    )


# ─── OverlapConflict ──────────────────────────────────────────────────────────

class TestOverlapConflict:
    def test_basic_creation(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.3)
        assert oc.idx1 == 0
        assert oc.idx2 == 1
        assert oc.iou == pytest.approx(0.3)

    def test_default_shift_vector(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.5)
        assert oc.shift_vector == (0.0, 0.0)

    def test_custom_shift_vector(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.5, shift_vector=(3.5, -1.2))
        assert oc.shift_vector[0] == pytest.approx(3.5)
        assert oc.shift_vector[1] == pytest.approx(-1.2)

    def test_repr_contains_indices(self):
        oc = OverlapConflict(idx1=4, idx2=7, iou=0.2)
        r = repr(oc)
        assert "4" in r
        assert "7" in r

    def test_iou_stored(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.75)
        assert oc.iou == pytest.approx(0.75)


# ─── compute_separation_vector ────────────────────────────────────────────────

class TestComputeSeparationVector:
    def test_empty_contours_returns_zero(self):
        c1 = np.empty((0, 2), dtype=float)
        c2 = np.empty((0, 2), dtype=float)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_first_empty_returns_zero(self):
        c1 = np.empty((0, 2), dtype=float)
        c2 = make_rect(0, 0)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_second_empty_returns_zero(self):
        c1 = make_rect(0, 0)
        c2 = np.empty((0, 2), dtype=float)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_coincident_centroids_returns_diagonal(self):
        c1 = make_rect(0, 0)
        c2 = make_rect(0, 0)  # Same centroid
        dx, dy = compute_separation_vector(c1, c2)
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(1.0)

    def test_returns_tuple(self):
        c1 = make_rect(0, 0)
        c2 = make_rect(100, 0)
        result = compute_separation_vector(c1, c2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_direction_right(self):
        """c2 is to the right of c1 → dx > 0."""
        c1 = make_rect(0, 0)
        c2 = make_rect(100, 0)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx > 0.0

    def test_direction_left(self):
        """c2 is to the left of c1 → dx < 0."""
        c1 = make_rect(100, 0)
        c2 = make_rect(0, 0)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx < 0.0

    def test_direction_up(self):
        """c2 is above c1 → dy > 0."""
        c1 = make_rect(0, 0)
        c2 = make_rect(0, 100)
        dx, dy = compute_separation_vector(c1, c2)
        assert dy > 0.0

    def test_opposite_directions_are_opposite(self):
        c1 = make_rect(0, 0, w=10, h=10)
        c2 = make_rect(50, 0, w=10, h=10)
        dx1, dy1 = compute_separation_vector(c1, c2)
        dx2, dy2 = compute_separation_vector(c2, c1)
        # Should point in opposite directions
        assert dx1 * dx2 < 0

    def test_float_types_returned(self):
        c1 = make_rect(0, 0)
        c2 = make_rect(50, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert isinstance(dx, float)
        assert isinstance(dy, float)


# ─── detect_overlap_conflicts ─────────────────────────────────────────────────

class TestDetectOverlapConflicts:
    def test_empty_state_returns_empty(self):
        state = make_empty_state()
        result = detect_overlap_conflicts(state, [])
        assert result == []

    def test_returns_list(self):
        state = make_empty_state()
        result = detect_overlap_conflicts(state, [])
        assert isinstance(result, list)

    def test_no_contours_no_conflicts(self):
        state = make_empty_state(5)
        result = detect_overlap_conflicts(state, [])
        assert result == []

    def test_empty_state_conflict_score_zero(self):
        state = make_empty_state()
        score = conflict_score(state, [])
        assert score == pytest.approx(0.0)


# ─── resolve_single_conflict ──────────────────────────────────────────────────

class TestResolveSingleConflict:
    def test_missing_move_idx_returns_state(self):
        """If move_idx not in state.placed → state returned unchanged."""
        state = make_empty_state()
        conflict = OverlapConflict(idx1=0, idx2=1, iou=0.3,
                                   shift_vector=(5.0, 0.0))
        result = resolve_single_conflict(state, conflict, [])
        assert result is state

    def test_fixed_idx1_missing_returns_state(self):
        state = make_empty_state()
        conflict = OverlapConflict(idx1=0, idx2=1, iou=0.3,
                                   shift_vector=(5.0, 0.0))
        result = resolve_single_conflict(state, conflict, [], fixed=0)
        assert result is state


# ─── resolve_all_conflicts ────────────────────────────────────────────────────

class TestResolveAllConflicts:
    def test_empty_state_returns_assembly_state(self):
        state = make_empty_state()
        result = resolve_all_conflicts(state, [])
        assert isinstance(result, AssemblyState)

    def test_empty_state_no_change(self):
        state = make_empty_state()
        result = resolve_all_conflicts(state, [], max_iter=5)
        assert result.n_fragments == state.n_fragments
        assert result.placed == {}


# ─── conflict_score ───────────────────────────────────────────────────────────

class TestConflictScore:
    def test_empty_state_returns_zero(self):
        state = make_empty_state()
        result = conflict_score(state, [])
        assert result == pytest.approx(0.0)

    def test_returns_float(self):
        state = make_empty_state()
        result = conflict_score(state, [])
        assert isinstance(result, float)

    def test_nonneg(self):
        state = make_empty_state()
        result = conflict_score(state, [])
        assert result >= 0.0
