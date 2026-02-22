"""Extra tests for puzzle_reconstruction.algorithms.overlap_resolver."""
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
from puzzle_reconstruction.assembly.assembly_state import (
    AssemblyState,
    create_state,
    place_fragment,
)


def _sq(x0, y0, size):
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float64)


def _state_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0)):
    state = place_fragment(create_state(2), 0, pos0)
    state = place_fragment(state, 1, pos1)
    return state


# ─── OverlapConflict extras ───────────────────────────────────────────────────

class TestOverlapConflictExtra:
    def test_iou_zero(self):
        oc = OverlapConflict(0, 1, iou=0.0)
        assert oc.iou == pytest.approx(0.0)

    def test_iou_one(self):
        oc = OverlapConflict(0, 1, iou=1.0)
        assert oc.iou == pytest.approx(1.0)

    def test_shift_vector_custom(self):
        oc = OverlapConflict(2, 3, iou=0.5, shift_vector=(7.0, -3.0))
        assert oc.shift_vector == (7.0, -3.0)

    def test_large_indices(self):
        oc = OverlapConflict(100, 999, iou=0.1)
        assert oc.idx1 == 100
        assert oc.idx2 == 999

    def test_repr_string(self):
        oc = OverlapConflict(0, 1, iou=0.5)
        assert isinstance(repr(oc), str)

    def test_shift_vector_negative(self):
        oc = OverlapConflict(0, 1, iou=0.3, shift_vector=(-5.0, -10.0))
        assert oc.shift_vector[0] == pytest.approx(-5.0)


# ─── compute_separation_vector extras ────────────────────────────────────────

class TestComputeSeparationVectorExtra:
    def test_returns_tuple(self):
        result = compute_separation_vector(_sq(0, 0, 50), _sq(60, 0, 50))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_c2_below_c1_dy_positive(self):
        c1 = _sq(0, 0, 50)
        c2 = _sq(0, 60, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert dy > 0

    def test_c2_left_of_c1_dx_negative(self):
        c1 = _sq(60, 0, 50)
        c2 = _sq(0, 0, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx < 0

    def test_both_empty_returns_zero(self):
        c1 = np.empty((0, 2), dtype=np.float64)
        c2 = np.empty((0, 2), dtype=np.float64)
        assert compute_separation_vector(c1, c2) == (0.0, 0.0)

    def test_single_point_contours(self):
        c1 = np.array([[0.0, 0.0]], dtype=np.float64)
        c2 = np.array([[10.0, 0.0]], dtype=np.float64)
        dx, dy = compute_separation_vector(c1, c2)
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_large_offset(self):
        c1 = _sq(0, 0, 50)
        c2 = _sq(1000, 1000, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx > 0
        assert dy > 0


# ─── detect_overlap_conflicts extras ─────────────────────────────────────────

class TestDetectOverlapConflictsExtra:
    def test_returns_list(self):
        cnts = [_sq(0, 0, 50), _sq(200, 0, 50)]
        state = _state_two()
        assert isinstance(detect_overlap_conflicts(state, cnts), list)

    def test_no_placed_no_conflicts(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = create_state(2)
        assert detect_overlap_conflicts(state, cnts) == []

    def test_partial_overlap_with_low_threshold(self):
        cnts = [_sq(0, 0, 50), _sq(25, 0, 50)]
        state = _state_two()
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        assert len(result) >= 1

    def test_iou_in_range(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        for c in result:
            assert 0.0 < c.iou <= 1.0

    def test_sorted_descending_iou(self):
        cnts = [_sq(0, 0, 50), _sq(5, 0, 50), _sq(40, 0, 50)]
        state = create_state(3)
        for i in range(3):
            state = place_fragment(state, i, (0.0, 0.0))
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        ious = [c.iou for c in result]
        assert ious == sorted(ious, reverse=True)

    def test_threshold_1_no_conflicts(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        assert detect_overlap_conflicts(state, cnts, threshold=1.0) == []


# ─── resolve_single_conflict extras ──────────────────────────────────────────

class TestResolveSingleConflictExtra:
    def _make_conflict(self, idx1=0, idx2=1, dx=10.0, dy=5.0):
        return OverlapConflict(idx1=idx1, idx2=idx2, iou=0.5,
                               shift_vector=(dx, dy))

    def test_returns_assembly_state(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        r = resolve_single_conflict(state, self._make_conflict(), cnts)
        assert isinstance(r, AssemblyState)

    def test_idx2_moves_by_shift(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0))
        r = resolve_single_conflict(state, self._make_conflict(dx=15.0, dy=0.0), cnts)
        assert r.placed[1].position[0] == pytest.approx(15.0)

    def test_idx1_unchanged_default(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two(pos0=(3.0, 7.0), pos1=(0.0, 0.0))
        r = resolve_single_conflict(state, self._make_conflict(), cnts)
        assert r.placed[0].position[0] == pytest.approx(3.0)
        assert r.placed[0].position[1] == pytest.approx(7.0)

    def test_fixed_1_moves_idx1_reverse(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        r = resolve_single_conflict(state, self._make_conflict(dx=6.0, dy=0.0), cnts, fixed=1)
        assert r.placed[0].position[0] == pytest.approx(-6.0)

    def test_large_shift(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        r = resolve_single_conflict(state,
                                    self._make_conflict(dx=1000.0, dy=1000.0), cnts)
        assert r.placed[1].position[0] == pytest.approx(1000.0)


# ─── resolve_all_conflicts extras ────────────────────────────────────────────

class TestResolveAllConflictsExtra:
    def test_returns_assembly_state(self):
        cnts = [_sq(0, 0, 50), _sq(200, 0, 50)]
        state = _state_two()
        r = resolve_all_conflicts(state, cnts)
        assert isinstance(r, AssemblyState)

    def test_no_conflicts_positions_unchanged(self):
        cnts = [_sq(0, 0, 50), _sq(200, 0, 50)]
        state = _state_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0))
        r = resolve_all_conflicts(state, cnts)
        assert r.placed[0].position == pytest.approx((0.0, 0.0))
        assert r.placed[1].position == pytest.approx((0.0, 0.0))

    def test_max_iter_1_runs(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        r = resolve_all_conflicts(state, cnts, max_iter=1, threshold=0.01)
        assert isinstance(r, AssemblyState)

    def test_after_resolve_fewer_or_equal_conflicts(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        r = resolve_all_conflicts(state, cnts, max_iter=10, threshold=0.01)
        before = len(detect_overlap_conflicts(state, cnts, threshold=0.01))
        after = len(detect_overlap_conflicts(r, cnts, threshold=0.01))
        assert after <= before

    def test_single_fragment_no_change(self):
        cnts = [_sq(0, 0, 50)]
        state = place_fragment(create_state(1), 0, (5.0, 5.0))
        r = resolve_all_conflicts(state, cnts)
        assert r.placed[0].position == pytest.approx((5.0, 5.0))


# ─── conflict_score extras ────────────────────────────────────────────────────

class TestConflictScoreExtra:
    def test_returns_float(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        s = conflict_score(state, cnts, threshold=0.01)
        assert isinstance(s, float)

    def test_nonneg(self):
        cnts = [_sq(0, 0, 50), _sq(20, 0, 50)]
        state = _state_two()
        assert conflict_score(state, cnts) >= 0.0

    def test_zero_for_separated(self):
        cnts = [_sq(0, 0, 50), _sq(200, 0, 50)]
        state = _state_two()
        assert conflict_score(state, cnts) == pytest.approx(0.0)

    def test_full_overlap_positive(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        s = conflict_score(state, cnts, threshold=0.01)
        assert s > 0.0

    def test_empty_state_zero(self):
        cnts = [_sq(0, 0, 50)]
        state = create_state(1)
        assert conflict_score(state, cnts) == pytest.approx(0.0)

    def test_decreases_after_resolve(self):
        cnts = [_sq(0, 0, 50), _sq(0, 0, 50)]
        state = _state_two()
        s_before = conflict_score(state, cnts, threshold=0.01)
        resolved = resolve_all_conflicts(state, cnts, max_iter=5, threshold=0.01)
        s_after = conflict_score(resolved, cnts, threshold=0.01)
        assert s_after <= s_before
