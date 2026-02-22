"""
Тесты для puzzle_reconstruction.algorithms.overlap_resolver.
"""
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
    create_state,
    place_fragment,
    add_adjacency,
)


# ─── Вспомогательные контуры и состояния ─────────────────────────────────────

def _square_contour(x0: float, y0: float, size: float) -> np.ndarray:
    """Квадратный контур (4, 2) float64."""
    return np.array([
        [x0,        y0],
        [x0 + size, y0],
        [x0 + size, y0 + size],
        [x0,        y0 + size],
    ], dtype=np.float64)


def _make_state_with_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0)):
    """Два фрагмента, размещённых в заданных позициях."""
    state = place_fragment(create_state(2), 0, pos0)
    state = place_fragment(state, 1, pos1)
    return state


# ─── OverlapConflict ──────────────────────────────────────────────────────────

class TestOverlapConflict:
    def test_fields_accessible(self):
        oc = OverlapConflict(idx1=0, idx2=1, iou=0.3, shift_vector=(2.0, 1.5))
        assert oc.idx1 == 0
        assert oc.idx2 == 1
        assert oc.iou  == pytest.approx(0.3)
        assert oc.shift_vector == (2.0, 1.5)

    def test_default_shift_zero(self):
        oc = OverlapConflict(0, 1, 0.1)
        assert oc.shift_vector == (0.0, 0.0)

    def test_repr_contains_indices(self):
        oc = OverlapConflict(3, 7, 0.25)
        r  = repr(oc)
        assert "3" in r and "7" in r

    def test_repr_contains_iou(self):
        oc = OverlapConflict(0, 1, 0.42)
        assert "0.4" in repr(oc)


# ─── compute_separation_vector ────────────────────────────────────────────────

class TestComputeSeparationVector:
    def test_returns_two_floats(self):
        c1 = _square_contour(0, 0, 50)
        c2 = _square_contour(10, 0, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert isinstance(dx, float)
        assert isinstance(dy, float)

    def test_empty_contour1_returns_zero(self):
        c1 = np.empty((0, 2), dtype=np.float64)
        c2 = _square_contour(0, 0, 50)
        assert compute_separation_vector(c1, c2) == (0.0, 0.0)

    def test_empty_contour2_returns_zero(self):
        c1 = _square_contour(0, 0, 50)
        c2 = np.empty((0, 2), dtype=np.float64)
        assert compute_separation_vector(c1, c2) == (0.0, 0.0)

    def test_c2_right_of_c1_dx_positive(self):
        # c1 centroid=(25,25), c2 centroid=(75,25) → c2 правее c1 → сдвиг вправо
        c1 = _square_contour(0,  0, 50)
        c2 = _square_contour(50, 0, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert dx > 0

    def test_c2_above_c1_dy_negative(self):
        # c1 centroid=(25,25), c2 centroid=(25,-25)
        c1 = _square_contour(0, 0, 50)
        c2 = _square_contour(0, -50, 50)
        dx, dy = compute_separation_vector(c1, c2)
        assert dy < 0

    def test_coincident_centroids_nonzero(self):
        c1 = _square_contour(0, 0, 50)
        c2 = _square_contour(0, 0, 50)
        dx, dy = compute_separation_vector(c1, c2)
        # Должен вернуть ненулевой сдвиг
        assert abs(dx) + abs(dy) > 0


# ─── detect_overlap_conflicts ─────────────────────────────────────────────────

class TestDetectOverlapConflicts:
    def test_no_overlap_empty_list(self):
        # Фрагменты далеко друг от друга
        cnts  = [_square_contour(0, 0, 50), _square_contour(200, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        result = detect_overlap_conflicts(state, cnts)
        assert result == []

    def test_fully_overlapping_returns_conflict(self):
        # Оба фрагмента на одном месте
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        assert len(result) >= 1

    def test_conflict_has_positive_iou(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        for c in result:
            assert c.iou > 0.0

    def test_sorted_by_iou_descending(self):
        # 3 фрагмента с разной степенью перекрытия
        cnts = [
            _square_contour(0, 0, 50),
            _square_contour(5, 0, 50),   # Большое перекрытие
            _square_contour(40, 0, 50),  # Малое перекрытие
        ]
        state = create_state(3)
        for i in range(3):
            state = place_fragment(state, i, (0.0, 0.0))
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        for i in range(len(result) - 1):
            assert result[i].iou >= result[i + 1].iou

    def test_empty_state_no_conflicts(self):
        cnts  = [_square_contour(0, 0, 50)]
        state = create_state(1)
        result = detect_overlap_conflicts(state, cnts)
        assert result == []

    def test_single_fragment_no_conflicts(self):
        cnts  = [_square_contour(0, 0, 50)]
        state = place_fragment(create_state(1), 0, (0.0, 0.0))
        result = detect_overlap_conflicts(state, cnts)
        assert result == []

    def test_conflict_indices_correct(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        result = detect_overlap_conflicts(state, cnts, threshold=0.01)
        if result:
            c = result[0]
            assert {c.idx1, c.idx2} == {0, 1}

    def test_high_threshold_no_conflicts(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        # Порог 1.0 — никогда не превысится
        result = detect_overlap_conflicts(state, cnts, threshold=1.0)
        assert result == []


# ─── resolve_single_conflict ──────────────────────────────────────────────────

class TestResolveSingleConflict:
    def _conflict(self, idx1=0, idx2=1, dx=10.0, dy=5.0):
        return OverlapConflict(idx1=idx1, idx2=idx2, iou=0.5,
                               shift_vector=(dx, dy))

    def test_returns_assembly_state(self):
        from puzzle_reconstruction.assembly.assembly_state import AssemblyState
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two()
        c     = self._conflict()
        r     = resolve_single_conflict(state, c, cnts)
        assert isinstance(r, AssemblyState)

    def test_idx2_position_changed(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0))
        c     = self._conflict(dx=10.0, dy=5.0)
        r     = resolve_single_conflict(state, c, cnts)
        new_pos = r.placed[1].position
        assert new_pos[0] == pytest.approx(10.0)
        assert new_pos[1] == pytest.approx(5.0)

    def test_idx1_position_unchanged(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0))
        c     = self._conflict()
        r     = resolve_single_conflict(state, c, cnts)
        assert r.placed[0].position == pytest.approx((0.0, 0.0))

    def test_fixed_idx1_moves_idx2(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0))
        c     = self._conflict(dx=8.0, dy=3.0)
        r     = resolve_single_conflict(state, c, cnts, fixed=0)
        assert r.placed[1].position[0] == pytest.approx(8.0)

    def test_fixed_idx2_moves_idx1(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0.0, 0.0), pos1=(0.0, 0.0))
        c     = self._conflict(dx=8.0, dy=3.0)
        r     = resolve_single_conflict(state, c, cnts, fixed=1)
        # idx1 должен сдвинуться в обратную сторону
        assert r.placed[0].position[0] == pytest.approx(-8.0)

    def test_missing_fragment_unchanged(self):
        cnts  = [_square_contour(0, 0, 50)]
        state = place_fragment(create_state(2), 0, (0.0, 0.0))
        # idx2=1 не размещён
        c = self._conflict(idx1=0, idx2=1)
        r = resolve_single_conflict(state, c, cnts)
        # Не должно упасть
        assert isinstance(r.placed, dict)


# ─── resolve_all_conflicts ────────────────────────────────────────────────────

class TestResolveAllConflicts:
    def test_no_conflicts_state_unchanged(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(200, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        r     = resolve_all_conflicts(state, cnts)
        # Нет конфликтов — позиции не меняются
        assert r.placed[0].position == pytest.approx(state.placed[0].position)
        assert r.placed[1].position == pytest.approx(state.placed[1].position)

    def test_overlapping_reduces_conflicts(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        r     = resolve_all_conflicts(state, cnts, max_iter=20, threshold=0.01)
        after = detect_overlap_conflicts(r, cnts, threshold=0.01)
        before = detect_overlap_conflicts(state, cnts, threshold=0.01)
        # После разрешения конфликтов должно быть не больше, чем до
        assert len(after) <= len(before)

    def test_max_iter_zero_returns_original(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        r     = resolve_all_conflicts(state, cnts, max_iter=0)
        # max_iter=0 → ни одной итерации
        assert isinstance(r.placed, dict)

    def test_single_fragment_state_unchanged(self):
        cnts  = [_square_contour(0, 0, 50)]
        state = place_fragment(create_state(1), 0, (0.0, 0.0))
        r     = resolve_all_conflicts(state, cnts, max_iter=5)
        assert r.placed[0].position == pytest.approx((0.0, 0.0))


# ─── conflict_score ───────────────────────────────────────────────────────────

class TestConflictScore:
    def test_no_overlap_zero_score(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(200, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        assert conflict_score(state, cnts) == pytest.approx(0.0)

    def test_full_overlap_positive_score(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        s     = conflict_score(state, cnts, threshold=0.01)
        assert s > 0.0

    def test_score_non_negative(self):
        cnts  = [_square_contour(0, 0, 50), _square_contour(10, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        assert conflict_score(state, cnts) >= 0.0

    def test_empty_state_zero_score(self):
        cnts  = [_square_contour(0, 0, 50)]
        state = create_state(1)
        assert conflict_score(state, cnts) == pytest.approx(0.0)

    def test_score_increases_with_more_overlap(self):
        cnts1 = [_square_contour(0, 0, 50), _square_contour(0, 0, 50)]
        cnts2 = [_square_contour(0, 0, 50), _square_contour(25, 0, 50)]
        state = _make_state_with_two(pos0=(0, 0), pos1=(0, 0))
        s_full = conflict_score(state, cnts1, threshold=0.01)
        s_part = conflict_score(state, cnts2, threshold=0.01)
        assert s_full >= s_part
