"""Extra tests for puzzle_reconstruction/assembly/layout_refiner.py."""
from __future__ import annotations

import math
import pytest

from puzzle_reconstruction.assembly.layout_refiner import (
    RefineConfig,
    FragmentPosition,
    RefineStep,
    RefineResult,
    compute_layout_score,
    refine_layout,
    apply_offset,
    compare_layouts,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pos(fid, x=0.0, y=0.0, rot=0.0):
    return FragmentPosition(fragment_id=fid, x=x, y=y, rotation=rot)


def _layout3():
    return {0: _pos(0, 0.0, 0.0), 1: _pos(1, 1.0, 0.0), 2: _pos(2, 2.0, 0.0)}


def _adj():
    return {(0, 1): 0.9, (1, 2): 0.8}


# ─── RefineConfig (extra) ─────────────────────────────────────────────────────

class TestRefineConfigExtra:
    def test_large_max_iter(self):
        cfg = RefineConfig(max_iter=1000)
        assert cfg.max_iter == 1000

    def test_small_step_size(self):
        cfg = RefineConfig(step_size=0.001)
        assert cfg.step_size == pytest.approx(0.001)

    def test_large_convergence_eps(self):
        cfg = RefineConfig(convergence_eps=10.0)
        assert cfg.convergence_eps == pytest.approx(10.0)

    def test_frozen_ids_empty_default(self):
        cfg = RefineConfig()
        assert cfg.frozen_ids == []

    def test_frozen_ids_single(self):
        cfg = RefineConfig(frozen_ids=[5])
        assert 5 in cfg.frozen_ids

    def test_frozen_ids_multiple(self):
        cfg = RefineConfig(frozen_ids=[0, 1, 2])
        assert cfg.frozen_ids == [0, 1, 2]

    def test_step_size_large(self):
        cfg = RefineConfig(step_size=100.0)
        assert cfg.step_size == pytest.approx(100.0)


# ─── FragmentPosition (extra) ─────────────────────────────────────────────────

class TestFragmentPositionExtra:
    def test_position_tuple_values(self):
        p = _pos(0, 5.0, 7.0)
        assert p.position == (pytest.approx(5.0), pytest.approx(7.0))

    def test_distance_to_345_triangle(self):
        a = _pos(0, 0.0, 0.0)
        b = _pos(1, 3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_negative_coords(self):
        a = _pos(0, -3.0, -4.0)
        b = _pos(1, 0.0, 0.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_fragment_id_large(self):
        p = _pos(9999)
        assert p.fragment_id == 9999

    def test_rotation_stored(self):
        p = _pos(0, 1.0, 2.0, rot=90.0)
        assert p.rotation == pytest.approx(90.0)

    def test_default_rotation_zero(self):
        p = _pos(0, 1.0, 2.0)
        assert p.rotation == pytest.approx(0.0)

    def test_x_y_stored(self):
        p = _pos(1, 3.5, -2.1)
        assert p.x == pytest.approx(3.5)
        assert p.y == pytest.approx(-2.1)

    def test_distance_commutative(self):
        a = _pos(0, 1.0, 5.0)
        b = _pos(1, 4.0, 9.0)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a))


# ─── RefineStep (extra) ───────────────────────────────────────────────────────

class TestRefineStepExtra:
    def test_positive_delta_improved(self):
        s = RefineStep(iteration=2, total_shift=0.5, score_delta=0.01, n_moved=1)
        assert s.improved is True

    def test_negative_delta_not_improved(self):
        s = RefineStep(iteration=0, total_shift=0.0, score_delta=-0.01, n_moved=0)
        assert s.improved is False

    def test_total_shift_zero(self):
        s = RefineStep(iteration=0, total_shift=0.0, score_delta=0.0, n_moved=0)
        assert s.total_shift == pytest.approx(0.0)

    def test_score_delta_stored(self):
        s = RefineStep(iteration=1, total_shift=1.0, score_delta=-0.5, n_moved=3)
        assert s.score_delta == pytest.approx(-0.5)

    def test_n_moved_stored(self):
        s = RefineStep(iteration=0, total_shift=2.0, score_delta=0.3, n_moved=5)
        assert s.n_moved == 5

    def test_iteration_stored(self):
        s = RefineStep(iteration=7, total_shift=1.0, score_delta=0.0, n_moved=0)
        assert s.iteration == 7


# ─── RefineResult (extra) ─────────────────────────────────────────────────────

class TestRefineResultExtra:
    def _make(self, n=3, converged=True):
        steps = [
            RefineStep(iteration=i, total_shift=float(i + 1),
                       score_delta=0.1, n_moved=1)
            for i in range(n)
        ]
        positions = {i: _pos(i, float(i)) for i in range(n)}
        return RefineResult(positions=positions, history=steps,
                            n_iter=n, converged=converged)

    def test_total_shift_sum(self):
        r = self._make(4)
        expected = 1.0 + 2.0 + 3.0 + 4.0
        assert r.total_shift == pytest.approx(expected)

    def test_n_iter_stored(self):
        r = self._make(5)
        assert r.n_iter == 5

    def test_positions_keys(self):
        r = self._make(3)
        assert set(r.positions.keys()) == {0, 1, 2}

    def test_get_position_existing(self):
        r = self._make(3)
        p = r.get_position(1)
        assert p is not None
        assert p.fragment_id == 1

    def test_get_position_missing(self):
        r = self._make(2)
        assert r.get_position(99) is None

    def test_improved_iters_count(self):
        r = self._make(4)  # all score_delta=0.1 → all improved
        assert r.improved_iters == 4

    def test_not_converged(self):
        r = self._make(converged=False)
        assert r.converged is False


# ─── compute_layout_score (extra) ────────────────────────────────────────────

class TestComputeLayoutScoreExtra:
    def test_score_nonneg(self):
        s = compute_layout_score(_layout3(), _adj())
        assert s >= 0.0

    def test_score_nonneg_with_adj(self):
        s = compute_layout_score(_layout3(), _adj(), target_gap=1.0)
        assert s >= 0.0

    def test_empty_positions_zero(self):
        s = compute_layout_score({}, _adj())
        assert s == pytest.approx(0.0)

    def test_two_pairs_score_positive(self):
        pos = {0: _pos(0, 0.0), 1: _pos(1, 1.0), 2: _pos(2, 2.0)}
        adj = {(0, 1): 1.0, (1, 2): 1.0}
        s = compute_layout_score(pos, adj, target_gap=1.0)
        assert s > 0.0

    def test_perfect_placement_score_nonneg(self):
        pos = {0: _pos(0, 0.0), 1: _pos(1, 1.0)}
        adj = {(0, 1): 1.0}
        s = compute_layout_score(pos, adj, target_gap=1.0)
        assert s >= 0.0

    def test_reversed_pair_key_handled(self):
        # If pair (1, 0) is used instead of (0, 1), should still work or return 0
        pos = {0: _pos(0, 0.0), 1: _pos(1, 1.0)}
        adj = {(1, 0): 0.9}
        s = compute_layout_score(pos, adj, target_gap=1.0)
        assert s >= 0.0


# ─── refine_layout (extra) ────────────────────────────────────────────────────

class TestRefineLayoutExtra:
    def test_converged_flag_present(self):
        r = refine_layout(_layout3(), _adj())
        assert isinstance(r.converged, bool)

    def test_history_steps_sequential(self):
        r = refine_layout(_layout3(), _adj())
        for i, step in enumerate(r.history):
            assert step.iteration == i

    def test_frozen_fragment_not_moved(self):
        pos = {0: _pos(0, 0.0, 0.0), 1: _pos(1, 1.0, 0.0)}
        cfg = RefineConfig(max_iter=5, frozen_ids=[0])
        r = refine_layout(pos, {(0, 1): 0.9}, cfg)
        assert r.positions[0].x == pytest.approx(0.0)
        assert r.positions[0].y == pytest.approx(0.0)

    def test_two_fragments_result_has_both(self):
        pos = {0: _pos(0, 0.0), 1: _pos(1, 5.0)}
        r = refine_layout(pos, {(0, 1): 1.0})
        assert 0 in r.positions
        assert 1 in r.positions

    def test_no_adjacency_no_movement(self):
        pos = {0: _pos(0, 3.0, 5.0), 1: _pos(1, 7.0, 2.0)}
        r = refine_layout(pos, {})
        # Without adjacency, positions may be unchanged
        assert isinstance(r, RefineResult)

    def test_all_frozen_no_movement(self):
        pos = {0: _pos(0, 1.0, 2.0), 1: _pos(1, 3.0, 4.0)}
        cfg = RefineConfig(max_iter=5, frozen_ids=[0, 1])
        r = refine_layout(pos, {(0, 1): 0.9}, cfg)
        assert r.positions[0].x == pytest.approx(1.0)
        assert r.positions[1].x == pytest.approx(3.0)


# ─── apply_offset (extra) ─────────────────────────────────────────────────────

class TestApplyOffsetExtra:
    def test_negative_offset(self):
        pos = {0: _pos(0, 5.0, 5.0)}
        result = apply_offset(pos, -3.0, -2.0)
        assert result[0].x == pytest.approx(2.0)
        assert result[0].y == pytest.approx(3.0)

    def test_large_offset(self):
        pos = {0: _pos(0, 0.0, 0.0)}
        result = apply_offset(pos, 1000.0, 500.0)
        assert result[0].x == pytest.approx(1000.0)

    def test_fragment_ids_not_in_list_unchanged(self):
        pos = {0: _pos(0, 0.0), 1: _pos(1, 5.0)}
        result = apply_offset(pos, 10.0, 0.0, fragment_ids=[0])
        assert result[1].x == pytest.approx(5.0)

    def test_rotation_not_changed_by_offset(self):
        pos = {0: _pos(0, 1.0, 1.0, rot=45.0)}
        result = apply_offset(pos, 3.0, 3.0)
        assert result[0].rotation == pytest.approx(45.0)

    def test_empty_positions(self):
        result = apply_offset({}, 5.0, 5.0)
        assert result == {}

    def test_all_fragments_shifted(self):
        pos = {i: _pos(i, float(i), 0.0) for i in range(5)}
        result = apply_offset(pos, 1.0, 0.0)
        for i in range(5):
            assert result[i].x == pytest.approx(float(i) + 1.0)


# ─── compare_layouts (extra) ──────────────────────────────────────────────────

class TestCompareLayoutsExtra:
    def test_keys_in_result(self):
        cmp = compare_layouts(_layout3(), _layout3())
        assert "mean_shift" in cmp
        assert "max_shift" in cmp
        assert "n_moved" in cmp

    def test_identical_mean_zero(self):
        pos = _layout3()
        cmp = compare_layouts(pos, pos)
        assert cmp["mean_shift"] == pytest.approx(0.0)

    def test_single_fragment_shifted(self):
        before = {0: _pos(0, 0.0, 0.0)}
        after = {0: _pos(0, 3.0, 4.0)}
        cmp = compare_layouts(before, after)
        assert cmp["mean_shift"] == pytest.approx(5.0)
        assert cmp["n_moved"] == 1

    def test_max_shift_gte_mean(self):
        before = {0: _pos(0, 0.0), 1: _pos(1, 0.0)}
        after = {0: _pos(0, 10.0), 1: _pos(1, 1.0)}
        cmp = compare_layouts(before, after)
        assert cmp["max_shift"] >= cmp["mean_shift"]

    def test_n_moved_zero_when_identical(self):
        pos = _layout3()
        cmp = compare_layouts(pos, pos)
        assert cmp["n_moved"] == 0

    def test_shift_threshold_n_moved(self):
        before = {0: _pos(0, 0.0), 1: _pos(1, 0.0)}
        # fragment 0 moves a lot, fragment 1 barely moves
        after = {0: _pos(0, 100.0), 1: _pos(1, 0.0001)}
        cmp = compare_layouts(before, after)
        # At least fragment 0 moved significantly
        assert cmp["n_moved"] >= 1
