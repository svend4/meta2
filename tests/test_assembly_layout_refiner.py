"""Tests for puzzle_reconstruction/assembly/layout_refiner.py."""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.layout_refiner import (
    RefineConfig,
    FragmentPosition,
    RefineStep,
    RefineResult,
    _score_pair,
    compute_layout_score,
    refine_layout,
    apply_offset,
    compare_layouts,
)


# ─── RefineConfig ─────────────────────────────────────────────────────────────

class TestRefineConfig:
    def test_default_values(self):
        cfg = RefineConfig()
        assert cfg.max_iter == 20
        assert cfg.step_size == 1.0
        assert cfg.convergence_eps == 0.01
        assert cfg.frozen_ids == []

    def test_custom_values(self):
        cfg = RefineConfig(max_iter=50, step_size=2.5, convergence_eps=0.001, frozen_ids=[1, 2])
        assert cfg.max_iter == 50
        assert cfg.step_size == 2.5
        assert cfg.convergence_eps == 0.001
        assert cfg.frozen_ids == [1, 2]

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(max_iter=0)

    def test_invalid_step_size_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(step_size=0.0)

    def test_negative_step_size_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(step_size=-1.0)

    def test_negative_convergence_eps_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(convergence_eps=-0.001)

    def test_zero_convergence_eps_ok(self):
        cfg = RefineConfig(convergence_eps=0.0)
        assert cfg.convergence_eps == 0.0


# ─── FragmentPosition ─────────────────────────────────────────────────────────

class TestFragmentPosition:
    def test_default_position(self):
        pos = FragmentPosition(fragment_id=1)
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.rotation == 0.0

    def test_custom_position(self):
        pos = FragmentPosition(fragment_id=2, x=10.0, y=20.0, rotation=90.0)
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.rotation == 90.0

    def test_position_property(self):
        pos = FragmentPosition(fragment_id=3, x=5.0, y=7.0)
        assert pos.position == (5.0, 7.0)

    def test_distance_to_same_point(self):
        pos = FragmentPosition(fragment_id=0, x=3.0, y=4.0)
        assert pos.distance_to(pos) == 0.0

    def test_distance_to_pythagorean(self):
        p1 = FragmentPosition(fragment_id=0, x=0.0, y=0.0)
        p2 = FragmentPosition(fragment_id=1, x=3.0, y=4.0)
        assert abs(p1.distance_to(p2) - 5.0) < 1e-9

    def test_distance_symmetric(self):
        p1 = FragmentPosition(fragment_id=0, x=1.0, y=2.0)
        p2 = FragmentPosition(fragment_id=1, x=4.0, y=6.0)
        assert abs(p1.distance_to(p2) - p2.distance_to(p1)) < 1e-9


# ─── RefineStep ───────────────────────────────────────────────────────────────

class TestRefineStep:
    def test_basic_step(self):
        step = RefineStep(iteration=0, total_shift=1.5, score_delta=0.3, n_moved=2)
        assert step.iteration == 0
        assert step.total_shift == 1.5
        assert step.score_delta == 0.3
        assert step.n_moved == 2

    def test_improved_true_when_positive_delta(self):
        step = RefineStep(iteration=0, total_shift=1.0, score_delta=0.1, n_moved=1)
        assert step.improved is True

    def test_improved_false_when_negative_delta(self):
        step = RefineStep(iteration=0, total_shift=1.0, score_delta=-0.1, n_moved=1)
        assert step.improved is False

    def test_negative_iteration_raises(self):
        with pytest.raises(ValueError):
            RefineStep(iteration=-1, total_shift=0.0, score_delta=0.0, n_moved=0)

    def test_negative_total_shift_raises(self):
        with pytest.raises(ValueError):
            RefineStep(iteration=0, total_shift=-1.0, score_delta=0.0, n_moved=0)


# ─── RefineResult ─────────────────────────────────────────────────────────────

class TestRefineResult:
    def _make_result(self):
        positions = {
            0: FragmentPosition(fragment_id=0, x=1.0, y=1.0),
            1: FragmentPosition(fragment_id=1, x=5.0, y=5.0),
        }
        history = [
            RefineStep(iteration=0, total_shift=2.0, score_delta=0.1, n_moved=1),
            RefineStep(iteration=1, total_shift=0.5, score_delta=0.05, n_moved=1),
        ]
        return RefineResult(positions=positions, history=history, n_iter=2, converged=False)

    def test_total_shift_sum(self):
        result = self._make_result()
        assert abs(result.total_shift - 2.5) < 1e-9

    def test_improved_iters_count(self):
        result = self._make_result()
        assert result.improved_iters == 2

    def test_get_position_existing(self):
        result = self._make_result()
        pos = result.get_position(0)
        assert pos is not None
        assert pos.x == 1.0

    def test_get_position_missing_returns_none(self):
        result = self._make_result()
        assert result.get_position(99) is None

    def test_negative_n_iter_raises(self):
        with pytest.raises(ValueError):
            RefineResult(positions={}, history=[], n_iter=-1, converged=False)


# ─── _score_pair ──────────────────────────────────────────────────────────────

class TestScorePair:
    def test_exact_target_gap_max_score(self):
        pa = FragmentPosition(fragment_id=0, x=0.0, y=0.0)
        pb = FragmentPosition(fragment_id=1, x=1.0, y=0.0)  # distance = 1.0
        score = _score_pair(pa, pb, adjacency_score=1.0, target_gap=1.0)
        assert abs(score - 1.0) < 1e-9  # exp(0) = 1

    def test_score_decreases_with_deviation(self):
        pa = FragmentPosition(fragment_id=0, x=0.0, y=0.0)
        pb1 = FragmentPosition(fragment_id=1, x=2.0, y=0.0)  # distance 2, deviation 1
        pb2 = FragmentPosition(fragment_id=1, x=5.0, y=0.0)  # distance 5, deviation 4
        s1 = _score_pair(pa, pb1, adjacency_score=1.0, target_gap=1.0)
        s2 = _score_pair(pa, pb2, adjacency_score=1.0, target_gap=1.0)
        assert s1 > s2

    def test_zero_adjacency_score(self):
        pa = FragmentPosition(fragment_id=0, x=0.0, y=0.0)
        pb = FragmentPosition(fragment_id=1, x=1.0, y=0.0)
        score = _score_pair(pa, pb, adjacency_score=0.0, target_gap=1.0)
        assert score == 0.0


# ─── compute_layout_score ─────────────────────────────────────────────────────

class TestComputeLayoutScore:
    def test_zero_score_no_adjacency(self):
        positions = {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=1.0, y=0.0),
        }
        score = compute_layout_score(positions, {})
        assert score == 0.0

    def test_score_with_adjacency(self):
        positions = {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=1.0, y=0.0),
        }
        adjacency = {(0, 1): 1.0}
        score = compute_layout_score(positions, adjacency, target_gap=1.0)
        assert score > 0.0

    def test_missing_position_skipped(self):
        positions = {0: FragmentPosition(fragment_id=0, x=0.0, y=0.0)}
        adjacency = {(0, 1): 1.0}
        score = compute_layout_score(positions, adjacency, target_gap=1.0)
        assert score == 0.0


# ─── refine_layout ────────────────────────────────────────────────────────────

class TestRefineLayout:
    def _make_positions(self):
        return {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=5.0, y=0.0),
            2: FragmentPosition(fragment_id=2, x=10.0, y=0.0),
        }

    def test_returns_refine_result(self):
        positions = self._make_positions()
        adjacency = {(0, 1): 1.0, (1, 2): 1.0}
        result = refine_layout(positions, adjacency)
        assert isinstance(result, RefineResult)

    def test_positions_dict_contains_all_ids(self):
        positions = self._make_positions()
        adjacency = {(0, 1): 1.0}
        result = refine_layout(positions, adjacency)
        assert set(result.positions.keys()) == {0, 1, 2}

    def test_history_length_le_max_iter(self):
        positions = self._make_positions()
        cfg = RefineConfig(max_iter=5)
        result = refine_layout(positions, {}, cfg)
        assert len(result.history) <= 5

    def test_n_iter_matches_history(self):
        positions = self._make_positions()
        result = refine_layout(positions, {})
        assert result.n_iter == len(result.history)

    def test_converged_flag_set(self):
        positions = {0: FragmentPosition(fragment_id=0, x=0.0, y=0.0)}
        cfg = RefineConfig(max_iter=5, convergence_eps=1000.0)
        result = refine_layout(positions, {}, cfg)
        assert result.converged is True

    def test_frozen_id_not_moved(self):
        positions = {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=10.0, y=0.0),
        }
        adjacency = {(0, 1): 10.0}
        cfg = RefineConfig(max_iter=10, frozen_ids=[0])
        result = refine_layout(positions, adjacency, cfg)
        assert result.positions[0].x == 0.0
        assert result.positions[0].y == 0.0

    def test_empty_positions(self):
        result = refine_layout({}, {})
        assert isinstance(result, RefineResult)
        assert len(result.positions) == 0


# ─── apply_offset ─────────────────────────────────────────────────────────────

class TestApplyOffset:
    def test_shifts_all_positions(self):
        positions = {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=10.0, y=10.0),
        }
        result = apply_offset(positions, dx=5.0, dy=3.0)
        assert result[0].x == 5.0
        assert result[0].y == 3.0
        assert result[1].x == 15.0
        assert result[1].y == 13.0

    def test_shifts_only_specified_ids(self):
        positions = {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=10.0, y=10.0),
        }
        result = apply_offset(positions, dx=5.0, dy=5.0, fragment_ids=[0])
        assert result[0].x == 5.0
        assert result[1].x == 10.0

    def test_returns_new_dict(self):
        positions = {0: FragmentPosition(fragment_id=0, x=0.0, y=0.0)}
        result = apply_offset(positions, dx=1.0, dy=1.0)
        assert result is not positions

    def test_zero_offset_unchanged(self):
        positions = {0: FragmentPosition(fragment_id=0, x=5.0, y=7.0)}
        result = apply_offset(positions, dx=0.0, dy=0.0)
        assert result[0].x == 5.0
        assert result[0].y == 7.0


# ─── compare_layouts ──────────────────────────────────────────────────────────

class TestCompareLayouts:
    def test_identical_layouts_zero_shift(self):
        positions = {0: FragmentPosition(fragment_id=0, x=5.0, y=5.0)}
        result = compare_layouts(positions, positions)
        assert result["mean_shift"] == 0.0
        assert result["max_shift"] == 0.0
        assert result["n_moved"] == 0

    def test_shift_computed_correctly(self):
        before = {0: FragmentPosition(fragment_id=0, x=0.0, y=0.0)}
        after = {0: FragmentPosition(fragment_id=0, x=3.0, y=4.0)}
        result = compare_layouts(before, after)
        assert abs(result["max_shift"] - 5.0) < 1e-9
        assert abs(result["mean_shift"] - 5.0) < 1e-9
        assert result["n_moved"] == 1

    def test_empty_layouts_returns_zeros(self):
        result = compare_layouts({}, {})
        assert result["mean_shift"] == 0.0
        assert result["max_shift"] == 0.0

    def test_only_common_ids_compared(self):
        before = {
            0: FragmentPosition(fragment_id=0, x=0.0, y=0.0),
            1: FragmentPosition(fragment_id=1, x=0.0, y=0.0),
        }
        after = {0: FragmentPosition(fragment_id=0, x=3.0, y=4.0)}
        result = compare_layouts(before, after)
        assert abs(result["max_shift"] - 5.0) < 1e-9
