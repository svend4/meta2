"""Extra tests for puzzle_reconstruction/assembly/layout_refiner.py."""
from __future__ import annotations

import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pos(fid, x=0.0, y=0.0):
    return FragmentPosition(fragment_id=fid, x=x, y=y)


def _positions():
    return {0: _pos(0, 0, 0), 1: _pos(1, 10, 0)}


def _adjacency():
    return {(0, 1): 0.9}


# ─── RefineConfig ───────────────────────────────────────────────────────────

class TestRefineConfigExtra:
    def test_defaults(self):
        c = RefineConfig()
        assert c.max_iter == 20
        assert c.step_size == 1.0
        assert c.convergence_eps == pytest.approx(0.01)

    def test_zero_max_iter_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(max_iter=0)

    def test_zero_step_size_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(step_size=0.0)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError):
            RefineConfig(convergence_eps=-0.1)

    def test_frozen_ids(self):
        c = RefineConfig(frozen_ids=[0, 1])
        assert c.frozen_ids == [0, 1]


# ─── FragmentPosition ──────────────────────────────────────────────────────

class TestFragmentPositionExtra:
    def test_position_property(self):
        fp = _pos(0, 10.0, 20.0)
        assert fp.position == (10.0, 20.0)

    def test_distance_to(self):
        a = _pos(0, 0.0, 0.0)
        b = _pos(1, 3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_same(self):
        a = _pos(0, 5.0, 5.0)
        assert a.distance_to(a) == pytest.approx(0.0)


# ─── RefineStep ─────────────────────────────────────────────────────────────

class TestRefineStepExtra:
    def test_valid(self):
        rs = RefineStep(iteration=0, total_shift=1.0,
                        score_delta=0.5, n_moved=1)
        assert rs.improved is True

    def test_negative_delta(self):
        rs = RefineStep(iteration=0, total_shift=1.0,
                        score_delta=-0.1, n_moved=1)
        assert rs.improved is False

    def test_negative_iteration_raises(self):
        with pytest.raises(ValueError):
            RefineStep(iteration=-1, total_shift=0.0,
                       score_delta=0.0, n_moved=0)

    def test_negative_shift_raises(self):
        with pytest.raises(ValueError):
            RefineStep(iteration=0, total_shift=-1.0,
                       score_delta=0.0, n_moved=0)


# ─── RefineResult ───────────────────────────────────────────────────────────

class TestRefineResultExtra:
    def test_valid(self):
        rr = RefineResult(positions={}, history=[], n_iter=0, converged=True)
        assert rr.converged is True

    def test_negative_n_iter_raises(self):
        with pytest.raises(ValueError):
            RefineResult(positions={}, history=[], n_iter=-1, converged=False)

    def test_total_shift(self):
        steps = [
            RefineStep(iteration=0, total_shift=1.5, score_delta=0.1, n_moved=1),
            RefineStep(iteration=1, total_shift=0.5, score_delta=0.0, n_moved=0),
        ]
        rr = RefineResult(positions={}, history=steps, n_iter=2, converged=True)
        assert rr.total_shift == pytest.approx(2.0)

    def test_improved_iters(self):
        steps = [
            RefineStep(iteration=0, total_shift=1.0, score_delta=0.5, n_moved=1),
            RefineStep(iteration=1, total_shift=0.5, score_delta=-0.1, n_moved=0),
            RefineStep(iteration=2, total_shift=0.3, score_delta=0.1, n_moved=1),
        ]
        rr = RefineResult(positions={}, history=steps, n_iter=3, converged=False)
        assert rr.improved_iters == 2

    def test_get_position(self):
        pos = _pos(5, 10.0, 20.0)
        rr = RefineResult(positions={5: pos}, history=[], n_iter=0, converged=True)
        assert rr.get_position(5) is pos
        assert rr.get_position(99) is None


# ─── compute_layout_score ──────────────────────────────────────────────────

class TestComputeLayoutScoreExtra:
    def test_empty(self):
        assert compute_layout_score({}, {}) == pytest.approx(0.0)

    def test_basic(self):
        positions = _positions()
        adj = _adjacency()
        score = compute_layout_score(positions, adj, target_gap=10.0)
        assert score > 0.0

    def test_missing_fragment(self):
        positions = {0: _pos(0)}
        adj = {(0, 1): 0.9}
        score = compute_layout_score(positions, adj)
        assert score == pytest.approx(0.0)


# ─── refine_layout ──────────────────────────────────────────────────────────

class TestRefineLayoutExtra:
    def test_basic(self):
        positions = _positions()
        adj = _adjacency()
        result = refine_layout(positions, adj, target_gap=10.0)
        assert isinstance(result, RefineResult)
        assert result.n_iter > 0

    def test_converges(self):
        positions = _positions()
        adj = _adjacency()
        cfg = RefineConfig(max_iter=100, step_size=0.5)
        result = refine_layout(positions, adj, cfg, target_gap=10.0)
        assert isinstance(result, RefineResult)

    def test_frozen_ids(self):
        positions = {0: _pos(0, 0, 0), 1: _pos(1, 50, 0)}
        adj = {(0, 1): 0.9}
        cfg = RefineConfig(max_iter=5, frozen_ids=[0])
        result = refine_layout(positions, adj, cfg, target_gap=1.0)
        # Fragment 0 should not have moved
        assert result.positions[0].x == pytest.approx(0.0)
        assert result.positions[0].y == pytest.approx(0.0)

    def test_empty_positions(self):
        result = refine_layout({}, {})
        assert result.n_iter >= 0


# ─── apply_offset ───────────────────────────────────────────────────────────

class TestApplyOffsetExtra:
    def test_all(self):
        positions = _positions()
        shifted = apply_offset(positions, dx=5.0, dy=10.0)
        assert shifted[0].x == pytest.approx(5.0)
        assert shifted[0].y == pytest.approx(10.0)
        assert shifted[1].x == pytest.approx(15.0)

    def test_specific_ids(self):
        positions = _positions()
        shifted = apply_offset(positions, dx=5.0, dy=0.0, fragment_ids=[0])
        assert shifted[0].x == pytest.approx(5.0)
        assert shifted[1].x == pytest.approx(10.0)  # unchanged

    def test_empty(self):
        result = apply_offset({}, dx=5, dy=5)
        assert result == {}


# ─── compare_layouts ────────────────────────────────────────────────────────

class TestCompareLayoutsExtra:
    def test_no_change(self):
        positions = _positions()
        result = compare_layouts(positions, positions)
        assert result["mean_shift"] == pytest.approx(0.0)
        assert result["n_moved"] == 0

    def test_with_change(self):
        before = _positions()
        after = apply_offset(before, dx=3.0, dy=4.0)
        result = compare_layouts(before, after)
        assert result["mean_shift"] == pytest.approx(5.0)
        assert result["max_shift"] == pytest.approx(5.0)
        assert result["n_moved"] == 2

    def test_no_common(self):
        a = {0: _pos(0)}
        b = {1: _pos(1)}
        result = compare_layouts(a, b)
        assert result["n_moved"] == 0
