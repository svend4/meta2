"""Extra tests for puzzle_reconstruction/assembly/fragment_arranger.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.fragment_arranger import (
    ArrangementParams,
    FragmentPlacement,
    arrange_grid,
    arrange_strip,
    center_placements,
    group_bbox,
    shift_placements,
    arrange,
    batch_arrange,
)


# ─── ArrangementParams ──────────────────────────────────────────────────────

class TestArrangementParamsExtra:
    def test_defaults(self):
        p = ArrangementParams()
        assert p.strategy == "strip"
        assert p.cols == 4
        assert p.gap == 4
        assert p.canvas_w == 512
        assert p.canvas_h == 512

    def test_valid_strategies(self):
        for s in ("grid", "strip", "center"):
            ArrangementParams(strategy=s)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(strategy="bad")

    def test_zero_cols_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(cols=0)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(gap=-1)

    def test_zero_canvas_w_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(canvas_w=0)

    def test_zero_canvas_h_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(canvas_h=0)


# ─── FragmentPlacement ──────────────────────────────────────────────────────

class TestFragmentPlacementExtra:
    def test_valid(self):
        p = FragmentPlacement(fragment_id=0, x=10, y=20, width=30, height=40)
        assert p.fragment_id == 0
        assert p.x == 10

    def test_bbox(self):
        p = FragmentPlacement(fragment_id=0, x=10, y=20, width=30, height=40)
        assert p.bbox == (10, 20, 30, 40)

    def test_center(self):
        p = FragmentPlacement(fragment_id=0, x=0, y=0, width=100, height=200)
        assert p.center == (50.0, 100.0)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=-1, x=0, y=0, width=10, height=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, x=-1, y=0, width=10, height=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, x=0, y=-1, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, x=0, y=0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            FragmentPlacement(fragment_id=0, x=0, y=0, width=10, height=0)


# ─── arrange_grid ───────────────────────────────────────────────────────────

class TestArrangeGridExtra:
    def test_empty(self):
        assert arrange_grid([]) == []

    def test_single(self):
        placements = arrange_grid([(50, 50)], cols=2, gap=0)
        assert len(placements) == 1
        assert placements[0].x == 0
        assert placements[0].y == 0

    def test_multiple_cols(self):
        sizes = [(30, 30), (30, 30), (30, 30), (30, 30)]
        placements = arrange_grid(sizes, cols=2, gap=0)
        assert len(placements) == 4
        # Row 0: items 0,1; Row 1: items 2,3
        assert placements[2].y > 0

    def test_with_gap(self):
        sizes = [(10, 10), (10, 10)]
        placements = arrange_grid(sizes, cols=2, gap=5)
        assert placements[1].x == 10 + 5  # width + gap

    def test_zero_cols_raises(self):
        with pytest.raises(ValueError):
            arrange_grid([(10, 10)], cols=0)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            arrange_grid([(10, 10)], gap=-1)


# ─── arrange_strip ──────────────────────────────────────────────────────────

class TestArrangeStripExtra:
    def test_empty(self):
        assert arrange_strip([], canvas_w=100) == []

    def test_single(self):
        placements = arrange_strip([(50, 50)], canvas_w=100, gap=0)
        assert len(placements) == 1
        assert placements[0].x == 0
        assert placements[0].y == 0

    def test_wraps_to_next_row(self):
        sizes = [(60, 30), (60, 30)]
        placements = arrange_strip(sizes, canvas_w=100, gap=0)
        # 60 fits, 60+60=120 > 100 → second wraps
        assert placements[1].y > 0

    def test_no_wrap_if_fits(self):
        sizes = [(40, 30), (40, 30)]
        placements = arrange_strip(sizes, canvas_w=100, gap=0)
        assert placements[1].y == 0

    def test_zero_canvas_w_raises(self):
        with pytest.raises(ValueError):
            arrange_strip([(10, 10)], canvas_w=0)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            arrange_strip([(10, 10)], canvas_w=100, gap=-1)


# ─── center_placements ─────────────────────────────────────────────────────

class TestCenterPlacementsExtra:
    def test_empty(self):
        assert center_placements([], 100, 100) == []

    def test_centers(self):
        p = FragmentPlacement(fragment_id=0, x=0, y=0, width=50, height=50)
        result = center_placements([p], canvas_w=200, canvas_h=200)
        assert len(result) == 1
        # Center of 50x50 in 200x200 → x=(200-50)//2=75
        assert result[0].x == 75
        assert result[0].y == 75

    def test_zero_canvas_w_raises(self):
        with pytest.raises(ValueError):
            center_placements([], canvas_w=0, canvas_h=100)

    def test_zero_canvas_h_raises(self):
        with pytest.raises(ValueError):
            center_placements([], canvas_w=100, canvas_h=0)

    def test_preserves_fragment_id(self):
        p = FragmentPlacement(fragment_id=5, x=0, y=0, width=10, height=10)
        result = center_placements([p], 100, 100)
        assert result[0].fragment_id == 5


# ─── group_bbox ─────────────────────────────────────────────────────────────

class TestGroupBboxExtra:
    def test_single(self):
        p = FragmentPlacement(fragment_id=0, x=10, y=20, width=30, height=40)
        bbox = group_bbox([p])
        assert bbox == (10, 20, 30, 40)

    def test_multiple(self):
        p1 = FragmentPlacement(fragment_id=0, x=0, y=0, width=50, height=50)
        p2 = FragmentPlacement(fragment_id=1, x=100, y=100, width=50, height=50)
        bbox = group_bbox([p1, p2])
        assert bbox == (0, 0, 150, 150)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            group_bbox([])


# ─── shift_placements ───────────────────────────────────────────────────────

class TestShiftPlacementsExtra:
    def test_positive_shift(self):
        p = FragmentPlacement(fragment_id=0, x=10, y=20, width=30, height=40)
        result = shift_placements([p], dx=5, dy=10)
        assert result[0].x == 15
        assert result[0].y == 30

    def test_zero_shift(self):
        p = FragmentPlacement(fragment_id=0, x=10, y=20, width=30, height=40)
        result = shift_placements([p], dx=0, dy=0)
        assert result[0].x == 10
        assert result[0].y == 20

    def test_negative_result_x_raises(self):
        p = FragmentPlacement(fragment_id=0, x=5, y=10, width=10, height=10)
        with pytest.raises(ValueError):
            shift_placements([p], dx=-10, dy=0)

    def test_negative_result_y_raises(self):
        p = FragmentPlacement(fragment_id=0, x=5, y=10, width=10, height=10)
        with pytest.raises(ValueError):
            shift_placements([p], dx=0, dy=-20)

    def test_empty(self):
        result = shift_placements([], dx=10, dy=10)
        assert result == []


# ─── arrange ────────────────────────────────────────────────────────────────

class TestArrangeExtra:
    def test_grid_strategy(self):
        params = ArrangementParams(strategy="grid", cols=2, gap=0)
        sizes = [(30, 30), (30, 30), (30, 30)]
        placements = arrange(sizes, params)
        assert len(placements) == 3

    def test_strip_strategy(self):
        params = ArrangementParams(strategy="strip", canvas_w=200)
        sizes = [(50, 50), (50, 50)]
        placements = arrange(sizes, params)
        assert len(placements) == 2

    def test_center_strategy(self):
        params = ArrangementParams(strategy="center", canvas_w=500, canvas_h=500)
        sizes = [(50, 50)]
        placements = arrange(sizes, params)
        assert len(placements) == 1
        # Should be approximately centered
        assert placements[0].x > 0

    def test_empty_sizes(self):
        params = ArrangementParams(strategy="grid")
        placements = arrange([], params)
        assert placements == []


# ─── batch_arrange ──────────────────────────────────────────────────────────

class TestBatchArrangeExtra:
    def test_empty(self):
        params = ArrangementParams()
        assert batch_arrange([], params) == []

    def test_length(self):
        params = ArrangementParams(strategy="grid", cols=2)
        size_lists = [
            [(30, 30), (30, 30)],
            [(40, 40)],
        ]
        results = batch_arrange(size_lists, params)
        assert len(results) == 2

    def test_result_type(self):
        params = ArrangementParams(strategy="strip", canvas_w=200)
        results = batch_arrange([[(50, 50)]], params)
        assert isinstance(results[0], list)
        assert isinstance(results[0][0], FragmentPlacement)
