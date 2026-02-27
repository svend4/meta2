"""Tests for puzzle_reconstruction/assembly/fragment_arranger.py"""
import pytest
import numpy as np

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


# ── ArrangementParams ─────────────────────────────────────────────────────────

class TestArrangementParams:
    def test_default_values(self):
        p = ArrangementParams()
        assert p.strategy == "strip"
        assert p.cols == 4
        assert p.gap == 4
        assert p.canvas_w == 512
        assert p.canvas_h == 512

    def test_valid_grid_strategy(self):
        p = ArrangementParams(strategy="grid")
        assert p.strategy == "grid"

    def test_valid_strip_strategy(self):
        p = ArrangementParams(strategy="strip")
        assert p.strategy == "strip"

    def test_valid_center_strategy(self):
        p = ArrangementParams(strategy="center")
        assert p.strategy == "center"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            ArrangementParams(strategy="random")

    def test_cols_less_than_1_raises(self):
        with pytest.raises(ValueError, match="cols должен быть >= 1"):
            ArrangementParams(cols=0)

    def test_gap_negative_raises(self):
        with pytest.raises(ValueError, match="gap должен быть >= 0"):
            ArrangementParams(gap=-1)

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError, match="canvas_w должен быть >= 1"):
            ArrangementParams(canvas_w=0)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError, match="canvas_h должен быть >= 1"):
            ArrangementParams(canvas_h=0)

    def test_valid_custom_params(self):
        p = ArrangementParams(strategy="grid", cols=3, gap=10, canvas_w=800, canvas_h=600)
        assert p.cols == 3
        assert p.gap == 10


# ── FragmentPlacement ─────────────────────────────────────────────────────────

class TestFragmentPlacement:
    def test_valid_construction(self):
        fp = FragmentPlacement(fragment_id=0, x=10, y=20, width=50, height=60)
        assert fp.fragment_id == 0
        assert fp.x == 10
        assert fp.y == 20

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError, match="fragment_id должен быть >= 0"):
            FragmentPlacement(fragment_id=-1, x=0, y=0, width=10, height=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x должен быть >= 0"):
            FragmentPlacement(fragment_id=0, x=-1, y=0, width=10, height=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y должен быть >= 0"):
            FragmentPlacement(fragment_id=0, x=0, y=-1, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="width должен быть >= 1"):
            FragmentPlacement(fragment_id=0, x=0, y=0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height должен быть >= 1"):
            FragmentPlacement(fragment_id=0, x=0, y=0, width=10, height=0)

    def test_bbox_property(self):
        fp = FragmentPlacement(fragment_id=0, x=5, y=10, width=20, height=30)
        assert fp.bbox == (5, 10, 20, 30)

    def test_center_property(self):
        fp = FragmentPlacement(fragment_id=0, x=0, y=0, width=10, height=20)
        cx, cy = fp.center
        assert cx == 5.0
        assert cy == 10.0


# ── arrange_grid ──────────────────────────────────────────────────────────────

class TestArrangeGrid:
    def test_empty_sizes_returns_empty(self):
        assert arrange_grid([], cols=4, gap=0) == []

    def test_single_item(self):
        result = arrange_grid([(50, 30)], cols=4, gap=0)
        assert len(result) == 1
        assert result[0].x == 0
        assert result[0].y == 0

    def test_correct_count(self):
        sizes = [(50, 50)] * 6
        result = arrange_grid(sizes, cols=3, gap=0)
        assert len(result) == 6

    def test_fragment_ids_sequential(self):
        sizes = [(50, 50)] * 4
        result = arrange_grid(sizes, cols=2, gap=0)
        ids = [r.fragment_id for r in result]
        assert ids == [0, 1, 2, 3]

    def test_cols_1_gives_single_column(self):
        sizes = [(50, 30), (50, 30), (50, 30)]
        result = arrange_grid(sizes, cols=1, gap=0)
        xs = [r.x for r in result]
        assert all(x == 0 for x in xs)

    def test_gap_applied(self):
        sizes = [(50, 50), (50, 50)]
        result = arrange_grid(sizes, cols=2, gap=10)
        # Second item should start at 50 + 10 = 60
        assert result[1].x == 60

    def test_cols_less_than_1_raises(self):
        with pytest.raises(ValueError):
            arrange_grid([(10, 10)], cols=0, gap=0)

    def test_gap_negative_raises(self):
        with pytest.raises(ValueError):
            arrange_grid([(10, 10)], cols=1, gap=-1)

    def test_widths_heights_correct(self):
        sizes = [(30, 40)]
        result = arrange_grid(sizes)
        assert result[0].width == 30
        assert result[0].height == 40

    def test_row_wrapping(self):
        sizes = [(50, 50)] * 5
        result = arrange_grid(sizes, cols=3, gap=0)
        # 4th item (idx=3) should be in row 1
        assert result[3].y > 0


# ── arrange_strip ─────────────────────────────────────────────────────────────

class TestArrangeStrip:
    def test_empty_sizes_returns_empty(self):
        assert arrange_strip([], canvas_w=512, gap=0) == []

    def test_single_item_at_origin(self):
        result = arrange_strip([(50, 50)], canvas_w=512, gap=0)
        assert result[0].x == 0
        assert result[0].y == 0

    def test_two_items_fit_in_row(self):
        result = arrange_strip([(50, 50), (50, 50)], canvas_w=200, gap=0)
        assert result[1].x == 50
        assert result[1].y == 0

    def test_item_wraps_to_new_row(self):
        result = arrange_strip([(100, 50), (100, 50)], canvas_w=150, gap=0)
        assert result[1].y > 0
        assert result[1].x == 0

    def test_gap_between_items(self):
        result = arrange_strip([(50, 50), (50, 50)], canvas_w=500, gap=10)
        assert result[1].x == 60

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            arrange_strip([(10, 10)], canvas_w=0, gap=0)

    def test_gap_negative_raises(self):
        with pytest.raises(ValueError):
            arrange_strip([(10, 10)], canvas_w=100, gap=-1)

    def test_count_preserved(self):
        sizes = [(30, 20)] * 10
        result = arrange_strip(sizes, canvas_w=200, gap=5)
        assert len(result) == 10

    def test_fragment_ids_sequential(self):
        sizes = [(30, 20)] * 3
        result = arrange_strip(sizes, canvas_w=200, gap=0)
        assert [r.fragment_id for r in result] == [0, 1, 2]


# ── center_placements ─────────────────────────────────────────────────────────

class TestCenterPlacements:
    def _make_placement(self, fid, x, y, w=50, h=50):
        return FragmentPlacement(fragment_id=fid, x=x, y=y, width=w, height=h)

    def test_empty_returns_empty(self):
        assert center_placements([], 512, 512) == []

    def test_canvas_w_zero_raises(self):
        p = self._make_placement(0, 0, 0)
        with pytest.raises(ValueError):
            center_placements([p], canvas_w=0, canvas_h=512)

    def test_canvas_h_zero_raises(self):
        p = self._make_placement(0, 0, 0)
        with pytest.raises(ValueError):
            center_placements([p], canvas_w=512, canvas_h=0)

    def test_single_placement_centered(self):
        p = self._make_placement(0, 0, 0, w=100, h=100)
        result = center_placements([p], canvas_w=500, canvas_h=500)
        # Group (100x100) centered in 500x500 -> offset = (500-100)//2 = 200
        assert result[0].x == 200
        assert result[0].y == 200

    def test_positions_non_negative(self):
        p = self._make_placement(0, 0, 0, w=600, h=600)
        result = center_placements([p], canvas_w=512, canvas_h=512)
        for r in result:
            assert r.x >= 0
            assert r.y >= 0

    def test_fragment_ids_preserved(self):
        ps = [self._make_placement(i, i * 60, 0) for i in range(3)]
        result = center_placements(ps, canvas_w=512, canvas_h=512)
        assert [r.fragment_id for r in result] == [0, 1, 2]

    def test_dimensions_unchanged(self):
        p = self._make_placement(0, 0, 0, w=80, h=90)
        result = center_placements([p], canvas_w=512, canvas_h=512)
        assert result[0].width == 80
        assert result[0].height == 90


# ── group_bbox ────────────────────────────────────────────────────────────────

class TestGroupBbox:
    def _make_placement(self, fid, x, y, w=50, h=50):
        return FragmentPlacement(fragment_id=fid, x=x, y=y, width=w, height=h)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            group_bbox([])

    def test_single_placement(self):
        p = self._make_placement(0, 10, 20, w=50, h=60)
        x, y, w, h = group_bbox([p])
        assert x == 10
        assert y == 20
        assert w == 50
        assert h == 60

    def test_two_placements(self):
        p1 = self._make_placement(0, 0, 0, w=50, h=50)
        p2 = self._make_placement(1, 100, 100, w=50, h=50)
        x, y, w, h = group_bbox([p1, p2])
        assert x == 0
        assert y == 0
        assert w == 150
        assert h == 150

    def test_returns_tuple_of_4(self):
        p = self._make_placement(0, 0, 0)
        result = group_bbox([p])
        assert len(result) == 4


# ── shift_placements ──────────────────────────────────────────────────────────

class TestShiftPlacements:
    def _make_placement(self, fid, x, y):
        return FragmentPlacement(fragment_id=fid, x=x, y=y, width=50, height=50)

    def test_basic_shift(self):
        p = self._make_placement(0, 10, 20)
        result = shift_placements([p], dx=5, dy=10)
        assert result[0].x == 15
        assert result[0].y == 30

    def test_zero_shift(self):
        p = self._make_placement(0, 10, 20)
        result = shift_placements([p], dx=0, dy=0)
        assert result[0].x == 10
        assert result[0].y == 20

    def test_negative_shift_raises(self):
        p = self._make_placement(0, 5, 5)
        with pytest.raises(ValueError):
            shift_placements([p], dx=-10, dy=0)

    def test_negative_dy_raises(self):
        p = self._make_placement(0, 5, 5)
        with pytest.raises(ValueError):
            shift_placements([p], dx=0, dy=-10)

    def test_empty_returns_empty(self):
        assert shift_placements([], dx=5, dy=5) == []

    def test_multiple_placements_shifted(self):
        ps = [self._make_placement(i, i * 10, 0) for i in range(3)]
        result = shift_placements(ps, dx=5, dy=3)
        for i, r in enumerate(result):
            assert r.x == i * 10 + 5
            assert r.y == 3

    def test_new_list_not_mutating(self):
        p = self._make_placement(0, 10, 10)
        result = shift_placements([p], dx=5, dy=5)
        assert result[0] is not p


# ── arrange ───────────────────────────────────────────────────────────────────

class TestArrange:
    def test_grid_strategy(self):
        sizes = [(50, 50)] * 4
        params = ArrangementParams(strategy="grid", cols=2)
        result = arrange(sizes, params)
        assert len(result) == 4

    def test_strip_strategy(self):
        sizes = [(50, 50)] * 4
        params = ArrangementParams(strategy="strip", canvas_w=300)
        result = arrange(sizes, params)
        assert len(result) == 4

    def test_center_strategy(self):
        sizes = [(50, 50)] * 4
        params = ArrangementParams(strategy="center", canvas_w=512, canvas_h=512)
        result = arrange(sizes, params)
        assert len(result) == 4

    def test_empty_sizes(self):
        params = ArrangementParams(strategy="grid")
        result = arrange([], params)
        assert result == []


# ── batch_arrange ─────────────────────────────────────────────────────────────

class TestBatchArrange:
    def test_output_length(self):
        size_lists = [[(50, 50)] * 3, [(30, 40)] * 2]
        params = ArrangementParams(strategy="grid")
        result = batch_arrange(size_lists, params)
        assert len(result) == 2

    def test_each_element_is_list(self):
        size_lists = [[(50, 50)] * 3]
        params = ArrangementParams(strategy="strip", canvas_w=512)
        result = batch_arrange(size_lists, params)
        assert isinstance(result[0], list)

    def test_empty_size_lists(self):
        params = ArrangementParams(strategy="grid")
        result = batch_arrange([], params)
        assert result == []
