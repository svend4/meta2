"""Тесты для puzzle_reconstruction.assembly.fragment_arranger."""
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


# ─── TestArrangementParams ────────────────────────────────────────────────────

class TestArrangementParams:
    def test_default_values(self):
        p = ArrangementParams()
        assert p.strategy == "strip"
        assert p.cols == 4
        assert p.gap == 4
        assert p.canvas_w == 512
        assert p.canvas_h == 512

    def test_valid_strategies(self):
        for s in ("grid", "strip", "center"):
            p = ArrangementParams(strategy=s)
            assert p.strategy == s

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(strategy="unknown")

    def test_cols_zero_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(cols=0)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(gap=-1)

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(canvas_w=0)

    def test_canvas_h_zero_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(canvas_h=0)


# ─── TestFragmentPlacement ────────────────────────────────────────────────────

class TestFragmentPlacement:
    def _make(self, **kw):
        defaults = dict(fragment_id=0, x=0, y=0, width=10, height=10)
        defaults.update(kw)
        return FragmentPlacement(**defaults)

    def test_basic_creation(self):
        p = self._make()
        assert p.fragment_id == 0
        assert p.width == 10

    def test_bbox_property(self):
        p = self._make(x=5, y=3, width=20, height=15)
        assert p.bbox == (5, 3, 20, 15)

    def test_center_property(self):
        p = self._make(x=0, y=0, width=10, height=10)
        assert p.center == (5.0, 5.0)

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            self._make(fragment_id=-1)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            self._make(x=-1)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            self._make(y=-1)

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            self._make(width=0)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            self._make(height=0)


# ─── TestArrangeGrid ──────────────────────────────────────────────────────────

class TestArrangeGrid:
    def _sizes(self, n=6):
        return [(32, 32)] * n

    def test_returns_list(self):
        result = arrange_grid(self._sizes(), cols=3, gap=4)
        assert isinstance(result, list)

    def test_correct_length(self):
        result = arrange_grid(self._sizes(6), cols=3, gap=4)
        assert len(result) == 6

    def test_empty_input(self):
        assert arrange_grid([], cols=3, gap=4) == []

    def test_fragment_ids_sequential(self):
        result = arrange_grid(self._sizes(4), cols=2, gap=0)
        ids = [p.fragment_id for p in result]
        assert ids == [0, 1, 2, 3]

    def test_positions_nonnegative(self):
        result = arrange_grid(self._sizes(6), cols=3, gap=4)
        for p in result:
            assert p.x >= 0
            assert p.y >= 0

    def test_cols_zero_raises(self):
        with pytest.raises(ValueError):
            arrange_grid(self._sizes(), cols=0, gap=4)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            arrange_grid(self._sizes(), cols=3, gap=-1)

    def test_single_col(self):
        result = arrange_grid([(10, 20), (10, 30)], cols=1, gap=0)
        assert result[0].y == 0
        assert result[1].y == 20


# ─── TestArrangeStrip ─────────────────────────────────────────────────────────

class TestArrangeStrip:
    def test_returns_list(self):
        result = arrange_strip([(30, 20)] * 4, canvas_w=100, gap=4)
        assert isinstance(result, list)

    def test_correct_length(self):
        result = arrange_strip([(30, 20)] * 5, canvas_w=200, gap=4)
        assert len(result) == 5

    def test_empty_input(self):
        assert arrange_strip([], canvas_w=100, gap=4) == []

    def test_positions_nonnegative(self):
        result = arrange_strip([(40, 20)] * 6, canvas_w=100, gap=4)
        for p in result:
            assert p.x >= 0
            assert p.y >= 0

    def test_wraps_to_next_row(self):
        # Ширина холста 50, фрагменты 30px — второй на строку не влезет
        result = arrange_strip([(30, 20), (30, 20)], canvas_w=50, gap=0)
        assert result[0].y == 0
        assert result[1].y == 20  # перенос

    def test_canvas_w_zero_raises(self):
        with pytest.raises(ValueError):
            arrange_strip([(10, 10)], canvas_w=0, gap=4)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError):
            arrange_strip([(10, 10)], canvas_w=100, gap=-1)


# ─── TestCenterPlacements ─────────────────────────────────────────────────────

class TestCenterPlacements:
    def test_returns_list(self):
        placements = arrange_strip([(20, 20)] * 2, canvas_w=200)
        result = center_placements(placements, canvas_w=200, canvas_h=200)
        assert isinstance(result, list)

    def test_same_length(self):
        placements = arrange_strip([(20, 20)] * 3, canvas_w=200)
        result = center_placements(placements, canvas_w=200, canvas_h=200)
        assert len(result) == len(placements)

    def test_positions_nonnegative(self):
        placements = arrange_strip([(20, 20)] * 3, canvas_w=200)
        result = center_placements(placements, canvas_w=200, canvas_h=200)
        for p in result:
            assert p.x >= 0
            assert p.y >= 0

    def test_empty_list(self):
        assert center_placements([], canvas_w=100, canvas_h=100) == []

    def test_canvas_w_zero_raises(self):
        p = [FragmentPlacement(fragment_id=0, x=0, y=0, width=10, height=10)]
        with pytest.raises(ValueError):
            center_placements(p, canvas_w=0, canvas_h=100)

    def test_canvas_h_zero_raises(self):
        p = [FragmentPlacement(fragment_id=0, x=0, y=0, width=10, height=10)]
        with pytest.raises(ValueError):
            center_placements(p, canvas_w=100, canvas_h=0)


# ─── TestGroupBbox ────────────────────────────────────────────────────────────

class TestGroupBbox:
    def test_single_fragment(self):
        p = [FragmentPlacement(fragment_id=0, x=5, y=3, width=10, height=8)]
        bbox = group_bbox(p)
        assert bbox == (5, 3, 10, 8)

    def test_two_fragments(self):
        placements = [
            FragmentPlacement(fragment_id=0, x=0, y=0, width=10, height=10),
            FragmentPlacement(fragment_id=1, x=20, y=15, width=5, height=5),
        ]
        bbox = group_bbox(placements)
        assert bbox[0] == 0
        assert bbox[1] == 0
        assert bbox[2] == 25  # 0+10 .. 20+5 = 25 width
        assert bbox[3] == 20  # 0+10 .. 15+5 = 20 height

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            group_bbox([])


# ─── TestShiftPlacements ──────────────────────────────────────────────────────

class TestShiftPlacements:
    def _placements(self):
        return [
            FragmentPlacement(fragment_id=0, x=10, y=10, width=5, height=5),
            FragmentPlacement(fragment_id=1, x=20, y=15, width=5, height=5),
        ]

    def test_shift_positive(self):
        shifted = shift_placements(self._placements(), dx=5, dy=3)
        assert shifted[0].x == 15
        assert shifted[0].y == 13

    def test_shift_zero(self):
        placements = self._placements()
        shifted = shift_placements(placements, dx=0, dy=0)
        for orig, s in zip(placements, shifted):
            assert orig.x == s.x and orig.y == s.y

    def test_negative_shift_valid(self):
        shifted = shift_placements(self._placements(), dx=-5, dy=-5)
        assert shifted[0].x == 5
        assert shifted[0].y == 5

    def test_x_negative_after_shift_raises(self):
        with pytest.raises(ValueError):
            shift_placements(self._placements(), dx=-20, dy=0)

    def test_y_negative_after_shift_raises(self):
        with pytest.raises(ValueError):
            shift_placements(self._placements(), dx=0, dy=-20)

    def test_empty_list(self):
        assert shift_placements([], dx=10, dy=10) == []


# ─── TestArrange ──────────────────────────────────────────────────────────────

class TestArrange:
    def _sizes(self):
        return [(30, 20)] * 4

    def test_grid_strategy(self):
        params = ArrangementParams(strategy="grid", cols=2, gap=4)
        result = arrange(self._sizes(), params)
        assert len(result) == 4

    def test_strip_strategy(self):
        params = ArrangementParams(strategy="strip", canvas_w=200, gap=4)
        result = arrange(self._sizes(), params)
        assert len(result) == 4

    def test_center_strategy(self):
        params = ArrangementParams(strategy="center", canvas_w=200, canvas_h=200, gap=4)
        result = arrange(self._sizes(), params)
        assert len(result) == 4
        for p in result:
            assert p.x >= 0 and p.y >= 0


# ─── TestBatchArrange ─────────────────────────────────────────────────────────

class TestBatchArrange:
    def test_returns_list_of_lists(self):
        params = ArrangementParams(strategy="strip", canvas_w=200, gap=4)
        size_lists = [[(30, 20)] * 3, [(40, 25)] * 2]
        result = batch_arrange(size_lists, params)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_inner_length(self):
        params = ArrangementParams(strategy="grid", cols=2)
        size_lists = [[(10, 10)] * 4, [(20, 20)] * 6]
        result = batch_arrange(size_lists, params)
        assert len(result[0]) == 4
        assert len(result[1]) == 6

    def test_empty_list(self):
        params = ArrangementParams()
        assert batch_arrange([], params) == []
