"""Extra tests for puzzle_reconstruction.assembly.fragment_arranger."""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.fragment_arranger import (
    ArrangementParams,
    FragmentPlacement,
    arrange,
    arrange_grid,
    arrange_strip,
    batch_arrange,
    center_placements,
    group_bbox,
    shift_placements,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fp(fid=0, x=0, y=0, w=10, h=10):
    return FragmentPlacement(fragment_id=fid, x=x, y=y, width=w, height=h)


def _sizes(n=4, w=30, h=20):
    return [(w, h)] * n


# ─── ArrangementParams extras ─────────────────────────────────────────────────

class TestArrangementParamsExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(ArrangementParams()), str)

    def test_gap_zero_valid(self):
        p = ArrangementParams(gap=0)
        assert p.gap == 0

    def test_cols_one_valid(self):
        p = ArrangementParams(cols=1)
        assert p.cols == 1

    def test_large_canvas(self):
        p = ArrangementParams(canvas_w=9999, canvas_h=9999)
        assert p.canvas_w == 9999

    def test_all_strategies_stored(self):
        for s in ("grid", "strip", "center"):
            assert ArrangementParams(strategy=s).strategy == s

    def test_cols_negative_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(cols=-1)

    def test_canvas_w_negative_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(canvas_w=-1)

    def test_canvas_h_negative_raises(self):
        with pytest.raises(ValueError):
            ArrangementParams(canvas_h=-1)


# ─── FragmentPlacement extras ─────────────────────────────────────────────────

class TestFragmentPlacementExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(_fp()), str)

    def test_center_offset_fragment(self):
        p = _fp(x=10, y=20, w=40, h=60)
        cx, cy = p.center
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(50.0)

    def test_bbox_is_tuple(self):
        p = _fp(x=1, y=2, w=3, h=4)
        assert isinstance(p.bbox, tuple)
        assert len(p.bbox) == 4

    def test_bbox_values_correct(self):
        p = _fp(x=7, y=3, w=12, h=8)
        assert p.bbox == (7, 3, 12, 8)

    def test_fragment_id_zero_valid(self):
        p = _fp(fid=0)
        assert p.fragment_id == 0

    def test_large_fragment_id(self):
        p = _fp(fid=9999)
        assert p.fragment_id == 9999

    def test_width_one_valid(self):
        p = _fp(w=1)
        assert p.width == 1

    def test_height_one_valid(self):
        p = _fp(h=1)
        assert p.height == 1


# ─── arrange_grid extras ──────────────────────────────────────────────────────

class TestArrangeGridExtra:
    def test_single_fragment(self):
        result = arrange_grid([(20, 15)], cols=1, gap=0)
        assert len(result) == 1
        assert result[0].fragment_id == 0
        assert result[0].x == 0
        assert result[0].y == 0

    def test_one_col_n_rows(self):
        result = arrange_grid([(10, 20)] * 5, cols=1, gap=0)
        assert len(result) == 5
        for i, p in enumerate(result):
            assert p.y == i * 20

    def test_gap_zero_adjacent(self):
        result = arrange_grid([(10, 10), (10, 10)], cols=2, gap=0)
        assert result[1].x == 10

    def test_all_positions_unique(self):
        result = arrange_grid(_sizes(9), cols=3, gap=0)
        coords = [(p.x, p.y) for p in result]
        assert len(set(coords)) == 9

    def test_fragment_ids_all_present(self):
        n = 8
        result = arrange_grid(_sizes(n), cols=4, gap=2)
        assert sorted(p.fragment_id for p in result) == list(range(n))

    def test_mixed_sizes(self):
        sizes = [(10, 20), (30, 15), (20, 25)]
        result = arrange_grid(sizes, cols=2, gap=0)
        assert len(result) == 3

    def test_all_fragment_placements(self):
        result = arrange_grid(_sizes(6), cols=3, gap=4)
        for r in result:
            assert isinstance(r, FragmentPlacement)

    def test_gap_applied_x(self):
        result = arrange_grid([(10, 10)] * 2, cols=2, gap=5)
        assert result[1].x == 15  # 10 + 5 gap


# ─── arrange_strip extras ─────────────────────────────────────────────────────

class TestArrangeStripExtra:
    def test_single_fragment(self):
        result = arrange_strip([(20, 15)], canvas_w=100, gap=0)
        assert len(result) == 1
        assert result[0].x == 0
        assert result[0].y == 0

    def test_all_fit_one_row(self):
        # 3 fragments of 10px each + 0 gap → 30px, canvas 100
        result = arrange_strip([(10, 10)] * 3, canvas_w=100, gap=0)
        assert all(p.y == 0 for p in result)

    def test_very_wide_canvas_no_wrap(self):
        result = arrange_strip([(20, 10)] * 5, canvas_w=10000, gap=4)
        assert all(p.y == 0 for p in result)

    def test_wrap_positions_nonnegative(self):
        result = arrange_strip([(40, 20)] * 4, canvas_w=50, gap=0)
        for p in result:
            assert p.x >= 0
            assert p.y >= 0

    def test_fragment_ids_sequential(self):
        result = arrange_strip([(20, 10)] * 4, canvas_w=100, gap=0)
        assert [p.fragment_id for p in result] == [0, 1, 2, 3]

    def test_gap_increases_x_step(self):
        result = arrange_strip([(10, 10)] * 2, canvas_w=200, gap=5)
        assert result[1].x == 15  # 10 + 5

    def test_different_heights_wrap_row_height(self):
        # First row: two heights 10 and 20, second row starts after max(10,20)=20
        result = arrange_strip([(100, 10), (100, 20), (100, 5)],
                                canvas_w=150, gap=0)
        # First fits (100 < 150), second wraps (100+100=200 > 150)
        assert result[1].y > 0


# ─── center_placements extras ─────────────────────────────────────────────────

class TestCenterPlacementsExtra:
    def test_single_fragment_nonneg(self):
        p = [_fp(x=0, y=0, w=20, h=20)]
        result = center_placements(p, canvas_w=200, canvas_h=200)
        assert result[0].x >= 0
        assert result[0].y >= 0

    def test_very_large_canvas(self):
        p = [_fp(x=0, y=0, w=10, h=10)]
        result = center_placements(p, canvas_w=1000, canvas_h=1000)
        assert result[0].x >= 0

    def test_length_preserved(self):
        placements = arrange_grid([(20, 20)] * 5, cols=2, gap=4)
        result = center_placements(placements, canvas_w=300, canvas_h=300)
        assert len(result) == 5

    def test_returns_fragment_placements(self):
        p = [_fp()]
        result = center_placements(p, canvas_w=100, canvas_h=100)
        assert all(isinstance(r, FragmentPlacement) for r in result)

    def test_ids_preserved(self):
        placements = [_fp(fid=i) for i in range(4)]
        result = center_placements(placements, canvas_w=200, canvas_h=200)
        assert [r.fragment_id for r in result] == [0, 1, 2, 3]


# ─── group_bbox extras ────────────────────────────────────────────────────────

class TestGroupBboxExtra:
    def test_three_fragments_bbox_x(self):
        placements = [
            _fp(fid=0, x=0, y=0, w=10, h=10),
            _fp(fid=1, x=15, y=0, w=10, h=10),
            _fp(fid=2, x=30, y=0, w=10, h=10),
        ]
        bbox = group_bbox(placements)
        assert bbox[0] == 0   # min x
        assert bbox[2] == 40  # total width (0..40)

    def test_three_fragments_bbox_y(self):
        placements = [
            _fp(fid=0, x=0, y=5, w=10, h=10),
            _fp(fid=1, x=0, y=20, w=10, h=10),
        ]
        bbox = group_bbox(placements)
        assert bbox[1] == 5
        assert bbox[3] == 25  # total height (5..30)

    def test_returns_tuple(self):
        p = [_fp()]
        assert isinstance(group_bbox(p), tuple)

    def test_single_fragment_bbox(self):
        p = [_fp(x=3, y=7, w=20, h=30)]
        bbox = group_bbox(p)
        assert bbox == (3, 7, 20, 30)

    def test_coincident_fragments(self):
        p = [_fp(x=5, y=5, w=10, h=10), _fp(fid=1, x=5, y=5, w=10, h=10)]
        bbox = group_bbox(p)
        assert bbox == (5, 5, 10, 10)


# ─── shift_placements extras ──────────────────────────────────────────────────

class TestShiftPlacementsExtra:
    def test_all_shifted_equally(self):
        placements = [_fp(fid=i, x=i * 10, y=i * 10) for i in range(4)]
        shifted = shift_placements(placements, dx=3, dy=2)
        for orig, s in zip(placements, shifted):
            assert s.x == orig.x + 3
            assert s.y == orig.y + 2

    def test_ids_preserved_after_shift(self):
        placements = [_fp(fid=i) for i in range(3)]
        shifted = shift_placements(placements, dx=1, dy=1)
        assert [s.fragment_id for s in shifted] == [0, 1, 2]

    def test_sizes_preserved_after_shift(self):
        placements = [_fp(w=15, h=25)]
        shifted = shift_placements(placements, dx=5, dy=3)
        assert shifted[0].width == 15
        assert shifted[0].height == 25

    def test_returns_new_list(self):
        placements = [_fp()]
        shifted = shift_placements(placements, dx=0, dy=0)
        assert shifted is not placements

    def test_large_positive_shift(self):
        placements = [_fp(x=0, y=0)]
        shifted = shift_placements(placements, dx=100, dy=100)
        assert shifted[0].x == 100
        assert shifted[0].y == 100


# ─── arrange extras ───────────────────────────────────────────────────────────

class TestArrangeExtra:
    def test_empty_sizes_returns_empty(self):
        params = ArrangementParams(strategy="grid", cols=2)
        assert arrange([], params) == []

    def test_single_fragment_grid(self):
        params = ArrangementParams(strategy="grid", cols=1)
        result = arrange([(20, 15)], params)
        assert len(result) == 1

    def test_fragment_ids_sequential_strip(self):
        params = ArrangementParams(strategy="strip", canvas_w=200)
        result = arrange(_sizes(5), params)
        assert [p.fragment_id for p in result] == list(range(5))

    def test_center_positions_nonneg(self):
        params = ArrangementParams(strategy="center", canvas_w=200, canvas_h=200)
        for p in arrange(_sizes(6), params):
            assert p.x >= 0
            assert p.y >= 0

    def test_returns_list(self):
        params = ArrangementParams(strategy="grid", cols=2)
        assert isinstance(arrange(_sizes(4), params), list)


# ─── batch_arrange extras ─────────────────────────────────────────────────────

class TestBatchArrangeExtra:
    def test_single_size_list(self):
        params = ArrangementParams(strategy="strip", canvas_w=200)
        result = batch_arrange([_sizes(3)], params)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_five_size_lists(self):
        params = ArrangementParams(strategy="grid", cols=2)
        size_lists = [_sizes(i + 2, w=10, h=10) for i in range(5)]
        result = batch_arrange(size_lists, params)
        assert len(result) == 5
        for i, r in enumerate(result):
            assert len(r) == i + 2

    def test_all_fragment_placements(self):
        params = ArrangementParams(strategy="strip", canvas_w=200)
        size_lists = [_sizes(4)]
        result = batch_arrange(size_lists, params)
        for r in result[0]:
            assert isinstance(r, FragmentPlacement)

    def test_single_fragment_in_each_list(self):
        params = ArrangementParams(strategy="grid", cols=1)
        size_lists = [[(10, 10)]] * 4
        result = batch_arrange(size_lists, params)
        assert all(len(r) == 1 for r in result)
