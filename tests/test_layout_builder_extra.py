"""Extra tests for puzzle_reconstruction/assembly/layout_builder.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.layout_builder import (
    AssemblyLayout,
    LayoutCell,
    add_cell,
    compute_bounding_box,
    create_layout,
    dict_to_layout,
    layout_to_dict,
    remove_cell,
    render_layout_image,
    snap_to_grid,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cell(idx: int = 0, x: float = 0.0, y: float = 0.0,
          w: float = 50.0, h: float = 50.0, rot: float = 0.0) -> LayoutCell:
    return LayoutCell(fragment_idx=idx, x=x, y=y, width=w, height=h, rotation=rot)


def _layout_with_cells(n: int = 3, gap: float = 10.0) -> AssemblyLayout:
    layout = create_layout()
    for i in range(n):
        add_cell(layout, i, x=float(i) * (50.0 + gap), y=0.0, width=50.0, height=50.0)
    return layout


# ─── LayoutCell (extra) ───────────────────────────────────────────────────────

class TestLayoutCellExtra:
    def test_default_rotation_zero(self):
        c = _cell()
        assert c.rotation == pytest.approx(0.0)

    def test_default_meta_empty(self):
        c = _cell()
        assert c.meta == {}

    def test_custom_rotation(self):
        c = _cell(rot=45.0)
        assert c.rotation == pytest.approx(45.0)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=0.0, height=50.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=-1.0, height=50.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=50.0, height=0.0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=50.0, height=-1.0)

    def test_negative_position_valid(self):
        c = LayoutCell(fragment_idx=0, x=-10.0, y=-20.0, width=50.0, height=50.0)
        assert c.x == pytest.approx(-10.0)
        assert c.y == pytest.approx(-20.0)

    def test_meta_stored(self):
        c = LayoutCell(fragment_idx=0, x=0, y=0, width=50.0, height=50.0,
                       meta={"score": 0.9})
        assert c.meta["score"] == pytest.approx(0.9)

    def test_fragment_idx_stored(self):
        c = _cell(idx=42)
        assert c.fragment_idx == 42

    def test_small_size_valid(self):
        c = LayoutCell(fragment_idx=0, x=0, y=0, width=0.001, height=0.001)
        assert c.width == pytest.approx(0.001)


# ─── AssemblyLayout (extra) ───────────────────────────────────────────────────

class TestAssemblyLayoutExtra:
    def test_default_cells_empty(self):
        layout = AssemblyLayout()
        assert layout.cells == []

    def test_default_canvas_zero(self):
        layout = AssemblyLayout()
        assert layout.canvas_w == pytest.approx(0.0)
        assert layout.canvas_h == pytest.approx(0.0)

    def test_default_params_empty(self):
        layout = AssemblyLayout()
        assert layout.params == {}

    def test_custom_canvas(self):
        layout = AssemblyLayout(canvas_w=800.0, canvas_h=600.0)
        assert layout.canvas_w == pytest.approx(800.0)
        assert layout.canvas_h == pytest.approx(600.0)

    def test_params_stored(self):
        layout = AssemblyLayout(params={"author": "test"})
        assert layout.params["author"] == "test"


# ─── create_layout (extra) ────────────────────────────────────────────────────

class TestCreateLayoutExtra:
    def test_returns_assembly_layout(self):
        assert isinstance(create_layout(), AssemblyLayout)

    def test_empty_cells(self):
        assert create_layout().cells == []

    def test_canvas_stored(self):
        layout = create_layout(canvas_w=640.0, canvas_h=480.0)
        assert layout.canvas_w == pytest.approx(640.0)
        assert layout.canvas_h == pytest.approx(480.0)

    def test_default_canvas_zero(self):
        layout = create_layout()
        assert layout.canvas_w == pytest.approx(0.0)

    def test_params_kwargs_stored(self):
        layout = create_layout(scale=2.0, mode="auto")
        assert layout.params["scale"] == pytest.approx(2.0)
        assert layout.params["mode"] == "auto"


# ─── add_cell (extra) ─────────────────────────────────────────────────────────

class TestAddCellExtra:
    def test_returns_layout(self):
        layout = create_layout()
        result = add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        assert result is layout

    def test_cell_added(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        assert len(layout.cells) == 1

    def test_multiple_cells_added(self):
        layout = create_layout()
        for i in range(5):
            add_cell(layout, i, float(i * 60), 0.0, 50.0, 50.0)
        assert len(layout.cells) == 5

    def test_duplicate_fragment_idx_replaced(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        add_cell(layout, 0, 100.0, 100.0, 60.0, 60.0)
        assert len(layout.cells) == 1
        assert layout.cells[0].x == pytest.approx(100.0)

    def test_zero_width_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            add_cell(layout, 0, 0.0, 0.0, 0.0, 50.0)

    def test_zero_height_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            add_cell(layout, 0, 0.0, 0.0, 50.0, 0.0)

    def test_rotation_stored(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0, rotation=90.0)
        assert layout.cells[0].rotation == pytest.approx(90.0)

    def test_meta_kwargs_stored(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0, score=0.9)
        assert layout.cells[0].meta["score"] == pytest.approx(0.9)


# ─── remove_cell (extra) ──────────────────────────────────────────────────────

class TestRemoveCellExtra:
    def test_returns_layout(self):
        layout = _layout_with_cells(3)
        result = remove_cell(layout, 0)
        assert result is layout

    def test_cell_removed(self):
        layout = _layout_with_cells(3)
        remove_cell(layout, 1)
        idxs = [c.fragment_idx for c in layout.cells]
        assert 1 not in idxs

    def test_other_cells_preserved(self):
        layout = _layout_with_cells(3)
        remove_cell(layout, 1)
        assert len(layout.cells) == 2

    def test_nonexistent_idx_noop(self):
        layout = _layout_with_cells(3)
        remove_cell(layout, 99)
        assert len(layout.cells) == 3

    def test_empty_layout_noop(self):
        layout = create_layout()
        remove_cell(layout, 0)
        assert len(layout.cells) == 0

    def test_remove_all_cells(self):
        layout = _layout_with_cells(3)
        for i in range(3):
            remove_cell(layout, i)
        assert len(layout.cells) == 0


# ─── compute_bounding_box (extra) ─────────────────────────────────────────────

class TestComputeBoundingBoxExtra:
    def test_empty_layout_returns_zeros(self):
        layout = create_layout()
        assert compute_bounding_box(layout) == (0.0, 0.0, 0.0, 0.0)

    def test_single_cell(self):
        layout = create_layout()
        add_cell(layout, 0, 10.0, 20.0, 50.0, 30.0)
        bb = compute_bounding_box(layout)
        assert bb[0] == pytest.approx(10.0)
        assert bb[1] == pytest.approx(20.0)
        assert bb[2] == pytest.approx(50.0)
        assert bb[3] == pytest.approx(30.0)

    def test_two_cells_side_by_side(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        add_cell(layout, 1, 50.0, 0.0, 50.0, 50.0)
        bb = compute_bounding_box(layout)
        assert bb[2] == pytest.approx(100.0)  # width
        assert bb[3] == pytest.approx(50.0)   # height

    def test_returns_4_tuple(self):
        layout = _layout_with_cells(2)
        bb = compute_bounding_box(layout)
        assert len(bb) == 4

    def test_negative_coordinates(self):
        layout = create_layout()
        add_cell(layout, 0, -20.0, -10.0, 40.0, 30.0)
        bb = compute_bounding_box(layout)
        assert bb[0] == pytest.approx(-20.0)
        assert bb[1] == pytest.approx(-10.0)

    def test_stacked_cells_height(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        add_cell(layout, 1, 0.0, 50.0, 50.0, 50.0)
        bb = compute_bounding_box(layout)
        assert bb[3] == pytest.approx(100.0)


# ─── snap_to_grid (extra) ─────────────────────────────────────────────────────

class TestSnapToGridExtra:
    def test_returns_layout(self):
        layout = _layout_with_cells(2)
        result = snap_to_grid(layout, grid_size=10.0)
        assert result is layout

    def test_zero_grid_raises(self):
        layout = _layout_with_cells(2)
        with pytest.raises(ValueError):
            snap_to_grid(layout, grid_size=0.0)

    def test_negative_grid_raises(self):
        layout = _layout_with_cells(2)
        with pytest.raises(ValueError):
            snap_to_grid(layout, grid_size=-5.0)

    def test_already_on_grid_unchanged(self):
        layout = create_layout()
        add_cell(layout, 0, 10.0, 20.0, 50.0, 50.0)
        snap_to_grid(layout, grid_size=10.0)
        assert layout.cells[0].x == pytest.approx(10.0)
        assert layout.cells[0].y == pytest.approx(20.0)

    def test_snaps_to_nearest(self):
        layout = create_layout()
        add_cell(layout, 0, 13.0, 17.0, 50.0, 50.0)
        snap_to_grid(layout, grid_size=10.0)
        assert layout.cells[0].x == pytest.approx(10.0)
        assert layout.cells[0].y == pytest.approx(20.0)

    def test_grid_1_no_change(self):
        layout = create_layout()
        add_cell(layout, 0, 13.0, 17.0, 50.0, 50.0)
        snap_to_grid(layout, grid_size=1.0)
        assert layout.cells[0].x == pytest.approx(13.0)

    def test_empty_layout_noop(self):
        layout = create_layout()
        snap_to_grid(layout, grid_size=10.0)
        assert len(layout.cells) == 0


# ─── render_layout_image (extra) ──────────────────────────────────────────────

class TestRenderLayoutImageExtra:
    def test_empty_layout_returns_1x1(self):
        layout = create_layout()
        img = render_layout_image(layout)
        assert img.shape == (1, 1)

    def test_returns_uint8(self):
        layout = _layout_with_cells(2)
        img = render_layout_image(layout)
        assert img.dtype == np.uint8

    def test_returns_2d(self):
        layout = _layout_with_cells(2)
        img = render_layout_image(layout)
        assert img.ndim == 2

    def test_image_larger_than_cells(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 100.0, 80.0)
        img = render_layout_image(layout, padding=5)
        assert img.shape[1] >= 100
        assert img.shape[0] >= 80

    def test_bg_color_applied(self):
        layout = create_layout()
        img = render_layout_image(layout, bg_color=200)
        assert img[0, 0] == 200

    def test_cell_color_visible(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        img = render_layout_image(layout, cell_color=150, bg_color=255)
        assert img.min() < 200

    def test_multiple_cells(self):
        layout = _layout_with_cells(3)
        img = render_layout_image(layout)
        assert img.shape[1] > 100

    def test_padding_affects_size(self):
        layout = create_layout()
        add_cell(layout, 0, 0.0, 0.0, 50.0, 50.0)
        img_small = render_layout_image(layout, padding=0)
        img_large = render_layout_image(layout, padding=20)
        assert img_large.shape[0] > img_small.shape[0]


# ─── layout_to_dict / dict_to_layout (extra) ──────────────────────────────────

class TestLayoutSerializationExtra:
    def test_roundtrip_empty_layout(self):
        layout = create_layout(canvas_w=100.0, canvas_h=200.0)
        d = layout_to_dict(layout)
        layout2 = dict_to_layout(d)
        assert layout2.canvas_w == pytest.approx(100.0)
        assert layout2.canvas_h == pytest.approx(200.0)
        assert len(layout2.cells) == 0

    def test_roundtrip_with_cells(self):
        layout = _layout_with_cells(3)
        d = layout_to_dict(layout)
        layout2 = dict_to_layout(d)
        assert len(layout2.cells) == 3

    def test_cell_fields_preserved(self):
        layout = create_layout()
        add_cell(layout, 7, 15.0, 25.0, 60.0, 40.0, rotation=30.0)
        d = layout_to_dict(layout)
        layout2 = dict_to_layout(d)
        c = layout2.cells[0]
        assert c.fragment_idx == 7
        assert c.x == pytest.approx(15.0)
        assert c.y == pytest.approx(25.0)
        assert c.width == pytest.approx(60.0)
        assert c.height == pytest.approx(40.0)
        assert c.rotation == pytest.approx(30.0)

    def test_layout_to_dict_has_cells_key(self):
        layout = _layout_with_cells(2)
        d = layout_to_dict(layout)
        assert "cells" in d

    def test_layout_to_dict_canvas(self):
        layout = create_layout(canvas_w=500.0, canvas_h=300.0)
        d = layout_to_dict(layout)
        assert d["canvas_w"] == pytest.approx(500.0)
        assert d["canvas_h"] == pytest.approx(300.0)

    def test_layout_to_dict_params(self):
        layout = create_layout(mode="auto")
        d = layout_to_dict(layout)
        assert d["params"]["mode"] == "auto"

    def test_dict_to_layout_missing_cells_ok(self):
        data = {"canvas_w": 100.0, "canvas_h": 100.0, "params": {}}
        layout = dict_to_layout(data)
        assert len(layout.cells) == 0

    def test_dict_to_layout_default_rotation(self):
        data = {
            "canvas_w": 0.0, "canvas_h": 0.0, "params": {},
            "cells": [{"fragment_idx": 0, "x": 0.0, "y": 0.0,
                       "width": 50.0, "height": 50.0}]
        }
        layout = dict_to_layout(data)
        assert layout.cells[0].rotation == pytest.approx(0.0)
