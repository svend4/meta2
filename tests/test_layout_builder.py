"""Tests for puzzle_reconstruction.assembly.layout_builder."""
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


# ─── LayoutCell ──────────────────────────────────────────────────────────────

class TestLayoutCell:
    def test_fields_stored(self):
        c = LayoutCell(fragment_idx=3, x=10.0, y=20.0, width=50.0, height=30.0)
        assert c.fragment_idx == 3
        assert c.x == pytest.approx(10.0)
        assert c.y == pytest.approx(20.0)
        assert c.width == pytest.approx(50.0)
        assert c.height == pytest.approx(30.0)
        assert c.rotation == pytest.approx(0.0)
        assert c.meta == {}

    def test_rotation_stored(self):
        c = LayoutCell(0, 0, 0, 10, 10, rotation=45.0)
        assert c.rotation == pytest.approx(45.0)

    def test_meta_stored(self):
        c = LayoutCell(1, 0, 0, 10, 10, meta={"score": 0.9})
        assert c.meta["score"] == pytest.approx(0.9)

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(0, 0, 0, width=0.0, height=10.0)

    def test_width_negative_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(0, 0, 0, width=-5.0, height=10.0)

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(0, 0, 0, width=10.0, height=0.0)

    def test_height_negative_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(0, 0, 0, width=10.0, height=-1.0)


# ─── AssemblyLayout ──────────────────────────────────────────────────────────

class TestAssemblyLayout:
    def test_defaults(self):
        layout = AssemblyLayout()
        assert layout.cells == []
        assert layout.canvas_w == pytest.approx(0.0)
        assert layout.canvas_h == pytest.approx(0.0)
        assert layout.params == {}

    def test_fields_stored(self):
        layout = AssemblyLayout(canvas_w=100.0, canvas_h=200.0, params={"dpi": 300})
        assert layout.canvas_w == pytest.approx(100.0)
        assert layout.canvas_h == pytest.approx(200.0)
        assert layout.params["dpi"] == 300


# ─── create_layout ───────────────────────────────────────────────────────────

class TestCreateLayout:
    def test_returns_assembly_layout(self):
        layout = create_layout()
        assert isinstance(layout, AssemblyLayout)

    def test_empty_cells(self):
        assert create_layout().cells == []

    def test_canvas_dimensions_stored(self):
        layout = create_layout(canvas_w=400.0, canvas_h=600.0)
        assert layout.canvas_w == pytest.approx(400.0)
        assert layout.canvas_h == pytest.approx(600.0)

    def test_params_stored(self):
        layout = create_layout(page="A4", dpi=300)
        assert layout.params["page"] == "A4"
        assert layout.params["dpi"] == 300

    def test_zero_canvas_default(self):
        layout = create_layout()
        assert layout.canvas_w == pytest.approx(0.0)
        assert layout.canvas_h == pytest.approx(0.0)


# ─── add_cell ────────────────────────────────────────────────────────────────

class TestAddCell:
    def test_adds_cell(self):
        layout = create_layout()
        add_cell(layout, 0, x=5.0, y=10.0, width=40.0, height=50.0)
        assert len(layout.cells) == 1

    def test_cell_fields(self):
        layout = create_layout()
        add_cell(layout, 3, x=1.0, y=2.0, width=30.0, height=20.0, rotation=90.0)
        c = layout.cells[0]
        assert c.fragment_idx == 3
        assert c.x == pytest.approx(1.0)
        assert c.y == pytest.approx(2.0)
        assert c.width == pytest.approx(30.0)
        assert c.height == pytest.approx(20.0)
        assert c.rotation == pytest.approx(90.0)

    def test_meta_stored_via_kwargs(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10, score=0.95)
        assert layout.cells[0].meta["score"] == pytest.approx(0.95)

    def test_replaces_existing_same_idx(self):
        layout = create_layout()
        add_cell(layout, 5, x=0.0, y=0.0, width=10.0, height=10.0)
        add_cell(layout, 5, x=20.0, y=30.0, width=15.0, height=25.0)
        assert len(layout.cells) == 1
        assert layout.cells[0].x == pytest.approx(20.0)

    def test_multiple_cells_appended(self):
        layout = create_layout()
        for i in range(5):
            add_cell(layout, i, x=i * 20.0, y=0.0, width=18.0, height=18.0)
        assert len(layout.cells) == 5

    def test_returns_same_layout(self):
        layout = create_layout()
        result = add_cell(layout, 0, 0, 0, 10, 10)
        assert result is layout

    def test_invalid_width_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            add_cell(layout, 0, 0, 0, width=0.0, height=10.0)


# ─── remove_cell ─────────────────────────────────────────────────────────────

class TestRemoveCell:
    def test_removes_existing_cell(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10)
        add_cell(layout, 1, 20, 0, 10, 10)
        remove_cell(layout, 0)
        assert len(layout.cells) == 1
        assert layout.cells[0].fragment_idx == 1

    def test_no_op_if_not_found(self):
        layout = create_layout()
        add_cell(layout, 2, 0, 0, 10, 10)
        remove_cell(layout, 99)
        assert len(layout.cells) == 1

    def test_returns_same_layout(self):
        layout = create_layout()
        result = remove_cell(layout, 0)
        assert result is layout

    def test_remove_all(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10)
        remove_cell(layout, 0)
        assert layout.cells == []

    def test_remove_middle(self):
        layout = create_layout()
        for i in range(3):
            add_cell(layout, i, float(i * 20), 0, 10, 10)
        remove_cell(layout, 1)
        idxs = [c.fragment_idx for c in layout.cells]
        assert 1 not in idxs
        assert 0 in idxs
        assert 2 in idxs


# ─── compute_bounding_box ────────────────────────────────────────────────────

class TestComputeBoundingBox:
    def test_empty_layout_returns_zeros(self):
        layout = create_layout()
        assert compute_bounding_box(layout) == (0.0, 0.0, 0.0, 0.0)

    def test_single_cell(self):
        layout = create_layout()
        add_cell(layout, 0, x=5.0, y=10.0, width=20.0, height=15.0)
        x, y, w, h = compute_bounding_box(layout)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(10.0)
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(15.0)

    def test_multiple_cells_tight_bbox(self):
        layout = create_layout()
        add_cell(layout, 0, x=0.0,  y=0.0,  width=10.0, height=10.0)
        add_cell(layout, 1, x=15.0, y=5.0,  width=10.0, height=20.0)
        add_cell(layout, 2, x=5.0,  y=20.0, width=30.0, height=5.0)
        x, y, w, h = compute_bounding_box(layout)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)
        assert w == pytest.approx(35.0)
        assert h == pytest.approx(25.0)

    def test_negative_coordinates(self):
        layout = create_layout()
        add_cell(layout, 0, x=-10.0, y=-5.0, width=20.0, height=10.0)
        x, y, w, h = compute_bounding_box(layout)
        assert x == pytest.approx(-10.0)
        assert y == pytest.approx(-5.0)
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(10.0)


# ─── snap_to_grid ────────────────────────────────────────────────────────────

class TestSnapToGrid:
    def test_integer_positions_unchanged_with_grid_1(self):
        layout = create_layout()
        add_cell(layout, 0, x=10.0, y=20.0, width=30.0, height=30.0)
        snap_to_grid(layout, grid_size=1.0)
        c = layout.cells[0]
        assert c.x == pytest.approx(10.0)
        assert c.y == pytest.approx(20.0)

    def test_rounds_to_grid(self):
        layout = create_layout()
        add_cell(layout, 0, x=13.0, y=17.0, width=10.0, height=10.0)
        snap_to_grid(layout, grid_size=10.0)
        c = layout.cells[0]
        assert c.x == pytest.approx(10.0)
        assert c.y == pytest.approx(20.0)

    def test_grid_size_zero_raises(self):
        with pytest.raises(ValueError):
            snap_to_grid(create_layout(), grid_size=0.0)

    def test_grid_size_negative_raises(self):
        with pytest.raises(ValueError):
            snap_to_grid(create_layout(), grid_size=-5.0)

    def test_returns_same_layout(self):
        layout = create_layout()
        result = snap_to_grid(layout, 1.0)
        assert result is layout

    def test_multiple_cells_all_snapped(self):
        layout = create_layout()
        for i in range(3):
            add_cell(layout, i, x=float(i * 7 + 3), y=float(i * 9 + 1), width=5.0, height=5.0)
        snap_to_grid(layout, grid_size=5.0)
        for c in layout.cells:
            assert c.x % 5.0 == pytest.approx(0.0, abs=1e-9)
            assert c.y % 5.0 == pytest.approx(0.0, abs=1e-9)


# ─── render_layout_image ─────────────────────────────────────────────────────

class TestRenderLayoutImage:
    def test_empty_layout_returns_1x1(self):
        layout = create_layout()
        img = render_layout_image(layout)
        assert img.shape == (1, 1)

    def test_returns_uint8(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 20, 20)
        img = render_layout_image(layout)
        assert img.dtype == np.uint8

    def test_2d_output(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 20, 20)
        img = render_layout_image(layout)
        assert img.ndim == 2

    def test_padding_increases_size(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 20, 20)
        img0 = render_layout_image(layout, padding=0)
        img2 = render_layout_image(layout, padding=10)
        assert img2.shape[0] > img0.shape[0]
        assert img2.shape[1] > img0.shape[1]

    def test_cells_visible_in_image(self):
        layout = create_layout()
        add_cell(layout, 0, 5, 5, 20, 20)
        img = render_layout_image(layout, padding=0, bg_color=255, cell_color=0)
        # Some pixels must be darker than background
        assert img.min() < 255

    def test_multiple_cells_rendered(self):
        layout = create_layout()
        add_cell(layout, 0, 0,  0,  20, 20)
        add_cell(layout, 1, 30, 0,  20, 20)
        add_cell(layout, 2, 0,  30, 20, 20)
        img = render_layout_image(layout, padding=2, bg_color=255, cell_color=128)
        # Image should contain non-background pixels
        assert np.any(img < 255)

    def test_values_in_range(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 30, 30)
        img = render_layout_image(layout)
        assert img.min() >= 0
        assert img.max() <= 255


# ─── layout_to_dict ──────────────────────────────────────────────────────────

class TestLayoutToDict:
    def test_returns_dict(self):
        assert isinstance(layout_to_dict(create_layout()), dict)

    def test_has_cells_key(self):
        assert "cells" in layout_to_dict(create_layout())

    def test_empty_cells(self):
        d = layout_to_dict(create_layout())
        assert d["cells"] == []

    def test_canvas_dimensions(self):
        layout = create_layout(canvas_w=200.0, canvas_h=300.0)
        d = layout_to_dict(layout)
        assert d["canvas_w"] == pytest.approx(200.0)
        assert d["canvas_h"] == pytest.approx(300.0)

    def test_params_stored(self):
        layout = create_layout(dpi=600)
        d = layout_to_dict(layout)
        assert d["params"]["dpi"] == 600

    def test_cell_fields_in_dict(self):
        layout = create_layout()
        add_cell(layout, 7, x=1.0, y=2.0, width=10.0, height=15.0, rotation=30.0)
        d = layout_to_dict(layout)
        cd = d["cells"][0]
        assert cd["fragment_idx"] == 7
        assert cd["x"] == pytest.approx(1.0)
        assert cd["y"] == pytest.approx(2.0)
        assert cd["width"] == pytest.approx(10.0)
        assert cd["height"] == pytest.approx(15.0)
        assert cd["rotation"] == pytest.approx(30.0)

    def test_meta_in_cell_dict(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 5, 5, score=0.75)
        d = layout_to_dict(layout)
        assert d["cells"][0]["meta"]["score"] == pytest.approx(0.75)


# ─── dict_to_layout ──────────────────────────────────────────────────────────

class TestDictToLayout:
    def test_returns_assembly_layout(self):
        d = layout_to_dict(create_layout())
        assert isinstance(dict_to_layout(d), AssemblyLayout)

    def test_roundtrip_empty(self):
        layout = create_layout(canvas_w=100.0, canvas_h=200.0, dpi=72)
        d = layout_to_dict(layout)
        restored = dict_to_layout(d)
        assert restored.canvas_w == pytest.approx(100.0)
        assert restored.canvas_h == pytest.approx(200.0)
        assert restored.params["dpi"] == 72
        assert restored.cells == []

    def test_roundtrip_cells(self):
        layout = create_layout()
        add_cell(layout, 3, x=5.0, y=10.0, width=20.0, height=30.0, rotation=15.0, score=0.8)
        d = layout_to_dict(layout)
        restored = dict_to_layout(d)
        assert len(restored.cells) == 1
        c = restored.cells[0]
        assert c.fragment_idx == 3
        assert c.x == pytest.approx(5.0)
        assert c.y == pytest.approx(10.0)
        assert c.width == pytest.approx(20.0)
        assert c.height == pytest.approx(30.0)
        assert c.rotation == pytest.approx(15.0)
        assert c.meta["score"] == pytest.approx(0.8)

    def test_roundtrip_multiple_cells(self):
        layout = create_layout()
        for i in range(6):
            add_cell(layout, i, x=float(i * 25), y=0.0, width=20.0, height=20.0)
        d = layout_to_dict(layout)
        restored = dict_to_layout(d)
        assert len(restored.cells) == 6
        idxs = sorted(c.fragment_idx for c in restored.cells)
        assert idxs == list(range(6))

    def test_empty_data_gives_defaults(self):
        restored = dict_to_layout({})
        assert isinstance(restored, AssemblyLayout)
        assert restored.canvas_w == pytest.approx(0.0)
        assert restored.cells == []
