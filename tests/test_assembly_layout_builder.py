"""Tests for puzzle_reconstruction/assembly/layout_builder.py."""
import pytest
import numpy as np

from puzzle_reconstruction.assembly.layout_builder import (
    LayoutCell,
    AssemblyLayout,
    create_layout,
    add_cell,
    remove_cell,
    compute_bounding_box,
    snap_to_grid,
    render_layout_image,
    layout_to_dict,
    dict_to_layout,
)


# ─── LayoutCell ───────────────────────────────────────────────────────────────

class TestLayoutCell:
    def test_basic_creation(self):
        cell = LayoutCell(fragment_idx=0, x=10.0, y=20.0, width=100.0, height=50.0)
        assert cell.fragment_idx == 0
        assert cell.x == 10.0
        assert cell.y == 20.0
        assert cell.width == 100.0
        assert cell.height == 50.0

    def test_default_rotation_is_zero(self):
        cell = LayoutCell(fragment_idx=1, x=0.0, y=0.0, width=10.0, height=10.0)
        assert cell.rotation == 0.0

    def test_custom_rotation(self):
        cell = LayoutCell(fragment_idx=2, x=0.0, y=0.0, width=10.0, height=10.0, rotation=90.0)
        assert cell.rotation == 90.0

    def test_default_meta_is_empty_dict(self):
        cell = LayoutCell(fragment_idx=3, x=0.0, y=0.0, width=10.0, height=10.0)
        assert cell.meta == {}

    def test_meta_is_stored(self):
        cell = LayoutCell(fragment_idx=4, x=0.0, y=0.0, width=10.0, height=10.0,
                          meta={"label": "A"})
        assert cell.meta["label"] == "A"

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0.0, y=0.0, width=0.0, height=10.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0.0, y=0.0, width=-5.0, height=10.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0.0, y=0.0, width=10.0, height=0.0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0.0, y=0.0, width=10.0, height=-1.0)


# ─── AssemblyLayout ───────────────────────────────────────────────────────────

class TestAssemblyLayout:
    def test_default_canvas_is_zero(self):
        layout = AssemblyLayout()
        assert layout.canvas_w == 0.0
        assert layout.canvas_h == 0.0

    def test_default_cells_is_empty(self):
        layout = AssemblyLayout()
        assert layout.cells == []

    def test_custom_canvas(self):
        layout = AssemblyLayout(canvas_w=800.0, canvas_h=600.0)
        assert layout.canvas_w == 800.0
        assert layout.canvas_h == 600.0


# ─── create_layout ────────────────────────────────────────────────────────────

class TestCreateLayout:
    def test_returns_empty_layout(self):
        layout = create_layout()
        assert isinstance(layout, AssemblyLayout)
        assert layout.cells == []

    def test_canvas_dimensions_stored(self):
        layout = create_layout(canvas_w=1920.0, canvas_h=1080.0)
        assert layout.canvas_w == 1920.0
        assert layout.canvas_h == 1080.0

    def test_params_stored(self):
        layout = create_layout(author="test", version=2)
        assert layout.params["author"] == "test"
        assert layout.params["version"] == 2

    def test_zero_canvas(self):
        layout = create_layout(canvas_w=0.0, canvas_h=0.0)
        assert layout.canvas_w == 0.0
        assert layout.canvas_h == 0.0


# ─── add_cell ─────────────────────────────────────────────────────────────────

class TestAddCell:
    def test_adds_one_cell(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=100.0, height=80.0)
        assert len(layout.cells) == 1

    def test_cell_properties_correct(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=5, x=10.0, y=20.0, width=100.0, height=80.0, rotation=45.0)
        cell = layout.cells[0]
        assert cell.fragment_idx == 5
        assert cell.x == 10.0
        assert cell.y == 20.0
        assert cell.width == 100.0
        assert cell.height == 80.0
        assert cell.rotation == 45.0

    def test_meta_passed_via_kwargs(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=10.0, height=10.0, label="X")
        assert layout.cells[0].meta["label"] == "X"

    def test_replaces_existing_cell(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=50.0, height=50.0)
        add_cell(layout, fragment_idx=0, x=99.0, y=99.0, width=10.0, height=10.0)
        assert len(layout.cells) == 1
        assert layout.cells[0].x == 99.0

    def test_add_multiple_cells(self):
        layout = create_layout()
        for i in range(5):
            add_cell(layout, fragment_idx=i, x=float(i * 100), y=0.0, width=90.0, height=80.0)
        assert len(layout.cells) == 5

    def test_returns_layout(self):
        layout = create_layout()
        result = add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=10.0, height=10.0)
        assert result is layout

    def test_invalid_width_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=0.0, height=10.0)


# ─── remove_cell ──────────────────────────────────────────────────────────────

class TestRemoveCell:
    def test_removes_existing_cell(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=3, x=0.0, y=0.0, width=10.0, height=10.0)
        remove_cell(layout, fragment_idx=3)
        assert len(layout.cells) == 0

    def test_remove_nonexistent_cell_ok(self):
        layout = create_layout()
        remove_cell(layout, fragment_idx=99)
        assert len(layout.cells) == 0

    def test_remove_specific_cell_from_multiple(self):
        layout = create_layout()
        for i in range(4):
            add_cell(layout, fragment_idx=i, x=float(i * 50), y=0.0, width=40.0, height=40.0)
        remove_cell(layout, fragment_idx=2)
        assert len(layout.cells) == 3
        assert all(c.fragment_idx != 2 for c in layout.cells)

    def test_returns_layout(self):
        layout = create_layout()
        result = remove_cell(layout, fragment_idx=0)
        assert result is layout


# ─── compute_bounding_box ─────────────────────────────────────────────────────

class TestComputeBoundingBox:
    def test_empty_layout_returns_zeros(self):
        layout = create_layout()
        assert compute_bounding_box(layout) == (0.0, 0.0, 0.0, 0.0)

    def test_single_cell(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=10.0, y=20.0, width=100.0, height=50.0)
        x_min, y_min, w, h = compute_bounding_box(layout)
        assert x_min == 10.0
        assert y_min == 20.0
        assert w == 100.0
        assert h == 50.0

    def test_multiple_cells(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=50.0, height=50.0)
        add_cell(layout, fragment_idx=1, x=60.0, y=60.0, width=40.0, height=40.0)
        x_min, y_min, w, h = compute_bounding_box(layout)
        assert x_min == 0.0
        assert y_min == 0.0
        assert w == 100.0
        assert h == 100.0

    def test_negative_positions(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=-10.0, y=-20.0, width=30.0, height=30.0)
        x_min, y_min, w, h = compute_bounding_box(layout)
        assert x_min == -10.0
        assert y_min == -20.0
        assert w == 30.0
        assert h == 30.0


# ─── snap_to_grid ─────────────────────────────────────────────────────────────

class TestSnapToGrid:
    def test_snaps_to_10px_grid(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=13.0, y=17.0, width=10.0, height=10.0)
        snap_to_grid(layout, grid_size=10.0)
        assert layout.cells[0].x == 10.0
        assert layout.cells[0].y == 20.0

    def test_already_aligned_unchanged(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=100.0, y=200.0, width=10.0, height=10.0)
        snap_to_grid(layout, grid_size=10.0)
        assert layout.cells[0].x == 100.0
        assert layout.cells[0].y == 200.0

    def test_grid_size_one(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=3.4, y=7.6, width=10.0, height=10.0)
        snap_to_grid(layout, grid_size=1.0)
        assert layout.cells[0].x == 3.0
        assert layout.cells[0].y == 8.0

    def test_invalid_grid_size_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            snap_to_grid(layout, grid_size=0.0)

    def test_negative_grid_size_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            snap_to_grid(layout, grid_size=-1.0)

    def test_returns_layout(self):
        layout = create_layout()
        result = snap_to_grid(layout, grid_size=5.0)
        assert result is layout


# ─── render_layout_image ──────────────────────────────────────────────────────

class TestRenderLayoutImage:
    def test_empty_layout_returns_1x1(self):
        layout = create_layout()
        img = render_layout_image(layout)
        assert img.shape == (1, 1)
        assert img.dtype == np.uint8

    def test_nonempty_returns_2d_array(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=50.0, height=50.0)
        img = render_layout_image(layout)
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_image_has_positive_size(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=100.0, height=80.0)
        img = render_layout_image(layout)
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_background_color_used(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=50.0, y=50.0, width=10.0, height=10.0)
        img = render_layout_image(layout, bg_color=128, cell_color=200)
        # There should be background pixels at value 128
        assert (img == 128).any()

    def test_cell_color_present(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=0, x=0.0, y=0.0, width=50.0, height=50.0)
        img = render_layout_image(layout, cell_color=200)
        assert (img == 200).any()


# ─── layout_to_dict / dict_to_layout ─────────────────────────────────────────

class TestSerialization:
    def test_roundtrip_empty_layout(self):
        layout = create_layout(canvas_w=800.0, canvas_h=600.0)
        d = layout_to_dict(layout)
        layout2 = dict_to_layout(d)
        assert layout2.canvas_w == 800.0
        assert layout2.canvas_h == 600.0
        assert layout2.cells == []

    def test_roundtrip_with_cells(self):
        layout = create_layout(canvas_w=500.0, canvas_h=400.0)
        add_cell(layout, fragment_idx=0, x=10.0, y=20.0, width=100.0, height=80.0, rotation=45.0)
        add_cell(layout, fragment_idx=1, x=120.0, y=20.0, width=100.0, height=80.0)
        d = layout_to_dict(layout)
        layout2 = dict_to_layout(d)
        assert len(layout2.cells) == 2
        c0 = next(c for c in layout2.cells if c.fragment_idx == 0)
        assert c0.x == 10.0
        assert c0.rotation == 45.0

    def test_dict_has_required_keys(self):
        layout = create_layout()
        d = layout_to_dict(layout)
        assert "canvas_w" in d
        assert "canvas_h" in d
        assert "cells" in d
        assert "params" in d

    def test_cell_dict_has_required_keys(self):
        layout = create_layout()
        add_cell(layout, fragment_idx=7, x=0.0, y=0.0, width=10.0, height=10.0)
        d = layout_to_dict(layout)
        cell_d = d["cells"][0]
        assert "fragment_idx" in cell_d
        assert "x" in cell_d
        assert "y" in cell_d
        assert "width" in cell_d
        assert "height" in cell_d

    def test_params_preserved(self):
        layout = create_layout(title="my layout", version=3)
        d = layout_to_dict(layout)
        layout2 = dict_to_layout(d)
        assert layout2.params.get("title") == "my layout"
        assert layout2.params.get("version") == 3
