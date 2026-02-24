"""Extra tests for puzzle_reconstruction/assembly/layout_builder.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── LayoutCell ─────────────────────────────────────────────────────────────

class TestLayoutCellExtra:
    def test_valid(self):
        c = LayoutCell(fragment_idx=0, x=10.0, y=20.0,
                       width=30.0, height=40.0)
        assert c.fragment_idx == 0

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=0, height=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=10, height=0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            LayoutCell(fragment_idx=0, x=0, y=0, width=-5, height=10)

    def test_repr(self):
        c = LayoutCell(fragment_idx=1, x=5.0, y=10.0,
                       width=20.0, height=30.0)
        s = repr(c)
        assert "idx=1" in s

    def test_rotation_default(self):
        c = LayoutCell(fragment_idx=0, x=0, y=0, width=10, height=10)
        assert c.rotation == 0.0


# ─── AssemblyLayout ─────────────────────────────────────────────────────────

class TestAssemblyLayoutExtra:
    def test_defaults(self):
        layout = AssemblyLayout()
        assert layout.cells == []
        assert layout.canvas_w == 0.0
        assert layout.canvas_h == 0.0

    def test_repr(self):
        layout = AssemblyLayout()
        s = repr(layout)
        assert "n_cells=0" in s


# ─── create_layout ──────────────────────────────────────────────────────────

class TestCreateLayoutExtra:
    def test_empty(self):
        layout = create_layout()
        assert layout.cells == []
        assert layout.canvas_w == 0.0

    def test_with_canvas(self):
        layout = create_layout(canvas_w=800, canvas_h=600)
        assert layout.canvas_w == 800.0
        assert layout.canvas_h == 600.0

    def test_with_params(self):
        layout = create_layout(mode="test")
        assert layout.params["mode"] == "test"


# ─── add_cell ───────────────────────────────────────────────────────────────

class TestAddCellExtra:
    def test_add_one(self):
        layout = create_layout()
        add_cell(layout, 0, 10, 20, 30, 40)
        assert len(layout.cells) == 1
        assert layout.cells[0].fragment_idx == 0

    def test_add_replaces(self):
        layout = create_layout()
        add_cell(layout, 0, 10, 20, 30, 40)
        add_cell(layout, 0, 50, 60, 70, 80)
        assert len(layout.cells) == 1
        assert layout.cells[0].x == 50.0

    def test_add_multiple(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10)
        add_cell(layout, 1, 20, 20, 10, 10)
        assert len(layout.cells) == 2

    def test_zero_width_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            add_cell(layout, 0, 0, 0, 0, 10)


# ─── remove_cell ────────────────────────────────────────────────────────────

class TestRemoveCellExtra:
    def test_remove_existing(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10)
        remove_cell(layout, 0)
        assert len(layout.cells) == 0

    def test_remove_nonexistent(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10)
        remove_cell(layout, 99)
        assert len(layout.cells) == 1


# ─── compute_bounding_box ──────────────────────────────────────────────────

class TestComputeBoundingBoxExtra:
    def test_empty(self):
        layout = create_layout()
        assert compute_bounding_box(layout) == (0.0, 0.0, 0.0, 0.0)

    def test_single(self):
        layout = create_layout()
        add_cell(layout, 0, 10, 20, 30, 40)
        bbox = compute_bounding_box(layout)
        assert bbox == (10.0, 20.0, 30.0, 40.0)

    def test_multiple(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 50, 50)
        add_cell(layout, 1, 100, 100, 50, 50)
        bbox = compute_bounding_box(layout)
        assert bbox == (0.0, 0.0, 150.0, 150.0)


# ─── snap_to_grid ───────────────────────────────────────────────────────────

class TestSnapToGridExtra:
    def test_snaps(self):
        layout = create_layout()
        add_cell(layout, 0, 3.7, 7.2, 10, 10)
        snap_to_grid(layout, grid_size=5.0)
        assert layout.cells[0].x == pytest.approx(5.0)
        assert layout.cells[0].y == pytest.approx(5.0)

    def test_zero_grid_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            snap_to_grid(layout, grid_size=0.0)

    def test_negative_grid_raises(self):
        layout = create_layout()
        with pytest.raises(ValueError):
            snap_to_grid(layout, grid_size=-1.0)


# ─── render_layout_image ───────────────────────────────────────────────────

class TestRenderLayoutImageExtra:
    def test_empty(self):
        layout = create_layout()
        img = render_layout_image(layout)
        assert img.shape == (1, 1)
        assert img.dtype == np.uint8

    def test_with_cells(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 50, 50)
        add_cell(layout, 1, 60, 0, 50, 50)
        img = render_layout_image(layout)
        assert img.ndim == 2
        assert img.dtype == np.uint8
        assert img.shape[0] > 0 and img.shape[1] > 0


# ─── layout_to_dict / dict_to_layout ───────────────────────────────────────

class TestLayoutSerializationExtra:
    def test_roundtrip(self):
        layout = create_layout(canvas_w=100, canvas_h=200)
        add_cell(layout, 0, 10, 20, 30, 40, rotation=90)
        d = layout_to_dict(layout)
        restored = dict_to_layout(d)
        assert restored.canvas_w == 100.0
        assert restored.canvas_h == 200.0
        assert len(restored.cells) == 1
        assert restored.cells[0].fragment_idx == 0
        assert restored.cells[0].rotation == 90.0

    def test_empty_roundtrip(self):
        layout = create_layout()
        d = layout_to_dict(layout)
        restored = dict_to_layout(d)
        assert len(restored.cells) == 0

    def test_dict_keys(self):
        layout = create_layout()
        add_cell(layout, 0, 0, 0, 10, 10)
        d = layout_to_dict(layout)
        assert "canvas_w" in d
        assert "cells" in d
        assert len(d["cells"]) == 1
