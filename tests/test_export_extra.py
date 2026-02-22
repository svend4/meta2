"""Additional tests for puzzle_reconstruction/export.py."""
import math
import numpy as np
import pytest
import cv2

from puzzle_reconstruction.export import (
    render_canvas,
    render_heatmap,
    render_mosaic,
    save_png,
    comparison_strip,
)
from puzzle_reconstruction.models import (
    Assembly, Fragment, ShapeClass,
    TangramSignature, FractalSignature,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_fragment(frag_id, h=80, w=60, color=(200, 180, 160)):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    frag = Fragment(fragment_id=frag_id, image=img, mask=mask, contour=contour)
    frag.tangram = TangramSignature(
        polygon=contour / np.array([w, h]),
        shape_class=ShapeClass.RECTANGLE,
        centroid=np.array([0.5, 0.5]),
        angle=0.0, scale=1.0, area=0.5,
    )
    frag.fractal = FractalSignature(
        fd_box=1.3, fd_divider=1.35,
        ifs_coeffs=np.zeros(8),
        css_image=[], chain_code="", curve=np.zeros((10, 2)),
    )
    return frag


def _make_assembly(n=4, spacing=120.0, color=(200, 180, 160)):
    frags = [_make_fragment(i, color=color) for i in range(n)]
    placements = {i: (np.array([i * spacing, 0.0]), 0.0) for i in range(n)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.zeros((n * 4, n * 4)),
        total_score=0.75,
        ocr_score=0.6,
    )


# ─── TestRenderCanvasExtra ────────────────────────────────────────────────────

class TestRenderCanvasExtra:
    def test_six_fragments_wider_canvas(self):
        asm2 = _make_assembly(2)
        asm6 = _make_assembly(6)
        c2 = render_canvas(asm2)
        c6 = render_canvas(asm6)
        assert c6.shape[1] > c2.shape[1]

    def test_different_bg_color(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm, bg_color=(0, 0, 0))
        assert canvas.dtype == np.uint8
        assert canvas.ndim == 3

    def test_large_margin(self):
        asm = _make_assembly(2)
        c_small = render_canvas(asm, margin=10)
        c_large = render_canvas(asm, margin=200)
        assert c_large.shape[0] >= c_small.shape[0]
        assert c_large.shape[1] >= c_small.shape[1]

    def test_two_fragments_canvas_nontrivial(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        assert canvas.shape[0] > 0
        assert canvas.shape[1] > 0

    def test_rotation_pi_half(self):
        frag = _make_fragment(0)
        asm = Assembly(
            fragments=[frag],
            placements={0: (np.array([200.0, 200.0]), math.pi / 2)},
            compat_matrix=np.array([]),
        )
        canvas = render_canvas(asm, margin=50)
        assert isinstance(canvas, np.ndarray)
        assert canvas.ndim == 3

    def test_canvas_values_uint8_range(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        assert canvas.min() >= 0
        assert canvas.max() <= 255


# ─── TestRenderHeatmapExtra ───────────────────────────────────────────────────

class TestRenderHeatmapExtra:
    def test_single_fragment_assembly(self):
        asm = _make_assembly(1)
        hm = render_heatmap(asm)
        assert isinstance(hm, np.ndarray)
        assert hm.ndim == 3

    def test_six_fragment_assembly(self):
        asm = _make_assembly(6)
        hm = render_heatmap(asm)
        assert hm.dtype == np.uint8

    def test_canvas_shape_param_matches_canvas(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas_shape=canvas.shape)
        assert hm.shape == canvas.shape

    def test_six_fragment_shape_matches_canvas(self):
        asm = _make_assembly(6)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas_shape=canvas.shape)
        assert hm.shape == canvas.shape

    def test_values_bounded(self):
        asm = _make_assembly(4)
        hm = render_heatmap(asm)
        assert int(hm.min()) >= 0
        assert int(hm.max()) <= 255


# ─── TestRenderMosaicExtra ────────────────────────────────────────────────────

class TestRenderMosaicExtra:
    def test_single_fragment_mosaic(self):
        asm = _make_assembly(1)
        mosaic = render_mosaic(asm, thumb_size=64)
        assert isinstance(mosaic, np.ndarray)
        assert mosaic.ndim == 3

    def test_gap_0_vs_gap_10_different_size(self):
        asm = _make_assembly(4)
        m0 = render_mosaic(asm, thumb_size=32, gap=0)
        m10 = render_mosaic(asm, thumb_size=32, gap=10)
        assert m10.shape[1] >= m0.shape[1] or m10.shape[0] >= m0.shape[0]

    def test_two_columns_layout(self):
        asm = _make_assembly(4)
        mosaic = render_mosaic(asm, max_cols=2, thumb_size=32)
        assert isinstance(mosaic, np.ndarray)

    def test_thumb_size_80(self):
        asm = _make_assembly(3)
        mosaic = render_mosaic(asm, thumb_size=80)
        assert mosaic.dtype == np.uint8

    def test_values_in_range(self):
        asm = _make_assembly(3)
        mosaic = render_mosaic(asm, thumb_size=32)
        assert mosaic.min() >= 0 and mosaic.max() <= 255


# ─── TestSavePngExtra ─────────────────────────────────────────────────────────

class TestSavePngExtra:
    def test_save_and_read_heatmap(self, tmp_path):
        asm = _make_assembly(2)
        hm = render_heatmap(asm)
        path = tmp_path / "heatmap.png"
        save_png(hm, path)
        assert path.exists()
        loaded = cv2.imread(str(path))
        assert loaded is not None

    def test_save_mosaic(self, tmp_path):
        asm = _make_assembly(3)
        mosaic = render_mosaic(asm, thumb_size=32)
        path = tmp_path / "mosaic.png"
        save_png(mosaic, path)
        assert path.stat().st_size > 0

    def test_string_path_accepted(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        path_str = str(tmp_path / "str_path.png")
        save_png(canvas, path_str)
        import os
        assert os.path.exists(path_str)

    def test_loaded_shape_matches_saved(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        path = tmp_path / "shape_check.png"
        save_png(canvas, path)
        loaded = cv2.imread(str(path))
        assert loaded.shape == canvas.shape


# ─── TestComparisonStripExtra ─────────────────────────────────────────────────

class TestComparisonStripExtra:
    def test_large_target_height(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=400)
        assert strip.shape[0] == 400

    def test_small_target_height(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=50)
        assert strip.shape[0] == 50

    def test_single_fragment(self):
        asm = _make_assembly(1)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=100)
        assert isinstance(strip, np.ndarray)
        assert strip.ndim == 3

    def test_values_in_range(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=150)
        assert strip.min() >= 0 and strip.max() <= 255

    def test_with_heatmap_3_channels(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas.shape)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, hm, target_height=150)
        assert strip.shape[2] == 3
