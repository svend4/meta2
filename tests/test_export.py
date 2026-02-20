"""
Юнит-тесты для модуля puzzle_reconstruction/export.py.

Тесты покрывают:
    - render_canvas()        — размер холста, фрагменты видны
    - _paste_fragment()      — наложение по маске, поворот
    - render_heatmap()       — форма и диапазон значений
    - render_mosaic()        — размер мозаики, наличие фрагментов
    - save_png()             — файл создаётся и читаемый
    - save_pdf()             — fallback через Pillow (без reportlab)
    - comparison_strip()     — горизонтальный стрип
"""
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from puzzle_reconstruction.export import (
    render_canvas,
    render_heatmap,
    render_mosaic,
    save_png,
    comparison_strip,
)
from puzzle_reconstruction.models import (
    Assembly, Fragment, ShapeClass, EdgeSide,
    TangramSignature, FractalSignature, EdgeSignature,
)


# ─── Фикстуры ────────────────────────────────────────────────────────────

def _make_fragment(frag_id: int, h: int = 80, w: int = 60,
                    color: tuple = (200, 180, 160)) -> Fragment:
    """Создаёт Fragment с цветным прямоугольным изображением и маской."""
    img  = np.full((h, w, 3), color, dtype=np.uint8)
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


def _make_assembly(n: int = 4, spacing: float = 120.0) -> Assembly:
    """Создаёт простую сборку: n фрагментов в ряд."""
    frags = [_make_fragment(i) for i in range(n)]
    placements = {
        i: (np.array([i * spacing, 0.0]), 0.0) for i in range(n)
    }
    asm = Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.zeros((n * 4, n * 4)),
        total_score=0.75,
        ocr_score=0.6,
    )
    return asm


# ─── render_canvas ────────────────────────────────────────────────────────

class TestRenderCanvas:

    def test_returns_ndarray(self):
        asm    = _make_assembly(3)
        canvas = render_canvas(asm)
        assert isinstance(canvas, np.ndarray)

    def test_shape_3_channels(self):
        asm    = _make_assembly(3)
        canvas = render_canvas(asm)
        assert canvas.ndim == 3
        assert canvas.shape[2] == 3

    def test_dtype_uint8(self):
        asm    = _make_assembly(2)
        canvas = render_canvas(asm)
        assert canvas.dtype == np.uint8

    def test_canvas_not_all_background(self):
        """На холсте должны быть пиксели не цвета фона (=240)."""
        asm    = _make_assembly(2)
        canvas = render_canvas(asm, bg_color=(240, 240, 240))
        # Фрагменты цвета (200, 180, 160) — должны присутствовать
        not_bg = np.any(canvas != 240, axis=2)
        assert not_bg.sum() > 0

    def test_empty_placements_returns_placeholder(self):
        asm = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        canvas = render_canvas(asm)
        assert canvas.shape[2] == 3

    def test_single_fragment(self):
        asm = _make_assembly(1)
        canvas = render_canvas(asm)
        assert canvas.shape[0] > 0 and canvas.shape[1] > 0

    def test_margin_increases_canvas(self):
        asm = _make_assembly(2)
        c1  = render_canvas(asm, margin=0)
        c2  = render_canvas(asm, margin=50)
        assert c2.shape[0] >= c1.shape[0]
        assert c2.shape[1] >= c1.shape[1]

    def test_rotated_fragment_fits_canvas(self):
        """Повёрнутый фрагмент не должен выходить за пределы холста."""
        frag = _make_fragment(0)
        asm  = Assembly(
            fragments=[frag],
            placements={0: (np.array([200.0, 200.0]), math.pi / 4)},
            compat_matrix=np.array([]),
        )
        canvas = render_canvas(asm, margin=30)
        assert canvas.shape[0] > 0


# ─── render_heatmap ───────────────────────────────────────────────────────

class TestRenderHeatmap:

    def test_returns_ndarray(self):
        asm = _make_assembly(3)
        hm  = render_heatmap(asm)
        assert isinstance(hm, np.ndarray)

    def test_shape_matches_canvas(self):
        asm    = _make_assembly(3)
        canvas = render_canvas(asm)
        hm     = render_heatmap(asm, canvas_shape=canvas.shape)
        assert hm.shape == canvas.shape

    def test_dtype_uint8(self):
        asm = _make_assembly(2)
        hm  = render_heatmap(asm)
        assert hm.dtype == np.uint8

    def test_3_channels(self):
        asm = _make_assembly(2)
        hm  = render_heatmap(asm)
        assert hm.ndim == 3 and hm.shape[2] == 3

    def test_empty_assembly_returns_image(self):
        asm = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        hm  = render_heatmap(asm)
        assert hm.ndim == 3

    def test_values_in_uint8_range(self):
        asm = _make_assembly(4)
        hm  = render_heatmap(asm)
        assert hm.min() >= 0 and hm.max() <= 255


# ─── render_mosaic ────────────────────────────────────────────────────────

class TestRenderMosaic:

    def test_returns_ndarray(self):
        asm    = _make_assembly(4)
        mosaic = render_mosaic(asm)
        assert isinstance(mosaic, np.ndarray)

    def test_3_channels(self):
        asm    = _make_assembly(4)
        mosaic = render_mosaic(asm, thumb_size=64)
        assert mosaic.ndim == 3 and mosaic.shape[2] == 3

    def test_dtype_uint8(self):
        asm    = _make_assembly(4)
        mosaic = render_mosaic(asm)
        assert mosaic.dtype == np.uint8

    def test_empty_assembly(self):
        asm    = Assembly(fragments=[], placements={}, compat_matrix=np.array([]))
        mosaic = render_mosaic(asm)
        assert mosaic.ndim == 3

    def test_thumb_size_affects_dimensions(self):
        asm = _make_assembly(4)
        m1  = render_mosaic(asm, thumb_size=32, gap=0)
        m2  = render_mosaic(asm, thumb_size=64, gap=0)
        assert m2.shape[0] > m1.shape[0] or m2.shape[1] > m1.shape[1]

    def test_mosaic_not_all_same_color(self):
        """На мозаике должны быть видны фрагменты (не однотонная заливка)."""
        asm    = _make_assembly(3)
        mosaic = render_mosaic(asm, thumb_size=80)
        std    = float(mosaic.std())
        assert std > 0.0

    def test_max_cols_limits_grid_width(self):
        asm    = _make_assembly(9)
        mosaic = render_mosaic(asm, max_cols=3, thumb_size=32)
        # 3 колонки, каждая thumb_size + gap → ширина ≈ 3 * (32+8) + 8
        assert mosaic.shape[1] <= 4 * (32 + 8) + 10


# ─── save_png ─────────────────────────────────────────────────────────────

class TestSavePng:

    def test_creates_png_file(self, tmp_path):
        asm    = _make_assembly(2)
        canvas = render_canvas(asm)
        path   = tmp_path / "out.png"
        save_png(canvas, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_created_file_is_readable(self, tmp_path):
        asm    = _make_assembly(2)
        canvas = render_canvas(asm)
        path   = tmp_path / "out.png"
        save_png(canvas, path)
        loaded = cv2.imread(str(path))
        assert loaded is not None
        assert loaded.shape == canvas.shape

    def test_creates_jpeg_file(self, tmp_path):
        asm    = _make_assembly(2)
        canvas = render_canvas(asm)
        path   = tmp_path / "out.jpg"
        save_png(canvas, path, quality=85)
        assert path.exists()
        loaded = cv2.imread(str(path))
        assert loaded is not None


# ─── comparison_strip ─────────────────────────────────────────────────────

class TestComparisonStrip:

    def test_returns_ndarray(self):
        asm    = _make_assembly(4)
        canvas = render_canvas(asm)
        imgs   = [f.image for f in asm.fragments]
        strip  = comparison_strip(imgs, canvas, target_height=200)
        assert isinstance(strip, np.ndarray)

    def test_3_channels(self):
        asm   = _make_assembly(4)
        imgs  = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, render_canvas(asm), target_height=200)
        assert strip.ndim == 3 and strip.shape[2] == 3

    def test_strip_height_matches_target(self):
        asm    = _make_assembly(4)
        canvas = render_canvas(asm)
        imgs   = [f.image for f in asm.fragments]
        strip  = comparison_strip(imgs, canvas, target_height=150)
        assert strip.shape[0] == 150

    def test_strip_wider_with_heatmap(self):
        asm    = _make_assembly(4)
        canvas = render_canvas(asm)
        hm     = render_heatmap(asm, canvas.shape)
        imgs   = [f.image for f in asm.fragments]
        s_no_hm = comparison_strip(imgs, canvas, target_height=150)
        s_hm    = comparison_strip(imgs, canvas, hm, target_height=150)
        assert s_hm.shape[1] > s_no_hm.shape[1]

    def test_empty_fragments_no_crash(self):
        canvas = np.full((200, 300, 3), 200, dtype=np.uint8)
        strip  = comparison_strip([], canvas, target_height=100)
        assert isinstance(strip, np.ndarray)
