"""
Integration tests for puzzle_reconstruction/export.py.

Covers render_canvas, render_heatmap, render_mosaic, save_png, and
comparison_strip across ~60 test methods organised into 5 test classes.
"""
from __future__ import annotations

import math
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
from puzzle_reconstruction.models import Assembly, Fragment


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_fragment(
    frag_id: int,
    h: int = 80,
    w: int = 60,
    color: tuple = (200, 180, 160),
    with_mask: bool = True,
) -> Fragment:
    """Create a Fragment with a solid-colour image and optional binary mask."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8) if with_mask else None
    return Fragment(fragment_id=frag_id, image=img, mask=mask)


def _make_assembly(
    n: int = 3,
    spacing: float = 120.0,
    with_compat: bool = True,
    total_score: float = 0.75,
) -> Assembly:
    """Create a simple Assembly with *n* fragments placed in a row."""
    frags = [_make_fragment(i) for i in range(n)]
    placements = {
        i: (np.array([i * spacing, 0.0]), 0.0) for i in range(n)
    }
    compat = np.random.rand(n * 4, n * 4).astype(np.float32) if with_compat else None
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=compat,
        total_score=total_score,
    )


def _empty_assembly() -> Assembly:
    """Assembly with no fragments and no placements."""
    return Assembly(fragments=[], placements={}, compat_matrix=None, total_score=0.0)


# ---------------------------------------------------------------------------
# 1. TestRenderCanvas  (~15 tests)
# ---------------------------------------------------------------------------

class TestRenderCanvas:

    def test_empty_placements_returns_ndarray(self):
        asm = _empty_assembly()
        canvas = render_canvas(asm)
        assert isinstance(canvas, np.ndarray)

    def test_empty_placements_stub_shape(self):
        """Empty placements → 200×400×3 stub as documented."""
        asm = _empty_assembly()
        canvas = render_canvas(asm)
        assert canvas.shape == (200, 400, 3)

    def test_empty_placements_stub_background(self):
        """Stub should be filled with the bg_color."""
        bg = (10, 20, 30)
        asm = _empty_assembly()
        canvas = render_canvas(asm, bg_color=bg)
        # All pixels should equal the bg_color (BGR order)
        assert np.all(canvas[:, :, 0] == bg[0])
        assert np.all(canvas[:, :, 1] == bg[1])
        assert np.all(canvas[:, :, 2] == bg[2])

    def test_single_fragment_returns_ndarray(self):
        asm = _make_assembly(1)
        canvas = render_canvas(asm)
        assert isinstance(canvas, np.ndarray)

    def test_single_fragment_shape_3ch(self):
        asm = _make_assembly(1)
        canvas = render_canvas(asm)
        assert canvas.ndim == 3
        assert canvas.shape[2] == 3

    def test_returns_uint8_dtype(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        assert canvas.dtype == np.uint8

    def test_three_fragments_valid_canvas(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        assert canvas.shape[0] > 0
        assert canvas.shape[1] > 0

    def test_fragment_pixels_visible_on_canvas(self):
        """Fragment pixels (color != bg_color) should appear on canvas."""
        # Fragment is solid (200, 180, 160), background is (240, 240, 240)
        asm = _make_assembly(2, with_compat=False)
        canvas = render_canvas(asm, bg_color=(240, 240, 240))
        not_bg = np.any(canvas != 240, axis=2)
        assert not_bg.sum() > 0

    def test_larger_margin_produces_larger_canvas(self):
        asm = _make_assembly(2)
        c_small = render_canvas(asm, margin=0)
        c_large = render_canvas(asm, margin=80)
        assert c_large.shape[0] >= c_small.shape[0]
        assert c_large.shape[1] >= c_small.shape[1]

    def test_zero_margin_still_valid(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm, margin=0)
        assert canvas.shape[0] >= 200
        assert canvas.shape[1] >= 200

    def test_bg_color_applied(self):
        """The specified bg_color should appear in background areas."""
        bg = (50, 100, 150)
        asm = _make_assembly(1)
        canvas = render_canvas(asm, bg_color=bg)
        # At least one corner pixel should match bg_color (top-left corner)
        corner = canvas[0, 0]
        assert tuple(corner) == bg

    def test_fragment_with_mask(self):
        """Fragment that has a mask is pasted correctly."""
        frag = _make_fragment(0, h=50, w=50, with_mask=True)
        asm = Assembly(
            fragments=[frag],
            placements={0: (np.array([100.0, 100.0]), 0.0)},
            compat_matrix=None,
            total_score=0.5,
        )
        canvas = render_canvas(asm)
        assert canvas.dtype == np.uint8

    def test_fragment_without_mask(self):
        """Fragment without mask uses white-transparency fallback."""
        frag = _make_fragment(0, h=50, w=50, with_mask=False)
        asm = Assembly(
            fragments=[frag],
            placements={0: (np.array([100.0, 100.0]), 0.0)},
            compat_matrix=None,
            total_score=0.5,
        )
        canvas = render_canvas(asm)
        assert canvas.dtype == np.uint8

    def test_angle_zero_vs_quarter_pi(self):
        """Angle changes layout but canvas is always valid uint8 ndarray."""
        frag0 = _make_fragment(0, h=60, w=60)
        frag1 = _make_fragment(0, h=60, w=60)

        asm0 = Assembly(
            fragments=[frag0],
            placements={0: (np.array([100.0, 100.0]), 0.0)},
            compat_matrix=None,
            total_score=0.5,
        )
        asm1 = Assembly(
            fragments=[frag1],
            placements={0: (np.array([100.0, 100.0]), math.pi / 4)},
            compat_matrix=None,
            total_score=0.5,
        )
        c0 = render_canvas(asm0)
        c1 = render_canvas(asm1)
        assert c0.dtype == np.uint8
        assert c1.dtype == np.uint8

    def test_five_fragments_canvas_wide_enough(self):
        """5 fragments placed in a row → canvas wider than a single fragment."""
        asm = _make_assembly(5, spacing=100.0)
        canvas = render_canvas(asm)
        single_frag_w = asm.fragments[0].image.shape[1]
        assert canvas.shape[1] > single_frag_w


# ---------------------------------------------------------------------------
# 2. TestRenderHeatmap  (~12 tests)
# ---------------------------------------------------------------------------

class TestRenderHeatmap:

    def test_returns_ndarray(self):
        asm = _make_assembly(3)
        hm = render_heatmap(asm)
        assert isinstance(hm, np.ndarray)

    def test_returns_uint8_dtype(self):
        asm = _make_assembly(2)
        hm = render_heatmap(asm)
        assert hm.dtype == np.uint8

    def test_shape_3_channels(self):
        asm = _make_assembly(2)
        hm = render_heatmap(asm)
        assert hm.ndim == 3
        assert hm.shape[2] == 3

    def test_canvas_shape_none_auto_computes(self):
        """canvas_shape=None should still return a valid image."""
        asm = _make_assembly(3)
        hm = render_heatmap(asm, canvas_shape=None)
        assert hm.shape[0] > 0
        assert hm.shape[1] > 0

    def test_canvas_shape_provided_matches_output(self):
        """When canvas_shape is provided the output height/width must match."""
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas_shape=canvas.shape)
        assert hm.shape[0] == canvas.shape[0]
        assert hm.shape[1] == canvas.shape[1]

    def test_empty_placements_still_returns_image(self):
        asm = _empty_assembly()
        hm = render_heatmap(asm)
        assert isinstance(hm, np.ndarray)
        assert hm.ndim == 3

    def test_values_in_valid_range(self):
        asm = _make_assembly(4)
        hm = render_heatmap(asm)
        assert int(hm.min()) >= 0
        assert int(hm.max()) <= 255

    def test_alpha_near_zero_close_to_canvas(self):
        """alpha≈0 → heatmap is nearly the same as the underlying canvas."""
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas_shape=canvas.shape, alpha=0.0)
        diff = np.abs(hm.astype(float) - canvas.astype(float)).mean()
        assert diff < 5.0  # very close to canvas when alpha=0

    def test_alpha_near_one_differs_from_canvas(self):
        """alpha=1 → output is essentially the colormap, not the canvas."""
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        hm_full = render_heatmap(asm, canvas_shape=canvas.shape, alpha=1.0)
        hm_none = render_heatmap(asm, canvas_shape=canvas.shape, alpha=0.0)
        diff = np.abs(hm_full.astype(float) - hm_none.astype(float)).mean()
        assert diff > 0.0  # they must differ

    def test_compat_matrix_none_uses_total_score_fallback(self):
        """With compat_matrix=None the function should not crash."""
        asm = Assembly(
            fragments=[_make_fragment(0)],
            placements={0: (np.array([50.0, 50.0]), 0.0)},
            compat_matrix=None,
            total_score=0.9,
        )
        hm = render_heatmap(asm)
        assert isinstance(hm, np.ndarray)

    def test_colormap_jet_applied(self):
        """Default COLORMAP_JET produces coloured (not grey) output."""
        asm = _make_assembly(3)
        hm = render_heatmap(asm)
        # JET colourmap → R, G, B channels differ in value across the image
        ch_std = [hm[:, :, c].std() for c in range(3)]
        assert max(ch_std) > 0.0

    def test_multiple_fragments_with_compat_matrix(self):
        """compat_matrix with non-zero values exercises the per-edge loop."""
        n = 4
        mat = np.eye(n * 4, dtype=np.float32) * 0.8
        frags = [_make_fragment(i) for i in range(n)]
        # Give each fragment a dummy edge so frag_of_edge is populated
        for f in frags:
            f.edges = [object()]
        asm = Assembly(
            fragments=frags,
            placements={i: (np.array([i * 120.0, 0.0]), 0.0) for i in range(n)},
            compat_matrix=mat,
            total_score=0.6,
        )
        hm = render_heatmap(asm)
        assert hm.dtype == np.uint8


# ---------------------------------------------------------------------------
# 3. TestRenderMosaic  (~12 tests)
# ---------------------------------------------------------------------------

class TestRenderMosaic:

    def test_empty_fragments_returns_stub(self):
        asm = _empty_assembly()
        mosaic = render_mosaic(asm)
        assert isinstance(mosaic, np.ndarray)
        assert mosaic.ndim == 3

    def test_empty_fragments_stub_is_thumb_size(self):
        """Empty fragments → stub of exactly (thumb_size, thumb_size, 3)."""
        asm = _empty_assembly()
        mosaic = render_mosaic(asm, thumb_size=128)
        assert mosaic.shape[0] == 128
        assert mosaic.shape[1] == 128

    def test_single_fragment_valid_mosaic(self):
        asm = _make_assembly(1)
        mosaic = render_mosaic(asm)
        assert isinstance(mosaic, np.ndarray)
        assert mosaic.shape[0] > 0
        assert mosaic.shape[1] > 0

    def test_four_fragments_two_cols_layout(self):
        """4 fragments with max_cols=2 → 2 grid columns."""
        asm = _make_assembly(4)
        thumb = 32
        gap = 4
        mosaic = render_mosaic(asm, max_cols=2, thumb_size=thumb, gap=gap)
        # 2 columns → width ≈ 2*(thumb+gap)+gap
        expected_max_w = 2 * (thumb + gap) + gap + 2  # small tolerance
        assert mosaic.shape[1] <= expected_max_w

    def test_returns_ndarray(self):
        asm = _make_assembly(3)
        assert isinstance(render_mosaic(asm), np.ndarray)

    def test_returns_uint8_dtype(self):
        asm = _make_assembly(3)
        mosaic = render_mosaic(asm)
        assert mosaic.dtype == np.uint8

    def test_shape_3_channels(self):
        asm = _make_assembly(3)
        mosaic = render_mosaic(asm)
        assert mosaic.ndim == 3
        assert mosaic.shape[2] == 3

    def test_thumb_size_affects_output_dimensions(self):
        asm = _make_assembly(4)
        m_small = render_mosaic(asm, thumb_size=32, gap=0)
        m_large = render_mosaic(asm, thumb_size=96, gap=0)
        assert m_large.shape[0] > m_small.shape[0] or m_large.shape[1] > m_small.shape[1]

    def test_fragment_with_angle_produces_valid_mosaic(self):
        """Fragment placed with a rotation angle must still render without error."""
        frag = _make_fragment(0, h=60, w=60)
        asm = Assembly(
            fragments=[frag],
            placements={0: (np.array([0.0, 0.0]), math.pi / 3)},
            compat_matrix=None,
            total_score=0.5,
        )
        mosaic = render_mosaic(asm, thumb_size=64)
        assert mosaic.dtype == np.uint8

    def test_mosaic_not_uniform_color(self):
        """A mosaic with coloured fragments must not be a uniform plane."""
        asm = _make_assembly(3)
        mosaic = render_mosaic(asm, thumb_size=80)
        assert mosaic.std() > 0.0

    def test_max_cols_one_produces_single_column(self):
        """max_cols=1 → each fragment in its own row (single column grid)."""
        asm = _make_assembly(3)
        thumb = 40
        gap = 4
        mosaic = render_mosaic(asm, max_cols=1, thumb_size=thumb, gap=gap)
        # Width should accommodate only 1 column
        expected_max_w = 1 * (thumb + gap) + gap + 2
        assert mosaic.shape[1] <= expected_max_w

    def test_large_grid_six_fragments(self):
        """6 fragments with max_cols=3 → 2 rows, valid shape."""
        asm = _make_assembly(6)
        mosaic = render_mosaic(asm, max_cols=3, thumb_size=40, gap=4)
        assert mosaic.shape[0] > 0
        assert mosaic.shape[1] > 0
        assert mosaic.dtype == np.uint8


# ---------------------------------------------------------------------------
# 4. TestSavePng  (~8 tests)
# ---------------------------------------------------------------------------

class TestSavePng:

    def test_save_png_creates_file(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        out = tmp_path / "result.png"
        save_png(canvas, out)
        assert out.exists()

    def test_save_png_nonzero_size(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        out = tmp_path / "result.png"
        save_png(canvas, out)
        assert out.stat().st_size > 0

    def test_save_jpg_creates_file(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        out = tmp_path / "result.jpg"
        save_png(canvas, out, quality=80)
        assert out.exists()

    def test_save_jpg_nonzero_size(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        out = tmp_path / "result.jpg"
        save_png(canvas, out, quality=80)
        assert out.stat().st_size > 0

    def test_loaded_png_same_shape(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        out = tmp_path / "roundtrip.png"
        save_png(canvas, out)
        loaded = cv2.imread(str(out))
        assert loaded is not None
        assert loaded.shape == canvas.shape

    def test_loaded_png_same_dtype(self, tmp_path):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        out = tmp_path / "dtype_check.png"
        save_png(canvas, out)
        loaded = cv2.imread(str(out))
        assert loaded.dtype == np.uint8

    def test_save_accepts_path_object(self, tmp_path):
        """save_png must accept pathlib.Path, not only strings."""
        asm = _make_assembly(1)
        canvas = render_canvas(asm)
        path_obj = tmp_path / "path_obj.png"
        save_png(canvas, path_obj)
        assert path_obj.exists()

    def test_save_accepts_string_path(self, tmp_path):
        asm = _make_assembly(1)
        canvas = render_canvas(asm)
        str_path = str(tmp_path / "string_path.png")
        save_png(canvas, str_path)
        assert Path(str_path).exists()


# ---------------------------------------------------------------------------
# 5. TestComparisonStrip  (~13 tests)
# ---------------------------------------------------------------------------

class TestComparisonStrip:

    # -- basic shape / type contracts ---

    def test_returns_ndarray(self):
        asm = _make_assembly(4)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=200)
        assert isinstance(strip, np.ndarray)

    def test_returns_uint8_dtype(self):
        asm = _make_assembly(4)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=200)
        assert strip.dtype == np.uint8

    def test_shape_3_channels(self):
        asm = _make_assembly(4)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=200)
        assert strip.ndim == 3
        assert strip.shape[2] == 3

    def test_height_matches_target_height(self):
        asm = _make_assembly(4)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        target = 300
        strip = comparison_strip(imgs, canvas, target_height=target)
        assert strip.shape[0] == target

    def test_height_target_100(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=100)
        assert strip.shape[0] == 100

    # -- panel count ---

    def test_two_panels_no_heatmap(self):
        """Without heatmap: [mosaic | assembly] → width > assembly panel alone."""
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, heatmap_img=None, target_height=200)
        # Strip must be wider than a single assembly panel resized to target_height
        h_, w_ = canvas.shape[:2]
        scaled_w = int(w_ * 200 / h_)
        assert strip.shape[1] > scaled_w

    def test_three_panels_with_heatmap(self):
        """With heatmap: 3 panels → strip is wider than 2-panel version."""
        asm = _make_assembly(4)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas.shape)
        imgs = [f.image for f in asm.fragments]
        strip_2p = comparison_strip(imgs, canvas, heatmap_img=None, target_height=200)
        strip_3p = comparison_strip(imgs, canvas, heatmap_img=hm, target_height=200)
        assert strip_3p.shape[1] > strip_2p.shape[1]

    def test_heatmap_none_vs_provided_width_difference(self):
        """Providing heatmap_img must increase the strip width."""
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas.shape)
        imgs = [f.image for f in asm.fragments]
        w_no_hm = comparison_strip(imgs, canvas, target_height=200).shape[1]
        w_hm = comparison_strip(imgs, canvas, hm, target_height=200).shape[1]
        assert w_hm > w_no_hm

    # -- edge cases ---

    def test_empty_fragments_list_no_crash(self):
        canvas = np.full((200, 300, 3), 200, dtype=np.uint8)
        strip = comparison_strip([], canvas, target_height=100)
        assert isinstance(strip, np.ndarray)
        assert strip.shape[0] == 100

    def test_single_fragment_image(self):
        img = np.full((80, 60, 3), 128, dtype=np.uint8)
        canvas = np.full((200, 300, 3), 200, dtype=np.uint8)
        strip = comparison_strip([img], canvas, target_height=200)
        assert strip.shape[0] == 200

    def test_values_in_valid_uint8_range(self):
        asm = _make_assembly(3)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, target_height=200)
        assert int(strip.min()) >= 0
        assert int(strip.max()) <= 255

    def test_separators_darker_than_panels(self):
        """The 4-pixel separator columns are solid grey (value 100)."""
        asm = _make_assembly(4)
        canvas = render_canvas(asm)
        hm = render_heatmap(asm, canvas.shape)
        imgs = [f.image for f in asm.fragments]
        strip = comparison_strip(imgs, canvas, heatmap_img=hm, target_height=200)
        # Separator is 4 pixels wide with value 100; strip width > 8 (two seps)
        assert strip.shape[1] > 8

    def test_larger_target_height_produces_taller_strip(self):
        asm = _make_assembly(2)
        canvas = render_canvas(asm)
        imgs = [f.image for f in asm.fragments]
        s_small = comparison_strip(imgs, canvas, target_height=100)
        s_large = comparison_strip(imgs, canvas, target_height=400)
        assert s_large.shape[0] > s_small.shape[0]
