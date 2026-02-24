"""Extra tests for puzzle_reconstruction/utils/visualizer.py."""
from __future__ import annotations

from types import SimpleNamespace
import pytest
import numpy as np
import cv2

from puzzle_reconstruction.utils.visualizer import (
    VisConfig,
    draw_word_boxes,
    draw_fragment_boxes,
    draw_edge_matches,
    draw_contour,
    draw_assembly_layout,
    draw_skew_angle,
    draw_confidence_bar,
    tile_images,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _word(x=5, y=5, w=20, h=10, line_idx=0):
    return SimpleNamespace(x=x, y=y, w=w, h=h, line_idx=line_idx)


def _frag_box(x=5, y=5, x2=25, y2=20, fid=0):
    return SimpleNamespace(x=x, y=y, x2=x2, y2=y2, fid=fid)


def _match(pt_src=(5, 10), pt_dst=(15, 10), confidence=0.8):
    return SimpleNamespace(pt_src=pt_src, pt_dst=pt_dst, confidence=confidence)


def _contour_pts(n=8) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 20
    xs = (32 + r * np.cos(t)).astype(np.int32)
    ys = (32 + r * np.sin(t)).astype(np.int32)
    return np.column_stack([xs, ys])


# ─── VisConfig ────────────────────────────────────────────────────────────────

class TestVisConfigExtra:
    def test_default_line_thickness(self):
        assert VisConfig().line_thickness == 1

    def test_default_font_scale(self):
        assert VisConfig().font_scale == pytest.approx(0.4)

    def test_default_tile_gap(self):
        assert VisConfig().tile_gap == 4

    def test_default_word_box_color(self):
        assert VisConfig().word_box_color == (0, 200, 0)

    def test_default_frag_box_color(self):
        assert VisConfig().frag_box_color == (255, 80, 0)

    def test_default_bg_color(self):
        assert VisConfig().bg_color == (255, 255, 255)

    def test_custom_thickness(self):
        cfg = VisConfig(line_thickness=3)
        assert cfg.line_thickness == 3

    def test_custom_font_scale(self):
        cfg = VisConfig(font_scale=1.0)
        assert cfg.font_scale == pytest.approx(1.0)


# ─── draw_word_boxes ──────────────────────────────────────────────────────────

class TestDrawWordBoxesExtra:
    def test_returns_ndarray(self):
        out = draw_word_boxes(_bgr(), [_word()])
        assert isinstance(out, np.ndarray)

    def test_dtype_uint8(self):
        assert draw_word_boxes(_bgr(), [_word()]).dtype == np.uint8

    def test_shape_preserved(self):
        img = _bgr(48, 80)
        out = draw_word_boxes(img, [_word()])
        assert out.shape == (48, 80, 3)

    def test_3_channels(self):
        out = draw_word_boxes(_bgr(), [_word()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_input_not_modified(self):
        img = _bgr()
        original = img.copy()
        draw_word_boxes(img, [_word()])
        np.testing.assert_array_equal(img, original)

    def test_grayscale_converted(self):
        out = draw_word_boxes(_gray(), [_word()])
        assert out.ndim == 3

    def test_multiple_words(self):
        words = [_word(x=i * 10, y=5) for i in range(3)]
        out = draw_word_boxes(_bgr(), words)
        assert out.shape[2] == 3

    def test_custom_cfg(self):
        cfg = VisConfig(word_box_color=(255, 0, 0))
        out = draw_word_boxes(_bgr(), [_word()], cfg=cfg)
        assert isinstance(out, np.ndarray)

    def test_label_false(self):
        out = draw_word_boxes(_bgr(), [_word()], label=False)
        assert out.ndim == 3


# ─── draw_fragment_boxes ──────────────────────────────────────────────────────

class TestDrawFragmentBoxesExtra:
    def test_returns_ndarray(self):
        out = draw_fragment_boxes(_bgr(), [_frag_box()])
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        img = _bgr(48, 80)
        out = draw_fragment_boxes(img, [_frag_box()])
        assert out.shape == (48, 80, 3)

    def test_dtype_uint8(self):
        assert draw_fragment_boxes(_bgr(), [_frag_box()]).dtype == np.uint8

    def test_empty_list(self):
        out = draw_fragment_boxes(_bgr(), [])
        assert out.shape == _bgr().shape

    def test_grayscale_input(self):
        out = draw_fragment_boxes(_gray(), [_frag_box()])
        assert out.ndim == 3

    def test_multiple_boxes(self):
        boxes = [_frag_box(x=i * 20, y=5, x2=i * 20 + 15, y2=20, fid=i) for i in range(3)]
        out = draw_fragment_boxes(_bgr(), boxes)
        assert isinstance(out, np.ndarray)

    def test_custom_cfg(self):
        cfg = VisConfig(frag_box_color=(0, 0, 255))
        out = draw_fragment_boxes(_bgr(), [_frag_box()], cfg=cfg)
        assert isinstance(out, np.ndarray)


# ─── draw_edge_matches ────────────────────────────────────────────────────────

class TestDrawEdgeMatchesExtra:
    def test_returns_ndarray(self):
        out = draw_edge_matches(_bgr(), _bgr(), [_match()])
        assert isinstance(out, np.ndarray)

    def test_3_channels(self):
        out = draw_edge_matches(_bgr(), _bgr(), [_match()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_dtype_uint8(self):
        assert draw_edge_matches(_bgr(), _bgr(), [_match()]).dtype == np.uint8

    def test_empty_matches(self):
        out = draw_edge_matches(_bgr(), _bgr(), [])
        assert isinstance(out, np.ndarray)

    def test_canvas_wider_than_images(self):
        img1 = _bgr(40, 50)
        img2 = _bgr(40, 60)
        out = draw_edge_matches(img1, img2, [_match()])
        assert out.shape[1] >= 50 + 60

    def test_canvas_height_max(self):
        img1 = _bgr(40, 50)
        img2 = _bgr(60, 50)
        out = draw_edge_matches(img1, img2, [_match()])
        assert out.shape[0] >= 60

    def test_max_matches_limit(self):
        matches = [_match() for _ in range(20)]
        out = draw_edge_matches(_bgr(), _bgr(), matches, max_matches=5)
        assert isinstance(out, np.ndarray)


# ─── draw_contour ─────────────────────────────────────────────────────────────

class TestDrawContourExtra:
    def test_returns_ndarray(self):
        pts = _contour_pts()
        out = draw_contour(_bgr(), pts)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        img = _bgr(48, 80)
        out = draw_contour(img, _contour_pts())
        assert out.shape == (48, 80, 3)

    def test_dtype_uint8(self):
        assert draw_contour(_bgr(), _contour_pts()).dtype == np.uint8

    def test_input_not_modified(self):
        img = _bgr()
        original = img.copy()
        draw_contour(img, _contour_pts())
        np.testing.assert_array_equal(img, original)

    def test_grayscale_converted(self):
        out = draw_contour(_gray(), _contour_pts())
        assert out.ndim == 3

    def test_n1_2_format(self):
        pts = _contour_pts().reshape(-1, 1, 2)
        out = draw_contour(_bgr(), pts)
        assert isinstance(out, np.ndarray)

    def test_fill_alpha_positive(self):
        pts = _contour_pts()
        out = draw_contour(_bgr(), pts, fill_alpha=0.5)
        assert isinstance(out, np.ndarray)

    def test_closed_false(self):
        pts = _contour_pts()
        out = draw_contour(_bgr(), pts, closed=False)
        assert isinstance(out, np.ndarray)


# ─── draw_assembly_layout ─────────────────────────────────────────────────────

class TestDrawAssemblyLayoutExtra:
    def test_returns_ndarray(self):
        out = draw_assembly_layout([_frag_box()])
        assert isinstance(out, np.ndarray)

    def test_3_channels(self):
        out = draw_assembly_layout([_frag_box()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_dtype_uint8(self):
        assert draw_assembly_layout([_frag_box()]).dtype == np.uint8

    def test_canvas_size(self):
        out = draw_assembly_layout([_frag_box()], canvas_wh=(300, 200))
        assert out.shape == (200, 300, 3)

    def test_empty_boxes(self):
        out = draw_assembly_layout([])
        assert isinstance(out, np.ndarray)

    def test_many_boxes_palette_cycles(self):
        boxes = [_frag_box(x=i*5, y=i*5, x2=i*5+4, y2=i*5+4, fid=i) for i in range(12)]
        out = draw_assembly_layout(boxes)
        assert isinstance(out, np.ndarray)

    def test_custom_cfg(self):
        cfg = VisConfig(bg_color=(0, 0, 0))
        out = draw_assembly_layout([_frag_box()], cfg=cfg)
        assert isinstance(out, np.ndarray)


# ─── draw_skew_angle ──────────────────────────────────────────────────────────

class TestDrawSkewAngleExtra:
    def test_returns_ndarray(self):
        out = draw_skew_angle(_bgr(), 5.0)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        img = _bgr(48, 80)
        out = draw_skew_angle(img, 3.0)
        assert out.shape == (48, 80, 3)

    def test_dtype_uint8(self):
        assert draw_skew_angle(_bgr(), 0.0).dtype == np.uint8

    def test_input_not_modified(self):
        img = _bgr()
        original = img.copy()
        draw_skew_angle(img, 10.0)
        np.testing.assert_array_equal(img, original)

    def test_zero_angle(self):
        out = draw_skew_angle(_bgr(), 0.0)
        assert isinstance(out, np.ndarray)

    def test_negative_angle(self):
        out = draw_skew_angle(_bgr(), -5.0)
        assert isinstance(out, np.ndarray)

    def test_grayscale_converted(self):
        out = draw_skew_angle(_gray(), 2.0)
        assert out.ndim == 3


# ─── draw_confidence_bar ──────────────────────────────────────────────────────

class TestDrawConfidenceBarExtra:
    def test_returns_ndarray(self):
        out = draw_confidence_bar(_bgr(), 0.5)
        assert isinstance(out, np.ndarray)

    def test_height_increased_by_bar_height(self):
        img = _bgr(40, 60)
        out = draw_confidence_bar(img, 0.5, bar_height=20)
        assert out.shape[0] == 60  # 40 + 20

    def test_width_preserved(self):
        img = _bgr(40, 60)
        out = draw_confidence_bar(img, 0.5)
        assert out.shape[1] == 60

    def test_3_channels(self):
        out = draw_confidence_bar(_bgr(), 0.7)
        assert out.ndim == 3

    def test_dtype_uint8(self):
        assert draw_confidence_bar(_bgr(), 0.5).dtype == np.uint8

    def test_confidence_one(self):
        out = draw_confidence_bar(_bgr(), 1.0)
        assert isinstance(out, np.ndarray)

    def test_confidence_zero(self):
        out = draw_confidence_bar(_bgr(), 0.0)
        assert isinstance(out, np.ndarray)

    def test_grade_label(self):
        out = draw_confidence_bar(_bgr(), 0.8, grade="A")
        assert isinstance(out, np.ndarray)

    def test_custom_bar_height(self):
        img = _bgr(32, 32)
        out = draw_confidence_bar(img, 0.5, bar_height=30)
        assert out.shape[0] == 62


# ─── tile_images ──────────────────────────────────────────────────────────────

class TestTileImagesExtra:
    def test_returns_ndarray(self):
        assert isinstance(tile_images([_bgr()]), np.ndarray)

    def test_dtype_uint8(self):
        assert tile_images([_bgr()]).dtype == np.uint8

    def test_3_channels(self):
        out = tile_images([_bgr()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_empty_list_placeholder(self):
        out = tile_images([])
        assert out.shape == (100, 100, 3)

    def test_single_image_shape(self):
        img = _bgr(40, 60)
        out = tile_images([img], n_cols=1)
        assert out.shape[0] >= 40 and out.shape[1] >= 60

    def test_grayscale_converted(self):
        out = tile_images([_gray()])
        assert out.ndim == 3

    def test_two_images_side_by_side(self):
        img = _bgr(32, 32)
        out = tile_images([img, img], n_cols=2)
        assert out.shape[1] >= 64  # at least 2 * 32

    def test_labels(self):
        out = tile_images([_bgr(), _bgr()], labels=["a", "b"])
        assert isinstance(out, np.ndarray)

    def test_single_column(self):
        imgs = [_bgr(16, 16)] * 3
        out = tile_images(imgs, n_cols=1)
        assert out.shape[0] >= 48  # 3 * 16

    def test_multiple_rows(self):
        imgs = [_bgr(16, 16)] * 4
        out = tile_images(imgs, n_cols=2)
        # 2 rows with gap
        assert out.shape[0] > 16
