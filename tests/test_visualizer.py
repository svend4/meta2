"""
Тесты для puzzle_reconstruction/utils/visualizer.py

Покрывает:
    VisConfig             — значения по умолчанию, кастомные параметры
    draw_word_boxes       — форма, dtype, нет изменения входа, grayscale
    draw_fragment_boxes   — форма, dtype, label on/off, grayscale
    draw_edge_matches     — ширина canvas, высота max, dtype, max_matches
    draw_contour          — форма, fill_alpha, открытый контур, (N,1,2) формат
    draw_assembly_layout  — размер canvas, dtype, цикл палитры, out_of_bounds
    draw_skew_angle       — форма, dtype, нулевой/отрицательный угол
    draw_confidence_bar   — высота, ширина, gradient, grade, clipping
    tile_images           — пустой список, n_cols, labels, размеры, grayscale
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

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


# ─── Stub-классы для тестов ───────────────────────────────────────────────────

@dataclass
class _WordBox:
    x: int; y: int; w: int; h: int
    line_idx: int = 0
    confidence: float = 1.0


@dataclass
class _FragBox:
    fid: int
    x: float; y: float; w: float; h: float
    rotation: float = 0.0

    @property
    def x2(self): return self.x + self.w

    @property
    def y2(self): return self.y + self.h


@dataclass
class _KpMatch:
    pt_src: Tuple[float, float]
    pt_dst: Tuple[float, float]
    distance: float
    confidence: float = 0.8


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def white100():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def gray80():
    return np.full((80, 80), 128, dtype=np.uint8)


@pytest.fixture
def words():
    return [
        _WordBox(x=5,  y=5,  w=30, h=15, line_idx=0),
        _WordBox(x=40, y=5,  w=25, h=15, line_idx=0),
        _WordBox(x=5,  y=30, w=35, h=15, line_idx=1),
    ]


@pytest.fixture
def frag_boxes():
    return [
        _FragBox(fid=1, x=10., y=10., w=60., h=40.),
        _FragBox(fid=2, x=80., y=10., w=60., h=40.),
    ]


@pytest.fixture
def matches():
    return [
        _KpMatch(pt_src=(10., 20.), pt_dst=(15., 25.),
                 distance=5.0, confidence=0.9),
        _KpMatch(pt_src=(30., 40.), pt_dst=(35., 45.),
                 distance=20.0, confidence=0.3),
    ]


@pytest.fixture
def square_contour():
    return np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)


# ─── VisConfig ────────────────────────────────────────────────────────────────

class TestVisConfig:
    def test_word_box_color_default(self):
        cfg = VisConfig()
        assert cfg.word_box_color == (0, 200, 0)

    def test_frag_box_color_default(self):
        cfg = VisConfig()
        assert cfg.frag_box_color == (255, 80, 0)

    def test_line_thickness_default(self):
        assert VisConfig().line_thickness == 1

    def test_font_scale_default(self):
        assert VisConfig().font_scale == pytest.approx(0.4)

    def test_tile_gap_default(self):
        assert VisConfig().tile_gap == 4

    def test_bg_color_default(self):
        assert VisConfig().bg_color == (255, 255, 255)

    def test_tile_bg_default(self):
        assert VisConfig().tile_bg == (200, 200, 200)

    def test_custom_colors(self):
        cfg = VisConfig(word_box_color=(1, 2, 3), line_thickness=3)
        assert cfg.word_box_color == (1, 2, 3)
        assert cfg.line_thickness == 3
        assert cfg.font_scale == pytest.approx(0.4)  # остальное по умолчанию


# ─── draw_word_boxes ──────────────────────────────────────────────────────────

class TestDrawWordBoxes:
    def test_returns_ndarray(self, white100, words):
        out = draw_word_boxes(white100, words)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self, white100, words):
        out = draw_word_boxes(white100, words)
        assert out.shape == white100.shape

    def test_dtype_uint8(self, white100, words):
        out = draw_word_boxes(white100, words)
        assert out.dtype == np.uint8

    def test_does_not_modify_input(self, white100, words):
        orig = white100.copy()
        draw_word_boxes(white100, words)
        np.testing.assert_array_equal(white100, orig)

    def test_grayscale_converted_to_bgr(self, gray80, words):
        out = draw_word_boxes(gray80, words)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_empty_words_list(self, white100):
        out = draw_word_boxes(white100, [])
        assert out.shape == white100.shape

    def test_label_false_no_crash(self, white100, words):
        out = draw_word_boxes(white100, words, label=False)
        assert isinstance(out, np.ndarray)

    def test_negative_line_idx_skips_label(self, white100):
        wbs = [_WordBox(x=5, y=5, w=20, h=10, line_idx=-1)]
        out = draw_word_boxes(white100, wbs, label=True)
        assert isinstance(out, np.ndarray)

    def test_custom_cfg(self, white100, words):
        cfg = VisConfig(word_box_color=(0, 0, 255), line_thickness=2)
        out = draw_word_boxes(white100, words, cfg=cfg)
        assert isinstance(out, np.ndarray)


# ─── draw_fragment_boxes ──────────────────────────────────────────────────────

class TestDrawFragmentBoxes:
    def test_returns_ndarray(self, frag_boxes):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        out = draw_fragment_boxes(canvas, frag_boxes)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self, frag_boxes):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        out = draw_fragment_boxes(canvas, frag_boxes)
        assert out.shape == canvas.shape

    def test_dtype_uint8(self, frag_boxes):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        out = draw_fragment_boxes(canvas, frag_boxes)
        assert out.dtype == np.uint8

    def test_empty_boxes(self, white100):
        out = draw_fragment_boxes(white100, [])
        assert out.shape == white100.shape

    def test_label_true_contains_fid(self):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        boxes = [_FragBox(fid=99, x=10., y=10., w=50., h=30.)]
        out = draw_fragment_boxes(canvas, boxes, label=True)
        assert isinstance(out, np.ndarray)

    def test_label_false_no_crash(self):
        canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
        boxes = [_FragBox(fid=1, x=10., y=10., w=50., h=30.)]
        out = draw_fragment_boxes(canvas, boxes, label=False)
        assert isinstance(out, np.ndarray)

    def test_grayscale_input_converted(self, gray80):
        out = draw_fragment_boxes(gray80, [])
        assert out.ndim == 3


# ─── draw_edge_matches ────────────────────────────────────────────────────────

class TestDrawEdgeMatches:
    def test_canvas_width(self, white100, matches):
        img2 = np.full((100, 80, 3), 200, dtype=np.uint8)
        cfg = VisConfig(tile_gap=4)
        out = draw_edge_matches(white100, img2, matches, cfg=cfg)
        assert out.shape[1] == 100 + 4 + 80

    def test_canvas_height_is_max(self, matches):
        img1 = np.full((80, 100, 3), 255, dtype=np.uint8)
        img2 = np.full((120, 100, 3), 200, dtype=np.uint8)
        out = draw_edge_matches(img1, img2, matches)
        assert out.shape[0] == 120

    def test_returns_bgr(self, white100, matches):
        out = draw_edge_matches(white100, white100, matches)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_dtype_uint8(self, white100, matches):
        out = draw_edge_matches(white100, white100, matches)
        assert out.dtype == np.uint8

    def test_empty_matches_no_crash(self, white100):
        out = draw_edge_matches(white100, white100, [])
        assert isinstance(out, np.ndarray)

    def test_max_matches_limit(self, white100):
        many = [_KpMatch((float(i), 0.), (float(i), 0.), 1.0)
                for i in range(100)]
        out = draw_edge_matches(white100, white100, many, max_matches=5)
        assert isinstance(out, np.ndarray)

    def test_grayscale_images_converted(self, gray80, matches):
        out = draw_edge_matches(gray80, gray80, matches)
        assert out.ndim == 3

    def test_equal_size_images_symmetric_width(self, white100, matches):
        cfg = VisConfig(tile_gap=0)
        out = draw_edge_matches(white100, white100, matches, cfg=cfg)
        assert out.shape[1] == 100 + 0 + 100


# ─── draw_contour ─────────────────────────────────────────────────────────────

class TestDrawContour:
    def test_returns_ndarray(self, white100, square_contour):
        out = draw_contour(white100, square_contour)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self, white100, square_contour):
        out = draw_contour(white100, square_contour)
        assert out.shape == white100.shape

    def test_dtype_uint8(self, white100, square_contour):
        out = draw_contour(white100, square_contour)
        assert out.dtype == np.uint8

    def test_does_not_modify_input(self, white100, square_contour):
        orig = white100.copy()
        draw_contour(white100, square_contour)
        np.testing.assert_array_equal(white100, orig)

    def test_fill_alpha_zero(self, white100, square_contour):
        out = draw_contour(white100, square_contour, fill_alpha=0.0)
        assert isinstance(out, np.ndarray)

    def test_fill_alpha_positive(self, white100, square_contour):
        out = draw_contour(white100, square_contour, fill_alpha=0.4)
        assert isinstance(out, np.ndarray)

    def test_open_contour(self, white100, square_contour):
        out = draw_contour(white100, square_contour, closed=False)
        assert out.shape == white100.shape

    def test_3d_contour_format(self, white100):
        pts = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]],
                       dtype=np.int32)
        out = draw_contour(white100, pts)
        assert out.shape == white100.shape

    def test_grayscale_input_converted(self, gray80, square_contour):
        # square_contour создан для 100×100, обрежем до 80×80
        c = np.clip(square_contour, 0, 79)
        out = draw_contour(gray80, c)
        assert out.ndim == 3


# ─── draw_assembly_layout ─────────────────────────────────────────────────────

class TestDrawAssemblyLayout:
    def test_returns_ndarray(self, frag_boxes):
        out = draw_assembly_layout(frag_boxes, canvas_wh=(300, 200))
        assert isinstance(out, np.ndarray)

    def test_canvas_shape(self, frag_boxes):
        out = draw_assembly_layout(frag_boxes, canvas_wh=(300, 200))
        assert out.shape == (200, 300, 3)

    def test_dtype_uint8(self, frag_boxes):
        out = draw_assembly_layout(frag_boxes, canvas_wh=(300, 200))
        assert out.dtype == np.uint8

    def test_empty_boxes(self):
        out = draw_assembly_layout([], canvas_wh=(150, 100))
        assert out.shape == (100, 150, 3)

    def test_many_boxes_palette_cycles(self):
        boxes = [_FragBox(fid=i, x=float(i * 8), y=float(i * 8),
                          w=20., h=15.) for i in range(12)]
        out = draw_assembly_layout(boxes, canvas_wh=(300, 300))
        assert isinstance(out, np.ndarray)

    def test_box_outside_canvas_no_crash(self):
        boxes = [_FragBox(fid=1, x=800., y=800., w=100., h=50.)]
        out = draw_assembly_layout(boxes, canvas_wh=(200, 200))
        assert isinstance(out, np.ndarray)

    def test_default_canvas_wh(self, frag_boxes):
        out = draw_assembly_layout(frag_boxes)
        assert out.shape == (400, 500, 3)

    def test_single_box(self):
        boxes = [_FragBox(fid=7, x=50., y=50., w=100., h=80.)]
        out = draw_assembly_layout(boxes, canvas_wh=(300, 250))
        assert isinstance(out, np.ndarray)


# ─── draw_skew_angle ──────────────────────────────────────────────────────────

class TestDrawSkewAngle:
    def test_returns_ndarray(self, white100):
        out = draw_skew_angle(white100, 5.0)
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self, white100):
        out = draw_skew_angle(white100, 5.0)
        assert out.shape == white100.shape

    def test_dtype_uint8(self, white100):
        out = draw_skew_angle(white100, 5.0)
        assert out.dtype == np.uint8

    def test_zero_angle(self, white100):
        out = draw_skew_angle(white100, 0.0)
        assert isinstance(out, np.ndarray)

    def test_negative_angle(self, white100):
        out = draw_skew_angle(white100, -20.0)
        assert isinstance(out, np.ndarray)

    def test_large_angle(self, white100):
        out = draw_skew_angle(white100, 45.0)
        assert isinstance(out, np.ndarray)

    def test_grayscale_converted(self, gray80):
        out = draw_skew_angle(gray80, 10.0)
        assert out.ndim == 3

    def test_does_not_modify_input(self, white100):
        orig = white100.copy()
        draw_skew_angle(white100, 3.0)
        np.testing.assert_array_equal(white100, orig)


# ─── draw_confidence_bar ──────────────────────────────────────────────────────

class TestDrawConfidenceBar:
    def test_returns_ndarray(self, white100):
        out = draw_confidence_bar(white100, 0.75)
        assert isinstance(out, np.ndarray)

    def test_height_increased_by_bar_height(self, white100):
        bar_h = 20
        out = draw_confidence_bar(white100, 0.75, bar_height=bar_h)
        assert out.shape[0] == white100.shape[0] + bar_h

    def test_custom_bar_height(self, white100):
        out = draw_confidence_bar(white100, 0.5, bar_height=30)
        assert out.shape[0] == white100.shape[0] + 30

    def test_width_preserved(self, white100):
        out = draw_confidence_bar(white100, 0.75)
        assert out.shape[1] == white100.shape[1]

    def test_dtype_uint8(self, white100):
        out = draw_confidence_bar(white100, 0.5)
        assert out.dtype == np.uint8

    def test_zero_confidence(self, white100):
        out = draw_confidence_bar(white100, 0.0)
        assert out.shape[0] > white100.shape[0]

    def test_full_confidence(self, white100):
        out = draw_confidence_bar(white100, 1.0)
        assert isinstance(out, np.ndarray)

    def test_with_grade_label(self, white100):
        out = draw_confidence_bar(white100, 0.85, grade="A")
        assert isinstance(out, np.ndarray)

    def test_confidence_above_one_clipped(self, white100):
        out = draw_confidence_bar(white100, 1.5)
        assert isinstance(out, np.ndarray)

    def test_negative_confidence_clipped(self, white100):
        out = draw_confidence_bar(white100, -0.5)
        assert isinstance(out, np.ndarray)

    def test_channels_preserved(self, white100):
        out = draw_confidence_bar(white100, 0.6)
        assert out.ndim == 3
        assert out.shape[2] == 3


# ─── tile_images ──────────────────────────────────────────────────────────────

class TestTileImages:
    def test_empty_list_returns_placeholder(self):
        out = tile_images([])
        assert isinstance(out, np.ndarray)
        assert out.ndim == 3

    def test_single_image(self, white100):
        out = tile_images([white100], n_cols=1)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 3

    def test_dtype_uint8(self):
        imgs = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(4)]
        out = tile_images(imgs, n_cols=2)
        assert out.dtype == np.uint8

    def test_width_with_gap(self):
        imgs = [np.zeros((50, 60, 3), dtype=np.uint8) for _ in range(3)]
        cfg = VisConfig(tile_gap=4)
        out = tile_images(imgs, n_cols=3, cfg=cfg)
        assert out.shape[1] == 3 * 60 + 2 * 4

    def test_height_two_rows(self):
        imgs = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(6)]
        cfg = VisConfig(tile_gap=4)
        out = tile_images(imgs, n_cols=3, cfg=cfg)
        assert out.shape[0] == 2 * 50 + 1 * 4

    def test_with_labels(self):
        imgs = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        out = tile_images(imgs, n_cols=3, labels=["A", "B", "C"])
        assert isinstance(out, np.ndarray)

    def test_fewer_labels_than_images(self):
        imgs = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        out = tile_images(imgs, n_cols=3, labels=["X"])
        assert isinstance(out, np.ndarray)

    def test_grayscale_converted_to_bgr(self):
        imgs = [np.zeros((50, 50), dtype=np.uint8) for _ in range(2)]
        out = tile_images(imgs, n_cols=2)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_images_resized_to_first(self):
        img1 = np.zeros((50, 60, 3), dtype=np.uint8)
        img2 = np.zeros((80, 40, 3), dtype=np.uint8)
        cfg = VisConfig(tile_gap=0)
        out = tile_images([img1, img2], n_cols=2, cfg=cfg)
        assert out.shape[1] == 2 * 60  # оба размером первого

    def test_single_column(self):
        imgs = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(3)]
        cfg = VisConfig(tile_gap=2)
        out = tile_images(imgs, n_cols=1, cfg=cfg)
        assert out.shape[0] == 3 * 30 + 2 * 2
        assert out.shape[1] == 40
