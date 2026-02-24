"""Extra tests for puzzle_reconstruction/utils/visualizer.py."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _canvas(h=100, w=100, channels=3):
    return np.full((h, w, channels), 255, dtype=np.uint8)


def _gray_canvas(h=100, w=100):
    return np.full((h, w), 200, dtype=np.uint8)


@dataclass
class FakeWordBox:
    x: int = 10
    y: int = 10
    w: int = 40
    h: int = 15
    line_idx: int = 0


@dataclass
class FakeFragmentBox:
    fid: int = 0
    x: float = 10.0
    y: float = 10.0
    x2: float = 50.0
    y2: float = 50.0


@dataclass
class FakeKeypointMatch:
    pt_src: Tuple[float, float] = (10.0, 10.0)
    pt_dst: Tuple[float, float] = (20.0, 20.0)
    confidence: float = 0.8


# ─── VisConfig ───────────────────────────────────────────────────────────────

class TestVisConfigExtra:
    def test_defaults(self):
        cfg = VisConfig()
        assert cfg.line_thickness == 1
        assert cfg.tile_gap == 4
        assert cfg.font_scale == pytest.approx(0.4)

    def test_custom(self):
        cfg = VisConfig(line_thickness=3, tile_gap=10)
        assert cfg.line_thickness == 3
        assert cfg.tile_gap == 10


# ─── draw_word_boxes ─────────────────────────────────────────────────────────

class TestDrawWordBoxesExtra:
    def test_returns_bgr(self):
        img = _canvas()
        out = draw_word_boxes(img, [FakeWordBox()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_empty_words(self):
        img = _canvas()
        out = draw_word_boxes(img, [])
        assert out.shape == img.shape

    def test_grayscale_input(self):
        img = _gray_canvas()
        out = draw_word_boxes(img, [FakeWordBox()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_no_label(self):
        img = _canvas()
        out = draw_word_boxes(img, [FakeWordBox()], label=False)
        assert out.shape[:2] == img.shape[:2]

    def test_negative_line_idx_no_label(self):
        img = _canvas()
        wb = FakeWordBox(line_idx=-1)
        out = draw_word_boxes(img, [wb], label=True)
        assert out.shape[:2] == img.shape[:2]


# ─── draw_fragment_boxes ─────────────────────────────────────────────────────

class TestDrawFragmentBoxesExtra:
    def test_returns_bgr(self):
        img = _canvas()
        out = draw_fragment_boxes(img, [FakeFragmentBox()])
        assert out.ndim == 3 and out.shape[2] == 3

    def test_empty_boxes(self):
        img = _canvas()
        out = draw_fragment_boxes(img, [])
        assert out.shape == img.shape

    def test_no_label(self):
        img = _canvas()
        out = draw_fragment_boxes(img, [FakeFragmentBox()], label=False)
        assert out.shape[:2] == img.shape[:2]


# ─── draw_edge_matches ───────────────────────────────────────────────────────

class TestDrawEdgeMatchesExtra:
    def test_side_by_side_width(self):
        img1 = _canvas(50, 40)
        img2 = _canvas(50, 60)
        out = draw_edge_matches(img1, img2, [])
        cfg = VisConfig()
        assert out.shape[1] == 40 + cfg.tile_gap + 60

    def test_height_is_max(self):
        img1 = _canvas(30, 40)
        img2 = _canvas(50, 40)
        out = draw_edge_matches(img1, img2, [])
        assert out.shape[0] == 50

    def test_with_matches(self):
        img1 = _canvas(50, 50)
        img2 = _canvas(50, 50)
        m = FakeKeypointMatch()
        out = draw_edge_matches(img1, img2, [m])
        assert out.ndim == 3

    def test_max_matches_limits(self):
        img1 = _canvas(50, 50)
        img2 = _canvas(50, 50)
        matches = [FakeKeypointMatch() for _ in range(100)]
        out = draw_edge_matches(img1, img2, matches, max_matches=5)
        assert out.ndim == 3


# ─── draw_contour ────────────────────────────────────────────────────────────

class TestDrawContourExtra:
    def test_returns_bgr(self):
        img = _canvas()
        contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        out = draw_contour(img, contour)
        assert out.ndim == 3 and out.shape[2] == 3

    def test_3d_contour_input(self):
        img = _canvas()
        contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]]])
        out = draw_contour(img, contour)
        assert out.shape[:2] == img.shape[:2]

    def test_fill_alpha(self):
        img = _canvas()
        contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        out = draw_contour(img, contour, fill_alpha=0.5)
        assert out.shape == img.shape

    def test_grayscale_input(self):
        img = _gray_canvas()
        contour = np.array([[10, 10], [50, 10], [50, 50]])
        out = draw_contour(img, contour)
        assert out.ndim == 3


# ─── draw_assembly_layout ────────────────────────────────────────────────────

class TestDrawAssemblyLayoutExtra:
    def test_empty_boxes(self):
        out = draw_assembly_layout([])
        assert out.shape == (400, 500, 3)

    def test_custom_canvas_size(self):
        out = draw_assembly_layout([], canvas_wh=(200, 150))
        assert out.shape == (150, 200, 3)

    def test_with_boxes(self):
        boxes = [FakeFragmentBox(fid=0), FakeFragmentBox(fid=1)]
        out = draw_assembly_layout(boxes)
        assert out.ndim == 3


# ─── draw_skew_angle ─────────────────────────────────────────────────────────

class TestDrawSkewAngleExtra:
    def test_returns_bgr(self):
        img = _canvas()
        out = draw_skew_angle(img, 5.0)
        assert out.ndim == 3 and out.shape[2] == 3

    def test_zero_angle(self):
        img = _canvas()
        out = draw_skew_angle(img, 0.0)
        assert out.shape[:2] == img.shape[:2]

    def test_grayscale_input(self):
        img = _gray_canvas()
        out = draw_skew_angle(img, 10.0)
        assert out.ndim == 3


# ─── draw_confidence_bar ─────────────────────────────────────────────────────

class TestDrawConfidenceBarExtra:
    def test_adds_height(self):
        img = _canvas(50, 100)
        out = draw_confidence_bar(img, 0.8)
        assert out.shape[0] == 50 + 20  # default bar_height=20
        assert out.shape[1] == 100

    def test_custom_bar_height(self):
        img = _canvas(50, 100)
        out = draw_confidence_bar(img, 0.5, bar_height=30)
        assert out.shape[0] == 50 + 30

    def test_with_grade(self):
        img = _canvas(50, 100)
        out = draw_confidence_bar(img, 0.9, grade="A")
        assert out.ndim == 3

    def test_zero_confidence(self):
        img = _canvas(50, 100)
        out = draw_confidence_bar(img, 0.0)
        assert out.shape[0] > 50

    def test_one_confidence(self):
        img = _canvas(50, 100)
        out = draw_confidence_bar(img, 1.0)
        assert out.shape[0] > 50


# ─── tile_images ─────────────────────────────────────────────────────────────

class TestTileImagesExtra:
    def test_empty_returns_fallback(self):
        out = tile_images([])
        assert out.shape == (100, 100, 3)

    def test_single_image(self):
        img = _canvas(40, 60)
        out = tile_images([img], n_cols=3)
        # 1 row x 3 cols grid but only 1 image
        assert out.ndim == 3

    def test_two_images_one_col(self):
        imgs = [_canvas(30, 40), _canvas(30, 40)]
        out = tile_images(imgs, n_cols=1)
        cfg = VisConfig()
        assert out.shape[0] == 30 * 2 + cfg.tile_gap

    def test_labels(self):
        imgs = [_canvas(30, 30), _canvas(30, 30)]
        out = tile_images(imgs, labels=["a", "b"])
        assert out.ndim == 3

    def test_grayscale_converted(self):
        imgs = [_gray_canvas(30, 30)]
        out = tile_images(imgs)
        assert out.ndim == 3 and out.shape[2] == 3
