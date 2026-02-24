"""Extra tests for puzzle_reconstruction/utils/render_utils.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from puzzle_reconstruction.utils.render_utils import (
    CanvasConfig,
    MosaicConfig,
    rotation_matrix_2d,
    bounding_box_of_rotated,
    compute_canvas_size,
    make_blank_canvas,
    resize_keep_aspect,
    pad_to_square,
    make_thumbnail,
    paste_with_mask,
    compute_grid_layout,
    make_mosaic,
    horizontal_concat,
)


# ─── CanvasConfig ─────────────────────────────────────────────────────────────

class TestCanvasConfigExtra:
    def test_default_bg_color(self):
        assert CanvasConfig().bg_color == (240, 240, 240)

    def test_default_margin(self):
        assert CanvasConfig().margin == 20

    def test_custom_margin(self):
        cfg = CanvasConfig(margin=10)
        assert cfg.margin == 10


# ─── MosaicConfig ─────────────────────────────────────────────────────────────

class TestMosaicConfigExtra:
    def test_default_thumb_size(self):
        assert MosaicConfig().thumb_size == 64

    def test_default_max_cols(self):
        assert MosaicConfig().max_cols == 8

    def test_default_gap(self):
        assert MosaicConfig().gap == 8

    def test_custom_config(self):
        cfg = MosaicConfig(thumb_size=32, max_cols=4, gap=4)
        assert cfg.thumb_size == 32 and cfg.max_cols == 4 and cfg.gap == 4


# ─── rotation_matrix_2d ───────────────────────────────────────────────────────

class TestRotationMatrix2dExtra:
    def test_shape(self):
        R = rotation_matrix_2d(0.0)
        assert R.shape == (2, 2)

    def test_identity_at_zero(self):
        R = rotation_matrix_2d(0.0)
        assert np.allclose(R, np.eye(2))

    def test_90_degrees(self):
        R = rotation_matrix_2d(math.pi / 2)
        expected = np.array([[0, -1], [1, 0]], dtype=np.float64)
        assert np.allclose(R, expected, atol=1e-10)

    def test_orthogonal(self):
        R = rotation_matrix_2d(1.23)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-10)


# ─── bounding_box_of_rotated ──────────────────────────────────────────────────

class TestBoundingBoxOfRotatedExtra:
    def test_zero_angle_unchanged(self):
        new_w, new_h = bounding_box_of_rotated(100, 50, 0.0)
        assert new_w == 100 and new_h == 50

    def test_90_degrees_swaps(self):
        new_w, new_h = bounding_box_of_rotated(100, 50, math.pi / 2)
        # ceil rounding may add 1 pixel
        assert abs(new_w - 50) <= 1 and abs(new_h - 100) <= 1

    def test_returns_positive(self):
        new_w, new_h = bounding_box_of_rotated(80, 60, 0.5)
        assert new_w > 0 and new_h > 0

    def test_45_degrees_square_unchanged(self):
        # A square rotated 45° has a larger bounding box
        new_w, new_h = bounding_box_of_rotated(100, 100, math.pi / 4)
        # should be > 100
        assert new_w > 100 and new_h > 100


# ─── compute_canvas_size ──────────────────────────────────────────────────────

class TestComputeCanvasSizeExtra:
    def test_empty_placements(self):
        w, h = compute_canvas_size({}, {})
        assert w > 0 and h > 0

    def test_single_fragment_no_rotation(self):
        pos = np.array([0.0, 0.0])
        w, h = compute_canvas_size(
            {0: (pos, 0.0)},
            {0: (100, 80)},
            margin=10,
        )
        assert w >= 100 + 20 and h >= 80 + 20

    def test_margin_adds_to_size(self):
        pos = np.array([0.0, 0.0])
        w1, h1 = compute_canvas_size({0: (pos, 0.0)}, {0: (50, 50)}, margin=5)
        w2, h2 = compute_canvas_size({0: (pos, 0.0)}, {0: (50, 50)}, margin=20)
        assert w2 > w1 and h2 > h1

    def test_missing_frag_id_skipped(self):
        pos = np.array([0.0, 0.0])
        # fid 99 not in frag_sizes → should not crash
        w, h = compute_canvas_size({99: (pos, 0.0)}, {0: (50, 50)})
        assert w > 0 and h > 0


# ─── make_blank_canvas ────────────────────────────────────────────────────────

class TestMakeBlankCanvasExtra:
    def test_shape(self):
        c = make_blank_canvas(80, 60)
        assert c.shape == (60, 80, 3)

    def test_color_filled(self):
        c = make_blank_canvas(10, 10, color=(100, 150, 200))
        assert np.all(c[:, :, 0] == 100)
        assert np.all(c[:, :, 1] == 150)
        assert np.all(c[:, :, 2] == 200)

    def test_dtype_uint8(self):
        c = make_blank_canvas(20, 20)
        assert c.dtype == np.uint8


# ─── resize_keep_aspect ───────────────────────────────────────────────────────

class TestResizeKeepAspectExtra:
    def test_longer_side_equals_target(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 50)
        assert max(result.shape[:2]) == 50

    def test_aspect_ratio_preserved(self):
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 200)
        h, w = result.shape[:2]
        assert abs(h / w - 2.0) < 0.1

    def test_zero_dimension_returns_copy(self):
        img = np.zeros((0, 10, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 64)
        assert result.shape[0] == 0


# ─── pad_to_square ────────────────────────────────────────────────────────────

class TestPadToSquareExtra:
    def test_output_shape(self):
        img = np.zeros((30, 20, 3), dtype=np.uint8)
        result = pad_to_square(img, 64)
        assert result.shape == (64, 64, 3)

    def test_fill_value(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = pad_to_square(img, 20, fill=128)
        # Bottom-right corner should be fill value
        assert result[15, 15, 0] == 128

    def test_grayscale_pad(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        result = pad_to_square(img, 20)
        assert result.shape == (20, 20)


# ─── make_thumbnail ───────────────────────────────────────────────────────────

class TestMakeThumbnailExtra:
    def test_output_is_square(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        t = make_thumbnail(img, 64)
        assert t.shape == (64, 64, 3)

    def test_small_thumb_size(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        t = make_thumbnail(img, 32)
        assert t.shape == (32, 32, 3)


# ─── paste_with_mask ──────────────────────────────────────────────────────────

class TestPasteWithMaskExtra:
    def _make_canvas(self, h=100, w=100):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _make_white_frag(self, h=20, w=20):
        return np.full((h, w, 3), 255, dtype=np.uint8)

    def _make_full_mask(self, h=20, w=20):
        return np.full((h, w), 255, dtype=np.uint8)

    def test_full_mask_overwrites(self):
        canvas = self._make_canvas()
        frag = self._make_white_frag()
        mask = self._make_full_mask()
        result = paste_with_mask(canvas, frag, mask, 0, 0)
        assert np.all(result[0:20, 0:20] == 255)

    def test_zero_mask_leaves_canvas(self):
        canvas = self._make_canvas()
        frag = self._make_white_frag()
        mask = np.zeros((20, 20), dtype=np.uint8)
        result = paste_with_mask(canvas, frag, mask, 0, 0)
        assert np.all(result[0:20, 0:20] == 0)

    def test_out_of_bounds_returns_canvas(self):
        canvas = self._make_canvas(50, 50)
        frag = self._make_white_frag()
        mask = self._make_full_mask()
        result = paste_with_mask(canvas, frag, mask, 200, 200)
        # No crash, canvas unchanged
        assert result.shape == (50, 50, 3)

    def test_returns_canvas(self):
        canvas = self._make_canvas()
        frag = self._make_white_frag()
        mask = self._make_full_mask()
        result = paste_with_mask(canvas, frag, mask, 5, 5)
        assert result is canvas


# ─── compute_grid_layout ──────────────────────────────────────────────────────

class TestComputeGridLayoutExtra:
    def test_zero_items(self):
        assert compute_grid_layout(0) == (0, 0)

    def test_single_item(self):
        rows, cols = compute_grid_layout(1)
        assert rows == 1 and cols == 1

    def test_respects_max_cols(self):
        rows, cols = compute_grid_layout(10, max_cols=4)
        assert cols <= 4

    def test_all_items_covered(self):
        n = 17
        rows, cols = compute_grid_layout(n, max_cols=5)
        assert rows * cols >= n

    def test_fewer_than_max_cols(self):
        rows, cols = compute_grid_layout(3, max_cols=8)
        assert cols == 3


# ─── make_mosaic ──────────────────────────────────────────────────────────────

class TestMakeMosaicExtra:
    def test_empty_returns_image(self):
        result = make_mosaic([])
        assert isinstance(result, np.ndarray) and result.ndim == 3

    def test_single_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = make_mosaic([img])
        assert result.ndim == 3

    def test_output_is_bgr(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = make_mosaic([img])
        assert result.shape[2] == 3

    def test_grayscale_input(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        result = make_mosaic([img])
        assert result.ndim == 3 and result.shape[2] == 3

    def test_multiple_images(self):
        imgs = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(5)]
        result = make_mosaic(imgs, MosaicConfig(thumb_size=32, max_cols=3))
        assert result.ndim == 3


# ─── horizontal_concat ────────────────────────────────────────────────────────

class TestHorizontalConcatExtra:
    def test_empty_returns_image(self):
        result = horizontal_concat([])
        assert result.ndim == 3

    def test_single_image(self):
        img = np.zeros((50, 80, 3), dtype=np.uint8)
        result = horizontal_concat([img], gap=0)
        assert result.shape == (50, 80, 3)

    def test_two_images_width(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = horizontal_concat([img, img], gap=0)
        assert result.shape[1] == 100

    def test_gap_adds_width(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        r_no_gap = horizontal_concat([img, img], gap=0)
        r_with_gap = horizontal_concat([img, img], gap=10)
        assert r_with_gap.shape[1] > r_no_gap.shape[1]

    def test_grayscale_converted(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        result = horizontal_concat([img], gap=0)
        assert result.shape[2] == 3
