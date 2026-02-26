"""Tests for puzzle_reconstruction.utils.render_utils"""
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

np.random.seed(42)


# ─── CanvasConfig / MosaicConfig ─────────────────────────────────────────────

def test_canvas_config_defaults():
    cfg = CanvasConfig()
    assert cfg.bg_color == (240, 240, 240)
    assert cfg.margin == 20


def test_mosaic_config_defaults():
    cfg = MosaicConfig()
    assert cfg.thumb_size == 64
    assert cfg.max_cols == 8
    assert cfg.gap == 8


# ─── rotation_matrix_2d ───────────────────────────────────────────────────────

def test_rotation_matrix_2d_shape():
    R = rotation_matrix_2d(0.0)
    assert R.shape == (2, 2)


def test_rotation_matrix_2d_identity():
    R = rotation_matrix_2d(0.0)
    np.testing.assert_allclose(R, np.eye(2), atol=1e-9)


def test_rotation_matrix_2d_90deg():
    R = rotation_matrix_2d(math.pi / 2)
    # Should rotate (1, 0) → (0, 1)
    v = R @ np.array([1.0, 0.0])
    assert abs(v[0]) < 1e-9
    assert abs(v[1] - 1.0) < 1e-9


# ─── bounding_box_of_rotated ──────────────────────────────────────────────────

def test_bounding_box_zero_angle():
    nw, nh = bounding_box_of_rotated(100, 50, 0.0)
    assert nw == 100
    assert nh == 50


def test_bounding_box_90deg():
    nw, nh = bounding_box_of_rotated(100, 50, math.pi / 2)
    # After 90deg rotation, width and height should swap
    assert abs(nw - 50) <= 1
    assert abs(nh - 100) <= 1


def test_bounding_box_nonneg():
    nw, nh = bounding_box_of_rotated(80, 60, math.pi / 4)
    assert nw > 0
    assert nh > 0


# ─── compute_canvas_size ──────────────────────────────────────────────────────

def test_compute_canvas_size_empty():
    w, h = compute_canvas_size({}, {}, margin=20)
    assert w > 0
    assert h > 0


def test_compute_canvas_size_single_frag():
    placements = {0: (np.array([0, 0]), 0.0)}
    sizes = {0: (100, 50)}
    w, h = compute_canvas_size(placements, sizes, margin=10)
    assert w >= 100 + 20
    assert h >= 50 + 20


def test_compute_canvas_size_returns_ints():
    placements = {0: (np.array([10, 20]), 0.0)}
    sizes = {0: (80, 60)}
    w, h = compute_canvas_size(placements, sizes, margin=5)
    assert isinstance(w, int)
    assert isinstance(h, int)


# ─── make_blank_canvas ───────────────────────────────────────────────────────

def test_make_blank_canvas_shape():
    canvas = make_blank_canvas(200, 100)
    assert canvas.shape == (100, 200, 3)
    assert canvas.dtype == np.uint8


def test_make_blank_canvas_color():
    canvas = make_blank_canvas(10, 10, color=(255, 0, 0))
    assert canvas[0, 0, 0] == 255
    assert canvas[0, 0, 1] == 0


# ─── resize_keep_aspect ───────────────────────────────────────────────────────

def test_resize_keep_aspect_downsample():
    img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    result = resize_keep_aspect(img, 50)
    assert max(result.shape[:2]) == 50


def test_resize_keep_aspect_empty():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    result = resize_keep_aspect(img, 64)
    assert result.shape == (0, 0, 3)


def test_resize_keep_aspect_preserves_ratio():
    img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    result = resize_keep_aspect(img, 100)
    h, w = result.shape[:2]
    assert max(h, w) == 100
    # Width was bigger, so width=100, height~50
    assert w == 100


# ─── pad_to_square ────────────────────────────────────────────────────────────

def test_pad_to_square_shape():
    img = np.random.randint(0, 255, (40, 60, 3), dtype=np.uint8)
    result = pad_to_square(img, 64)
    assert result.shape == (64, 64, 3)


def test_pad_to_square_fill_value():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = pad_to_square(img, 20, fill=128)
    # Bottom-right region should be filled with 128
    assert result[15, 15, 0] == 128


def test_pad_to_square_gray():
    img = np.zeros((10, 10), dtype=np.uint8)
    result = pad_to_square(img, 20, fill=100)
    assert result.shape == (20, 20)
    assert result[15, 15] == 100


# ─── make_thumbnail ───────────────────────────────────────────────────────────

def test_make_thumbnail_shape():
    img = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
    thumb = make_thumbnail(img, thumb_size=64)
    assert thumb.shape == (64, 64, 3)


def test_make_thumbnail_dtype():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    thumb = make_thumbnail(img, thumb_size=32)
    assert thumb.dtype == np.uint8


# ─── paste_with_mask ──────────────────────────────────────────────────────────

def test_paste_with_mask_basic():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    fragment = np.full((30, 30, 3), 200, dtype=np.uint8)
    mask = np.full((30, 30), 255, dtype=np.uint8)
    result = paste_with_mask(canvas, fragment, mask, x=10, y=10)
    assert result[20, 20, 0] == 200


def test_paste_with_mask_zero_mask():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    fragment = np.full((30, 30, 3), 200, dtype=np.uint8)
    mask = np.zeros((30, 30), dtype=np.uint8)
    result = paste_with_mask(canvas, fragment, mask, x=10, y=10)
    assert result[20, 20, 0] == 0  # canvas unchanged


def test_paste_with_mask_out_of_bounds():
    canvas = np.zeros((50, 50, 3), dtype=np.uint8)
    fragment = np.full((30, 30, 3), 100, dtype=np.uint8)
    mask = np.full((30, 30), 255, dtype=np.uint8)
    # Paste at out-of-bounds position
    result = paste_with_mask(canvas, fragment, mask, x=200, y=200)
    assert result.shape == (50, 50, 3)


# ─── compute_grid_layout ──────────────────────────────────────────────────────

def test_compute_grid_layout_zero():
    r, c = compute_grid_layout(0)
    assert r == 0
    assert c == 0


def test_compute_grid_layout_basic():
    r, c = compute_grid_layout(10, max_cols=4)
    assert c == 4
    assert r == 3  # ceil(10/4)


def test_compute_grid_layout_single():
    r, c = compute_grid_layout(1)
    assert r == 1
    assert c == 1


def test_compute_grid_layout_exact():
    r, c = compute_grid_layout(8, max_cols=4)
    assert r == 2
    assert c == 4


# ─── make_mosaic ──────────────────────────────────────────────────────────────

def test_make_mosaic_empty():
    mosaic = make_mosaic([])
    assert isinstance(mosaic, np.ndarray)
    assert mosaic.ndim == 3


def test_make_mosaic_single():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    mosaic = make_mosaic([img])
    assert isinstance(mosaic, np.ndarray)


def test_make_mosaic_multiple():
    imgs = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(4)]
    cfg = MosaicConfig(thumb_size=32, max_cols=2, gap=4)
    mosaic = make_mosaic(imgs, cfg)
    assert mosaic.shape[2] == 3


# ─── horizontal_concat ────────────────────────────────────────────────────────

def test_horizontal_concat_empty():
    result = horizontal_concat([])
    assert result.ndim == 3


def test_horizontal_concat_basic():
    imgs = [np.zeros((50, 30, 3), dtype=np.uint8),
            np.zeros((50, 40, 3), dtype=np.uint8)]
    result = horizontal_concat(imgs, gap=0)
    assert result.shape[0] == 50
    assert result.shape[1] == 70


def test_horizontal_concat_different_heights():
    imgs = [np.zeros((50, 30, 3), dtype=np.uint8),
            np.zeros((100, 30, 3), dtype=np.uint8)]
    result = horizontal_concat(imgs, gap=0)
    assert result.shape[0] == 100
