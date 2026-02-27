"""Tests for puzzle_reconstruction.utils.visualizer (numpy/pytest only, no cv2 imports in tests)"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.visualizer import (
    VisConfig,
    _ensure_bgr,
    draw_contour,
    draw_skew_angle,
    draw_confidence_bar,
    tile_images,
)


# ── VisConfig ─────────────────────────────────────────────────────────────────

def test_vis_config_defaults():
    cfg = VisConfig()
    assert cfg.line_thickness == 1
    assert cfg.font_scale == pytest.approx(0.4)
    assert cfg.tile_gap == 4


def test_vis_config_custom():
    cfg = VisConfig(line_thickness=3, font_scale=1.0, tile_gap=10)
    assert cfg.line_thickness == 3
    assert cfg.font_scale == pytest.approx(1.0)
    assert cfg.tile_gap == 10


def test_vis_config_colors_are_tuples():
    cfg = VisConfig()
    assert isinstance(cfg.word_box_color, tuple)
    assert len(cfg.word_box_color) == 3


# ── _ensure_bgr ───────────────────────────────────────────────────────────────

def test_ensure_bgr_grayscale_2d():
    gray = np.zeros((10, 10), dtype=np.uint8)
    bgr = _ensure_bgr(gray)
    assert bgr.ndim == 3
    assert bgr.shape[2] == 3


def test_ensure_bgr_already_bgr():
    bgr_in = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_out = _ensure_bgr(bgr_in)
    assert bgr_out.shape == (10, 10, 3)


def test_ensure_bgr_bgra():
    bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    bgr = _ensure_bgr(bgra)
    assert bgr.shape[2] == 3


def test_ensure_bgr_returns_copy():
    img = np.ones((5, 5, 3), dtype=np.uint8) * 128
    out = _ensure_bgr(img)
    out[0, 0, 0] = 0
    # Original should be unchanged
    assert img[0, 0, 0] == 128


def test_ensure_bgr_output_dtype():
    gray = np.zeros((8, 8), dtype=np.uint8)
    bgr = _ensure_bgr(gray)
    assert bgr.dtype == np.uint8


# ── draw_contour ──────────────────────────────────────────────────────────────

def test_draw_contour_output_shape():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    contour = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])
    out = draw_contour(img, contour)
    assert out.shape == img.shape


def test_draw_contour_output_dtype():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    contour = np.array([[10, 10], [40, 10], [40, 40], [10, 40]])
    out = draw_contour(img, contour)
    assert out.dtype == np.uint8


def test_draw_contour_does_not_modify_input():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    img_copy = img.copy()
    contour = np.array([[10, 10], [40, 10], [40, 40]])
    draw_contour(img, contour)
    np.testing.assert_array_equal(img, img_copy)


def test_draw_contour_grayscale_input():
    img = np.ones((50, 50), dtype=np.uint8) * 200
    contour = np.array([[10, 10], [40, 10], [40, 40]])
    out = draw_contour(img, contour)
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_draw_contour_3d_points():
    img = np.ones((60, 60, 3), dtype=np.uint8) * 128
    contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])
    out = draw_contour(img, contour)
    assert out.shape == (60, 60, 3)


def test_draw_contour_with_fill():
    img = np.ones((60, 60, 3), dtype=np.uint8) * 255
    contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
    out = draw_contour(img, contour, fill_alpha=0.3)
    assert out.shape == (60, 60, 3)


# ── draw_skew_angle ───────────────────────────────────────────────────────────

def test_draw_skew_angle_output_shape():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    out = draw_skew_angle(img, angle_deg=15.0)
    assert out.shape == (100, 100, 3)


def test_draw_skew_angle_output_dtype():
    img = np.ones((80, 80, 3), dtype=np.uint8) * 200
    out = draw_skew_angle(img, angle_deg=0.0)
    assert out.dtype == np.uint8


def test_draw_skew_angle_zero_angle():
    img = np.ones((80, 80, 3), dtype=np.uint8) * 200
    out = draw_skew_angle(img, angle_deg=0.0)
    assert out.shape == img.shape


def test_draw_skew_angle_negative_angle():
    img = np.ones((80, 80, 3), dtype=np.uint8) * 200
    out = draw_skew_angle(img, angle_deg=-5.0)
    assert out.shape == img.shape


def test_draw_skew_angle_grayscale_input():
    img = np.ones((80, 80), dtype=np.uint8) * 200
    out = draw_skew_angle(img, angle_deg=10.0)
    assert out.ndim == 3


def test_draw_skew_angle_modifies_pixels():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    out = draw_skew_angle(img, angle_deg=30.0)
    # The line should change some pixels
    assert not np.array_equal(img, out)


# ── draw_confidence_bar ───────────────────────────────────────────────────────

def test_draw_confidence_bar_adds_height():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 200
    out = draw_confidence_bar(img, confidence=0.7)
    assert out.shape[0] > 50
    assert out.shape[1] == 100


def test_draw_confidence_bar_output_dtype():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 200
    out = draw_confidence_bar(img, confidence=0.5)
    assert out.dtype == np.uint8


def test_draw_confidence_bar_zero_confidence():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 200
    out = draw_confidence_bar(img, confidence=0.0)
    assert out.shape[0] > 50


def test_draw_confidence_bar_full_confidence():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 200
    out = draw_confidence_bar(img, confidence=1.0)
    assert out.shape[0] > 50


def test_draw_confidence_bar_with_grade():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 200
    out = draw_confidence_bar(img, confidence=0.8, grade="A")
    assert out.shape[0] > 50


def test_draw_confidence_bar_custom_height():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 200
    out = draw_confidence_bar(img, confidence=0.5, bar_height=30)
    assert out.shape[0] == 50 + 30


# ── tile_images ───────────────────────────────────────────────────────────────

def test_tile_images_empty_returns_placeholder():
    out = tile_images([])
    assert out.ndim == 3
    assert out.shape[2] == 3


def test_tile_images_single_image():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    out = tile_images([img], n_cols=1)
    assert out.ndim == 3


def test_tile_images_output_shape_3cols():
    imgs = [np.zeros((30, 30, 3), dtype=np.uint8) for _ in range(6)]
    out = tile_images(imgs, n_cols=3)
    # 2 rows of 3 cols, each 30x30, with gap=4
    expected_h = 2 * 30 + 1 * 4
    expected_w = 3 * 30 + 2 * 4
    assert out.shape == (expected_h, expected_w, 3)


def test_tile_images_grayscale_converted():
    gray = np.zeros((30, 30), dtype=np.uint8)
    out = tile_images([gray], n_cols=1)
    assert out.shape[2] == 3


def test_tile_images_with_labels():
    imgs = [np.zeros((30, 30, 3), dtype=np.uint8) for _ in range(3)]
    labels = ["A", "B", "C"]
    out = tile_images(imgs, n_cols=3, labels=labels)
    assert out.ndim == 3


def test_tile_images_partial_last_row():
    imgs = [np.zeros((30, 30, 3), dtype=np.uint8) for _ in range(5)]
    out = tile_images(imgs, n_cols=3)
    # 5 images -> 2 rows (3 + 2)
    expected_h = 2 * 30 + 1 * 4
    assert out.shape[0] == expected_h


def test_tile_images_output_dtype():
    imgs = [np.zeros((20, 20, 3), dtype=np.uint8)]
    out = tile_images(imgs)
    assert out.dtype == np.uint8


def test_tile_images_n_cols_1():
    imgs = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    out = tile_images(imgs, n_cols=1)
    # 3 rows, 1 col
    expected_h = 3 * 20 + 2 * 4
    assert out.shape[0] == expected_h
    assert out.shape[1] == 20
