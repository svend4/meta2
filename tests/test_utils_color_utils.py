"""Tests for puzzle_reconstruction.utils.color_utils (no cv2 in tests)."""
import numpy as np
import pytest

from puzzle_reconstruction.utils.color_utils import (
    to_gray,
    to_lab,
    to_hsv,
    from_lab,
    compute_histogram,
    compare_histograms,
    dominant_colors,
    color_distance,
    strip_histogram,
)

np.random.seed(7)


# ── to_gray ────────────────────────────────────────────────────────────────────

def test_to_gray_bgr_returns_2d():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    gray = to_gray(img)
    assert gray.ndim == 2
    assert gray.shape == (50, 50)


def test_to_gray_already_gray():
    img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    gray = to_gray(img)
    assert gray.ndim == 2
    assert gray.shape == (30, 30)


def test_to_gray_dtype():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert to_gray(img).dtype == np.uint8


# ── to_lab ─────────────────────────────────────────────────────────────────────

def test_to_lab_shape():
    img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    lab = to_lab(img)
    assert lab.shape == (20, 20, 3)


def test_to_lab_dtype_float32():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    lab = to_lab(img)
    assert lab.dtype == np.float32


def test_to_lab_from_gray():
    img = np.ones((10, 10), dtype=np.uint8) * 128
    lab = to_lab(img)
    assert lab.shape == (10, 10, 3)


# ── to_hsv ─────────────────────────────────────────────────────────────────────

def test_to_hsv_shape():
    img = np.random.randint(0, 255, (15, 15, 3), dtype=np.uint8)
    hsv = to_hsv(img)
    assert hsv.shape == (15, 15, 3)


def test_to_hsv_dtype():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv = to_hsv(img)
    assert hsv.dtype == np.uint8


# ── from_lab ───────────────────────────────────────────────────────────────────

def test_from_lab_roundtrip():
    img = np.random.randint(30, 200, (20, 20, 3), dtype=np.uint8)
    lab = to_lab(img)
    bgr = from_lab(lab)
    assert bgr.shape == img.shape
    assert bgr.dtype == np.uint8


def test_from_lab_shape():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 100
    lab = to_lab(img)
    bgr = from_lab(lab)
    assert bgr.shape == (10, 10, 3)


# ── compute_histogram ──────────────────────────────────────────────────────────

def test_compute_histogram_shape():
    img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    h = compute_histogram(img, bins=64)
    assert h.shape == (64,)


def test_compute_histogram_normalized_sums_to_one():
    img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    h = compute_histogram(img, normalize=True)
    assert abs(h.sum() - 1.0) < 1e-5


def test_compute_histogram_not_normalized():
    img = np.ones((10, 10), dtype=np.uint8) * 100
    h = compute_histogram(img, bins=256, normalize=False)
    assert h.sum() == 100  # 10*10 = 100 pixels all at value 100


def test_compute_histogram_bgr_channel():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # Red channel
    h = compute_histogram(img, channel=2, normalize=False)
    assert h[200] == 100


def test_compute_histogram_dtype():
    img = np.zeros((10, 10), dtype=np.uint8)
    h = compute_histogram(img)
    assert h.dtype == np.float32


# ── compare_histograms ─────────────────────────────────────────────────────────

def test_compare_histograms_identical_correlation():
    img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    h = compute_histogram(img)
    result = compare_histograms(h, h, method="correlation")
    assert abs(result - 1.0) < 1e-5


def test_compare_histograms_identical_bhattacharyya():
    img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    h = compute_histogram(img)
    result = compare_histograms(h, h, method="bhattacharyya")
    assert abs(result) < 1e-5  # identical → 0


def test_compare_histograms_invalid_method():
    h = np.ones(256, dtype=np.float32) / 256
    with pytest.raises(ValueError):
        compare_histograms(h, h, method="unknown")


def test_compare_histograms_chi_non_negative():
    img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    h = compute_histogram(img)
    h2 = np.roll(h, 10)
    h2 = h2 / h2.sum()
    result = compare_histograms(h, h2.astype(np.float32), method="chi")
    assert result >= 0.0


# ── dominant_colors ────────────────────────────────────────────────────────────

def test_dominant_colors_shape():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    colors = dominant_colors(img, k=3)
    assert colors.shape == (3, 3)


def test_dominant_colors_dtype():
    img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    colors = dominant_colors(img, k=2)
    assert colors.dtype == np.uint8


def test_dominant_colors_k1():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 100
    colors = dominant_colors(img, k=1)
    assert colors.shape == (1, 3)


def test_dominant_colors_gray():
    img = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    colors = dominant_colors(img, k=2)
    assert colors.shape == (2, 3)


# ── color_distance ─────────────────────────────────────────────────────────────

def test_color_distance_identical():
    c = np.array([100, 100, 100], dtype=np.uint8)
    assert color_distance(c, c, space="lab") == pytest.approx(0.0, abs=1e-4)


def test_color_distance_rgb():
    c1 = np.array([0, 0, 0], dtype=np.uint8)
    c2 = np.array([255, 0, 0], dtype=np.uint8)
    d = color_distance(c1, c2, space="rgb")
    assert d == pytest.approx(255.0, rel=1e-3)


def test_color_distance_nonnegative():
    c1 = np.array([100, 50, 200], dtype=np.uint8)
    c2 = np.array([200, 150, 100], dtype=np.uint8)
    assert color_distance(c1, c2, space="lab") >= 0.0


def test_color_distance_invalid_space():
    c = np.array([100, 100, 100], dtype=np.uint8)
    with pytest.raises(ValueError):
        color_distance(c, c, space="xyz")


# ── strip_histogram ────────────────────────────────────────────────────────────

def test_strip_histogram_top():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    h = strip_histogram(img, side=0, border_px=10)
    assert h.shape == (256,)
    assert abs(h.sum() - 1.0) < 1e-5


def test_strip_histogram_all_sides():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    for side in range(4):
        h = strip_histogram(img, side=side, border_px=5)
        assert h.shape[0] > 0


def test_strip_histogram_invalid_side():
    img = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        strip_histogram(img, side=5)
