"""Extra tests for puzzle_reconstruction/utils/color_utils.py."""
from __future__ import annotations

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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _color(b, g, r):
    return np.array([b, g, r], dtype=np.uint8)


# ─── to_gray (extra) ─────────────────────────────────────────────────────────

class TestToGrayExtra:
    def test_ndim_2(self):
        assert to_gray(_bgr()).ndim == 2

    def test_dtype_uint8(self):
        assert to_gray(_bgr()).dtype == np.uint8

    def test_shape_preserved(self):
        h, w = 20, 30
        assert to_gray(_bgr(h, w)).shape == (h, w)

    def test_gray_input_unchanged(self):
        img = _gray(val=50)
        r = to_gray(img)
        np.testing.assert_array_equal(r, img)

    def test_white_bgr_high(self):
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        assert to_gray(img).min() >= 250

    def test_black_bgr_zero(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert to_gray(img).max() == 0


# ─── to_lab (extra) ──────────────────────────────────────────────────────────

class TestToLabExtra:
    def test_dtype_float32(self):
        assert to_lab(_bgr()).dtype == np.float32

    def test_ndim_3(self):
        assert to_lab(_bgr()).ndim == 3

    def test_shape_hw3(self):
        h, w = 16, 24
        assert to_lab(_bgr(h, w)).shape == (h, w, 3)

    def test_L_in_range(self):
        lab = to_lab(_bgr())
        L = lab[:, :, 0]
        assert L.min() >= 0.0
        assert L.max() <= 100.0 + 1e-3

    def test_white_L_near_100(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        lab = to_lab(img)
        assert lab[:, :, 0].mean() > 90.0

    def test_gray_input_ok(self):
        lab = to_lab(_gray())
        assert lab.shape == (32, 32, 3)

    def test_black_L_near_zero(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        lab = to_lab(img)
        assert lab[:, :, 0].mean() < 10.0


# ─── to_hsv (extra) ──────────────────────────────────────────────────────────

class TestToHsvExtra:
    def test_dtype_uint8(self):
        assert to_hsv(_bgr()).dtype == np.uint8

    def test_ndim_3(self):
        assert to_hsv(_bgr()).ndim == 3

    def test_shape_hw3(self):
        h, w = 16, 24
        assert to_hsv(_bgr(h, w)).shape == (h, w, 3)

    def test_H_in_range(self):
        hsv = to_hsv(_bgr())
        assert hsv[:, :, 0].min() >= 0
        assert hsv[:, :, 0].max() <= 180

    def test_S_in_range(self):
        hsv = to_hsv(_bgr())
        assert hsv[:, :, 1].min() >= 0
        assert hsv[:, :, 1].max() <= 255

    def test_white_S_zero(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        hsv = to_hsv(img)
        assert hsv[:, :, 1].max() == 0

    def test_gray_input_ok(self):
        hsv = to_hsv(_gray())
        assert hsv.shape == (32, 32, 3)


# ─── from_lab (extra) ────────────────────────────────────────────────────────

class TestFromLabExtra:
    def test_dtype_uint8(self):
        lab = to_lab(_bgr())
        assert from_lab(lab).dtype == np.uint8

    def test_ndim_3(self):
        lab = to_lab(_bgr())
        assert from_lab(lab).ndim == 3

    def test_shape_hw3(self):
        h, w = 16, 20
        lab = to_lab(_bgr(h, w))
        assert from_lab(lab).shape == (h, w, 3)

    def test_roundtrip_close(self):
        img = _bgr()
        recovered = from_lab(to_lab(img))
        diff = np.abs(img.astype(np.float32) - recovered.astype(np.float32))
        assert diff.mean() < 5.0

    def test_white_roundtrip(self):
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        assert from_lab(to_lab(img)).min() > 240


# ─── compute_histogram (extra) ───────────────────────────────────────────────

class TestComputeHistogramExtra:
    def test_dtype_float32(self):
        assert compute_histogram(_gray()).dtype == np.float32

    def test_length_equals_bins(self):
        assert len(compute_histogram(_gray(), bins=32)) == 32

    def test_normalized_sum_one(self):
        h = compute_histogram(_gray(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_unnormalized_sum_n_pixels(self):
        img = _gray(10, 10)
        h = compute_histogram(img, normalize=False)
        assert h.sum() == pytest.approx(100.0, abs=1e-3)

    def test_nonneg(self):
        assert compute_histogram(_gray()).min() >= 0.0

    def test_uniform_single_peak(self):
        img = _gray(val=100)
        h = compute_histogram(img, bins=256, normalize=False)
        assert h[100] > 0
        assert h[:100].sum() == 0
        assert h[101:].sum() == 0

    def test_bgr_channel_0(self):
        r = compute_histogram(_bgr(), channel=0)
        assert len(r) == 256


# ─── compare_histograms (extra) ──────────────────────────────────────────────

class TestCompareHistogramsExtra:
    def test_correlation_identical_is_one(self):
        h = compute_histogram(_gray())
        assert compare_histograms(h, h, method="correlation") == pytest.approx(1.0, abs=1e-5)

    def test_chi_identical_is_zero(self):
        h = compute_histogram(_gray())
        assert compare_histograms(h, h, method="chi") == pytest.approx(0.0, abs=1e-5)

    def test_bhattacharyya_identical_is_zero(self):
        h = compute_histogram(_gray())
        assert compare_histograms(h, h, method="bhattacharyya") == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        h = compute_histogram(_gray())
        assert isinstance(compare_histograms(h, h), float)

    def test_unknown_method_raises(self):
        h = compute_histogram(_gray())
        with pytest.raises(ValueError):
            compare_histograms(h, h, method="xyz_unknown")

    def test_different_images_correlation_less_one(self):
        h1 = compute_histogram(_gray(val=50))
        h2 = compute_histogram(_gray(val=200))
        assert compare_histograms(h1, h2, method="correlation") < 1.0


# ─── dominant_colors (extra) ─────────────────────────────────────────────────

class TestDominantColorsExtra:
    def test_returns_ndarray(self):
        assert isinstance(dominant_colors(_bgr(), k=3), np.ndarray)

    def test_shape_k_3(self):
        r = dominant_colors(_bgr(), k=3)
        assert r.shape == (3, 3)

    def test_dtype_uint8(self):
        assert dominant_colors(_bgr(), k=3).dtype == np.uint8

    def test_values_in_0_255(self):
        r = dominant_colors(_bgr(), k=3)
        assert r.min() >= 0
        assert r.max() <= 255

    def test_k1_returns_1_row(self):
        r = dominant_colors(_bgr(), k=1)
        assert r.shape == (1, 3)

    def test_gray_input(self):
        r = dominant_colors(_gray(), k=2)
        assert r.shape == (2, 3)


# ─── color_distance (extra) ──────────────────────────────────────────────────

class TestColorDistanceExtra:
    def test_returns_float(self):
        c = _color(100, 150, 200)
        assert isinstance(color_distance(c, c), float)

    def test_same_color_zero(self):
        c = _color(100, 150, 200)
        assert color_distance(c, c) == pytest.approx(0.0, abs=1e-3)

    def test_nonneg(self):
        c1 = _color(0, 0, 0)
        c2 = _color(200, 100, 50)
        assert color_distance(c1, c2) >= 0.0

    def test_symmetric(self):
        c1 = _color(100, 50, 200)
        c2 = _color(30, 180, 90)
        assert color_distance(c1, c2) == pytest.approx(
            color_distance(c2, c1), abs=1e-3
        )

    def test_rgb_space_known(self):
        c1 = _color(0, 0, 0)
        c2 = _color(255, 0, 0)
        assert color_distance(c1, c2, space="rgb") == pytest.approx(255.0, abs=1e-3)

    def test_lab_vs_rgb_differ(self):
        c1 = _color(0, 0, 0)
        c2 = _color(255, 0, 0)
        d_lab = color_distance(c1, c2, space="lab")
        d_rgb = color_distance(c1, c2, space="rgb")
        assert d_lab != d_rgb

    def test_unknown_space_raises(self):
        c = _color(100, 100, 100)
        with pytest.raises(ValueError):
            color_distance(c, c, space="xyz_unknown")


# ─── strip_histogram (extra) ─────────────────────────────────────────────────

class TestStripHistogramExtra:
    def test_dtype_float32(self):
        assert strip_histogram(_gray()).dtype == np.float32

    def test_length_equals_bins(self):
        assert len(strip_histogram(_gray(), bins=64)) == 64

    def test_sum_to_one(self):
        h = strip_histogram(_gray(), bins=32)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_all_sides_accepted(self):
        img = _bgr()
        for side in range(4):
            h = strip_histogram(img, side=side, border_px=4)
            assert h.dtype == np.float32

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            strip_histogram(_gray(), side=5)

    def test_uniform_image_single_peak(self):
        img = _gray(val=100)
        h = strip_histogram(img, side=0, bins=256, border_px=4)
        assert h[100] == pytest.approx(1.0, abs=1e-5)

    def test_bgr_channel0_accepted(self):
        h = strip_histogram(_bgr(), side=0, channel=0, bins=32)
        assert h.dtype == np.float32
        assert len(h) == 32
