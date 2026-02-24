"""Extra tests for puzzle_reconstruction/utils/color_utils.py."""
from __future__ import annotations

import pytest
import numpy as np
import cv2

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=32, w=32) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(h=32, w=32) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w), dtype=np.uint8)


# ─── to_gray ──────────────────────────────────────────────────────────────────

class TestToGrayExtra:
    def test_returns_ndarray(self):
        assert isinstance(to_gray(_bgr()), np.ndarray)

    def test_bgr_gives_2d(self):
        out = to_gray(_bgr())
        assert out.ndim == 2

    def test_gray_returns_gray(self):
        g = _gray()
        out = to_gray(g)
        assert out.ndim == 2

    def test_dtype_uint8(self):
        assert to_gray(_bgr()).dtype == np.uint8

    def test_shape(self):
        out = to_gray(_bgr(16, 24))
        assert out.shape == (16, 24)

    def test_grayscale_copy(self):
        g = _gray()
        out = to_gray(g)
        np.testing.assert_array_equal(out, g)


# ─── to_lab ───────────────────────────────────────────────────────────────────

class TestToLabExtra:
    def test_returns_ndarray(self):
        assert isinstance(to_lab(_bgr()), np.ndarray)

    def test_shape_3_channels(self):
        out = to_lab(_bgr())
        assert out.ndim == 3 and out.shape[2] == 3

    def test_dtype_float32(self):
        assert to_lab(_bgr()).dtype == np.float32

    def test_spatial_shape_preserved(self):
        out = to_lab(_bgr(16, 24))
        assert out.shape[:2] == (16, 24)

    def test_grayscale_input_ok(self):
        out = to_lab(_gray())
        assert out.ndim == 3


# ─── to_hsv ───────────────────────────────────────────────────────────────────

class TestToHsvExtra:
    def test_returns_ndarray(self):
        assert isinstance(to_hsv(_bgr()), np.ndarray)

    def test_shape_3_channels(self):
        out = to_hsv(_bgr())
        assert out.ndim == 3 and out.shape[2] == 3

    def test_dtype_uint8(self):
        assert to_hsv(_bgr()).dtype == np.uint8

    def test_grayscale_input_ok(self):
        out = to_hsv(_gray())
        assert out.ndim == 3


# ─── from_lab ─────────────────────────────────────────────────────────────────

class TestFromLabExtra:
    def test_returns_ndarray(self):
        lab = to_lab(_bgr())
        assert isinstance(from_lab(lab), np.ndarray)

    def test_dtype_uint8(self):
        assert from_lab(to_lab(_bgr())).dtype == np.uint8

    def test_shape_3_channels(self):
        out = from_lab(to_lab(_bgr(16, 24)))
        assert out.ndim == 3 and out.shape[2] == 3

    def test_spatial_shape_preserved(self):
        bgr = _bgr(16, 24)
        out = from_lab(to_lab(bgr))
        assert out.shape[:2] == (16, 24)


# ─── compute_histogram ────────────────────────────────────────────────────────

class TestComputeHistogramExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_histogram(_bgr()), np.ndarray)

    def test_dtype_float32(self):
        assert compute_histogram(_bgr()).dtype == np.float32

    def test_length_matches_bins(self):
        assert len(compute_histogram(_bgr(), bins=64)) == 64

    def test_normalized_sums_to_one(self):
        h = compute_histogram(_bgr(), normalize=True)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_not_normalized(self):
        h = compute_histogram(_bgr(), normalize=False)
        assert h.sum() > 1.0

    def test_grayscale_input(self):
        h = compute_histogram(_gray())
        assert isinstance(h, np.ndarray)


# ─── compare_histograms ───────────────────────────────────────────────────────

class TestCompareHistogramsExtra:
    def _hist(self):
        return compute_histogram(_bgr())

    def test_returns_float(self):
        h = self._hist()
        assert isinstance(compare_histograms(h, h), float)

    def test_correlation_identical_is_one(self):
        h = self._hist()
        val = compare_histograms(h, h, method="correlation")
        assert val == pytest.approx(1.0, abs=1e-5)

    def test_chi_identical_is_zero(self):
        h = self._hist()
        val = compare_histograms(h, h, method="chi")
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_bhattacharyya_identical_is_zero(self):
        h = self._hist()
        val = compare_histograms(h, h, method="bhattacharyya")
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_invalid_method_raises(self):
        h = self._hist()
        with pytest.raises(ValueError):
            compare_histograms(h, h, method="unknown")


# ─── dominant_colors ──────────────────────────────────────────────────────────

class TestDominantColorsExtra:
    def test_returns_ndarray(self):
        assert isinstance(dominant_colors(_bgr(), k=3), np.ndarray)

    def test_shape_k_by_3(self):
        out = dominant_colors(_bgr(), k=2)
        assert out.shape == (2, 3)

    def test_dtype_uint8(self):
        assert dominant_colors(_bgr(), k=2).dtype == np.uint8

    def test_k_one_returns_one_row(self):
        out = dominant_colors(_bgr(), k=1)
        assert out.shape == (1, 3)

    def test_grayscale_input(self):
        out = dominant_colors(_gray(), k=2)
        assert out.shape == (2, 3)


# ─── color_distance ───────────────────────────────────────────────────────────

class TestColorDistanceExtra:
    def test_returns_float(self):
        c = np.array([128, 128, 128], dtype=np.uint8)
        assert isinstance(color_distance(c, c), float)

    def test_identical_colors_zero(self):
        c = np.array([100, 150, 200], dtype=np.uint8)
        assert color_distance(c, c) == pytest.approx(0.0, abs=1e-5)

    def test_nonneg(self):
        c1 = np.array([0, 0, 0], dtype=np.uint8)
        c2 = np.array([255, 255, 255], dtype=np.uint8)
        assert color_distance(c1, c2) >= 0.0

    def test_lab_space(self):
        c1 = np.array([0, 0, 0], dtype=np.uint8)
        c2 = np.array([255, 255, 255], dtype=np.uint8)
        d = color_distance(c1, c2, space="lab")
        assert d > 0.0

    def test_rgb_space(self):
        c1 = np.array([0, 0, 0], dtype=np.uint8)
        c2 = np.array([255, 0, 0], dtype=np.uint8)
        d = color_distance(c1, c2, space="rgb")
        assert d > 0.0

    def test_unknown_space_raises(self):
        c = np.array([100, 100, 100], dtype=np.uint8)
        with pytest.raises(ValueError):
            color_distance(c, c, space="xyz")


# ─── strip_histogram ──────────────────────────────────────────────────────────

class TestStripHistogramExtra:
    def test_returns_ndarray(self):
        assert isinstance(strip_histogram(_bgr()), np.ndarray)

    def test_length_matches_bins(self):
        out = strip_histogram(_bgr(), bins=32)
        assert len(out) == 32

    def test_dtype_float32(self):
        assert strip_histogram(_bgr()).dtype == np.float32

    def test_side_0_top(self):
        out = strip_histogram(_bgr(), side=0)
        assert isinstance(out, np.ndarray)

    def test_side_1_right(self):
        out = strip_histogram(_bgr(), side=1)
        assert isinstance(out, np.ndarray)

    def test_side_2_bottom(self):
        out = strip_histogram(_bgr(), side=2)
        assert isinstance(out, np.ndarray)

    def test_side_3_left(self):
        out = strip_histogram(_bgr(), side=3)
        assert isinstance(out, np.ndarray)

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            strip_histogram(_bgr(), side=4)

    def test_grayscale_input(self):
        out = strip_histogram(_gray(), side=0)
        assert isinstance(out, np.ndarray)
