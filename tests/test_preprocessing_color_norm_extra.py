"""Extra tests for puzzle_reconstruction/preprocessing/color_norm.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.color_norm import (
    normalize_color,
    clahe_normalize,
    white_balance,
    gamma_correction,
    normalize_brightness,
    batch_normalize,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── normalize_color ────────────────────────────────────────────────────────

class TestNormalizeColorExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = normalize_color(img)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        out = normalize_color(img)
        assert out.dtype == np.uint8

    def test_empty_image(self):
        img = np.array([], dtype=np.uint8)
        out = normalize_color(img)
        assert out.size == 0


# ─── clahe_normalize ────────────────────────────────────────────────────────

class TestClaheNormalizeExtra:
    def test_bgr_shape_preserved(self):
        img = _bgr()
        out = clahe_normalize(img)
        assert out.shape == img.shape

    def test_grayscale(self):
        img = _gray()
        out = clahe_normalize(img)
        assert out.shape == img.shape
        assert out.ndim == 2

    def test_dtype_uint8(self):
        img = _bgr()
        out = clahe_normalize(img)
        assert out.dtype == np.uint8


# ─── white_balance ──────────────────────────────────────────────────────────

class TestWhiteBalanceExtra:
    def test_neutral_image_unchanged(self):
        img = _bgr(val=128)
        out = white_balance(img)
        # Neutral image should be roughly unchanged
        assert np.allclose(out.astype(float), img.astype(float), atol=2)

    def test_non_3_channel_returns_same(self):
        img = _gray()
        out = white_balance(img)
        assert np.array_equal(out, img)

    def test_dtype_uint8(self):
        img = _bgr()
        out = white_balance(img)
        assert out.dtype == np.uint8

    def test_dark_image_returns_same(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = white_balance(img)
        assert np.array_equal(out, img)


# ─── gamma_correction ───────────────────────────────────────────────────────

class TestGammaCorrectionExtra:
    def test_identity_gamma(self):
        img = _bgr()
        out = gamma_correction(img, gamma=1.0)
        assert np.array_equal(out, img)

    def test_shape_preserved(self):
        img = _bgr()
        out = gamma_correction(img, gamma=0.5)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        out = gamma_correction(img, gamma=2.0)
        assert out.dtype == np.uint8

    def test_grayscale(self):
        img = _gray()
        out = gamma_correction(img, gamma=0.5)
        assert out.ndim == 2


# ─── normalize_brightness ───────────────────────────────────────────────────

class TestNormalizeBrightnessExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = normalize_brightness(img, target=200.0)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        out = normalize_brightness(img)
        assert out.dtype == np.uint8

    def test_dark_image_no_crash(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = normalize_brightness(img)
        assert np.array_equal(out, img)

    def test_with_mask(self):
        img = _bgr(val=100)
        mask = np.ones((50, 50), dtype=np.uint8)
        out = normalize_brightness(img, target=200.0, mask=mask)
        assert out.dtype == np.uint8


# ─── batch_normalize ────────────────────────────────────────────────────────

class TestBatchNormalizeExtra:
    def test_empty(self):
        assert batch_normalize([]) == []

    def test_preserves_count(self):
        imgs = [_bgr(), _bgr()]
        result = batch_normalize(imgs)
        assert len(result) == 2

    def test_result_shapes(self):
        imgs = [_bgr(30, 40), _bgr(50, 60)]
        result = batch_normalize(imgs)
        assert result[0].shape == (30, 40, 3)
        assert result[1].shape == (50, 60, 3)
