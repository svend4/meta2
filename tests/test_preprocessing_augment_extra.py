"""Extra tests for puzzle_reconstruction/preprocessing/augment.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.augment import (
    random_crop,
    random_rotate,
    add_gaussian_noise,
    add_salt_pepper,
    brightness_jitter,
    jpeg_compress,
    simulate_scan_noise,
    augment_batch,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _rng(seed=42):
    return np.random.RandomState(seed)


# ─── random_crop ─────────────────────────────────────────────────────────────

class TestRandomCropExtra:
    def test_preserves_shape(self):
        img = _bgr()
        out = random_crop(img, rng=_rng())
        assert out.shape == img.shape

    def test_grayscale(self):
        img = _gray()
        out = random_crop(img, rng=_rng())
        assert out.shape == img.shape

    def test_full_scale(self):
        img = _bgr()
        out = random_crop(img, min_scale=1.0, max_scale=1.0, rng=_rng())
        assert out.shape == img.shape

    def test_dtype_preserved(self):
        img = _bgr()
        out = random_crop(img, rng=_rng())
        assert out.dtype == np.uint8


# ─── random_rotate ───────────────────────────────────────────────────────────

class TestRandomRotateExtra:
    def test_preserves_shape_no_expand(self):
        img = _bgr()
        out = random_rotate(img, max_angle=15.0, expand=False, rng=_rng())
        assert out.shape == img.shape

    def test_expand_may_change_shape(self):
        img = _bgr(100, 50)
        out = random_rotate(img, max_angle=30.0, expand=True, rng=_rng())
        # With expand, shape may differ
        assert out.ndim == 3

    def test_zero_angle(self):
        img = _bgr()
        out = random_rotate(img, max_angle=0.0, rng=_rng())
        assert out.shape == img.shape

    def test_grayscale(self):
        img = _gray()
        out = random_rotate(img, rng=_rng())
        assert out.ndim == 2


# ─── add_gaussian_noise ─────────────────────────────────────────────────────

class TestAddGaussianNoiseExtra:
    def test_zero_sigma_returns_same(self):
        img = _bgr()
        out = add_gaussian_noise(img, sigma=0.0)
        assert np.array_equal(out, img)

    def test_negative_sigma_returns_same(self):
        img = _bgr()
        out = add_gaussian_noise(img, sigma=-5.0)
        assert np.array_equal(out, img)

    def test_dtype_uint8(self):
        img = _bgr()
        out = add_gaussian_noise(img, sigma=20.0, rng=_rng())
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        img = _bgr()
        out = add_gaussian_noise(img, sigma=10.0, rng=_rng())
        assert out.shape == img.shape

    def test_adds_noise(self):
        img = _bgr(50, 50, 128)
        out = add_gaussian_noise(img, sigma=30.0, rng=_rng())
        assert not np.array_equal(out, img)


# ─── add_salt_pepper ────────────────────────────────────────────────────────

class TestAddSaltPepperExtra:
    def test_zero_amount(self):
        img = _bgr()
        out = add_salt_pepper(img, amount=0.0, rng=_rng())
        assert np.array_equal(out, img)

    def test_shape_preserved(self):
        img = _bgr()
        out = add_salt_pepper(img, amount=0.05, rng=_rng())
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        out = add_salt_pepper(img, rng=_rng())
        assert out.dtype == np.uint8


# ─── brightness_jitter ──────────────────────────────────────────────────────

class TestBrightnessJitterExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = brightness_jitter(img, rng=_rng())
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        out = brightness_jitter(img, rng=_rng())
        assert out.dtype == np.uint8


# ─── jpeg_compress ──────────────────────────────────────────────────────────

class TestJpegCompressExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = jpeg_compress(img, quality=50)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        out = jpeg_compress(img)
        assert out.dtype == np.uint8

    def test_quality_clamped(self):
        img = _bgr()
        # Should not raise even with extreme values
        out = jpeg_compress(img, quality=0)
        assert out.dtype == np.uint8
        out = jpeg_compress(img, quality=200)
        assert out.dtype == np.uint8


# ─── simulate_scan_noise ────────────────────────────────────────────────────

class TestSimulateScanNoiseExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = simulate_scan_noise(img, rng=_rng())
        assert out.shape == img.shape

    def test_all_disabled(self):
        img = _bgr()
        out = simulate_scan_noise(img, gaussian_sigma=0, sp_amount=0,
                                   jpeg_quality=100, yellowing=0, rng=_rng())
        assert np.array_equal(out, img)

    def test_dtype_uint8(self):
        img = _bgr()
        out = simulate_scan_noise(img, rng=_rng())
        assert out.dtype == np.uint8


# ─── augment_batch ──────────────────────────────────────────────────────────

class TestAugmentBatchExtra:
    def test_output_length(self):
        imgs = [_bgr(), _bgr()]
        result = augment_batch(imgs, n_augments=2, seed=42)
        assert len(result) == 2 * (1 + 2)

    def test_empty_input(self):
        result = augment_batch([], n_augments=3)
        assert result == []

    def test_zero_augments(self):
        imgs = [_bgr()]
        result = augment_batch(imgs, n_augments=0)
        assert len(result) == 1

    def test_all_flags_disabled(self):
        imgs = [_bgr()]
        result = augment_batch(imgs, n_augments=1, rotate=False,
                               crop=False, noise=False, jitter=False,
                               jpeg=False, seed=42)
        assert len(result) == 2
