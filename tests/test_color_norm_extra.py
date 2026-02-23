"""Extra tests for puzzle_reconstruction.preprocessing.color_norm."""
import numpy as np
import pytest
import cv2

from puzzle_reconstruction.preprocessing.color_norm import (
    batch_normalize,
    clahe_normalize,
    gamma_correction,
    normalize_brightness,
    normalize_color,
    white_balance,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _channel(h=64, w=64, b=100, g=120, r=140):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def _rand(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray2d(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── white_balance extras ─────────────────────────────────────────────────────

class TestWhiteBalanceExtra:
    def test_small_image_4x4(self):
        img = _rand(h=4, w=4)
        out = white_balance(img)
        assert out.shape == (4, 4, 3)
        assert out.dtype == np.uint8

    def test_large_image_256x256(self):
        img = _rand(h=256, w=256)
        out = white_balance(img)
        assert out.shape == (256, 256, 3)

    def test_non_square(self):
        img = _rand(h=32, w=96)
        out = white_balance(img)
        assert out.shape == (32, 96, 3)

    def test_pure_red_corrected(self):
        img = _channel(b=50, g=50, r=200)
        out = white_balance(img)
        # R channel mean should decrease, B/G should increase
        assert out[:, :, 2].mean() < img[:, :, 2].mean() + 1

    def test_values_in_0_255(self):
        for seed in range(5):
            out = white_balance(_rand(seed=seed))
            assert out.min() >= 0
            assert out.max() <= 255

    def test_shape_preserved_bgr(self):
        img = _rand(h=48, w=80)
        assert white_balance(img).shape == (48, 80, 3)

    def test_all_zeros_no_crash(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        out = white_balance(img)
        assert out is not None

    def test_all_255_no_crash(self):
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        out = white_balance(img)
        assert out.dtype == np.uint8


# ─── clahe_normalize extras ───────────────────────────────────────────────────

class TestClaheNormalizeExtra:
    def test_small_image(self):
        img = _rand(h=8, w=8)
        out = clahe_normalize(img)
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8

    def test_large_image(self):
        img = _rand(h=256, w=256)
        out = clahe_normalize(img)
        assert out.shape == (256, 256, 3)

    def test_non_square(self):
        img = _rand(h=32, w=80)
        out = clahe_normalize(img)
        assert out.shape == (32, 80, 3)

    def test_values_in_range(self):
        out = clahe_normalize(_rand())
        assert out.min() >= 0
        assert out.max() <= 255

    def test_grayscale_large_image(self):
        gray = np.random.default_rng(1).integers(50, 200, (128, 128), dtype=np.uint8)
        out = clahe_normalize(gray)
        assert out.dtype == np.uint8
        assert out.ndim == 2

    def test_clip_limit_low_valid(self):
        out = clahe_normalize(_rand(), clip_limit=0.5)
        assert out.dtype == np.uint8

    def test_clip_limit_high_valid(self):
        out = clahe_normalize(_rand(), clip_limit=16.0)
        assert out.dtype == np.uint8

    def test_idempotent_shape(self):
        img = _rand()
        out1 = clahe_normalize(img)
        out2 = clahe_normalize(out1)
        assert out2.shape == img.shape


# ─── gamma_correction extras ──────────────────────────────────────────────────

class TestGammaCorrectionExtra:
    def test_gamma_0_1_very_bright(self):
        img = _bgr(val=100)
        out = gamma_correction(img, gamma=0.1)
        assert float(out.mean()) > float(img.mean())

    def test_gamma_10_very_dark(self):
        img = _bgr(val=200)
        out = gamma_correction(img, gamma=10.0)
        assert float(out.mean()) < float(img.mean())

    def test_non_square(self):
        img = _rand(h=32, w=80)
        out = gamma_correction(img, gamma=1.5)
        assert out.shape == (32, 80, 3)

    def test_grayscale_2d(self):
        gray = _gray2d(val=128)
        out = gamma_correction(gray, gamma=0.5)
        assert out.dtype == np.uint8

    def test_small_image_4x4(self):
        img = _rand(h=4, w=4)
        out = gamma_correction(img, gamma=1.2)
        assert out.shape == (4, 4, 3)

    def test_values_in_range_high_gamma(self):
        out = gamma_correction(_rand(), gamma=5.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_values_in_range_low_gamma(self):
        out = gamma_correction(_rand(), gamma=0.2)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_gamma_2_mid_value(self):
        img = _bgr(val=127)
        out = gamma_correction(img, gamma=2.0)
        # 127/255 ^ 2 * 255 ≈ 63
        assert float(out.mean()) < float(img.mean())


# ─── normalize_brightness extras ─────────────────────────────────────────────

class TestNormalizeBrightnessExtra:
    def test_target_50(self):
        img = _bgr(val=200)
        out = normalize_brightness(img, target=50.0)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        assert float(gray.mean()) < 150.0

    def test_target_255_max(self):
        img = _bgr(val=100)
        out = normalize_brightness(img, target=255.0)
        assert out.dtype == np.uint8

    def test_non_square(self):
        img = _rand(h=32, w=96)
        out = normalize_brightness(img, target=128.0)
        assert out.shape == (32, 96, 3)

    def test_uint8_output(self):
        out = normalize_brightness(_rand(), target=150.0)
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        out = normalize_brightness(_rand(), target=100.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_large_image(self):
        img = _rand(h=256, w=256)
        out = normalize_brightness(img, target=128.0)
        assert out.shape == (256, 256, 3)


# ─── normalize_color extras ───────────────────────────────────────────────────

class TestNormalizeColorExtra:
    def test_small_image_4x4(self):
        img = _rand(h=4, w=4)
        out = normalize_color(img)
        assert out is not None

    def test_large_image(self):
        img = _rand(h=256, w=256)
        out = normalize_color(img)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8

    def test_non_square(self):
        img = _rand(h=32, w=96)
        out = normalize_color(img)
        assert out.shape == (32, 96, 3)

    def test_values_in_range(self):
        out = normalize_color(_rand(seed=42))
        assert out.min() >= 0
        assert out.max() <= 255

    def test_multiple_seeds_no_crash(self):
        for seed in range(5):
            out = normalize_color(_rand(seed=seed))
            assert out.dtype == np.uint8


# ─── batch_normalize extras ───────────────────────────────────────────────────

class TestBatchNormalizeExtra:
    def test_single_image(self):
        result = batch_normalize([_rand()])
        assert len(result) == 1
        assert result[0].dtype == np.uint8

    def test_ten_images(self):
        images = [_rand(seed=i) for i in range(10)]
        result = batch_normalize(images)
        assert len(result) == 10

    def test_all_uint8(self):
        images = [_rand(seed=i) for i in range(4)]
        for r in batch_normalize(images):
            assert r.dtype == np.uint8

    def test_shapes_preserved(self):
        images = [_rand(h=48, w=64, seed=i) for i in range(3)]
        for r in batch_normalize(images):
            assert r.shape == (48, 64, 3)

    def test_values_in_range(self):
        images = [_rand(seed=i) for i in range(3)]
        for r in batch_normalize(images):
            assert r.min() >= 0
            assert r.max() <= 255

    def test_reference_idx_last(self):
        images = [_rand(seed=i) for i in range(4)]
        result = batch_normalize(images, reference_idx=3)
        assert len(result) == 4
        for r in result:
            assert r.dtype == np.uint8

    def test_non_square_images(self):
        images = [_rand(h=32, w=64, seed=i) for i in range(3)]
        result = batch_normalize(images)
        for r in result:
            assert r.shape == (32, 64, 3)
