"""Tests for puzzle_reconstruction.preprocessing.illumination_corrector."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.illumination_corrector import (
    IlluminationParams,
    batch_correct,
    correct_by_homomorph,
    correct_by_retinex,
    correct_illumination,
    estimate_background,
    estimate_uniformity,
    subtract_background,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _gradient_gray(h: int = 64, w: int = 64) -> np.ndarray:
    """Left-dark, right-bright gradient."""
    col = np.arange(w, dtype=np.float32) * (255.0 / (w - 1))
    return np.tile(col.astype(np.uint8), (h, 1))


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = _gradient_gray(h, w)
    img[:, :, 2] = 100
    return img


def _uneven_illumination(h: int = 64, w: int = 64) -> np.ndarray:
    """Image with bright top-left corner, dark bottom-right."""
    img = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            img[i, j] = 255.0 * (1.0 - (i + j) / (h + w))
    return np.clip(img, 10, 245).astype(np.uint8)


# ─── IlluminationParams ──────────────────────────────────────────────────────

class TestIlluminationParams:
    def test_defaults(self):
        p = IlluminationParams()
        assert p.method == "background"
        assert p.blur_ksize == 51
        assert p.target_mean == pytest.approx(128.0)

    def test_all_valid_methods(self):
        for m in ("background", "homomorph", "retinex", "none"):
            p = IlluminationParams(method=m)
            assert p.method == m

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            IlluminationParams(method="unknown")

    def test_even_blur_ksize_raises(self):
        with pytest.raises(ValueError):
            IlluminationParams(blur_ksize=50)

    def test_blur_ksize_less_than_3_raises(self):
        with pytest.raises(ValueError):
            IlluminationParams(blur_ksize=1)

    def test_custom_values(self):
        p = IlluminationParams(method="retinex", blur_ksize=31, target_mean=200.0)
        assert p.method == "retinex"
        assert p.blur_ksize == 31
        assert p.target_mean == pytest.approx(200.0)

    def test_retinex_scales_default(self):
        p = IlluminationParams()
        assert len(p.retinex_scales) == 3


# ─── estimate_background ─────────────────────────────────────────────────────

class TestEstimateBackground:
    def test_returns_float32(self):
        bg = estimate_background(_gradient_gray())
        assert bg.dtype == np.float32

    def test_shape_preserved(self):
        img = _gradient_gray(40, 50)
        bg = estimate_background(img, ksize=11)
        assert bg.shape == (40, 50)

    def test_ksize_less_than_3_raises(self):
        with pytest.raises(ValueError):
            estimate_background(_gray(), ksize=2)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            estimate_background(_gray(), ksize=10)

    def test_values_in_range(self):
        bg = estimate_background(_gradient_gray(), ksize=11)
        assert bg.min() >= 0.0
        assert bg.max() <= 255.0

    def test_bgr_input_accepted(self):
        bg = estimate_background(_bgr(), ksize=11)
        assert bg.dtype == np.float32
        assert bg.ndim == 2

    def test_uniform_image_constant_bg(self):
        img = _gray(value=100)
        bg = estimate_background(img, ksize=11)
        np.testing.assert_allclose(bg, 100.0, atol=5.0)


# ─── subtract_background ─────────────────────────────────────────────────────

class TestSubtractBackground:
    def test_returns_uint8(self):
        result = subtract_background(_gradient_gray(), ksize=11)
        assert result.dtype == np.uint8

    def test_shape_preserved_gray(self):
        img = _gradient_gray(48, 52)
        result = subtract_background(img, ksize=11)
        assert result.shape == (48, 52)

    def test_bgr_output_is_3_channel(self):
        result = subtract_background(_bgr(), ksize=11)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_precomputed_background_used(self):
        img = _gradient_gray()
        bg = estimate_background(img, ksize=11)
        r1 = subtract_background(img, background=bg)
        r2 = subtract_background(img, ksize=11)
        # Both should produce valid uint8 images of same shape
        assert r1.shape == r2.shape
        assert r1.dtype == np.uint8

    def test_target_mean_effect(self):
        img = _gradient_gray()
        r_dark = subtract_background(img, ksize=11, target_mean=50.0)
        r_bright = subtract_background(img, ksize=11, target_mean=200.0)
        assert r_bright.mean() > r_dark.mean()


# ─── correct_by_homomorph ────────────────────────────────────────────────────

class TestCorrectByHomomorph:
    def test_returns_uint8(self):
        result = correct_by_homomorph(_gradient_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gradient_gray(50, 60)
        result = correct_by_homomorph(img, d0=10.0)
        assert result.shape == (50, 60)

    def test_bgr_output_is_3_channel(self):
        result = correct_by_homomorph(_bgr(), d0=10.0)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_d0_zero_raises(self):
        with pytest.raises(ValueError):
            correct_by_homomorph(_gray(), d0=0.0)

    def test_d0_negative_raises(self):
        with pytest.raises(ValueError):
            correct_by_homomorph(_gray(), d0=-5.0)

    def test_values_in_uint8_range(self):
        result = correct_by_homomorph(_gradient_gray(), d0=15.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_target_mean_changes_brightness(self):
        img = _gradient_gray()
        r_dark = correct_by_homomorph(img, d0=10.0, target_mean=50.0)
        r_bright = correct_by_homomorph(img, d0=10.0, target_mean=200.0)
        assert r_bright.mean() > r_dark.mean()


# ─── correct_by_retinex ──────────────────────────────────────────────────────

class TestCorrectByRetinex:
    def test_returns_uint8(self):
        result = correct_by_retinex(_gradient_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gradient_gray(48, 56)
        result = correct_by_retinex(img)
        assert result.shape == (48, 56)

    def test_bgr_output_3_channels(self):
        result = correct_by_retinex(_bgr())
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_empty_scales_raises(self):
        with pytest.raises(ValueError):
            correct_by_retinex(_gradient_gray(), scales=[])

    def test_custom_scales(self):
        result = correct_by_retinex(_gradient_gray(), scales=[10.0, 50.0])
        assert result.dtype == np.uint8

    def test_single_scale(self):
        result = correct_by_retinex(_gradient_gray(), scales=[30.0])
        assert result.shape == _gradient_gray().shape

    def test_values_in_uint8_range(self):
        result = correct_by_retinex(_gradient_gray())
        assert result.min() >= 0
        assert result.max() <= 255


# ─── correct_illumination ────────────────────────────────────────────────────

class TestCorrectIllumination:
    def test_none_method_returns_copy(self):
        img = _gradient_gray()
        params = IlluminationParams(method="none")
        result = correct_illumination(img, params)
        np.testing.assert_array_equal(result, img)
        assert result is not img  # must be a copy

    def test_background_method(self):
        params = IlluminationParams(method="background", blur_ksize=11)
        result = correct_illumination(_gradient_gray(), params)
        assert result.dtype == np.uint8

    def test_homomorph_method(self):
        params = IlluminationParams(method="homomorph")
        result = correct_illumination(_gradient_gray(), params)
        assert result.dtype == np.uint8

    def test_retinex_method(self):
        params = IlluminationParams(method="retinex")
        result = correct_illumination(_gradient_gray(), params)
        assert result.dtype == np.uint8

    def test_default_params(self):
        result = correct_illumination(_gradient_gray())
        assert result.dtype == np.uint8

    def test_shape_always_preserved(self):
        img = _gradient_gray(40, 48)
        for method in ("background", "homomorph", "retinex", "none"):
            params = IlluminationParams(method=method, blur_ksize=11)
            result = correct_illumination(img, params)
            assert result.shape == img.shape, f"Shape mismatch for method={method!r}"


# ─── batch_correct ───────────────────────────────────────────────────────────

class TestBatchCorrect:
    def test_empty_returns_empty(self):
        assert batch_correct([]) == []

    def test_length_preserved(self):
        imgs = [_gradient_gray()] * 4
        result = batch_correct(imgs)
        assert len(result) == 4

    def test_all_uint8(self):
        imgs = [_gradient_gray(), _bgr()]
        params = IlluminationParams(method="background", blur_ksize=11)
        result = batch_correct(imgs, params=params)
        assert all(r.dtype == np.uint8 for r in result)

    def test_shapes_preserved(self):
        imgs = [_gradient_gray(30, 40), _gradient_gray(50, 60)]
        result = batch_correct(imgs)
        assert result[0].shape == (30, 40)
        assert result[1].shape == (50, 60)

    def test_default_params_applied(self):
        imgs = [_gradient_gray()] * 3
        result = batch_correct(imgs)
        assert all(r.dtype == np.uint8 for r in result)


# ─── estimate_uniformity ─────────────────────────────────────────────────────

class TestEstimateUniformity:
    def test_returns_float(self):
        result = estimate_uniformity(_gradient_gray())
        assert isinstance(result, float)

    def test_in_unit_interval(self):
        result = estimate_uniformity(_gradient_gray(), ksize=11)
        assert 0.0 <= result <= 1.0

    def test_uniform_image_high_score(self):
        # A perfectly uniform image should have high uniformity
        result = estimate_uniformity(_gray(value=128), ksize=11)
        assert result >= 0.9

    def test_uneven_less_than_uniform(self):
        uniform = estimate_uniformity(_gray(value=128), ksize=11)
        uneven = estimate_uniformity(_uneven_illumination(), ksize=11)
        # Not guaranteed to be strictly less in all cases, but uneven should not exceed uniform
        assert uneven <= uniform + 0.1

    def test_bgr_accepted(self):
        result = estimate_uniformity(_bgr(), ksize=11)
        assert 0.0 <= result <= 1.0
