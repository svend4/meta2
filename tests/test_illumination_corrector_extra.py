"""Extra tests for puzzle_reconstruction.preprocessing.illumination_corrector."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _gradient(h=64, w=64):
    col = np.arange(w, dtype=np.float32) * (255.0 / max(w - 1, 1))
    return np.tile(col.astype(np.uint8), (h, 1))


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = _gradient(h, w)
    img[:, :, 2] = 100
    return img


def _uneven(h=64, w=64):
    img = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            img[i, j] = 255.0 * (1.0 - (i + j) / (h + w))
    return np.clip(img, 10, 245).astype(np.uint8)


# ─── IlluminationParams extras ────────────────────────────────────────────────

class TestIlluminationParamsExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(IlluminationParams()), str)

    def test_none_method_valid(self):
        p = IlluminationParams(method="none")
        assert p.method == "none"

    def test_background_method_valid(self):
        p = IlluminationParams(method="background")
        assert p.method == "background"

    def test_homomorph_method_valid(self):
        p = IlluminationParams(method="homomorph")
        assert p.method == "homomorph"

    def test_retinex_method_valid(self):
        p = IlluminationParams(method="retinex")
        assert p.method == "retinex"

    def test_blur_ksize_3_valid(self):
        p = IlluminationParams(blur_ksize=3)
        assert p.blur_ksize == 3

    def test_blur_ksize_99_valid(self):
        p = IlluminationParams(blur_ksize=99)
        assert p.blur_ksize == 99

    def test_target_mean_zero_valid(self):
        p = IlluminationParams(target_mean=0.0)
        assert p.target_mean == pytest.approx(0.0)

    def test_target_mean_255_valid(self):
        p = IlluminationParams(target_mean=255.0)
        assert p.target_mean == pytest.approx(255.0)

    def test_retinex_scales_len(self):
        p = IlluminationParams()
        assert len(p.retinex_scales) >= 1


# ─── estimate_background extras ───────────────────────────────────────────────

class TestEstimateBackgroundExtra:
    def test_small_image_ksize_3(self):
        img = _gray(h=8, w=8)
        bg = estimate_background(img, ksize=3)
        assert bg.shape == (8, 8)
        assert bg.dtype == np.float32

    def test_large_image(self):
        img = _gradient(h=128, w=128)
        bg = estimate_background(img, ksize=11)
        assert bg.shape == (128, 128)

    def test_non_square(self):
        img = _gradient(h=32, w=64)
        bg = estimate_background(img, ksize=7)
        assert bg.shape == (32, 64)

    def test_bgr_output_2d(self):
        bg = estimate_background(_bgr(), ksize=11)
        assert bg.ndim == 2

    def test_values_nonneg(self):
        bg = estimate_background(_gradient(), ksize=11)
        assert float(bg.min()) >= 0.0

    def test_values_le_255(self):
        bg = estimate_background(_gradient(), ksize=11)
        assert float(bg.max()) <= 255.0

    def test_uniform_bg_approx_constant(self):
        img = _gray(value=80)
        bg = estimate_background(img, ksize=11)
        np.testing.assert_allclose(bg, 80.0, atol=10.0)


# ─── subtract_background extras ───────────────────────────────────────────────

class TestSubtractBackgroundExtra:
    def test_small_image(self):
        img = _gray(h=8, w=8)
        result = subtract_background(img, ksize=3)
        assert result.shape == (8, 8)
        assert result.dtype == np.uint8

    def test_large_image(self):
        img = _gradient(h=128, w=128)
        result = subtract_background(img, ksize=11)
        assert result.shape == (128, 128)

    def test_values_in_uint8_range(self):
        result = subtract_background(_gradient(), ksize=11)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bgr_shape_preserved(self):
        result = subtract_background(_bgr(), ksize=11)
        assert result.shape == (64, 64, 3)

    def test_target_mean_100_darker(self):
        result100 = subtract_background(_gradient(), ksize=11, target_mean=100.0)
        result200 = subtract_background(_gradient(), ksize=11, target_mean=200.0)
        assert result200.mean() > result100.mean()

    def test_precomputed_bg_accepted(self):
        img = _gradient()
        bg = estimate_background(img, ksize=11)
        result = subtract_background(img, background=bg)
        assert result.dtype == np.uint8
        assert result.shape == img.shape


# ─── correct_by_homomorph extras ─────────────────────────────────────────────

class TestCorrectByHomomorphExtra:
    def test_small_image(self):
        img = _gray(h=16, w=16)
        result = correct_by_homomorph(img, d0=5.0)
        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_large_image(self):
        img = _gradient(h=128, w=128)
        result = correct_by_homomorph(img, d0=15.0)
        assert result.shape == (128, 128)

    def test_non_square(self):
        img = _gradient(h=32, w=80)
        result = correct_by_homomorph(img, d0=10.0)
        assert result.shape == (32, 80)

    def test_bgr_3channels(self):
        result = correct_by_homomorph(_bgr(), d0=10.0)
        assert result.shape == (64, 64, 3)

    def test_values_uint8_range(self):
        result = correct_by_homomorph(_gradient(), d0=10.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_d0_large_valid(self):
        result = correct_by_homomorph(_gradient(), d0=500.0)
        assert result.dtype == np.uint8


# ─── correct_by_retinex extras ────────────────────────────────────────────────

class TestCorrectByRetinexExtra:
    def test_small_image(self):
        img = _gray(h=16, w=16)
        result = correct_by_retinex(img, scales=[5.0])
        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_large_image(self):
        img = _gradient(h=128, w=128)
        result = correct_by_retinex(img)
        assert result.shape == (128, 128)

    def test_three_scales(self):
        result = correct_by_retinex(_gradient(), scales=[10.0, 50.0, 100.0])
        assert result.dtype == np.uint8

    def test_non_square(self):
        img = _gradient(h=32, w=80)
        result = correct_by_retinex(img, scales=[15.0])
        assert result.shape == (32, 80)

    def test_bgr_3channels(self):
        result = correct_by_retinex(_bgr())
        assert result.shape == (64, 64, 3)

    def test_values_in_uint8_range(self):
        result = correct_by_retinex(_gradient())
        assert result.min() >= 0
        assert result.max() <= 255


# ─── correct_illumination extras ──────────────────────────────────────────────

class TestCorrectIlluminationExtra:
    def test_none_returns_different_object(self):
        img = _gradient()
        params = IlluminationParams(method="none")
        result = correct_illumination(img, params)
        assert result is not img

    def test_none_values_identical(self):
        img = _gradient()
        params = IlluminationParams(method="none")
        result = correct_illumination(img, params)
        np.testing.assert_array_equal(result, img)

    def test_bgr_background_method(self):
        params = IlluminationParams(method="background", blur_ksize=11)
        result = correct_illumination(_bgr(), params)
        assert result.ndim == 3

    def test_bgr_homomorph_method(self):
        params = IlluminationParams(method="homomorph")
        result = correct_illumination(_bgr(), params)
        assert result.ndim == 3

    def test_bgr_retinex_method(self):
        params = IlluminationParams(method="retinex")
        result = correct_illumination(_bgr(), params)
        assert result.ndim == 3

    def test_all_methods_uint8(self):
        img = _gradient(40, 48)
        for method in ("background", "homomorph", "retinex", "none"):
            params = IlluminationParams(method=method, blur_ksize=11)
            result = correct_illumination(img, params)
            assert result.dtype == np.uint8, f"Failed for method={method!r}"

    def test_non_square_shape_preserved(self):
        img = _gradient(h=32, w=80)
        params = IlluminationParams(method="background", blur_ksize=7)
        result = correct_illumination(img, params)
        assert result.shape == (32, 80)


# ─── batch_correct extras ─────────────────────────────────────────────────────

class TestBatchCorrectExtra:
    def test_single_image(self):
        result = batch_correct([_gradient()])
        assert len(result) == 1
        assert result[0].dtype == np.uint8

    def test_five_images(self):
        imgs = [_gradient() for _ in range(5)]
        result = batch_correct(imgs)
        assert len(result) == 5

    def test_mixed_types(self):
        imgs = [_gray(), _bgr(), _gradient()]
        params = IlluminationParams(method="background", blur_ksize=11)
        result = batch_correct(imgs, params=params)
        assert len(result) == 3
        for r in result:
            assert r.dtype == np.uint8

    def test_shapes_preserved_varied(self):
        imgs = [_gradient(h=32, w=40), _gradient(h=48, w=64)]
        result = batch_correct(imgs)
        assert result[0].shape == (32, 40)
        assert result[1].shape == (48, 64)

    def test_none_method_passthrough(self):
        img = _gradient()
        params = IlluminationParams(method="none")
        result = batch_correct([img], params=params)
        np.testing.assert_array_equal(result[0], img)


# ─── estimate_uniformity extras ───────────────────────────────────────────────

class TestEstimateUniformityExtra:
    def test_value_in_unit_interval(self):
        result = estimate_uniformity(_gradient(), ksize=11)
        assert 0.0 <= result <= 1.0

    def test_uniform_gray_near_one(self):
        result = estimate_uniformity(_gray(value=100), ksize=11)
        assert result >= 0.8

    def test_returns_float(self):
        result = estimate_uniformity(_gray())
        assert isinstance(result, float)

    def test_bgr_image_valid(self):
        result = estimate_uniformity(_bgr(), ksize=11)
        assert 0.0 <= result <= 1.0

    def test_small_image(self):
        img = _gray(h=8, w=8)
        result = estimate_uniformity(img, ksize=3)
        assert 0.0 <= result <= 1.0

    def test_large_image(self):
        img = _gradient(h=128, w=128)
        result = estimate_uniformity(img, ksize=11)
        assert 0.0 <= result <= 1.0

    def test_uneven_not_greater_than_uniform(self):
        u_uniform = estimate_uniformity(_gray(value=128), ksize=11)
        u_uneven = estimate_uniformity(_uneven(), ksize=11)
        assert u_uneven <= u_uniform + 0.1
