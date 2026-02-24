"""Extra tests for puzzle_reconstruction/preprocessing/illumination_normalizer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.illumination_normalizer import (
    IllumConfig,
    IllumResult,
    apply_clahe,
    batch_normalize_illumination,
    estimate_illumination,
    normalize_illumination,
    normalize_mean_std,
    subtract_background,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _const(h: int = 64, w: int = 64, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h: int = 64, w: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _mask_center(h: int = 64, w: int = 64) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    return m


# ─── IllumConfig (extra) ──────────────────────────────────────────────────────

class TestIllumConfigExtra:
    def test_default_blur_ksize(self):
        assert IllumConfig().blur_ksize == 51

    def test_default_target_mean(self):
        assert IllumConfig().target_mean == pytest.approx(128.0)

    def test_default_target_std(self):
        assert IllumConfig().target_std == pytest.approx(60.0)

    def test_default_clip_limit(self):
        assert IllumConfig().clip_limit == pytest.approx(2.0)

    def test_default_tile_grid_size(self):
        assert IllumConfig().tile_grid_size == (8, 8)

    def test_custom_blur_ksize_odd(self):
        cfg = IllumConfig(blur_ksize=21)
        assert cfg.blur_ksize == 21

    def test_even_blur_ksize_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=10)

    def test_blur_ksize_below_3_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=1)

    def test_target_mean_zero_valid(self):
        assert IllumConfig(target_mean=0.0).target_mean == pytest.approx(0.0)

    def test_target_mean_255_valid(self):
        assert IllumConfig(target_mean=255.0).target_mean == pytest.approx(255.0)

    def test_target_mean_negative_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_mean=-1.0)

    def test_target_mean_above_255_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_mean=256.0)

    def test_target_std_zero_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_std=0.0)

    def test_target_std_negative_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_std=-5.0)

    def test_clip_limit_zero_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(clip_limit=0.0)

    def test_clip_limit_negative_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(clip_limit=-1.0)

    def test_tile_grid_size_zero_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(tile_grid_size=(0, 8))

    def test_tile_grid_size_custom(self):
        cfg = IllumConfig(tile_grid_size=(4, 4))
        assert cfg.tile_grid_size == (4, 4)


# ─── IllumResult (extra) ──────────────────────────────────────────────────────

class TestIllumResultExtra:
    def test_shape_property_2d(self):
        r = IllumResult(image=_const(), original_mean=128.0, original_std=50.0, method="mean_std")
        assert r.shape == (64, 64)

    def test_shape_property_3d(self):
        r = IllumResult(image=_bgr(), original_mean=100.0, original_std=40.0, method="clahe")
        assert r.shape == (64, 64, 3)

    def test_method_stored(self):
        r = IllumResult(image=_const(), original_mean=128.0, original_std=50.0, method="background")
        assert r.method == "background"

    def test_invalid_ndim_raises(self):
        bad = np.zeros((4, 4, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            IllumResult(image=bad, original_mean=0.0, original_std=0.0, method="x")

    def test_negative_original_mean_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=-1.0, original_std=0.0, method="x")

    def test_original_mean_above_255_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=300.0, original_std=0.0, method="x")

    def test_negative_original_std_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=100.0, original_std=-5.0, method="x")

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_const(), original_mean=100.0, original_std=10.0, method="")

    def test_original_mean_zero_valid(self):
        r = IllumResult(image=_const(), original_mean=0.0, original_std=0.0, method="x")
        assert r.original_mean == pytest.approx(0.0)

    def test_original_std_zero_valid(self):
        r = IllumResult(image=_const(), original_mean=100.0, original_std=0.0, method="x")
        assert r.original_std == pytest.approx(0.0)


# ─── estimate_illumination (extra) ────────────────────────────────────────────

class TestEstimateIlluminationExtra:
    def test_returns_float64(self):
        result = estimate_illumination(_gray())
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        img = _gray(32, 48)
        result = estimate_illumination(img)
        assert result.shape == (32, 48)

    def test_constant_image_uniform_bg(self):
        bg = estimate_illumination(_const(val=100))
        # Blurred constant image should still be roughly constant
        assert bg.min() > 0
        assert bg.max() <= 255.0

    def test_invalid_ksize_even_raises(self):
        with pytest.raises(ValueError):
            estimate_illumination(_gray(), blur_ksize=10)

    def test_invalid_ksize_too_small_raises(self):
        with pytest.raises(ValueError):
            estimate_illumination(_gray(), blur_ksize=1)

    def test_ksize_3_valid(self):
        result = estimate_illumination(_gray(), blur_ksize=3)
        assert result.shape == _gray().shape

    def test_ksize_5_valid(self):
        result = estimate_illumination(_gray(), blur_ksize=5)
        assert result.dtype == np.float64

    def test_bgr_input(self):
        result = estimate_illumination(_bgr())
        assert result.shape == (64, 64)

    def test_values_nonneg(self):
        result = estimate_illumination(_gray())
        assert (result >= 0).all()


# ─── subtract_background (extra) ──────────────────────────────────────────────

class TestSubtractBackgroundExtra:
    def test_returns_uint8(self):
        result = subtract_background(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(32, 48)
        result = subtract_background(img)
        assert result.shape == (32, 48)

    def test_output_in_0_255(self):
        result = subtract_background(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image_offset(self):
        # After background subtraction with offset=128, constant image → ~128
        result = subtract_background(_const(val=128), offset=128.0)
        assert result.dtype == np.uint8

    def test_bgr_input(self):
        result = subtract_background(_bgr())
        assert result.shape == (64, 64)

    def test_invalid_ksize_raises(self):
        with pytest.raises(ValueError):
            subtract_background(_gray(), blur_ksize=4)

    def test_offset_zero(self):
        result = subtract_background(_gray(), offset=0.0)
        assert result.dtype == np.uint8

    def test_offset_255(self):
        result = subtract_background(_const(val=0), offset=255.0)
        assert result.max() <= 255


# ─── normalize_mean_std (extra) ───────────────────────────────────────────────

class TestNormalizeMeanStdExtra:
    def test_returns_uint8(self):
        result = normalize_mean_std(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(32, 48)
        result = normalize_mean_std(img)
        assert result.shape == (32, 48)

    def test_output_in_0_255(self):
        result = normalize_mean_std(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_target_mean_applied(self):
        result = normalize_mean_std(_gray(64, 64, seed=5), target_mean=128.0, target_std=50.0)
        mean = float(result.astype(np.float32).mean())
        assert 50 <= mean <= 200

    def test_invalid_target_mean_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_mean=-1.0)

    def test_invalid_target_std_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_std=0.0)

    def test_with_mask(self):
        result = normalize_mean_std(_gray(), mask=_mask_center())
        assert result.dtype == np.uint8

    def test_all_mask_zero_returns_image(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = normalize_mean_std(_gray(), mask=mask)
        assert result.dtype == np.uint8

    def test_bgr_input(self):
        result = normalize_mean_std(_bgr())
        assert result.shape == (64, 64)

    def test_constant_image_normalized(self):
        img = _const(val=50)
        result = normalize_mean_std(img, target_mean=128.0, target_std=50.0)
        assert result.dtype == np.uint8


# ─── apply_clahe (extra) ──────────────────────────────────────────────────────

class TestApplyClaheExtra:
    def test_returns_uint8(self):
        result = apply_clahe(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved_gray(self):
        img = _gray(32, 48)
        result = apply_clahe(img)
        assert result.shape == (32, 48)

    def test_shape_preserved_bgr(self):
        img = _bgr(32, 48)
        result = apply_clahe(img)
        assert result.shape == (32, 48, 3)

    def test_output_in_0_255(self):
        result = apply_clahe(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_invalid_clip_limit_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=0.0)

    def test_negative_clip_limit_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=-1.0)

    def test_invalid_tile_grid_size_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), tile_grid_size=(0, 8))

    def test_custom_clip_limit(self):
        result = apply_clahe(_gray(), clip_limit=4.0)
        assert result.dtype == np.uint8

    def test_custom_tile_grid_size(self):
        result = apply_clahe(_gray(), tile_grid_size=(4, 4))
        assert result.shape == (64, 64)


# ─── normalize_illumination (extra) ───────────────────────────────────────────

class TestNormalizeIlluminationExtra:
    def test_returns_illum_result(self):
        r = normalize_illumination(_gray())
        assert isinstance(r, IllumResult)

    def test_method_mean_std_stored(self):
        r = normalize_illumination(_gray(), method="mean_std")
        assert r.method == "mean_std"

    def test_method_background_stored(self):
        r = normalize_illumination(_gray(), method="background")
        assert r.method == "background"

    def test_method_clahe_stored(self):
        r = normalize_illumination(_gray(), method="clahe")
        assert r.method == "clahe"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            normalize_illumination(_gray(), method="unknown")

    def test_original_mean_in_range(self):
        r = normalize_illumination(_gray())
        assert 0.0 <= r.original_mean <= 255.0

    def test_original_std_nonneg(self):
        r = normalize_illumination(_gray())
        assert r.original_std >= 0.0

    def test_output_image_uint8(self):
        r = normalize_illumination(_gray())
        assert r.image.dtype == np.uint8

    def test_output_shape_preserved(self):
        r = normalize_illumination(_gray(32, 48))
        assert r.shape[:2] == (32, 48)

    def test_none_cfg_uses_default(self):
        r = normalize_illumination(_gray(), cfg=None)
        assert isinstance(r, IllumResult)

    def test_with_mask_mean_std(self):
        r = normalize_illumination(_gray(), method="mean_std", mask=_mask_center())
        assert r.image.dtype == np.uint8

    def test_bgr_input(self):
        r = normalize_illumination(_bgr(), method="mean_std")
        assert isinstance(r, IllumResult)


# ─── batch_normalize_illumination (extra) ─────────────────────────────────────

class TestBatchNormalizeIlluminationExtra:
    def test_empty_list_returns_empty(self):
        assert batch_normalize_illumination([]) == []

    def test_single_image(self):
        results = batch_normalize_illumination([_gray()])
        assert len(results) == 1
        assert isinstance(results[0], IllumResult)

    def test_multiple_images(self):
        imgs = [_gray(seed=i) for i in range(4)]
        results = batch_normalize_illumination(imgs)
        assert len(results) == 4

    def test_all_are_illum_results(self):
        imgs = [_gray(), _bgr(), _const()]
        results = batch_normalize_illumination(imgs, method="mean_std")
        for r in results:
            assert isinstance(r, IllumResult)

    def test_method_propagated(self):
        results = batch_normalize_illumination([_gray()], method="clahe")
        assert results[0].method == "clahe"

    def test_cfg_propagated(self):
        cfg = IllumConfig(target_mean=100.0)
        results = batch_normalize_illumination([_gray()], method="mean_std", cfg=cfg)
        assert isinstance(results[0], IllumResult)
