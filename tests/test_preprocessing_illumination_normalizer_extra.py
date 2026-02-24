"""Extra tests for puzzle_reconstruction/preprocessing/illumination_normalizer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.illumination_normalizer import (
    IllumConfig,
    IllumResult,
    estimate_illumination,
    subtract_background,
    normalize_mean_std,
    apply_clahe,
    normalize_illumination,
    batch_normalize_illumination,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── IllumConfig ──────────────────────────────────────────────────────────────

class TestIllumConfigExtra:
    def test_defaults(self):
        cfg = IllumConfig()
        assert cfg.blur_ksize == 51
        assert cfg.target_mean == pytest.approx(128.0)
        assert cfg.target_std == pytest.approx(60.0)
        assert cfg.clip_limit == pytest.approx(2.0)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=50)

    def test_small_ksize_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(blur_ksize=1)

    def test_target_mean_out_of_range_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_mean=300.0)

    def test_zero_target_std_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(target_std=0.0)

    def test_zero_clip_limit_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(clip_limit=0.0)

    def test_zero_tile_raises(self):
        with pytest.raises(ValueError):
            IllumConfig(tile_grid_size=(0, 8))


# ─── IllumResult ──────────────────────────────────────────────────────────────

class TestIllumResultExtra:
    def test_valid(self):
        r = IllumResult(image=_gray(), original_mean=128.0,
                        original_std=30.0, method="mean_std")
        assert r.shape == (50, 50)

    def test_1d_image_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=np.array([1, 2, 3], dtype=np.uint8),
                        original_mean=128.0, original_std=30.0, method="x")

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_gray(), original_mean=-1.0,
                        original_std=30.0, method="x")

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_gray(), original_mean=128.0,
                        original_std=-1.0, method="x")

    def test_empty_method_raises(self):
        with pytest.raises(ValueError):
            IllumResult(image=_gray(), original_mean=128.0,
                        original_std=30.0, method="")


# ─── estimate_illumination ────────────────────────────────────────────────────

class TestEstimateIlluminationExtra:
    def test_shape(self):
        bg = estimate_illumination(_gray())
        assert bg.shape == (50, 50)

    def test_dtype_float64(self):
        bg = estimate_illumination(_gray())
        assert bg.dtype == np.float64

    def test_bgr_input(self):
        bg = estimate_illumination(_bgr())
        assert bg.ndim == 2

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            estimate_illumination(_gray(), blur_ksize=50)


# ─── subtract_background ──────────────────────────────────────────────────────

class TestSubtractBackgroundNormExtra:
    def test_dtype_uint8(self):
        out = subtract_background(_gray())
        assert out.dtype == np.uint8

    def test_shape_gray(self):
        out = subtract_background(_gray())
        assert out.ndim == 2

    def test_shape_bgr(self):
        out = subtract_background(_bgr())
        # Converts to gray internally
        assert out.ndim == 2


# ─── normalize_mean_std ───────────────────────────────────────────────────────

class TestNormalizeMeanStdExtra:
    def test_dtype_uint8(self):
        out = normalize_mean_std(_gray())
        assert out.dtype == np.uint8

    def test_with_mask(self):
        img = _gray()
        mask = np.ones((50, 50), dtype=np.uint8)
        out = normalize_mean_std(img, mask=mask)
        assert out.dtype == np.uint8

    def test_out_of_range_mean_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_mean=300.0)

    def test_zero_std_raises(self):
        with pytest.raises(ValueError):
            normalize_mean_std(_gray(), target_std=0.0)

    def test_empty_mask(self):
        img = _gray()
        mask = np.zeros((50, 50), dtype=np.uint8)
        out = normalize_mean_std(img, mask=mask)
        assert out.dtype == np.uint8


# ─── apply_clahe ──────────────────────────────────────────────────────────────

class TestApplyClaheNormExtra:
    def test_gray(self):
        out = apply_clahe(_gray())
        assert out.ndim == 2
        assert out.dtype == np.uint8

    def test_bgr(self):
        out = apply_clahe(_bgr())
        assert out.ndim == 3

    def test_zero_clip_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), clip_limit=0.0)

    def test_zero_tile_raises(self):
        with pytest.raises(ValueError):
            apply_clahe(_gray(), tile_grid_size=(0, 8))


# ─── normalize_illumination ──────────────────────────────────────────────────

class TestNormalizeIlluminationNormExtra:
    def test_mean_std(self):
        r = normalize_illumination(_gray(), method="mean_std")
        assert isinstance(r, IllumResult)
        assert r.method == "mean_std"

    def test_background(self):
        r = normalize_illumination(_gray(), method="background")
        assert r.method == "background"

    def test_clahe(self):
        r = normalize_illumination(_gray(), method="clahe")
        assert r.method == "clahe"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_illumination(_gray(), method="bad")

    def test_bgr_input(self):
        r = normalize_illumination(_bgr(), method="mean_std")
        assert isinstance(r, IllumResult)

    def test_original_stats(self):
        r = normalize_illumination(_gray(val=100), method="mean_std")
        assert r.original_mean == pytest.approx(100.0, abs=1.0)


# ─── batch_normalize_illumination ─────────────────────────────────────────────

class TestBatchNormalizeIlluminationExtra:
    def test_empty(self):
        assert batch_normalize_illumination([]) == []

    def test_length(self):
        results = batch_normalize_illumination([_gray(), _gray()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_normalize_illumination([_gray()])
        assert isinstance(results[0], IllumResult)
