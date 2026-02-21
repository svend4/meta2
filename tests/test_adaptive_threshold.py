"""Tests for puzzle_reconstruction.preprocessing.adaptive_threshold."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.adaptive_threshold import (
    ThresholdParams,
    adaptive_gaussian,
    adaptive_mean,
    apply_threshold,
    batch_threshold,
    bernsen_threshold,
    global_threshold,
    niblack_threshold,
    sauvola_threshold,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _striped(h: int = 32, w: int = 32) -> np.ndarray:
    """Image with alternating bright/dark columns."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, ::2] = 200
    return img


# ─── ThresholdParams ─────────────────────────────────────────────────────────

class TestThresholdParams:
    def test_default_method_otsu(self):
        p = ThresholdParams()
        assert p.method == "otsu"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(method="unknown_method")

    def test_block_size_below_3_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=4)

    def test_valid_block_size(self):
        p = ThresholdParams(block_size=11)
        assert p.block_size == 11

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=-1)

    def test_threshold_above_255_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=256)

    def test_all_valid_methods(self):
        for method in ("global", "otsu", "adaptive_mean", "adaptive_gaussian",
                       "niblack", "sauvola", "bernsen"):
            p = ThresholdParams(method=method)
            assert p.method == method

    def test_default_params_empty(self):
        p = ThresholdParams()
        assert p.params == {}


# ─── global_threshold ────────────────────────────────────────────────────────

class TestGlobalThreshold:
    def test_returns_uint8(self):
        result = global_threshold(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray(24, 32)
        result = global_threshold(img)
        assert result.shape == (24, 32)

    def test_values_only_0_or_255(self):
        result = global_threshold(_gray())
        assert set(np.unique(result)).issubset({0, 255})

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            global_threshold(_gray(), threshold=300)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            global_threshold(_gray(), threshold=-1)

    def test_otsu_returns_binary(self):
        result = global_threshold(_gray(), use_otsu=True)
        assert set(np.unique(result)).issubset({0, 255})

    def test_color_image_accepted(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        result = global_threshold(img, threshold=100)
        assert result.shape == (32, 32)

    def test_all_black_image(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        result = global_threshold(img, threshold=10)
        assert np.all(result == 0)

    def test_all_white_image(self):
        img = np.full((16, 16), 255, dtype=np.uint8)
        result = global_threshold(img, threshold=10)
        assert np.all(result == 255)


# ─── adaptive_mean ───────────────────────────────────────────────────────────

class TestAdaptiveMean:
    def test_returns_uint8(self):
        result = adaptive_mean(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = adaptive_mean(_gray(24, 32))
        assert result.shape == (24, 32)

    def test_values_only_0_or_255(self):
        result = adaptive_mean(_gray())
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_even_raises(self):
        with pytest.raises(ValueError):
            adaptive_mean(_gray(), block_size=4)

    def test_block_size_below_3_raises(self):
        with pytest.raises(ValueError):
            adaptive_mean(_gray(), block_size=1)

    def test_striped_image_produces_nonzero(self):
        result = adaptive_mean(_striped())
        assert np.any(result > 0)

    def test_color_image_accepted(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        result = adaptive_mean(img)
        assert result.shape == (32, 32)


# ─── adaptive_gaussian ───────────────────────────────────────────────────────

class TestAdaptiveGaussian:
    def test_returns_uint8(self):
        result = adaptive_gaussian(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = adaptive_gaussian(_gray(24, 32))
        assert result.shape == (24, 32)

    def test_values_only_0_or_255(self):
        result = adaptive_gaussian(_gray())
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_even_raises(self):
        with pytest.raises(ValueError):
            adaptive_gaussian(_gray(), block_size=6)

    def test_block_size_below_3_raises(self):
        with pytest.raises(ValueError):
            adaptive_gaussian(_gray(), block_size=2)

    def test_striped_image_produces_nonzero(self):
        result = adaptive_gaussian(_striped())
        assert np.any(result > 0)


# ─── niblack_threshold ───────────────────────────────────────────────────────

class TestNiblackThreshold:
    def test_returns_uint8(self):
        result = niblack_threshold(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = niblack_threshold(_gray(16, 16))
        assert result.shape == (16, 16)

    def test_values_only_0_or_255(self):
        result = niblack_threshold(_gray(8, 8))
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_even_raises(self):
        with pytest.raises(ValueError):
            niblack_threshold(_gray(8, 8), block_size=4)

    def test_block_size_below_3_raises(self):
        with pytest.raises(ValueError):
            niblack_threshold(_gray(8, 8), block_size=2)

    def test_constant_image_all_zero(self):
        img = np.full((8, 8), 100, dtype=np.uint8)
        result = niblack_threshold(img, k=-0.2)
        # σ=0 → T=µ, so pixels equal to µ are NOT > T → 0
        assert np.all(result == 0)


# ─── sauvola_threshold ───────────────────────────────────────────────────────

class TestSauvolaThreshold:
    def test_returns_uint8(self):
        result = sauvola_threshold(_gray(8, 8))
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = sauvola_threshold(_gray(8, 8))
        assert result.shape == (8, 8)

    def test_values_only_0_or_255(self):
        result = sauvola_threshold(_gray(8, 8))
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_even_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_gray(8, 8), block_size=4)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_gray(8, 8), r=0.0)

    def test_r_negative_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_gray(8, 8), r=-1.0)


# ─── bernsen_threshold ───────────────────────────────────────────────────────

class TestBernsenThreshold:
    def test_returns_uint8(self):
        result = bernsen_threshold(_gray(8, 8))
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        result = bernsen_threshold(_gray(8, 8))
        assert result.shape == (8, 8)

    def test_values_only_0_or_255(self):
        result = bernsen_threshold(_gray(8, 8))
        assert set(np.unique(result)).issubset({0, 255})

    def test_block_size_even_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_gray(8, 8), block_size=4)

    def test_negative_contrast_threshold_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_gray(8, 8), contrast_threshold=-1.0)

    def test_low_contrast_image_all_zero(self):
        # Constant image → contrast = 0 < default threshold → all background
        img = np.full((8, 8), 128, dtype=np.uint8)
        result = bernsen_threshold(img, contrast_threshold=10.0)
        assert np.all(result == 0)


# ─── apply_threshold ─────────────────────────────────────────────────────────

class TestApplyThreshold:
    def test_otsu_method(self):
        p = ThresholdParams(method="otsu")
        result = apply_threshold(_gray(), p)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32)

    def test_global_method(self):
        p = ThresholdParams(method="global", threshold=128)
        result = apply_threshold(_gray(), p)
        assert set(np.unique(result)).issubset({0, 255})

    def test_adaptive_mean_method(self):
        p = ThresholdParams(method="adaptive_mean", block_size=11)
        result = apply_threshold(_gray(), p)
        assert result.shape == (32, 32)

    def test_niblack_method(self):
        p = ThresholdParams(method="niblack", block_size=7)
        result = apply_threshold(_gray(8, 8), p)
        assert result.shape == (8, 8)

    def test_sauvola_method(self):
        p = ThresholdParams(method="sauvola", block_size=7)
        result = apply_threshold(_gray(8, 8), p)
        assert result.shape == (8, 8)

    def test_bernsen_method(self):
        p = ThresholdParams(method="bernsen", block_size=7)
        result = apply_threshold(_gray(8, 8), p)
        assert result.shape == (8, 8)


# ─── batch_threshold ─────────────────────────────────────────────────────────

class TestBatchThreshold:
    def test_returns_list(self):
        p = ThresholdParams(method="otsu")
        result = batch_threshold([_gray()], p)
        assert isinstance(result, list)

    def test_length_matches(self):
        p = ThresholdParams(method="otsu")
        result = batch_threshold([_gray(), _gray(seed=1)], p)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        p = ThresholdParams(method="otsu")
        assert batch_threshold([], p) == []

    def test_all_results_uint8(self):
        p = ThresholdParams(method="adaptive_gaussian")
        result = batch_threshold([_gray(), _gray(seed=2)], p)
        assert all(r.dtype == np.uint8 for r in result)
