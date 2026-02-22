"""Extra tests for puzzle_reconstruction.preprocessing.adaptive_threshold."""
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


def _gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _striped(h=32, w=32):
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, ::2] = 200
    return img


# ─── TestThresholdParamsExtra ─────────────────────────────────────────────────

class TestThresholdParamsExtra:
    def test_default_otsu(self):
        assert ThresholdParams().method == "otsu"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            ThresholdParams(method="bad_method")

    def test_block_below_3(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=2)

    def test_even_block(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=4)

    def test_valid_block(self):
        assert ThresholdParams(block_size=11).block_size == 11

    def test_threshold_below_0(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=-1)

    def test_threshold_above_255(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=256)

    @pytest.mark.parametrize("method", [
        "global", "otsu", "adaptive_mean", "adaptive_gaussian",
        "niblack", "sauvola", "bernsen",
    ])
    def test_all_methods(self, method):
        assert ThresholdParams(method=method).method == method

    def test_params_default_empty(self):
        assert ThresholdParams().params == {}

    def test_threshold_0_ok(self):
        assert ThresholdParams(threshold=0).threshold == 0

    def test_threshold_255_ok(self):
        assert ThresholdParams(threshold=255).threshold == 255


# ─── TestGlobalThresholdExtra ─────────────────────────────────────────────────

class TestGlobalThresholdExtra:
    def test_dtype(self):
        assert global_threshold(_gray()).dtype == np.uint8

    def test_shape(self):
        assert global_threshold(_gray(24, 32)).shape == (24, 32)

    def test_binary(self):
        assert set(np.unique(global_threshold(_gray()))).issubset({0, 255})

    def test_invalid_high(self):
        with pytest.raises(ValueError):
            global_threshold(_gray(), threshold=300)

    def test_invalid_neg(self):
        with pytest.raises(ValueError):
            global_threshold(_gray(), threshold=-1)

    def test_otsu(self):
        assert set(np.unique(global_threshold(_gray(), use_otsu=True))).issubset({0, 255})

    def test_color(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        assert global_threshold(img, threshold=100).shape == (32, 32)

    def test_all_black(self):
        assert np.all(global_threshold(np.zeros((16, 16), dtype=np.uint8), threshold=10) == 0)

    def test_all_white(self):
        assert np.all(global_threshold(np.full((16, 16), 255, dtype=np.uint8), threshold=10) == 255)


# ─── TestAdaptiveMeanExtra ────────────────────────────────────────────────────

class TestAdaptiveMeanExtra:
    def test_dtype(self):
        assert adaptive_mean(_gray()).dtype == np.uint8

    def test_shape(self):
        assert adaptive_mean(_gray(24, 32)).shape == (24, 32)

    def test_binary(self):
        assert set(np.unique(adaptive_mean(_gray()))).issubset({0, 255})

    def test_even_block_raises(self):
        with pytest.raises(ValueError):
            adaptive_mean(_gray(), block_size=4)

    def test_block_1_raises(self):
        with pytest.raises(ValueError):
            adaptive_mean(_gray(), block_size=1)

    def test_striped(self):
        assert np.any(adaptive_mean(_striped()) > 0)

    def test_color(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        assert adaptive_mean(img).shape == (32, 32)


# ─── TestAdaptiveGaussianExtra ────────────────────────────────────────────────

class TestAdaptiveGaussianExtra:
    def test_dtype(self):
        assert adaptive_gaussian(_gray()).dtype == np.uint8

    def test_shape(self):
        assert adaptive_gaussian(_gray(24, 32)).shape == (24, 32)

    def test_binary(self):
        assert set(np.unique(adaptive_gaussian(_gray()))).issubset({0, 255})

    def test_even_block_raises(self):
        with pytest.raises(ValueError):
            adaptive_gaussian(_gray(), block_size=6)

    def test_block_2_raises(self):
        with pytest.raises(ValueError):
            adaptive_gaussian(_gray(), block_size=2)

    def test_striped(self):
        assert np.any(adaptive_gaussian(_striped()) > 0)


# ─── TestNiblackThresholdExtra ────────────────────────────────────────────────

class TestNiblackThresholdExtra:
    def test_dtype(self):
        assert niblack_threshold(_gray()).dtype == np.uint8

    def test_shape(self):
        assert niblack_threshold(_gray(16, 16)).shape == (16, 16)

    def test_binary(self):
        assert set(np.unique(niblack_threshold(_gray(8, 8)))).issubset({0, 255})

    def test_even_block_raises(self):
        with pytest.raises(ValueError):
            niblack_threshold(_gray(8, 8), block_size=4)

    def test_block_2_raises(self):
        with pytest.raises(ValueError):
            niblack_threshold(_gray(8, 8), block_size=2)

    def test_constant_zero(self):
        img = np.full((8, 8), 100, dtype=np.uint8)
        assert np.all(niblack_threshold(img, k=-0.2) == 0)


# ─── TestSauvolaThresholdExtra ────────────────────────────────────────────────

class TestSauvolaThresholdExtra:
    def test_dtype(self):
        assert sauvola_threshold(_gray(8, 8)).dtype == np.uint8

    def test_shape(self):
        assert sauvola_threshold(_gray(8, 8)).shape == (8, 8)

    def test_binary(self):
        assert set(np.unique(sauvola_threshold(_gray(8, 8)))).issubset({0, 255})

    def test_even_block_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_gray(8, 8), block_size=4)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_gray(8, 8), r=0.0)

    def test_r_neg_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_gray(8, 8), r=-1.0)


# ─── TestBernsenThresholdExtra ────────────────────────────────────────────────

class TestBernsenThresholdExtra:
    def test_dtype(self):
        assert bernsen_threshold(_gray(8, 8)).dtype == np.uint8

    def test_shape(self):
        assert bernsen_threshold(_gray(8, 8)).shape == (8, 8)

    def test_binary(self):
        assert set(np.unique(bernsen_threshold(_gray(8, 8)))).issubset({0, 255})

    def test_even_block_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_gray(8, 8), block_size=4)

    def test_neg_contrast_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_gray(8, 8), contrast_threshold=-1.0)

    def test_constant_all_zero(self):
        img = np.full((8, 8), 128, dtype=np.uint8)
        assert np.all(bernsen_threshold(img, contrast_threshold=10.0) == 0)


# ─── TestApplyThresholdExtra ──────────────────────────────────────────────────

class TestApplyThresholdExtra:
    @pytest.mark.parametrize("method", [
        "otsu", "global", "adaptive_mean", "adaptive_gaussian",
    ])
    def test_methods_return_binary(self, method):
        p = ThresholdParams(method=method, threshold=128, block_size=11)
        result = apply_threshold(_gray(), p)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_niblack_method(self):
        p = ThresholdParams(method="niblack", block_size=7)
        assert apply_threshold(_gray(8, 8), p).shape == (8, 8)

    def test_sauvola_method(self):
        p = ThresholdParams(method="sauvola", block_size=7)
        assert apply_threshold(_gray(8, 8), p).shape == (8, 8)

    def test_bernsen_method(self):
        p = ThresholdParams(method="bernsen", block_size=7)
        assert apply_threshold(_gray(8, 8), p).shape == (8, 8)


# ─── TestBatchThresholdExtra ─────────────────────────────────────────────────

class TestBatchThresholdExtra:
    def test_returns_list(self):
        p = ThresholdParams(method="otsu")
        assert isinstance(batch_threshold([_gray()], p), list)

    def test_length(self):
        p = ThresholdParams(method="otsu")
        assert len(batch_threshold([_gray(), _gray(seed=1)], p)) == 2

    def test_empty(self):
        assert batch_threshold([], ThresholdParams(method="otsu")) == []

    def test_all_uint8(self):
        p = ThresholdParams(method="adaptive_gaussian")
        assert all(r.dtype == np.uint8 for r in batch_threshold([_gray(), _gray(seed=2)], p))
