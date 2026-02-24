"""Extra tests for puzzle_reconstruction/preprocessing/adaptive_threshold.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.adaptive_threshold import (
    ThresholdParams,
    global_threshold,
    adaptive_mean,
    adaptive_gaussian,
    niblack_threshold,
    sauvola_threshold,
    bernsen_threshold,
    apply_threshold,
    batch_threshold,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=32) -> np.ndarray:
    """Horizontal ramp 0…255."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32, val=100) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _small_gray(h=8, w=8, val=128) -> np.ndarray:
    """Small image for slow pixel-loop tests."""
    return np.full((h, w), val, dtype=np.uint8)


# ─── ThresholdParams ──────────────────────────────────────────────────────────

class TestThresholdParamsExtra:
    def test_default_method(self):
        assert ThresholdParams().method == "otsu"

    def test_default_block_size(self):
        assert ThresholdParams().block_size == 11

    def test_default_k(self):
        assert ThresholdParams().k == pytest.approx(0.2)

    def test_default_threshold(self):
        assert ThresholdParams().threshold == 128

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(method="unknown")

    def test_block_size_less_than_3_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=4)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=-1)

    def test_threshold_gt_255_raises(self):
        with pytest.raises(ValueError):
            ThresholdParams(threshold=256)

    def test_valid_methods(self):
        for m in ("global", "otsu", "adaptive_mean", "adaptive_gaussian",
                  "niblack", "sauvola", "bernsen"):
            p = ThresholdParams(method=m)
            assert p.method == m

    def test_odd_block_size_ok(self):
        p = ThresholdParams(block_size=15)
        assert p.block_size == 15

    def test_k_negative_ok(self):
        p = ThresholdParams(k=-0.2)
        assert p.k == pytest.approx(-0.2)


# ─── global_threshold ─────────────────────────────────────────────────────────

class TestGlobalThresholdExtra:
    def test_returns_uint8(self):
        result = global_threshold(_gray())
        assert result.dtype == np.uint8

    def test_output_shape_gray(self):
        img = _gray(20, 30)
        result = global_threshold(img)
        assert result.shape == (20, 30)

    def test_output_shape_bgr(self):
        img = _bgr(20, 30)
        result = global_threshold(img)
        assert result.shape == (20, 30)

    def test_binary_values_only(self):
        result = global_threshold(_ramp())
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            global_threshold(_gray(), threshold=300)

    def test_otsu_flag(self):
        result = global_threshold(_ramp(), use_otsu=True)
        assert result.dtype == np.uint8

    def test_low_threshold_all_white(self):
        result = global_threshold(_gray(val=200), threshold=50)
        assert np.all(result == 255)

    def test_high_threshold_all_black(self):
        result = global_threshold(_gray(val=50), threshold=200)
        assert np.all(result == 0)


# ─── adaptive_mean ────────────────────────────────────────────────────────────

class TestAdaptiveMeanExtra:
    def test_returns_uint8(self):
        assert adaptive_mean(_gray()).dtype == np.uint8

    def test_output_shape(self):
        img = _gray(24, 32)
        result = adaptive_mean(img)
        assert result.shape == (24, 32)

    def test_binary_values_only(self):
        unique = np.unique(adaptive_mean(_ramp()))
        assert set(unique).issubset({0, 255})

    def test_block_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            adaptive_mean(_gray(), block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            adaptive_mean(_gray(), block_size=10)

    def test_custom_block_size(self):
        result = adaptive_mean(_gray(), block_size=15)
        assert result.shape == _gray().shape

    def test_bgr_input(self):
        result = adaptive_mean(_bgr())
        assert result.shape == (32, 32)


# ─── adaptive_gaussian ────────────────────────────────────────────────────────

class TestAdaptiveGaussianExtra:
    def test_returns_uint8(self):
        assert adaptive_gaussian(_gray()).dtype == np.uint8

    def test_output_shape(self):
        img = _gray(16, 24)
        result = adaptive_gaussian(img)
        assert result.shape == (16, 24)

    def test_binary_values_only(self):
        unique = np.unique(adaptive_gaussian(_ramp()))
        assert set(unique).issubset({0, 255})

    def test_block_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            adaptive_gaussian(_gray(), block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            adaptive_gaussian(_gray(), block_size=8)

    def test_constant_image_output(self):
        result = adaptive_gaussian(_gray(val=128))
        assert result.dtype == np.uint8

    def test_bgr_input(self):
        result = adaptive_gaussian(_bgr())
        assert result.shape == (32, 32)


# ─── niblack_threshold ────────────────────────────────────────────────────────

class TestNiblackThresholdExtra:
    def test_returns_uint8(self):
        result = niblack_threshold(_small_gray())
        assert result.dtype == np.uint8

    def test_output_shape(self):
        img = _small_gray(6, 8)
        result = niblack_threshold(img, block_size=3)
        assert result.shape == (6, 8)

    def test_binary_values_only(self):
        img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        result = niblack_threshold(img, block_size=3)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_block_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            niblack_threshold(_small_gray(), block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            niblack_threshold(_small_gray(), block_size=4)

    def test_uniform_image_all_zero(self):
        img = _small_gray(val=128)
        # k=-0.2: T = 128 + (-0.2)*0 = 128; pixel 128 is not > 128 => 0
        result = niblack_threshold(img, block_size=3, k=-0.2)
        assert np.all(result == 0)

    def test_bgr_input(self):
        result = niblack_threshold(_bgr(8, 8, 100), block_size=3)
        assert result.shape == (8, 8)


# ─── sauvola_threshold ────────────────────────────────────────────────────────

class TestSauvolaThresholdExtra:
    def test_returns_uint8(self):
        result = sauvola_threshold(_small_gray(), block_size=3)
        assert result.dtype == np.uint8

    def test_output_shape(self):
        img = _small_gray(6, 8)
        result = sauvola_threshold(img, block_size=3)
        assert result.shape == (6, 8)

    def test_binary_values_only(self):
        img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        result = sauvola_threshold(img, block_size=3)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_block_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_small_gray(), block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_small_gray(), block_size=4)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_small_gray(), block_size=3, r=0.0)

    def test_r_negative_raises(self):
        with pytest.raises(ValueError):
            sauvola_threshold(_small_gray(), block_size=3, r=-1.0)

    def test_bgr_input(self):
        result = sauvola_threshold(_bgr(8, 8, 100), block_size=3)
        assert result.shape == (8, 8)


# ─── bernsen_threshold ────────────────────────────────────────────────────────

class TestBernsenThresholdExtra:
    def test_returns_uint8(self):
        result = bernsen_threshold(_small_gray(), block_size=3)
        assert result.dtype == np.uint8

    def test_output_shape(self):
        img = _small_gray(6, 8)
        result = bernsen_threshold(img, block_size=3)
        assert result.shape == (6, 8)

    def test_binary_values_only(self):
        img = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        result = bernsen_threshold(img, block_size=3)
        unique = np.unique(result)
        assert set(unique).issubset({0, 255})

    def test_block_size_lt_3_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_small_gray(), block_size=2)

    def test_even_block_size_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_small_gray(), block_size=4)

    def test_negative_contrast_threshold_raises(self):
        with pytest.raises(ValueError):
            bernsen_threshold(_small_gray(), block_size=3, contrast_threshold=-1.0)

    def test_uniform_image_all_zero(self):
        img = _small_gray(val=100)
        # contrast=0 < default 15 => all background => 0
        result = bernsen_threshold(img, block_size=3, contrast_threshold=15.0)
        assert np.all(result == 0)

    def test_bgr_input(self):
        result = bernsen_threshold(_bgr(8, 8, 100), block_size=3)
        assert result.shape == (8, 8)


# ─── apply_threshold ──────────────────────────────────────────────────────────

class TestApplyThresholdExtra:
    def _apply(self, method, **kwargs):
        params = ThresholdParams(method=method, **kwargs)
        img = _gray() if method not in ("niblack", "sauvola", "bernsen") \
              else _small_gray()
        return apply_threshold(img, params), img.shape

    def test_otsu_returns_uint8(self):
        result, shape = self._apply("otsu")
        assert result.dtype == np.uint8
        assert result.shape == shape

    def test_global_returns_correct_shape(self):
        result, shape = self._apply("global", threshold=100)
        assert result.shape == shape

    def test_adaptive_mean_applied(self):
        result, shape = self._apply("adaptive_mean", block_size=11)
        assert result.shape == shape

    def test_adaptive_gaussian_applied(self):
        result, shape = self._apply("adaptive_gaussian", block_size=11)
        assert result.shape == shape

    def test_niblack_applied(self):
        result, shape = self._apply("niblack", block_size=3)
        assert result.shape == shape

    def test_sauvola_applied(self):
        result, shape = self._apply("sauvola", block_size=3)
        assert result.shape == shape

    def test_bernsen_applied(self):
        result, shape = self._apply("bernsen", block_size=3)
        assert result.shape == shape

    def test_all_methods_return_binary(self):
        for m in ("global", "otsu", "adaptive_mean", "adaptive_gaussian"):
            params = ThresholdParams(method=m)
            img = _ramp()
            result = apply_threshold(img, params)
            assert set(np.unique(result)).issubset({0, 255})


# ─── batch_threshold ──────────────────────────────────────────────────────────

class TestBatchThresholdExtra:
    def test_returns_list(self):
        params = ThresholdParams(method="otsu")
        result = batch_threshold([_gray()], params)
        assert isinstance(result, list)

    def test_length_matches(self):
        params = ThresholdParams(method="otsu")
        imgs = [_gray(), _gray(16, 16)]
        result = batch_threshold(imgs, params)
        assert len(result) == 2

    def test_empty_list(self):
        params = ThresholdParams(method="otsu")
        assert batch_threshold([], params) == []

    def test_each_element_uint8(self):
        params = ThresholdParams(method="global", threshold=128)
        for img in batch_threshold([_gray(), _ramp()], params):
            assert img.dtype == np.uint8

    def test_shape_preserved(self):
        params = ThresholdParams(method="otsu")
        imgs = [_gray(10, 20), _gray(15, 25)]
        results = batch_threshold(imgs, params)
        for orig, res in zip(imgs, results):
            assert res.shape == orig.shape[:2]
