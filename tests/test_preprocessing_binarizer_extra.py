"""Extra tests for puzzle_reconstruction/preprocessing/binarizer.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.binarizer import (
    BinarizeResult,
    binarize_otsu,
    binarize_adaptive,
    binarize_sauvola,
    binarize_niblack,
    binarize_bernsen,
    auto_binarize,
    batch_binarize,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32, val=100) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _binary_values_ok(result: BinarizeResult) -> bool:
    return set(np.unique(result.binary)).issubset({0, 255})


# ─── BinarizeResult ───────────────────────────────────────────────────────────

class TestBinarizeResultExtra:
    def _make(self, val=128, inverted=False) -> BinarizeResult:
        binary = np.full((8, 8), val, dtype=np.uint8)
        return BinarizeResult(binary=binary, method="otsu",
                              threshold=0.0, inverted=inverted)

    def test_method_stored(self):
        r = self._make()
        assert r.method == "otsu"

    def test_threshold_stored(self):
        r = BinarizeResult(binary=np.zeros((4, 4), dtype=np.uint8),
                           method="global", threshold=128.0)
        assert r.threshold == pytest.approx(128.0)

    def test_inverted_stored(self):
        r = self._make(inverted=True)
        assert r.inverted is True

    def test_foreground_ratio_all_white(self):
        r = self._make(val=255)
        assert r.foreground_ratio == pytest.approx(1.0)

    def test_foreground_ratio_all_black(self):
        r = self._make(val=0)
        assert r.foreground_ratio == pytest.approx(0.0)

    def test_foreground_ratio_half(self):
        b = np.zeros((4, 4), dtype=np.uint8)
        b[:, :2] = 255
        r = BinarizeResult(binary=b, method="otsu", threshold=0.0)
        assert r.foreground_ratio == pytest.approx(0.5)

    def test_repr_contains_method(self):
        r = self._make()
        assert "otsu" in repr(r)

    def test_default_params_empty(self):
        r = self._make()
        assert r.params == {}


# ─── binarize_otsu ────────────────────────────────────────────────────────────

class TestBinarizeOtsuExtra:
    def test_returns_binarize_result(self):
        assert isinstance(binarize_otsu(_ramp()), BinarizeResult)

    def test_method_name(self):
        assert binarize_otsu(_gray()).method == "otsu"

    def test_output_shape_gray(self):
        img = _gray(20, 30)
        r = binarize_otsu(img)
        assert r.binary.shape == (20, 30)

    def test_output_shape_bgr(self):
        img = _bgr(20, 30)
        r = binarize_otsu(img)
        assert r.binary.shape == (20, 30)

    def test_binary_values_only(self):
        assert _binary_values_ok(binarize_otsu(_ramp()))

    def test_dtype_uint8(self):
        assert binarize_otsu(_ramp()).binary.dtype == np.uint8

    def test_inverted_flag_stored(self):
        r = binarize_otsu(_ramp(), invert=True)
        assert r.inverted is True

    def test_invert_flips_values(self):
        r1 = binarize_otsu(_ramp(), invert=False)
        r2 = binarize_otsu(_ramp(), invert=True)
        assert not np.array_equal(r1.binary, r2.binary)


# ─── binarize_adaptive ────────────────────────────────────────────────────────

class TestBinarizeAdaptiveExtra:
    def test_returns_binarize_result(self):
        assert isinstance(binarize_adaptive(_gray()), BinarizeResult)

    def test_gaussian_method_name(self):
        r = binarize_adaptive(_gray(), adaptive_method="gaussian")
        assert "gaussian" in r.method

    def test_mean_method_name(self):
        r = binarize_adaptive(_gray(), adaptive_method="mean")
        assert "mean" in r.method

    def test_output_shape(self):
        img = _gray(24, 40)
        r = binarize_adaptive(img)
        assert r.binary.shape == (24, 40)

    def test_binary_values_only(self):
        assert _binary_values_ok(binarize_adaptive(_ramp()))

    def test_threshold_zero_for_adaptive(self):
        r = binarize_adaptive(_gray())
        assert r.threshold == pytest.approx(0.0)

    def test_params_stored(self):
        r = binarize_adaptive(_gray(), block_size=11, c=2.0)
        assert "block_size" in r.params
        assert "c" in r.params

    def test_even_block_size_corrected(self):
        # block_size is forced odd via max(3, int(bs) | 1)
        r = binarize_adaptive(_gray(), block_size=10)
        assert r.params["block_size"] % 2 == 1

    def test_inverted_flag(self):
        r = binarize_adaptive(_gray(), invert=True)
        assert r.inverted is True


# ─── binarize_sauvola ─────────────────────────────────────────────────────────

class TestBinarizeSauvolaExtra:
    def test_returns_binarize_result(self):
        assert isinstance(binarize_sauvola(_gray()), BinarizeResult)

    def test_method_name(self):
        assert binarize_sauvola(_gray()).method == "sauvola"

    def test_output_shape(self):
        img = _gray(20, 24)
        r = binarize_sauvola(img)
        assert r.binary.shape == (20, 24)

    def test_binary_values_only(self):
        assert _binary_values_ok(binarize_sauvola(_ramp()))

    def test_dtype_uint8(self):
        assert binarize_sauvola(_gray()).binary.dtype == np.uint8

    def test_params_stored(self):
        r = binarize_sauvola(_gray(), window_size=15, k=0.2, r=128.0)
        assert "window_size" in r.params
        assert "k" in r.params
        assert "r" in r.params

    def test_inverted_flag(self):
        r = binarize_sauvola(_gray(), invert=True)
        assert r.inverted is True

    def test_bgr_input(self):
        r = binarize_sauvola(_bgr())
        assert r.binary.shape == (32, 32)


# ─── binarize_niblack ─────────────────────────────────────────────────────────

class TestBinarizeNiblackExtra:
    def test_returns_binarize_result(self):
        assert isinstance(binarize_niblack(_gray()), BinarizeResult)

    def test_method_name(self):
        assert binarize_niblack(_gray()).method == "niblack"

    def test_output_shape(self):
        img = _gray(20, 24)
        r = binarize_niblack(img)
        assert r.binary.shape == (20, 24)

    def test_binary_values_only(self):
        assert _binary_values_ok(binarize_niblack(_ramp()))

    def test_dtype_uint8(self):
        assert binarize_niblack(_gray()).binary.dtype == np.uint8

    def test_params_stored(self):
        r = binarize_niblack(_gray(), window_size=15, k=-0.2)
        assert "window_size" in r.params
        assert "k" in r.params

    def test_inverted_flag(self):
        r = binarize_niblack(_gray(), invert=True)
        assert r.inverted is True

    def test_bgr_input(self):
        r = binarize_niblack(_bgr())
        assert r.binary.shape == (32, 32)


# ─── binarize_bernsen ─────────────────────────────────────────────────────────

class TestBinarizeBernsenExtra:
    def test_returns_binarize_result(self):
        assert isinstance(binarize_bernsen(_gray()), BinarizeResult)

    def test_method_name(self):
        assert binarize_bernsen(_gray()).method == "bernsen"

    def test_output_shape(self):
        img = _gray(20, 24)
        r = binarize_bernsen(img)
        assert r.binary.shape == (20, 24)

    def test_binary_values_only(self):
        assert _binary_values_ok(binarize_bernsen(_ramp()))

    def test_dtype_uint8(self):
        assert binarize_bernsen(_gray()).binary.dtype == np.uint8

    def test_params_stored(self):
        r = binarize_bernsen(_gray(), window_size=15, contrast_thresh=15.0)
        assert "window_size" in r.params
        assert "contrast_thresh" in r.params

    def test_uniform_image_all_zero(self):
        img = _gray(val=128)
        r = binarize_bernsen(img, contrast_thresh=50.0)
        # Uniform → contrast 0 < 50 → all background → 0
        assert np.all(r.binary == 0)

    def test_inverted_flag(self):
        r = binarize_bernsen(_gray(), invert=True)
        assert r.inverted is True


# ─── auto_binarize ────────────────────────────────────────────────────────────

class TestAutoBinarizeExtra:
    def test_returns_binarize_result(self):
        assert isinstance(auto_binarize(_ramp()), BinarizeResult)

    def test_binary_values_only(self):
        assert _binary_values_ok(auto_binarize(_ramp()))

    def test_output_shape(self):
        img = _gray(20, 24)
        r = auto_binarize(img)
        assert r.binary.shape == (20, 24)

    def test_dtype_uint8(self):
        assert auto_binarize(_ramp()).binary.dtype == np.uint8

    def test_high_entropy_uses_otsu(self):
        # Noise image has entropy ~7.8 > 6.5 → otsu
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        r = auto_binarize(noise)
        assert r.method == "otsu"

    def test_low_entropy_uses_sauvola(self):
        # Uniform image has very low entropy → sauvola
        r = auto_binarize(_gray(val=100))
        assert r.method == "sauvola"

    def test_inverted_passed_through(self):
        r = auto_binarize(_ramp(), invert=True)
        assert r.inverted is True


# ─── batch_binarize ───────────────────────────────────────────────────────────

class TestBatchBinarizeExtra:
    def test_returns_list(self):
        result = batch_binarize([_gray()], method="otsu")
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_gray(), _gray(16, 16)]
        assert len(batch_binarize(imgs, method="otsu")) == 2

    def test_empty_list(self):
        assert batch_binarize([], method="otsu") == []

    def test_each_element_is_result(self):
        for r in batch_binarize([_gray(), _ramp()], method="otsu"):
            assert isinstance(r, BinarizeResult)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_binarize([_gray()], method="unknown_method")

    def test_adaptive_method(self):
        results = batch_binarize([_gray()], method="adaptive")
        assert len(results) == 1

    def test_sauvola_method(self):
        results = batch_binarize([_gray()], method="sauvola")
        assert results[0].method == "sauvola"

    def test_niblack_method(self):
        results = batch_binarize([_gray()], method="niblack")
        assert results[0].method == "niblack"

    def test_bernsen_method(self):
        results = batch_binarize([_gray()], method="bernsen")
        assert results[0].method == "bernsen"

    def test_auto_method(self):
        results = batch_binarize([_ramp()], method="auto")
        assert isinstance(results[0], BinarizeResult)
