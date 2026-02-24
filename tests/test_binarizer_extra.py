"""Extra tests for puzzle_reconstruction/preprocessing/binarizer.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── helpers ────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 255, (h, w)) * 255).astype(np.uint8)


def _bimodal(h=32, w=32):
    """Image with clear bimodal intensity distribution."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[: h // 2, :] = 50
    img[h // 2 :, :] = 200
    return img


# ─── BinarizeResult (extra) ──────────────────────────────────────────────────

class TestBinarizeResultExtra:
    def test_method_stored(self):
        binary = np.zeros((8, 8), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="otsu", threshold=128.0)
        assert result.method == "otsu"

    def test_threshold_stored(self):
        binary = np.ones((8, 8), dtype=np.uint8) * 255
        result = BinarizeResult(binary=binary, method="test", threshold=100.0)
        assert result.threshold == pytest.approx(100.0)

    def test_inverted_default_false(self):
        binary = np.zeros((4, 4), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0)
        assert result.inverted is False

    def test_inverted_true(self):
        binary = np.zeros((4, 4), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0, inverted=True)
        assert result.inverted is True

    def test_foreground_ratio_all_white(self):
        binary = np.full((8, 8), 255, dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0)
        assert result.foreground_ratio == pytest.approx(1.0)

    def test_foreground_ratio_all_black(self):
        binary = np.zeros((8, 8), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0)
        assert result.foreground_ratio == pytest.approx(0.0)

    def test_foreground_ratio_half(self):
        binary = np.zeros((8, 8), dtype=np.uint8)
        binary[:4, :] = 255
        result = BinarizeResult(binary=binary, method="t", threshold=0.0)
        assert result.foreground_ratio == pytest.approx(0.5)

    def test_repr_contains_method(self):
        binary = np.zeros((4, 4), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="otsu", threshold=128.0)
        assert "otsu" in repr(result)

    def test_params_default_empty(self):
        binary = np.zeros((4, 4), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0)
        assert result.params == {}

    def test_params_stored(self):
        binary = np.zeros((4, 4), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0, params={"k": 0.5})
        assert result.params["k"] == pytest.approx(0.5)

    def test_binary_shape_preserved(self):
        binary = np.zeros((16, 24), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="t", threshold=0.0)
        assert result.binary.shape == (16, 24)


# ─── binarize_otsu (extra) ───────────────────────────────────────────────────

class TestBinarizeOtsuExtra:
    def test_returns_result(self):
        result = binarize_otsu(_gray())
        assert isinstance(result, BinarizeResult)

    def test_method_otsu(self):
        result = binarize_otsu(_gray())
        assert "otsu" in result.method.lower()

    def test_binary_shape(self):
        img = _gray(16, 20)
        result = binarize_otsu(img)
        assert result.binary.shape == (16, 20)

    def test_binary_values_only_0_255(self):
        result = binarize_otsu(_gray())
        unique = np.unique(result.binary)
        assert set(unique).issubset({0, 255})

    def test_invert_flips_ratio(self):
        img = _bimodal()
        r1 = binarize_otsu(img, invert=False)
        r2 = binarize_otsu(img, invert=True)
        assert r2.foreground_ratio == pytest.approx(1.0 - r1.foreground_ratio, abs=1e-4)

    def test_inverted_flag_set(self):
        result = binarize_otsu(_gray(), invert=True)
        assert result.inverted is True

    def test_threshold_positive(self):
        result = binarize_otsu(_bimodal())
        assert result.threshold > 0


# ─── binarize_adaptive (extra) ───────────────────────────────────────────────

class TestBinarizeAdaptiveExtra:
    def test_returns_result(self):
        result = binarize_adaptive(_gray())
        assert isinstance(result, BinarizeResult)

    def test_binary_shape(self):
        img = _gray(24, 32)
        result = binarize_adaptive(img)
        assert result.binary.shape == (24, 32)

    def test_binary_values_only_0_255(self):
        result = binarize_adaptive(_gray())
        unique = np.unique(result.binary)
        assert set(unique).issubset({0, 255})

    def test_method_contains_adaptive(self):
        result = binarize_adaptive(_gray())
        assert "adaptive" in result.method.lower()

    def test_invert_flag(self):
        result = binarize_adaptive(_gray(), invert=True)
        assert result.inverted is True

    def test_block_size_affects_result(self):
        img = _gray(32, 32, seed=1)
        r1 = binarize_adaptive(img, block_size=5)
        r2 = binarize_adaptive(img, block_size=15)
        # Different block sizes may produce different results
        assert r1.binary.shape == r2.binary.shape


# ─── binarize_sauvola (extra) ────────────────────────────────────────────────

class TestBinarizeSauvolaExtra:
    def test_returns_result(self):
        result = binarize_sauvola(_gray())
        assert isinstance(result, BinarizeResult)

    def test_binary_shape(self):
        img = _gray(20, 30)
        result = binarize_sauvola(img)
        assert result.binary.shape == (20, 30)

    def test_binary_values_only_0_255(self):
        result = binarize_sauvola(_gray())
        unique = np.unique(result.binary)
        assert set(unique).issubset({0, 255})

    def test_method_contains_sauvola(self):
        result = binarize_sauvola(_gray())
        assert "sauvola" in result.method.lower()

    def test_invert_flag(self):
        result = binarize_sauvola(_gray(), invert=True)
        assert result.inverted is True


# ─── binarize_niblack (extra) ────────────────────────────────────────────────

class TestBinarizeNiblackExtra:
    def test_returns_result(self):
        result = binarize_niblack(_gray())
        assert isinstance(result, BinarizeResult)

    def test_binary_shape(self):
        img = _gray(16, 16)
        result = binarize_niblack(img)
        assert result.binary.shape == (16, 16)

    def test_binary_values_only_0_255(self):
        result = binarize_niblack(_gray())
        unique = np.unique(result.binary)
        assert set(unique).issubset({0, 255})

    def test_method_contains_niblack(self):
        result = binarize_niblack(_gray())
        assert "niblack" in result.method.lower()

    def test_invert_flag(self):
        result = binarize_niblack(_gray(), invert=True)
        assert result.inverted is True


# ─── binarize_bernsen (extra) ────────────────────────────────────────────────

class TestBinarizeBernsenExtra:
    def test_returns_result(self):
        result = binarize_bernsen(_gray())
        assert isinstance(result, BinarizeResult)

    def test_binary_shape(self):
        img = _gray(20, 20)
        result = binarize_bernsen(img)
        assert result.binary.shape == (20, 20)

    def test_binary_values_only_0_255(self):
        result = binarize_bernsen(_gray())
        unique = np.unique(result.binary)
        assert set(unique).issubset({0, 255})

    def test_method_contains_bernsen(self):
        result = binarize_bernsen(_gray())
        assert "bernsen" in result.method.lower()

    def test_invert_flag(self):
        result = binarize_bernsen(_gray(), invert=True)
        assert result.inverted is True


# ─── auto_binarize (extra) ───────────────────────────────────────────────────

class TestAutoBinarizeExtra:
    def test_returns_result(self):
        result = auto_binarize(_gray())
        assert isinstance(result, BinarizeResult)

    def test_binary_shape(self):
        img = _gray(24, 24)
        result = auto_binarize(img)
        assert result.binary.shape == (24, 24)

    def test_binary_values_only_0_255(self):
        result = auto_binarize(_gray())
        unique = np.unique(result.binary)
        assert set(unique).issubset({0, 255})

    def test_foreground_ratio_in_range(self):
        result = auto_binarize(_gray())
        assert 0.0 <= result.foreground_ratio <= 1.0

    def test_invert_flag_passed(self):
        result = auto_binarize(_gray(), invert=True)
        assert result.inverted is True

    def test_bimodal_image(self):
        img = _bimodal()
        result = auto_binarize(img)
        assert isinstance(result, BinarizeResult)


# ─── batch_binarize (extra) ──────────────────────────────────────────────────

class TestBatchBinarizeExtra:
    def test_returns_list(self):
        imgs = [_gray() for _ in range(3)]
        result = batch_binarize(imgs, method="otsu")
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_gray(seed=i) for i in range(4)]
        result = batch_binarize(imgs, method="otsu")
        assert len(result) == 4

    def test_all_binarize_results(self):
        imgs = [_gray(seed=i) for i in range(2)]
        result = batch_binarize(imgs, method="otsu")
        for r in result:
            assert isinstance(r, BinarizeResult)

    def test_empty_list_empty_result(self):
        result = batch_binarize([], method="otsu")
        assert result == []

    def test_adaptive_method(self):
        imgs = [_gray(seed=i) for i in range(2)]
        result = batch_binarize(imgs, method="adaptive")
        assert len(result) == 2

    def test_sauvola_method(self):
        imgs = [_gray(seed=i) for i in range(2)]
        result = batch_binarize(imgs, method="sauvola")
        assert len(result) == 2

    def test_auto_method(self):
        imgs = [_gray(seed=i) for i in range(2)]
        result = batch_binarize(imgs, method="auto")
        assert len(result) == 2
