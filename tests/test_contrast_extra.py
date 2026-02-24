"""Extra tests for puzzle_reconstruction/preprocessing/contrast.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.contrast import (
    ContrastResult,
    measure_contrast,
    enhance_clahe,
    enhance_histeq,
    enhance_gamma,
    enhance_stretch,
    enhance_retinex,
    auto_enhance,
    batch_enhance,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray_low(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((32, 32)) * 15 + 120).astype(np.uint8)


def _gray_high():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:16, :] = 20
    img[16:, :] = 220
    return img


def _bgr(seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((32, 32, 3)) * 200 + 30).astype(np.uint8)


def _constant(val=128):
    return np.full((16, 16), val, dtype=np.uint8)


def _gradient():
    img = np.zeros((32, 32), dtype=np.uint8)
    for i in range(32):
        img[i, :] = i * 4
    return img


# ─── ContrastResult (extra) ──────────────────────────────────────────────────

class TestContrastResultExtra:
    def test_improvement_positive(self):
        res = ContrastResult(
            enhanced=np.zeros((8, 8), dtype=np.uint8),
            method="clahe", contrast_before=20.0, contrast_after=50.0,
        )
        assert res.improvement == pytest.approx(30.0)

    def test_improvement_negative(self):
        res = ContrastResult(
            enhanced=np.zeros((8, 8), dtype=np.uint8),
            method="x", contrast_before=50.0, contrast_after=30.0,
        )
        assert res.improvement == pytest.approx(-20.0)

    def test_improvement_ratio_positive(self):
        res = ContrastResult(
            enhanced=np.zeros((8, 8), dtype=np.uint8),
            method="x", contrast_before=40.0, contrast_after=60.0,
        )
        assert res.improvement_ratio == pytest.approx(0.5)

    def test_improvement_ratio_zero_before(self):
        res = ContrastResult(
            enhanced=np.zeros((8, 8), dtype=np.uint8),
            method="x", contrast_before=0.0, contrast_after=10.0,
        )
        assert res.improvement_ratio == pytest.approx(0.0)

    def test_default_params_empty(self):
        res = ContrastResult(
            enhanced=np.zeros((4, 4), dtype=np.uint8),
            method="x", contrast_before=0.0, contrast_after=0.0,
        )
        assert res.params == {}

    def test_repr_contains_method(self):
        res = ContrastResult(
            enhanced=np.zeros((4, 4), dtype=np.uint8),
            method="gamma", contrast_before=20.0, contrast_after=40.0,
        )
        assert "gamma" in repr(res)

    def test_method_stored(self):
        res = ContrastResult(
            enhanced=np.zeros((4, 4), dtype=np.uint8),
            method="retinex", contrast_before=10.0, contrast_after=20.0,
        )
        assert res.method == "retinex"


# ─── measure_contrast (extra) ────────────────────────────────────────────────

class TestMeasureContrastExtra:
    def test_returns_float(self):
        assert isinstance(measure_contrast(_gray_low()), float)

    def test_constant_zero(self):
        assert measure_contrast(_constant()) == pytest.approx(0.0)

    def test_nonneg(self):
        assert measure_contrast(_gray_low()) >= 0.0

    def test_high_gt_low(self):
        assert measure_contrast(_gray_high()) > measure_contrast(_gray_low())

    def test_bgr_accepted(self):
        val = measure_contrast(_bgr())
        assert isinstance(val, float)
        assert val >= 0.0


# ─── enhance_clahe (extra) ───────────────────────────────────────────────────

class TestEnhanceClaheExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_clahe(_gray_low()), ContrastResult)

    def test_method_is_clahe(self):
        assert enhance_clahe(_gray_low()).method == "clahe"

    def test_gray_shape_preserved(self):
        img = _gray_low()
        assert enhance_clahe(img).enhanced.shape == img.shape

    def test_bgr_shape_preserved(self):
        img = _bgr()
        assert enhance_clahe(img).enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_clahe(_gray_low()).enhanced.dtype == np.uint8

    def test_clip_limit_stored(self):
        res = enhance_clahe(_gray_low(), clip_limit=3.0)
        assert res.params["clip_limit"] == pytest.approx(3.0)

    def test_tile_size_stored(self):
        res = enhance_clahe(_gray_low(), tile_size=4)
        assert res.params["tile_size"] == 4

    def test_does_not_modify_input(self):
        img = _gray_low()
        orig = img.copy()
        enhance_clahe(img)
        np.testing.assert_array_equal(img, orig)


# ─── enhance_histeq (extra) ──────────────────────────────────────────────────

class TestEnhanceHisteqExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_histeq(_gray_low()), ContrastResult)

    def test_method_is_histeq(self):
        assert enhance_histeq(_gray_low()).method == "histeq"

    def test_gray_shape_preserved(self):
        img = _gray_low()
        assert enhance_histeq(img).enhanced.shape == img.shape

    def test_bgr_shape_preserved(self):
        img = _bgr()
        assert enhance_histeq(img).enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_histeq(_gray_low()).enhanced.dtype == np.uint8

    def test_does_not_modify_input(self):
        img = _gray_low()
        orig = img.copy()
        enhance_histeq(img)
        np.testing.assert_array_equal(img, orig)

    def test_contrast_improves(self):
        res = enhance_histeq(_gray_low())
        assert res.contrast_after >= res.contrast_before


# ─── enhance_gamma (extra) ───────────────────────────────────────────────────

class TestEnhanceGammaExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_gamma(_gradient()), ContrastResult)

    def test_method_is_gamma(self):
        assert enhance_gamma(_gradient()).method == "gamma"

    def test_gamma_stored(self):
        res = enhance_gamma(_gradient(), gamma=2.0)
        assert res.params["gamma"] == pytest.approx(2.0)

    def test_shape_preserved_gray(self):
        img = _gradient()
        assert enhance_gamma(img).enhanced.shape == img.shape

    def test_shape_preserved_bgr(self):
        img = _bgr()
        assert enhance_gamma(img).enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_gamma(_gradient()).enhanced.dtype == np.uint8

    def test_gamma_one_near_identity(self):
        img = _gradient()
        res = enhance_gamma(img, gamma=1.0)
        diff = np.abs(res.enhanced.astype(int) - img.astype(int))
        assert diff.max() <= 2

    def test_same_gamma_same_result(self):
        img = _gradient()
        r1 = enhance_gamma(img, gamma=2.2)
        r2 = enhance_gamma(img, gamma=2.2)
        np.testing.assert_array_equal(r1.enhanced, r2.enhanced)


# ─── enhance_stretch (extra) ─────────────────────────────────────────────────

class TestEnhanceStretchExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_stretch(_gray_low()), ContrastResult)

    def test_method_is_stretch(self):
        assert enhance_stretch(_gray_low()).method == "stretch"

    def test_p_low_stored(self):
        res = enhance_stretch(_gray_low(), p_low=5.0, p_high=95.0)
        assert res.params["p_low"] == pytest.approx(5.0)

    def test_p_high_stored(self):
        res = enhance_stretch(_gray_low(), p_low=5.0, p_high=95.0)
        assert res.params["p_high"] == pytest.approx(95.0)

    def test_shape_preserved_gray(self):
        img = _gray_low()
        assert enhance_stretch(img).enhanced.shape == img.shape

    def test_shape_preserved_bgr(self):
        img = _bgr()
        assert enhance_stretch(img).enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_stretch(_gray_low()).enhanced.dtype == np.uint8

    def test_constant_no_crash(self):
        res = enhance_stretch(_constant())
        assert isinstance(res, ContrastResult)

    def test_does_not_modify_input(self):
        img = _gray_low()
        orig = img.copy()
        enhance_stretch(img)
        np.testing.assert_array_equal(img, orig)


# ─── enhance_retinex (extra) ─────────────────────────────────────────────────

class TestEnhanceRetinexExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_retinex(_gradient()), ContrastResult)

    def test_method_is_retinex(self):
        assert enhance_retinex(_gradient()).method == "retinex"

    def test_sigma_stored(self):
        res = enhance_retinex(_gradient(), sigma=20.0)
        assert res.params["sigma"] == pytest.approx(20.0)

    def test_gray_shape_preserved(self):
        img = _gradient()
        assert enhance_retinex(img).enhanced.shape == img.shape

    def test_bgr_shape_preserved(self):
        img = _bgr()
        assert enhance_retinex(img).enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_retinex(_gradient()).enhanced.dtype == np.uint8

    def test_constant_no_crash(self):
        res = enhance_retinex(_constant())
        assert isinstance(res, ContrastResult)

    def test_output_in_range(self):
        res = enhance_retinex(_gradient())
        assert res.enhanced.min() >= 0
        assert res.enhanced.max() <= 255


# ─── auto_enhance (extra) ────────────────────────────────────────────────────

class TestAutoEnhanceExtra:
    def test_returns_contrast_result(self):
        assert isinstance(auto_enhance(_gradient()), ContrastResult)

    def test_shape_preserved(self):
        img = _gradient()
        assert auto_enhance(img).enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert auto_enhance(_gradient()).enhanced.dtype == np.uint8

    def test_very_low_contrast_clahe(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        rng = np.random.default_rng(5)
        img = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
        res = auto_enhance(img)
        assert res.method == "clahe"

    def test_bgr_accepted(self):
        res = auto_enhance(_bgr())
        assert isinstance(res, ContrastResult)


# ─── batch_enhance (extra) ───────────────────────────────────────────────────

class TestBatchEnhanceExtra:
    def test_returns_list(self):
        result = batch_enhance([_gradient(), _gradient()], method="gamma")
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_gradient() for _ in range(3)]
        assert len(batch_enhance(imgs, method="clahe")) == 3

    def test_each_is_contrast_result(self):
        results = batch_enhance([_gradient()], method="histeq")
        assert isinstance(results[0], ContrastResult)

    def test_empty_list_empty_result(self):
        assert batch_enhance([], method="gamma") == []

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_enhance([_gradient()], method="xyz_unknown")

    def test_kwargs_forwarded(self):
        results = batch_enhance([_gradient()], method="gamma", gamma=2.5)
        assert results[0].params["gamma"] == pytest.approx(2.5)

    def test_all_methods_work(self):
        img = _gradient()
        for method in ("auto", "clahe", "histeq", "gamma", "stretch", "retinex"):
            results = batch_enhance([img], method=method)
            assert len(results) == 1
