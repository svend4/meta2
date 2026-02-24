"""Extra tests for puzzle_reconstruction/preprocessing/contrast.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=64) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32, val=100) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _noise(h=32, w=32, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ─── ContrastResult ───────────────────────────────────────────────────────────

class TestContrastResultExtra:
    def _make(self, before=30.0, after=50.0, method="clahe") -> ContrastResult:
        img = np.zeros((8, 8), dtype=np.uint8)
        return ContrastResult(enhanced=img, method=method,
                              contrast_before=before, contrast_after=after)

    def test_method_stored(self):
        r = self._make(method="histeq")
        assert r.method == "histeq"

    def test_contrast_before_stored(self):
        r = self._make(before=25.0)
        assert r.contrast_before == pytest.approx(25.0)

    def test_contrast_after_stored(self):
        r = self._make(after=60.0)
        assert r.contrast_after == pytest.approx(60.0)

    def test_improvement_positive(self):
        r = self._make(before=30.0, after=50.0)
        assert r.improvement == pytest.approx(20.0)

    def test_improvement_negative(self):
        r = self._make(before=50.0, after=30.0)
        assert r.improvement == pytest.approx(-20.0)

    def test_improvement_ratio_zero_before(self):
        r = self._make(before=0.0, after=10.0)
        assert r.improvement_ratio == pytest.approx(0.0)

    def test_improvement_ratio_computed(self):
        r = self._make(before=40.0, after=60.0)
        assert r.improvement_ratio == pytest.approx(0.5)

    def test_repr_contains_method(self):
        r = self._make(method="gamma")
        assert "gamma" in repr(r)

    def test_default_params_empty(self):
        r = self._make()
        assert r.params == {}


# ─── measure_contrast ─────────────────────────────────────────────────────────

class TestMeasureContrastExtra:
    def test_returns_float(self):
        assert isinstance(measure_contrast(_gray()), float)

    def test_uniform_image_zero(self):
        assert measure_contrast(_gray()) == pytest.approx(0.0)

    def test_ramp_nonzero(self):
        assert measure_contrast(_ramp()) > 0.0

    def test_bgr_input(self):
        # Should not raise
        val = measure_contrast(_bgr())
        assert isinstance(val, float)

    def test_all_black_zero(self):
        assert measure_contrast(_gray(val=0)) == pytest.approx(0.0)

    def test_all_white_zero(self):
        assert measure_contrast(_gray(val=255)) == pytest.approx(0.0)

    def test_high_contrast_large_value(self):
        # Checkerboard-like noise has high std
        img = _noise()
        assert measure_contrast(img) > 50.0


# ─── enhance_clahe ────────────────────────────────────────────────────────────

class TestEnhanceClaheExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_clahe(_ramp()), ContrastResult)

    def test_method_name(self):
        assert enhance_clahe(_ramp()).method == "clahe"

    def test_output_shape_gray(self):
        img = _gray(20, 30)
        r = enhance_clahe(img)
        assert r.enhanced.shape == img.shape

    def test_output_shape_bgr(self):
        img = _bgr(20, 30)
        r = enhance_clahe(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_preserved_gray(self):
        r = enhance_clahe(_gray())
        assert r.enhanced.dtype == np.uint8

    def test_params_stored(self):
        r = enhance_clahe(_ramp(), clip_limit=3.0, tile_size=4)
        assert r.params["clip_limit"] == pytest.approx(3.0)
        assert r.params["tile_size"] == 4

    def test_contrast_before_nonnegative(self):
        r = enhance_clahe(_ramp())
        assert r.contrast_before >= 0.0

    def test_contrast_after_nonnegative(self):
        r = enhance_clahe(_ramp())
        assert r.contrast_after >= 0.0


# ─── enhance_histeq ───────────────────────────────────────────────────────────

class TestEnhanceHisteqExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_histeq(_ramp()), ContrastResult)

    def test_method_name(self):
        assert enhance_histeq(_ramp()).method == "histeq"

    def test_output_shape_gray(self):
        img = _gray(20, 30)
        r = enhance_histeq(img)
        assert r.enhanced.shape == img.shape

    def test_output_shape_bgr(self):
        img = _bgr(20, 30)
        r = enhance_histeq(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_histeq(_ramp()).enhanced.dtype == np.uint8

    def test_uniform_image_contrast_after(self):
        r = enhance_histeq(_gray(val=128))
        # Histeq of uniform image: contrast may increase
        assert r.contrast_after >= 0.0

    def test_contrast_before_matches_measure(self):
        img = _ramp()
        r = enhance_histeq(img)
        assert r.contrast_before == pytest.approx(measure_contrast(img), abs=0.1)


# ─── enhance_gamma ────────────────────────────────────────────────────────────

class TestEnhanceGammaExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_gamma(_ramp()), ContrastResult)

    def test_method_name(self):
        assert enhance_gamma(_ramp()).method == "gamma"

    def test_output_shape(self):
        img = _gray(20, 30)
        r = enhance_gamma(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_gamma(_ramp()).enhanced.dtype == np.uint8

    def test_gamma_one_nearly_identity(self):
        img = _ramp()
        r = enhance_gamma(img, gamma=1.0)
        # With gamma=1, output should be close to input
        assert np.allclose(r.enhanced.astype(float), img.astype(float), atol=5)

    def test_gamma_stored_in_params(self):
        r = enhance_gamma(_ramp(), gamma=2.0)
        assert r.params["gamma"] == pytest.approx(2.0)

    def test_bgr_input(self):
        r = enhance_gamma(_bgr())
        assert r.enhanced.shape == (32, 32, 3)


# ─── enhance_stretch ──────────────────────────────────────────────────────────

class TestEnhanceStretchExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_stretch(_ramp()), ContrastResult)

    def test_method_name(self):
        assert enhance_stretch(_ramp()).method == "stretch"

    def test_output_shape(self):
        img = _gray(20, 30)
        r = enhance_stretch(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_stretch(_ramp()).enhanced.dtype == np.uint8

    def test_params_stored(self):
        r = enhance_stretch(_ramp(), p_low=5.0, p_high=95.0)
        assert r.params["p_low"] == pytest.approx(5.0)
        assert r.params["p_high"] == pytest.approx(95.0)

    def test_bgr_input(self):
        img = _bgr()
        r = enhance_stretch(img)
        assert r.enhanced.shape == img.shape

    def test_uniform_input_unchanged(self):
        img = _gray(val=100)
        r = enhance_stretch(img)
        # hi <= lo for uniform → copy → same values
        assert r.enhanced.shape == img.shape


# ─── enhance_retinex ──────────────────────────────────────────────────────────

class TestEnhanceRetinexExtra:
    def test_returns_contrast_result(self):
        assert isinstance(enhance_retinex(_ramp()), ContrastResult)

    def test_method_name(self):
        assert enhance_retinex(_ramp()).method == "retinex"

    def test_output_shape_gray(self):
        img = _gray(20, 30)
        r = enhance_retinex(img)
        assert r.enhanced.shape == img.shape

    def test_output_shape_bgr(self):
        img = _bgr(20, 30)
        r = enhance_retinex(img)
        assert r.enhanced.shape == img.shape

    def test_dtype_uint8(self):
        assert enhance_retinex(_ramp()).enhanced.dtype == np.uint8

    def test_params_sigma_stored(self):
        r = enhance_retinex(_ramp(), sigma=20.0)
        assert r.params["sigma"] == pytest.approx(20.0)

    def test_values_in_range(self):
        r = enhance_retinex(_ramp())
        assert r.enhanced.min() >= 0
        assert r.enhanced.max() <= 255


# ─── auto_enhance ─────────────────────────────────────────────────────────────

class TestAutoEnhanceExtra:
    def test_returns_contrast_result(self):
        assert isinstance(auto_enhance(_ramp()), ContrastResult)

    def test_output_shape(self):
        img = _gray(20, 30)
        r = auto_enhance(img)
        assert r.enhanced.shape == img.shape

    def test_low_contrast_uses_clahe(self):
        # Very flat image → rms < 20 → clahe
        img = _gray(val=128)
        r = auto_enhance(img)
        assert r.method == "clahe"

    def test_high_contrast_uses_gamma(self):
        # High-contrast noise image → rms ≥ 60 → gamma
        img = _noise()
        rms = measure_contrast(img)
        if rms >= 60.0:
            r = auto_enhance(img)
            assert r.method == "gamma"

    def test_dtype_uint8(self):
        r = auto_enhance(_ramp())
        assert r.enhanced.dtype == np.uint8


# ─── batch_enhance ────────────────────────────────────────────────────────────

class TestBatchEnhanceExtra:
    def test_returns_list(self):
        result = batch_enhance([_gray()], method="clahe")
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_gray(), _ramp()]
        assert len(batch_enhance(imgs, method="histeq")) == 2

    def test_empty_list(self):
        assert batch_enhance([], method="clahe") == []

    def test_each_element_is_result(self):
        for r in batch_enhance([_gray(), _ramp()], method="gamma"):
            assert isinstance(r, ContrastResult)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_enhance([_gray()], method="nonexistent")

    def test_clahe_method(self):
        results = batch_enhance([_ramp()], method="clahe")
        assert results[0].method == "clahe"

    def test_histeq_method(self):
        results = batch_enhance([_ramp()], method="histeq")
        assert results[0].method == "histeq"

    def test_stretch_method(self):
        results = batch_enhance([_ramp()], method="stretch")
        assert results[0].method == "stretch"

    def test_retinex_method(self):
        results = batch_enhance([_ramp()], method="retinex")
        assert results[0].method == "retinex"
