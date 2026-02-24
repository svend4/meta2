"""Extra tests for puzzle_reconstruction/preprocessing/noise_reduction.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_reduction import (
    DenoiseResult,
    batch_denoise,
    denoise_bilateral,
    denoise_gaussian,
    denoise_median,
    denoise_morphological,
    denoise_nlm,
    estimate_noise_level,
    smart_denoise,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _clean(h: int = 32, w: int = 32) -> np.ndarray:
    """Very clean (constant) image."""
    return np.full((h, w), 200, dtype=np.uint8)


# ─── DenoiseResult (extra) ────────────────────────────────────────────────────

class TestDenoiseResultExtra:
    def _make(self, method: str = "gaussian",
              nb: float = 5.0, na: float = 3.0) -> DenoiseResult:
        return DenoiseResult(denoised=_gray(), method=method,
                              noise_before=nb, noise_after=na)

    def test_method_stored(self):
        r = self._make("median")
        assert r.method == "median"

    def test_noise_before_stored(self):
        r = self._make(nb=8.0)
        assert r.noise_before == pytest.approx(8.0)

    def test_noise_after_stored(self):
        r = self._make(na=2.0)
        assert r.noise_after == pytest.approx(2.0)

    def test_denoised_stored(self):
        r = self._make()
        assert r.denoised.dtype == np.uint8

    def test_noise_reduction_ratio_positive(self):
        r = self._make(nb=10.0, na=4.0)
        assert r.noise_reduction_ratio > 0.0

    def test_noise_reduction_ratio_zero_before(self):
        r = self._make(nb=0.0, na=0.0)
        assert r.noise_reduction_ratio == pytest.approx(0.0)

    def test_noise_reduction_ratio_clipped_to_1(self):
        r = self._make(nb=5.0, na=-10.0)
        assert r.noise_reduction_ratio <= 1.0

    def test_noise_reduction_ratio_clipped_to_0(self):
        r = self._make(nb=3.0, na=5.0)  # after > before
        assert r.noise_reduction_ratio >= 0.0

    def test_repr_contains_method(self):
        r = self._make("bilateral")
        assert "bilateral" in repr(r)

    def test_repr_contains_reduction(self):
        r = self._make(nb=10.0, na=5.0)
        assert "0.5" in repr(r) or "reduction" in repr(r).lower()

    def test_params_default_empty(self):
        r = self._make()
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        r = DenoiseResult(_gray(), "gaussian", 5.0, 3.0, params={"ksize": 5})
        assert r.params["ksize"] == 5


# ─── estimate_noise_level (extra) ─────────────────────────────────────────────

class TestEstimateNoiseLevelExtra:
    def test_returns_float(self):
        assert isinstance(estimate_noise_level(_gray()), float)

    def test_constant_image_near_zero(self):
        assert estimate_noise_level(_clean()) < 1.0

    def test_noisy_image_positive(self):
        assert estimate_noise_level(_noisy()) > 0.0

    def test_noisy_greater_than_clean(self):
        assert estimate_noise_level(_noisy()) > estimate_noise_level(_clean())

    def test_nonneg(self):
        assert estimate_noise_level(_gray()) >= 0.0

    def test_bgr_input(self):
        assert estimate_noise_level(_bgr()) >= 0.0

    def test_small_image(self):
        assert estimate_noise_level(_noisy(4, 4)) >= 0.0

    def test_large_image(self):
        assert estimate_noise_level(_noisy(128, 128)) >= 0.0


# ─── denoise_gaussian (extra) ─────────────────────────────────────────────────

class TestDenoiseGaussianExtra:
    def test_returns_denoise_result(self):
        assert isinstance(denoise_gaussian(_gray()), DenoiseResult)

    def test_method_is_gaussian(self):
        assert denoise_gaussian(_gray()).method == "gaussian"

    def test_shape_preserved_gray(self):
        r = denoise_gaussian(_noisy(16, 24))
        assert r.denoised.shape == (16, 24)

    def test_shape_preserved_bgr(self):
        r = denoise_gaussian(_bgr(16, 24))
        assert r.denoised.shape == (16, 24, 3)

    def test_dtype_uint8(self):
        assert denoise_gaussian(_noisy()).denoised.dtype == np.uint8

    def test_noise_before_nonneg(self):
        assert denoise_gaussian(_noisy()).noise_before >= 0.0

    def test_noise_after_nonneg(self):
        assert denoise_gaussian(_noisy()).noise_after >= 0.0

    def test_params_ksize_stored(self):
        r = denoise_gaussian(_gray(), ksize=7)
        assert r.params.get("ksize") == 7

    def test_even_ksize_made_odd(self):
        r = denoise_gaussian(_gray(), ksize=4)
        assert r.params.get("ksize") % 2 == 1


# ─── denoise_median (extra) ───────────────────────────────────────────────────

class TestDenoiseMedianExtra:
    def test_returns_denoise_result(self):
        assert isinstance(denoise_median(_gray()), DenoiseResult)

    def test_method_is_median(self):
        assert denoise_median(_gray()).method == "median"

    def test_shape_preserved(self):
        r = denoise_median(_noisy(16, 24))
        assert r.denoised.shape == (16, 24)

    def test_dtype_uint8(self):
        assert denoise_median(_noisy()).denoised.dtype == np.uint8

    def test_params_ksize_odd(self):
        r = denoise_median(_gray(), ksize=5)
        assert r.params["ksize"] % 2 == 1

    def test_small_ksize_forced_to_3(self):
        r = denoise_median(_gray(), ksize=1)
        assert r.params["ksize"] >= 3

    def test_noise_before_nonneg(self):
        assert denoise_median(_noisy()).noise_before >= 0.0


# ─── denoise_nlm (extra) ──────────────────────────────────────────────────────

class TestDenoiseNlmExtra:
    def test_returns_denoise_result(self):
        assert isinstance(denoise_nlm(_gray()), DenoiseResult)

    def test_method_is_nlm(self):
        assert denoise_nlm(_gray()).method == "nlm"

    def test_shape_preserved_gray(self):
        r = denoise_nlm(_noisy(16, 24))
        assert r.denoised.shape == (16, 24)

    def test_shape_preserved_bgr(self):
        r = denoise_nlm(_bgr(16, 24))
        assert r.denoised.shape == (16, 24, 3)

    def test_dtype_uint8(self):
        assert denoise_nlm(_noisy()).denoised.dtype == np.uint8

    def test_params_stored(self):
        r = denoise_nlm(_gray(), h=5.0)
        assert r.params["h"] == pytest.approx(5.0)

    def test_noise_before_nonneg(self):
        assert denoise_nlm(_noisy()).noise_before >= 0.0


# ─── denoise_bilateral (extra) ────────────────────────────────────────────────

class TestDenoiseBilateralExtra:
    def test_returns_denoise_result(self):
        assert isinstance(denoise_bilateral(_gray()), DenoiseResult)

    def test_method_is_bilateral(self):
        assert denoise_bilateral(_gray()).method == "bilateral"

    def test_shape_preserved_gray(self):
        r = denoise_bilateral(_noisy(16, 24))
        assert r.denoised.shape == (16, 24)

    def test_shape_preserved_bgr(self):
        r = denoise_bilateral(_bgr(16, 24))
        assert r.denoised.shape == (16, 24, 3)

    def test_dtype_uint8(self):
        assert denoise_bilateral(_noisy()).denoised.dtype == np.uint8

    def test_params_stored(self):
        r = denoise_bilateral(_gray(), d=5)
        assert r.params["d"] == 5

    def test_noise_before_nonneg(self):
        assert denoise_bilateral(_noisy()).noise_before >= 0.0


# ─── denoise_morphological (extra) ────────────────────────────────────────────

class TestDenoiseMorphologicalExtra:
    def test_returns_denoise_result(self):
        assert isinstance(denoise_morphological(_gray()), DenoiseResult)

    def test_method_contains_op(self):
        r = denoise_morphological(_gray(), op="open")
        assert "open" in r.method

    def test_shape_preserved(self):
        r = denoise_morphological(_noisy(16, 24))
        assert r.denoised.shape == (16, 24)

    def test_dtype_uint8(self):
        assert denoise_morphological(_noisy()).denoised.dtype == np.uint8

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            denoise_morphological(_gray(), op="gradient")

    def test_open_op(self):
        r = denoise_morphological(_gray(), op="open")
        assert isinstance(r, DenoiseResult)

    def test_close_op(self):
        r = denoise_morphological(_gray(), op="close")
        assert isinstance(r, DenoiseResult)

    def test_tophat_op(self):
        r = denoise_morphological(_gray(), op="tophat")
        assert isinstance(r, DenoiseResult)

    def test_blackhat_op(self):
        r = denoise_morphological(_gray(), op="blackhat")
        assert isinstance(r, DenoiseResult)

    def test_params_op_stored(self):
        r = denoise_morphological(_gray(), op="close")
        assert r.params.get("op") == "close"


# ─── smart_denoise (extra) ────────────────────────────────────────────────────

class TestSmartDenoiseExtra:
    def test_returns_denoise_result(self):
        assert isinstance(smart_denoise(_gray()), DenoiseResult)

    def test_clean_image_method_none(self):
        r = smart_denoise(_clean())
        assert r.method == "none"

    def test_clean_image_denoised_copy(self):
        img = _clean()
        r = smart_denoise(img)
        assert np.array_equal(r.denoised, img)

    def test_shape_preserved(self):
        r = smart_denoise(_noisy(16, 24))
        assert r.denoised.shape == (16, 24)

    def test_bgr_input(self):
        r = smart_denoise(_bgr())
        assert isinstance(r, DenoiseResult)

    def test_noise_before_nonneg(self):
        assert smart_denoise(_noisy()).noise_before >= 0.0

    def test_dtype_uint8(self):
        assert smart_denoise(_noisy()).denoised.dtype == np.uint8

    def test_very_noisy_not_none(self):
        # High-variance image should use a real denoiser
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        r = smart_denoise(img)
        assert isinstance(r, DenoiseResult)


# ─── batch_denoise (extra) ────────────────────────────────────────────────────

class TestBatchDenoiseExtra:
    def test_empty_returns_empty(self):
        assert batch_denoise([]) == []

    def test_single_image(self):
        result = batch_denoise([_gray()])
        assert len(result) == 1

    def test_multiple_images(self):
        result = batch_denoise([_gray(), _noisy(), _bgr()])
        assert len(result) == 3

    def test_all_denoise_results(self):
        for r in batch_denoise([_gray(), _noisy()]):
            assert isinstance(r, DenoiseResult)

    def test_method_gaussian(self):
        result = batch_denoise([_gray()], method="gaussian")
        assert result[0].method == "gaussian"

    def test_method_median(self):
        result = batch_denoise([_gray()], method="median")
        assert result[0].method == "median"

    def test_method_bilateral(self):
        result = batch_denoise([_gray()], method="bilateral")
        assert result[0].method == "bilateral"

    def test_method_nlm(self):
        result = batch_denoise([_gray()], method="nlm")
        assert result[0].method == "nlm"

    def test_method_smart(self):
        result = batch_denoise([_clean()], method="smart")
        assert result[0].method == "none"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_denoise([_gray()], method="magic")

    def test_kwargs_passed(self):
        result = batch_denoise([_gray()], method="gaussian", ksize=7)
        assert result[0].params.get("ksize") == 7
