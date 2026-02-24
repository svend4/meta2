"""Extra tests for puzzle_reconstruction/preprocessing/noise_reduction.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_reduction import (
    DenoiseResult,
    estimate_noise_level,
    denoise_gaussian,
    denoise_median,
    denoise_nlm,
    denoise_bilateral,
    denoise_morphological,
    smart_denoise,
    batch_denoise,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=50, w=50):
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (h, w), dtype=np.uint8)


# ─── DenoiseResult ────────────────────────────────────────────────────────────

class TestDenoiseResultExtra:
    def test_fields(self):
        r = DenoiseResult(denoised=_gray(), method="gaussian",
                          noise_before=10.0, noise_after=5.0)
        assert r.method == "gaussian"

    def test_noise_reduction_ratio(self):
        r = DenoiseResult(denoised=_gray(), method="x",
                          noise_before=10.0, noise_after=5.0)
        assert r.noise_reduction_ratio == pytest.approx(0.5)

    def test_noise_reduction_zero_before(self):
        r = DenoiseResult(denoised=_gray(), method="x",
                          noise_before=0.0, noise_after=0.0)
        assert r.noise_reduction_ratio == pytest.approx(0.0)

    def test_repr(self):
        r = DenoiseResult(denoised=_gray(), method="gaussian",
                          noise_before=10.0, noise_after=5.0)
        s = repr(r)
        assert "gaussian" in s


# ─── estimate_noise_level ─────────────────────────────────────────────────────

class TestEstimateNoiseLevelExtra:
    def test_nonnegative(self):
        assert estimate_noise_level(_gray()) >= 0

    def test_uniform_low(self):
        level = estimate_noise_level(_gray())
        assert level < 1.0

    def test_bgr_input(self):
        level = estimate_noise_level(_bgr())
        assert isinstance(level, float)


# ─── denoise_gaussian ─────────────────────────────────────────────────────────

class TestDenoiseGaussianExtra:
    def test_returns_result(self):
        r = denoise_gaussian(_gray())
        assert isinstance(r, DenoiseResult)
        assert r.method == "gaussian"

    def test_shape_preserved(self):
        r = denoise_gaussian(_gray())
        assert r.denoised.shape == (50, 50)

    def test_bgr(self):
        r = denoise_gaussian(_bgr())
        assert r.denoised.shape == (50, 50, 3)


# ─── denoise_median ───────────────────────────────────────────────────────────

class TestDenoiseMedianExtra:
    def test_returns_result(self):
        r = denoise_median(_gray())
        assert r.method == "median"

    def test_shape_preserved(self):
        r = denoise_median(_gray())
        assert r.denoised.shape == (50, 50)


# ─── denoise_nlm ──────────────────────────────────────────────────────────────

class TestDenoiseNlmExtra:
    def test_returns_result(self):
        r = denoise_nlm(_gray())
        assert r.method == "nlm"

    def test_gray(self):
        r = denoise_nlm(_gray())
        assert r.denoised.ndim == 2

    def test_bgr(self):
        r = denoise_nlm(_bgr())
        assert r.denoised.ndim == 3


# ─── denoise_bilateral ────────────────────────────────────────────────────────

class TestDenoiseBilateralExtra:
    def test_returns_result(self):
        r = denoise_bilateral(_gray())
        assert r.method == "bilateral"

    def test_gray(self):
        r = denoise_bilateral(_gray())
        assert r.denoised.ndim == 2

    def test_bgr(self):
        r = denoise_bilateral(_bgr())
        assert r.denoised.ndim == 3


# ─── denoise_morphological ────────────────────────────────────────────────────

class TestDenoiseMorphologicalExtra:
    def test_open(self):
        r = denoise_morphological(_gray(), op="open")
        assert "open" in r.method

    def test_close(self):
        r = denoise_morphological(_gray(), op="close")
        assert "close" in r.method

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError):
            denoise_morphological(_gray(), op="bad")

    def test_shape_preserved(self):
        r = denoise_morphological(_gray())
        assert r.denoised.shape == (50, 50)


# ─── smart_denoise ────────────────────────────────────────────────────────────

class TestSmartDenoiseExtra:
    def test_clean_image_is_none(self):
        r = smart_denoise(_gray())
        assert r.method == "none"

    def test_noisy_image_filtered(self):
        r = smart_denoise(_noisy())
        assert r.method != "none"

    def test_shape_preserved(self):
        r = smart_denoise(_gray())
        assert r.denoised.shape == (50, 50)


# ─── batch_denoise ────────────────────────────────────────────────────────────

class TestBatchDenoiseExtra:
    def test_empty(self):
        assert batch_denoise([]) == []

    def test_length(self):
        results = batch_denoise([_gray(), _gray()])
        assert len(results) == 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_denoise([_gray()], method="bad")

    def test_all_methods(self):
        for m in ("smart", "gaussian", "median", "nlm", "bilateral", "morphological"):
            results = batch_denoise([_gray()], method=m)
            assert len(results) == 1
