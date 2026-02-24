"""Extra tests for puzzle_reconstruction/preprocessing/patch_normalizer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.patch_normalizer import (
    NormalizationParams,
    equalize_histogram,
    stretch_contrast,
    standardize_patch,
    normalize_patch,
    batch_normalize,
    compute_normalization_stats,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── NormalizationParams ──────────────────────────────────────────────────────

class TestNormalizationParamsExtra:
    def test_defaults(self):
        p = NormalizationParams()
        assert p.method == "clahe"
        assert p.clip_limit == pytest.approx(2.0)

    def test_valid_methods(self):
        for m in ("equalize", "clahe", "stretch", "standardize", "none"):
            NormalizationParams(method=m)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NormalizationParams(method="bad")


# ─── equalize_histogram ───────────────────────────────────────────────────────

class TestEqualizeHistogramExtra:
    def test_global(self):
        out = equalize_histogram(_gray(), method="global")
        assert out.ndim == 2 and out.dtype == np.uint8

    def test_clahe(self):
        out = equalize_histogram(_gray(), method="clahe")
        assert out.ndim == 2

    def test_bgr_input(self):
        out = equalize_histogram(_bgr(), method="global")
        assert out.ndim == 2

    def test_bad_method_raises(self):
        with pytest.raises(ValueError):
            equalize_histogram(_gray(), method="bad")


# ─── stretch_contrast ─────────────────────────────────────────────────────────

class TestStretchContrastExtra:
    def test_dtype_uint8(self):
        out = stretch_contrast(_gray())
        assert out.dtype == np.uint8

    def test_bgr_input(self):
        out = stretch_contrast(_bgr())
        assert out.ndim == 2

    def test_bad_pct_raises(self):
        with pytest.raises(ValueError):
            stretch_contrast(_gray(), low_pct=98, high_pct=2)

    def test_uniform_image(self):
        out = stretch_contrast(_gray(val=128))
        assert out.dtype == np.uint8


# ─── standardize_patch ────────────────────────────────────────────────────────

class TestStandardizePatchExtra:
    def test_dtype_uint8(self):
        out = standardize_patch(_gray())
        assert out.dtype == np.uint8

    def test_uniform_image(self):
        out = standardize_patch(_gray(val=100), target_mean=128.0)
        # Uniform → all target_mean
        assert np.allclose(out, 128, atol=1)

    def test_bgr_input(self):
        out = standardize_patch(_bgr())
        assert out.ndim == 2


# ─── normalize_patch ──────────────────────────────────────────────────────────

class TestNormalizePatchExtra:
    def test_default_clahe(self):
        out = normalize_patch(_gray())
        assert out.dtype == np.uint8

    def test_none_method(self):
        p = NormalizationParams(method="none")
        out = normalize_patch(_gray(), p)
        assert out.dtype == np.uint8

    def test_equalize(self):
        p = NormalizationParams(method="equalize")
        out = normalize_patch(_gray(), p)
        assert out.ndim == 2

    def test_stretch(self):
        p = NormalizationParams(method="stretch")
        out = normalize_patch(_gray(), p)
        assert out.ndim == 2

    def test_standardize(self):
        p = NormalizationParams(method="standardize")
        out = normalize_patch(_gray(), p)
        assert out.ndim == 2


# ─── batch_normalize ──────────────────────────────────────────────────────────

class TestBatchNormalizePatchExtra:
    def test_empty(self):
        assert batch_normalize([]) == []

    def test_length(self):
        results = batch_normalize([_gray(), _gray()])
        assert len(results) == 2

    def test_dtype(self):
        results = batch_normalize([_gray()])
        assert results[0].dtype == np.uint8


# ─── compute_normalization_stats ──────────────────────────────────────────────

class TestComputeNormalizationStatsExtra:
    def test_keys(self):
        stats = compute_normalization_stats([_gray()])
        assert "mean" in stats
        assert "std" in stats
        assert "n_images" in stats

    def test_n_images(self):
        stats = compute_normalization_stats([_gray(), _gray()])
        assert stats["n_images"] == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_normalization_stats([])

    def test_uniform_mean(self):
        stats = compute_normalization_stats([_gray(val=100)])
        assert stats["mean"] == pytest.approx(100.0)
