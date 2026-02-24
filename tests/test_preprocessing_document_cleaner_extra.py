"""Extra tests for puzzle_reconstruction/preprocessing/document_cleaner.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.document_cleaner import (
    CleanResult,
    remove_shadow,
    remove_border_artifacts,
    normalize_illumination,
    remove_blobs,
    auto_clean,
    batch_clean,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


# ─── CleanResult ──────────────────────────────────────────────────────────────

class TestCleanResultExtra:
    def test_fields(self):
        r = CleanResult(cleaned=_gray(), method="shadow",
                        artifacts_removed=3)
        assert r.method == "shadow"
        assert r.artifacts_removed == 3

    def test_repr(self):
        r = CleanResult(cleaned=_bgr(), method="blobs",
                        artifacts_removed=5)
        s = repr(r)
        assert "blobs" in s
        assert "artifacts=5" in s

    def test_default_artifacts(self):
        r = CleanResult(cleaned=_gray(), method="shadow")
        assert r.artifacts_removed == 0

    def test_default_params(self):
        r = CleanResult(cleaned=_gray(), method="shadow")
        assert r.params == {}


# ─── remove_shadow ────────────────────────────────────────────────────────────

class TestRemoveShadowExtra:
    def test_bgr_shape(self):
        img = _bgr()
        r = remove_shadow(img)
        assert r.cleaned.shape == img.shape
        assert r.method == "shadow"

    def test_gray_shape(self):
        img = _gray()
        r = remove_shadow(img)
        assert r.cleaned.shape == img.shape

    def test_dtype_uint8(self):
        r = remove_shadow(_bgr())
        assert r.cleaned.dtype == np.uint8

    def test_params_populated(self):
        r = remove_shadow(_bgr(), block_size=21, c=5.0)
        assert "block_size" in r.params
        assert "c" in r.params


# ─── remove_border_artifacts ──────────────────────────────────────────────────

class TestRemoveBorderArtifactsExtra:
    def test_shape_preserved(self):
        img = _bgr()
        r = remove_border_artifacts(img, border_px=3)
        assert r.cleaned.shape == img.shape
        assert r.method == "border"

    def test_zero_border(self):
        img = _bgr()
        r = remove_border_artifacts(img, border_px=0)
        assert np.array_equal(r.cleaned, img)

    def test_gray(self):
        img = _gray()
        r = remove_border_artifacts(img, border_px=2)
        assert r.cleaned.ndim == 2

    def test_border_filled(self):
        img = _gray(h=20, w=20, val=100)
        r = remove_border_artifacts(img, border_px=2, fill=255)
        # Top border should be filled
        assert np.all(r.cleaned[:2, :] == 255)

    def test_params(self):
        r = remove_border_artifacts(_bgr(), border_px=5, fill=200)
        assert r.params["border_px"] == 5
        assert r.params["fill"] == 200


# ─── normalize_illumination ──────────────────────────────────────────────────

class TestNormalizeIlluminationExtra:
    def test_shape_preserved(self):
        img = _bgr()
        r = normalize_illumination(img)
        assert r.cleaned.shape == img.shape
        assert r.method == "illumination"

    def test_gray(self):
        img = _gray()
        r = normalize_illumination(img)
        assert r.cleaned.ndim == 2

    def test_dtype_uint8(self):
        r = normalize_illumination(_bgr())
        assert r.cleaned.dtype == np.uint8

    def test_params(self):
        r = normalize_illumination(_bgr(), sigma=30.0)
        assert r.params["sigma"] == pytest.approx(30.0)


# ─── remove_blobs ─────────────────────────────────────────────────────────────

class TestRemoveBlobsExtra:
    def test_shape_preserved(self):
        img = _bgr()
        r = remove_blobs(img)
        assert r.cleaned.shape == img.shape
        assert r.method == "blobs"

    def test_gray(self):
        img = _gray()
        r = remove_blobs(img)
        assert r.cleaned.ndim == 2

    def test_artifacts_count_nonneg(self):
        r = remove_blobs(_bgr())
        assert r.artifacts_removed >= 0

    def test_white_image_no_artifacts(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        r = remove_blobs(img)
        assert r.artifacts_removed == 0


# ─── auto_clean ──────────────────────────────────────────────────────────────

class TestAutoCleanExtra:
    def test_shape_preserved(self):
        img = _bgr()
        r = auto_clean(img)
        assert r.cleaned.shape == img.shape
        assert r.method == "auto"

    def test_gray(self):
        img = _gray()
        r = auto_clean(img)
        assert r.cleaned.ndim == 2

    def test_params(self):
        r = auto_clean(_bgr(), shadow_block=31, illum_sigma=20.0)
        assert r.params["shadow_block"] == 31
        assert r.params["illum_sigma"] == pytest.approx(20.0)

    def test_dtype_uint8(self):
        r = auto_clean(_bgr())
        assert r.cleaned.dtype == np.uint8


# ─── batch_clean ──────────────────────────────────────────────────────────────

class TestBatchCleanExtra:
    def test_empty(self):
        assert batch_clean([]) == []

    def test_length(self):
        results = batch_clean([_bgr(), _bgr()])
        assert len(results) == 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_clean([_bgr()], method="unknown")

    def test_all_methods(self):
        for method in ("shadow", "border", "illumination", "blobs", "auto"):
            results = batch_clean([_bgr()], method=method)
            assert len(results) == 1
            assert results[0].method == method

    def test_result_types(self):
        results = batch_clean([_bgr()])
        assert isinstance(results[0], CleanResult)
