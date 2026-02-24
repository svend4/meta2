"""Extra tests for puzzle_reconstruction/preprocessing/edge_enhancer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.edge_enhancer import (
    EdgeEnhanceParams,
    unsharp_mask,
    laplacian_enhance,
    hybrid_enhance,
    gradient_scale_enhance,
    sharpness_measure,
    apply_edge_enhance,
    batch_edge_enhance,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _edge_image():
    """Image with a sharp edge for testing enhancement."""
    img = np.zeros((50, 50), dtype=np.uint8)
    img[:, 25:] = 255
    return img


# ─── EdgeEnhanceParams ────────────────────────────────────────────────────────

class TestEdgeEnhanceParamsExtra:
    def test_defaults(self):
        p = EdgeEnhanceParams()
        assert p.method == "unsharp"
        assert p.strength == pytest.approx(1.5)
        assert p.blur_sigma == pytest.approx(1.0)
        assert p.kernel_size == 5
        assert p.clip is True

    def test_valid_methods(self):
        for m in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            EdgeEnhanceParams(method=m)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(method="bad")

    def test_zero_strength_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(strength=0.0)

    def test_negative_strength_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(strength=-1.0)

    def test_zero_blur_sigma_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(blur_sigma=0.0)

    def test_small_kernel_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(kernel_size=1)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(kernel_size=4)


# ─── unsharp_mask ─────────────────────────────────────────────────────────────

class TestUnsharpMaskExtra:
    def test_shape_preserved_bgr(self):
        img = _bgr()
        out = unsharp_mask(img)
        assert out.shape == img.shape

    def test_shape_preserved_gray(self):
        img = _gray()
        out = unsharp_mask(img)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        out = unsharp_mask(_bgr())
        assert out.dtype == np.uint8

    def test_zero_strength_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_bgr(), strength=0.0)

    def test_zero_blur_sigma_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_bgr(), blur_sigma=0.0)

    def test_bad_kernel_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_bgr(), kernel_size=2)


# ─── laplacian_enhance ────────────────────────────────────────────────────────

class TestLaplacianEnhanceExtra:
    def test_shape_preserved_bgr(self):
        img = _bgr()
        out = laplacian_enhance(img)
        assert out.shape == img.shape

    def test_shape_preserved_gray(self):
        img = _gray()
        out = laplacian_enhance(img)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        out = laplacian_enhance(_bgr())
        assert out.dtype == np.uint8

    def test_zero_strength_raises(self):
        with pytest.raises(ValueError):
            laplacian_enhance(_bgr(), strength=0.0)


# ─── hybrid_enhance ───────────────────────────────────────────────────────────

class TestHybridEnhanceExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = hybrid_enhance(img)
        assert out.shape == img.shape

    def test_gray(self):
        img = _gray()
        out = hybrid_enhance(img)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        out = hybrid_enhance(_bgr())
        assert out.dtype == np.uint8


# ─── gradient_scale_enhance ───────────────────────────────────────────────────

class TestGradientScaleEnhanceExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = gradient_scale_enhance(img)
        assert out.shape == img.shape

    def test_gray(self):
        img = _gray()
        out = gradient_scale_enhance(img)
        assert out.shape == img.shape

    def test_dtype_uint8(self):
        out = gradient_scale_enhance(_bgr())
        assert out.dtype == np.uint8

    def test_zero_strength_raises(self):
        with pytest.raises(ValueError):
            gradient_scale_enhance(_bgr(), strength=0.0)

    def test_edge_image_enhanced(self):
        img = _edge_image()
        out = gradient_scale_enhance(img, strength=2.0)
        # Near the edge, values should differ from input
        assert not np.array_equal(out, img)


# ─── sharpness_measure ────────────────────────────────────────────────────────

class TestSharpnessMeasureExtra:
    def test_nonnegative(self):
        assert sharpness_measure(_bgr()) >= 0

    def test_edge_sharper_than_flat(self):
        flat = _gray()
        edge = _edge_image()
        assert sharpness_measure(edge) > sharpness_measure(flat)

    def test_gray_input(self):
        val = sharpness_measure(_gray())
        assert isinstance(val, float)

    def test_bgr_input(self):
        val = sharpness_measure(_bgr())
        assert isinstance(val, float)


# ─── apply_edge_enhance ───────────────────────────────────────────────────────

class TestApplyEdgeEnhanceExtra:
    def test_unsharp(self):
        p = EdgeEnhanceParams(method="unsharp")
        out = apply_edge_enhance(_bgr(), p)
        assert out.shape == _bgr().shape

    def test_laplacian(self):
        p = EdgeEnhanceParams(method="laplacian")
        out = apply_edge_enhance(_bgr(), p)
        assert out.shape == _bgr().shape

    def test_hybrid(self):
        p = EdgeEnhanceParams(method="hybrid")
        out = apply_edge_enhance(_bgr(), p)
        assert out.shape == _bgr().shape

    def test_gradient_scale(self):
        p = EdgeEnhanceParams(method="gradient_scale")
        out = apply_edge_enhance(_bgr(), p)
        assert out.shape == _bgr().shape


# ─── batch_edge_enhance ───────────────────────────────────────────────────────

class TestBatchEdgeEnhanceExtra:
    def test_empty(self):
        p = EdgeEnhanceParams()
        assert batch_edge_enhance([], p) == []

    def test_length(self):
        p = EdgeEnhanceParams()
        results = batch_edge_enhance([_bgr(), _bgr()], p)
        assert len(results) == 2

    def test_dtype(self):
        p = EdgeEnhanceParams()
        results = batch_edge_enhance([_bgr()], p)
        assert results[0].dtype == np.uint8
