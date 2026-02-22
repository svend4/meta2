"""Tests for puzzle_reconstruction/preprocessing/edge_sharpener.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.edge_sharpener import (
    SharpenerConfig,
    SharpenerResult,
    unsharp_mask,
    laplacian_sharpen,
    high_pass_sharpen,
    sharpen_image,
    sharpen_edges,
    batch_sharpen,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=50, w=50, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr(h=50, w=50):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 130
    img[:, :, 2] = 200
    return img


def make_gradient(h=50, w=50):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_binary_edge(h=50, w=50):
    em = np.zeros((h, w), dtype=np.uint8)
    em[h // 2, :] = 255
    return em


# ─── SharpenerConfig ──────────────────────────────────────────────────────────

class TestSharpenerConfig:
    def test_defaults(self):
        cfg = SharpenerConfig()
        assert cfg.method == "unsharp"
        assert cfg.strength == pytest.approx(1.0)
        assert cfg.sigma == pytest.approx(1.0)
        assert cfg.ksize == 3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(method="unknown")

    def test_valid_methods(self):
        for m in ("unsharp", "laplacian", "high_pass"):
            cfg = SharpenerConfig(method=m)
            assert cfg.method == m

    def test_negative_strength_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(strength=-0.1)

    def test_zero_strength_valid(self):
        cfg = SharpenerConfig(strength=0.0)
        assert cfg.strength == pytest.approx(0.0)

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(sigma=-1.0)

    def test_invalid_ksize_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(ksize=4)

    def test_valid_ksizes(self):
        for k in (1, 3, 5):
            cfg = SharpenerConfig(ksize=k)
            assert cfg.ksize == k


# ─── SharpenerResult ──────────────────────────────────────────────────────────

class TestSharpenerResult:
    def test_basic_creation(self):
        img = make_gray()
        r = SharpenerResult(image=img, method="unsharp", strength=1.0)
        assert r.method == "unsharp"
        assert r.strength == pytest.approx(1.0)

    def test_negative_strength_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            SharpenerResult(image=img, method="unsharp", strength=-0.5)

    def test_shape_property(self):
        img = make_gray(30, 40)
        r = SharpenerResult(image=img, method="unsharp", strength=1.0)
        assert r.shape == (30, 40)

    def test_dtype_property(self):
        img = make_gray()
        r = SharpenerResult(image=img, method="unsharp", strength=1.0)
        assert r.dtype == np.uint8

    def test_is_sharpened_true(self):
        img = make_gray()
        r = SharpenerResult(image=img, method="unsharp", strength=1.0)
        assert r.is_sharpened is True

    def test_is_sharpened_false_when_zero(self):
        img = make_gray()
        r = SharpenerResult(image=img, method="unsharp", strength=0.0)
        assert r.is_sharpened is False

    def test_params_stored(self):
        img = make_gray()
        r = SharpenerResult(image=img, method="unsharp", strength=1.0,
                            params={"sigma": 2.0})
        assert r.params["sigma"] == pytest.approx(2.0)


# ─── unsharp_mask ─────────────────────────────────────────────────────────────

class TestUnsharpMask:
    def test_sigma_zero_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            unsharp_mask(img, sigma=0.0)

    def test_sigma_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            unsharp_mask(img, sigma=-1.0)

    def test_strength_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            unsharp_mask(img, strength=-0.1)

    def test_strength_zero_returns_copy(self):
        img = make_gradient()
        result = unsharp_mask(img, strength=0.0)
        np.testing.assert_array_equal(result, img)

    def test_returns_uint8(self):
        img = make_gradient()
        result = unsharp_mask(img)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = make_gradient(h=30, w=40)
        result = unsharp_mask(img)
        assert result.shape == (30, 40)

    def test_bgr_input(self):
        img = make_bgr()
        result = unsharp_mask(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_values_clipped_in_range(self):
        img = make_gradient()
        result = unsharp_mask(img, strength=5.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image_unchanged(self):
        img = make_gray(value=128)
        result = unsharp_mask(img)
        np.testing.assert_array_equal(result, img)


# ─── laplacian_sharpen ────────────────────────────────────────────────────────

class TestLaplacianSharpen:
    def test_invalid_ksize_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            laplacian_sharpen(img, ksize=4)

    def test_negative_alpha_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            laplacian_sharpen(img, alpha=-0.1)

    def test_alpha_zero_returns_copy(self):
        img = make_gradient()
        result = laplacian_sharpen(img, alpha=0.0)
        np.testing.assert_array_equal(result, img)

    def test_returns_uint8(self):
        img = make_gradient()
        result = laplacian_sharpen(img)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = make_gradient(h=35, w=45)
        result = laplacian_sharpen(img)
        assert result.shape == (35, 45)

    def test_bgr_input(self):
        img = make_bgr()
        result = laplacian_sharpen(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_values_in_range(self):
        img = make_gradient()
        result = laplacian_sharpen(img, alpha=2.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_valid_ksizes(self):
        img = make_gradient()
        for k in (1, 3, 5):
            result = laplacian_sharpen(img, ksize=k)
            assert result.dtype == np.uint8


# ─── high_pass_sharpen ────────────────────────────────────────────────────────

class TestHighPassSharpen:
    def test_sigma_zero_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_pass_sharpen(img, sigma=0.0)

    def test_sigma_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_pass_sharpen(img, sigma=-1.0)

    def test_strength_negative_raises(self):
        img = make_gray()
        with pytest.raises(ValueError):
            high_pass_sharpen(img, strength=-0.5)

    def test_strength_zero_returns_copy(self):
        img = make_gradient()
        result = high_pass_sharpen(img, strength=0.0)
        np.testing.assert_array_equal(result, img)

    def test_returns_uint8(self):
        img = make_gradient()
        result = high_pass_sharpen(img)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = make_gradient(h=40, w=60)
        result = high_pass_sharpen(img)
        assert result.shape == (40, 60)

    def test_bgr_input(self):
        img = make_bgr()
        result = high_pass_sharpen(img)
        assert result.shape == img.shape

    def test_values_in_range(self):
        img = make_gradient()
        result = high_pass_sharpen(img, strength=3.0)
        assert result.min() >= 0
        assert result.max() <= 255


# ─── sharpen_image ────────────────────────────────────────────────────────────

class TestSharpenImage:
    def test_returns_sharpener_result(self):
        img = make_gradient()
        r = sharpen_image(img)
        assert isinstance(r, SharpenerResult)

    def test_default_config(self):
        img = make_gradient()
        r = sharpen_image(img)
        assert r.method == "unsharp"

    def test_method_unsharp(self):
        img = make_gradient()
        cfg = SharpenerConfig(method="unsharp")
        r = sharpen_image(img, cfg)
        assert r.method == "unsharp"

    def test_method_laplacian(self):
        img = make_gradient()
        cfg = SharpenerConfig(method="laplacian")
        r = sharpen_image(img, cfg)
        assert r.method == "laplacian"

    def test_method_high_pass(self):
        img = make_gradient()
        cfg = SharpenerConfig(method="high_pass")
        r = sharpen_image(img, cfg)
        assert r.method == "high_pass"

    def test_strength_zero_returns_copy(self):
        img = make_gradient()
        cfg = SharpenerConfig(strength=0.0)
        r = sharpen_image(img, cfg)
        np.testing.assert_array_equal(r.image, img)
        assert not r.is_sharpened

    def test_output_dtype_uint8(self):
        img = make_gradient()
        r = sharpen_image(img)
        assert r.image.dtype == np.uint8

    def test_shape_preserved(self):
        img = make_gradient(h=30, w=40)
        r = sharpen_image(img)
        assert r.image.shape == (30, 40)

    def test_bgr_input(self):
        img = make_bgr()
        r = sharpen_image(img)
        assert r.image.shape == img.shape


# ─── sharpen_edges ────────────────────────────────────────────────────────────

class TestSharpenEdges:
    def test_sigma_zero_raises(self):
        img = make_gradient()
        with pytest.raises(ValueError):
            sharpen_edges(img, sigma=0.0)

    def test_strength_negative_raises(self):
        img = make_gradient()
        with pytest.raises(ValueError):
            sharpen_edges(img, strength=-1.0)

    def test_returns_uint8(self):
        img = make_gradient()
        result = sharpen_edges(img)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        img = make_gradient(h=40, w=60)
        result = sharpen_edges(img)
        assert result.shape == (40, 60)

    def test_with_blank_mask_unchanged(self):
        img = make_gradient()
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = sharpen_edges(img, edge_mask=mask)
        np.testing.assert_array_equal(result, img)

    def test_with_provided_mask(self):
        img = make_gradient()
        mask = make_binary_edge()
        result = sharpen_edges(img, edge_mask=mask)
        assert result.dtype == np.uint8

    def test_no_mask_auto_canny(self):
        img = make_gradient()
        result = sharpen_edges(img)
        assert result.dtype == np.uint8


# ─── batch_sharpen ────────────────────────────────────────────────────────────

class TestBatchSharpen:
    def test_empty_list(self):
        result = batch_sharpen([])
        assert result == []

    def test_returns_list(self):
        imgs = [make_gradient() for _ in range(3)]
        result = batch_sharpen(imgs)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_all_uint8(self):
        imgs = [make_gradient() for _ in range(3)]
        result = batch_sharpen(imgs)
        assert all(img.dtype == np.uint8 for img in result)

    def test_shapes_preserved(self):
        imgs = [make_gradient(h=30, w=40) for _ in range(2)]
        result = batch_sharpen(imgs)
        assert all(img.shape == (30, 40) for img in result)

    def test_custom_config(self):
        cfg = SharpenerConfig(method="laplacian", strength=0.5)
        imgs = [make_gradient() for _ in range(2)]
        result = batch_sharpen(imgs, cfg=cfg)
        assert len(result) == 2
