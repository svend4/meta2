"""Extra tests for puzzle_reconstruction/preprocessing/edge_sharpener.py"""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.edge_sharpener import (
    SharpenerConfig,
    SharpenerResult,
    batch_sharpen,
    high_pass_sharpen,
    laplacian_sharpen,
    sharpen_edges,
    sharpen_image,
    unsharp_mask,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=50, w=50):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 130
    img[:, :, 2] = 200
    return img


def _gradient(h=50, w=50):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def _noisy(h=50, w=50, seed=42):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


# ─── TestSharpenerConfigExtra ─────────────────────────────────────────────────

class TestSharpenerConfigExtra:
    def test_ksize_7_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(ksize=7)

    def test_ksize_9_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(ksize=9)

    def test_strength_large(self):
        cfg = SharpenerConfig(strength=10.0)
        assert cfg.strength == pytest.approx(10.0)

    def test_sigma_large(self):
        cfg = SharpenerConfig(sigma=10.0)
        assert cfg.sigma == pytest.approx(10.0)

    def test_sigma_small_positive(self):
        cfg = SharpenerConfig(sigma=0.001)
        assert cfg.sigma == pytest.approx(0.001)

    def test_strength_exactly_zero(self):
        cfg = SharpenerConfig(strength=0.0)
        assert cfg.strength == pytest.approx(0.0)

    def test_ksize_even_raises(self):
        with pytest.raises(ValueError):
            SharpenerConfig(ksize=6)

    def test_all_methods_valid(self):
        for m in ("unsharp", "laplacian", "high_pass"):
            cfg = SharpenerConfig(method=m)
            assert cfg.method == m


# ─── TestSharpenerResultExtra ─────────────────────────────────────────────────

class TestSharpenerResultExtra:
    def test_bgr_shape(self):
        img = _bgr(30, 40)
        r = SharpenerResult(image=img, method="laplacian", strength=1.0)
        assert r.shape == (30, 40, 3)

    def test_dtype_property(self):
        img = _gray()
        r = SharpenerResult(image=img, method="unsharp", strength=1.0)
        assert r.dtype == np.uint8

    def test_params_multiple_keys(self):
        img = _gray()
        r = SharpenerResult(image=img, method="unsharp", strength=1.5,
                            params={"sigma": 2.0, "ksize": 5})
        assert r.params["sigma"] == pytest.approx(2.0)
        assert r.params["ksize"] == 5

    def test_is_sharpened_strength_positive(self):
        img = _gray()
        r = SharpenerResult(image=img, method="laplacian", strength=0.5)
        assert r.is_sharpened is True

    def test_is_sharpened_false_for_zero(self):
        img = _gray()
        r = SharpenerResult(image=img, method="high_pass", strength=0.0)
        assert r.is_sharpened is False

    def test_non_square_shape(self):
        img = _gray(h=30, w=60)
        r = SharpenerResult(image=img, method="unsharp", strength=1.0)
        assert r.shape == (30, 60)


# ─── TestUnsharpMaskExtra ─────────────────────────────────────────────────────

class TestUnsharpMaskExtra:
    def test_non_square_gray(self):
        img = _gradient(h=30, w=60)
        result = unsharp_mask(img)
        assert result.shape == (30, 60)

    def test_large_sigma(self):
        img = _gradient()
        result = unsharp_mask(img, sigma=5.0)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_large_strength(self):
        img = _gradient()
        result = unsharp_mask(img, strength=3.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bgr_non_square(self):
        img = _bgr(h=30, w=60)
        result = unsharp_mask(img)
        assert result.shape == (30, 60, 3)

    def test_all_black_constant(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        result = unsharp_mask(img)
        np.testing.assert_array_equal(result, img)

    def test_all_white_constant(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        result = unsharp_mask(img)
        np.testing.assert_array_equal(result, img)

    def test_noisy_image_shape_preserved(self):
        img = _noisy()
        result = unsharp_mask(img)
        assert result.shape == img.shape

    def test_various_sigma_values(self):
        img = _gradient()
        for s in (0.5, 1.0, 2.0, 4.0):
            result = unsharp_mask(img, sigma=s)
            assert result.dtype == np.uint8


# ─── TestLaplacianSharpenExtra ────────────────────────────────────────────────

class TestLaplacianSharpenExtra:
    def test_non_square_gray(self):
        img = _gradient(h=30, w=60)
        result = laplacian_sharpen(img)
        assert result.shape == (30, 60)

    def test_bgr_non_square(self):
        img = _bgr(h=30, w=60)
        result = laplacian_sharpen(img)
        assert result.shape == (30, 60, 3)

    def test_large_alpha(self):
        img = _gradient()
        result = laplacian_sharpen(img, alpha=5.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_ksize_7_raises(self):
        img = _gradient()
        with pytest.raises(ValueError):
            laplacian_sharpen(img, ksize=7)

    def test_all_black_unchanged(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        result = laplacian_sharpen(img)
        np.testing.assert_array_equal(result, img)

    def test_all_white_unchanged(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        result = laplacian_sharpen(img)
        np.testing.assert_array_equal(result, img)

    def test_valid_ksize_1(self):
        img = _gradient()
        result = laplacian_sharpen(img, ksize=1)
        assert result.dtype == np.uint8

    def test_noisy_dtype_preserved(self):
        img = _noisy()
        result = laplacian_sharpen(img)
        assert result.dtype == np.uint8


# ─── TestHighPassSharpenExtra ─────────────────────────────────────────────────

class TestHighPassSharpenExtra:
    def test_non_square_gray(self):
        img = _gradient(h=30, w=60)
        result = high_pass_sharpen(img)
        assert result.shape == (30, 60)

    def test_bgr_non_square(self):
        img = _bgr(h=30, w=60)
        result = high_pass_sharpen(img)
        assert result.shape == (30, 60, 3)

    def test_large_sigma(self):
        img = _gradient()
        result = high_pass_sharpen(img, sigma=5.0)
        assert result.dtype == np.uint8

    def test_large_strength(self):
        img = _gradient()
        result = high_pass_sharpen(img, strength=5.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_all_black_unchanged(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        result = high_pass_sharpen(img)
        np.testing.assert_array_equal(result, img)

    def test_all_white_unchanged(self):
        img = np.full((50, 50), 255, dtype=np.uint8)
        result = high_pass_sharpen(img)
        np.testing.assert_array_equal(result, img)

    def test_noisy_shape_preserved(self):
        img = _noisy()
        result = high_pass_sharpen(img)
        assert result.shape == img.shape


# ─── TestSharpenImageExtra ────────────────────────────────────────────────────

class TestSharpenImageExtra:
    def test_non_square_all_methods(self):
        img = _gradient(h=30, w=60)
        for method in ("unsharp", "laplacian", "high_pass"):
            cfg = SharpenerConfig(method=method)
            r = sharpen_image(img, cfg)
            assert r.image.shape == (30, 60)

    def test_bgr_all_methods(self):
        img = _bgr()
        for method in ("unsharp", "laplacian", "high_pass"):
            cfg = SharpenerConfig(method=method)
            r = sharpen_image(img, cfg)
            assert r.image.shape == img.shape

    def test_strength_stored_in_result(self):
        img = _gradient()
        cfg = SharpenerConfig(strength=2.0)
        r = sharpen_image(img, cfg)
        assert r.strength == pytest.approx(2.0)

    def test_is_sharpened_for_strength_above_zero(self):
        img = _gradient()
        cfg = SharpenerConfig(strength=0.5)
        r = sharpen_image(img, cfg)
        assert r.is_sharpened is True

    def test_output_in_0_255(self):
        img = _gradient()
        r = sharpen_image(img)
        assert int(r.image.min()) >= 0
        assert int(r.image.max()) <= 255

    def test_constant_black_unchanged(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        r = sharpen_image(img)
        np.testing.assert_array_equal(r.image, img)


# ─── TestSharpenEdgesExtra ────────────────────────────────────────────────────

class TestSharpenEdgesExtra:
    def test_non_square_shape(self):
        img = _gradient(h=30, w=60)
        result = sharpen_edges(img)
        assert result.shape == (30, 60)

    def test_bgr_shape(self):
        img = _bgr()
        result = sharpen_edges(img)
        assert result.shape == img.shape

    def test_large_sigma(self):
        img = _gradient()
        result = sharpen_edges(img, sigma=3.0)
        assert result.dtype == np.uint8

    def test_large_strength(self):
        img = _gradient()
        result = sharpen_edges(img, strength=5.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_full_mask_all_ones_applies_everywhere(self):
        img = _noisy()
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        result = sharpen_edges(img, edge_mask=mask)
        assert result.dtype == np.uint8
        assert result.shape == img.shape

    def test_blank_mask_image_unchanged(self):
        img = _gradient()
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = sharpen_edges(img, edge_mask=mask)
        np.testing.assert_array_equal(result, img)


# ─── TestBatchSharpenExtra ────────────────────────────────────────────────────

class TestBatchSharpenExtra:
    def test_ten_images(self):
        imgs = [_gradient() for _ in range(10)]
        result = batch_sharpen(imgs)
        assert len(result) == 10

    def test_mixed_sizes(self):
        imgs = [_gradient(h=30, w=30), _gradient(h=50, w=60), _gradient(h=64, w=32)]
        result = batch_sharpen(imgs)
        assert result[0].shape == (30, 30)
        assert result[1].shape == (50, 60)
        assert result[2].shape == (64, 32)

    def test_all_uint8(self):
        imgs = [_gradient() for _ in range(5)]
        result = batch_sharpen(imgs)
        assert all(img.dtype == np.uint8 for img in result)

    def test_bgr_images(self):
        imgs = [_bgr() for _ in range(3)]
        result = batch_sharpen(imgs)
        assert all(img.ndim == 3 for img in result)

    def test_laplacian_config(self):
        cfg = SharpenerConfig(method="laplacian", strength=0.5)
        imgs = [_gradient() for _ in range(3)]
        result = batch_sharpen(imgs, cfg=cfg)
        assert len(result) == 3

    def test_high_pass_config(self):
        cfg = SharpenerConfig(method="high_pass", sigma=2.0)
        imgs = [_gradient() for _ in range(2)]
        result = batch_sharpen(imgs, cfg=cfg)
        assert all(img.dtype == np.uint8 for img in result)

    def test_output_in_range(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        result = batch_sharpen(imgs)
        for img in result:
            assert int(img.min()) >= 0
            assert int(img.max()) <= 255

    def test_single_bgr(self):
        result = batch_sharpen([_bgr()])
        assert result[0].ndim == 3
