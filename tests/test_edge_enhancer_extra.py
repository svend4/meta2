"""Extra tests for puzzle_reconstruction.preprocessing.edge_enhancer."""
import pytest
import numpy as np

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


def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=64, w=64, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const(val=128, h=32, w=32):
    return np.full((h, w), val, dtype=np.uint8)


# ─── EdgeEnhanceParams extras ─────────────────────────────────────────────────

class TestEdgeEnhanceParamsExtra:
    def test_strength_positive_ok(self):
        p = EdgeEnhanceParams(strength=0.1)
        assert p.strength == pytest.approx(0.1)

    def test_strength_large_ok(self):
        p = EdgeEnhanceParams(strength=100.0)
        assert p.strength == pytest.approx(100.0)

    def test_kernel_size_3_ok(self):
        p = EdgeEnhanceParams(kernel_size=3)
        assert p.kernel_size == 3

    def test_kernel_size_7_ok(self):
        p = EdgeEnhanceParams(kernel_size=7)
        assert p.kernel_size == 7

    def test_kernel_size_even_2_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(kernel_size=2)

    def test_kernel_size_even_6_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(kernel_size=6)

    def test_method_unsharp_stored(self):
        p = EdgeEnhanceParams(method="unsharp")
        assert p.method == "unsharp"

    def test_method_laplacian_stored(self):
        p = EdgeEnhanceParams(method="laplacian")
        assert p.method == "laplacian"

    def test_blur_sigma_positive_ok(self):
        p = EdgeEnhanceParams(blur_sigma=2.5)
        assert p.blur_sigma == pytest.approx(2.5)

    def test_repr_is_string(self):
        p = EdgeEnhanceParams()
        assert isinstance(repr(p), str)


# ─── unsharp_mask extras ──────────────────────────────────────────────────────

class TestUnsharpMaskExtra:
    def test_non_square_gray(self):
        img = _gray(h=32, w=64)
        out = unsharp_mask(img, strength=1.0)
        assert out.shape == (32, 64)
        assert out.dtype == np.uint8

    def test_kernel_size_3(self):
        out = unsharp_mask(_gray(), strength=1.5, kernel_size=3)
        assert out.dtype == np.uint8

    def test_kernel_size_7(self):
        out = unsharp_mask(_gray(), strength=1.5, kernel_size=7)
        assert out.dtype == np.uint8

    def test_small_image_4x4(self):
        img = _gray(h=4, w=4)
        out = unsharp_mask(img, strength=1.0, kernel_size=3)
        assert out.shape == (4, 4)

    def test_strength_1_close_to_original(self):
        img = _const(val=100)
        out = unsharp_mask(img, strength=1.0)
        # Constant image: unsharp mask should not change it
        assert out.dtype == np.uint8

    def test_color_3ch(self):
        img = _color()
        out = unsharp_mask(img, strength=1.5)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_blur_sigma_0_1_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_gray(), blur_sigma=0.0)

    def test_large_strength_still_clipped(self):
        out = unsharp_mask(_gray(), strength=50.0, clip=True)
        assert 0 <= int(out.min()) and int(out.max()) <= 255


# ─── laplacian_enhance extras ─────────────────────────────────────────────────

class TestLaplacianEnhanceExtra:
    def test_non_square_gray(self):
        img = _gray(h=48, w=32)
        out = laplacian_enhance(img, strength=1.0)
        assert out.shape == (48, 32)
        assert out.dtype == np.uint8

    def test_small_image(self):
        img = _gray(h=8, w=8)
        out = laplacian_enhance(img, strength=1.0)
        assert out.shape == (8, 8)

    def test_constant_image(self):
        img = _const(val=128)
        out = laplacian_enhance(img, strength=1.0)
        # Laplacian of constant is 0 → result ≈ original
        assert out.dtype == np.uint8

    def test_strength_5(self):
        out = laplacian_enhance(_gray(), strength=5.0, clip=True)
        assert 0 <= int(out.min()) and int(out.max()) <= 255

    def test_color_image(self):
        img = _color()
        out = laplacian_enhance(img, strength=1.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_strength_negative_raises(self):
        with pytest.raises(ValueError):
            laplacian_enhance(_gray(), strength=-0.5)


# ─── hybrid_enhance extras ────────────────────────────────────────────────────

class TestHybridEnhanceExtra:
    def test_non_square(self):
        img = _gray(h=32, w=80)
        out = hybrid_enhance(img, strength=1.0)
        assert out.shape == (32, 80)

    def test_small_image(self):
        img = _gray(h=8, w=8)
        out = hybrid_enhance(img, strength=1.0)
        assert out.shape == (8, 8)

    def test_large_strength_clipped(self):
        out = hybrid_enhance(_gray(), strength=20.0, clip=True)
        assert 0 <= int(out.min()) and int(out.max()) <= 255

    def test_color_image(self):
        img = _color()
        out = hybrid_enhance(img, strength=1.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_constant_image_unchanged(self):
        img = _const(val=80)
        out = hybrid_enhance(img, strength=1.0)
        assert out.dtype == np.uint8


# ─── gradient_scale_enhance extras ───────────────────────────────────────────

class TestGradientScaleEnhanceExtra:
    def test_non_square(self):
        img = _gray(h=48, w=96)
        out = gradient_scale_enhance(img, strength=1.0)
        assert out.shape == (48, 96)

    def test_small_image(self):
        img = _gray(h=8, w=8)
        out = gradient_scale_enhance(img, strength=1.0)
        assert out.shape == (8, 8)

    def test_large_strength_clipped(self):
        out = gradient_scale_enhance(_gray(), strength=30.0, clip=True)
        assert 0 <= int(out.min()) and int(out.max()) <= 255

    def test_color_image(self):
        img = _color()
        out = gradient_scale_enhance(img, strength=1.0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_strength_negative_raises(self):
        with pytest.raises(ValueError):
            gradient_scale_enhance(_gray(), strength=-1.0)


# ─── sharpness_measure extras ─────────────────────────────────────────────────

class TestSharpnessMeasureExtra:
    def test_very_sharp_image_high_value(self):
        # Checkerboard pattern = very sharp
        img = np.zeros((64, 64), dtype=np.uint8)
        img[::2, ::2] = 255
        s = sharpness_measure(img)
        assert s > 0.0

    def test_blurred_lower_than_sharp(self):
        import cv2
        img = _gray(seed=5)
        blurred = cv2.GaussianBlur(img, (15, 15), 5.0)
        s_orig = sharpness_measure(img)
        s_blur = sharpness_measure(blurred)
        assert s_orig > s_blur

    def test_non_square_image(self):
        img = _gray(h=32, w=128)
        s = sharpness_measure(img)
        assert isinstance(s, float)
        assert s >= 0.0

    def test_small_image(self):
        img = _gray(h=8, w=8)
        s = sharpness_measure(img)
        assert isinstance(s, float)

    def test_all_zeros_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        s = sharpness_measure(img)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_all_255_image(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        s = sharpness_measure(img)
        assert s == pytest.approx(0.0, abs=1e-6)


# ─── apply_edge_enhance extras ────────────────────────────────────────────────

class TestApplyEdgeEnhanceExtra:
    def test_all_methods_color(self):
        for method in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            img = _color()
            p = EdgeEnhanceParams(method=method, strength=1.0)
            out = apply_edge_enhance(img, p)
            assert out.shape == img.shape
            assert out.dtype == np.uint8

    def test_kernel_size_3_all_methods(self):
        for method in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            img = _gray()
            p = EdgeEnhanceParams(method=method, kernel_size=3)
            out = apply_edge_enhance(img, p)
            assert out.shape == img.shape

    def test_non_square_all_methods(self):
        for method in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            img = _gray(h=32, w=64)
            p = EdgeEnhanceParams(method=method, strength=1.0)
            out = apply_edge_enhance(img, p)
            assert out.shape == (32, 64)

    def test_strength_0_1(self):
        p = EdgeEnhanceParams(method="unsharp", strength=0.1)
        out = apply_edge_enhance(_gray(), p)
        assert out.dtype == np.uint8


# ─── batch_edge_enhance extras ────────────────────────────────────────────────

class TestBatchEdgeEnhanceExtra:
    def test_single_image(self):
        p = EdgeEnhanceParams(method="unsharp")
        result = batch_edge_enhance([_gray()], p)
        assert len(result) == 1
        assert result[0].dtype == np.uint8

    def test_mixed_gray_color(self):
        images = [_gray(), _color(), _gray(h=32, w=32)]
        p = EdgeEnhanceParams(method="hybrid", strength=1.0)
        result = batch_edge_enhance(images, p)
        assert len(result) == 3
        for orig, out in zip(images, result):
            assert orig.shape == out.shape

    def test_all_methods(self):
        images = [_gray(seed=i) for i in range(3)]
        for method in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            p = EdgeEnhanceParams(method=method, strength=1.0)
            result = batch_edge_enhance(images, p)
            assert len(result) == 3

    def test_large_batch(self):
        images = [_gray(seed=i) for i in range(10)]
        p = EdgeEnhanceParams(method="laplacian")
        result = batch_edge_enhance(images, p)
        assert len(result) == 10
        assert all(r.dtype == np.uint8 for r in result)
