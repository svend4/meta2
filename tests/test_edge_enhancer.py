"""Тесты для puzzle_reconstruction.preprocessing.edge_enhancer."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _color(h=64, w=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── TestEdgeEnhanceParams ────────────────────────────────────────────────────

class TestEdgeEnhanceParams:
    def test_default_values(self):
        p = EdgeEnhanceParams()
        assert p.method == "unsharp"
        assert p.strength == pytest.approx(1.5)
        assert p.kernel_size == 5

    def test_valid_methods(self):
        for m in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            p = EdgeEnhanceParams(method=m)
            assert p.method == m

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(method="sobel_only")

    def test_strength_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(strength=0.0)

    def test_negative_strength_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(strength=-1.0)

    def test_blur_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(blur_sigma=0.0)

    def test_kernel_size_less_than_3_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(kernel_size=1)

    def test_even_kernel_size_raises(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(kernel_size=4)


# ─── TestUnsharpMask ──────────────────────────────────────────────────────────

class TestUnsharpMask:
    def test_output_shape_gray(self):
        img = _gray()
        out = unsharp_mask(img, strength=1.5, blur_sigma=1.0, kernel_size=5)
        assert out.shape == img.shape

    def test_output_shape_color(self):
        img = _color()
        out = unsharp_mask(img, strength=1.5, blur_sigma=1.0, kernel_size=5)
        assert out.shape == img.shape

    def test_output_dtype_uint8(self):
        out = unsharp_mask(_gray(), strength=1.5)
        assert out.dtype == np.uint8

    def test_clipped_values_in_range(self):
        out = unsharp_mask(_gray(), strength=10.0, clip=True)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_strength_zero_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_gray(), strength=0.0)

    def test_blur_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_gray(), blur_sigma=0.0)

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(_gray(), kernel_size=4)

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            unsharp_mask(np.ones((4, 4, 3, 2), dtype=np.uint8))

    def test_constant_image_unchanged(self):
        img = np.full((32, 32), 100, dtype=np.uint8)
        out = unsharp_mask(img, strength=2.0, blur_sigma=1.0, kernel_size=5)
        # Постоянное изображение: blur(img) == img → маска = 0 → нет изменений
        np.testing.assert_array_equal(out, img)


# ─── TestLaplacianEnhance ─────────────────────────────────────────────────────

class TestLaplacianEnhance:
    def test_output_shape_gray(self):
        out = laplacian_enhance(_gray(), strength=1.0)
        assert out.shape == _gray().shape

    def test_output_shape_color(self):
        img = _color()
        out = laplacian_enhance(img, strength=1.0)
        assert out.shape == img.shape

    def test_output_dtype_uint8(self):
        out = laplacian_enhance(_gray(), strength=1.0)
        assert out.dtype == np.uint8

    def test_clipped_values_in_range(self):
        out = laplacian_enhance(_gray(), strength=10.0, clip=True)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_strength_zero_raises(self):
        with pytest.raises(ValueError):
            laplacian_enhance(_gray(), strength=0.0)

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            laplacian_enhance(np.ones((4, 4, 3, 2), dtype=np.uint8))


# ─── TestHybridEnhance ────────────────────────────────────────────────────────

class TestHybridEnhance:
    def test_output_shape_gray(self):
        out = hybrid_enhance(_gray(), strength=1.0)
        assert out.shape == _gray().shape

    def test_output_shape_color(self):
        img = _color()
        out = hybrid_enhance(img, strength=1.0)
        assert out.shape == img.shape

    def test_output_dtype_uint8(self):
        out = hybrid_enhance(_gray(), strength=1.0)
        assert out.dtype == np.uint8

    def test_clipped_in_range(self):
        out = hybrid_enhance(_gray(), strength=5.0, clip=True)
        assert out.min() >= 0
        assert out.max() <= 255


# ─── TestGradientScaleEnhance ─────────────────────────────────────────────────

class TestGradientScaleEnhance:
    def test_output_shape_gray(self):
        out = gradient_scale_enhance(_gray(), strength=1.0)
        assert out.shape == _gray().shape

    def test_output_shape_color(self):
        img = _color()
        out = gradient_scale_enhance(img, strength=1.0)
        assert out.shape == img.shape

    def test_output_dtype_uint8(self):
        out = gradient_scale_enhance(_gray(), strength=1.0)
        assert out.dtype == np.uint8

    def test_clipped_in_range(self):
        out = gradient_scale_enhance(_gray(), strength=10.0, clip=True)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_strength_zero_raises(self):
        with pytest.raises(ValueError):
            gradient_scale_enhance(_gray(), strength=0.0)

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            gradient_scale_enhance(np.ones((4, 4, 3, 2), dtype=np.uint8))


# ─── TestSharpnessMeasure ─────────────────────────────────────────────────────

class TestSharpnessMeasure:
    def test_returns_float(self):
        s = sharpness_measure(_gray())
        assert isinstance(s, float)

    def test_nonnegative(self):
        s = sharpness_measure(_gray())
        assert s >= 0.0

    def test_constant_image_zero_sharpness(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        s = sharpness_measure(img)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_random_image_nonzero(self):
        s = sharpness_measure(_gray())
        assert s > 0.0

    def test_color_image(self):
        s = sharpness_measure(_color())
        assert isinstance(s, float)
        assert s >= 0.0

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            sharpness_measure(np.ones((4, 4, 3, 2), dtype=np.uint8))


# ─── TestApplyEdgeEnhance ─────────────────────────────────────────────────────

class TestApplyEdgeEnhance:
    def _apply(self, method):
        p = EdgeEnhanceParams(method=method, strength=1.0, kernel_size=5)
        return apply_edge_enhance(_gray(), p)

    def test_unsharp(self):
        out = self._apply("unsharp")
        assert out.dtype == np.uint8

    def test_laplacian(self):
        out = self._apply("laplacian")
        assert out.dtype == np.uint8

    def test_hybrid(self):
        out = self._apply("hybrid")
        assert out.dtype == np.uint8

    def test_gradient_scale(self):
        out = self._apply("gradient_scale")
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        img = _gray()
        p = EdgeEnhanceParams(method="unsharp")
        out = apply_edge_enhance(img, p)
        assert out.shape == img.shape


# ─── TestBatchEdgeEnhance ─────────────────────────────────────────────────────

class TestBatchEdgeEnhance:
    def test_returns_list(self):
        images = [_gray() for _ in range(3)]
        p = EdgeEnhanceParams(method="unsharp", strength=1.0)
        result = batch_edge_enhance(images, p)
        assert isinstance(result, list)

    def test_correct_length(self):
        images = [_gray() for _ in range(4)]
        p = EdgeEnhanceParams(method="laplacian", strength=1.0)
        result = batch_edge_enhance(images, p)
        assert len(result) == 4

    def test_empty_list(self):
        p = EdgeEnhanceParams()
        result = batch_edge_enhance([], p)
        assert result == []

    def test_each_uint8(self):
        images = [_gray(), _color()]
        p = EdgeEnhanceParams(method="gradient_scale", strength=1.0)
        result = batch_edge_enhance(images, p)
        assert all(r.dtype == np.uint8 for r in result)

    def test_shapes_preserved(self):
        images = [_gray(32, 32), _gray(48, 64)]
        p = EdgeEnhanceParams(method="hybrid", strength=1.0)
        result = batch_edge_enhance(images, p)
        for orig, res in zip(images, result):
            assert orig.shape == res.shape
