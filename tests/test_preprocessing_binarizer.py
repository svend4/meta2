"""Tests for puzzle_reconstruction/preprocessing/binarizer.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.binarizer import (
    BinarizeResult,
    binarize_otsu,
    binarize_adaptive,
    binarize_sauvola,
    binarize_niblack,
    binarize_bernsen,
    auto_binarize,
    batch_binarize,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_gray(h=40, w=40, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_bgr(h=40, w=40, value=(100, 150, 200)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = value
    return img


def make_gradient(h=40, w=40):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_checkerboard(h=40, w=40):
    r, c = np.mgrid[:h, :w]
    return (((r + c) % 2) * 255).astype(np.uint8)


# ─── BinarizeResult ───────────────────────────────────────────────────────────

class TestBinarizeResult:
    def test_basic_creation(self):
        binary = np.zeros((10, 10), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="otsu", threshold=128.0)
        assert result.method == "otsu"
        assert result.threshold == pytest.approx(128.0)

    def test_default_inverted(self):
        binary = np.zeros((5, 5), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="test", threshold=0.0)
        assert result.inverted is False

    def test_default_params_empty(self):
        binary = np.zeros((5, 5), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="test", threshold=0.0)
        assert result.params == {}

    def test_foreground_ratio_all_zero(self):
        binary = np.zeros((10, 10), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="test", threshold=0.0)
        assert result.foreground_ratio == pytest.approx(0.0)

    def test_foreground_ratio_all_255(self):
        binary = np.full((10, 10), 255, dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="test", threshold=0.0)
        assert result.foreground_ratio == pytest.approx(1.0)

    def test_foreground_ratio_half(self):
        binary = np.zeros((10, 10), dtype=np.uint8)
        binary[:5, :] = 255
        result = BinarizeResult(binary=binary, method="test", threshold=0.0)
        assert result.foreground_ratio == pytest.approx(0.5)

    def test_custom_params(self):
        binary = np.zeros((5, 5), dtype=np.uint8)
        result = BinarizeResult(binary=binary, method="sauvola",
                                threshold=0.0, params={"k": 0.2})
        assert result.params["k"] == 0.2


# ─── binarize_otsu ────────────────────────────────────────────────────────────

class TestBinarizeOtsu:
    def test_returns_binarize_result(self):
        img = make_gradient()
        result = binarize_otsu(img)
        assert isinstance(result, BinarizeResult)

    def test_method_name(self):
        img = make_gradient()
        result = binarize_otsu(img)
        assert result.method == "otsu"

    def test_output_shape(self):
        img = make_gray(30, 40)
        result = binarize_otsu(img)
        assert result.binary.shape == (30, 40)

    def test_output_dtype_uint8(self):
        img = make_gradient()
        result = binarize_otsu(img)
        assert result.binary.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient()
        result = binarize_otsu(img)
        assert set(np.unique(result.binary)).issubset({0, 255})

    def test_bgr_input(self):
        img = make_bgr()
        result = binarize_otsu(img)
        assert result.binary.ndim == 2

    def test_invert_flag(self):
        img = make_gradient()
        r_normal = binarize_otsu(img, invert=False)
        r_invert = binarize_otsu(img, invert=True)
        assert r_invert.inverted is True
        # Inverted is bitwise_not of normal
        np.testing.assert_array_equal(
            r_invert.binary, 255 - r_normal.binary
        )

    def test_threshold_stored(self):
        img = make_gradient()
        result = binarize_otsu(img)
        assert result.threshold >= 0.0


# ─── binarize_adaptive ────────────────────────────────────────────────────────

class TestBinarizeAdaptive:
    def test_returns_binarize_result(self):
        img = make_gradient()
        result = binarize_adaptive(img)
        assert isinstance(result, BinarizeResult)

    def test_output_shape(self):
        img = make_gradient(50, 50)
        result = binarize_adaptive(img)
        assert result.binary.shape == (50, 50)

    def test_output_dtype_uint8(self):
        img = make_gradient()
        result = binarize_adaptive(img)
        assert result.binary.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient()
        result = binarize_adaptive(img)
        assert set(np.unique(result.binary)).issubset({0, 255})

    def test_method_mean(self):
        img = make_gradient()
        result = binarize_adaptive(img, adaptive_method="mean")
        assert "mean" in result.method

    def test_method_gaussian(self):
        img = make_gradient()
        result = binarize_adaptive(img, adaptive_method="gaussian")
        assert "gaussian" in result.method

    def test_bgr_input(self):
        img = make_bgr()
        result = binarize_adaptive(img)
        assert result.binary.ndim == 2

    def test_invert_flag(self):
        img = make_gradient()
        r_normal = binarize_adaptive(img, invert=False)
        r_invert = binarize_adaptive(img, invert=True)
        assert r_invert.inverted is True

    def test_block_size_stored_in_params(self):
        img = make_gradient()
        result = binarize_adaptive(img, block_size=15)
        assert result.params.get("block_size") is not None


# ─── binarize_sauvola ─────────────────────────────────────────────────────────

class TestBinarizeSauvola:
    def test_returns_binarize_result(self):
        img = make_gradient()
        result = binarize_sauvola(img)
        assert isinstance(result, BinarizeResult)

    def test_method_name(self):
        img = make_gradient()
        result = binarize_sauvola(img)
        assert result.method == "sauvola"

    def test_output_shape(self):
        img = make_gradient(30, 30)
        result = binarize_sauvola(img)
        assert result.binary.shape == (30, 30)

    def test_output_dtype_uint8(self):
        img = make_gradient()
        result = binarize_sauvola(img)
        assert result.binary.dtype == np.uint8

    def test_output_binary(self):
        img = make_checkerboard()
        result = binarize_sauvola(img)
        assert set(np.unique(result.binary)).issubset({0, 255})

    def test_bgr_input(self):
        img = make_bgr()
        result = binarize_sauvola(img)
        assert result.binary.ndim == 2

    def test_invert_flag(self):
        img = make_gradient()
        r_normal = binarize_sauvola(img, invert=False)
        r_invert = binarize_sauvola(img, invert=True)
        assert r_invert.inverted is True

    def test_custom_k_r(self):
        img = make_gradient()
        result = binarize_sauvola(img, k=0.5, r=64.0)
        assert result.binary.dtype == np.uint8


# ─── binarize_niblack ─────────────────────────────────────────────────────────

class TestBinarizeNiblack:
    def test_returns_binarize_result(self):
        img = make_gradient()
        result = binarize_niblack(img)
        assert isinstance(result, BinarizeResult)

    def test_method_name(self):
        img = make_gradient()
        result = binarize_niblack(img)
        assert result.method == "niblack"

    def test_output_shape(self):
        img = make_gradient(25, 25)
        result = binarize_niblack(img)
        assert result.binary.shape == (25, 25)

    def test_output_dtype_uint8(self):
        img = make_gradient()
        result = binarize_niblack(img)
        assert result.binary.dtype == np.uint8

    def test_output_binary(self):
        img = make_checkerboard()
        result = binarize_niblack(img)
        assert set(np.unique(result.binary)).issubset({0, 255})

    def test_bgr_input(self):
        img = make_bgr()
        result = binarize_niblack(img)
        assert result.binary.ndim == 2

    def test_invert_flag(self):
        img = make_gradient()
        r_normal = binarize_niblack(img, invert=False)
        r_invert = binarize_niblack(img, invert=True)
        assert r_invert.inverted is True

    def test_gradient_produces_binary(self):
        img = make_gradient(30, 30)
        result = binarize_niblack(img, window_size=7, k=-0.2)
        assert set(np.unique(result.binary)).issubset({0, 255})


# ─── binarize_bernsen ─────────────────────────────────────────────────────────

class TestBinarizeBernsen:
    def test_returns_binarize_result(self):
        img = make_checkerboard()
        result = binarize_bernsen(img)
        assert isinstance(result, BinarizeResult)

    def test_method_name(self):
        img = make_checkerboard()
        result = binarize_bernsen(img)
        assert result.method == "bernsen"

    def test_output_shape(self):
        img = make_gradient(30, 30)
        result = binarize_bernsen(img)
        assert result.binary.shape == (30, 30)

    def test_output_dtype_uint8(self):
        img = make_checkerboard()
        result = binarize_bernsen(img)
        assert result.binary.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient()
        result = binarize_bernsen(img)
        assert set(np.unique(result.binary)).issubset({0, 255})

    def test_constant_image_all_background(self):
        """Constant image → contrast = 0 < contrast_thresh → all 0."""
        img = make_gray(20, 20, value=128)
        result = binarize_bernsen(img, contrast_thresh=50.0)
        assert (result.binary == 0).all()

    def test_bgr_input(self):
        img = make_bgr()
        result = binarize_bernsen(img)
        assert result.binary.ndim == 2

    def test_invert_flag(self):
        img = make_gradient()
        r_normal = binarize_bernsen(img, invert=False)
        r_invert = binarize_bernsen(img, invert=True)
        assert r_invert.inverted is True


# ─── auto_binarize ────────────────────────────────────────────────────────────

class TestAutoBinarize:
    def test_returns_binarize_result(self):
        img = make_gradient()
        result = auto_binarize(img)
        assert isinstance(result, BinarizeResult)

    def test_output_dtype_uint8(self):
        img = make_gradient()
        result = auto_binarize(img)
        assert result.binary.dtype == np.uint8

    def test_output_binary(self):
        img = make_gradient()
        result = auto_binarize(img)
        assert set(np.unique(result.binary)).issubset({0, 255})

    def test_high_entropy_uses_otsu(self):
        """Large gradient image → high entropy → Otsu."""
        img = np.linspace(0, 255, 256 * 10, dtype=np.uint8).reshape(10, 256)
        result = auto_binarize(img)
        assert result.method == "otsu"

    def test_low_entropy_uses_sauvola(self):
        """Constant image → entropy = 0 → Sauvola."""
        img = make_gray(30, 30, value=128)
        result = auto_binarize(img)
        assert result.method == "sauvola"

    def test_bgr_input(self):
        img = make_bgr()
        result = auto_binarize(img)
        assert result.binary.ndim == 2

    def test_invert_flag(self):
        img = make_gradient()
        r_normal = auto_binarize(img, invert=False)
        r_invert = auto_binarize(img, invert=True)
        assert r_invert.inverted is True


# ─── batch_binarize ───────────────────────────────────────────────────────────

class TestBatchBinarize:
    def test_returns_list(self):
        imgs = [make_gradient() for _ in range(3)]
        results = batch_binarize(imgs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_binarize_result(self):
        imgs = [make_gradient() for _ in range(3)]
        results = batch_binarize(imgs)
        assert all(isinstance(r, BinarizeResult) for r in results)

    def test_empty_list(self):
        results = batch_binarize([], method="otsu")
        assert results == []

    def test_unknown_method_raises(self):
        imgs = [make_gradient()]
        with pytest.raises(ValueError):
            batch_binarize(imgs, method="unknown_method")

    def test_otsu_method(self):
        imgs = [make_gradient() for _ in range(2)]
        results = batch_binarize(imgs, method="otsu")
        assert all(r.method == "otsu" for r in results)

    def test_sauvola_method(self):
        imgs = [make_gradient() for _ in range(2)]
        results = batch_binarize(imgs, method="sauvola")
        assert all(r.method == "sauvola" for r in results)

    def test_adaptive_mean_method(self):
        imgs = [make_gradient() for _ in range(2)]
        results = batch_binarize(imgs, method="adaptive_mean")
        assert all(isinstance(r, BinarizeResult) for r in results)

    def test_niblack_method(self):
        imgs = [make_gradient() for _ in range(2)]
        results = batch_binarize(imgs, method="niblack")
        assert all(r.method == "niblack" for r in results)

    def test_bernsen_method(self):
        imgs = [make_checkerboard() for _ in range(2)]
        results = batch_binarize(imgs, method="bernsen")
        assert all(r.method == "bernsen" for r in results)

    def test_auto_method(self):
        imgs = [make_gradient() for _ in range(2)]
        results = batch_binarize(imgs, method="auto")
        assert all(isinstance(r, BinarizeResult) for r in results)

    def test_shapes_match_inputs(self):
        imgs = [make_gradient(h + 10, 30) for h in range(3)]
        results = batch_binarize(imgs, method="otsu")
        for img, result in zip(imgs, results):
            assert result.binary.shape == img.shape
