"""Additional tests for puzzle_reconstruction/preprocessing/segmentation.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.segmentation import (
    _adaptive_mask,
    _keep_largest_component,
    _morphological_clean,
    _otsu_mask,
    _to_gray,
    segment_fragment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white_rect(h: int = 64, w: int = 64, margin: int = 8) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[margin:h - margin, margin:w - margin] = 200
    return img


def _all_white(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _all_black(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray_2d(h: int = 64, w: int = 64, margin: int = 8) -> np.ndarray:
    g = np.zeros((h, w), dtype=np.uint8)
    g[margin:h - margin, margin:w - margin] = 180
    return g


# ─── TestSegmentFragmentExtra ─────────────────────────────────────────────────

class TestSegmentFragmentExtra:
    def test_large_image_ok(self):
        img = _white_rect(h=256, w=256, margin=20)
        mask = segment_fragment(img)
        assert mask.shape == (256, 256)

    def test_non_square_large(self):
        img = _white_rect(h=128, w=64, margin=10)
        mask = segment_fragment(img)
        assert mask.shape == (128, 64)

    def test_morph_kernel_7_no_crash(self):
        img = _white_rect()
        mask = segment_fragment(img, morph_kernel=7)
        assert mask.dtype == np.uint8

    def test_all_methods_return_same_shape(self):
        img = _white_rect(h=80, w=80)
        for method in ("otsu", "adaptive", "grabcut"):
            mask = segment_fragment(img, method=method)
            assert mask.shape == (80, 80), f"Method {method} returned wrong shape"

    def test_all_white_image_no_crash(self):
        mask = segment_fragment(_all_white())
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_all_black_image_no_crash(self):
        mask = segment_fragment(_all_black())
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_output_values_binary(self):
        for method in ("otsu", "adaptive"):
            mask = segment_fragment(_white_rect(), method=method)
            unique = set(np.unique(mask))
            assert unique.issubset({0, 255}), f"{method}: non-binary values {unique}"

    def test_grabcut_binary_output(self):
        mask = segment_fragment(_white_rect(h=80, w=80), method="grabcut")
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_various_morph_kernels_consistent_shape(self):
        img = _white_rect()
        for k in (1, 3, 5):
            m = segment_fragment(img, morph_kernel=k)
            assert m.shape == (64, 64)


# ─── TestToGrayExtra ──────────────────────────────────────────────────────────

class TestToGrayExtra:
    def test_3channel_to_2d(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = _to_gray(img)
        assert result.ndim == 2
        assert result.shape == (32, 32)

    def test_grayscale_2d_stays_2d(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = _to_gray(img)
        assert result.ndim == 2

    def test_copy_returned_for_2d(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        result = _to_gray(img)
        result[:] = 0
        assert img[0, 0] == 100  # original unchanged

    def test_dtype_preserved_as_uint8(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = _to_gray(img)
        assert result.dtype == np.uint8


# ─── TestOtsuMaskExtra ────────────────────────────────────────────────────────

class TestOtsuMaskExtra:
    def test_binary_values(self):
        gray = _gray_2d()
        mask = _otsu_mask(gray)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_shape_matches(self):
        gray = _gray_2d(48, 80)
        mask = _otsu_mask(gray)
        assert mask.shape == (48, 80)

    def test_dtype_uint8(self):
        mask = _otsu_mask(_gray_2d())
        assert mask.dtype == np.uint8

    def test_all_same_gray_no_crash(self):
        gray = np.full((32, 32), 128, dtype=np.uint8)
        mask = _otsu_mask(gray)
        assert mask.dtype == np.uint8


# ─── TestMorphologicalCleanExtra ──────────────────────────────────────────────

class TestMorphologicalCleanExtra:
    def test_shape_preserved(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        result = _morphological_clean(mask, kernel_size=3)
        assert result.shape == (64, 64)

    def test_dtype_uint8(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        result = _morphological_clean(mask, kernel_size=3)
        assert result.dtype == np.uint8

    def test_kernel_1_no_crash(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 4:28] = 255
        result = _morphological_clean(mask, kernel_size=1)
        assert result.shape == (32, 32)


# ─── TestKeepLargestComponentExtra ────────────────────────────────────────────

class TestKeepLargestComponentExtra:
    def test_single_component_unchanged(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 4:28] = 255
        result = _keep_largest_component(mask)
        assert result.dtype == np.uint8
        assert result.max() == 255

    def test_all_zeros_no_crash(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = _keep_largest_component(mask)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8

    def test_two_components_keeps_larger(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[4:10, 4:10] = 255     # small: 6×6 = 36 px
        mask[20:60, 20:60] = 255   # large: 40×40 = 1600 px
        result = _keep_largest_component(mask)
        # Small component (4:10, 4:10) should be zero
        assert result[5, 5] == 0
        # Large component should be non-zero
        assert result[30, 30] == 255

    def test_shape_and_dtype_preserved(self):
        mask = np.zeros((48, 80), dtype=np.uint8)
        mask[10:40, 10:70] = 255
        result = _keep_largest_component(mask)
        assert result.shape == (48, 80)
        assert result.dtype == np.uint8
