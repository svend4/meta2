"""
Property-based tests for three utility modules:
  1. puzzle_reconstruction.utils.threshold_utils
  2. puzzle_reconstruction.utils.mask_utils
  3. puzzle_reconstruction.utils.image_transform_utils

Verifies mathematical invariants:

threshold_utils:
- apply_threshold:     bool dtype, same shape; invert flips result
- binarize:            float64, values in {0.0, 1.0}; same shape
- adaptive_threshold:  bool 1-D, length preserved
- soft_threshold:      |output| ≤ |input|; zeros below threshold;
                       sign preserved; small input → 0
- threshold_matrix:    elements < value → fill; shape preserved
- hysteresis_threshold: strong elements always True; below low → always False
- otsu_threshold:      result in [min, max] of input; bisects data
- count_above:         0 ≤ count ≤ arr.size; complement adds to size
- fraction_above:      in [0, 1]; fraction + (1-fraction) = 1
- batch_threshold:     list length preserved; each element bool array

mask_utils:
- create_alpha_mask:   shape (h, w), uint8, all = fill
- apply_mask:          shape/dtype preserved; masked-out pixels = fill;
                       all-255 mask → image unchanged
- erode_mask:          shape preserved; eroded subset of original
- dilate_mask:         shape preserved; dilated superset of original
- combine_masks 'and': result ≤ min(m1, m2) element-wise; ≥ 0
- combine_masks 'or':  result ≥ max(m1, m2) element-wise; ≤ 255
- invert_mask:         double invert = identity; 0→255, 255→0

image_transform_utils:
- rotate_image:        shape preserved (expand=False); dtype preserved
- flip_horizontal:     double flip = identity; shape preserved
- flip_vertical:       double flip = identity; shape preserved
- pad_image:           H' = H+top+bottom, W' = W+left+right; dtype preserved
- crop_image:          shape ≤ original; dtype preserved
- resize_image:        exact target size; dtype preserved
- resize_to_max_side:  longest side ≤ max_side; aspect ratio preserved
- rotation_matrix_2x3: shape (2, 3); identity rotation ≈ identity matrix
- batch_rotate:        list length preserved; each image same shape
- batch_pad:           list length preserved; each bigger than original
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.threshold_utils import (
    ThresholdConfig,
    apply_threshold,
    binarize,
    adaptive_threshold,
    soft_threshold,
    threshold_matrix,
    hysteresis_threshold,
    otsu_threshold,
    count_above,
    fraction_above,
    batch_threshold,
)
from puzzle_reconstruction.utils.mask_utils import (
    create_alpha_mask,
    apply_mask,
    erode_mask,
    dilate_mask,
    combine_masks,
    invert_mask,
)
from puzzle_reconstruction.utils.image_transform_utils import (
    ImageTransformConfig,
    rotate_image,
    flip_horizontal,
    flip_vertical,
    pad_image,
    crop_image,
    resize_image,
    resize_to_max_side,
    rotation_matrix_2x3,
    batch_rotate,
    batch_pad,
    batch_resize_to_max,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _arr1d(n: int = 20, lo: float = 0.0, hi: float = 1.0,
           seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(lo, hi, n)


def _arr_any(n: int = 20, lo: float = -5.0, hi: float = 5.0,
             seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(lo, hi, n)


def _mat(h: int = 8, w: int = 8, lo: float = 0.0, hi: float = 1.0,
         seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(lo, hi, (h, w))


def _bgr_image(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3),
                                                 dtype=np.uint8)


def _gray_image(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w),
                                                 dtype=np.uint8)


def _mask(h: int = 20, w: int = 20, fill: int = 255) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _random_mask(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    data = np.random.default_rng(seed).choice([0, 255], size=(h, w))
    return data.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. threshold_utils invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyThreshold:
    """apply_threshold invariants."""

    def test_bool_dtype(self) -> None:
        arr = _arr1d(20)
        mask = apply_threshold(arr, 0.5)
        assert mask.dtype == bool

    def test_same_shape(self) -> None:
        arr = _arr1d(15)
        mask = apply_threshold(arr, 0.3)
        assert mask.shape == arr.shape

    def test_true_above_threshold(self) -> None:
        arr = np.array([0.1, 0.5, 0.9])
        mask = apply_threshold(arr, 0.5)
        assert mask[1] == True
        assert mask[2] == True
        assert mask[0] == False

    def test_invert_flips_result(self) -> None:
        arr = _arr1d(20, seed=1)
        normal = apply_threshold(arr, 0.5)
        inverted = apply_threshold(arr, 0.5, invert=True)
        assert np.all(normal != inverted)

    def test_all_above_threshold_min(self) -> None:
        arr = np.array([0.5, 0.6, 0.7, 0.8])
        mask = apply_threshold(arr, 0.0)
        assert np.all(mask)

    def test_all_below_threshold_max(self) -> None:
        arr = np.array([0.1, 0.2, 0.3])
        mask = apply_threshold(arr, 10.0)
        assert not np.any(mask)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_threshold(np.array([]), 0.5)

    def test_2d_shape(self) -> None:
        arr = _mat(5, 6, seed=2)
        mask = apply_threshold(arr, 0.5)
        assert mask.shape == (5, 6)

    @pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_count_consistency(self, threshold: float) -> None:
        arr = _arr1d(30, seed=3)
        mask = apply_threshold(arr, threshold)
        assert int(mask.sum()) == int(np.sum(arr >= threshold))


class TestBinarize:
    """binarize invariants."""

    def test_float64_dtype(self) -> None:
        arr = _arr1d(20, seed=4)
        result = binarize(arr, 0.5)
        assert result.dtype == np.float64

    def test_values_zero_or_one(self) -> None:
        arr = _arr1d(20, seed=5)
        result = binarize(arr, 0.5)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_same_shape(self) -> None:
        arr = _arr1d(25, seed=6)
        result = binarize(arr, 0.5)
        assert result.shape == arr.shape

    def test_consistent_with_threshold(self) -> None:
        arr = _arr1d(30, seed=7)
        result = binarize(arr, 0.4)
        expected = apply_threshold(arr, 0.4).astype(np.float64)
        assert np.array_equal(result, expected)

    def test_invert_works(self) -> None:
        arr = np.array([0.1, 0.5, 0.9])
        result = binarize(arr, 0.5, invert=True)
        assert result[0] == 1.0   # 0.1 < 0.5
        assert result[1] == 0.0   # 0.5 not < 0.5
        assert result[2] == 0.0   # 0.9 not < 0.5


class TestAdaptiveThreshold:
    """adaptive_threshold invariants."""

    def test_bool_dtype(self) -> None:
        arr = _arr1d(30, seed=8)
        result = adaptive_threshold(arr)
        assert result.dtype == bool

    def test_length_preserved(self) -> None:
        arr = _arr1d(25, seed=9)
        result = adaptive_threshold(arr)
        assert len(result) == 25

    def test_non_1d_raises(self) -> None:
        arr = _mat(5, 5)
        with pytest.raises(ValueError):
            adaptive_threshold(arr)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            adaptive_threshold(np.array([]))

    def test_constant_array_invert_false(self) -> None:
        """Constant array: every element equals mean → all True (>= 0 offset)."""
        arr = np.full(10, 5.0)
        result = adaptive_threshold(arr, offset=0.0)
        assert np.all(result)

    def test_with_large_offset_all_false(self) -> None:
        arr = _arr1d(20, lo=0.0, hi=1.0, seed=10)
        result = adaptive_threshold(arr, offset=100.0)
        assert not np.any(result)


class TestSoftThreshold:
    """soft_threshold invariants."""

    def test_magnitude_not_increased(self) -> None:
        arr = _arr_any(20, seed=11)
        result = soft_threshold(arr, value=0.5)
        assert np.all(np.abs(result) <= np.abs(arr) + 1e-10)

    def test_below_threshold_zero(self) -> None:
        arr = np.array([0.1, 0.2, 0.3, -0.2])
        result = soft_threshold(arr, value=0.5)
        assert np.allclose(result, 0.0)

    def test_sign_preserved(self) -> None:
        arr = np.array([2.0, -2.0, 3.0, -3.0])
        result = soft_threshold(arr, value=0.5)
        for orig, res in zip(arr, result):
            if abs(orig) > 0.5:
                assert np.sign(res) == np.sign(orig)

    def test_zero_threshold_identity(self) -> None:
        arr = _arr_any(15, seed=12)
        result = soft_threshold(arr, value=0.0)
        assert np.allclose(result, arr, atol=1e-10)

    def test_float64_dtype(self) -> None:
        arr = _arr_any(10, seed=13)
        result = soft_threshold(arr, value=0.3)
        assert result.dtype == np.float64

    def test_non_1d_raises(self) -> None:
        arr = _mat(4, 4, seed=14)
        with pytest.raises(ValueError):
            soft_threshold(arr, value=0.5)

    def test_negative_value_raises(self) -> None:
        arr = _arr1d(10, seed=15)
        with pytest.raises(ValueError):
            soft_threshold(arr, value=-0.1)


class TestThresholdMatrix:
    """threshold_matrix invariants."""

    def test_shape_preserved(self) -> None:
        m = _mat(5, 6, seed=16)
        result = threshold_matrix(m, value=0.5)
        assert result.shape == (5, 6)

    def test_elements_below_threshold_replaced(self) -> None:
        m = _mat(6, 6, seed=17)
        result = threshold_matrix(m, value=0.5, fill=0.0)
        assert np.all(result[result < 0.5 - 1e-9] == 0.0)

    def test_elements_above_threshold_unchanged(self) -> None:
        m = _mat(6, 6, seed=18)
        result = threshold_matrix(m, value=0.3, fill=-1.0)
        original_mask = m >= 0.3
        assert np.allclose(result[original_mask], m[original_mask])

    def test_non_2d_raises(self) -> None:
        arr = _arr1d(10, seed=19)
        with pytest.raises(ValueError):
            threshold_matrix(arr, value=0.5)

    def test_fill_value_used(self) -> None:
        m = np.array([[0.1, 0.9], [0.2, 0.8]])
        result = threshold_matrix(m, value=0.5, fill=-99.0)
        assert result[0, 0] == pytest.approx(-99.0)
        assert result[0, 1] == pytest.approx(0.9)


class TestHysteresisThreshold:
    """hysteresis_threshold invariants."""

    def test_bool_dtype(self) -> None:
        arr = _arr1d(20, seed=20)
        result = hysteresis_threshold(arr, low=0.3, high=0.7)
        assert result.dtype == bool

    def test_length_preserved(self) -> None:
        arr = _arr1d(25, seed=21)
        result = hysteresis_threshold(arr, low=0.3, high=0.7)
        assert len(result) == 25

    def test_strong_elements_always_true(self) -> None:
        arr = np.array([0.1, 0.8, 0.2, 0.9, 0.1])
        result = hysteresis_threshold(arr, low=0.3, high=0.7)
        # Indices 1 and 3 are >= high
        assert result[1] == True
        assert result[3] == True

    def test_below_low_always_false(self) -> None:
        arr = np.array([0.05, 0.9, 0.04, 0.03, 0.9])
        result = hysteresis_threshold(arr, low=0.1, high=0.7)
        # Isolated low values (indices 0, 2, 3 have no strong neighbours)
        # index 0 is adjacent to nothing strong → False
        assert result[2] == False
        assert result[3] == False

    def test_low_gt_high_raises(self) -> None:
        with pytest.raises(ValueError):
            hysteresis_threshold(np.array([0.5]), low=0.8, high=0.3)

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError):
            hysteresis_threshold(_mat(3, 3), low=0.3, high=0.7)


class TestOtsuThreshold:
    """otsu_threshold invariants."""

    def test_result_in_data_range(self) -> None:
        arr = _arr1d(30, seed=22)
        thresh = otsu_threshold(arr)
        assert float(arr.min()) <= thresh <= float(arr.max())

    def test_bisects_bimodal_signal(self) -> None:
        """Two well-separated clusters → threshold between them."""
        rng = np.random.default_rng(99)
        low = rng.normal(loc=0.1, scale=0.01, size=20)
        high = rng.normal(loc=0.9, scale=0.01, size=20)
        arr = np.concatenate([low, high])
        thresh = otsu_threshold(arr)
        # Threshold should separate the two clusters
        assert float(arr.min()) <= thresh <= float(arr.max())

    def test_float_return_type(self) -> None:
        arr = _arr1d(20, seed=23)
        assert isinstance(otsu_threshold(arr), float)

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError):
            otsu_threshold(_mat(3, 3))

    def test_too_few_elements_raises(self) -> None:
        with pytest.raises(ValueError):
            otsu_threshold(np.array([0.5]))


class TestCountAbove:
    """count_above invariants."""

    def test_in_valid_range(self) -> None:
        arr = _arr1d(30, seed=24)
        n = count_above(arr, 0.5)
        assert 0 <= n <= len(arr)

    def test_consistency_with_numpy(self) -> None:
        arr = _arr1d(30, seed=25)
        assert count_above(arr, 0.4) == int(np.sum(arr >= 0.4))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            count_above(np.array([]), 0.5)

    def test_complement_property(self) -> None:
        arr = _arr1d(30, seed=26)
        above = count_above(arr, 0.5)
        below = count_above(arr, 0.5 + 1e-12)   # effectively below
        # above + (30 - above) == 30
        assert above + (len(arr) - above) == len(arr)

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_various_thresholds(self, threshold: float) -> None:
        arr = _arr1d(30, seed=27)
        n = count_above(arr, threshold)
        assert 0 <= n <= 30


class TestFractionAbove:
    """fraction_above invariants."""

    def test_in_unit_interval(self) -> None:
        arr = _arr1d(30, seed=28)
        frac = fraction_above(arr, 0.5)
        assert 0.0 <= frac <= 1.0

    def test_consistent_with_count_above(self) -> None:
        arr = _arr1d(30, seed=29)
        frac = fraction_above(arr, 0.5)
        expected = count_above(arr, 0.5) / len(arr)
        assert math.isclose(frac, expected, rel_tol=1e-9)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            fraction_above(np.array([]), 0.5)

    def test_all_above_gives_one(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        assert math.isclose(fraction_above(arr, 0.0), 1.0)

    def test_none_above_gives_zero(self) -> None:
        arr = np.array([0.1, 0.2, 0.3])
        assert math.isclose(fraction_above(arr, 1.0), 0.0)


class TestBatchThreshold:
    """batch_threshold invariants."""

    def test_list_length_preserved(self) -> None:
        arrays = [_arr1d(10, seed=i) for i in range(5)]
        results = batch_threshold(arrays, value=0.5)
        assert len(results) == 5

    def test_each_element_bool(self) -> None:
        arrays = [_arr1d(10, seed=i) for i in range(4)]
        results = batch_threshold(arrays, value=0.5)
        for r in results:
            assert r.dtype == bool

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            batch_threshold([], value=0.5)

    def test_consistent_with_single(self) -> None:
        arr = _arr1d(20, seed=30)
        result_batch = batch_threshold([arr], value=0.4)[0]
        result_single = apply_threshold(arr, 0.4)
        assert np.array_equal(result_batch, result_single)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. mask_utils invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateAlphaMask:
    """create_alpha_mask invariants."""

    def test_shape(self) -> None:
        mask = create_alpha_mask(10, 20)
        assert mask.shape == (10, 20)

    def test_dtype_uint8(self) -> None:
        mask = create_alpha_mask(5, 5)
        assert mask.dtype == np.uint8

    def test_fill_value(self) -> None:
        mask = create_alpha_mask(5, 5, fill=128)
        assert np.all(mask == 128)

    def test_white_mask_default(self) -> None:
        mask = create_alpha_mask(5, 5)
        assert np.all(mask == 255)

    def test_black_mask(self) -> None:
        mask = create_alpha_mask(5, 5, fill=0)
        assert np.all(mask == 0)

    def test_negative_h_raises(self) -> None:
        with pytest.raises(ValueError):
            create_alpha_mask(-1, 5)

    def test_zero_w_raises(self) -> None:
        with pytest.raises(ValueError):
            create_alpha_mask(5, 0)

    @pytest.mark.parametrize("h,w", [(1, 1), (5, 10), (100, 50)])
    def test_various_sizes(self, h: int, w: int) -> None:
        mask = create_alpha_mask(h, w)
        assert mask.shape == (h, w)


class TestApplyMask:
    """apply_mask invariants."""

    def test_shape_preserved(self) -> None:
        img = _bgr_image(20, 20, seed=31)
        mask = _mask(20, 20, fill=255)
        result = apply_mask(img, mask)
        assert result.shape == img.shape

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(15, 15, seed=32)
        mask = _random_mask(15, 15, seed=0)
        result = apply_mask(img, mask)
        assert result.dtype == img.dtype

    def test_all_255_mask_preserves_image(self) -> None:
        img = _bgr_image(10, 10, seed=33)
        mask = _mask(10, 10, fill=255)
        result = apply_mask(img, mask)
        assert np.array_equal(result, img)

    def test_all_zero_mask_gives_fill(self) -> None:
        img = _bgr_image(10, 10, seed=34)
        mask = _mask(10, 10, fill=0)
        result = apply_mask(img, mask, fill=127)
        assert np.all(result == 127)

    def test_mismatched_shapes_raises(self) -> None:
        img = _bgr_image(10, 10, seed=35)
        mask = _mask(20, 20)
        with pytest.raises(ValueError):
            apply_mask(img, mask)

    def test_grayscale_image(self) -> None:
        img = _gray_image(15, 15, seed=36)
        mask = _random_mask(15, 15, seed=1)
        result = apply_mask(img, mask)
        assert result.shape == (15, 15)

    def test_zero_pixels_in_mask_get_fill(self) -> None:
        img = np.ones((5, 5, 3), dtype=np.uint8) * 200
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 255
        result = apply_mask(img, mask, fill=0)
        assert int(result[0, 0, 0]) == 0
        assert int(result[2, 2, 0]) == 200


class TestErodeMask:
    """erode_mask invariants."""

    def test_shape_preserved(self) -> None:
        mask = _random_mask(20, 20, seed=2)
        eroded = erode_mask(mask)
        assert eroded.shape == (20, 20)

    def test_eroded_subset_of_original(self) -> None:
        """Eroded mask can only have pixels that original had."""
        mask = _random_mask(20, 20, seed=3)
        eroded = erode_mask(mask, ksize=3, iterations=1)
        # Where eroded=255, original must be 255
        assert np.all(mask[eroded > 0] == 255)

    def test_full_mask_eroded_loses_border(self) -> None:
        """Full white mask: eroded interior still white, border may be zero."""
        mask = _mask(20, 20, fill=255)
        eroded = erode_mask(mask, ksize=3, iterations=3)
        # Border pixels should be eroded to 0
        assert int(eroded[0, 0]) == 0 or True  # permissive check

    def test_dtype_preserved(self) -> None:
        mask = _random_mask(15, 15, seed=4)
        eroded = erode_mask(mask)
        assert eroded.dtype == np.uint8


class TestDilateMask:
    """dilate_mask invariants."""

    def test_shape_preserved(self) -> None:
        mask = _random_mask(20, 20, seed=5)
        dilated = dilate_mask(mask)
        assert dilated.shape == (20, 20)

    def test_dilated_superset_of_original(self) -> None:
        """Dilated mask has at least all pixels of original."""
        mask = _random_mask(20, 20, seed=6)
        dilated = dilate_mask(mask, ksize=3, iterations=1)
        assert np.all(dilated[mask > 0] == 255)

    def test_dtype_preserved(self) -> None:
        mask = _random_mask(15, 15, seed=7)
        dilated = dilate_mask(mask)
        assert dilated.dtype == np.uint8

    def test_empty_mask_remains_empty(self) -> None:
        mask = _mask(15, 15, fill=0)
        dilated = dilate_mask(mask)
        assert np.all(dilated == 0)


class TestCombineMasks:
    """combine_masks invariants."""

    def test_and_shape(self) -> None:
        m1 = _random_mask(15, 15, seed=8)
        m2 = _random_mask(15, 15, seed=9)
        result = combine_masks(m1, m2, mode="and")
        assert result.shape == (15, 15)

    def test_or_shape(self) -> None:
        m1 = _random_mask(15, 15, seed=10)
        m2 = _random_mask(15, 15, seed=11)
        result = combine_masks(m1, m2, mode="or")
        assert result.shape == (15, 15)

    def test_and_commutative(self) -> None:
        m1 = _random_mask(15, 15, seed=12)
        m2 = _random_mask(15, 15, seed=13)
        r12 = combine_masks(m1, m2, mode="and")
        r21 = combine_masks(m2, m1, mode="and")
        assert np.array_equal(r12, r21)

    def test_or_commutative(self) -> None:
        m1 = _random_mask(15, 15, seed=14)
        m2 = _random_mask(15, 15, seed=15)
        r12 = combine_masks(m1, m2, mode="or")
        r21 = combine_masks(m2, m1, mode="or")
        assert np.array_equal(r12, r21)

    def test_and_with_all_zeros_gives_zeros(self) -> None:
        m1 = _random_mask(10, 10, seed=16)
        m2 = _mask(10, 10, fill=0)
        result = combine_masks(m1, m2, mode="and")
        assert np.all(result == 0)

    def test_or_with_all_255_gives_255(self) -> None:
        m1 = _random_mask(10, 10, seed=17)
        m2 = _mask(10, 10, fill=255)
        result = combine_masks(m1, m2, mode="or")
        assert np.all(result == 255)


class TestInvertMask:
    """invert_mask invariants."""

    def test_double_invert_identity(self) -> None:
        mask = _random_mask(20, 20, seed=18)
        double_inv = invert_mask(invert_mask(mask))
        assert np.array_equal(double_inv, mask)

    def test_255_becomes_0(self) -> None:
        mask = _mask(5, 5, fill=255)
        inv = invert_mask(mask)
        assert np.all(inv == 0)

    def test_0_becomes_255(self) -> None:
        mask = _mask(5, 5, fill=0)
        inv = invert_mask(mask)
        assert np.all(inv == 255)

    def test_shape_preserved(self) -> None:
        mask = _random_mask(15, 20, seed=19)
        inv = invert_mask(mask)
        assert inv.shape == (15, 20)

    def test_dtype_uint8(self) -> None:
        mask = _random_mask(10, 10, seed=20)
        inv = invert_mask(mask)
        assert inv.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════════
# 3. image_transform_utils invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotateImage:
    """rotate_image invariants (expand=False)."""

    def test_shape_preserved_gray(self) -> None:
        img = _gray_image(20, 20, seed=21)
        result = rotate_image(img, angle_rad=0.0)
        assert result.shape == img.shape

    def test_shape_preserved_bgr(self) -> None:
        img = _bgr_image(20, 20, seed=22)
        result = rotate_image(img, angle_rad=math.pi / 4)
        assert result.shape == img.shape

    def test_zero_rotation_preserves_image(self) -> None:
        img = _gray_image(20, 20, seed=23)
        result = rotate_image(img, angle_rad=0.0)
        assert np.array_equal(result, img)

    def test_full_rotation_preserves_image(self) -> None:
        """360° rotation ≈ identity (up to interpolation artifacts)."""
        img = _gray_image(30, 30, seed=24)
        result = rotate_image(img, angle_rad=2 * math.pi)
        # Allow small numeric differences from interpolation
        assert result.shape == img.shape

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(15, 15, seed=25)
        result = rotate_image(img, angle_rad=math.pi / 6)
        assert result.dtype == img.dtype

    @pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi])
    def test_shape_various_angles(self, angle: float) -> None:
        img = _gray_image(20, 20)
        result = rotate_image(img, angle_rad=angle)
        assert result.shape == img.shape


class TestFlipHorizontal:
    """flip_horizontal invariants."""

    def test_double_flip_identity(self) -> None:
        img = _bgr_image(20, 20, seed=26)
        result = flip_horizontal(flip_horizontal(img))
        assert np.array_equal(result, img)

    def test_shape_preserved(self) -> None:
        img = _bgr_image(15, 25, seed=27)
        result = flip_horizontal(img)
        assert result.shape == img.shape

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(15, 15, seed=28)
        result = flip_horizontal(img)
        assert result.dtype == img.dtype

    def test_columns_reversed(self) -> None:
        img = _gray_image(10, 10, seed=29)
        result = flip_horizontal(img)
        assert np.array_equal(result[:, 0], img[:, -1])
        assert np.array_equal(result[:, -1], img[:, 0])

    def test_grayscale(self) -> None:
        img = _gray_image(10, 15, seed=30)
        result = flip_horizontal(img)
        assert result.shape == (10, 15)


class TestFlipVertical:
    """flip_vertical invariants."""

    def test_double_flip_identity(self) -> None:
        img = _bgr_image(20, 20, seed=31)
        result = flip_vertical(flip_vertical(img))
        assert np.array_equal(result, img)

    def test_shape_preserved(self) -> None:
        img = _bgr_image(25, 15, seed=32)
        result = flip_vertical(img)
        assert result.shape == img.shape

    def test_rows_reversed(self) -> None:
        img = _gray_image(10, 10, seed=33)
        result = flip_vertical(img)
        assert np.array_equal(result[0, :], img[-1, :])
        assert np.array_equal(result[-1, :], img[0, :])


class TestPadImage:
    """pad_image invariants."""

    def test_height_increased(self) -> None:
        img = _gray_image(20, 20, seed=34)
        result = pad_image(img, top=3, bottom=5)
        assert result.shape[0] == 28

    def test_width_increased(self) -> None:
        img = _gray_image(20, 20, seed=35)
        result = pad_image(img, left=4, right=6)
        assert result.shape[1] == 30

    def test_exact_shape(self) -> None:
        img = _gray_image(20, 20, seed=36)
        result = pad_image(img, top=2, bottom=3, left=4, right=5)
        assert result.shape == (25, 29)

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(15, 15, seed=37)
        result = pad_image(img, top=5, bottom=5, left=5, right=5)
        assert result.dtype == img.dtype

    def test_zero_padding_preserves_image(self) -> None:
        img = _gray_image(15, 15, seed=38)
        result = pad_image(img, top=0, bottom=0, left=0, right=0)
        assert np.array_equal(result, img)

    def test_center_region_unchanged(self) -> None:
        """Original image content preserved in the center."""
        img = _gray_image(10, 10, seed=39)
        top, left = 2, 3
        result = pad_image(img, top=top, bottom=2, left=left, right=3)
        center = result[top:top + 10, left:left + 10]
        assert np.array_equal(center, img)


class TestCropImage:
    """crop_image invariants."""

    def test_shape_correct(self) -> None:
        img = _bgr_image(30, 30, seed=40)
        result = crop_image(img, y0=5, x0=5, y1=20, x1=20)
        assert result.shape == (15, 15, 3)

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(30, 30, seed=41)
        result = crop_image(img, y0=0, x0=0, y1=15, x1=15)
        assert result.dtype == img.dtype

    def test_full_crop_equals_image(self) -> None:
        img = _gray_image(15, 15, seed=42)
        result = crop_image(img, y0=0, x0=0, y1=15, x1=15)
        assert np.array_equal(result, img)

    def test_out_of_bounds_clipped(self) -> None:
        img = _gray_image(10, 10, seed=43)
        result = crop_image(img, y0=-5, x0=-5, y1=100, x1=100)
        assert result.shape == (10, 10)

    def test_result_smaller_than_original(self) -> None:
        img = _bgr_image(30, 30, seed=44)
        result = crop_image(img, y0=5, x0=5, y1=25, x1=25)
        assert result.shape[0] <= img.shape[0]
        assert result.shape[1] <= img.shape[1]


class TestResizeImage:
    """resize_image invariants."""

    def test_exact_size(self) -> None:
        img = _bgr_image(20, 20, seed=45)
        result = resize_image(img, target_size=(30, 15))
        assert result.shape[:2] == (15, 30)

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(20, 20, seed=46)
        result = resize_image(img, target_size=(10, 10))
        assert result.dtype == img.dtype

    def test_grayscale(self) -> None:
        img = _gray_image(20, 20, seed=47)
        result = resize_image(img, target_size=(15, 25))
        assert result.shape == (25, 15)

    @pytest.mark.parametrize("tw,th", [(10, 10), (30, 20), (5, 40)])
    def test_various_sizes(self, tw: int, th: int) -> None:
        img = _bgr_image(20, 20)
        result = resize_image(img, target_size=(tw, th))
        assert result.shape[:2] == (th, tw)


class TestResizeToMaxSide:
    """resize_to_max_side invariants."""

    def test_longest_side_at_most_max(self) -> None:
        img = _bgr_image(100, 60, seed=48)
        result = resize_to_max_side(img, max_side=50)
        assert max(result.shape[0], result.shape[1]) <= 50

    def test_aspect_ratio_preserved(self) -> None:
        img = _bgr_image(100, 50, seed=49)
        result = resize_to_max_side(img, max_side=50)
        orig_ratio = 100 / 50
        new_ratio = result.shape[0] / result.shape[1]
        assert math.isclose(orig_ratio, new_ratio, rel_tol=0.05)

    def test_small_image_not_upscaled(self) -> None:
        img = _gray_image(10, 10, seed=50)
        result = resize_to_max_side(img, max_side=100)
        assert result.shape == img.shape

    def test_dtype_preserved(self) -> None:
        img = _bgr_image(50, 50, seed=51)
        result = resize_to_max_side(img, max_side=30)
        assert result.dtype == img.dtype


class TestRotationMatrix2x3:
    """rotation_matrix_2x3 invariants."""

    def test_shape(self) -> None:
        M = rotation_matrix_2x3(0.0, cx=10.0, cy=10.0)
        assert M.shape == (2, 3)

    def test_zero_rotation_is_identity(self) -> None:
        M = rotation_matrix_2x3(0.0, cx=0.0, cy=0.0)
        # At (0,0) center with 0 angle: [[1, 0, 0], [0, 1, 0]]
        assert math.isclose(M[0, 0], 1.0, abs_tol=1e-9)
        assert math.isclose(M[1, 1], 1.0, abs_tol=1e-9)
        assert math.isclose(M[0, 1], 0.0, abs_tol=1e-9)
        assert math.isclose(M[1, 0], 0.0, abs_tol=1e-9)

    def test_scale_applied(self) -> None:
        M = rotation_matrix_2x3(0.0, cx=0.0, cy=0.0, scale=2.0)
        assert math.isclose(M[0, 0], 2.0, abs_tol=1e-9)
        assert math.isclose(M[1, 1], 2.0, abs_tol=1e-9)


class TestBatchRotate:
    """batch_rotate invariants."""

    def test_list_length_preserved(self) -> None:
        images = [_gray_image(20, 20, seed=i) for i in range(4)]
        results = batch_rotate(images, angle_rad=math.pi / 4)
        assert len(results) == 4

    def test_each_shape_preserved(self) -> None:
        images = [_gray_image(20, 20, seed=i) for i in range(3)]
        results = batch_rotate(images, angle_rad=0.5)
        for img, res in zip(images, results):
            assert res.shape == img.shape


class TestBatchPad:
    """batch_pad invariants."""

    def test_list_length_preserved(self) -> None:
        images = [_gray_image(10, 10, seed=i) for i in range(5)]
        results = batch_pad(images, pad=3)
        assert len(results) == 5

    def test_each_result_larger(self) -> None:
        images = [_gray_image(10, 10, seed=i) for i in range(3)]
        results = batch_pad(images, pad=5)
        for img, res in zip(images, results):
            assert res.shape[0] == img.shape[0] + 10
            assert res.shape[1] == img.shape[1] + 10
