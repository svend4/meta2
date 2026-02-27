"""
Property-based tests for:
  1. puzzle_reconstruction.utils.array_utils
  2. puzzle_reconstruction.utils.bbox_utils

Verifies mathematical invariants:

array_utils:
- normalize_array:        output ∈ [low, high]; same shape; constant → all low;
                          translation invariant; scaling invariant
- pad_to_shape:           output shape = target; original data preserved in top-left;
                          ValueError if target < arr on any axis
- crop_center:            output shape = size; data from center of arr;
                          ValueError if size > arr
- stack_arrays:           output.shape[0] = len(arrays); output shape ≥ each input
- chunk_array:            each chunk ≤ chunk_size; concat = original; total elems = n
- sliding_window:         all windows have exactly size elements; count formula correct
- flatten/unflatten:      round-trip identity
- compute_pairwise_norms: symmetric; diagonal = 0; non-negative;
                          l2 satisfies triangle inequality

bbox_utils:
- BBox:                   area = w * h; invalid w/h raises
- bbox_iou:               ∈ [0, 1]; symmetric; self = 1; disjoint = 0
- bbox_intersection:      None if disjoint; area ≤ min(a.area, b.area); symmetric
- bbox_union:             area ≥ max(a.area, b.area); both boxes inside union
- expand_bbox:            expanded area ≥ original; 0 padding preserves box
- bbox_center:            cx = x + w/2; cy = y + h/2
- bbox_aspect_ratio:      ∈ (0, 1]; square → 1.0; w > h → ratio = h/w
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.array_utils import (
    normalize_array,
    pad_to_shape,
    crop_center,
    stack_arrays,
    chunk_array,
    sliding_window,
    flatten_images,
    unflatten_images,
    compute_pairwise_norms,
)
from puzzle_reconstruction.utils.bbox_utils import (
    BBox,
    bbox_iou,
    bbox_intersection,
    bbox_union,
    expand_bbox,
    bbox_center,
    bbox_aspect_ratio,
)

RNG = np.random.default_rng(42)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_arr(shape: Tuple[int, ...], lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
    return RNG.uniform(lo, hi, size=shape)


def _rand_bbox(max_coord: int = 50) -> BBox:
    x = int(RNG.integers(0, max_coord))
    y = int(RNG.integers(0, max_coord))
    w = int(RNG.integers(1, max_coord + 1))
    h = int(RNG.integers(1, max_coord + 1))
    return BBox(x=x, y=y, w=w, h=h)


# ═══════════════════════════════════════════════════════════════════════════════
# normalize_array
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeArray:

    def test_output_in_range(self):
        arr = _rand_arr((30,))
        out = normalize_array(arr, low=0.0, high=1.0)
        assert float(out.min()) >= 0.0 - 1e-12
        assert float(out.max()) <= 1.0 + 1e-12

    def test_output_in_custom_range(self):
        arr = _rand_arr((20, 10))
        out = normalize_array(arr, low=-3.0, high=7.0)
        assert float(out.min()) >= -3.0 - 1e-10
        assert float(out.max()) <= 7.0 + 1e-10

    def test_same_shape(self):
        arr = _rand_arr((5, 8, 3))
        out = normalize_array(arr)
        assert out.shape == arr.shape

    def test_constant_array_maps_to_low(self):
        arr = np.full((10, 10), 5.0)
        out = normalize_array(arr, low=2.0, high=9.0)
        assert np.allclose(out, 2.0)

    def test_translation_invariant(self):
        arr = _rand_arr((15,))
        out1 = normalize_array(arr, low=0.0, high=1.0)
        out2 = normalize_array(arr + 100.0, low=0.0, high=1.0)
        assert np.allclose(out1, out2, atol=1e-10)

    def test_scaling_invariant(self):
        arr = _rand_arr((15,))
        out1 = normalize_array(arr, low=0.0, high=1.0)
        out2 = normalize_array(arr * 3.0, low=0.0, high=1.0)
        assert np.allclose(out1, out2, atol=1e-10)

    def test_min_maps_to_low_max_maps_to_high(self):
        arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        out = normalize_array(arr, low=0.0, high=10.0)
        assert abs(out[np.argmin(arr)] - 0.0) < 1e-10
        assert abs(out[np.argmax(arr)] - 10.0) < 1e-10

    def test_single_element(self):
        arr = np.array([42.0])
        out = normalize_array(arr, low=1.0, high=2.0)
        assert out.shape == (1,)
        assert abs(float(out[0]) - 1.0) < 1e-10

    def test_dtype_preserved_when_specified(self):
        arr = _rand_arr((10,))
        out = normalize_array(arr, dtype=np.float32)
        assert out.dtype == np.float32

    def test_default_dtype_is_float64(self):
        arr = _rand_arr((10,))
        out = normalize_array(arr)
        assert out.dtype == np.float64

    def test_negative_range(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        out = normalize_array(arr, low=-2.0, high=-1.0)
        assert float(out.min()) >= -2.0 - 1e-10
        assert float(out.max()) <= -1.0 + 1e-10

    def test_all_same_value_multiple_elements(self):
        arr = np.ones(20) * 7.0
        out = normalize_array(arr, low=0.5, high=0.5)
        assert np.allclose(out, 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# pad_to_shape
# ═══════════════════════════════════════════════════════════════════════════════

class TestPadToShape:

    def test_output_shape_topleft(self):
        arr = np.ones((3, 4), dtype=float)
        out = pad_to_shape(arr, (6, 8))
        assert out.shape == (6, 8)

    def test_output_shape_center(self):
        arr = np.ones((3, 4), dtype=float)
        out = pad_to_shape(arr, (6, 8), align="center")
        assert out.shape == (6, 8)

    def test_topleft_data_preserved(self):
        arr = np.arange(12, dtype=float).reshape(3, 4)
        out = pad_to_shape(arr, (5, 7))
        assert np.allclose(out[:3, :4], arr)

    def test_pad_value_used(self):
        arr = np.ones((2, 2))
        out = pad_to_shape(arr, (4, 4), value=-1.0)
        # Bottom-right should be padded with -1
        assert np.allclose(out[2:, :], -1.0)
        assert np.allclose(out[:, 2:], -1.0)

    def test_exact_shape_no_change(self):
        arr = _rand_arr((4, 4))
        out = pad_to_shape(arr, (4, 4))
        assert np.allclose(out, arr)

    def test_raises_if_target_too_small(self):
        arr = np.ones((5, 5))
        with pytest.raises(ValueError):
            pad_to_shape(arr, (3, 6))

    def test_3d_array(self):
        arr = np.ones((3, 4, 3), dtype=np.uint8)
        out = pad_to_shape(arr, (6, 8, 3))
        assert out.shape == (6, 8, 3)

    def test_center_alignment_symmetric(self):
        arr = np.ones((2, 2))
        out = pad_to_shape(arr, (6, 6), value=0.0, align="center")
        assert out.shape == (6, 6)
        # The arr should be roughly in the center
        assert float(out[2:4, 2:4].sum()) == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════════════════════
# crop_center
# ═══════════════════════════════════════════════════════════════════════════════

class TestCropCenter:

    def test_output_shape(self):
        arr = np.zeros((10, 12))
        out = crop_center(arr, (4, 6))
        assert out.shape == (4, 6)

    def test_output_shape_3d(self):
        arr = np.zeros((10, 12, 3))
        out = crop_center(arr, (4, 6))
        assert out.shape == (4, 6, 3)

    def test_full_size_is_identity(self):
        arr = _rand_arr((8, 8))
        out = crop_center(arr, (8, 8))
        assert np.allclose(out, arr)

    def test_center_pixel_preserved(self):
        arr = np.zeros((10, 10))
        arr[5, 5] = 999.0
        out = crop_center(arr, (2, 2))
        # Center (5,5) should appear in the cropped region when crop is (2,2)
        # y0 = 5 - 1 = 4, x0 = 5 - 1 = 4 → region [4:6, 4:6]
        assert 999.0 in out

    def test_raises_if_too_large(self):
        arr = np.zeros((4, 4))
        with pytest.raises(ValueError):
            crop_center(arr, (5, 3))

    def test_single_row_crop(self):
        arr = _rand_arr((10, 10))
        out = crop_center(arr, (1, 10))
        assert out.shape == (1, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# stack_arrays
# ═══════════════════════════════════════════════════════════════════════════════

class TestStackArrays:

    def test_output_n_equals_num_arrays(self):
        arrays = [_rand_arr((3, 4)) for _ in range(5)]
        out = stack_arrays(arrays, axis=0)
        assert out.shape[0] == 5

    def test_output_size_at_least_max(self):
        shapes = [(3, 4), (5, 2), (2, 6)]
        arrays = [np.ones(s) for s in shapes]
        out = stack_arrays(arrays, axis=0)
        assert out.shape[1] >= max(s[0] for s in shapes)
        assert out.shape[2] >= max(s[1] for s in shapes)

    def test_single_array(self):
        a = _rand_arr((4, 5))
        out = stack_arrays([a], axis=0)
        assert out.shape == (1, 4, 5)
        assert np.allclose(out[0], a)

    def test_same_shapes_no_padding(self):
        arrays = [_rand_arr((4, 4)) for _ in range(3)]
        out = stack_arrays(arrays, axis=0)
        for i, arr in enumerate(arrays):
            assert np.allclose(out[i], arr)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            stack_arrays([])

    def test_pad_value_propagated(self):
        arrays = [np.ones((2, 2)), np.ones((4, 4))]
        out = stack_arrays(arrays, axis=0, value=-1.0)
        # First array should be padded with -1 in extended region
        assert float(out[0, 2, 2]) == pytest.approx(-1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# chunk_array
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkArray:

    def test_all_chunks_le_chunk_size(self):
        arr = _rand_arr((17,))
        chunks = chunk_array(arr, chunk_size=5, axis=0)
        for c in chunks:
            assert c.shape[0] <= 5

    def test_concat_restores_original(self):
        arr = _rand_arr((17,))
        chunks = chunk_array(arr, chunk_size=5, axis=0)
        restored = np.concatenate(chunks, axis=0)
        assert np.allclose(restored, arr)

    def test_total_elements_preserved(self):
        arr = _rand_arr((13,))
        chunks = chunk_array(arr, chunk_size=4, axis=0)
        total = sum(c.shape[0] for c in chunks)
        assert total == 13

    def test_exact_division(self):
        arr = _rand_arr((12,))
        chunks = chunk_array(arr, chunk_size=4, axis=0)
        assert len(chunks) == 3
        assert all(c.shape[0] == 4 for c in chunks)

    def test_chunk_size_larger_than_arr(self):
        arr = _rand_arr((5,))
        chunks = chunk_array(arr, chunk_size=20, axis=0)
        assert len(chunks) == 1
        assert np.allclose(chunks[0], arr)

    def test_raises_if_chunk_size_zero(self):
        arr = _rand_arr((5,))
        with pytest.raises(ValueError):
            chunk_array(arr, chunk_size=0)

    def test_along_axis_1(self):
        arr = _rand_arr((4, 10))
        chunks = chunk_array(arr, chunk_size=3, axis=1)
        restored = np.concatenate(chunks, axis=1)
        assert np.allclose(restored, arr)


# ═══════════════════════════════════════════════════════════════════════════════
# sliding_window
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlidingWindow:

    def test_each_window_has_correct_size(self):
        arr = _rand_arr((20,))
        windows = list(sliding_window(arr, size=5, step=1))
        for w in windows:
            assert w.shape[0] == 5

    def test_window_count_formula(self):
        n, size, step = 20, 5, 2
        arr = _rand_arr((n,))
        windows = list(sliding_window(arr, size=size, step=step))
        expected = len(range(0, n - size + 1, step))
        assert len(windows) == expected

    def test_step_1_count(self):
        arr = _rand_arr((10,))
        windows = list(sliding_window(arr, size=3, step=1))
        assert len(windows) == 8  # 10 - 3 + 1

    def test_step_equals_size_non_overlapping(self):
        arr = np.arange(12, dtype=float)
        windows = list(sliding_window(arr, size=4, step=4))
        assert len(windows) == 3
        for i, w in enumerate(windows):
            assert np.allclose(w, arr[i * 4 : i * 4 + 4])

    def test_window_larger_than_arr_empty(self):
        arr = _rand_arr((3,))
        windows = list(sliding_window(arr, size=5, step=1))
        assert len(windows) == 0

    def test_raises_if_size_zero(self):
        arr = _rand_arr((10,))
        with pytest.raises(ValueError):
            list(sliding_window(arr, size=0))

    def test_raises_if_step_zero(self):
        arr = _rand_arr((10,))
        with pytest.raises(ValueError):
            list(sliding_window(arr, size=3, step=0))


# ═══════════════════════════════════════════════════════════════════════════════
# flatten_images / unflatten_images
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlattenUnflatten:

    def test_flatten_shape(self):
        images = [_rand_arr((4, 5)) for _ in range(3)]
        flat = flatten_images(images)
        assert flat.shape == (3, 20)

    def test_flatten_unflatten_roundtrip(self):
        shape = (4, 5)
        images = [_rand_arr(shape) for _ in range(3)]
        flat = flatten_images(images)
        restored = unflatten_images(flat, shape)
        assert len(restored) == 3
        for orig, rest in zip(images, restored):
            assert np.allclose(orig.ravel(), rest.ravel())

    def test_flatten_3d_images(self):
        images = [_rand_arr((4, 5, 3)) for _ in range(2)]
        flat = flatten_images(images)
        assert flat.shape == (2, 60)

    def test_flatten_raises_on_empty(self):
        with pytest.raises(ValueError):
            flatten_images([])

    def test_flatten_raises_on_mismatched_shapes(self):
        with pytest.raises(ValueError):
            flatten_images([np.ones((3, 3)), np.ones((4, 4))])

    def test_unflatten_raises_on_wrong_d(self):
        matrix = np.ones((3, 10))
        with pytest.raises(ValueError):
            unflatten_images(matrix, (3, 4))  # 3*4=12 ≠ 10

    def test_single_image_roundtrip(self):
        img = _rand_arr((8, 8))
        flat = flatten_images([img])
        restored = unflatten_images(flat, (8, 8))
        assert np.allclose(img.ravel(), restored[0].ravel())


# ═══════════════════════════════════════════════════════════════════════════════
# compute_pairwise_norms
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePairwiseNorms:

    def _matrix(self, n: int = 5, d: int = 8) -> np.ndarray:
        return _rand_arr((n, d))

    def test_output_shape(self):
        m = self._matrix(5, 8)
        for metric in ("l2", "l1", "cosine"):
            out = compute_pairwise_norms(m, metric=metric)
            assert out.shape == (5, 5)

    def test_diagonal_is_zero(self):
        m = self._matrix(5, 8)
        for metric in ("l2", "l1", "cosine"):
            out = compute_pairwise_norms(m, metric=metric)
            assert np.allclose(np.diag(out), 0.0, atol=1e-10)

    def test_non_negative(self):
        m = self._matrix(5, 8)
        for metric in ("l2", "l1", "cosine"):
            out = compute_pairwise_norms(m, metric=metric)
            assert float(out.min()) >= -1e-10

    def test_symmetric(self):
        m = self._matrix(5, 8)
        for metric in ("l2", "l1", "cosine"):
            out = compute_pairwise_norms(m, metric=metric)
            assert np.allclose(out, out.T, atol=1e-10)

    def test_l2_triangle_inequality(self):
        m = self._matrix(4, 6)
        out = compute_pairwise_norms(m, metric="l2")
        n = out.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert out[i, j] <= out[i, k] + out[k, j] + 1e-9

    def test_l2_known_value(self):
        m = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = compute_pairwise_norms(m, metric="l2")
        assert abs(out[0, 1] - 5.0) < 1e-10

    def test_l1_known_value(self):
        m = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = compute_pairwise_norms(m, metric="l1")
        assert abs(out[0, 1] - 7.0) < 1e-10

    def test_cosine_self_is_zero(self):
        m = self._matrix(4, 6)
        out = compute_pairwise_norms(m, metric="cosine")
        assert np.allclose(np.diag(out), 0.0, atol=1e-8)

    def test_cosine_range(self):
        m = self._matrix(5, 8)
        out = compute_pairwise_norms(m, metric="cosine")
        assert float(out.min()) >= -1e-10
        assert float(out.max()) <= 2.0 + 1e-10

    def test_raises_unknown_metric(self):
        m = self._matrix(3, 4)
        with pytest.raises(ValueError):
            compute_pairwise_norms(m, metric="euclidean")


# ═══════════════════════════════════════════════════════════════════════════════
# BBox
# ═══════════════════════════════════════════════════════════════════════════════

class TestBBox:

    def test_area(self):
        b = BBox(x=0, y=0, w=5, h=3)
        assert b.area == 15

    def test_x2_y2(self):
        b = BBox(x=2, y=3, w=4, h=5)
        assert b.x2 == 6
        assert b.y2 == 8

    def test_to_tuple(self):
        b = BBox(x=1, y=2, w=3, h=4)
        assert b.to_tuple() == (1, 2, 3, 4)

    def test_raises_zero_width(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=0, h=5)

    def test_raises_zero_height(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=5, h=0)

    def test_raises_negative_width(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=-1, h=5)

    def test_unit_box_area_one(self):
        b = BBox(x=0, y=0, w=1, h=1)
        assert b.area == 1

    def test_area_large(self):
        b = BBox(x=100, y=200, w=50, h=30)
        assert b.area == 1500


# ═══════════════════════════════════════════════════════════════════════════════
# bbox_iou
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxIou:

    def test_range_zero_to_one(self):
        for _ in range(30):
            a = _rand_bbox()
            b = _rand_bbox()
            iou = bbox_iou(a, b)
            assert 0.0 <= iou <= 1.0 + 1e-10

    def test_self_iou_is_one(self):
        b = BBox(x=5, y=5, w=10, h=10)
        assert abs(bbox_iou(b, b) - 1.0) < 1e-10

    def test_symmetric(self):
        a = BBox(x=0, y=0, w=10, h=10)
        b = BBox(x=5, y=5, w=10, h=10)
        assert abs(bbox_iou(a, b) - bbox_iou(b, a)) < 1e-10

    def test_disjoint_is_zero(self):
        a = BBox(x=0, y=0, w=5, h=5)
        b = BBox(x=10, y=10, w=5, h=5)
        assert bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = BBox(x=0, y=0, w=4, h=4)
        b = BBox(x=2, y=0, w=4, h=4)
        iou = bbox_iou(a, b)
        # intersection = 2x4=8; union = 4x4+4x4-8=24; iou=8/24=1/3
        assert abs(iou - 1.0 / 3.0) < 1e-10

    def test_contained_box(self):
        outer = BBox(x=0, y=0, w=10, h=10)
        inner = BBox(x=2, y=2, w=4, h=4)
        iou = bbox_iou(outer, inner)
        # inter = 16; union = 100; iou = 16/100
        assert abs(iou - 16.0 / 100.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# bbox_intersection
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxIntersection:

    def test_disjoint_returns_none(self):
        a = BBox(x=0, y=0, w=5, h=5)
        b = BBox(x=10, y=10, w=5, h=5)
        assert bbox_intersection(a, b) is None

    def test_symmetric(self):
        a = BBox(x=0, y=0, w=8, h=8)
        b = BBox(x=4, y=4, w=8, h=8)
        inter_ab = bbox_intersection(a, b)
        inter_ba = bbox_intersection(b, a)
        assert inter_ab is not None and inter_ba is not None
        assert inter_ab.to_tuple() == inter_ba.to_tuple()

    def test_intersection_area_le_both_areas(self):
        a = BBox(x=0, y=0, w=10, h=10)
        b = BBox(x=3, y=3, w=10, h=10)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert inter.area <= a.area
        assert inter.area <= b.area

    def test_self_intersection_equals_self(self):
        b = BBox(x=5, y=5, w=8, h=6)
        inter = bbox_intersection(b, b)
        assert inter is not None
        assert inter.to_tuple() == b.to_tuple()

    def test_known_intersection(self):
        a = BBox(x=0, y=0, w=4, h=4)
        b = BBox(x=2, y=0, w=4, h=4)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert inter.x == 2 and inter.w == 2 and inter.h == 4


# ═══════════════════════════════════════════════════════════════════════════════
# bbox_union
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxUnion:

    def test_union_area_ge_both(self):
        for _ in range(20):
            a = _rand_bbox(20)
            b = _rand_bbox(20)
            u = bbox_union(a, b)
            assert u.area >= a.area
            assert u.area >= b.area

    def test_self_union_equals_self(self):
        b = BBox(x=2, y=3, w=5, h=7)
        u = bbox_union(b, b)
        assert u.to_tuple() == b.to_tuple()

    def test_symmetric(self):
        a = BBox(x=0, y=0, w=5, h=5)
        b = BBox(x=3, y=3, w=5, h=5)
        u1 = bbox_union(a, b)
        u2 = bbox_union(b, a)
        assert u1.to_tuple() == u2.to_tuple()

    def test_both_boxes_inside_union(self):
        a = BBox(x=0, y=0, w=5, h=5)
        b = BBox(x=3, y=3, w=5, h=5)
        u = bbox_union(a, b)
        assert u.x <= a.x and u.x <= b.x
        assert u.y <= a.y and u.y <= b.y
        assert u.x2 >= a.x2 and u.x2 >= b.x2
        assert u.y2 >= a.y2 and u.y2 >= b.y2

    def test_disjoint_union(self):
        a = BBox(x=0, y=0, w=2, h=2)
        b = BBox(x=10, y=10, w=2, h=2)
        u = bbox_union(a, b)
        assert u.x == 0 and u.y == 0
        assert u.x2 == 12 and u.y2 == 12


# ═══════════════════════════════════════════════════════════════════════════════
# expand_bbox
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpandBbox:

    def test_expanded_area_ge_original(self):
        b = BBox(x=5, y=5, w=10, h=10)
        expanded = expand_bbox(b, padding=3)
        assert expanded.area >= b.area

    def test_zero_padding_preserves_box(self):
        b = BBox(x=5, y=5, w=10, h=10)
        expanded = expand_bbox(b, padding=0)
        assert expanded.x == b.x
        assert expanded.y == b.y
        assert expanded.w == b.w
        assert expanded.h == b.h

    def test_negative_padding_raises(self):
        b = BBox(x=5, y=5, w=10, h=10)
        with pytest.raises(ValueError):
            expand_bbox(b, padding=-1)

    def test_x_y_not_below_zero(self):
        b = BBox(x=1, y=1, w=5, h=5)
        expanded = expand_bbox(b, padding=5)
        assert expanded.x >= 0
        assert expanded.y >= 0

    def test_max_bounds_clip(self):
        b = BBox(x=5, y=5, w=10, h=10)
        expanded = expand_bbox(b, padding=10, max_h=20, max_w=20)
        assert expanded.x2 <= 20
        assert expanded.y2 <= 20


# ═══════════════════════════════════════════════════════════════════════════════
# bbox_center
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxCenter:

    def test_center_formula(self):
        b = BBox(x=4, y=6, w=10, h=8)
        cx, cy = bbox_center(b)
        assert abs(cx - 9.0) < 1e-10
        assert abs(cy - 10.0) < 1e-10

    def test_unit_box_center(self):
        b = BBox(x=0, y=0, w=1, h=1)
        cx, cy = bbox_center(b)
        assert abs(cx - 0.5) < 1e-10
        assert abs(cy - 0.5) < 1e-10

    def test_center_inside_box(self):
        for _ in range(20):
            b = _rand_bbox()
            cx, cy = bbox_center(b)
            assert b.x <= cx <= b.x2
            assert b.y <= cy <= b.y2


# ═══════════════════════════════════════════════════════════════════════════════
# bbox_aspect_ratio
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxAspectRatio:

    def test_range_zero_to_one(self):
        for _ in range(30):
            b = _rand_bbox()
            r = bbox_aspect_ratio(b)
            assert 0.0 < r <= 1.0 + 1e-10

    def test_square_is_one(self):
        b = BBox(x=0, y=0, w=7, h=7)
        assert abs(bbox_aspect_ratio(b) - 1.0) < 1e-10

    def test_wide_box(self):
        b = BBox(x=0, y=0, w=10, h=2)
        r = bbox_aspect_ratio(b)
        assert abs(r - 0.2) < 1e-10

    def test_tall_box(self):
        b = BBox(x=0, y=0, w=2, h=10)
        r = bbox_aspect_ratio(b)
        assert abs(r - 0.2) < 1e-10

    def test_symmetric_aspect_ratio(self):
        b1 = BBox(x=0, y=0, w=4, h=6)
        b2 = BBox(x=0, y=0, w=6, h=4)
        assert abs(bbox_aspect_ratio(b1) - bbox_aspect_ratio(b2)) < 1e-10
