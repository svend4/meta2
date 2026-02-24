"""Extra tests for puzzle_reconstruction/utils/array_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=8, w=8, val=100) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _mat(n=4, d=3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, d)).astype(np.float64)


# ─── normalize_array ──────────────────────────────────────────────────────────

class TestNormalizeArrayExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_array(np.array([1.0, 2.0])), np.ndarray)

    def test_min_becomes_low(self):
        a = np.array([0.0, 5.0, 10.0])
        out = normalize_array(a, low=0.0, high=1.0)
        assert out.min() == pytest.approx(0.0)

    def test_max_becomes_high(self):
        a = np.array([0.0, 5.0, 10.0])
        out = normalize_array(a, low=0.0, high=1.0)
        assert out.max() == pytest.approx(1.0)

    def test_custom_range(self):
        a = np.array([0.0, 10.0])
        out = normalize_array(a, low=-1.0, high=1.0)
        assert out[0] == pytest.approx(-1.0)
        assert out[1] == pytest.approx(1.0)

    def test_constant_array_all_low(self):
        a = np.full(5, 7.0)
        out = normalize_array(a, low=2.0)
        assert np.allclose(out, 2.0)

    def test_default_dtype_float64(self):
        a = np.array([0, 1, 2], dtype=np.int32)
        out = normalize_array(a)
        assert out.dtype == np.float64

    def test_custom_dtype(self):
        a = np.array([0.0, 1.0])
        out = normalize_array(a, dtype=np.float32)
        assert out.dtype == np.float32

    def test_shape_preserved(self):
        a = np.ones((3, 4, 2))
        out = normalize_array(a)
        assert out.shape == a.shape


# ─── pad_to_shape ─────────────────────────────────────────────────────────────

class TestPadToShapeExtra:
    def test_returns_ndarray(self):
        a = np.zeros((2, 2))
        assert isinstance(pad_to_shape(a, (4, 4)), np.ndarray)

    def test_output_shape(self):
        a = np.zeros((2, 3))
        out = pad_to_shape(a, (5, 7))
        assert out.shape == (5, 7)

    def test_data_preserved_topleft(self):
        a = np.ones((2, 2), dtype=np.uint8)
        out = pad_to_shape(a, (4, 4), value=0, align="topleft")
        np.testing.assert_array_equal(out[:2, :2], a)
        assert out[3, 3] == 0

    def test_center_align(self):
        a = np.ones((2, 2), dtype=np.uint8)
        out = pad_to_shape(a, (4, 4), value=0, align="center")
        # Content should be near center
        assert out[1, 1] == 1

    def test_fill_value_used(self):
        a = np.zeros((2, 2))
        out = pad_to_shape(a, (4, 4), value=9.0)
        assert out[3, 3] == pytest.approx(9.0)

    def test_exceeds_shape_raises(self):
        a = np.zeros((5, 5))
        with pytest.raises(ValueError):
            pad_to_shape(a, (3, 3))

    def test_same_shape_unchanged(self):
        a = np.ones((3, 3), dtype=np.float32)
        out = pad_to_shape(a, (3, 3))
        np.testing.assert_array_equal(out, a)


# ─── crop_center ──────────────────────────────────────────────────────────────

class TestCropCenterExtra:
    def test_returns_ndarray(self):
        a = np.zeros((8, 8))
        assert isinstance(crop_center(a, (4, 4)), np.ndarray)

    def test_output_shape(self):
        a = np.zeros((10, 12))
        out = crop_center(a, (6, 8))
        assert out.shape == (6, 8)

    def test_center_value_preserved(self):
        a = np.zeros((8, 8), dtype=np.float32)
        a[4, 4] = 99.0
        out = crop_center(a, (4, 4))
        assert out[2, 2] == pytest.approx(99.0)

    def test_exceeds_size_raises(self):
        a = np.zeros((4, 4))
        with pytest.raises(ValueError):
            crop_center(a, (6, 4))

    def test_exact_size_same_array(self):
        a = np.ones((6, 8))
        out = crop_center(a, (6, 8))
        np.testing.assert_array_equal(out, a)

    def test_3d_array_ok(self):
        a = np.zeros((10, 10, 3))
        out = crop_center(a, (4, 4))
        assert out.shape == (4, 4, 3)


# ─── stack_arrays ─────────────────────────────────────────────────────────────

class TestStackArraysExtra:
    def test_returns_ndarray(self):
        a = np.zeros((4, 4))
        assert isinstance(stack_arrays([a, a]), np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_arrays([])

    def test_uniform_shape_stack(self):
        arrs = [np.ones((4, 4)) for _ in range(3)]
        out = stack_arrays(arrs)
        assert out.shape == (3, 4, 4)

    def test_different_shapes_padded(self):
        a = np.ones((2, 2))
        b = np.ones((4, 4))
        out = stack_arrays([a, b])
        assert out.shape[1] == 4 and out.shape[2] == 4

    def test_fill_value_for_padding(self):
        a = np.ones((2, 2), dtype=np.float64)
        b = np.ones((4, 4), dtype=np.float64)
        out = stack_arrays([a, b], value=7.0)
        # Padded part of first array should be 7.0
        assert out[0, 3, 3] == pytest.approx(7.0)


# ─── chunk_array ──────────────────────────────────────────────────────────────

class TestChunkArrayExtra:
    def test_returns_list(self):
        a = np.arange(10)
        assert isinstance(chunk_array(a, 3), list)

    def test_chunk_size_lt_1_raises(self):
        with pytest.raises(ValueError):
            chunk_array(np.arange(5), 0)

    def test_even_split(self):
        a = np.arange(9)
        chunks = chunk_array(a, 3)
        assert len(chunks) == 3
        for c in chunks:
            assert len(c) == 3

    def test_last_chunk_smaller(self):
        a = np.arange(10)
        chunks = chunk_array(a, 3)
        assert len(chunks) == 4
        assert len(chunks[-1]) == 1

    def test_chunk_size_larger_than_array(self):
        a = np.arange(5)
        chunks = chunk_array(a, 10)
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], a)

    def test_along_axis_1(self):
        a = np.ones((4, 6))
        chunks = chunk_array(a, 2, axis=1)
        assert len(chunks) == 3
        assert chunks[0].shape == (4, 2)


# ─── sliding_window ───────────────────────────────────────────────────────────

class TestSlidingWindowExtra:
    def test_returns_generator(self):
        import types
        gen = sliding_window(np.arange(5), size=3)
        assert isinstance(gen, types.GeneratorType)

    def test_size_lt_1_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=0))

    def test_step_lt_1_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=2, step=0))

    def test_correct_number_of_windows(self):
        a = np.arange(10)
        windows = list(sliding_window(a, size=3, step=1))
        assert len(windows) == 8  # 10-3+1

    def test_step_two(self):
        a = np.arange(10)
        windows = list(sliding_window(a, size=2, step=2))
        assert len(windows) == 5

    def test_window_content_correct(self):
        a = np.arange(5)
        windows = list(sliding_window(a, size=3, step=1))
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[1], [1, 2, 3])

    def test_size_equals_array_length(self):
        a = np.arange(5)
        windows = list(sliding_window(a, size=5))
        assert len(windows) == 1


# ─── flatten_images ───────────────────────────────────────────────────────────

class TestFlattenImagesExtra:
    def test_returns_ndarray(self):
        imgs = [_img(4, 4)]
        assert isinstance(flatten_images(imgs), np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            flatten_images([])

    def test_shape_n_by_d(self):
        imgs = [_img(4, 4) for _ in range(3)]
        out = flatten_images(imgs)
        assert out.shape == (3, 16)

    def test_shape_mismatch_raises(self):
        imgs = [_img(4, 4), _img(8, 8)]
        with pytest.raises(ValueError):
            flatten_images(imgs)

    def test_custom_dtype(self):
        imgs = [_img(4, 4)]
        out = flatten_images(imgs, dtype=np.float32)
        assert out.dtype == np.float32

    def test_values_preserved(self):
        a = np.arange(16, dtype=np.uint8).reshape(4, 4)
        out = flatten_images([a])
        np.testing.assert_array_equal(out[0], a.ravel())


# ─── unflatten_images ─────────────────────────────────────────────────────────

class TestUnflattenImagesExtra:
    def test_returns_list(self):
        mat = np.zeros((2, 16))
        assert isinstance(unflatten_images(mat, (4, 4)), list)

    def test_length_matches_rows(self):
        mat = np.zeros((3, 16))
        out = unflatten_images(mat, (4, 4))
        assert len(out) == 3

    def test_shape_matches_img_shape(self):
        mat = np.zeros((2, 16))
        for img in unflatten_images(mat, (4, 4)):
            assert img.shape == (4, 4)

    def test_dimension_mismatch_raises(self):
        mat = np.zeros((2, 9))
        with pytest.raises(ValueError):
            unflatten_images(mat, (4, 4))

    def test_roundtrip_flatten_unflatten(self):
        imgs = [np.arange(16, dtype=np.uint8).reshape(4, 4)]
        mat = flatten_images(imgs)
        recovered = unflatten_images(mat, (4, 4))
        np.testing.assert_array_equal(recovered[0], imgs[0])


# ─── compute_pairwise_norms ───────────────────────────────────────────────────

class TestComputePairwiseNormsExtra:
    def test_returns_ndarray(self):
        M = _mat()
        assert isinstance(compute_pairwise_norms(M), np.ndarray)

    def test_shape_n_by_n(self):
        M = _mat(4, 3)
        out = compute_pairwise_norms(M)
        assert out.shape == (4, 4)

    def test_diagonal_zero_l2(self):
        M = _mat()
        out = compute_pairwise_norms(M, "l2")
        np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)

    def test_diagonal_zero_l1(self):
        M = _mat()
        out = compute_pairwise_norms(M, "l1")
        np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)

    def test_diagonal_zero_cosine(self):
        M = _mat()
        out = compute_pairwise_norms(M, "cosine")
        np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)

    def test_symmetric_l2(self):
        M = _mat()
        out = compute_pairwise_norms(M, "l2")
        np.testing.assert_allclose(out, out.T, atol=1e-10)

    def test_l2_nonneg(self):
        M = _mat()
        assert (compute_pairwise_norms(M, "l2") >= 0).all()

    def test_cosine_in_range(self):
        M = _mat()
        out = compute_pairwise_norms(M, "cosine")
        assert (out >= 0).all() and (out <= 2.0 + 1e-9).all()

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            compute_pairwise_norms(_mat(), "euclidean")
