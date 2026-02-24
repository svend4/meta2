"""Extra tests for puzzle_reconstruction/utils/array_utils.py."""
from __future__ import annotations

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


# ─── normalize_array (extra) ─────────────────────────────────────────────────

class TestNormalizeArrayExtra:
    def test_int_input_ok(self):
        a = np.array([0, 128, 255], dtype=np.int32)
        r = normalize_array(a)
        assert r.min() == pytest.approx(0.0)
        assert r.max() == pytest.approx(1.0)

    def test_two_elements(self):
        a = np.array([10.0, 20.0])
        r = normalize_array(a)
        assert r[0] == pytest.approx(0.0)
        assert r[1] == pytest.approx(1.0)

    def test_all_same_returns_low(self):
        a = np.full(10, 5.0)
        r = normalize_array(a, low=0.0, high=1.0)
        assert np.all(r == pytest.approx(0.0))

    def test_custom_range_minus_one_one(self):
        a = np.array([0.0, 50.0, 100.0])
        r = normalize_array(a, low=-1.0, high=1.0)
        assert r[1] == pytest.approx(0.0)

    def test_3d_preserved(self):
        a = np.random.rand(2, 3, 4)
        r = normalize_array(a)
        assert r.shape == (2, 3, 4)

    def test_float32_dtype(self):
        a = np.array([0.0, 1.0])
        r = normalize_array(a, dtype=np.float32)
        assert r.dtype == np.float32


# ─── pad_to_shape (extra) ────────────────────────────────────────────────────

class TestPadToShapeExtra:
    def test_1d_padding(self):
        a = np.ones(3)
        r = pad_to_shape(a, (5,), value=0.0)
        assert r.shape == (5,)

    def test_fill_value_correct(self):
        a = np.zeros((2, 2), dtype=np.float64)
        r = pad_to_shape(a, (4, 4), value=7.0)
        assert r[3, 3] == pytest.approx(7.0)
        assert r[0, 0] == pytest.approx(0.0)

    def test_center_alignment_symmetric(self):
        a = np.ones((2, 2), dtype=np.int32)
        r = pad_to_shape(a, (6, 6), value=0, align="center")
        # offset = (6-2)//2 = 2
        assert r[2, 2] == 1
        assert r[0, 0] == 0

    def test_same_shape_no_change(self):
        a = np.arange(6).reshape(2, 3)
        r = pad_to_shape(a, (2, 3))
        np.testing.assert_array_equal(r, a)

    def test_uint8_preserved(self):
        a = np.zeros((3, 3), dtype=np.uint8)
        r = pad_to_shape(a, (5, 5))
        assert r.dtype == np.uint8


# ─── crop_center (extra) ─────────────────────────────────────────────────────

class TestCropCenterExtra:
    def test_same_size_no_change(self):
        a = np.arange(9).reshape(3, 3)
        r = crop_center(a, (3, 3))
        np.testing.assert_array_equal(r, a)

    def test_1x1_center_pixel(self):
        a = np.arange(25).reshape(5, 5)
        r = crop_center(a, (1, 1))
        assert r[0, 0] == a[2, 2]

    def test_preserves_dtype(self):
        a = np.ones((8, 8), dtype=np.float32)
        r = crop_center(a, (4, 4))
        assert r.dtype == np.float32

    def test_even_crop_from_even(self):
        a = np.arange(16).reshape(4, 4)
        r = crop_center(a, (2, 2))
        assert r.shape == (2, 2)

    def test_3d_preserves_channels(self):
        a = np.ones((10, 10, 3), dtype=np.uint8)
        r = crop_center(a, (4, 4))
        assert r.shape == (4, 4, 3)


# ─── stack_arrays (extra) ────────────────────────────────────────────────────

class TestStackArraysExtra:
    def test_single_array(self):
        r = stack_arrays([np.ones((3, 3))])
        assert r.shape == (1, 3, 3)

    def test_many_arrays(self):
        arrays = [np.ones((2, 2)) * i for i in range(10)]
        r = stack_arrays(arrays)
        assert r.shape == (10, 2, 2)

    def test_different_dtypes_promoted(self):
        a1 = np.ones((2, 2), dtype=np.int32)
        a2 = np.ones((2, 2), dtype=np.float64)
        r = stack_arrays([a1, a2])
        assert r.shape == (2, 2, 2)

    def test_first_layer_values(self):
        a1 = np.ones((2, 2)) * 5
        a2 = np.ones((4, 4)) * 10
        r = stack_arrays([a1, a2], value=0.0)
        assert r[0, 0, 0] == pytest.approx(5.0)

    def test_padding_value_applied(self):
        a1 = np.ones((1, 1))
        a2 = np.ones((3, 3)) * 2
        r = stack_arrays([a1, a2], value=-1.0)
        assert r[0, 2, 2] == pytest.approx(-1.0)


# ─── chunk_array (extra) ─────────────────────────────────────────────────────

class TestChunkArrayExtra:
    def test_chunk_size_one(self):
        a = np.arange(5)
        chunks = chunk_array(a, chunk_size=1)
        assert len(chunks) == 5

    def test_all_chunks_correct_size(self):
        a = np.arange(12)
        chunks = chunk_array(a, chunk_size=4)
        for c in chunks:
            assert len(c) <= 4

    def test_2d_chunks_rows(self):
        a = np.ones((6, 3))
        chunks = chunk_array(a, chunk_size=2, axis=0)
        assert len(chunks) == 3
        assert chunks[0].shape == (2, 3)

    def test_reconstruction_matches(self):
        a = np.arange(17)
        chunks = chunk_array(a, chunk_size=5)
        np.testing.assert_array_equal(np.concatenate(chunks), a)

    def test_large_chunk_size(self):
        a = np.arange(3)
        chunks = chunk_array(a, chunk_size=1000)
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], a)


# ─── sliding_window (extra) ──────────────────────────────────────────────────

class TestSlidingWindowExtra:
    def test_step_equals_size(self):
        a = np.arange(10)
        wins = list(sliding_window(a, size=5, step=5))
        assert len(wins) == 2

    def test_single_window(self):
        a = np.arange(5)
        wins = list(sliding_window(a, size=5, step=1))
        assert len(wins) == 1

    def test_window_values(self):
        a = np.arange(6)
        wins = list(sliding_window(a, size=3, step=1))
        np.testing.assert_array_equal(wins[0], [0, 1, 2])
        np.testing.assert_array_equal(wins[1], [1, 2, 3])

    def test_large_step(self):
        a = np.arange(20)
        wins = list(sliding_window(a, size=3, step=10))
        assert len(wins) == 2

    def test_2d_axis_0(self):
        a = np.ones((10, 5))
        wins = list(sliding_window(a, size=4, step=3, axis=0))
        for w in wins:
            assert w.shape == (4, 5)


# ─── flatten_images (extra) ──────────────────────────────────────────────────

class TestFlattenImagesExtra:
    def test_dtype_preserved(self):
        imgs = [np.ones((3, 3), dtype=np.uint8)]
        m = flatten_images(imgs)
        assert m.dtype == np.uint8

    def test_round_trip_values(self):
        img = np.arange(12, dtype=np.float32).reshape(3, 4)
        m = flatten_images([img])
        np.testing.assert_array_equal(m[0], img.ravel())

    def test_multiple_images(self):
        imgs = [np.ones((2, 2), dtype=np.float32) * i for i in range(5)]
        m = flatten_images(imgs)
        assert m.shape == (5, 4)

    def test_3d_flattened_correctly(self):
        imgs = [np.ones((2, 3, 4), dtype=np.uint8)] * 2
        m = flatten_images(imgs)
        assert m.shape == (2, 24)

    def test_custom_dtype_conversion(self):
        imgs = [np.ones((2, 2), dtype=np.uint8) * 255]
        m = flatten_images(imgs, dtype=np.float64)
        assert m.dtype == np.float64
        assert m[0, 0] == pytest.approx(255.0)


# ─── unflatten_images (extra) ────────────────────────────────────────────────

class TestUnflattenImagesExtra:
    def test_round_trip(self):
        imgs = [np.arange(6, dtype=np.float32).reshape(2, 3) for _ in range(3)]
        flat = flatten_images(imgs)
        restored = unflatten_images(flat, (2, 3))
        for orig, rest in zip(imgs, restored):
            np.testing.assert_array_equal(orig, rest)

    def test_3d_shape(self):
        m = np.zeros((2, 24))
        imgs = unflatten_images(m, (2, 3, 4))
        assert imgs[0].shape == (2, 3, 4)

    def test_single_image(self):
        m = np.arange(9, dtype=float).reshape(1, 9)
        imgs = unflatten_images(m, (3, 3))
        assert len(imgs) == 1
        assert imgs[0].shape == (3, 3)

    def test_many_images(self):
        m = np.zeros((10, 16))
        imgs = unflatten_images(m, (4, 4))
        assert len(imgs) == 10

    def test_values_preserved(self):
        vals = np.arange(4, dtype=float).reshape(1, 4)
        imgs = unflatten_images(vals, (2, 2))
        np.testing.assert_array_equal(imgs[0].ravel(), [0, 1, 2, 3])


# ─── compute_pairwise_norms (extra) ──────────────────────────────────────────

class TestComputePairwiseNormsExtra:
    def test_l2_nonneg(self):
        m = np.random.rand(4, 5)
        d = compute_pairwise_norms(m, metric="l2")
        assert (d >= -1e-10).all()

    def test_l1_nonneg(self):
        m = np.random.rand(4, 5)
        d = compute_pairwise_norms(m, metric="l1")
        assert (d >= -1e-10).all()

    def test_l1_symmetric(self):
        m = np.random.rand(5, 8)
        d = compute_pairwise_norms(m, metric="l1")
        np.testing.assert_allclose(d, d.T, atol=1e-10)

    def test_cosine_symmetric(self):
        m = np.random.rand(5, 6) + 0.1
        d = compute_pairwise_norms(m, metric="cosine")
        np.testing.assert_allclose(d, d.T, atol=1e-10)

    def test_single_row(self):
        m = np.array([[1.0, 2.0, 3.0]])
        d = compute_pairwise_norms(m, metric="l2")
        assert d.shape == (1, 1)
        assert d[0, 0] == pytest.approx(0.0)

    def test_two_rows_l2(self):
        m = np.array([[0.0, 0.0], [1.0, 0.0]])
        d = compute_pairwise_norms(m, metric="l2")
        assert d[0, 1] == pytest.approx(1.0)
        assert d[1, 0] == pytest.approx(1.0)
