"""Tests for puzzle_reconstruction/utils/array_utils.py"""
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


# ─── normalize_array ──────────────────────────────────────────────────────────

class TestNormalizeArray:
    def test_basic_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_array(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_custom_range(self):
        arr = np.array([0.0, 10.0])
        result = normalize_array(arr, low=2.0, high=8.0)
        assert result[0] == pytest.approx(2.0)
        assert result[-1] == pytest.approx(8.0)

    def test_constant_array_returns_low(self):
        arr = np.full((5,), 42.0)
        result = normalize_array(arr, low=0.0, high=1.0)
        assert np.allclose(result, 0.0)

    def test_output_dtype_default_float64(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = normalize_array(arr)
        assert result.dtype == np.float64

    def test_output_dtype_custom(self):
        arr = np.array([0.0, 1.0, 2.0])
        result = normalize_array(arr, dtype=np.float32)
        assert result.dtype == np.float32

    def test_shape_preserved(self):
        arr = np.arange(12).reshape(3, 4)
        result = normalize_array(arr)
        assert result.shape == (3, 4)

    def test_min_value_is_low(self):
        arr = np.array([5.0, 10.0, 20.0])
        result = normalize_array(arr, low=-1.0, high=1.0)
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_integer_input(self):
        arr = np.array([0, 100, 200, 255], dtype=np.uint8)
        result = normalize_array(arr)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_single_element(self):
        arr = np.array([7.0])
        result = normalize_array(arr, low=3.0, high=9.0)
        assert result[0] == pytest.approx(3.0)  # constant → low


# ─── pad_to_shape ─────────────────────────────────────────────────────────────

class TestPadToShape:
    def test_basic_topleft(self):
        arr = np.ones((3, 3), dtype=np.float32)
        result = pad_to_shape(arr, (5, 5))
        assert result.shape == (5, 5)
        assert (result[:3, :3] == 1.0).all()
        assert result[4, 4] == 0.0

    def test_basic_center(self):
        arr = np.ones((2, 2), dtype=np.float32)
        result = pad_to_shape(arr, (4, 4), align="center")
        assert result.shape == (4, 4)
        assert result[1, 1] == 1.0  # center region
        assert result[0, 0] == 0.0  # corner → fill

    def test_no_padding_needed(self):
        arr = np.eye(3, dtype=np.float32)
        result = pad_to_shape(arr, (3, 3))
        np.testing.assert_array_equal(result, arr)

    def test_custom_fill_value(self):
        arr = np.zeros((2, 2), dtype=np.float32)
        result = pad_to_shape(arr, (4, 4), value=9.0)
        assert result[3, 3] == pytest.approx(9.0)

    def test_dtype_preserved(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = pad_to_shape(arr, (4, 4))
        assert result.dtype == np.uint8

    def test_exceeds_target_raises(self):
        arr = np.ones((5, 5))
        with pytest.raises(ValueError):
            pad_to_shape(arr, (3, 3))

    def test_3d_array(self):
        arr = np.ones((2, 2, 3), dtype=np.uint8)
        result = pad_to_shape(arr, (4, 4, 3))
        assert result.shape == (4, 4, 3)
        assert (result[:2, :2, :] == 1).all()


# ─── crop_center ──────────────────────────────────────────────────────────────

class TestCropCenter:
    def test_basic_2d(self):
        arr = np.arange(100).reshape(10, 10)
        cropped = crop_center(arr, (4, 4))
        assert cropped.shape == (4, 4)

    def test_center_value(self):
        arr = np.zeros((10, 10))
        arr[5, 5] = 99
        cropped = crop_center(arr, (3, 3))
        # center pixel (5,5) should be in the cropped region
        assert 99 in cropped

    def test_exact_size(self):
        arr = np.eye(5)
        cropped = crop_center(arr, (5, 5))
        np.testing.assert_array_equal(cropped, arr)

    def test_3d_array(self):
        arr = np.ones((10, 10, 3), dtype=np.uint8)
        cropped = crop_center(arr, (4, 6))
        assert cropped.shape == (4, 6, 3)

    def test_exceeds_size_raises(self):
        arr = np.ones((5, 5))
        with pytest.raises(ValueError):
            crop_center(arr, (6, 4))

    def test_exceeds_width_raises(self):
        arr = np.ones((5, 5))
        with pytest.raises(ValueError):
            crop_center(arr, (4, 6))

    def test_1x1_crop(self):
        arr = np.arange(9).reshape(3, 3)
        cropped = crop_center(arr, (1, 1))
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == arr[1, 1]  # center element


# ─── stack_arrays ─────────────────────────────────────────────────────────────

class TestStackArrays:
    def test_same_size_arrays(self):
        arrays = [np.ones((3, 3)) * i for i in range(4)]
        result = stack_arrays(arrays, axis=0)
        assert result.shape == (4, 3, 3)

    def test_different_sizes_padded(self):
        a = np.ones((2, 2))
        b = np.ones((4, 3))
        result = stack_arrays([a, b], axis=0)
        assert result.shape == (2, 4, 3)

    def test_fill_value(self):
        a = np.ones((2, 2))
        b = np.ones((4, 4))
        result = stack_arrays([a, b], axis=0, value=-1.0)
        # a is padded to (4,4), so padded region should be -1
        assert result[0, 3, 3] == -1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_arrays([])

    def test_single_array(self):
        a = np.ones((3, 4))
        result = stack_arrays([a])
        assert result.shape == (1, 3, 4)

    def test_output_shape_correct(self):
        arrays = [np.zeros((i + 1, 5)) for i in range(3)]
        result = stack_arrays(arrays, axis=0)
        assert result.shape[0] == 3
        assert result.shape[1] == 3   # max height
        assert result.shape[2] == 5


# ─── chunk_array ──────────────────────────────────────────────────────────────

class TestChunkArray:
    def test_even_split(self):
        arr = np.arange(12)
        chunks = chunk_array(arr, chunk_size=4)
        assert len(chunks) == 3
        np.testing.assert_array_equal(chunks[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(chunks[2], [8, 9, 10, 11])

    def test_last_chunk_smaller(self):
        arr = np.arange(10)
        chunks = chunk_array(arr, chunk_size=4)
        assert len(chunks) == 3
        assert len(chunks[-1]) == 2

    def test_chunk_size_one(self):
        arr = np.arange(5)
        chunks = chunk_array(arr, chunk_size=1)
        assert len(chunks) == 5

    def test_chunk_size_larger_than_array(self):
        arr = np.arange(3)
        chunks = chunk_array(arr, chunk_size=10)
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], arr)

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError):
            chunk_array(np.arange(5), chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError):
            chunk_array(np.arange(5), chunk_size=-2)

    def test_axis_1(self):
        arr = np.ones((4, 9))
        chunks = chunk_array(arr, chunk_size=3, axis=1)
        assert len(chunks) == 3
        for c in chunks:
            assert c.shape == (4, 3)

    def test_2d_array_axis_0(self):
        arr = np.arange(12).reshape(4, 3)
        chunks = chunk_array(arr, chunk_size=2)
        assert len(chunks) == 2
        assert chunks[0].shape == (2, 3)


# ─── sliding_window ───────────────────────────────────────────────────────────

class TestSlidingWindow:
    def test_basic(self):
        arr = np.arange(10)
        windows = list(sliding_window(arr, size=3))
        assert len(windows) == 8
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[-1], [7, 8, 9])

    def test_step_2(self):
        arr = np.arange(10)
        windows = list(sliding_window(arr, size=3, step=2))
        # starts: 0, 2, 4, 6
        assert len(windows) == 4

    def test_window_size_equals_array(self):
        arr = np.arange(5)
        windows = list(sliding_window(arr, size=5))
        assert len(windows) == 1
        np.testing.assert_array_equal(windows[0], arr)

    def test_window_larger_than_array(self):
        arr = np.arange(3)
        windows = list(sliding_window(arr, size=5))
        assert len(windows) == 0

    def test_size_zero_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=0))

    def test_step_zero_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=2, step=0))

    def test_size_negative_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=-1))

    def test_axis_1(self):
        arr = np.arange(12).reshape(3, 4)
        windows = list(sliding_window(arr, size=2, axis=1))
        assert len(windows) == 3
        for w in windows:
            assert w.shape == (3, 2)

    def test_window_content_correct(self):
        arr = np.array([10, 20, 30, 40, 50])
        windows = list(sliding_window(arr, size=2, step=2))
        np.testing.assert_array_equal(windows[0], [10, 20])
        np.testing.assert_array_equal(windows[1], [30, 40])


# ─── flatten_images ───────────────────────────────────────────────────────────

class TestFlattenImages:
    def test_basic(self):
        imgs = [np.ones((4, 4), dtype=np.uint8) * i for i in range(3)]
        mat = flatten_images(imgs)
        assert mat.shape == (3, 16)

    def test_values_correct(self):
        img0 = np.arange(9, dtype=np.float32).reshape(3, 3)
        img1 = np.zeros((3, 3), dtype=np.float32)
        mat = flatten_images([img0, img1])
        np.testing.assert_array_equal(mat[0], img0.ravel())
        np.testing.assert_array_equal(mat[1], img1.ravel())

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            flatten_images([])

    def test_shape_mismatch_raises(self):
        imgs = [np.ones((4, 4)), np.ones((3, 4))]
        with pytest.raises(ValueError):
            flatten_images(imgs)

    def test_custom_dtype(self):
        imgs = [np.ones((2, 2), dtype=np.uint8)]
        mat = flatten_images(imgs, dtype=np.float64)
        assert mat.dtype == np.float64

    def test_3d_images(self):
        imgs = [np.ones((4, 4, 3), dtype=np.uint8) for _ in range(5)]
        mat = flatten_images(imgs)
        assert mat.shape == (5, 48)  # 4*4*3 = 48


# ─── unflatten_images ─────────────────────────────────────────────────────────

class TestUnflattenImages:
    def test_basic_roundtrip(self):
        imgs = [np.arange(9, dtype=np.float32).reshape(3, 3) + i for i in range(4)]
        mat = flatten_images(imgs)
        restored = unflatten_images(mat, (3, 3))
        for orig, rest in zip(imgs, restored):
            np.testing.assert_array_equal(orig, rest)

    def test_output_length(self):
        mat = np.zeros((5, 16))
        imgs = unflatten_images(mat, (4, 4))
        assert len(imgs) == 5

    def test_output_shape(self):
        mat = np.zeros((3, 12))
        imgs = unflatten_images(mat, (3, 4))
        for img in imgs:
            assert img.shape == (3, 4)

    def test_shape_mismatch_raises(self):
        mat = np.zeros((2, 9))
        with pytest.raises(ValueError):
            unflatten_images(mat, (3, 4))  # 3*4=12 != 9

    def test_3d_shape(self):
        mat = np.zeros((2, 24))
        imgs = unflatten_images(mat, (2, 4, 3))
        for img in imgs:
            assert img.shape == (2, 4, 3)


# ─── compute_pairwise_norms ───────────────────────────────────────────────────

class TestComputePairwiseNorms:
    def _make_matrix(self):
        return np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def test_l2_diagonal_zero(self):
        mat = self._make_matrix()
        dist = compute_pairwise_norms(mat, metric="l2")
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)

    def test_l2_symmetric(self):
        mat = self._make_matrix()
        dist = compute_pairwise_norms(mat, metric="l2")
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_l2_known_value(self):
        mat = np.array([[0.0, 0.0], [3.0, 4.0]])
        dist = compute_pairwise_norms(mat, metric="l2")
        assert dist[0, 1] == pytest.approx(5.0)
        assert dist[1, 0] == pytest.approx(5.0)

    def test_l1_known_value(self):
        mat = np.array([[0.0, 0.0], [3.0, 4.0]])
        dist = compute_pairwise_norms(mat, metric="l1")
        assert dist[0, 1] == pytest.approx(7.0)

    def test_cosine_same_vector_zero(self):
        mat = np.array([[1.0, 0.0], [1.0, 0.0]])
        dist = compute_pairwise_norms(mat, metric="cosine")
        assert dist[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_cosine_orthogonal_vectors(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        dist = compute_pairwise_norms(mat, metric="cosine")
        assert dist[0, 1] == pytest.approx(1.0, abs=1e-9)

    def test_output_shape(self):
        mat = np.random.rand(5, 3)
        dist = compute_pairwise_norms(mat)
        assert dist.shape == (5, 5)

    def test_unknown_metric_raises(self):
        mat = np.ones((3, 3))
        with pytest.raises(ValueError):
            compute_pairwise_norms(mat, metric="euclidean")

    def test_output_dtype_float64(self):
        mat = np.ones((3, 2))
        dist = compute_pairwise_norms(mat)
        assert dist.dtype == np.float64

    def test_non_negative(self):
        mat = np.random.rand(4, 5)
        for metric in ("l2", "l1", "cosine"):
            dist = compute_pairwise_norms(mat, metric=metric)
            assert (dist >= 0.0).all()
