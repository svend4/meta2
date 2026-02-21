"""
Тесты для puzzle_reconstruction.utils.array_utils.
"""
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
    def test_default_range_zero_one(self):
        a = np.array([0.0, 100.0, 255.0])
        r = normalize_array(a)
        assert r.min() == pytest.approx(0.0)
        assert r.max() == pytest.approx(1.0)

    def test_custom_range(self):
        a = np.array([0.0, 50.0, 100.0])
        r = normalize_array(a, low=-1.0, high=1.0)
        assert r.min() == pytest.approx(-1.0)
        assert r.max() == pytest.approx(1.0)

    def test_flat_array_returns_low(self):
        a = np.full((4, 4), 128.0)
        r = normalize_array(a, low=0.5)
        assert np.all(r == pytest.approx(0.5))

    def test_output_dtype_float64_by_default(self):
        a = np.array([0, 128, 255], dtype=np.uint8)
        r = normalize_array(a)
        assert r.dtype == np.float64

    def test_custom_dtype(self):
        a = np.array([0.0, 1.0, 2.0])
        r = normalize_array(a, dtype=np.float32)
        assert r.dtype == np.float32

    def test_preserves_shape(self):
        a = np.random.rand(3, 4, 5)
        r = normalize_array(a)
        assert r.shape == (3, 4, 5)

    def test_2d_array(self):
        a = np.arange(16, dtype=float).reshape(4, 4)
        r = normalize_array(a)
        assert r.min() == pytest.approx(0.0)
        assert r.max() == pytest.approx(1.0)

    def test_single_element(self):
        a = np.array([42.0])
        r = normalize_array(a)
        assert r[0] == pytest.approx(0.0)

    def test_negative_values(self):
        a = np.array([-100.0, 0.0, 100.0])
        r = normalize_array(a)
        assert r[0] == pytest.approx(0.0)
        assert r[-1] == pytest.approx(1.0)


# ─── pad_to_shape ─────────────────────────────────────────────────────────────

class TestPadToShape:
    def test_no_padding_needed(self):
        a = np.ones((4, 4), dtype=np.uint8)
        r = pad_to_shape(a, (4, 4))
        np.testing.assert_array_equal(r, a)

    def test_topleft_origin(self):
        a = np.ones((2, 3), dtype=np.uint8) * 5
        r = pad_to_shape(a, (4, 5), value=0)
        assert r[0, 0] == 5
        assert r[0, 2] == 5
        assert r[3, 4] == 0
        assert r[2, 0] == 0

    def test_center_alignment(self):
        a = np.ones((2, 2), dtype=np.int32) * 9
        r = pad_to_shape(a, (4, 4), value=0, align="center")
        # Центр: offset = (4-2)//2 = 1
        assert r[1, 1] == 9
        assert r[2, 2] == 9
        assert r[0, 0] == 0

    def test_preserves_dtype(self):
        a = np.zeros((3, 3), dtype=np.float32)
        r = pad_to_shape(a, (5, 5))
        assert r.dtype == np.float32

    def test_output_shape_correct(self):
        a = np.ones((3, 4))
        r = pad_to_shape(a, (7, 9))
        assert r.shape == (7, 9)

    def test_3d_array(self):
        a = np.ones((2, 2, 3), dtype=np.uint8) * 255
        r = pad_to_shape(a, (4, 4, 3))
        assert r.shape == (4, 4, 3)
        assert r[0, 0, 0] == 255
        assert r[3, 3, 0] == 0

    def test_too_large_raises(self):
        a = np.ones((5, 5))
        with pytest.raises(ValueError):
            pad_to_shape(a, (3, 3))

    def test_fill_value(self):
        a = np.zeros((2, 2), dtype=np.float32)
        r = pad_to_shape(a, (4, 4), value=99.0)
        # Верхний левый квадрат 2×2 должен быть 0
        assert r[0, 0] == pytest.approx(0.0)
        # Остальные — 99.0
        assert r[3, 3] == pytest.approx(99.0)


# ─── crop_center ──────────────────────────────────────────────────────────────

class TestCropCenter:
    def test_exact_size(self):
        a = np.ones((8, 8))
        r = crop_center(a, (8, 8))
        assert r.shape == (8, 8)

    def test_crop_shape(self):
        a = np.zeros((16, 20))
        r = crop_center(a, (4, 6))
        assert r.shape == (4, 6)

    def test_center_values(self):
        a = np.zeros((10, 10), dtype=np.int32)
        a[5, 5] = 1  # Центральный пиксель
        r = crop_center(a, (1, 1))
        assert r[0, 0] == 1

    def test_3d_array(self):
        a = np.ones((16, 16, 3), dtype=np.uint8)
        r = crop_center(a, (8, 8))
        assert r.shape == (8, 8, 3)

    def test_crop_too_large_raises(self):
        a = np.zeros((4, 4))
        with pytest.raises(ValueError):
            crop_center(a, (8, 4))

    def test_crop_height_too_large_raises(self):
        a = np.zeros((4, 8))
        with pytest.raises(ValueError):
            crop_center(a, (5, 4))

    def test_odd_size_offset(self):
        a = np.arange(25).reshape(5, 5)
        r = crop_center(a, (3, 3))
        # Offset = (5-3)//2 = 1
        np.testing.assert_array_equal(r, a[1:4, 1:4])


# ─── stack_arrays ─────────────────────────────────────────────────────────────

class TestStackArrays:
    def test_same_size_no_padding(self):
        arrays = [np.ones((4, 4)) * i for i in range(3)]
        r = stack_arrays(arrays)
        assert r.shape == (3, 4, 4)

    def test_different_sizes_padded(self):
        a1 = np.ones((2, 3))
        a2 = np.ones((4, 5))
        r  = stack_arrays([a1, a2])
        assert r.shape == (2, 4, 5)

    def test_fill_value_in_padding(self):
        a1 = np.ones((2, 2), dtype=np.float32)
        a2 = np.ones((4, 4), dtype=np.float32) * 2
        r  = stack_arrays([a1, a2], value=-1.0)
        # В первом слое за пределами (2,2) должно быть -1
        assert r[0, 3, 3] == pytest.approx(-1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_arrays([])

    def test_axis_0_adds_new_dimension(self):
        arrays = [np.ones((3, 3)) for _ in range(5)]
        r = stack_arrays(arrays, axis=0)
        assert r.shape == (5, 3, 3)

    def test_3d_arrays(self):
        a1 = np.ones((4, 4, 3), dtype=np.uint8)
        a2 = np.ones((6, 8, 3), dtype=np.uint8) * 2
        r  = stack_arrays([a1, a2])
        assert r.shape == (2, 6, 8, 3)


# ─── chunk_array ──────────────────────────────────────────────────────────────

class TestChunkArray:
    def test_equal_chunks(self):
        a = np.arange(12)
        chunks = chunk_array(a, chunk_size=4)
        assert len(chunks) == 3
        assert len(chunks[0]) == 4

    def test_unequal_last_chunk(self):
        a = np.arange(10)
        chunks = chunk_array(a, chunk_size=3)
        assert len(chunks) == 4
        assert len(chunks[-1]) == 1

    def test_chunk_size_equals_length(self):
        a = np.arange(5)
        chunks = chunk_array(a, chunk_size=5)
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], a)

    def test_chunk_size_larger_than_array(self):
        a = np.arange(3)
        chunks = chunk_array(a, chunk_size=10)
        assert len(chunks) == 1

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError):
            chunk_array(np.arange(5), chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError):
            chunk_array(np.arange(5), chunk_size=-1)

    def test_2d_array_axis_0(self):
        a = np.ones((10, 4))
        chunks = chunk_array(a, chunk_size=3, axis=0)
        assert len(chunks) == 4
        assert chunks[0].shape == (3, 4)
        assert chunks[-1].shape == (1, 4)

    def test_2d_array_axis_1(self):
        a = np.ones((3, 8))
        chunks = chunk_array(a, chunk_size=4, axis=1)
        assert len(chunks) == 2
        assert chunks[0].shape == (3, 4)

    def test_concatenation_equals_original(self):
        a = np.arange(13)
        chunks = chunk_array(a, chunk_size=4)
        np.testing.assert_array_equal(np.concatenate(chunks), a)


# ─── sliding_window ───────────────────────────────────────────────────────────

class TestSlidingWindow:
    def test_basic_step1(self):
        a = np.arange(5)
        windows = list(sliding_window(a, size=3, step=1))
        assert len(windows) == 3
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[-1], [2, 3, 4])

    def test_step_2(self):
        a = np.arange(10)
        windows = list(sliding_window(a, size=4, step=2))
        assert len(windows) == 4

    def test_size_equals_length(self):
        a = np.arange(5)
        windows = list(sliding_window(a, size=5, step=1))
        assert len(windows) == 1

    def test_size_larger_than_array_empty(self):
        a = np.arange(3)
        windows = list(sliding_window(a, size=5, step=1))
        assert len(windows) == 0

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=0))

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError):
            list(sliding_window(np.arange(5), size=2, step=0))

    def test_2d_array_axis_0(self):
        a = np.ones((8, 4))
        windows = list(sliding_window(a, size=3, step=2, axis=0))
        assert len(windows) == 3
        assert windows[0].shape == (3, 4)

    def test_window_contents_correct(self):
        a = np.arange(6)
        windows = list(sliding_window(a, size=3, step=2))
        np.testing.assert_array_equal(windows[0], [0, 1, 2])
        np.testing.assert_array_equal(windows[1], [2, 3, 4])


# ─── flatten_images ───────────────────────────────────────────────────────────

class TestFlattenImages:
    def test_basic_2d(self):
        imgs = [np.ones((4, 4), dtype=np.uint8) * i for i in range(3)]
        m = flatten_images(imgs)
        assert m.shape == (3, 16)

    def test_basic_3d(self):
        imgs = [np.ones((4, 4, 3), dtype=np.uint8)] * 2
        m = flatten_images(imgs)
        assert m.shape == (2, 48)

    def test_single_image(self):
        img = np.arange(9, dtype=np.float32).reshape(3, 3)
        m   = flatten_images([img])
        assert m.shape == (1, 9)
        np.testing.assert_array_equal(m[0], img.ravel())

    def test_custom_dtype(self):
        imgs = [np.ones((2, 2), dtype=np.uint8)] * 2
        m    = flatten_images(imgs, dtype=np.float32)
        assert m.dtype == np.float32

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            flatten_images([])

    def test_different_shapes_raises(self):
        imgs = [np.ones((4, 4)), np.ones((3, 4))]
        with pytest.raises(ValueError):
            flatten_images(imgs)

    def test_row_values_correct(self):
        img0 = np.zeros((2, 2), dtype=np.uint8)
        img1 = np.full((2, 2), 255, dtype=np.uint8)
        m    = flatten_images([img0, img1])
        assert np.all(m[0] == 0)
        assert np.all(m[1] == 255)


# ─── unflatten_images ─────────────────────────────────────────────────────────

class TestUnflattenImages:
    def test_basic_round_trip(self):
        imgs_in  = [np.arange(9, dtype=np.float32).reshape(3, 3) for _ in range(4)]
        flat     = flatten_images(imgs_in, dtype=np.float32)
        imgs_out = unflatten_images(flat, (3, 3))
        for a, b in zip(imgs_in, imgs_out):
            np.testing.assert_array_equal(a, b)

    def test_output_length_matches(self):
        m = np.ones((5, 12))
        imgs = unflatten_images(m, (3, 4))
        assert len(imgs) == 5

    def test_output_shape_correct(self):
        m = np.zeros((3, 48))
        imgs = unflatten_images(m, (4, 4, 3))
        assert imgs[0].shape == (4, 4, 3)

    def test_wrong_d_raises(self):
        m = np.ones((3, 10))
        with pytest.raises(ValueError):
            unflatten_images(m, (3, 3))  # 3*3=9 ≠ 10

    def test_single_row(self):
        m = np.arange(6, dtype=float).reshape(1, 6)
        imgs = unflatten_images(m, (2, 3))
        assert imgs[0].shape == (2, 3)
        np.testing.assert_array_equal(imgs[0].ravel(), np.arange(6.0))


# ─── compute_pairwise_norms ───────────────────────────────────────────────────

class TestComputePairwiseNorms:
    def test_l2_diagonal_zero(self):
        m = np.random.rand(4, 10).astype(np.float64)
        d = compute_pairwise_norms(m, metric="l2")
        np.testing.assert_array_almost_equal(np.diag(d), 0.0)

    def test_l2_symmetric(self):
        m = np.random.rand(5, 8)
        d = compute_pairwise_norms(m, metric="l2")
        np.testing.assert_array_almost_equal(d, d.T)

    def test_l2_shape(self):
        m = np.random.rand(6, 10)
        d = compute_pairwise_norms(m, metric="l2")
        assert d.shape == (6, 6)

    def test_l2_known_values(self):
        m = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = compute_pairwise_norms(m, metric="l2")
        assert d[0, 1] == pytest.approx(5.0)

    def test_l1_diagonal_zero(self):
        m = np.random.rand(4, 6)
        d = compute_pairwise_norms(m, metric="l1")
        np.testing.assert_array_almost_equal(np.diag(d), 0.0)

    def test_l1_known_values(self):
        m = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = compute_pairwise_norms(m, metric="l1")
        assert d[0, 1] == pytest.approx(7.0)

    def test_cosine_diagonal_zero(self):
        m = np.random.rand(4, 8) + 0.1
        d = compute_pairwise_norms(m, metric="cosine")
        np.testing.assert_array_almost_equal(np.diag(d), 0.0, decimal=5)

    def test_cosine_range_zero_two(self):
        m = np.random.rand(5, 6) + 0.1
        d = compute_pairwise_norms(m, metric="cosine")
        assert np.all(d >= -1e-9)
        assert np.all(d <= 2.0 + 1e-9)

    def test_cosine_identical_rows_zero_dist(self):
        row = np.array([[1.0, 2.0, 3.0]])
        m   = np.tile(row, (4, 1))
        d   = compute_pairwise_norms(m, metric="cosine")
        np.testing.assert_array_almost_equal(d, 0.0, decimal=5)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            compute_pairwise_norms(np.ones((3, 3)), metric="unknown")

    def test_dtype_float64(self):
        m = np.random.rand(3, 4).astype(np.float32)
        d = compute_pairwise_norms(m, metric="l2")
        assert d.dtype == np.float64
