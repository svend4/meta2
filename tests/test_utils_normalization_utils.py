"""Tests for puzzle_reconstruction/utils/normalization_utils.py"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.normalization_utils import (
    l1_normalize,
    l2_normalize,
    minmax_normalize,
    zscore_normalize,
    softmax,
    clamp,
    symmetrize_matrix,
    zero_diagonal,
    normalize_rows,
    batch_l2_normalize,
)


# ─── l1_normalize ─────────────────────────────────────────────────────────────

class TestL1Normalize:
    def test_not_1d_raises(self):
        arr = np.ones((3, 3))
        with pytest.raises(ValueError):
            l1_normalize(arr)

    def test_zero_vector_returns_zeros(self):
        arr = np.zeros(5)
        result = l1_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_l1_norm_is_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = l1_normalize(arr)
        assert float(np.abs(result).sum()) == pytest.approx(1.0)

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3])
        result = l1_normalize(arr)
        assert result.dtype == np.float64

    def test_length_preserved(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = l1_normalize(arr)
        assert len(result) == 3

    def test_single_element(self):
        result = l1_normalize(np.array([5.0]))
        assert result[0] == pytest.approx(1.0)

    def test_negative_values(self):
        arr = np.array([-1.0, 2.0, -3.0])
        result = l1_normalize(arr)
        assert float(np.abs(result).sum()) == pytest.approx(1.0)


# ─── l2_normalize ─────────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_not_1d_raises(self):
        arr = np.ones((3, 3))
        with pytest.raises(ValueError):
            l2_normalize(arr)

    def test_zero_vector_returns_zeros(self):
        arr = np.zeros(5)
        result = l2_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_l2_norm_is_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = l2_normalize(arr)
        assert float(np.linalg.norm(result)) == pytest.approx(1.0)

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3])
        result = l2_normalize(arr)
        assert result.dtype == np.float64

    def test_length_preserved(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = l2_normalize(arr)
        assert len(result) == 3

    def test_direction_preserved(self):
        arr = np.array([3.0, 4.0])
        result = l2_normalize(arr)
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.8)


# ─── minmax_normalize ─────────────────────────────────────────────────────────

class TestMinmaxNormalize:
    def test_not_1d_raises(self):
        arr = np.ones((3, 3))
        with pytest.raises(ValueError):
            minmax_normalize(arr)

    def test_constant_array_returns_zeros(self):
        arr = np.full(5, 7.0)
        result = minmax_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_range_01(self):
        arr = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        result = minmax_normalize(arr)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3])
        result = minmax_normalize(arr)
        assert result.dtype == np.float64

    def test_monotone_preserved(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = minmax_normalize(arr)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_single_element_returns_zero(self):
        result = minmax_normalize(np.array([5.0]))
        assert float(result[0]) == pytest.approx(0.0)


# ─── zscore_normalize ────────────────────────────────────────────────────────

class TestZscoreNormalize:
    def test_not_1d_raises(self):
        arr = np.ones((3, 3))
        with pytest.raises(ValueError):
            zscore_normalize(arr)

    def test_constant_returns_zeros(self):
        arr = np.full(5, 3.0)
        result = zscore_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_zero_mean(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore_normalize(arr)
        assert float(result.mean()) == pytest.approx(0.0, abs=1e-9)

    def test_unit_std(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore_normalize(arr)
        assert float(result.std()) == pytest.approx(1.0, abs=1e-6)

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3])
        result = zscore_normalize(arr)
        assert result.dtype == np.float64

    def test_length_preserved(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = zscore_normalize(arr)
        assert len(result) == 3


# ─── softmax ──────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_not_1d_raises(self):
        arr = np.ones((3, 3))
        with pytest.raises(ValueError):
            softmax(arr)

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError):
            softmax(np.array([1.0, 2.0]), temperature=0.0)

    def test_temperature_negative_raises(self):
        with pytest.raises(ValueError):
            softmax(np.array([1.0, 2.0]), temperature=-1.0)

    def test_sums_to_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = softmax(arr)
        assert float(result.sum()) == pytest.approx(1.0)

    def test_all_in_01(self):
        arr = np.array([0.0, 1.0, 2.0, -1.0])
        result = softmax(arr)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_larger_input_gets_larger_prob(self):
        arr = np.array([0.0, 10.0])
        result = softmax(arr)
        assert result[1] > result[0]

    def test_uniform_input(self):
        arr = np.ones(5)
        result = softmax(arr)
        np.testing.assert_allclose(result, np.full(5, 0.2), atol=1e-9)

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3])
        result = softmax(arr)
        assert result.dtype == np.float64


# ─── clamp ───────────────────────────────────────────────────────────────────

class TestClamp:
    def test_lo_greater_than_hi_raises(self):
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            clamp(arr, lo=5.0, hi=1.0)

    def test_values_below_lo_clamped(self):
        arr = np.array([-5.0, 0.0, 5.0])
        result = clamp(arr, lo=0.0, hi=10.0)
        assert float(result[0]) == pytest.approx(0.0)

    def test_values_above_hi_clamped(self):
        arr = np.array([0.0, 5.0, 15.0])
        result = clamp(arr, lo=0.0, hi=10.0)
        assert float(result[-1]) == pytest.approx(10.0)

    def test_values_in_range_unchanged(self):
        arr = np.array([2.0, 5.0, 8.0])
        result = clamp(arr, lo=0.0, hi=10.0)
        np.testing.assert_allclose(result, arr)

    def test_any_shape(self):
        arr = np.ones((3, 4, 5)) * 20.0
        result = clamp(arr, lo=0.0, hi=10.0)
        assert float(result.max()) == pytest.approx(10.0)

    def test_lo_equals_hi(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = clamp(arr, lo=5.0, hi=5.0)
        assert np.all(result == pytest.approx(5.0))


# ─── symmetrize_matrix ───────────────────────────────────────────────────────

class TestSymmetrizeMatrix:
    def test_not_2d_raises(self):
        arr = np.ones((3,))
        with pytest.raises(ValueError):
            symmetrize_matrix(arr)

    def test_non_square_raises(self):
        arr = np.ones((3, 4))
        with pytest.raises(ValueError):
            symmetrize_matrix(arr)

    def test_result_is_symmetric(self):
        mat = np.array([[1.0, 2.0], [4.0, 8.0]])
        result = symmetrize_matrix(mat)
        np.testing.assert_allclose(result, result.T)

    def test_already_symmetric_unchanged(self):
        mat = np.array([[1.0, 2.0], [2.0, 4.0]])
        result = symmetrize_matrix(mat)
        np.testing.assert_allclose(result, mat)

    def test_dtype_float64(self):
        mat = np.eye(3)
        result = symmetrize_matrix(mat)
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        mat = np.eye(4)
        result = symmetrize_matrix(mat)
        assert result.shape == (4, 4)


# ─── zero_diagonal ───────────────────────────────────────────────────────────

class TestZeroDiagonal:
    def test_not_2d_raises(self):
        arr = np.ones((3,))
        with pytest.raises(ValueError):
            zero_diagonal(arr)

    def test_diagonal_all_zeros(self):
        mat = np.ones((4, 4))
        result = zero_diagonal(mat)
        np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_off_diagonal_unchanged(self):
        mat = np.ones((3, 3))
        result = zero_diagonal(mat)
        mask = ~np.eye(3, dtype=bool)
        np.testing.assert_array_equal(result[mask], np.ones(6))

    def test_returns_copy(self):
        mat = np.eye(3)
        result = zero_diagonal(mat)
        # Original should be unchanged
        assert mat[0, 0] == pytest.approx(1.0)

    def test_dtype_float64(self):
        mat = np.eye(3)
        result = zero_diagonal(mat)
        assert result.dtype == np.float64

    def test_non_square_works(self):
        mat = np.ones((2, 3))
        result = zero_diagonal(mat)
        assert result.shape == (2, 3)


# ─── normalize_rows ──────────────────────────────────────────────────────────

class TestNormalizeRows:
    def test_not_2d_raises(self):
        arr = np.ones((3,))
        with pytest.raises(ValueError):
            normalize_rows(arr)

    def test_unknown_method_raises(self):
        mat = np.ones((3, 3))
        with pytest.raises(ValueError):
            normalize_rows(mat, method="unknown")

    def test_l2_each_row_unit_norm(self):
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_rows(mat, method="l2")
        for row in result:
            assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-9)

    def test_l1_each_row_unit_l1_norm(self):
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_rows(mat, method="l1")
        for row in result:
            assert float(np.abs(row).sum()) == pytest.approx(1.0, abs=1e-9)

    def test_minmax_each_row_in_01(self):
        mat = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]])
        result = normalize_rows(mat, method="minmax")
        for row in result:
            assert float(row.max()) <= 1.0 + 1e-9
            assert float(row.min()) >= -1e-9

    def test_shape_preserved(self):
        mat = np.ones((3, 5))
        result = normalize_rows(mat)
        assert result.shape == (3, 5)

    def test_dtype_float64(self):
        mat = np.eye(3)
        result = normalize_rows(mat)
        assert result.dtype == np.float64


# ─── batch_l2_normalize ──────────────────────────────────────────────────────

class TestBatchL2Normalize:
    def test_empty_list(self):
        result = batch_l2_normalize([])
        assert result == []

    def test_length_preserved(self):
        vecs = [np.array([1.0, 2.0, 3.0]) for _ in range(5)]
        result = batch_l2_normalize(vecs)
        assert len(result) == 5

    def test_each_result_unit_norm(self):
        vecs = [np.array([3.0, 4.0]), np.array([1.0, 0.0, 0.0])]
        result = batch_l2_normalize(vecs)
        for v in result:
            assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-9)

    def test_zero_vector_returns_zeros(self):
        vecs = [np.zeros(3)]
        result = batch_l2_normalize(vecs)
        np.testing.assert_array_equal(result[0], np.zeros(3))

    def test_not_1d_raises(self):
        vecs = [np.ones((3, 3))]
        with pytest.raises(ValueError):
            batch_l2_normalize(vecs)

    def test_dtype_float64(self):
        vecs = [np.array([1, 2, 3])]
        result = batch_l2_normalize(vecs)
        assert result[0].dtype == np.float64
