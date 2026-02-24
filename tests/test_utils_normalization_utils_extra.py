"""Extra tests for puzzle_reconstruction/utils/normalization_utils.py"""
import numpy as np
import pytest

from puzzle_reconstruction.utils.normalization_utils import (
    batch_l2_normalize,
    clamp,
    l1_normalize,
    l2_normalize,
    minmax_normalize,
    normalize_rows,
    softmax,
    symmetrize_matrix,
    zero_diagonal,
    zscore_normalize,
)


# ─── TestL1NormalizeExtra ─────────────────────────────────────────────────────

class TestL1NormalizeExtra:
    def test_large_vector(self):
        arr = np.arange(1, 101, dtype=float)
        result = l1_normalize(arr)
        assert abs(float(np.abs(result).sum()) - 1.0) < 1e-9

    def test_all_ones(self):
        arr = np.ones(10)
        result = l1_normalize(arr)
        assert abs(float(np.abs(result).sum()) - 1.0) < 1e-9

    def test_single_neg(self):
        result = l1_normalize(np.array([-7.0]))
        assert abs(float(np.abs(result).sum()) - 1.0) < 1e-9

    def test_two_elements(self):
        result = l1_normalize(np.array([3.0, 7.0]))
        assert abs(float(np.abs(result).sum()) - 1.0) < 1e-9

    def test_float32_input(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = l1_normalize(arr)
        assert result.dtype == np.float64

    def test_all_zero_returns_zeros(self):
        arr = np.zeros(8)
        result = l1_normalize(arr)
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_preserves_sign_ratios(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = l1_normalize(arr)
        assert result[0] < result[1] < result[2]


# ─── TestL2NormalizeExtra ─────────────────────────────────────────────────────

class TestL2NormalizeExtra:
    def test_large_vector(self):
        arr = np.arange(1, 101, dtype=float)
        result = l2_normalize(arr)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-9

    def test_all_ones_50(self):
        arr = np.ones(50)
        result = l2_normalize(arr)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-9

    def test_single_positive(self):
        result = l2_normalize(np.array([5.0]))
        assert result[0] == pytest.approx(1.0)

    def test_single_negative(self):
        result = l2_normalize(np.array([-5.0]))
        assert result[0] == pytest.approx(-1.0)

    def test_float32_output_is_float64(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = l2_normalize(arr)
        assert result.dtype == np.float64

    def test_all_zero_returns_zeros(self):
        result = l2_normalize(np.zeros(5))
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_unit_vector_unchanged(self):
        arr = np.array([0.6, 0.8])
        result = l2_normalize(arr)
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.8)


# ─── TestMinmaxNormalizeExtra ─────────────────────────────────────────────────

class TestMinmaxNormalizeExtra:
    def test_range_is_0_to_1(self):
        arr = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        result = minmax_normalize(arr)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_negative_values(self):
        arr = np.array([-5.0, 0.0, 5.0])
        result = minmax_normalize(arr)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_all_same_returns_zeros(self):
        result = minmax_normalize(np.full(6, 42.0))
        np.testing.assert_array_equal(result, np.zeros(6))

    def test_large_range(self):
        arr = np.array([0.0, 1000.0])
        result = minmax_normalize(arr)
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[1]) == pytest.approx(1.0)

    def test_float32_input(self):
        arr = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        result = minmax_normalize(arr)
        assert result.dtype == np.float64

    def test_two_elements(self):
        result = minmax_normalize(np.array([2.0, 8.0]))
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[1]) == pytest.approx(1.0)


# ─── TestZscoreNormalizeExtra ─────────────────────────────────────────────────

class TestZscoreNormalizeExtra:
    def test_large_vector_zero_mean(self):
        arr = np.arange(100, dtype=float)
        result = zscore_normalize(arr)
        assert abs(float(result.mean())) < 1e-9

    def test_large_vector_unit_std(self):
        arr = np.arange(100, dtype=float)
        result = zscore_normalize(arr)
        assert abs(float(result.std()) - 1.0) < 1e-6

    def test_negative_values(self):
        arr = np.array([-10.0, 0.0, 10.0])
        result = zscore_normalize(arr)
        assert abs(float(result.mean())) < 1e-9

    def test_float32_input(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = zscore_normalize(arr)
        assert result.dtype == np.float64

    def test_constant_returns_zeros(self):
        result = zscore_normalize(np.full(8, 5.0))
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_two_elements_symmetric(self):
        result = zscore_normalize(np.array([-1.0, 1.0]))
        assert abs(float(result.mean())) < 1e-9


# ─── TestSoftmaxExtra ─────────────────────────────────────────────────────────

class TestSoftmaxExtra:
    def test_large_vector_sums_to_one(self):
        arr = np.arange(20, dtype=float)
        result = softmax(arr)
        assert abs(float(result.sum()) - 1.0) < 1e-9

    def test_temperature_high_approaches_uniform(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = softmax(arr, temperature=1000.0)
        expected = np.ones(5) / 5.0
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_temperature_low_approaches_argmax(self):
        arr = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
        result = softmax(arr, temperature=0.001)
        assert result[2] > 0.99

    def test_all_negative_sums_to_one(self):
        arr = np.array([-5.0, -3.0, -1.0])
        result = softmax(arr)
        assert abs(float(result.sum()) - 1.0) < 1e-9

    def test_output_float64(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = softmax(arr.astype(float))
        assert result.dtype == np.float64

    def test_single_element(self):
        result = softmax(np.array([5.0]))
        assert result[0] == pytest.approx(1.0)


# ─── TestClampExtra ───────────────────────────────────────────────────────────

class TestClampExtra:
    def test_3d_array(self):
        arr = np.arange(27, dtype=float).reshape(3, 3, 3)
        result = clamp(arr, lo=5.0, hi=20.0)
        assert float(result.min()) == pytest.approx(5.0)
        assert float(result.max()) == pytest.approx(20.0)

    def test_large_range(self):
        arr = np.array([-1000.0, 0.0, 1000.0])
        result = clamp(arr, lo=-100.0, hi=100.0)
        assert float(result[0]) == pytest.approx(-100.0)
        assert float(result[-1]) == pytest.approx(100.0)

    def test_negative_range(self):
        arr = np.array([-20.0, -10.0, -5.0])
        result = clamp(arr, lo=-15.0, hi=-8.0)
        assert float(result[0]) == pytest.approx(-15.0)
        assert float(result[-1]) == pytest.approx(-8.0)

    def test_all_in_range_unchanged(self):
        arr = np.array([3.0, 5.0, 7.0])
        result = clamp(arr, lo=0.0, hi=10.0)
        np.testing.assert_allclose(result, arr)

    def test_matrix_input(self):
        arr = np.array([[0.0, 5.0], [10.0, 15.0]])
        result = clamp(arr, lo=3.0, hi=12.0)
        assert float(result[0, 0]) == pytest.approx(3.0)
        assert float(result[1, 1]) == pytest.approx(12.0)

    def test_lo_equal_hi_everything_clamped(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = clamp(arr, lo=3.0, hi=3.0)
        assert np.all(result == pytest.approx(3.0))


# ─── TestSymmetrizeMatrixExtra ────────────────────────────────────────────────

class TestSymmetrizeMatrixExtra:
    def test_5x5_symmetric(self):
        rng = np.random.default_rng(0)
        mat = rng.uniform(0, 1, (5, 5))
        result = symmetrize_matrix(mat)
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_1x1_unchanged(self):
        mat = np.array([[7.0]])
        result = symmetrize_matrix(mat)
        assert result[0, 0] == pytest.approx(7.0)

    def test_identity_unchanged(self):
        mat = np.eye(4)
        result = symmetrize_matrix(mat)
        np.testing.assert_allclose(result, mat)

    def test_dtype_float64(self):
        mat = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = symmetrize_matrix(mat.astype(float))
        assert result.dtype == np.float64

    def test_average_off_diag(self):
        mat = np.array([[0.0, 4.0], [2.0, 0.0]])
        result = symmetrize_matrix(mat)
        assert result[0, 1] == pytest.approx(3.0)
        assert result[1, 0] == pytest.approx(3.0)


# ─── TestZeroDiagonalExtra ────────────────────────────────────────────────────

class TestZeroDiagonalExtra:
    def test_5x5_diagonal_zeros(self):
        mat = np.ones((5, 5))
        result = zero_diagonal(mat)
        np.testing.assert_array_equal(np.diag(result), np.zeros(5))

    def test_10x10_diagonal_zeros(self):
        mat = np.full((10, 10), 3.0)
        result = zero_diagonal(mat)
        np.testing.assert_array_equal(np.diag(result), np.zeros(10))

    def test_off_diagonal_unchanged_5x5(self):
        mat = np.ones((5, 5)) * 2.0
        result = zero_diagonal(mat)
        mask = ~np.eye(5, dtype=bool)
        assert np.all(result[mask] == pytest.approx(2.0))

    def test_1x1_becomes_zero(self):
        mat = np.array([[5.0]])
        result = zero_diagonal(mat)
        assert result[0, 0] == pytest.approx(0.0)

    def test_original_unchanged(self):
        mat = np.eye(4)
        _ = zero_diagonal(mat)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_float32_input(self):
        mat = np.ones((3, 3), dtype=np.float32)
        result = zero_diagonal(mat)
        assert result.dtype == np.float64


# ─── TestNormalizeRowsExtra ───────────────────────────────────────────────────

class TestNormalizeRowsExtra:
    def test_l2_5x5(self):
        mat = np.random.default_rng(0).uniform(0.1, 1.0, (5, 5))
        result = normalize_rows(mat, method="l2")
        for row in result:
            assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-9

    def test_l1_5x5(self):
        mat = np.random.default_rng(0).uniform(0.1, 1.0, (5, 5))
        result = normalize_rows(mat, method="l1")
        for row in result:
            assert abs(float(np.abs(row).sum()) - 1.0) < 1e-9

    def test_minmax_3x4(self):
        mat = np.array([[1.0, 5.0, 3.0, 2.0], [10.0, 20.0, 15.0, 25.0],
                        [0.0, 1.0, 0.5, 0.8]])
        result = normalize_rows(mat, method="minmax")
        for row in result:
            assert float(row.max()) <= 1.0 + 1e-9

    def test_zero_row_l2_returns_zeros(self):
        mat = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        result = normalize_rows(mat, method="l2")
        np.testing.assert_array_equal(result[0], np.zeros(3))

    def test_shape_preserved_2x10(self):
        mat = np.random.default_rng(1).uniform(0, 1, (2, 10))
        result = normalize_rows(mat, method="l2")
        assert result.shape == (2, 10)

    def test_default_method_l2(self):
        mat = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = normalize_rows(mat)
        assert abs(float(np.linalg.norm(result[0])) - 1.0) < 1e-9


# ─── TestBatchL2NormalizeExtra ────────────────────────────────────────────────

class TestBatchL2NormalizeExtra:
    def test_ten_vectors(self):
        vecs = [np.random.default_rng(i).uniform(0.1, 1.0, 10) for i in range(10)]
        result = batch_l2_normalize(vecs)
        assert len(result) == 10
        for v in result:
            assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9

    def test_various_lengths(self):
        vecs = [np.ones(n) for n in [3, 5, 10, 20]]
        result = batch_l2_normalize(vecs)
        for v in result:
            assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9

    def test_all_float64(self):
        vecs = [np.array([1.0, 2.0, 3.0], dtype=np.float32) for _ in range(3)]
        result = batch_l2_normalize(vecs)
        assert all(v.dtype == np.float64 for v in result)

    def test_zero_vectors_return_zeros(self):
        vecs = [np.zeros(5), np.zeros(3)]
        result = batch_l2_normalize(vecs)
        np.testing.assert_array_equal(result[0], np.zeros(5))
        np.testing.assert_array_equal(result[1], np.zeros(3))

    def test_single_element_vectors(self):
        vecs = [np.array([3.0]), np.array([-4.0])]
        result = batch_l2_normalize(vecs)
        assert result[0][0] == pytest.approx(1.0)
        assert result[1][0] == pytest.approx(-1.0)

    def test_lengths_preserved(self):
        vecs = [np.ones(n) for n in [4, 6, 8]]
        result = batch_l2_normalize(vecs)
        assert [len(v) for v in result] == [4, 6, 8]
