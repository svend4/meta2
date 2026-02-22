"""Tests for puzzle_reconstruction.utils.sparse_utils."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.sparse_utils import (
    SparseEntry,
    diagonal_zeros,
    from_sparse_entries,
    matrix_sparsity,
    normalize_matrix,
    sparse_top_k,
    symmetrize_matrix,
    threshold_matrix,
    to_sparse_entries,
    top_k_per_row,
)


# ─── SparseEntry ─────────────────────────────────────────────────────────────

class TestSparseEntry:
    def test_fields_stored(self):
        e = SparseEntry(row=2, col=3, value=0.7)
        assert e.row == 2
        assert e.col == 3
        assert e.value == pytest.approx(0.7)

    def test_negative_row_raises(self):
        with pytest.raises(ValueError):
            SparseEntry(row=-1, col=0, value=1.0)

    def test_negative_col_raises(self):
        with pytest.raises(ValueError):
            SparseEntry(row=0, col=-1, value=1.0)

    def test_zero_row_allowed(self):
        e = SparseEntry(row=0, col=0, value=0.5)
        assert e.row == 0

    def test_negative_value_allowed(self):
        e = SparseEntry(row=0, col=0, value=-0.5)
        assert e.value == pytest.approx(-0.5)

    def test_zero_value_allowed(self):
        e = SparseEntry(row=1, col=2, value=0.0)
        assert e.value == pytest.approx(0.0)


# ─── to_sparse_entries ───────────────────────────────────────────────────────

class TestToSparseEntries:
    def test_returns_list(self):
        mat = np.eye(3)
        assert isinstance(to_sparse_entries(mat), list)

    def test_count_matches_nonzero(self):
        mat = np.eye(3)
        entries = to_sparse_entries(mat)
        assert len(entries) == 3

    def test_all_sparse_entries(self):
        mat = np.eye(3)
        entries = to_sparse_entries(mat)
        assert all(isinstance(e, SparseEntry) for e in entries)

    def test_zero_matrix_returns_empty(self):
        mat = np.zeros((4, 4))
        assert to_sparse_entries(mat) == []

    def test_threshold_filters(self):
        mat = np.array([[0.1, 0.5], [0.3, 0.8]])
        entries = to_sparse_entries(mat, threshold=0.4)
        assert len(entries) == 2
        values = [e.value for e in entries]
        assert all(v > 0.4 for v in values)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            to_sparse_entries(np.zeros((3,)))

    def test_correct_coordinates(self):
        mat = np.zeros((3, 3))
        mat[1, 2] = 0.9
        entries = to_sparse_entries(mat)
        assert len(entries) == 1
        assert entries[0].row == 1
        assert entries[0].col == 2
        assert entries[0].value == pytest.approx(0.9)

    def test_negative_values_included(self):
        mat = np.array([[-0.5, 0.0], [0.0, 0.3]])
        entries = to_sparse_entries(mat, threshold=0.0)
        vals = [e.value for e in entries]
        assert -0.5 in vals


# ─── from_sparse_entries ─────────────────────────────────────────────────────

class TestFromSparseEntries:
    def test_returns_array(self):
        entries = [SparseEntry(0, 0, 1.0)]
        mat = from_sparse_entries(entries, 3, 3)
        assert isinstance(mat, np.ndarray)

    def test_dtype_float64(self):
        mat = from_sparse_entries([], 3, 3)
        assert mat.dtype == np.float64

    def test_shape_correct(self):
        mat = from_sparse_entries([], 4, 5)
        assert mat.shape == (4, 5)

    def test_empty_entries_all_zero(self):
        mat = from_sparse_entries([], 3, 3)
        assert np.all(mat == 0.0)

    def test_values_placed_correctly(self):
        entries = [SparseEntry(1, 2, 0.7), SparseEntry(0, 0, 0.3)]
        mat = from_sparse_entries(entries, 3, 3)
        assert mat[1, 2] == pytest.approx(0.7)
        assert mat[0, 0] == pytest.approx(0.3)

    def test_roundtrip(self):
        original = np.array([[0.1, 0.0, 0.5], [0.0, 0.9, 0.0]])
        entries = to_sparse_entries(original, threshold=0.0)
        recovered = from_sparse_entries(entries, 2, 3)
        np.testing.assert_array_almost_equal(recovered, original)

    def test_zero_n_rows_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], 0, 3)

    def test_zero_n_cols_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], 3, 0)

    def test_out_of_bounds_raises(self):
        entries = [SparseEntry(5, 0, 1.0)]
        with pytest.raises(ValueError):
            from_sparse_entries(entries, 3, 3)


# ─── sparse_top_k ────────────────────────────────────────────────────────────

class TestSparseTopK:
    def test_returns_list(self):
        mat = np.array([[0.9, 0.1, 0.5], [0.3, 0.8, 0.2]])
        result = sparse_top_k(mat, k=1)
        assert isinstance(result, list)

    def test_at_most_k_per_row(self):
        mat = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = sparse_top_k(mat, k=2)
        from collections import Counter
        row_counts = Counter(e.row for e in result)
        assert all(v <= 2 for v in row_counts.values())

    def test_top_values_selected(self):
        mat = np.array([[0.1, 0.9, 0.5]])
        result = sparse_top_k(mat, k=1)
        assert result[0].value == pytest.approx(0.9)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(np.zeros((3, 3)), k=0)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(np.zeros((3,)), k=1)

    def test_k_larger_than_cols(self):
        mat = np.array([[0.1, 0.9]])
        result = sparse_top_k(mat, k=10)
        assert len(result) == 2  # only 2 columns


# ─── threshold_matrix ────────────────────────────────────────────────────────

class TestThresholdMatrix:
    def test_returns_array(self):
        mat = np.array([[0.1, 0.5], [0.3, 0.9]])
        result = threshold_matrix(mat, threshold=0.4)
        assert isinstance(result, np.ndarray)

    def test_dtype_float64(self):
        mat = np.array([[0.1, 0.5], [0.3, 0.9]])
        result = threshold_matrix(mat, threshold=0.4)
        assert result.dtype == np.float64

    def test_below_threshold_zeroed(self):
        mat = np.array([[0.1, 0.5], [0.3, 0.9]])
        result = threshold_matrix(mat, threshold=0.4)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.5)

    def test_fill_value_used(self):
        mat = np.array([[0.1, 0.9]])
        result = threshold_matrix(mat, threshold=0.5, fill=-1.0)
        assert result[0, 0] == pytest.approx(-1.0)

    def test_all_zeroed_below_max(self):
        mat = np.ones((3, 3))
        result = threshold_matrix(mat, threshold=2.0)
        assert np.all(result == 0.0)

    def test_all_kept_above_min(self):
        mat = np.ones((3, 3)) * 5.0
        result = threshold_matrix(mat, threshold=0.0)
        np.testing.assert_array_equal(result, mat)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.zeros((3,)), threshold=0.5)

    def test_original_not_modified(self):
        mat = np.array([[0.1, 0.9]])
        original = mat.copy()
        threshold_matrix(mat, threshold=0.5)
        np.testing.assert_array_equal(mat, original)


# ─── symmetrize_matrix ───────────────────────────────────────────────────────

class TestSymmetrizeMatrix:
    def test_returns_symmetric(self):
        mat = np.array([[0.0, 0.3], [0.9, 0.0]])
        result = symmetrize_matrix(mat)
        np.testing.assert_array_equal(result, result.T)

    def test_takes_max(self):
        mat = np.array([[0.0, 0.3], [0.9, 0.0]])
        result = symmetrize_matrix(mat)
        assert result[0, 1] == pytest.approx(0.9)
        assert result[1, 0] == pytest.approx(0.9)

    def test_already_symmetric_unchanged(self):
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = symmetrize_matrix(mat)
        np.testing.assert_array_equal(result, mat)

    def test_not_square_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.zeros((2, 3)))

    def test_dtype_float64(self):
        mat = np.eye(3)
        result = symmetrize_matrix(mat)
        assert result.dtype == np.float64


# ─── normalize_matrix ────────────────────────────────────────────────────────

class TestNormalizeMatrix:
    def test_row_max_is_one(self):
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = normalize_matrix(mat, axis=1)
        np.testing.assert_array_almost_equal(result[:, 2], [1.0, 1.0])

    def test_col_max_is_one_with_axis_zero(self):
        mat = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 1.0]])
        result = normalize_matrix(mat, axis=0)
        np.testing.assert_array_almost_equal(result[2, 0], 1.0)
        np.testing.assert_array_almost_equal(result[0, 1], 1.0)

    def test_zero_row_stays_zero(self):
        mat = np.array([[0.0, 0.0], [1.0, 2.0]])
        result = normalize_matrix(mat, axis=1)
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.zeros((3,)), axis=1)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.zeros((3, 3)), axis=2)

    def test_values_in_unit_interval(self):
        mat = np.random.default_rng(0).random((4, 4))
        result = normalize_matrix(mat)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-9


# ─── diagonal_zeros ──────────────────────────────────────────────────────────

class TestDiagonalZeros:
    def test_diagonal_is_zero(self):
        mat = np.ones((4, 4))
        result = diagonal_zeros(mat)
        np.testing.assert_array_equal(np.diag(result), np.zeros(4))

    def test_off_diagonal_unchanged(self):
        mat = np.array([[1.0, 0.5], [0.3, 1.0]])
        result = diagonal_zeros(mat)
        assert result[0, 1] == pytest.approx(0.5)
        assert result[1, 0] == pytest.approx(0.3)

    def test_not_square_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.ones((2, 3)))

    def test_original_not_modified(self):
        mat = np.ones((3, 3))
        original = mat.copy()
        diagonal_zeros(mat)
        np.testing.assert_array_equal(mat, original)

    def test_dtype_float64(self):
        mat = np.ones((3, 3))
        result = diagonal_zeros(mat)
        assert result.dtype == np.float64


# ─── matrix_sparsity ─────────────────────────────────────────────────────────

class TestMatrixSparsity:
    def test_all_zeros_returns_one(self):
        mat = np.zeros((4, 4))
        assert matrix_sparsity(mat) == pytest.approx(1.0)

    def test_no_zeros_returns_zero(self):
        mat = np.ones((4, 4))
        assert matrix_sparsity(mat) == pytest.approx(0.0)

    def test_half_zeros(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert matrix_sparsity(mat) == pytest.approx(0.5)

    def test_empty_matrix_returns_one(self):
        mat = np.zeros((0, 0))
        assert matrix_sparsity(mat) == pytest.approx(1.0)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            matrix_sparsity(np.zeros((3,)))

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(0)
        mat = rng.random((5, 5))
        mat[mat < 0.5] = 0.0
        s = matrix_sparsity(mat)
        assert 0.0 <= s <= 1.0


# ─── top_k_per_row ───────────────────────────────────────────────────────────

class TestTopKPerRow:
    def test_returns_array(self):
        mat = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
        result = top_k_per_row(mat, k=1)
        assert isinstance(result, np.ndarray)

    def test_dtype_float64(self):
        mat = np.array([[0.1, 0.9]])
        result = top_k_per_row(mat, k=1)
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        mat = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
        result = top_k_per_row(mat, k=1)
        assert result.shape == mat.shape

    def test_k1_keeps_max_per_row(self):
        mat = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
        result = top_k_per_row(mat, k=1)
        assert result[0, 1] == pytest.approx(0.9)
        assert result[1, 2] == pytest.approx(0.8)

    def test_non_top_zeroed(self):
        mat = np.array([[0.1, 0.9, 0.5]])
        result = top_k_per_row(mat, k=1)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(0.0)

    def test_k_ge_cols_keeps_all(self):
        mat = np.array([[0.1, 0.9, 0.5]])
        result = top_k_per_row(mat, k=10)
        np.testing.assert_array_equal(result, mat)

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(np.zeros((3, 3)), k=0)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(np.zeros((3,)), k=1)

    def test_sum_preserved_when_k_ge_cols(self):
        mat = np.array([[0.3, 0.7], [0.5, 0.5]])
        result = top_k_per_row(mat, k=2)
        np.testing.assert_array_almost_equal(result, mat)
