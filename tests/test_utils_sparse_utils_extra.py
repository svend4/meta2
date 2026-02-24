"""Extra tests for puzzle_reconstruction/utils/sparse_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.sparse_utils import (
    SparseEntry,
    to_sparse_entries,
    from_sparse_entries,
    sparse_top_k,
    threshold_matrix,
    symmetrize_matrix,
    normalize_matrix,
    diagonal_zeros,
    matrix_sparsity,
    top_k_per_row,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mat(n=4, val=1.0) -> np.ndarray:
    return np.full((n, n), val, dtype=np.float64)


def _eye(n=4) -> np.ndarray:
    return np.eye(n, dtype=np.float64)


def _sparse_mat() -> np.ndarray:
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 1] = 1.0
    M[1, 2] = 2.0
    M[3, 0] = 3.0
    return M


# ─── SparseEntry ──────────────────────────────────────────────────────────────

class TestSparseEntryExtra:
    def test_stores_row(self):
        e = SparseEntry(row=2, col=3, value=0.5)
        assert e.row == 2

    def test_stores_col(self):
        e = SparseEntry(row=0, col=1, value=1.0)
        assert e.col == 1

    def test_stores_value(self):
        e = SparseEntry(row=0, col=0, value=-3.14)
        assert e.value == pytest.approx(-3.14)

    def test_negative_row_raises(self):
        with pytest.raises(ValueError):
            SparseEntry(row=-1, col=0, value=1.0)

    def test_negative_col_raises(self):
        with pytest.raises(ValueError):
            SparseEntry(row=0, col=-1, value=1.0)

    def test_zero_row_col_ok(self):
        e = SparseEntry(row=0, col=0, value=0.0)
        assert e.row == 0 and e.col == 0


# ─── to_sparse_entries ────────────────────────────────────────────────────────

class TestToSparseEntriesExtra:
    def test_returns_list(self):
        assert isinstance(to_sparse_entries(_sparse_mat()), list)

    def test_all_elements_sparse_entry(self):
        for e in to_sparse_entries(_sparse_mat()):
            assert isinstance(e, SparseEntry)

    def test_zero_matrix_empty(self):
        M = np.zeros((4, 4))
        assert len(to_sparse_entries(M)) == 0

    def test_threshold_filters(self):
        M = np.array([[0.5, 0.0], [0.0, 1.5]])
        entries = to_sparse_entries(M, threshold=1.0)
        assert len(entries) == 1
        assert entries[0].value == pytest.approx(1.5)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            to_sparse_entries(np.zeros((2, 2, 2)))

    def test_count_correct(self):
        M = _sparse_mat()
        entries = to_sparse_entries(M, threshold=0.0)
        assert len(entries) == 3

    def test_row_col_order(self):
        M = _sparse_mat()
        entries = to_sparse_entries(M, threshold=0.0)
        rows = [e.row for e in entries]
        assert rows == sorted(rows)


# ─── from_sparse_entries ──────────────────────────────────────────────────────

class TestFromSparseEntriesExtra:
    def test_returns_ndarray(self):
        entries = [SparseEntry(0, 0, 1.0)]
        assert isinstance(from_sparse_entries(entries, 2, 2), np.ndarray)

    def test_dtype_float64(self):
        entries = [SparseEntry(0, 0, 1.0)]
        out = from_sparse_entries(entries, 2, 2)
        assert out.dtype == np.float64

    def test_shape_correct(self):
        out = from_sparse_entries([], 3, 5)
        assert out.shape == (3, 5)

    def test_empty_entries_all_zeros(self):
        out = from_sparse_entries([], 4, 4)
        assert np.all(out == 0.0)

    def test_values_placed_correctly(self):
        entries = [SparseEntry(1, 2, 7.0)]
        out = from_sparse_entries(entries, 3, 4)
        assert out[1, 2] == pytest.approx(7.0)

    def test_invalid_n_rows_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], 0, 3)

    def test_invalid_n_cols_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], 3, 0)

    def test_out_of_bounds_raises(self):
        entries = [SparseEntry(5, 5, 1.0)]
        with pytest.raises(ValueError):
            from_sparse_entries(entries, 3, 3)

    def test_roundtrip(self):
        M = _sparse_mat()
        entries = to_sparse_entries(M, threshold=0.0)
        out = from_sparse_entries(entries, 4, 4)
        np.testing.assert_allclose(out, M)


# ─── sparse_top_k ─────────────────────────────────────────────────────────────

class TestSparseTopKExtra:
    def test_returns_list(self):
        assert isinstance(sparse_top_k(_sparse_mat(), 2), list)

    def test_k_lt_1_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(_sparse_mat(), 0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(np.zeros((2, 2, 2)), 1)

    def test_k_1_per_row(self):
        M = np.array([[3.0, 1.0], [2.0, 4.0]])
        entries = sparse_top_k(M, 1)
        rows = [e.row for e in entries]
        assert sorted(rows) == [0, 1]

    def test_values_descending_per_row(self):
        M = np.array([[1.0, 5.0, 3.0]])
        entries = sparse_top_k(M, 2)
        row0 = sorted([e.value for e in entries if e.row == 0], reverse=True)
        assert row0[0] >= row0[-1]

    def test_all_sparse_entry_type(self):
        for e in sparse_top_k(_sparse_mat(), 2):
            assert isinstance(e, SparseEntry)


# ─── threshold_matrix ─────────────────────────────────────────────────────────

class TestThresholdMatrixExtra:
    def test_returns_ndarray(self):
        assert isinstance(threshold_matrix(_mat(), 0.5), np.ndarray)

    def test_dtype_float64(self):
        assert threshold_matrix(_mat(), 0.5).dtype == np.float64

    def test_shape_preserved(self):
        M = np.zeros((3, 5), dtype=np.float64)
        assert threshold_matrix(M, 0.0).shape == (3, 5)

    def test_fills_below_threshold(self):
        M = np.array([[0.5, 1.5], [0.2, 2.0]])
        out = threshold_matrix(M, 1.0, fill=0.0)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(1.5)
        assert out[1, 0] == pytest.approx(0.0)

    def test_custom_fill(self):
        M = np.ones((3, 3))
        out = threshold_matrix(M, 2.0, fill=-1.0)
        assert np.all(out == -1.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.zeros((2, 2, 2)), 0.5)

    def test_original_not_modified(self):
        M = np.ones((3, 3))
        original = M.copy()
        threshold_matrix(M, 0.5)
        np.testing.assert_array_equal(M, original)


# ─── symmetrize_matrix ────────────────────────────────────────────────────────

class TestSymmetrizeMatrixExtra:
    def test_returns_ndarray(self):
        assert isinstance(symmetrize_matrix(_mat(3)), np.ndarray)

    def test_dtype_float64(self):
        assert symmetrize_matrix(_mat(3)).dtype == np.float64

    def test_shape_preserved(self):
        M = np.eye(5)
        assert symmetrize_matrix(M).shape == (5, 5)

    def test_result_symmetric(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = symmetrize_matrix(M)
        np.testing.assert_allclose(out, out.T)

    def test_max_rule(self):
        M = np.array([[0.0, 1.0], [5.0, 0.0]])
        out = symmetrize_matrix(M)
        assert out[0, 1] == pytest.approx(5.0)
        assert out[1, 0] == pytest.approx(5.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.zeros((3, 4)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.zeros((2, 2, 2)))


# ─── normalize_matrix ─────────────────────────────────────────────────────────

class TestNormalizeMatrixExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_matrix(_mat(3)), np.ndarray)

    def test_dtype_float64(self):
        assert normalize_matrix(_mat(3)).dtype == np.float64

    def test_row_max_one(self):
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = normalize_matrix(M, axis=1)
        np.testing.assert_allclose(out.max(axis=1), 1.0)

    def test_col_max_one(self):
        M = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        out = normalize_matrix(M, axis=0)
        np.testing.assert_allclose(out.max(axis=0), 1.0)

    def test_zero_row_stays_zero(self):
        M = np.array([[0.0, 0.0], [1.0, 2.0]])
        out = normalize_matrix(M, axis=1)
        assert np.all(out[0] == 0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.zeros((2, 2, 2)))

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(_mat(3), axis=2)


# ─── diagonal_zeros ───────────────────────────────────────────────────────────

class TestDiagonalZerosExtra:
    def test_returns_ndarray(self):
        assert isinstance(diagonal_zeros(_eye()), np.ndarray)

    def test_dtype_float64(self):
        assert diagonal_zeros(_eye()).dtype == np.float64

    def test_diagonal_is_zero(self):
        M = _mat(4, 1.0)
        out = diagonal_zeros(M)
        assert np.all(np.diag(out) == 0.0)

    def test_off_diagonal_unchanged(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = diagonal_zeros(M)
        assert out[0, 1] == pytest.approx(2.0)
        assert out[1, 0] == pytest.approx(3.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.zeros((3, 4)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.zeros((2, 2, 2)))

    def test_original_not_modified(self):
        M = _mat(3, 5.0)
        original = M.copy()
        diagonal_zeros(M)
        np.testing.assert_array_equal(M, original)


# ─── matrix_sparsity ──────────────────────────────────────────────────────────

class TestMatrixSparsityExtra:
    def test_returns_float(self):
        assert isinstance(matrix_sparsity(_mat()), float)

    def test_all_zeros_is_one(self):
        M = np.zeros((4, 4))
        assert matrix_sparsity(M) == pytest.approx(1.0)

    def test_all_nonzero_is_zero(self):
        M = _mat(4, 1.0)
        assert matrix_sparsity(M) == pytest.approx(0.0)

    def test_half_zeros(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert matrix_sparsity(M) == pytest.approx(0.5)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            matrix_sparsity(np.zeros((2, 2, 2)))

    def test_result_in_range(self):
        M = _sparse_mat()
        s = matrix_sparsity(M)
        assert 0.0 <= s <= 1.0


# ─── top_k_per_row ────────────────────────────────────────────────────────────

class TestTopKPerRowExtra:
    def test_returns_ndarray(self):
        assert isinstance(top_k_per_row(_mat(3), 2), np.ndarray)

    def test_dtype_float64(self):
        assert top_k_per_row(_mat(3), 2).dtype == np.float64

    def test_shape_preserved(self):
        M = np.zeros((4, 5))
        assert top_k_per_row(M, 2).shape == (4, 5)

    def test_k_lt_1_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(_mat(3), 0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(np.zeros((2, 2, 2)), 1)

    def test_row_has_at_most_k_nonzero(self):
        M = np.array([[3.0, 1.0, 2.0, 4.0]])
        out = top_k_per_row(M, 2)
        assert np.count_nonzero(out[0]) == 2

    def test_top_values_kept(self):
        M = np.array([[1.0, 5.0, 3.0]])
        out = top_k_per_row(M, 1)
        assert out[0, 1] == pytest.approx(5.0)

    def test_original_not_modified(self):
        M = _mat(3, 1.0)
        original = M.copy()
        top_k_per_row(M, 2)
        np.testing.assert_array_equal(M, original)
