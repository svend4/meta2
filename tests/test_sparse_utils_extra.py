"""Extra tests for puzzle_reconstruction/utils/sparse_utils.py."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mat(data=None) -> np.ndarray:
    if data is None:
        data = [[0.9, 0.2, 0.0],
                [0.3, 0.8, 0.5],
                [0.0, 0.1, 0.7]]
    return np.array(data, dtype=np.float64)


def _identity(n: int = 3) -> np.ndarray:
    return np.eye(n, dtype=np.float64)


def _zeros(n: int = 3) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


# ─── SparseEntry (extra) ──────────────────────────────────────────────────────

class TestSparseEntryExtra:
    def test_row_stored(self):
        e = SparseEntry(row=2, col=3, value=0.5)
        assert e.row == 2

    def test_col_stored(self):
        e = SparseEntry(row=0, col=5, value=1.0)
        assert e.col == 5

    def test_value_stored(self):
        e = SparseEntry(row=1, col=1, value=0.75)
        assert e.value == pytest.approx(0.75)

    def test_negative_row_raises(self):
        with pytest.raises(ValueError):
            SparseEntry(row=-1, col=0, value=0.5)

    def test_negative_col_raises(self):
        with pytest.raises(ValueError):
            SparseEntry(row=0, col=-1, value=0.5)

    def test_zero_row_col_ok(self):
        e = SparseEntry(row=0, col=0, value=0.0)
        assert e.row == 0 and e.col == 0

    def test_negative_value_ok(self):
        e = SparseEntry(row=0, col=0, value=-1.5)
        assert e.value == pytest.approx(-1.5)

    def test_large_indices_ok(self):
        e = SparseEntry(row=999, col=999, value=1.0)
        assert e.row == 999


# ─── to_sparse_entries (extra) ────────────────────────────────────────────────

class TestToSparseEntriesExtra:
    def test_returns_list(self):
        assert isinstance(to_sparse_entries(_mat()), list)

    def test_all_elements_sparse_entry(self):
        for e in to_sparse_entries(_mat()):
            assert isinstance(e, SparseEntry)

    def test_zero_matrix_returns_empty(self):
        result = to_sparse_entries(_zeros())
        assert result == []

    def test_identity_has_n_entries(self):
        result = to_sparse_entries(_identity(3), threshold=0.0)
        assert len(result) == 3

    def test_threshold_filters_small_values(self):
        m = np.array([[0.1, 0.9], [0.01, 0.8]])
        result = to_sparse_entries(m, threshold=0.5)
        assert all(abs(e.value) > 0.5 for e in result)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            to_sparse_entries(np.array([1.0, 2.0, 3.0]))

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            to_sparse_entries(np.ones((2, 2, 2)))

    def test_values_match_matrix(self):
        m = _mat()
        entries = to_sparse_entries(m, threshold=0.0)
        for e in entries:
            assert m[e.row, e.col] == pytest.approx(e.value)

    def test_row_col_in_bounds(self):
        m = _mat()
        r, c = m.shape
        for e in to_sparse_entries(m):
            assert 0 <= e.row < r
            assert 0 <= e.col < c


# ─── from_sparse_entries (extra) ──────────────────────────────────────────────

class TestFromSparseEntriesExtra:
    def test_returns_ndarray(self):
        result = from_sparse_entries([], 3, 3)
        assert isinstance(result, np.ndarray)

    def test_dtype_float64(self):
        result = from_sparse_entries([], 3, 3)
        assert result.dtype == np.float64

    def test_empty_entries_all_zeros(self):
        result = from_sparse_entries([], 2, 3)
        assert np.all(result == 0.0)

    def test_shape_correct(self):
        result = from_sparse_entries([], 4, 5)
        assert result.shape == (4, 5)

    def test_zero_rows_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], 0, 3)

    def test_zero_cols_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], 3, 0)

    def test_out_of_bounds_entry_raises(self):
        e = SparseEntry(row=5, col=0, value=1.0)
        with pytest.raises(ValueError):
            from_sparse_entries([e], n_rows=3, n_cols=3)

    def test_roundtrip_preserves_values(self):
        m = _mat()
        entries = to_sparse_entries(m, threshold=0.0)
        recovered = from_sparse_entries(entries, *m.shape)
        assert np.allclose(m, recovered)

    def test_single_entry(self):
        e = SparseEntry(row=1, col=2, value=0.75)
        result = from_sparse_entries([e], 3, 4)
        assert result[1, 2] == pytest.approx(0.75)

    def test_non_entry_positions_zero(self):
        e = SparseEntry(row=0, col=0, value=1.0)
        result = from_sparse_entries([e], 2, 2)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[1, 0] == pytest.approx(0.0)


# ─── sparse_top_k (extra) ─────────────────────────────────────────────────────

class TestSparseTopKExtra:
    def test_returns_list(self):
        assert isinstance(sparse_top_k(_mat(), 2), list)

    def test_all_sparse_entry(self):
        for e in sparse_top_k(_mat(), 2):
            assert isinstance(e, SparseEntry)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(_mat(), 0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(_mat(), -1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            sparse_top_k(np.array([1.0, 2.0]), 1)

    def test_at_most_k_per_row(self):
        m = _mat()
        k = 2
        entries = sparse_top_k(m, k)
        for r in range(m.shape[0]):
            count = sum(1 for e in entries if e.row == r)
            assert count <= k

    def test_k_larger_than_cols(self):
        m = _mat()  # 3 cols
        entries = sparse_top_k(m, k=100)
        for r in range(m.shape[0]):
            count = sum(1 for e in entries if e.row == r)
            assert count <= m.shape[1]

    def test_top_1_is_max_per_row(self):
        m = _mat()
        entries = sparse_top_k(m, 1)
        for r in range(m.shape[0]):
            row_entries = [e for e in entries if e.row == r]
            if row_entries:
                assert row_entries[0].value == pytest.approx(m[r].max())

    def test_empty_matrix_empty_result(self):
        m = np.zeros((3, 3))
        entries = sparse_top_k(m, 2)
        assert isinstance(entries, list)


# ─── threshold_matrix (extra) ─────────────────────────────────────────────────

class TestThresholdMatrixExtra:
    def test_returns_ndarray(self):
        assert isinstance(threshold_matrix(_mat(), 0.5), np.ndarray)

    def test_dtype_float64(self):
        assert threshold_matrix(_mat(), 0.5).dtype == np.float64

    def test_shape_preserved(self):
        m = _mat()
        assert threshold_matrix(m, 0.5).shape == m.shape

    def test_values_below_threshold_zeroed(self):
        m = np.array([[0.3, 0.8], [0.2, 0.9]])
        result = threshold_matrix(m, 0.5)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.8)

    def test_values_above_threshold_preserved(self):
        m = _mat()
        result = threshold_matrix(m, 0.5)
        orig_vals = m[m >= 0.5]
        res_vals = result[result >= 0.5]
        assert np.allclose(np.sort(orig_vals), np.sort(res_vals))

    def test_custom_fill_value(self):
        m = np.array([[0.2, 0.8], [0.1, 0.9]])
        result = threshold_matrix(m, 0.5, fill=-1.0)
        assert result[0, 0] == pytest.approx(-1.0)
        assert result[1, 0] == pytest.approx(-1.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            threshold_matrix(np.array([1.0, 2.0, 3.0]), 0.5)

    def test_original_not_modified(self):
        m = _mat()
        original = m.copy()
        threshold_matrix(m, 0.5)
        assert np.allclose(m, original)

    def test_zero_threshold_preserves_positives(self):
        m = np.array([[0.5, 0.0], [0.0, 0.7]])
        result = threshold_matrix(m, 0.0)
        assert result[0, 0] == pytest.approx(0.5)
        assert result[1, 1] == pytest.approx(0.7)


# ─── symmetrize_matrix (extra) ────────────────────────────────────────────────

class TestSymmetrizeMatrixExtra:
    def test_returns_ndarray(self):
        assert isinstance(symmetrize_matrix(_identity()), np.ndarray)

    def test_dtype_float64(self):
        assert symmetrize_matrix(_identity()).dtype == np.float64

    def test_shape_preserved(self):
        m = _mat()
        assert symmetrize_matrix(m).shape == m.shape

    def test_result_is_symmetric(self):
        m = _mat()
        result = symmetrize_matrix(m)
        assert np.allclose(result, result.T)

    def test_uses_max(self):
        m = np.array([[0.0, 0.3], [0.8, 0.0]], dtype=np.float64)
        result = symmetrize_matrix(m)
        assert result[0, 1] == pytest.approx(0.8)
        assert result[1, 0] == pytest.approx(0.8)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.ones((2, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.array([1.0, 2.0]))

    def test_symmetric_input_unchanged(self):
        m = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = symmetrize_matrix(m)
        assert np.allclose(result, m)

    def test_diagonal_preserved(self):
        m = _mat()
        result = symmetrize_matrix(m)
        assert np.allclose(np.diag(result), np.diag(m))


# ─── normalize_matrix (extra) ─────────────────────────────────────────────────

class TestNormalizeMatrixExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_matrix(_mat()), np.ndarray)

    def test_dtype_float64(self):
        assert normalize_matrix(_mat()).dtype == np.float64

    def test_shape_preserved(self):
        m = _mat()
        assert normalize_matrix(m).shape == m.shape

    def test_row_max_is_one(self):
        m = np.array([[0.5, 0.9], [0.3, 0.6]])
        result = normalize_matrix(m, axis=1)
        assert result[0].max() == pytest.approx(1.0)
        assert result[1].max() == pytest.approx(1.0)

    def test_col_max_is_one(self):
        m = np.array([[0.5, 0.9], [0.3, 0.6]])
        result = normalize_matrix(m, axis=0)
        assert result[:, 0].max() == pytest.approx(1.0)
        assert result[:, 1].max() == pytest.approx(1.0)

    def test_zero_row_stays_zero(self):
        m = np.array([[0.0, 0.0], [0.5, 0.8]])
        result = normalize_matrix(m, axis=1)
        assert np.all(result[0] == pytest.approx(0.0))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.array([1.0, 2.0]))

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(_mat(), axis=2)

    def test_original_not_modified(self):
        m = _mat()
        original = m.copy()
        normalize_matrix(m)
        assert np.allclose(m, original)

    def test_values_in_0_1(self):
        result = normalize_matrix(_mat())
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-12


# ─── diagonal_zeros (extra) ───────────────────────────────────────────────────

class TestDiagonalZerosExtra:
    def test_returns_ndarray(self):
        assert isinstance(diagonal_zeros(_identity()), np.ndarray)

    def test_dtype_float64(self):
        assert diagonal_zeros(_identity()).dtype == np.float64

    def test_diagonal_is_zero(self):
        result = diagonal_zeros(_identity(4))
        assert np.all(np.diag(result) == 0.0)

    def test_off_diagonal_preserved(self):
        m = np.array([[5.0, 2.0], [3.0, 4.0]])
        result = diagonal_zeros(m)
        assert result[0, 1] == pytest.approx(2.0)
        assert result[1, 0] == pytest.approx(3.0)

    def test_shape_preserved(self):
        m = _mat()
        assert diagonal_zeros(m).shape == m.shape

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.ones((2, 3)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.array([1.0, 2.0]))

    def test_original_not_modified(self):
        m = _identity()
        original = m.copy()
        diagonal_zeros(m)
        assert np.allclose(m, original)

    def test_zero_matrix_stays_zero(self):
        result = diagonal_zeros(_zeros())
        assert np.all(result == 0.0)


# ─── matrix_sparsity (extra) ──────────────────────────────────────────────────

class TestMatrixSparsityExtra:
    def test_returns_float(self):
        assert isinstance(matrix_sparsity(_mat()), float)

    def test_all_zeros_sparsity_one(self):
        assert matrix_sparsity(_zeros()) == pytest.approx(1.0)

    def test_all_nonzero_sparsity_zero(self):
        m = np.ones((3, 3), dtype=np.float64)
        assert matrix_sparsity(m) == pytest.approx(0.0)

    def test_empty_matrix_sparsity_one(self):
        m = np.zeros((0, 0), dtype=np.float64)
        assert matrix_sparsity(m) == pytest.approx(1.0)

    def test_identity_sparsity(self):
        # 3x3 identity: 3 nonzero, 6 zero => 6/9
        result = matrix_sparsity(_identity(3))
        assert result == pytest.approx(6.0 / 9.0)

    def test_sparsity_in_0_1(self):
        result = matrix_sparsity(_mat())
        assert 0.0 <= result <= 1.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            matrix_sparsity(np.array([1.0, 2.0]))

    def test_half_zeros(self):
        m = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert matrix_sparsity(m) == pytest.approx(0.5)


# ─── top_k_per_row (extra) ────────────────────────────────────────────────────

class TestTopKPerRowExtra:
    def test_returns_ndarray(self):
        assert isinstance(top_k_per_row(_mat(), 2), np.ndarray)

    def test_dtype_float64(self):
        assert top_k_per_row(_mat(), 2).dtype == np.float64

    def test_shape_preserved(self):
        m = _mat()
        assert top_k_per_row(m, 2).shape == m.shape

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(_mat(), 0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(_mat(), -1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(np.array([1.0, 2.0, 3.0]), 1)

    def test_at_most_k_nonzero_per_row(self):
        m = _mat()
        result = top_k_per_row(m, k=2)
        for r in range(m.shape[0]):
            n_nonzero = np.count_nonzero(result[r])
            assert n_nonzero <= 2

    def test_zeros_outside_topk(self):
        m = np.array([[0.9, 0.2, 0.5], [0.1, 0.8, 0.3]])
        result = top_k_per_row(m, k=1)
        # Row 0: only 0.9 kept; row 1: only 0.8 kept
        assert result[0, 0] == pytest.approx(0.9)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[1, 1] == pytest.approx(0.8)
        assert result[1, 0] == pytest.approx(0.0)

    def test_k_larger_than_cols_keeps_all(self):
        m = _mat()  # 3 cols
        result = top_k_per_row(m, k=100)
        assert np.allclose(result, m)

    def test_original_not_modified(self):
        m = _mat()
        original = m.copy()
        top_k_per_row(m, 2)
        assert np.allclose(m, original)

    def test_zero_matrix_result_zero(self):
        m = _zeros(3)
        result = top_k_per_row(m, 2)
        assert np.all(result == 0.0)
