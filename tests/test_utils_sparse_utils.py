"""Тесты для puzzle_reconstruction/utils/sparse_utils.py."""
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


# ─── SparseEntry ──────────────────────────────────────────────────────────────

class TestSparseEntry:
    def test_creation_basic(self):
        e = SparseEntry(row=0, col=0, value=1.0)
        assert e.row == 0
        assert e.col == 0
        assert e.value == 1.0

    def test_negative_row_raises(self):
        with pytest.raises(ValueError, match="row"):
            SparseEntry(row=-1, col=0, value=1.0)

    def test_negative_col_raises(self):
        with pytest.raises(ValueError, match="col"):
            SparseEntry(row=0, col=-1, value=1.0)

    def test_zero_indices_allowed(self):
        e = SparseEntry(row=0, col=0, value=0.0)
        assert e.row == 0
        assert e.col == 0

    def test_large_indices_allowed(self):
        e = SparseEntry(row=100, col=200, value=99.9)
        assert e.row == 100
        assert e.col == 200

    def test_negative_value_allowed(self):
        e = SparseEntry(row=0, col=0, value=-5.0)
        assert e.value == -5.0


# ─── to_sparse_entries ────────────────────────────────────────────────────────

class TestToSparseEntries:
    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            to_sparse_entries(np.array([1.0, 2.0]))

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            to_sparse_entries(np.ones((2, 2, 2)))

    def test_empty_matrix_returns_empty(self):
        mat = np.zeros((3, 3))
        entries = to_sparse_entries(mat)
        assert entries == []

    def test_strict_threshold_excludes_equal(self):
        mat = np.array([[0.5, 0.0], [0.0, 0.3]])
        # threshold=0.5: abs(v) > 0.5, so 0.5 is NOT included (strict)
        entries = to_sparse_entries(mat, threshold=0.5)
        assert all(abs(e.value) > 0.5 for e in entries)
        assert len(entries) == 0

    def test_strict_threshold_includes_above(self):
        mat = np.array([[0.6, 0.0], [0.0, 0.3]])
        entries = to_sparse_entries(mat, threshold=0.5)
        assert len(entries) == 1
        assert entries[0].value == pytest.approx(0.6)

    def test_default_threshold_0_excludes_zeros(self):
        mat = np.array([[1.0, 0.0], [0.0, 2.0]])
        entries = to_sparse_entries(mat)
        assert len(entries) == 2
        values = {e.value for e in entries}
        assert 1.0 in values
        assert 2.0 in values

    def test_returns_sparse_entry_objects(self):
        mat = np.array([[1.0, 2.0]])
        entries = to_sparse_entries(mat)
        for e in entries:
            assert isinstance(e, SparseEntry)

    def test_row_col_correct(self):
        mat = np.zeros((3, 4))
        mat[1, 2] = 5.0
        entries = to_sparse_entries(mat)
        assert len(entries) == 1
        assert entries[0].row == 1
        assert entries[0].col == 2

    def test_negative_values_included(self):
        mat = np.array([[-1.0, 0.0], [0.0, -2.0]])
        entries = to_sparse_entries(mat)
        assert len(entries) == 2

    def test_order_row_then_col(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        entries = to_sparse_entries(mat)
        rows = [e.row for e in entries]
        assert rows == sorted(rows)  # rows non-decreasing


# ─── from_sparse_entries ──────────────────────────────────────────────────────

class TestFromSparseEntries:
    def test_n_rows_zero_raises(self):
        with pytest.raises(ValueError, match="n_rows"):
            from_sparse_entries([], 0, 3)

    def test_n_cols_zero_raises(self):
        with pytest.raises(ValueError, match="n_cols"):
            from_sparse_entries([], 3, 0)

    def test_negative_n_rows_raises(self):
        with pytest.raises(ValueError):
            from_sparse_entries([], -1, 3)

    def test_out_of_bounds_row_raises(self):
        entries = [SparseEntry(row=5, col=0, value=1.0)]
        with pytest.raises(ValueError):
            from_sparse_entries(entries, 3, 3)

    def test_out_of_bounds_col_raises(self):
        entries = [SparseEntry(row=0, col=5, value=1.0)]
        with pytest.raises(ValueError):
            from_sparse_entries(entries, 3, 3)

    def test_empty_entries_returns_zeros(self):
        mat = from_sparse_entries([], 3, 4)
        assert mat.shape == (3, 4)
        np.testing.assert_array_equal(mat, 0.0)

    def test_values_placed_correctly(self):
        entries = [SparseEntry(row=1, col=2, value=7.5)]
        mat = from_sparse_entries(entries, 3, 4)
        assert mat[1, 2] == pytest.approx(7.5)
        assert mat[0, 0] == 0.0

    def test_returns_float64(self):
        mat = from_sparse_entries([], 2, 2)
        assert mat.dtype == np.float64

    def test_roundtrip(self):
        original = np.array([[0.0, 1.5, 0.0], [2.5, 0.0, 3.5]])
        entries = to_sparse_entries(original)
        restored = from_sparse_entries(entries, 2, 3)
        np.testing.assert_allclose(restored, original)


# ─── sparse_top_k ─────────────────────────────────────────────────────────────

class TestSparseTopK:
    def test_k_zero_raises(self):
        mat = np.ones((3, 3))
        with pytest.raises(ValueError, match="k"):
            sparse_top_k(mat, 0)

    def test_k_negative_raises(self):
        mat = np.ones((3, 3))
        with pytest.raises(ValueError):
            sparse_top_k(mat, -1)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            sparse_top_k(np.array([1.0, 2.0, 3.0]), 1)

    def test_returns_sparse_entries(self):
        mat = np.array([[3.0, 1.0, 2.0]])
        entries = sparse_top_k(mat, 2)
        for e in entries:
            assert isinstance(e, SparseEntry)

    def test_top_k_per_row(self):
        mat = np.array([
            [5.0, 3.0, 1.0],
            [2.0, 4.0, 6.0],
        ])
        entries = sparse_top_k(mat, 2)
        row0_entries = [e for e in entries if e.row == 0]
        row1_entries = [e for e in entries if e.row == 1]
        assert len(row0_entries) == 2
        assert len(row1_entries) == 2
        # Row 0 top-2: indices 0 (5.0) and 1 (3.0)
        assert any(e.col == 0 for e in row0_entries)
        # Row 1 top-2: indices 2 (6.0) and 1 (4.0)
        assert any(e.col == 2 for e in row1_entries)

    def test_k_larger_than_cols_clipped(self):
        mat = np.array([[1.0, 2.0, 3.0]])
        entries = sparse_top_k(mat, k=10)
        assert len(entries) == 3  # clipped to n_cols

    def test_descending_within_row(self):
        mat = np.array([[1.0, 4.0, 2.0, 3.0]])
        entries = sparse_top_k(mat, 2)
        assert len(entries) == 2
        values = [e.value for e in entries]
        assert values == sorted(values, reverse=True)


# ─── threshold_matrix ─────────────────────────────────────────────────────────

class TestThresholdMatrix:
    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            threshold_matrix(np.array([1.0, 2.0]), 0.5)

    def test_strict_less_than_threshold(self):
        mat = np.array([[0.5, 1.0], [0.3, 0.8]])
        result = threshold_matrix(mat, 0.5)
        # Values < 0.5 are replaced; 0.5 is kept (not strictly less)
        assert result[0, 0] == pytest.approx(0.5)  # 0.5 >= 0.5: kept
        assert result[1, 0] == pytest.approx(0.0)  # 0.3 < 0.5: replaced

    def test_fill_value(self):
        mat = np.array([[0.1, 1.0]])
        result = threshold_matrix(mat, 0.5, fill=99.0)
        assert result[0, 0] == pytest.approx(99.0)
        assert result[0, 1] == pytest.approx(1.0)

    def test_returns_float64(self):
        mat = np.array([[1.0, 2.0]])
        result = threshold_matrix(mat, 0.5)
        assert result.dtype == np.float64

    def test_returns_copy(self):
        mat = np.array([[1.0, 2.0]])
        result = threshold_matrix(mat, 0.5)
        mat[0, 0] = 99.0
        assert result[0, 0] != 99.0

    def test_all_above_threshold_unchanged(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = threshold_matrix(mat, 0.5)
        np.testing.assert_allclose(result, mat.astype(np.float64))


# ─── symmetrize_matrix ────────────────────────────────────────────────────────

class TestSymmetrizeMatrix:
    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.ones((2, 3)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            symmetrize_matrix(np.ones((2, 2, 2)))

    def test_returns_symmetric(self):
        mat = np.array([[1.0, 3.0], [2.0, 4.0]])
        result = symmetrize_matrix(mat)
        np.testing.assert_array_equal(result, result.T)

    def test_max_symmetric(self):
        mat = np.array([[0.0, 0.3], [0.9, 0.0]])
        result = symmetrize_matrix(mat)
        # result[0,1] = max(0.3, 0.9) = 0.9
        assert result[0, 1] == pytest.approx(0.9)
        assert result[1, 0] == pytest.approx(0.9)

    def test_already_symmetric_unchanged(self):
        mat = np.array([[1.0, 2.0], [2.0, 3.0]])
        result = symmetrize_matrix(mat)
        np.testing.assert_allclose(result, mat.astype(np.float64))

    def test_returns_float64(self):
        mat = np.ones((3, 3))
        result = symmetrize_matrix(mat)
        assert result.dtype == np.float64

    def test_1x1_matrix(self):
        mat = np.array([[5.0]])
        result = symmetrize_matrix(mat)
        assert result[0, 0] == pytest.approx(5.0)


# ─── normalize_matrix ─────────────────────────────────────────────────────────

class TestNormalizeMatrix:
    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            normalize_matrix(np.array([1.0, 2.0]))

    def test_invalid_axis_raises(self):
        mat = np.ones((2, 2))
        with pytest.raises(ValueError, match="axis"):
            normalize_matrix(mat, axis=2)

    def test_axis_row_default(self):
        mat = np.array([[2.0, 4.0], [1.0, 3.0]])
        result = normalize_matrix(mat, axis=1)
        # Row 0: max=4.0, so [0.5, 1.0]
        np.testing.assert_allclose(result[0], [0.5, 1.0], atol=1e-9)
        # Row 1: max=3.0, so [1/3, 1.0]
        np.testing.assert_allclose(result[1], [1.0 / 3.0, 1.0], atol=1e-9)

    def test_axis_col(self):
        mat = np.array([[2.0, 1.0], [4.0, 3.0]])
        result = normalize_matrix(mat, axis=0)
        # Col 0: max=4.0, so [0.5, 1.0]
        np.testing.assert_allclose(result[:, 0], [0.5, 1.0], atol=1e-9)

    def test_zero_row_stays_zero(self):
        mat = np.array([[0.0, 0.0], [1.0, 2.0]])
        result = normalize_matrix(mat, axis=1)
        np.testing.assert_array_equal(result[0], [0.0, 0.0])

    def test_returns_float64(self):
        mat = np.ones((2, 2))
        result = normalize_matrix(mat)
        assert result.dtype == np.float64

    def test_values_in_0_1(self):
        mat = np.random.default_rng(0).uniform(0, 10, (4, 5))
        result = normalize_matrix(mat, axis=1)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-9)


# ─── diagonal_zeros ───────────────────────────────────────────────────────────

class TestDiagonalZeros:
    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.ones((2, 3)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            diagonal_zeros(np.ones((2, 2, 2)))

    def test_diagonal_is_zero(self):
        mat = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
        result = diagonal_zeros(mat)
        np.testing.assert_array_equal(np.diag(result), [0.0, 0.0, 0.0])

    def test_off_diagonal_unchanged(self):
        mat = np.array([[5.0, 2.0], [3.0, 7.0]])
        result = diagonal_zeros(mat)
        assert result[0, 1] == pytest.approx(2.0)
        assert result[1, 0] == pytest.approx(3.0)

    def test_returns_copy(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = diagonal_zeros(mat)
        mat[0, 0] = 99.0
        assert result[0, 0] == 0.0

    def test_returns_float64(self):
        mat = np.ones((3, 3))
        result = diagonal_zeros(mat)
        assert result.dtype == np.float64


# ─── matrix_sparsity ──────────────────────────────────────────────────────────

class TestMatrixSparsity:
    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            matrix_sparsity(np.array([1.0, 2.0]))

    def test_empty_returns_1(self):
        mat = np.zeros((0, 3))
        assert matrix_sparsity(mat) == pytest.approx(1.0)

    def test_empty_0x0_returns_1(self):
        mat = np.zeros((0, 0))
        assert matrix_sparsity(mat) == pytest.approx(1.0)

    def test_all_zeros_returns_1(self):
        mat = np.zeros((3, 4))
        assert matrix_sparsity(mat) == pytest.approx(1.0)

    def test_all_nonzero_returns_0(self):
        mat = np.ones((3, 4))
        assert matrix_sparsity(mat) == pytest.approx(0.0)

    def test_half_zero(self):
        mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert matrix_sparsity(mat) == pytest.approx(0.5)

    def test_result_in_0_1(self):
        rng = np.random.default_rng(42)
        mat = rng.uniform(-1, 1, (5, 5))
        s = matrix_sparsity(mat)
        assert 0.0 <= s <= 1.0


# ─── top_k_per_row ────────────────────────────────────────────────────────────

class TestTopKPerRow:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k"):
            top_k_per_row(np.ones((3, 3)), 0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            top_k_per_row(np.ones((3, 3)), -1)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            top_k_per_row(np.array([1.0, 2.0, 3.0]), 1)

    def test_zeros_outside_top_k(self):
        mat = np.array([[3.0, 1.0, 4.0, 1.5]])
        result = top_k_per_row(mat, k=2)
        # Top-2: indices 2 (4.0) and 0 (3.0); rest should be 0
        assert result[0, 0] == pytest.approx(3.0)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(4.0)
        assert result[0, 3] == pytest.approx(0.0)

    def test_top_k_values_preserved(self):
        mat = np.array([[5.0, 3.0, 1.0]])
        result = top_k_per_row(mat, k=2)
        assert result[0, 0] == pytest.approx(5.0)
        assert result[0, 1] == pytest.approx(3.0)
        assert result[0, 2] == pytest.approx(0.0)

    def test_returns_float64(self):
        mat = np.ones((2, 2))
        result = top_k_per_row(mat, k=1)
        assert result.dtype == np.float64

    def test_k_larger_than_cols_keeps_all(self):
        mat = np.array([[1.0, 2.0, 3.0]])
        result = top_k_per_row(mat, k=10)
        np.testing.assert_allclose(result, mat.astype(np.float64))

    def test_multiple_rows(self):
        mat = np.array([
            [4.0, 2.0, 1.0],
            [1.0, 5.0, 3.0],
        ])
        result = top_k_per_row(mat, k=1)
        # Row 0: only index 0 (4.0) kept
        assert result[0, 0] == pytest.approx(4.0)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(0.0)
        # Row 1: only index 1 (5.0) kept
        assert result[1, 0] == pytest.approx(0.0)
        assert result[1, 1] == pytest.approx(5.0)
        assert result[1, 2] == pytest.approx(0.0)

    def test_returns_copy_not_inplace(self):
        mat = np.array([[3.0, 1.0, 2.0]])
        result = top_k_per_row(mat, k=1)
        mat[0, 0] = 99.0
        assert result[0, 0] == pytest.approx(3.0)
