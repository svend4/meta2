"""Extra tests for puzzle_reconstruction/utils/score_matrix_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.score_matrix_utils import (
    ScoreMatrixConfig,
    MatrixStats,
    RankEntry,
    zero_diagonal,
    symmetrize,
    threshold_matrix,
    normalize_rows,
    top_k_indices,
    matrix_stats,
    top_k_per_row,
    filter_by_threshold,
    intra_fragment_mask,
    apply_intra_fragment_mask,
    batch_matrix_stats,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sym_matrix(n=4):
    m = np.random.RandomState(42).rand(n, n)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


# ─── ScoreMatrixConfig ────────────────────────────────────────────────────────

class TestScoreMatrixConfigExtra:
    def test_defaults(self):
        cfg = ScoreMatrixConfig()
        assert cfg.threshold == pytest.approx(0.0)
        assert cfg.top_k == 10 and cfg.symmetrize is True

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ScoreMatrixConfig(threshold=1.5)

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            ScoreMatrixConfig(top_k=0)


# ─── MatrixStats ──────────────────────────────────────────────────────────────

class TestMatrixStatsExtra:
    def test_to_dict_keys(self):
        ms = MatrixStats(n_edges=4, n_nonzero=6, mean_score=0.5,
                          max_score=0.9, min_score=0.1, sparsity=0.5,
                          top_pair=(0, 1))
        d = ms.to_dict()
        for k in ("n_edges", "n_nonzero", "mean_score", "sparsity", "top_pair"):
            assert k in d


# ─── RankEntry ────────────────────────────────────────────────────────────────

class TestRankEntryExtra:
    def test_lt_higher_score_first(self):
        a = RankEntry(idx=0, score=0.9)
        b = RankEntry(idx=1, score=0.3)
        assert a < b  # a has higher score so a < b in sort-descending order

    def test_sort_descending(self):
        entries = [RankEntry(idx=0, score=0.3), RankEntry(idx=1, score=0.9)]
        s = sorted(entries)
        assert s[0].score == pytest.approx(0.9)


# ─── zero_diagonal ───────────────────────────────────────────────────────────

class TestZeroDiagonalExtra:
    def test_diagonal_zeroed(self):
        m = np.ones((3, 3))
        r = zero_diagonal(m)
        assert all(r[i, i] == 0.0 for i in range(3))

    def test_off_diagonal_preserved(self):
        m = np.ones((3, 3))
        r = zero_diagonal(m)
        assert r[0, 1] == pytest.approx(1.0)


# ─── symmetrize ───────────────────────────────────────────────────────────────

class TestSymmetrizeExtra:
    def test_result_is_symmetric(self):
        m = np.array([[0, 1], [3, 0]], dtype=float)
        s = symmetrize(m)
        assert np.allclose(s, s.T)

    def test_average_values(self):
        m = np.array([[0, 2], [4, 0]], dtype=float)
        s = symmetrize(m)
        assert s[0, 1] == pytest.approx(3.0)


# ─── threshold_matrix ────────────────────────────────────────────────────────

class TestThresholdMatrixExtra:
    def test_below_threshold_zeroed(self):
        m = np.array([[0.1, 0.5], [0.3, 0.9]])
        r = threshold_matrix(m, 0.4)
        assert r[0, 0] == pytest.approx(0.0)
        assert r[0, 1] == pytest.approx(0.5)


# ─── normalize_rows ──────────────────────────────────────────────────────────

class TestNormalizeRowsExtra:
    def test_rows_sum_to_one(self):
        m = np.array([[1, 2], [3, 4]], dtype=float)
        n = normalize_rows(m)
        for i in range(2):
            assert n[i].sum() == pytest.approx(1.0)

    def test_zero_row_stays_zero(self):
        m = np.array([[0, 0], [1, 1]], dtype=float)
        n = normalize_rows(m)
        assert n[0].sum() == pytest.approx(0.0)


# ─── top_k_indices ────────────────────────────────────────────────────────────

class TestTopKIndicesExtra:
    def test_returns_k(self):
        row = np.array([0.1, 0.9, 0.5])
        idx = top_k_indices(row, 2)
        assert len(idx) == 2

    def test_first_is_max(self):
        row = np.array([0.1, 0.9, 0.5])
        assert top_k_indices(row, 1)[0] == 1

    def test_k_zero_returns_empty(self):
        assert len(top_k_indices(np.array([1, 2, 3]), 0)) == 0


# ─── matrix_stats ─────────────────────────────────────────────────────────────

class TestMatrixStatsComputeExtra:
    def test_identity_returns_zero_nonzero(self):
        m = np.eye(3)
        s = matrix_stats(m)
        assert s.n_nonzero == 0

    def test_full_matrix(self):
        m = _sym_matrix(4)
        s = matrix_stats(m)
        assert s.n_nonzero > 0 and 0.0 <= s.sparsity <= 1.0


# ─── top_k_per_row ───────────────────────────────────────────────────────────

class TestTopKPerRowExtra:
    def test_returns_n_lists(self):
        m = _sym_matrix(4)
        result = top_k_per_row(m, 2)
        assert len(result) == 4

    def test_entries_sorted_descending(self):
        m = _sym_matrix(4)
        result = top_k_per_row(m, 3)
        for row_entries in result:
            scores = [e.score for e in row_entries]
            assert scores == sorted(scores, reverse=True)


# ─── filter_by_threshold ─────────────────────────────────────────────────────

class TestFilterByThresholdExtra:
    def test_returns_pairs(self):
        m = _sym_matrix(4)
        filtered, pairs = filter_by_threshold(m, 0.5)
        for r, c, s in pairs:
            assert s > 0.5


# ─── intra_fragment_mask ──────────────────────────────────────────────────────

class TestIntraFragmentMaskExtra:
    def test_shape(self):
        mask = intra_fragment_mask([3, 2])
        assert mask.shape == (5, 5)

    def test_diagonal_blocks_true(self):
        mask = intra_fragment_mask([2, 3])
        assert mask[0, 1] is np.bool_(True)
        assert mask[2, 4] is np.bool_(True)

    def test_cross_blocks_false(self):
        mask = intra_fragment_mask([2, 3])
        assert mask[0, 2] is np.bool_(False)


# ─── apply_intra_fragment_mask ────────────────────────────────────────────────

class TestApplyIntraFragmentMaskExtra:
    def test_intra_zeroed(self):
        m = np.ones((5, 5))
        r = apply_intra_fragment_mask(m, [2, 3])
        assert r[0, 1] == pytest.approx(0.0)
        assert r[0, 2] == pytest.approx(1.0)


# ─── batch_matrix_stats ──────────────────────────────────────────────────────

class TestBatchMatrixStatsExtra:
    def test_length(self):
        matrices = [_sym_matrix(3), _sym_matrix(4)]
        result = batch_matrix_stats(matrices)
        assert len(result) == 2
