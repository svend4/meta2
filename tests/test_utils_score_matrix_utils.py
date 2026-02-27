"""Tests for puzzle_reconstruction.utils.score_matrix_utils."""
import pytest
import numpy as np

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

np.random.seed(42)


# ─── ScoreMatrixConfig ───────────────────────────────────────────────────────

def test_config_defaults():
    cfg = ScoreMatrixConfig()
    assert cfg.threshold == pytest.approx(0.0)
    assert cfg.top_k == 10
    assert cfg.symmetrize is True


def test_config_invalid_threshold():
    with pytest.raises(ValueError):
        ScoreMatrixConfig(threshold=1.5)


def test_config_invalid_top_k():
    with pytest.raises(ValueError):
        ScoreMatrixConfig(top_k=0)


def test_config_valid():
    cfg = ScoreMatrixConfig(threshold=0.5, top_k=5, symmetrize=False)
    assert cfg.threshold == pytest.approx(0.5)
    assert cfg.top_k == 5
    assert cfg.symmetrize is False


# ─── zero_diagonal ───────────────────────────────────────────────────────────

def test_zero_diagonal_basic():
    m = np.ones((4, 4))
    result = zero_diagonal(m)
    np.testing.assert_array_equal(np.diag(result), 0.0)


def test_zero_diagonal_off_diag_unchanged():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = zero_diagonal(m)
    assert result[0, 1] == pytest.approx(2.0)
    assert result[1, 0] == pytest.approx(3.0)


def test_zero_diagonal_does_not_modify_original():
    m = np.ones((3, 3))
    _ = zero_diagonal(m)
    assert m[0, 0] == pytest.approx(1.0)  # original unchanged


# ─── symmetrize ──────────────────────────────────────────────────────────────

def test_symmetrize_basic():
    m = np.array([[0.0, 0.8], [0.4, 0.0]])
    result = symmetrize(m)
    assert result[0, 1] == pytest.approx(0.6)
    assert result[1, 0] == pytest.approx(0.6)


def test_symmetrize_already_symmetric():
    m = np.array([[0.0, 0.5], [0.5, 0.0]])
    result = symmetrize(m)
    np.testing.assert_allclose(result, m)


def test_symmetrize_result_is_symmetric():
    rng = np.random.default_rng(0)
    m = rng.random((5, 5))
    result = symmetrize(m)
    np.testing.assert_allclose(result, result.T, atol=1e-10)


# ─── threshold_matrix ────────────────────────────────────────────────────────

def test_threshold_matrix_zeroes_below():
    m = np.array([[0.0, 0.3], [0.7, 0.0]])
    result = threshold_matrix(m, 0.5)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[1, 0] == pytest.approx(0.7)


def test_threshold_matrix_all_zeroed():
    m = np.array([[0.0, 0.1], [0.2, 0.0]])
    result = threshold_matrix(m, 0.5)
    np.testing.assert_array_equal(result, 0.0)


def test_threshold_matrix_zero_threshold():
    m = np.array([[0.5, 0.3], [0.2, 0.0]])
    result = threshold_matrix(m, 0.0)
    assert result[0, 0] == pytest.approx(0.5)


# ─── normalize_rows ──────────────────────────────────────────────────────────

def test_normalize_rows_sums_to_one():
    m = np.array([[1.0, 3.0], [2.0, 2.0]])
    result = normalize_rows(m)
    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_normalize_rows_zero_row():
    m = np.array([[0.0, 0.0], [2.0, 2.0]])
    result = normalize_rows(m)
    # zero row stays zero
    np.testing.assert_array_equal(result[0], [0.0, 0.0])


def test_normalize_rows_shape_preserved():
    m = np.ones((4, 5))
    result = normalize_rows(m)
    assert result.shape == (4, 5)


# ─── top_k_indices ───────────────────────────────────────────────────────────

def test_top_k_indices_basic():
    row = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    result = top_k_indices(row, 2)
    assert len(result) == 2
    assert result[0] == 3  # 0.9 is max


def test_top_k_indices_k_zero():
    row = np.array([0.1, 0.5, 0.3])
    result = top_k_indices(row, 0)
    assert len(result) == 0


def test_top_k_indices_k_larger():
    row = np.array([0.1, 0.5])
    result = top_k_indices(row, 10)
    assert len(result) == 2


# ─── matrix_stats ────────────────────────────────────────────────────────────

def test_matrix_stats_returns_stats():
    m = np.array([[0.0, 0.8, 0.3], [0.8, 0.0, 0.5], [0.3, 0.5, 0.0]])
    stats = matrix_stats(m)
    assert isinstance(stats, MatrixStats)


def test_matrix_stats_n_edges():
    m = np.zeros((5, 5))
    m[0, 1] = 0.8
    m[1, 0] = 0.8
    stats = matrix_stats(m)
    assert stats.n_edges == 5


def test_matrix_stats_max_score():
    m = np.array([[0.0, 0.9], [0.9, 0.0]])
    stats = matrix_stats(m)
    assert stats.max_score == pytest.approx(0.9)


def test_matrix_stats_sparsity():
    m = np.zeros((3, 3))
    stats = matrix_stats(m)
    assert stats.sparsity == pytest.approx(1.0)


def test_matrix_stats_all_nonzero():
    m = np.ones((3, 3))
    np.fill_diagonal(m, 0)
    stats = matrix_stats(m)
    assert stats.sparsity == pytest.approx(0.0)


# ─── top_k_per_row ───────────────────────────────────────────────────────────

def test_top_k_per_row_length():
    m = np.array([[0.0, 0.8, 0.3], [0.8, 0.0, 0.5], [0.3, 0.5, 0.0]])
    result = top_k_per_row(m, k=2)
    assert len(result) == 3


def test_top_k_per_row_no_self():
    m = np.eye(3) * 0.5 + np.array([[0, 0.8, 0.3], [0.8, 0, 0.5], [0.3, 0.5, 0]])
    result = top_k_per_row(m, k=2, exclude_self=True)
    for row_result in result:
        for entry in row_result:
            assert isinstance(entry, RankEntry)


def test_top_k_per_row_descending():
    m = np.array([[0.0, 0.3, 0.9], [0.3, 0.0, 0.6], [0.9, 0.6, 0.0]])
    result = top_k_per_row(m, k=2)
    # row 0: top should be index 2 with score 0.9
    assert result[0][0].idx == 2
    assert result[0][0].score == pytest.approx(0.9)


# ─── filter_by_threshold ─────────────────────────────────────────────────────

def test_filter_by_threshold_returns_tuple():
    m = np.array([[0.0, 0.8], [0.8, 0.0]])
    filtered, pairs = filter_by_threshold(m, 0.5)
    assert isinstance(filtered, np.ndarray)
    assert isinstance(pairs, list)


def test_filter_by_threshold_pairs_sorted():
    m = np.array([[0.0, 0.9, 0.6], [0.9, 0.0, 0.7], [0.6, 0.7, 0.0]])
    _, pairs = filter_by_threshold(m, 0.5)
    scores = [p[2] for p in pairs]
    assert scores == sorted(scores, reverse=True)


def test_filter_by_threshold_nothing_above():
    m = np.array([[0.0, 0.1], [0.1, 0.0]])
    _, pairs = filter_by_threshold(m, 0.5)
    assert pairs == []


# ─── intra_fragment_mask ─────────────────────────────────────────────────────

def test_intra_fragment_mask_shape():
    mask = intra_fragment_mask([3, 2])
    assert mask.shape == (5, 5)


def test_intra_fragment_mask_diagonal_blocks_true():
    mask = intra_fragment_mask([2, 3])
    # First 2x2 block should be True
    assert mask[0, 1] is np.bool_(True)
    assert mask[1, 0] is np.bool_(True)
    # Cross block should be False
    assert mask[0, 2] is np.bool_(False)


def test_intra_fragment_mask_single_fragment():
    mask = intra_fragment_mask([4])
    assert mask.all()


# ─── apply_intra_fragment_mask ───────────────────────────────────────────────

def test_apply_intra_fragment_mask_zeros_blocks():
    m = np.ones((4, 4))
    result = apply_intra_fragment_mask(m, [2, 2])
    # intra-fragment entries (block 0-1,0-1 and 2-3,2-3) should be 0
    assert result[0, 1] == pytest.approx(0.0)
    assert result[2, 3] == pytest.approx(0.0)


def test_apply_intra_fragment_mask_preserves_cross():
    m = np.ones((4, 4))
    result = apply_intra_fragment_mask(m, [2, 2])
    # cross-fragment entries should be 1
    assert result[0, 2] == pytest.approx(1.0)
    assert result[1, 3] == pytest.approx(1.0)


# ─── batch_matrix_stats ──────────────────────────────────────────────────────

def test_batch_matrix_stats_length():
    matrices = [np.zeros((3, 3)), np.ones((4, 4))]
    np.fill_diagonal(matrices[1], 0)
    results = batch_matrix_stats(matrices)
    assert len(results) == 2


def test_batch_matrix_stats_returns_list_of_stats():
    matrices = [np.zeros((3, 3))]
    results = batch_matrix_stats(matrices)
    assert isinstance(results[0], MatrixStats)
