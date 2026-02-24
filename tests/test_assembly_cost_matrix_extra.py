"""Extra tests for puzzle_reconstruction/assembly/cost_matrix.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.cost_matrix import (
    CostMatrix,
    apply_forbidden_mask,
    build_combined,
    build_from_distances,
    build_from_scores,
    normalize_costs,
    to_assignment_matrix,
    top_k_candidates,
)


def _score_mat(n=4, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    np.fill_diagonal(m, 1.0)
    return m


def _cm(n=4, fill=0.5):
    m = np.full((n, n), fill, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return CostMatrix(matrix=m, n_fragments=n, method="test")


# ─── CostMatrix (extra) ─────────────────────────────────────────────────────

class TestCostMatrixExtra:
    def test_params_stored(self):
        m = np.zeros((3, 3), dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=3, method="t", params={"key": 42})
        assert cm.params["key"] == 42

    def test_matrix_values_preserved(self):
        m = np.eye(3, dtype=np.float32) * 0.7
        cm = CostMatrix(matrix=m, n_fragments=3, method="t")
        np.testing.assert_allclose(cm.matrix, m)

    def test_1x1_ok(self):
        m = np.zeros((1, 1), dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=1, method="t")
        assert cm.n_fragments == 1

    def test_large_matrix_ok(self):
        n = 50
        m = np.zeros((n, n), dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=n, method="t")
        assert cm.n_fragments == n


# ─── build_from_scores (extra) ───────────────────────────────────────────────

class TestBuildFromScoresExtra:
    def test_2x2_matrix(self):
        m = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        cm = build_from_scores(m)
        assert cm.n_fragments == 2

    def test_symmetric_input_symmetric_output(self):
        m = _score_mat(5, seed=1)
        m = (m + m.T) / 2
        cm = build_from_scores(m)
        np.testing.assert_allclose(cm.matrix, cm.matrix.T, atol=1e-5)

    def test_output_nonneg(self):
        cm = build_from_scores(_score_mat(4))
        assert cm.matrix.min() >= -1e-6

    def test_n_fragments_matches(self):
        cm = build_from_scores(_score_mat(7))
        assert cm.n_fragments == 7

    def test_diagonal_zero(self):
        cm = build_from_scores(_score_mat(3))
        np.testing.assert_allclose(np.diag(cm.matrix), 0.0, atol=1e-6)


# ─── build_from_distances (extra) ────────────────────────────────────────────

class TestBuildFromDistancesExtra:
    def test_symmetric_input(self):
        d = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float32)
        cm = build_from_distances(d)
        assert cm.matrix[0, 1] == cm.matrix[1, 0]

    def test_zero_matrix(self):
        d = np.zeros((3, 3), dtype=np.float32)
        cm = build_from_distances(d)
        np.testing.assert_allclose(cm.matrix, 0.0, atol=1e-6)

    def test_n_fragments_correct(self):
        d = np.eye(4, dtype=np.float32) * 10
        cm = build_from_distances(d)
        assert cm.n_fragments == 4

    def test_large_values_ok(self):
        d = np.full((3, 3), 1e6, dtype=np.float32)
        np.fill_diagonal(d, 0.0)
        cm = build_from_distances(d, normalize=True)
        off_diag = cm.matrix[~np.eye(3, dtype=bool)]
        assert off_diag.max() <= 1.0 + 1e-5


# ─── build_combined (extra) ──────────────────────────────────────────────────

class TestBuildCombinedExtra:
    def test_three_matrices(self):
        cms = [_cm(n=3) for _ in range(3)]
        result = build_combined(cms)
        assert result.n_fragments == 3

    def test_custom_weights_applied(self):
        cms = [_cm(n=3), _cm(n=3)]
        result = build_combined(cms, weights=[1.0, 0.0])
        assert result.n_fragments == 3

    def test_result_dtype_float32(self):
        result = build_combined([_cm(n=3)])
        assert result.matrix.dtype == np.float32

    def test_diagonal_always_zero(self):
        cms = [_cm(n=4) for _ in range(3)]
        result = build_combined(cms)
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-6)


# ─── apply_forbidden_mask (extra) ────────────────────────────────────────────

class TestApplyForbiddenMaskExtra:
    def test_all_forbidden(self):
        cm = _cm(n=3)
        mask = np.ones((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask, fill_value=5.0)
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-6)
        assert result.matrix[0, 1] == pytest.approx(5.0)

    def test_no_forbidden_same(self):
        cm = _cm(n=3)
        mask = np.zeros((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask)
        np.testing.assert_allclose(result.matrix, cm.matrix, atol=1e-6)

    def test_single_pair_forbidden(self):
        cm = _cm(n=4, fill=0.3)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = True
        result = apply_forbidden_mask(cm, mask, fill_value=9.0)
        assert result.matrix[1, 2] == pytest.approx(9.0)
        assert result.matrix[0, 1] == pytest.approx(0.3, abs=1e-5)

    def test_params_n_forbidden(self):
        cm = _cm(n=3)
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 1] = True
        mask[1, 0] = True
        result = apply_forbidden_mask(cm, mask)
        assert result.params["n_forbidden"] == 2


# ─── normalize_costs (extra) ─────────────────────────────────────────────────

class TestNormalizeCostsExtra:
    def test_minmax_diagonal_zero(self):
        cm = _cm(n=5, fill=0.7)
        result = normalize_costs(cm, method="minmax")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_zscore_diagonal_zero(self):
        m = np.random.default_rng(0).uniform(0.1, 0.9, (4, 4)).astype(np.float32)
        np.fill_diagonal(m, 0.0)
        cm = CostMatrix(matrix=m, n_fragments=4, method="t")
        result = normalize_costs(cm, method="zscore")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_rank_diagonal_zero(self):
        cm = _cm(n=4)
        result = normalize_costs(cm, method="rank")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_result_is_cost_matrix(self):
        cm = _cm(n=3)
        result = normalize_costs(cm, method="minmax")
        assert isinstance(result, CostMatrix)

    def test_n_fragments_preserved(self):
        cm = _cm(n=6)
        result = normalize_costs(cm, method="minmax")
        assert result.n_fragments == 6


# ─── to_assignment_matrix (extra) ────────────────────────────────────────────

class TestToAssignmentMatrixExtra:
    def test_1x1(self):
        cm = _cm(n=1)
        result = to_assignment_matrix(cm)
        assert result.shape == (1, 1)

    def test_diagonal_positive(self):
        cm = _cm(n=3, fill=0.5)
        result = to_assignment_matrix(cm)
        for i in range(3):
            assert result[i, i] > 0

    def test_result_ndarray(self):
        cm = _cm(n=4)
        result = to_assignment_matrix(cm)
        assert isinstance(result, np.ndarray)


# ─── top_k_candidates (extra) ────────────────────────────────────────────────

class TestTopKCandidatesExtra:
    def test_k_1_returns_single_candidate(self):
        cm = _cm(n=5)
        result = top_k_candidates(cm, k=1)
        for cands in result.values():
            assert len(cands) == 1

    def test_all_keys_present(self):
        cm = _cm(n=4)
        result = top_k_candidates(cm, k=2)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_candidate_ids_unique(self):
        cm = _cm(n=5)
        result = top_k_candidates(cm, k=3)
        for cands in result.values():
            ids = [c[0] for c in cands]
            assert len(ids) == len(set(ids))

    def test_costs_nonneg(self):
        cm = _cm(n=4, fill=0.5)
        result = top_k_candidates(cm, k=3)
        for cands in result.values():
            for _, cost in cands:
                assert cost >= 0.0
