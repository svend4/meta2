"""Extra tests for puzzle_reconstruction/assembly/cost_matrix.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.assembly.cost_matrix import (
    CostMatrix,
    build_from_scores,
    build_from_distances,
    build_combined,
    apply_forbidden_mask,
    normalize_costs,
    to_assignment_matrix,
    top_k_candidates,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _scores(n=4, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n)).astype(np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _cm(n=4, fill=0.5):
    m = np.full((n, n), fill, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return CostMatrix(matrix=m, n_fragments=n, method="test")


# ─── CostMatrix (extra) ──────────────────────────────────────────────────────

class TestCostMatrixExtra2:
    def test_method_stored(self):
        cm = CostMatrix(matrix=np.zeros((3, 3), dtype=np.float32),
                        n_fragments=3, method="test_method")
        assert cm.method == "test_method"

    def test_params_stored(self):
        cm = CostMatrix(matrix=np.zeros((3, 3), dtype=np.float32),
                        n_fragments=3, method="t", params={"alpha": 0.5})
        assert cm.params["alpha"] == pytest.approx(0.5)

    def test_non_square_raises(self):
        m = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            CostMatrix(matrix=m, n_fragments=3, method="t")

    def test_n_mismatch_raises(self):
        m = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            CostMatrix(matrix=m, n_fragments=5, method="t")

    def test_1x1_ok(self):
        cm = CostMatrix(matrix=np.zeros((1, 1), dtype=np.float32),
                        n_fragments=1, method="t")
        assert cm.n_fragments == 1

    def test_matrix_values_preserved(self):
        m = np.eye(3, dtype=np.float32) * 0.7
        cm = CostMatrix(matrix=m, n_fragments=3, method="t")
        np.testing.assert_allclose(cm.matrix, m)


# ─── build_from_scores (extra) ───────────────────────────────────────────────

class TestBuildFromScoresExtra2:
    def test_returns_cost_matrix(self):
        assert isinstance(build_from_scores(_scores(4)), CostMatrix)

    def test_n_fragments_correct(self):
        assert build_from_scores(_scores(5)).n_fragments == 5

    def test_diagonal_zero(self):
        cm = build_from_scores(_scores(4))
        np.testing.assert_allclose(np.diag(cm.matrix), 0.0, atol=1e-6)

    def test_dtype_float32(self):
        assert build_from_scores(_scores(3)).matrix.dtype == np.float32

    def test_method_name(self):
        assert build_from_scores(_scores(3)).method == "from_scores"

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_from_scores(np.ones((3, 4), dtype=np.float32))

    def test_invert_high_score_low_cost(self):
        n = 4
        m = np.zeros((n, n), dtype=np.float32)
        m[0, 1] = m[1, 0] = 1.0
        m[0, 2] = m[2, 0] = 0.0
        cm = build_from_scores(m, invert=True)
        assert cm.matrix[0, 1] < cm.matrix[0, 2]


# ─── build_from_distances (extra) ────────────────────────────────────────────

class TestBuildFromDistancesExtra2:
    def test_returns_cost_matrix(self):
        d = np.abs(_scores(4))
        assert isinstance(build_from_distances(d), CostMatrix)

    def test_diagonal_zero(self):
        d = np.abs(_scores(5))
        cm = build_from_distances(d)
        np.testing.assert_allclose(np.diag(cm.matrix), 0.0, atol=1e-6)

    def test_normalize_max_one(self):
        d = np.abs(_scores(5))
        cm = build_from_distances(d, normalize=True)
        assert cm.matrix.max() <= 1.0 + 1e-6

    def test_normalize_false_preserves(self):
        d = np.array([[0., 10.], [10., 0.]], dtype=np.float32)
        cm = build_from_distances(d, normalize=False)
        assert cm.matrix[0, 1] == pytest.approx(10.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_from_distances(np.ones((3, 5), dtype=np.float32))

    def test_method_name(self):
        d = np.abs(_scores(3))
        cm = build_from_distances(d)
        assert cm.method == "from_distances"


# ─── build_combined (extra) ──────────────────────────────────────────────────

class TestBuildCombinedExtra2:
    def test_returns_cost_matrix(self):
        result = build_combined([_cm(4), _cm(4)])
        assert isinstance(result, CostMatrix)

    def test_n_fragments_correct(self):
        result = build_combined([_cm(5)])
        assert result.n_fragments == 5

    def test_diagonal_zero(self):
        result = build_combined([_cm(4), _cm(4)])
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-6)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_combined([])

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_combined([_cm(3), _cm(5)])

    def test_single_matrix_identity(self):
        cm = _cm(4)
        result = build_combined([cm])
        np.testing.assert_allclose(result.matrix, cm.matrix, atol=1e-6)

    def test_method_name(self):
        result = build_combined([_cm(3)])
        assert result.method == "combined"

    def test_dtype_float32(self):
        result = build_combined([_cm(3), _cm(3)])
        assert result.matrix.dtype == np.float32


# ─── apply_forbidden_mask (extra) ────────────────────────────────────────────

class TestApplyForbiddenMaskExtra2:
    def test_forbidden_set_to_fill(self):
        cm = _cm(4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = mask[1, 0] = True
        result = apply_forbidden_mask(cm, mask, fill_value=2.0)
        assert result.matrix[0, 1] == pytest.approx(2.0)
        assert result.matrix[1, 0] == pytest.approx(2.0)

    def test_diagonal_still_zero(self):
        cm = _cm(3)
        mask = np.ones((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask, fill_value=5.0)
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-6)

    def test_no_forbidden_unchanged(self):
        cm = _cm(3)
        mask = np.zeros((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask)
        np.testing.assert_allclose(result.matrix, cm.matrix, atol=1e-6)

    def test_wrong_shape_raises(self):
        cm = _cm(4)
        with pytest.raises(ValueError):
            apply_forbidden_mask(cm, np.zeros((3, 3), dtype=bool))

    def test_method_masked(self):
        cm = _cm(3)
        mask = np.zeros((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask)
        assert result.method == "masked"

    def test_n_forbidden_in_params(self):
        cm = _cm(4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = True
        mask[1, 0] = True
        result = apply_forbidden_mask(cm, mask)
        assert result.params["n_forbidden"] == 2


# ─── normalize_costs (extra) ─────────────────────────────────────────────────

class TestNormalizeCostsExtra2:
    def test_returns_cost_matrix(self):
        assert isinstance(normalize_costs(_cm(4), method="minmax"), CostMatrix)

    def test_diagonal_zero_minmax(self):
        result = normalize_costs(_cm(5), method="minmax")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_diagonal_zero_zscore(self):
        cm = CostMatrix(
            matrix=np.random.default_rng(0).uniform(0.1, 0.9, (4, 4)).astype(np.float32),
            n_fragments=4, method="t"
        )
        np.fill_diagonal(cm.matrix, 0.0)
        result = normalize_costs(cm, method="zscore")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_diagonal_zero_rank(self):
        result = normalize_costs(_cm(4), method="rank")
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_costs(_cm(3), method="xyz_unknown")

    def test_minmax_method_in_name(self):
        result = normalize_costs(_cm(3), method="minmax")
        assert "minmax" in result.method


# ─── to_assignment_matrix (extra) ────────────────────────────────────────────

class TestToAssignmentMatrixExtra2:
    def test_shape_preserved(self):
        cm = _cm(4)
        m = to_assignment_matrix(cm)
        assert m.shape == (4, 4)

    def test_diagonal_larger_than_max(self):
        cm = _cm(4, fill=0.5)
        m = to_assignment_matrix(cm)
        mx = float(cm.matrix.max())
        for d in np.diag(m):
            assert float(d) > mx

    def test_off_diagonal_unchanged(self):
        cm = _cm(4)
        m = to_assignment_matrix(cm)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert m[i, j] == pytest.approx(cm.matrix[i, j])

    def test_dtype_float32(self):
        assert to_assignment_matrix(_cm(3)).dtype == np.float32

    def test_1x1(self):
        cm = _cm(1)
        m = to_assignment_matrix(cm)
        assert m.shape == (1, 1)


# ─── top_k_candidates (extra) ────────────────────────────────────────────────

class TestTopKCandidatesExtra2:
    def test_all_keys_present(self):
        cm = _cm(5)
        result = top_k_candidates(cm, k=2)
        assert set(result.keys()) == set(range(5))

    def test_k_1_one_candidate(self):
        cm = _cm(4)
        result = top_k_candidates(cm, k=1)
        for cands in result.values():
            assert len(cands) == 1

    def test_self_not_in_candidates(self):
        cm = _cm(4)
        result = top_k_candidates(cm, k=3)
        for i, cands in result.items():
            for j, _ in cands:
                assert j != i

    def test_sorted_ascending(self):
        cm = _cm(5)
        result = top_k_candidates(cm, k=3)
        for cands in result.values():
            costs = [c for _, c in cands]
            assert costs == sorted(costs)

    def test_k_exceeds_n_capped(self):
        cm = _cm(3)
        result = top_k_candidates(cm, k=100)
        for cands in result.values():
            assert len(cands) <= 2  # n-1 = 2

    def test_known_order(self):
        m = np.array([[0., 1., 3., 2.],
                      [1., 0., 4., 2.],
                      [3., 4., 0., 1.],
                      [2., 2., 1., 0.]], dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=4, method="test")
        result = top_k_candidates(cm, k=2)
        idxs_0 = [j for j, _ in result[0]]
        assert idxs_0[0] == 1  # cost 1 is lowest
        assert idxs_0[1] == 3  # cost 2 is next
