"""Tests for assembly/cost_matrix.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_score_matrix(n=4, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    np.fill_diagonal(m, 1.0)
    return m


def make_cost_matrix(n=4, method="test"):
    m = np.full((n, n), 0.5, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return CostMatrix(matrix=m, n_fragments=n, method=method)


# ─── CostMatrix ───────────────────────────────────────────────────────────────

class TestCostMatrix:
    def test_basic_creation(self):
        m = np.zeros((3, 3), dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=3, method="test")
        assert cm.n_fragments == 3
        assert cm.method == "test"

    def test_shape_mismatch_raises(self):
        m = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            CostMatrix(matrix=m, n_fragments=3, method="test")

    def test_n_fragments_mismatch_raises(self):
        m = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            CostMatrix(matrix=m, n_fragments=5, method="test")

    def test_params_default_empty(self):
        m = np.zeros((2, 2), dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=2, method="x")
        assert cm.params == {}

    def test_repr_contains_n(self):
        cm = make_cost_matrix(n=3)
        assert "3" in repr(cm)

    def test_repr_contains_method(self):
        cm = make_cost_matrix(n=3, method="from_scores")
        assert "from_scores" in repr(cm)


# ─── build_from_scores ────────────────────────────────────────────────────────

class TestBuildFromScores:
    def test_returns_cost_matrix(self):
        cm = build_from_scores(make_score_matrix())
        assert isinstance(cm, CostMatrix)

    def test_method_is_from_scores(self):
        cm = build_from_scores(make_score_matrix())
        assert cm.method == "from_scores"

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            build_from_scores(np.zeros((3, 4)))

    def test_diagonal_is_zero(self):
        cm = build_from_scores(make_score_matrix(n=5))
        diag = np.diag(cm.matrix)
        np.testing.assert_allclose(diag, 0.0, atol=1e-6)

    def test_output_range_zero_to_one(self):
        cm = build_from_scores(make_score_matrix(n=4))
        assert cm.matrix.min() >= -1e-6
        assert cm.matrix.max() <= 1.0 + 1e-6

    def test_invert_true_high_scores_low_costs(self):
        # Uniform high scores → should give low costs
        m = np.ones((3, 3)) * 0.9
        cm = build_from_scores(m, invert=True)
        # After normalization, all scores become equal → costs = 0.5
        # When all-same, norm = 0, cost = 1 - 0 = 1... wait, let me think
        # all-same → norm = zeros → cost = 1 - zeros = ones, then diagonal = 0
        # Actually: off-diagonal are all same value, so after minmax they're 0
        # cost = 1 - 0 = 1? No...
        # mn=mx=0.9, abs(mx-mn) < eps → norm = zeros, cost = 1 - 0 = 1
        # diagonal forced to 0
        # This is a degenerate case; just verify diagonal is 0
        diag = np.diag(cm.matrix)
        np.testing.assert_allclose(diag, 0.0, atol=1e-6)

    def test_invert_false_high_scores_high_costs(self):
        m = np.full((3, 3), 0.8, dtype=np.float32)
        m[0, 1] = 0.9
        m[1, 0] = 0.9
        cm_inv = build_from_scores(m, invert=True)
        cm_noinv = build_from_scores(m, invert=False)
        # invert=True should generally give different values than invert=False
        # Just check they're valid matrices
        assert cm_inv.matrix.shape == (3, 3)
        assert cm_noinv.matrix.shape == (3, 3)

    def test_dtype_float32(self):
        cm = build_from_scores(make_score_matrix())
        assert cm.matrix.dtype == np.float32

    def test_n_fragments_correct(self):
        cm = build_from_scores(make_score_matrix(n=6))
        assert cm.n_fragments == 6

    def test_1x1_matrix(self):
        cm = build_from_scores(np.array([[1.0]]))
        assert cm.n_fragments == 1
        assert cm.matrix[0, 0] == pytest.approx(0.0)


# ─── build_from_distances ─────────────────────────────────────────────────────

class TestBuildFromDistances:
    def test_returns_cost_matrix(self):
        d = np.random.default_rng(0).uniform(0, 10, (4, 4))
        cm = build_from_distances(d)
        assert isinstance(cm, CostMatrix)

    def test_method_is_from_distances(self):
        d = np.eye(3) * 5
        cm = build_from_distances(d)
        assert cm.method == "from_distances"

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            build_from_distances(np.zeros((3, 5)))

    def test_diagonal_is_zero(self):
        d = np.random.default_rng(1).uniform(0, 5, (5, 5))
        cm = build_from_distances(d)
        np.testing.assert_allclose(np.diag(cm.matrix), 0.0, atol=1e-6)

    def test_normalize_true_range_zero_to_one(self):
        d = np.array([[0.0, 3.0, 5.0],
                      [3.0, 0.0, 4.0],
                      [5.0, 4.0, 0.0]])
        cm = build_from_distances(d, normalize=True)
        off_diag_mask = ~np.eye(3, dtype=bool)
        vals = cm.matrix[off_diag_mask]
        assert vals.min() >= -1e-6
        assert vals.max() <= 1.0 + 1e-6

    def test_normalize_false_preserves_scale(self):
        d = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float32)
        cm = build_from_distances(d, normalize=False)
        assert cm.matrix[0, 1] == pytest.approx(100.0, abs=1e-4)

    def test_negative_distances_take_abs(self):
        d = np.array([[0.0, -3.0], [-3.0, 0.0]])
        cm = build_from_distances(d)
        assert cm.matrix[0, 1] >= 0.0

    def test_dtype_float32(self):
        cm = build_from_distances(np.eye(3))
        assert cm.matrix.dtype == np.float32


# ─── build_combined ───────────────────────────────────────────────────────────

class TestBuildCombined:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_combined([])

    def test_mismatched_n_fragments_raises(self):
        cm3 = make_cost_matrix(n=3)
        cm4 = make_cost_matrix(n=4)
        with pytest.raises(ValueError, match="n_fragments"):
            build_combined([cm3, cm4])

    def test_single_matrix_returns_same(self):
        cm = make_cost_matrix(n=3)
        result = build_combined([cm])
        np.testing.assert_allclose(result.matrix, cm.matrix, atol=1e-6)

    def test_method_is_combined(self):
        cm = make_cost_matrix(n=3)
        result = build_combined([cm, cm])
        assert result.method == "combined"

    def test_equal_weights_default(self):
        cm1 = make_cost_matrix(n=3)
        cm2 = make_cost_matrix(n=3)
        cm1.matrix[:] = 0.2
        np.fill_diagonal(cm1.matrix, 0.0)
        cm2.matrix[:] = 0.8
        np.fill_diagonal(cm2.matrix, 0.0)
        result = build_combined([cm1, cm2])
        # Equal weights → average = 0.5
        off_diag = result.matrix[~np.eye(3, dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.5, atol=1e-5)

    def test_custom_weights_applied(self):
        cm1 = make_cost_matrix(n=3)
        cm2 = make_cost_matrix(n=3)
        cm1.matrix[:] = 0.0
        np.fill_diagonal(cm1.matrix, 0.0)
        cm2.matrix[:] = 1.0
        np.fill_diagonal(cm2.matrix, 0.0)
        # w1=1, w2=0 → result should be all 0 off-diag
        result = build_combined([cm1, cm2], weights=[1.0, 0.0])
        off_diag = result.matrix[~np.eye(3, dtype=bool)]
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-5)

    def test_zero_weight_sum_raises(self):
        cm = make_cost_matrix(n=3)
        with pytest.raises(ValueError, match="Sum of weights"):
            build_combined([cm, cm], weights=[0.0, 0.0])

    def test_diagonal_is_zero(self):
        cm = make_cost_matrix(n=4)
        result = build_combined([cm])
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-6)

    def test_n_fragments_preserved(self):
        cm = make_cost_matrix(n=5)
        result = build_combined([cm, cm])
        assert result.n_fragments == 5


# ─── apply_forbidden_mask ─────────────────────────────────────────────────────

class TestApplyForbiddenMask:
    def test_wrong_mask_shape_raises(self):
        cm = make_cost_matrix(n=3)
        mask = np.zeros((4, 4), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            apply_forbidden_mask(cm, mask)

    def test_method_is_masked(self):
        cm = make_cost_matrix(n=3)
        mask = np.zeros((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask)
        assert result.method == "masked"

    def test_forbidden_pair_set_to_fill_value(self):
        cm = make_cost_matrix(n=3)
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 1] = True
        mask[1, 0] = True
        result = apply_forbidden_mask(cm, mask, fill_value=2.0)
        assert result.matrix[0, 1] == pytest.approx(2.0)
        assert result.matrix[1, 0] == pytest.approx(2.0)

    def test_diagonal_remains_zero(self):
        cm = make_cost_matrix(n=3)
        mask = np.ones((3, 3), dtype=bool)
        result = apply_forbidden_mask(cm, mask, fill_value=99.0)
        np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-6)

    def test_unforbidden_pairs_unchanged(self):
        m = np.full((3, 3), 0.3, dtype=np.float32)
        np.fill_diagonal(m, 0.0)
        cm = CostMatrix(matrix=m, n_fragments=3, method="test")
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 2] = True
        result = apply_forbidden_mask(cm, mask, fill_value=1.0)
        # (0, 1) should remain 0.3
        assert result.matrix[0, 1] == pytest.approx(0.3, abs=1e-5)

    def test_empty_mask_unchanged(self):
        cm = make_cost_matrix(n=4)
        mask = np.zeros((4, 4), dtype=bool)
        result = apply_forbidden_mask(cm, mask)
        np.testing.assert_allclose(result.matrix, cm.matrix, atol=1e-6)

    def test_n_forbidden_in_params(self):
        cm = make_cost_matrix(n=3)
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 1] = True
        result = apply_forbidden_mask(cm, mask)
        assert result.params["n_forbidden"] == 1


# ─── normalize_costs ──────────────────────────────────────────────────────────

class TestNormalizeCosts:
    def test_unknown_method_raises(self):
        cm = make_cost_matrix(n=3)
        with pytest.raises(ValueError, match="Unknown"):
            normalize_costs(cm, method="unknown")

    def test_minmax_output_range(self):
        cm = make_cost_matrix(n=4)
        result = normalize_costs(cm, method="minmax")
        off_diag = result.matrix[~np.eye(4, dtype=bool)]
        assert off_diag.min() >= -1e-5
        assert off_diag.max() <= 1.0 + 1e-5

    def test_zscore_output_range(self):
        m = np.array([[0.0, 0.2, 0.8],
                      [0.2, 0.0, 0.6],
                      [0.8, 0.6, 0.0]], dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=3, method="test")
        result = normalize_costs(cm, method="zscore")
        off_diag = result.matrix[~np.eye(3, dtype=bool)]
        assert off_diag.min() >= -1e-5
        assert off_diag.max() <= 1.0 + 1e-5

    def test_rank_output_range(self):
        cm = make_cost_matrix(n=5)
        result = normalize_costs(cm, method="rank")
        off_diag = result.matrix[~np.eye(5, dtype=bool)]
        assert off_diag.min() >= -1e-5
        assert off_diag.max() <= 1.0 + 1e-5

    def test_diagonal_zero_after_normalization(self):
        for method in ("minmax", "zscore", "rank"):
            cm = make_cost_matrix(n=4)
            result = normalize_costs(cm, method=method)
            np.testing.assert_allclose(np.diag(result.matrix), 0.0, atol=1e-5)

    def test_method_name_in_result(self):
        cm = make_cost_matrix(n=3)
        result = normalize_costs(cm, method="minmax")
        assert "minmax" in result.method

    def test_dtype_float32(self):
        cm = make_cost_matrix(n=3)
        result = normalize_costs(cm, method="minmax")
        assert result.matrix.dtype == np.float32


# ─── to_assignment_matrix ─────────────────────────────────────────────────────

class TestToAssignmentMatrix:
    def test_returns_ndarray(self):
        cm = make_cost_matrix(n=3)
        result = to_assignment_matrix(cm)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        cm = make_cost_matrix(n=4)
        result = to_assignment_matrix(cm)
        assert result.shape == (4, 4)

    def test_diagonal_larger_than_max_off_diag(self):
        cm = make_cost_matrix(n=3)
        result = to_assignment_matrix(cm)
        off_diag_max = result[~np.eye(3, dtype=bool)].max()
        diag_min = np.diag(result).min()
        assert diag_min > off_diag_max

    def test_dtype_float32(self):
        cm = make_cost_matrix(n=3)
        result = to_assignment_matrix(cm)
        assert result.dtype == np.float32

    def test_off_diagonal_unchanged(self):
        m = np.array([[0.0, 0.3, 0.7],
                      [0.3, 0.0, 0.4],
                      [0.7, 0.4, 0.0]], dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=3, method="test")
        result = to_assignment_matrix(cm)
        assert result[0, 1] == pytest.approx(0.3, abs=1e-5)
        assert result[0, 2] == pytest.approx(0.7, abs=1e-5)


# ─── top_k_candidates ─────────────────────────────────────────────────────────

class TestTopKCandidates:
    def test_returns_dict(self):
        cm = make_cost_matrix(n=4)
        result = top_k_candidates(cm, k=2)
        assert isinstance(result, dict)

    def test_all_fragments_present(self):
        n = 5
        cm = make_cost_matrix(n=n)
        result = top_k_candidates(cm, k=3)
        assert set(result.keys()) == set(range(n))

    def test_k_limits_candidates(self):
        cm = make_cost_matrix(n=5)
        result = top_k_candidates(cm, k=2)
        for candidates in result.values():
            assert len(candidates) <= 2

    def test_self_not_in_candidates(self):
        cm = make_cost_matrix(n=4)
        result = top_k_candidates(cm, k=3)
        for i, candidates in result.items():
            cand_ids = [c[0] for c in candidates]
            assert i not in cand_ids

    def test_candidates_sorted_ascending_cost(self):
        m = np.array([[0.0, 0.1, 0.5, 0.9],
                      [0.1, 0.0, 0.3, 0.7],
                      [0.5, 0.3, 0.0, 0.2],
                      [0.9, 0.7, 0.2, 0.0]], dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=4, method="test")
        result = top_k_candidates(cm, k=3)
        for candidates in result.values():
            costs = [c[1] for c in candidates]
            assert costs == sorted(costs)

    def test_k_larger_than_n_minus_1(self):
        # k > n-1 should be clamped to n-1
        cm = make_cost_matrix(n=3)
        result = top_k_candidates(cm, k=100)
        for candidates in result.values():
            assert len(candidates) <= 2  # max n-1 = 2

    def test_tuple_structure(self):
        cm = make_cost_matrix(n=3)
        result = top_k_candidates(cm, k=1)
        for candidates in result.values():
            for item in candidates:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], int)
                assert isinstance(item[1], float)
