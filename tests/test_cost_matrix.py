"""
Тесты для puzzle_reconstruction.assembly.cost_matrix.
"""
import pytest
import numpy as np

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


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _uniform_scores(n: int, value: float = 0.8) -> np.ndarray:
    m = np.full((n, n), value, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _random_scores(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m   = rng.random((n, n)).astype(np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _simple_cost_matrix(n: int = 4) -> CostMatrix:
    m = _random_scores(n)
    return build_from_scores(m)


# ─── CostMatrix ───────────────────────────────────────────────────────────────

class TestCostMatrixDataclass:
    def test_fields_accessible(self):
        m  = np.zeros((3, 3), dtype=np.float32)
        cm = CostMatrix(matrix=m, n_fragments=3, method="test")
        assert cm.n_fragments == 3
        assert cm.method == "test"
        assert cm.params == {}

    def test_wrong_shape_raises(self):
        m = np.zeros((4, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            CostMatrix(matrix=m, n_fragments=4, method="test")

    def test_wrong_n_raises(self):
        m = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            CostMatrix(matrix=m, n_fragments=5, method="test")

    def test_repr_contains_n(self):
        cm = _simple_cost_matrix(4)
        assert "4" in repr(cm)

    def test_repr_contains_method(self):
        cm = _simple_cost_matrix(3)
        assert "from_scores" in repr(cm)


# ─── build_from_scores ────────────────────────────────────────────────────────

class TestBuildFromScores:
    def test_returns_cost_matrix(self):
        cm = build_from_scores(_random_scores(4))
        assert isinstance(cm, CostMatrix)

    def test_n_fragments_correct(self):
        cm = build_from_scores(_random_scores(5))
        assert cm.n_fragments == 5

    def test_diagonal_zero(self):
        cm = build_from_scores(_random_scores(4))
        np.testing.assert_array_almost_equal(np.diag(cm.matrix), 0.0)

    def test_values_in_zero_one(self):
        cm = build_from_scores(_random_scores(6))
        assert cm.matrix.min() >= -1e-6
        assert cm.matrix.max() <= 1.0 + 1e-6

    def test_invert_true_high_score_low_cost(self):
        n   = 4
        m   = np.zeros((n, n), dtype=np.float32)
        m[0, 1] = m[1, 0] = 1.0   # высокая оценка
        m[0, 2] = m[2, 0] = 0.0   # низкая оценка
        cm  = build_from_scores(m, invert=True)
        # Высокая оценка → низкая стоимость
        assert cm.matrix[0, 1] < cm.matrix[0, 2]

    def test_invert_false(self):
        m   = _random_scores(4)
        cm  = build_from_scores(m, invert=False)
        np.testing.assert_array_almost_equal(np.diag(cm.matrix), 0.0)
        assert cm.matrix.max() <= 1.0 + 1e-6

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_from_scores(np.ones((3, 4), dtype=np.float32))

    def test_uniform_scores_all_equal(self):
        m  = _uniform_scores(4, value=0.5)
        cm = build_from_scores(m)
        # Все off-diagonal одинаковы → все равны 0 или 0.5
        off = cm.matrix[~np.eye(4, dtype=bool)]
        assert np.all(np.abs(off - off[0]) < 1e-5)

    def test_method_name(self):
        cm = build_from_scores(_random_scores(3))
        assert cm.method == "from_scores"

    def test_dtype_float32(self):
        cm = build_from_scores(_random_scores(3))
        assert cm.matrix.dtype == np.float32


# ─── build_from_distances ─────────────────────────────────────────────────────

class TestBuildFromDistances:
    def test_returns_cost_matrix(self):
        d  = np.abs(_random_scores(4))
        cm = build_from_distances(d)
        assert isinstance(cm, CostMatrix)

    def test_diagonal_zero(self):
        d  = np.abs(_random_scores(5))
        cm = build_from_distances(d)
        np.testing.assert_array_almost_equal(np.diag(cm.matrix), 0.0)

    def test_normalize_true_max_one(self):
        d  = np.abs(_random_scores(5))
        cm = build_from_distances(d, normalize=True)
        assert cm.matrix.max() <= 1.0 + 1e-6

    def test_normalize_false_values_preserved(self):
        d  = np.array([[0., 10.], [10., 0.]], dtype=np.float32)
        cm = build_from_distances(d, normalize=False)
        assert cm.matrix[0, 1] == pytest.approx(10.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            build_from_distances(np.ones((3, 5), dtype=np.float32))

    def test_method_name(self):
        cm = build_from_distances(_random_scores(3))
        assert cm.method == "from_distances"

    def test_negative_values_made_positive(self):
        d  = np.full((3, 3), -5.0, dtype=np.float32)
        np.fill_diagonal(d, 0.0)
        cm = build_from_distances(d, normalize=False)
        assert cm.matrix.min() >= -1e-6


# ─── build_combined ───────────────────────────────────────────────────────────

class TestBuildCombined:
    def test_returns_cost_matrix(self):
        cm1 = build_from_scores(_random_scores(4))
        cm2 = build_from_scores(_random_scores(4, seed=7))
        result = build_combined([cm1, cm2])
        assert isinstance(result, CostMatrix)

    def test_n_fragments_correct(self):
        cm1 = build_from_scores(_random_scores(4))
        cm2 = build_from_scores(_random_scores(4, seed=7))
        result = build_combined([cm1, cm2])
        assert result.n_fragments == 4

    def test_equal_weights_is_mean(self):
        n   = 4
        cm1 = build_from_scores(_random_scores(n, seed=1))
        cm2 = build_from_scores(_random_scores(n, seed=2))
        combined = build_combined([cm1, cm2])
        expected = (cm1.matrix + cm2.matrix) / 2.0
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_almost_equal(combined.matrix, expected, decimal=5)

    def test_custom_weights_normalized(self):
        cm1 = build_from_scores(_random_scores(3, seed=1))
        cm2 = build_from_scores(_random_scores(3, seed=2))
        # weights=[2, 1] → [2/3, 1/3]
        result = build_combined([cm1, cm2], weights=[2.0, 1.0])
        expected = cm1.matrix * (2.0 / 3) + cm2.matrix * (1.0 / 3)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_almost_equal(result.matrix, expected, decimal=5)

    def test_single_matrix(self):
        cm     = build_from_scores(_random_scores(4))
        result = build_combined([cm])
        np.testing.assert_array_almost_equal(result.matrix, cm.matrix, decimal=5)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_combined([])

    def test_size_mismatch_raises(self):
        cm1 = build_from_scores(_random_scores(3))
        cm2 = build_from_scores(_random_scores(5))
        with pytest.raises(ValueError):
            build_combined([cm1, cm2])

    def test_zero_weight_sum_raises(self):
        cm1 = build_from_scores(_random_scores(3))
        with pytest.raises(ValueError):
            build_combined([cm1], weights=[0.0])

    def test_diagonal_zero(self):
        cm1 = build_from_scores(_random_scores(4))
        cm2 = build_from_scores(_random_scores(4, seed=7))
        result = build_combined([cm1, cm2])
        np.testing.assert_array_almost_equal(np.diag(result.matrix), 0.0)

    def test_method_name(self):
        cm = build_from_scores(_random_scores(3))
        result = build_combined([cm])
        assert result.method == "combined"


# ─── apply_forbidden_mask ─────────────────────────────────────────────────────

class TestApplyForbiddenMask:
    def test_forbidden_pairs_set_to_fill(self):
        cm   = _simple_cost_matrix(4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = mask[1, 0] = True
        result = apply_forbidden_mask(cm, mask, fill_value=2.0)
        assert result.matrix[0, 1] == pytest.approx(2.0)
        assert result.matrix[1, 0] == pytest.approx(2.0)

    def test_diagonal_still_zero(self):
        cm   = _simple_cost_matrix(4)
        mask = np.ones((4, 4), dtype=bool)
        result = apply_forbidden_mask(cm, mask, fill_value=9.9)
        np.testing.assert_array_almost_equal(np.diag(result.matrix), 0.0)

    def test_non_forbidden_unchanged(self):
        cm      = _simple_cost_matrix(4)
        mask    = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = True
        result  = apply_forbidden_mask(cm, mask, fill_value=1.0)
        # Другие пары не изменились
        assert result.matrix[0, 2] == pytest.approx(cm.matrix[0, 2])

    def test_wrong_shape_raises(self):
        cm   = _simple_cost_matrix(4)
        mask = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError):
            apply_forbidden_mask(cm, mask)

    def test_method_name(self):
        cm   = _simple_cost_matrix(4)
        mask = np.zeros((4, 4), dtype=bool)
        result = apply_forbidden_mask(cm, mask)
        assert result.method == "masked"

    def test_params_n_forbidden(self):
        cm   = _simple_cost_matrix(4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 1] = mask[1, 0] = True
        result = apply_forbidden_mask(cm, mask)
        assert result.params["n_forbidden"] == 2


# ─── normalize_costs ──────────────────────────────────────────────────────────

class TestNormalizeCosts:
    @pytest.mark.parametrize("method", ["minmax", "zscore", "rank"])
    def test_returns_cost_matrix(self, method):
        cm     = _simple_cost_matrix(4)
        result = normalize_costs(cm, method=method)
        assert isinstance(result, CostMatrix)

    @pytest.mark.parametrize("method", ["minmax", "zscore", "rank"])
    def test_diagonal_zero(self, method):
        cm     = _simple_cost_matrix(5)
        result = normalize_costs(cm, method=method)
        np.testing.assert_array_almost_equal(np.diag(result.matrix), 0.0, decimal=5)

    @pytest.mark.parametrize("method", ["minmax", "zscore", "rank"])
    def test_values_in_zero_one(self, method):
        cm     = _simple_cost_matrix(5)
        result = normalize_costs(cm, method=method)
        assert result.matrix.min() >= -1e-5
        assert result.matrix.max() <= 1.0 + 1e-5

    def test_minmax_max_is_one(self):
        cm     = _simple_cost_matrix(4)
        result = normalize_costs(cm, "minmax")
        off    = result.matrix[~np.eye(4, dtype=bool)]
        assert off.max() == pytest.approx(1.0, abs=1e-5)

    def test_rank_method_order_preserved(self):
        n   = 4
        m   = np.array([[0., 0.1, 0.5, 0.9],
                         [0.1, 0., 0.3, 0.8],
                         [0.5, 0.3, 0., 0.2],
                         [0.9, 0.8, 0.2, 0.]], dtype=np.float32)
        cm     = CostMatrix(matrix=m, n_fragments=n, method="test")
        result = normalize_costs(cm, "rank")
        # Ранги сохраняют относительный порядок
        assert result.matrix[0, 1] < result.matrix[0, 3]

    def test_unknown_method_raises(self):
        cm = _simple_cost_matrix(3)
        with pytest.raises(ValueError):
            normalize_costs(cm, method="unknown")

    def test_method_name_contains_method(self):
        cm     = _simple_cost_matrix(3)
        result = normalize_costs(cm, "minmax")
        assert "minmax" in result.method


# ─── to_assignment_matrix ─────────────────────────────────────────────────────

class TestToAssignmentMatrix:
    def test_shape_preserved(self):
        cm = _simple_cost_matrix(4)
        m  = to_assignment_matrix(cm)
        assert m.shape == (4, 4)

    def test_diagonal_larger_than_max(self):
        cm  = _simple_cost_matrix(4)
        m   = to_assignment_matrix(cm)
        mx  = float(cm.matrix.max())
        for d in np.diag(m):
            assert float(d) > mx

    def test_off_diagonal_unchanged(self):
        cm = _simple_cost_matrix(4)
        m  = to_assignment_matrix(cm)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert m[i, j] == pytest.approx(cm.matrix[i, j])

    def test_dtype_float32(self):
        cm = _simple_cost_matrix(3)
        m  = to_assignment_matrix(cm)
        assert m.dtype == np.float32

    def test_returns_ndarray(self):
        cm = _simple_cost_matrix(3)
        m  = to_assignment_matrix(cm)
        assert isinstance(m, np.ndarray)


# ─── top_k_candidates ─────────────────────────────────────────────────────────

class TestTopKCandidates:
    def test_returns_dict(self):
        cm     = _simple_cost_matrix(4)
        result = top_k_candidates(cm, k=2)
        assert isinstance(result, dict)

    def test_all_fragments_present(self):
        n      = 5
        cm     = _simple_cost_matrix(n)
        result = top_k_candidates(cm, k=3)
        assert set(result.keys()) == set(range(n))

    def test_correct_length_per_fragment(self):
        n      = 5
        cm     = _simple_cost_matrix(n)
        result = top_k_candidates(cm, k=3)
        for candidates in result.values():
            assert len(candidates) == 3

    def test_self_not_in_candidates(self):
        n      = 4
        cm     = _simple_cost_matrix(n)
        result = top_k_candidates(cm, k=2)
        for i, candidates in result.items():
            for j, _ in candidates:
                assert j != i

    def test_sorted_by_cost_ascending(self):
        n      = 5
        cm     = _simple_cost_matrix(n)
        result = top_k_candidates(cm, k=3)
        for candidates in result.values():
            costs = [c for _, c in candidates]
            assert costs == sorted(costs)

    def test_k_larger_than_n_minus_1(self):
        n      = 3
        cm     = _simple_cost_matrix(n)
        result = top_k_candidates(cm, k=100)
        # Максимум n-1 кандидатов
        for candidates in result.values():
            assert len(candidates) <= n - 1

    def test_k_zero_empty_lists(self):
        n      = 3
        cm     = _simple_cost_matrix(n)
        result = top_k_candidates(cm, k=0)
        for candidates in result.values():
            assert candidates == []

    def test_known_order(self):
        m = np.array([[0., 1., 3., 2.],
                      [1., 0., 4., 2.],
                      [3., 4., 0., 1.],
                      [2., 2., 1., 0.]], dtype=np.float32)
        cm     = CostMatrix(matrix=m, n_fragments=4, method="test")
        result = top_k_candidates(cm, k=2)
        # Для фрагмента 0: кандидаты — 1 (стоимость 1) и 3 (стоимость 2)
        idxs_0 = [j for j, _ in result[0]]
        assert idxs_0[0] == 1
        assert idxs_0[1] == 3
