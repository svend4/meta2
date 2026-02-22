"""Тесты для puzzle_reconstruction.scoring.score_normalizer."""
import numpy as np
import pytest

from puzzle_reconstruction.scoring.score_normalizer import (
    NormMethod,
    NormalizedMatrix,
    batch_normalize_matrices,
    combine_score_matrices,
    minmax_normalize_matrix,
    normalize_score_matrix,
    rank_normalize_matrix,
    sigmoid_normalize_matrix,
    softmax_normalize_matrix,
    zscore_normalize_matrix,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mat(seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((4, 4))


def _mat_wide():
    return np.arange(12, dtype=np.float64).reshape(3, 4)


# ─── TestNormMethod ───────────────────────────────────────────────────────────

class TestNormMethod:
    def test_defaults(self):
        nm = NormMethod()
        assert nm.method == "minmax"
        assert nm.axis is None
        assert nm.temperature == 1.0
        assert nm.eps == 1e-10

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NormMethod(method="l2")

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            NormMethod(axis=2)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            NormMethod(temperature=0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            NormMethod(temperature=-1.0)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError):
            NormMethod(eps=0.0)

    def test_all_methods_valid(self):
        for m in ("minmax", "zscore", "rank", "softmax", "sigmoid"):
            nm = NormMethod(method=m)
            assert nm.method == m

    def test_axis_0_valid(self):
        NormMethod(axis=0)

    def test_axis_1_valid(self):
        NormMethod(axis=1)


# ─── TestNormalizedMatrix ─────────────────────────────────────────────────────

class TestNormalizedMatrix:
    def test_shape_prop(self):
        data = np.zeros((3, 5))
        nm = NormalizedMatrix(method="minmax", data=data, min_val=0.0, max_val=1.0)
        assert nm.shape == (3, 5)

    def test_method_stored(self):
        nm = NormalizedMatrix(method="zscore", data=np.eye(2), min_val=0.0, max_val=1.0)
        assert nm.method == "zscore"

    def test_min_max_stored(self):
        nm = NormalizedMatrix(method="minmax", data=np.zeros((2, 2)),
                              min_val=-5.0, max_val=10.0)
        assert nm.min_val == -5.0
        assert nm.max_val == 10.0


# ─── TestMinmaxNormalizeMatrix ────────────────────────────────────────────────

class TestMinmaxNormalizeMatrix:
    def test_output_range_0_to_1(self):
        r = minmax_normalize_matrix(_mat())
        assert float(r.data.min()) >= 0.0
        assert float(r.data.max()) <= 1.0 + 1e-12

    def test_min_is_0_max_is_1(self):
        r = minmax_normalize_matrix(_mat_wide())
        assert abs(float(r.data.min())) < 1e-9
        assert abs(float(r.data.max()) - 1.0) < 1e-9

    def test_shape_preserved(self):
        r = minmax_normalize_matrix(_mat())
        assert r.shape == (4, 4)

    def test_method_is_minmax(self):
        r = minmax_normalize_matrix(_mat())
        assert r.method == "minmax"

    def test_uniform_matrix_no_crash(self):
        m = np.ones((3, 3)) * 5.0
        r = minmax_normalize_matrix(m)
        assert r.data.shape == (3, 3)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            minmax_normalize_matrix(np.ones(5))

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError):
            minmax_normalize_matrix(_mat(), eps=-1.0)

    def test_min_max_recorded(self):
        m = np.array([[1.0, 3.0], [2.0, 4.0]])
        r = minmax_normalize_matrix(m)
        assert r.min_val == 1.0
        assert r.max_val == 4.0


# ─── TestZscoreNormalizeMatrix ────────────────────────────────────────────────

class TestZscoreNormalizeMatrix:
    def test_mean_approx_zero(self):
        r = zscore_normalize_matrix(_mat())
        assert abs(float(r.data.mean())) < 1e-9

    def test_std_approx_one(self):
        r = zscore_normalize_matrix(_mat())
        assert abs(float(r.data.std()) - 1.0) < 1e-9

    def test_shape_preserved(self):
        assert zscore_normalize_matrix(_mat()).shape == (4, 4)

    def test_method_is_zscore(self):
        assert zscore_normalize_matrix(_mat()).method == "zscore"

    def test_uniform_matrix_no_crash(self):
        m = np.full((3, 3), 7.0)
        r = zscore_normalize_matrix(m)
        assert r.data.shape == (3, 3)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            zscore_normalize_matrix(np.ones((2, 2, 2)))

    def test_min_val_stores_mean(self):
        m = np.array([[0.0, 2.0], [2.0, 4.0]])
        r = zscore_normalize_matrix(m)
        assert abs(r.min_val - float(m.mean())) < 1e-9


# ─── TestRankNormalizeMatrix ──────────────────────────────────────────────────

class TestRankNormalizeMatrix:
    def test_output_range_0_to_1(self):
        r = rank_normalize_matrix(_mat())
        assert float(r.data.min()) >= 0.0
        assert float(r.data.max()) <= 1.0 + 1e-12

    def test_min_rank_is_0_max_is_1(self):
        r = rank_normalize_matrix(_mat_wide())
        assert abs(float(r.data.min())) < 1e-9
        assert abs(float(r.data.max()) - 1.0) < 1e-9

    def test_shape_preserved(self):
        assert rank_normalize_matrix(_mat()).shape == (4, 4)

    def test_method_is_rank(self):
        assert rank_normalize_matrix(_mat()).method == "rank"

    def test_monotone_with_values(self):
        m = np.array([[3.0, 1.0], [4.0, 2.0]])
        r = rank_normalize_matrix(m)
        flat_m = m.ravel()
        flat_r = r.data.ravel()
        order = np.argsort(flat_m)
        assert (flat_r[order] == sorted(flat_r)).all()

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            rank_normalize_matrix(np.ones(5))


# ─── TestSoftmaxNormalizeMatrix ───────────────────────────────────────────────

class TestSoftmaxNormalizeMatrix:
    def test_global_sum_to_1(self):
        r = softmax_normalize_matrix(_mat())
        assert abs(float(r.data.sum()) - 1.0) < 1e-9

    def test_axis1_rows_sum_to_1(self):
        r = softmax_normalize_matrix(_mat(), axis=1)
        row_sums = r.data.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)

    def test_axis0_cols_sum_to_1(self):
        r = softmax_normalize_matrix(_mat(), axis=0)
        col_sums = r.data.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-9)

    def test_all_positive(self):
        r = softmax_normalize_matrix(_mat())
        assert float(r.data.min()) > 0.0

    def test_method_is_softmax(self):
        assert softmax_normalize_matrix(_mat()).method == "softmax"

    def test_shape_preserved(self):
        assert softmax_normalize_matrix(_mat()).shape == (4, 4)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            softmax_normalize_matrix(_mat(), axis=2)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            softmax_normalize_matrix(_mat(), temperature=0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            softmax_normalize_matrix(np.ones(5))


# ─── TestSigmoidNormalizeMatrix ───────────────────────────────────────────────

class TestSigmoidNormalizeMatrix:
    def test_output_range_0_to_1(self):
        r = sigmoid_normalize_matrix(_mat())
        assert float(r.data.min()) > 0.0
        assert float(r.data.max()) < 1.0

    def test_shape_preserved(self):
        assert sigmoid_normalize_matrix(_mat()).shape == (4, 4)

    def test_method_is_sigmoid(self):
        assert sigmoid_normalize_matrix(_mat()).method == "sigmoid"

    def test_zero_maps_to_half(self):
        m = np.zeros((2, 2))
        r = sigmoid_normalize_matrix(m)
        assert np.allclose(r.data, 0.5, atol=1e-9)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            sigmoid_normalize_matrix(np.ones((2, 2, 2)))


# ─── TestNormalizeScoreMatrix ─────────────────────────────────────────────────

class TestNormalizeScoreMatrix:
    def test_default_cfg_uses_minmax(self):
        r = normalize_score_matrix(_mat())
        assert r.method == "minmax"

    def test_none_cfg_uses_minmax(self):
        r = normalize_score_matrix(_mat(), None)
        assert r.method == "minmax"

    def test_zscore_dispatch(self):
        cfg = NormMethod(method="zscore")
        r = normalize_score_matrix(_mat(), cfg)
        assert r.method == "zscore"

    def test_rank_dispatch(self):
        cfg = NormMethod(method="rank")
        r = normalize_score_matrix(_mat(), cfg)
        assert r.method == "rank"

    def test_softmax_dispatch(self):
        cfg = NormMethod(method="softmax")
        r = normalize_score_matrix(_mat(), cfg)
        assert r.method == "softmax"

    def test_sigmoid_dispatch(self):
        cfg = NormMethod(method="sigmoid")
        r = normalize_score_matrix(_mat(), cfg)
        assert r.method == "sigmoid"


# ─── TestCombineScoreMatrices ─────────────────────────────────────────────────

class TestCombineScoreMatrices:
    def test_equal_weights_is_mean(self):
        m1 = np.ones((3, 3)) * 2.0
        m2 = np.ones((3, 3)) * 4.0
        result = combine_score_matrices([m1, m2])
        assert np.allclose(result, 3.0, atol=1e-9)

    def test_custom_weights(self):
        m1 = np.ones((2, 2)) * 0.0
        m2 = np.ones((2, 2)) * 1.0
        result = combine_score_matrices([m1, m2], weights=[0.0, 1.0])
        assert np.allclose(result, 1.0, atol=1e-9)

    def test_single_matrix(self):
        m = _mat()
        result = combine_score_matrices([m])
        assert np.allclose(result, m, atol=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_score_matrices([])

    def test_shape_mismatch_raises(self):
        m1 = np.ones((2, 2))
        m2 = np.ones((3, 3))
        with pytest.raises(ValueError):
            combine_score_matrices([m1, m2])

    def test_wrong_number_of_weights_raises(self):
        m = _mat()
        with pytest.raises(ValueError):
            combine_score_matrices([m, m], weights=[1.0])

    def test_negative_weight_raises(self):
        m = _mat()
        with pytest.raises(ValueError):
            combine_score_matrices([m, m], weights=[1.0, -0.1])

    def test_zero_total_weight_raises(self):
        m = _mat()
        with pytest.raises(ValueError):
            combine_score_matrices([m], weights=[0.0])

    def test_output_shape(self):
        m1 = _mat()
        m2 = _mat(seed=1)
        result = combine_score_matrices([m1, m2])
        assert result.shape == (4, 4)


# ─── TestBatchNormalizeMatrices ───────────────────────────────────────────────

class TestBatchNormalizeMatrices:
    def test_empty_batch(self):
        assert batch_normalize_matrices([]) == []

    def test_single_matrix(self):
        result = batch_normalize_matrices([_mat()])
        assert len(result) == 1
        assert isinstance(result[0], NormalizedMatrix)

    def test_multiple_matrices(self):
        mats = [_mat(seed=i) for i in range(5)]
        result = batch_normalize_matrices(mats)
        assert len(result) == 5

    def test_all_methods_in_batch(self):
        mats = [_mat(seed=i) for i in range(3)]
        for method in ("minmax", "zscore", "rank"):
            cfg = NormMethod(method=method)
            result = batch_normalize_matrices(mats, cfg)
            assert all(r.method == method for r in result)

    def test_none_cfg_uses_defaults(self):
        result = batch_normalize_matrices([_mat()], None)
        assert result[0].method == "minmax"
