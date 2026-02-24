"""Extra tests for puzzle_reconstruction/scoring/score_normalizer.py"""
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

def _mat(seed=0, n=4):
    return np.random.default_rng(seed).random((n, n))


def _wide(r=3, c=5):
    return np.arange(float(r * c)).reshape(r, c)


# ─── TestNormMethodExtra ──────────────────────────────────────────────────────

class TestNormMethodExtra:
    def test_temperature_large(self):
        nm = NormMethod(temperature=100.0)
        assert nm.temperature == pytest.approx(100.0)

    def test_temperature_small_positive(self):
        nm = NormMethod(temperature=0.001)
        assert nm.temperature == pytest.approx(0.001)

    def test_eps_small_positive(self):
        nm = NormMethod(eps=1e-15)
        assert nm.eps == pytest.approx(1e-15)

    def test_axis_none_default(self):
        nm = NormMethod()
        assert nm.axis is None

    def test_all_methods_roundtrip(self):
        for m in ("minmax", "zscore", "rank", "softmax", "sigmoid"):
            assert NormMethod(method=m).method == m

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError):
            NormMethod(eps=-1e-10)


# ─── TestNormalizedMatrixExtra ────────────────────────────────────────────────

class TestNormalizedMatrixExtra:
    def test_data_preserved(self):
        data = np.arange(6.0).reshape(2, 3)
        nm = NormalizedMatrix(method="minmax", data=data, min_val=0.0, max_val=5.0)
        np.testing.assert_array_equal(nm.data, data)

    def test_shape_non_square(self):
        data = np.zeros((3, 7))
        nm = NormalizedMatrix(method="rank", data=data, min_val=0.0, max_val=1.0)
        assert nm.shape == (3, 7)

    def test_min_val_float(self):
        nm = NormalizedMatrix(method="minmax", data=np.eye(2),
                              min_val=3.14, max_val=9.99)
        assert isinstance(nm.min_val, float)

    def test_max_val_stored(self):
        nm = NormalizedMatrix(method="sigmoid", data=np.zeros((2, 2)),
                              min_val=0.0, max_val=42.0)
        assert nm.max_val == pytest.approx(42.0)


# ─── TestMinmaxNormalizeMatrixExtra ───────────────────────────────────────────

class TestMinmaxNormalizeMatrixExtra:
    def test_wide_matrix_shape(self):
        r = minmax_normalize_matrix(_wide(3, 7))
        assert r.shape == (3, 7)

    def test_single_column_matrix(self):
        m = np.array([[1.0], [2.0], [3.0]])
        r = minmax_normalize_matrix(m)
        assert r.shape == (3, 1)
        assert float(r.data.min()) >= 0.0
        assert float(r.data.max()) <= 1.0 + 1e-12

    def test_seed_1_range(self):
        r = minmax_normalize_matrix(_mat(seed=1))
        assert float(r.data.min()) >= 0.0
        assert float(r.data.max()) <= 1.0 + 1e-9

    def test_two_identical_values(self):
        m = np.array([[5.0, 5.0], [5.0, 5.0]])
        r = minmax_normalize_matrix(m)
        assert r.data.shape == (2, 2)

    def test_large_matrix(self):
        m = np.random.default_rng(7).random((20, 20))
        r = minmax_normalize_matrix(m)
        assert r.shape == (20, 20)
        assert abs(float(r.data.min())) < 1e-9
        assert abs(float(r.data.max()) - 1.0) < 1e-9


# ─── TestZscoreNormalizeMatrixExtra ───────────────────────────────────────────

class TestZscoreNormalizeMatrixExtra:
    def test_wide_matrix(self):
        r = zscore_normalize_matrix(_wide(2, 6))
        assert r.shape == (2, 6)

    def test_mean_near_zero_large_matrix(self):
        m = np.random.default_rng(3).random((10, 10))
        r = zscore_normalize_matrix(m)
        assert abs(float(r.data.mean())) < 1e-9

    def test_std_near_one_large_matrix(self):
        m = np.random.default_rng(3).random((10, 10))
        r = zscore_normalize_matrix(m)
        assert abs(float(r.data.std()) - 1.0) < 1e-6

    def test_max_val_stores_std(self):
        m = np.array([[0.0, 1.0], [2.0, 3.0]])
        r = zscore_normalize_matrix(m)
        expected_std = float(m.std())
        assert abs(r.max_val - expected_std) < 1e-9

    def test_method_tag(self):
        assert zscore_normalize_matrix(_mat(seed=5)).method == "zscore"


# ─── TestRankNormalizeMatrixExtra ─────────────────────────────────────────────

class TestRankNormalizeMatrixExtra:
    def test_non_square_shape(self):
        r = rank_normalize_matrix(_wide(2, 5))
        assert r.shape == (2, 5)

    def test_all_same_values(self):
        m = np.full((3, 3), 7.0)
        r = rank_normalize_matrix(m)
        assert r.shape == (3, 3)
        assert float(r.data.max()) <= 1.0 + 1e-9

    def test_large_matrix_range(self):
        m = np.random.default_rng(9).random((8, 8))
        r = rank_normalize_matrix(m)
        assert float(r.data.min()) >= 0.0
        assert float(r.data.max()) <= 1.0 + 1e-12

    def test_monotone_preserving(self):
        m = np.array([[1.0, 4.0], [2.0, 3.0]])
        r = rank_normalize_matrix(m)
        flat_m = m.ravel()
        flat_r = r.data.ravel()
        for i in range(len(flat_m)):
            for j in range(len(flat_m)):
                if flat_m[i] < flat_m[j]:
                    assert flat_r[i] <= flat_r[j]


# ─── TestSoftmaxNormalizeMatrixExtra ─────────────────────────────────────────

class TestSoftmaxNormalizeMatrixExtra:
    def test_global_sum_large_matrix(self):
        r = softmax_normalize_matrix(_mat(seed=3, n=6))
        assert abs(float(r.data.sum()) - 1.0) < 1e-9

    def test_all_positive_nonzero(self):
        r = softmax_normalize_matrix(_mat(seed=2))
        assert float(r.data.min()) > 0.0

    def test_axis0_col_sums_large(self):
        r = softmax_normalize_matrix(_mat(seed=4, n=6), axis=0)
        col_sums = r.data.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-9)

    def test_axis1_row_sums_large(self):
        r = softmax_normalize_matrix(_mat(seed=5, n=6), axis=1)
        row_sums = r.data.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)

    def test_temperature_half_sharpens(self):
        m = _mat(seed=0)
        r_default = softmax_normalize_matrix(m)
        r_sharp = softmax_normalize_matrix(m, temperature=0.5)
        # Lower temperature should make distribution more peaked
        assert float(r_sharp.data.max()) >= float(r_default.data.max())

    def test_temperature_large_flattens(self):
        m = _mat(seed=0)
        r_flat = softmax_normalize_matrix(m, temperature=100.0)
        # Very high temperature → nearly uniform
        assert float(r_flat.data.max()) < 1.0 / m.size + 0.1


# ─── TestSigmoidNormalizeMatrixExtra ─────────────────────────────────────────

class TestSigmoidNormalizeMatrixExtra:
    def test_large_positive_near_one(self):
        m = np.full((2, 2), 100.0)
        r = sigmoid_normalize_matrix(m)
        assert float(r.data.min()) > 0.99

    def test_large_negative_near_zero(self):
        m = np.full((2, 2), -100.0)
        r = sigmoid_normalize_matrix(m)
        assert float(r.data.max()) < 0.01

    def test_non_square_shape(self):
        m = np.random.default_rng(1).random((3, 5)) - 0.5
        r = sigmoid_normalize_matrix(m)
        assert r.shape == (3, 5)

    def test_negative_values_lt_half(self):
        m = np.full((2, 2), -1.0)
        r = sigmoid_normalize_matrix(m)
        assert float(r.data.max()) < 0.5

    def test_positive_values_gt_half(self):
        m = np.full((2, 2), 1.0)
        r = sigmoid_normalize_matrix(m)
        assert float(r.data.min()) > 0.5


# ─── TestNormalizeScoreMatrixExtra ────────────────────────────────────────────

class TestNormalizeScoreMatrixExtra:
    def test_minmax_default_range(self):
        r = normalize_score_matrix(_mat())
        assert float(r.data.min()) >= 0.0
        assert float(r.data.max()) <= 1.0 + 1e-12

    def test_zscore_mean_zero(self):
        from puzzle_reconstruction.scoring.score_normalizer import NormMethod
        cfg = NormMethod(method="zscore")
        r = normalize_score_matrix(_mat(), cfg)
        assert abs(float(r.data.mean())) < 1e-9

    def test_softmax_sums_to_one(self):
        cfg = NormMethod(method="softmax")
        r = normalize_score_matrix(_mat(), cfg)
        assert abs(float(r.data.sum()) - 1.0) < 1e-9

    def test_sigmoid_in_zero_one(self):
        cfg = NormMethod(method="sigmoid")
        r = normalize_score_matrix(_mat(), cfg)
        assert float(r.data.min()) > 0.0
        assert float(r.data.max()) < 1.0

    def test_wide_matrix_all_methods(self):
        m = _wide(3, 5)
        for method in ("minmax", "zscore", "rank", "softmax", "sigmoid"):
            cfg = NormMethod(method=method)
            r = normalize_score_matrix(m, cfg)
            assert r.shape == (3, 5)


# ─── TestCombineScoreMatricesExtra ────────────────────────────────────────────

class TestCombineScoreMatricesExtra:
    def test_three_equal_weight_matrices(self):
        m1 = np.ones((2, 2)) * 1.0
        m2 = np.ones((2, 2)) * 2.0
        m3 = np.ones((2, 2)) * 3.0
        result = combine_score_matrices([m1, m2, m3])
        assert np.allclose(result, 2.0, atol=1e-9)

    def test_large_matrix_combine(self):
        mats = [np.random.default_rng(i).random((6, 6)) for i in range(4)]
        result = combine_score_matrices(mats)
        assert result.shape == (6, 6)

    def test_weights_sum_not_one_ok(self):
        m1 = np.ones((2, 2)) * 2.0
        m2 = np.ones((2, 2)) * 4.0
        result = combine_score_matrices([m1, m2], weights=[2.0, 2.0])
        assert np.allclose(result, 3.0, atol=1e-9)

    def test_non_square_matrices(self):
        m1 = np.ones((3, 5))
        m2 = np.ones((3, 5)) * 3.0
        result = combine_score_matrices([m1, m2])
        assert result.shape == (3, 5)

    def test_asymmetric_weights(self):
        m1 = np.ones((2, 2)) * 0.0
        m2 = np.ones((2, 2)) * 10.0
        result = combine_score_matrices([m1, m2], weights=[1.0, 9.0])
        assert np.allclose(result, 9.0, atol=1e-9)


# ─── TestBatchNormalizeMatricesExtra ──────────────────────────────────────────

class TestBatchNormalizeMatricesExtra:
    def test_ten_matrices(self):
        mats = [_mat(seed=i) for i in range(10)]
        result = batch_normalize_matrices(mats)
        assert len(result) == 10

    def test_all_normalized_in_range(self):
        mats = [_mat(seed=i) for i in range(5)]
        for nm in batch_normalize_matrices(mats):
            assert float(nm.data.min()) >= 0.0
            assert float(nm.data.max()) <= 1.0 + 1e-12

    def test_zscore_all_have_zero_mean(self):
        from puzzle_reconstruction.scoring.score_normalizer import NormMethod
        cfg = NormMethod(method="zscore")
        mats = [_mat(seed=i) for i in range(4)]
        for nm in batch_normalize_matrices(mats, cfg):
            assert abs(float(nm.data.mean())) < 1e-9

    def test_wide_matrices(self):
        mats = [_wide(2, 5) * (i + 1) for i in range(3)]
        result = batch_normalize_matrices(mats)
        assert all(r.shape == (2, 5) for r in result)

    def test_softmax_each_sums_to_one(self):
        from puzzle_reconstruction.scoring.score_normalizer import NormMethod
        cfg = NormMethod(method="softmax")
        mats = [_mat(seed=i) for i in range(3)]
        for nm in batch_normalize_matrices(mats, cfg):
            assert abs(float(nm.data.sum()) - 1.0) < 1e-9
