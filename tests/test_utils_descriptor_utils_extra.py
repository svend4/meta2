"""Extra tests for puzzle_reconstruction/utils/descriptor_utils.py"""
import pytest
import numpy as np

from puzzle_reconstruction.utils.descriptor_utils import (
    DescriptorConfig,
    l2_normalize,
    l1_normalize,
    batch_l2_normalize,
    l2_distance,
    cosine_distance,
    chi2_distance,
    l1_distance,
    descriptor_distance,
    pairwise_l2,
    pairwise_cosine,
    DescriptorMatch,
    nn_match,
    ratio_test,
    mean_pool,
    max_pool,
    batch_nn_match,
    top_k_matches,
    filter_matches_by_distance,
)

RNG = np.random.default_rng(0)
MAT4x8 = RNG.random((4, 8))


# ─── DescriptorConfig ─────────────────────────────────────────────────────────

class TestDescriptorConfig:
    def test_defaults(self):
        cfg = DescriptorConfig()
        assert cfg.metric == "l2"
        assert cfg.normalize is True
        assert cfg.eps == pytest.approx(1e-8)

    def test_custom(self):
        cfg = DescriptorConfig(metric="cosine", normalize=False, eps=1e-6)
        assert cfg.metric == "cosine"
        assert cfg.normalize is False
        assert cfg.eps == pytest.approx(1e-6)

    def test_l1_metric(self):
        cfg = DescriptorConfig(metric="l1")
        assert cfg.metric == "l1"

    def test_chi2_metric(self):
        cfg = DescriptorConfig(metric="chi2")
        assert cfg.metric == "chi2"


# ─── l2_normalize ─────────────────────────────────────────────────────────────

class TestL2Normalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        result = l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-9)

    def test_norm_is_one(self):
        v = np.array([3.0, 4.0])
        result = l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-9

    def test_zero_vector_returned_as_copy(self):
        v = np.array([0.0, 0.0, 0.0])
        result = l2_normalize(v)
        np.testing.assert_allclose(result, v)

    def test_returns_ndarray(self):
        v = np.array([1.0, 2.0])
        assert isinstance(l2_normalize(v), np.ndarray)

    def test_direction_preserved(self):
        v = np.array([3.0, 4.0])
        result = l2_normalize(v)
        assert abs(result[0] - 0.6) < 1e-9
        assert abs(result[1] - 0.8) < 1e-9


# ─── l1_normalize ─────────────────────────────────────────────────────────────

class TestL1Normalize:
    def test_sum_is_one(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        result = l1_normalize(v)
        assert abs(np.abs(result).sum() - 1.0) < 1e-9

    def test_zero_vector_safe(self):
        v = np.array([0.0, 0.0])
        result = l1_normalize(v)
        np.testing.assert_allclose(result, v)

    def test_returns_ndarray(self):
        v = np.array([2.0, 2.0])
        assert isinstance(l1_normalize(v), np.ndarray)

    def test_values_correct(self):
        v = np.array([1.0, 3.0])
        result = l1_normalize(v)
        assert abs(result[0] - 0.25) < 1e-9
        assert abs(result[1] - 0.75) < 1e-9

    def test_single_element(self):
        v = np.array([5.0])
        result = l1_normalize(v)
        assert abs(result[0] - 1.0) < 1e-9


# ─── batch_l2_normalize ───────────────────────────────────────────────────────

class TestBatchL2Normalize:
    def test_shape_preserved(self):
        result = batch_l2_normalize(MAT4x8.copy())
        assert result.shape == (4, 8)

    def test_rows_unit_norm(self):
        result = batch_l2_normalize(MAT4x8.copy())
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-9)

    def test_zero_rows_safe(self):
        mat = np.zeros((3, 4))
        result = batch_l2_normalize(mat)
        assert result.shape == (3, 4)

    def test_returns_ndarray(self):
        result = batch_l2_normalize(MAT4x8.copy())
        assert isinstance(result, np.ndarray)

    def test_single_row(self):
        mat = np.array([[3.0, 4.0]])
        result = batch_l2_normalize(mat)
        assert abs(np.linalg.norm(result[0]) - 1.0) < 1e-9


# ─── Distance metrics ─────────────────────────────────────────────────────────

class TestL2Distance:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert l2_distance(v, v) == pytest.approx(0.0)

    def test_orthogonal_unit_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert l2_distance(a, b) == pytest.approx(np.sqrt(2.0))

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert l2_distance(a, b) == pytest.approx(5.0)

    def test_returns_float(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert isinstance(l2_distance(a, b), float)


class TestCosineDistance:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(0.5, abs=1e-9)

    def test_zero_vector_returns_one(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(1.0)

    def test_result_in_zero_one(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        d = cosine_distance(a, b)
        assert 0.0 <= d <= 1.0

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert isinstance(cosine_distance(a, b), float)


class TestChi2Distance:
    def test_identical_histograms(self):
        a = np.array([0.25, 0.25, 0.25, 0.25])
        assert chi2_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_nonneg(self):
        a = np.array([0.1, 0.4, 0.3, 0.2])
        b = np.array([0.2, 0.3, 0.3, 0.2])
        assert chi2_distance(a, b) >= 0.0

    def test_returns_float(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert isinstance(chi2_distance(a, b), float)


class TestL1Distance:
    def test_identical(self):
        v = np.array([1.0, 2.0])
        assert l1_distance(v, v) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert l1_distance(a, b) == pytest.approx(2.0)

    def test_returns_float(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert isinstance(l1_distance(a, b), float)


# ─── descriptor_distance ──────────────────────────────────────────────────────

class TestDescriptorDistance:
    def test_l2_metric(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert descriptor_distance(a, b, metric="l2") == pytest.approx(5.0)

    def test_cosine_metric(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert descriptor_distance(a, b, metric="cosine") == pytest.approx(0.0, abs=1e-9)

    def test_l1_metric(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert descriptor_distance(a, b, metric="l1") == pytest.approx(2.0)

    def test_chi2_metric(self):
        a = np.array([0.5, 0.5])
        d = descriptor_distance(a, a, metric="chi2")
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_unknown_metric_raises(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        with pytest.raises(ValueError):
            descriptor_distance(a, b, metric="unknown")

    def test_returns_float(self):
        a = np.array([1.0])
        b = np.array([0.0])
        result = descriptor_distance(a, b, metric="l2")
        assert isinstance(result, float)


# ─── pairwise_l2 ──────────────────────────────────────────────────────────────

class TestPairwiseL2:
    def test_shape(self):
        a = MAT4x8.copy()
        b = np.random.default_rng(1).random((3, 8))
        result = pairwise_l2(a, b)
        assert result.shape == (4, 3)

    def test_self_distance_zero(self):
        a = MAT4x8.copy()
        result = pairwise_l2(a, a)
        diag = np.diag(result)
        np.testing.assert_allclose(diag, np.zeros(4), atol=1e-9)

    def test_nonneg(self):
        result = pairwise_l2(MAT4x8, MAT4x8)
        assert (result >= 0).all()

    def test_returns_ndarray(self):
        result = pairwise_l2(MAT4x8, MAT4x8)
        assert isinstance(result, np.ndarray)

    def test_square_on_same_input(self):
        result = pairwise_l2(MAT4x8, MAT4x8)
        assert result.shape == (4, 4)


# ─── pairwise_cosine ──────────────────────────────────────────────────────────

class TestPairwiseCosine:
    def test_shape(self):
        a = MAT4x8.copy()
        b = np.random.default_rng(2).random((5, 8))
        result = pairwise_cosine(a, b)
        assert result.shape == (4, 5)

    def test_values_in_zero_one(self):
        result = pairwise_cosine(MAT4x8, MAT4x8)
        assert (result >= -1e-9).all()
        assert (result <= 1.0 + 1e-9).all()

    def test_self_cosine_distance_near_zero(self):
        result = pairwise_cosine(MAT4x8, MAT4x8)
        diag = np.diag(result)
        np.testing.assert_allclose(diag, np.zeros(4), atol=1e-9)

    def test_returns_ndarray(self):
        result = pairwise_cosine(MAT4x8, MAT4x8)
        assert isinstance(result, np.ndarray)


# ─── DescriptorMatch ──────────────────────────────────────────────────────────

class TestDescriptorMatch:
    def test_fields(self):
        m = DescriptorMatch(query_idx=0, train_idx=1, distance=0.5)
        assert m.query_idx == 0
        assert m.train_idx == 1
        assert m.distance == pytest.approx(0.5)

    def test_is_dataclass(self):
        from dataclasses import fields
        assert len(fields(DescriptorMatch)) >= 3


# ─── nn_match ─────────────────────────────────────────────────────────────────

class TestNnMatch:
    def test_returns_list(self):
        q = MAT4x8.copy()
        t = np.random.default_rng(3).random((6, 8))
        result = nn_match(q, t)
        assert isinstance(result, list)

    def test_length_equals_query_count(self):
        q = MAT4x8.copy()
        t = np.random.default_rng(4).random((6, 8))
        result = nn_match(q, t)
        assert len(result) == 4

    def test_empty_query_returns_empty(self):
        q = np.zeros((0, 8))
        t = MAT4x8.copy()
        result = nn_match(q, t)
        assert result == []

    def test_empty_train_returns_empty(self):
        q = MAT4x8.copy()
        t = np.zeros((0, 8))
        result = nn_match(q, t)
        assert result == []

    def test_match_type(self):
        q = MAT4x8[:2].copy()
        t = MAT4x8[2:].copy()
        result = nn_match(q, t)
        for m in result:
            assert isinstance(m, DescriptorMatch)

    def test_train_idx_in_range(self):
        q = MAT4x8.copy()
        t = np.random.default_rng(5).random((6, 8))
        result = nn_match(q, t)
        for m in result:
            assert 0 <= m.train_idx < 6


# ─── ratio_test ───────────────────────────────────────────────────────────────

class TestRatioTest:
    def test_returns_list(self):
        q = MAT4x8.copy()
        t = np.random.default_rng(6).random((10, 8))
        result = ratio_test(q, t)
        assert isinstance(result, list)

    def test_empty_query_returns_empty(self):
        q = np.zeros((0, 8))
        t = MAT4x8.copy()
        result = ratio_test(q, t)
        assert result == []

    def test_too_few_train_returns_empty(self):
        q = MAT4x8.copy()
        t = MAT4x8[:1].copy()
        result = ratio_test(q, t)
        assert result == []

    def test_invalid_ratio_raises(self):
        q = MAT4x8.copy()
        t = np.random.default_rng(7).random((5, 8))
        with pytest.raises(ValueError):
            ratio_test(q, t, ratio=1.5)

    def test_match_type(self):
        q = MAT4x8.copy()
        t = np.random.default_rng(8).random((8, 8))
        result = ratio_test(q, t, ratio=0.9)
        for m in result:
            assert isinstance(m, DescriptorMatch)


# ─── mean_pool / max_pool ─────────────────────────────────────────────────────

class TestMeanPool:
    def test_shape(self):
        result = mean_pool(MAT4x8)
        assert result.shape == (8,)

    def test_returns_ndarray(self):
        assert isinstance(mean_pool(MAT4x8), np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_pool(np.zeros((0, 8)))

    def test_single_row(self):
        mat = np.array([[1.0, 2.0, 3.0]])
        result = mean_pool(mat)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_values(self):
        mat = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = mean_pool(mat)
        np.testing.assert_allclose(result, [1.0, 2.0])


class TestMaxPool:
    def test_shape(self):
        result = max_pool(MAT4x8)
        assert result.shape == (8,)

    def test_returns_ndarray(self):
        assert isinstance(max_pool(MAT4x8), np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            max_pool(np.zeros((0, 8)))

    def test_values(self):
        mat = np.array([[1.0, 3.0], [2.0, 0.0]])
        result = max_pool(mat)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_single_row(self):
        mat = np.array([[5.0, 7.0]])
        result = max_pool(mat)
        np.testing.assert_allclose(result, [5.0, 7.0])


# ─── batch_nn_match ───────────────────────────────────────────────────────────

class TestBatchNnMatch:
    def test_returns_list_of_lists(self):
        q1 = MAT4x8[:2].copy()
        q2 = MAT4x8[2:].copy()
        t = np.random.default_rng(9).random((5, 8))
        result = batch_nn_match([q1, q2], t)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_empty_queries_list(self):
        t = MAT4x8.copy()
        result = batch_nn_match([], t)
        assert result == []

    def test_each_inner_length(self):
        queries = [MAT4x8[:i+1].copy() for i in range(3)]
        t = np.random.default_rng(10).random((6, 8))
        result = batch_nn_match(queries, t)
        for i, inner in enumerate(result):
            assert len(inner) == i + 1


# ─── top_k_matches ────────────────────────────────────────────────────────────

class TestTopKMatches:
    def _make_matches(self, distances):
        return [DescriptorMatch(i, i, d) for i, d in enumerate(distances)]

    def test_top_1(self):
        matches = self._make_matches([0.5, 0.1, 0.8])
        result = top_k_matches(matches, 1)
        assert len(result) == 1
        assert result[0].distance == pytest.approx(0.1)

    def test_sorted_ascending(self):
        matches = self._make_matches([0.9, 0.3, 0.6])
        result = top_k_matches(matches, 3)
        dists = [m.distance for m in result]
        assert dists == sorted(dists)

    def test_k_larger_than_list(self):
        matches = self._make_matches([0.5, 0.2])
        result = top_k_matches(matches, 10)
        assert len(result) == 2

    def test_empty_input(self):
        result = top_k_matches([], 5)
        assert result == []

    def test_returns_list(self):
        matches = self._make_matches([0.1, 0.2])
        result = top_k_matches(matches, 1)
        assert isinstance(result, list)


# ─── filter_matches_by_distance ───────────────────────────────────────────────

class TestFilterMatchesByDistance:
    def _make_matches(self, distances):
        return [DescriptorMatch(i, i, d) for i, d in enumerate(distances)]

    def test_filters_out_high_distance(self):
        matches = self._make_matches([0.1, 0.5, 0.9])
        result = filter_matches_by_distance(matches, 0.5)
        assert len(result) == 2
        assert all(m.distance <= 0.5 for m in result)

    def test_keeps_all_below_threshold(self):
        matches = self._make_matches([0.1, 0.2, 0.3])
        result = filter_matches_by_distance(matches, 1.0)
        assert len(result) == 3

    def test_empty_input(self):
        result = filter_matches_by_distance([], 0.5)
        assert result == []

    def test_filters_all_out(self):
        matches = self._make_matches([0.8, 0.9])
        result = filter_matches_by_distance(matches, 0.5)
        assert result == []

    def test_returns_list(self):
        matches = self._make_matches([0.1])
        result = filter_matches_by_distance(matches, 1.0)
        assert isinstance(result, list)
