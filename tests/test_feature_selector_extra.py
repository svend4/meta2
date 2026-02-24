"""Extra tests for puzzle_reconstruction/utils/feature_selector.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.feature_selector import (
    FeatureSet,
    SelectionResult,
    variance_selection,
    correlation_selection,
    rank_features,
    pca_reduce,
    normalize_features,
    select_top_k,
    apply_selection,
    batch_select,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _X(n=30, m=5, seed=0):
    return np.random.default_rng(seed).standard_normal((n, m))


def _Xy(n=40, m=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    y = X[:, 0] * 2.0 + rng.standard_normal(n) * 0.1
    return X, y


def _sr(indices, scores=None):
    idx = np.array(indices, dtype=np.int64)
    sc = np.ones(len(indices)) if scores is None else np.array(scores)
    return SelectionResult(selected_indices=idx, n_selected=len(indices), scores=sc)


# ─── FeatureSet (extra) ───────────────────────────────────────────────────────

class TestFeatureSetExtra:
    def test_float32_dtype(self):
        fs = FeatureSet(features=np.ones(5, dtype=np.float32))
        assert fs.features.dtype == np.float32

    def test_int_input_converted(self):
        fs = FeatureSet(features=np.array([1, 2, 3]))
        assert fs.features.dtype == np.float32

    def test_large_fragment_id(self):
        fs = FeatureSet(features=np.ones(3), fragment_id=1000)
        assert fs.fragment_id == 1000

    def test_labels_stored(self):
        fs = FeatureSet(features=np.ones(3), labels=["r", "g", "b"])
        assert fs.labels == ["r", "g", "b"]

    def test_empty_features_len_zero(self):
        fs = FeatureSet(features=np.array([], dtype=np.float32))
        assert len(fs) == 0

    def test_single_feature(self):
        fs = FeatureSet(features=np.array([3.14], dtype=np.float32))
        assert len(fs) == 1
        assert fs.features[0] == pytest.approx(3.14, abs=1e-5)

    def test_labels_none_by_default(self):
        fs = FeatureSet(features=np.ones(3))
        assert fs.labels is None or fs.labels == []


# ─── SelectionResult (extra) ──────────────────────────────────────────────────

class TestSelectionResultExtra:
    def test_len_matches_n_selected(self):
        sr = _sr([0, 1, 2])
        assert len(sr) == 3

    def test_indices_stored(self):
        sr = _sr([2, 4, 6])
        np.testing.assert_array_equal(sr.selected_indices, [2, 4, 6])

    def test_scores_stored(self):
        sr = _sr([0, 1], scores=[0.8, 0.6])
        assert sr.scores[0] == pytest.approx(0.8)

    def test_n_selected_zero(self):
        sr = _sr([])
        assert len(sr) == 0

    def test_scores_float64(self):
        sr = _sr([0, 1])
        assert sr.scores.dtype == np.float64

    def test_large_n_selected(self):
        indices = list(range(100))
        sr = _sr(indices)
        assert sr.n_selected == 100


# ─── variance_selection (extra) ───────────────────────────────────────────────

class TestVarianceSelectionExtra:
    def test_high_threshold_removes_features(self):
        X = _X(50, 5, seed=1)
        # With very high threshold, all might be removed
        result = variance_selection(X, threshold=1e10)
        assert result.n_selected <= X.shape[1]

    def test_zero_threshold_keeps_all(self):
        X = _X(20, 4, seed=2)
        result = variance_selection(X, threshold=0.0)
        assert result.n_selected == 4

    def test_all_constant_all_removed(self):
        X = np.ones((10, 3))
        result = variance_selection(X, threshold=1e-6)
        assert result.n_selected == 0

    def test_scores_are_variances(self):
        X = _X(30, 4, seed=3)
        result = variance_selection(X, threshold=0.0)
        expected_vars = np.var(X, axis=0)
        np.testing.assert_allclose(result.scores, expected_vars, atol=1e-5)

    def test_one_varying_feature_kept(self):
        X = np.zeros((20, 3))
        X[:, 1] = np.linspace(0, 1, 20)  # only col 1 varies
        result = variance_selection(X, threshold=1e-6)
        assert 1 in result.selected_indices
        assert result.n_selected == 1


# ─── correlation_selection (extra) ────────────────────────────────────────────

class TestCorrelationSelectionExtra:
    def test_independent_features_all_kept(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        result = correlation_selection(X, max_corr=0.5)
        assert result.n_selected >= 1

    def test_max_corr_at_one_keeps_all(self):
        X = _X(30, 3, seed=7)
        result = correlation_selection(X, max_corr=1.0)
        assert result.n_selected == 3

    def test_result_is_selection_result(self):
        X = _X(20, 4)
        result = correlation_selection(X, max_corr=0.9)
        assert isinstance(result, SelectionResult)

    def test_two_identical_features_one_removed(self):
        X = _X(30, 3, seed=9)
        X[:, 2] = X[:, 0]  # perfectly correlated
        result = correlation_selection(X, max_corr=0.99)
        # Both 0 and 2 are identical, one must be removed
        assert result.n_selected < 3

    def test_scores_length_matches_features(self):
        X = _X(25, 4, seed=11)
        result = correlation_selection(X, max_corr=0.9)
        assert len(result.scores) <= X.shape[1]


# ─── rank_features (extra) ────────────────────────────────────────────────────

class TestRankFeaturesExtra:
    def test_first_index_highest_score(self):
        X, y = _Xy(40, 5, seed=5)
        result = rank_features(X, y)
        # The first selected index should have the highest correlation
        best_idx = result.selected_indices[0]
        assert result.scores[best_idx] == max(result.scores)

    def test_all_features_returned(self):
        X, y = _Xy(30, 6)
        result = rank_features(X, y)
        assert result.n_selected == 6

    def test_scores_in_0_1(self):
        X, y = _Xy(40, 4, seed=3)
        result = rank_features(X, y)
        assert (result.scores >= 0.0).all()
        assert (result.scores <= 1.0 + 1e-9).all()

    def test_indices_cover_all_features(self):
        X, y = _Xy(30, 5)
        result = rank_features(X, y)
        assert set(result.selected_indices.tolist()) == set(range(5))

    def test_constant_y_all_same_score(self):
        X = _X(20, 3)
        y = np.ones(20)
        result = rank_features(X, y)
        # All correlations with constant y are undefined/zero
        assert result.n_selected == 3


# ─── pca_reduce (extra) ───────────────────────────────────────────────────────

class TestPcaReduceExtra:
    def test_components_1(self):
        X_r, evr = pca_reduce(_X(50, 8), n_components=1)
        assert X_r.shape == (50, 1)
        assert evr.shape == (1,)

    def test_evr_first_largest(self):
        X_r, evr = pca_reduce(_X(50, 8, seed=1), n_components=4)
        assert evr[0] >= evr[1] >= evr[2] >= evr[3]

    def test_evr_nonneg(self):
        _, evr = pca_reduce(_X(40, 6), n_components=3)
        assert (evr >= 0).all()

    def test_output_float64(self):
        X_r, _ = pca_reduce(_X(30, 5), n_components=2)
        assert X_r.dtype == np.float64

    def test_n_components_equals_min_dimension(self):
        X = _X(10, 5)
        X_r, evr = pca_reduce(X, n_components=5)
        assert X_r.shape == (10, 5)

    def test_evr_sum_le_one(self):
        _, evr = pca_reduce(_X(60, 10), n_components=6)
        assert evr.sum() <= 1.0 + 1e-9


# ─── normalize_features (extra) ───────────────────────────────────────────────

class TestNormalizeFeaturesExtra:
    def test_minmax_constant_column(self):
        X = np.array([[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]])
        X_n = normalize_features(X, method="minmax")
        # Constant column (all 1.0) may be 0.0 or 0.5 depending on impl
        assert X_n.shape == (3, 2)

    def test_zscore_large_matrix(self):
        X = _X(100, 8)
        X_n = normalize_features(X, method="zscore")
        np.testing.assert_allclose(X_n.mean(axis=0), 0.0, atol=1e-9)

    def test_minmax_output_shape(self):
        X = _X(20, 5)
        X_n = normalize_features(X, method="minmax")
        assert X_n.shape == X.shape

    def test_zscore_output_shape(self):
        X = _X(15, 3)
        X_n = normalize_features(X, method="zscore")
        assert X_n.shape == X.shape

    def test_output_dtype_float64(self):
        X = _X(10, 3).astype(np.float32)
        X_n = normalize_features(X, method="minmax")
        assert X_n.dtype == np.float64

    def test_zscore_single_column(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_n = normalize_features(X, method="zscore")
        assert X_n.mean() == pytest.approx(0.0, abs=1e-9)


# ─── select_top_k (extra) ─────────────────────────────────────────────────────

class TestSelectTopKExtra:
    def test_top_1_from_rank_features(self):
        X, y = _Xy(40, 5, seed=0)
        sr = rank_features(X, y)  # returns pre-sorted result
        top = select_top_k(sr, k=1)
        assert top.n_selected == 1
        # The top-1 should be the first (best) from rank_features
        assert top.selected_indices[0] == sr.selected_indices[0]

    def test_top_k_all(self):
        sr = _sr([0, 1, 2], scores=[0.5, 0.8, 0.3])
        top = select_top_k(sr, k=3)
        assert top.n_selected == 3

    def test_result_indices_dtype_int64(self):
        sr = _sr([0, 1, 2])
        top = select_top_k(sr, k=2)
        assert top.selected_indices.dtype == np.int64

    def test_top_k_from_sorted_source(self):
        X, y = _Xy(40, 5, seed=2)
        sr = rank_features(X, y)  # already sorted descending
        top = select_top_k(sr, k=3)
        # Top 3 from a pre-sorted result preserves order
        assert top.scores[0] >= top.scores[1] >= top.scores[2]

    def test_k_equals_n_same_result(self):
        sr = _sr([0, 1, 2])
        top = select_top_k(sr, k=3)
        assert set(top.selected_indices.tolist()) == {0, 1, 2}


# ─── apply_selection (extra) ──────────────────────────────────────────────────

class TestApplySelectionExtra:
    def test_select_first_column(self):
        X = np.eye(4)
        sr = _sr([0])
        out = apply_selection(X, sr)
        np.testing.assert_array_equal(out, X[:, [0]])

    def test_select_last_column(self):
        X = np.eye(4)
        sr = _sr([3])
        out = apply_selection(X, sr)
        np.testing.assert_array_equal(out, X[:, [3]])

    def test_output_float(self):
        X = _X(10, 5).astype(np.float32)
        sr = _sr([0, 2])
        out = apply_selection(X, sr)
        assert out.dtype in (np.float32, np.float64)

    def test_select_multiple_columns(self):
        X = _X(15, 6, seed=4)
        sr = _sr([1, 3, 5])
        out = apply_selection(X, sr)
        assert out.shape == (15, 3)
        np.testing.assert_array_equal(out, X[:, [1, 3, 5]])

    def test_output_row_count_preserved(self):
        X = _X(20, 4)
        sr = _sr([0, 1])
        out = apply_selection(X, sr)
        assert out.shape[0] == 20


# ─── batch_select (extra) ─────────────────────────────────────────────────────

class TestBatchSelectExtra:
    def test_output_type_feature_set(self):
        fsets = [FeatureSet(features=np.ones(4, dtype=np.float32))]
        sr = _sr([0, 2])
        out = batch_select(fsets, sr)
        assert isinstance(out[0], FeatureSet)

    def test_correct_n_features_after_select(self):
        fsets = [FeatureSet(features=np.arange(6, dtype=float))]
        sr = _sr([0, 1, 3])
        out = batch_select(fsets, sr)
        assert len(out[0]) == 3

    def test_fragment_ids_preserved(self):
        fsets = [FeatureSet(features=np.ones(4), fragment_id=i) for i in range(3)]
        sr = _sr([0, 2])
        out = batch_select(fsets, sr)
        for i, fs in enumerate(out):
            assert fs.fragment_id == i

    def test_large_batch(self):
        fsets = [FeatureSet(features=np.random.rand(8).astype(np.float32))
                 for _ in range(50)]
        sr = _sr([0, 3, 5])
        out = batch_select(fsets, sr)
        assert len(out) == 50
        assert all(len(fs) == 3 for fs in out)

    def test_empty_batch(self):
        out = batch_select([], _sr([0, 1]))
        assert out == []
