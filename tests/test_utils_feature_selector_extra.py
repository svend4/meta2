"""Extra tests for puzzle_reconstruction/utils/feature_selector.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _X(n=20, d=5, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def _sel(n=5) -> SelectionResult:
    idx = np.arange(n, dtype=np.int64)
    scores = np.ones(n, dtype=np.float64)
    return SelectionResult(selected_indices=idx, n_selected=n, scores=scores)


# ─── FeatureSet ───────────────────────────────────────────────────────────────

class TestFeatureSetExtra:
    def test_stores_features(self):
        fs = FeatureSet(features=np.array([1.0, 2.0, 3.0]))
        assert len(fs) == 3

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.zeros((2, 2)))

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.array([1.0]), fragment_id=-1)

    def test_labels_mismatch_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.array([1.0, 2.0]), labels=["only_one"])

    def test_labels_match_ok(self):
        fs = FeatureSet(features=np.array([1.0, 2.0]), labels=["a", "b"])
        assert fs.labels == ["a", "b"]

    def test_fragment_id_stored(self):
        fs = FeatureSet(features=np.array([0.0]), fragment_id=5)
        assert fs.fragment_id == 5

    def test_default_labels_none(self):
        fs = FeatureSet(features=np.array([1.0]))
        assert fs.labels is None


# ─── SelectionResult ──────────────────────────────────────────────────────────

class TestSelectionResultExtra:
    def test_stores_n_selected(self):
        s = _sel(4)
        assert s.n_selected == 4

    def test_len_equals_n_selected(self):
        s = _sel(3)
        assert len(s) == 3

    def test_negative_n_selected_raises(self):
        with pytest.raises(ValueError):
            SelectionResult(selected_indices=np.array([]), n_selected=-1,
                            scores=np.array([]))

    def test_indices_dtype(self):
        s = _sel(3)
        assert s.selected_indices.dtype == np.int64

    def test_scores_dtype(self):
        s = _sel(3)
        assert s.scores.dtype == np.float64


# ─── variance_selection ───────────────────────────────────────────────────────

class TestVarianceSelectionExtra:
    def test_returns_selection_result(self):
        assert isinstance(variance_selection(_X()), SelectionResult)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            variance_selection(np.array([1.0, 2.0]))

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            variance_selection(_X(), threshold=-0.1)

    def test_zero_threshold_keeps_all(self):
        X = _X(20, 4)
        result = variance_selection(X, threshold=0.0)
        assert result.n_selected == 4

    def test_high_threshold_removes_const(self):
        X = np.ones((10, 3))
        result = variance_selection(X, threshold=0.01)
        assert result.n_selected == 0

    def test_scores_length_equals_d(self):
        X = _X(20, 5)
        result = variance_selection(X)
        assert len(result.scores) == 5


# ─── correlation_selection ────────────────────────────────────────────────────

class TestCorrelationSelectionExtra:
    def test_returns_selection_result(self):
        assert isinstance(correlation_selection(_X()), SelectionResult)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            correlation_selection(np.array([1.0, 2.0]))

    def test_invalid_max_corr_raises(self):
        with pytest.raises(ValueError):
            correlation_selection(_X(), max_corr=0.0)

    def test_removes_duplicate_col(self):
        X = np.column_stack([np.linspace(0, 1, 20)] * 3)
        result = correlation_selection(X, max_corr=0.9)
        assert result.n_selected < 3

    def test_independent_features_kept(self):
        rng = np.random.default_rng(42)
        X = rng.random((50, 3))
        result = correlation_selection(X, max_corr=0.99)
        assert result.n_selected >= 1


# ─── rank_features ────────────────────────────────────────────────────────────

class TestRankFeaturesExtra:
    def test_returns_selection_result(self):
        X = _X()
        y = np.arange(20, dtype=float)
        assert isinstance(rank_features(X, y), SelectionResult)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            rank_features(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_mismatched_rows_raises(self):
        with pytest.raises(ValueError):
            rank_features(_X(10, 3), np.arange(5, dtype=float))

    def test_n_selected_equals_d(self):
        X = _X(20, 4)
        y = np.arange(20, dtype=float)
        result = rank_features(X, y)
        assert result.n_selected == 4

    def test_scores_nonneg(self):
        X = _X()
        y = np.arange(20, dtype=float)
        result = rank_features(X, y)
        assert (result.scores >= 0).all()


# ─── pca_reduce ───────────────────────────────────────────────────────────────

class TestPcaReduceExtra:
    def test_returns_tuple(self):
        result = pca_reduce(_X(20, 5), 3)
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_shape(self):
        X = _X(20, 5)
        X_red, evr = pca_reduce(X, 2)
        assert X_red.shape == (20, 2)
        assert evr.shape == (2,)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(np.array([1.0, 2.0]), 1)

    def test_too_many_components_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(_X(5, 3), 10)

    def test_zero_components_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(_X(), 0)

    def test_evr_sums_leq_one(self):
        _, evr = pca_reduce(_X(30, 6), 3)
        assert evr.sum() <= 1.0 + 1e-9


# ─── normalize_features ───────────────────────────────────────────────────────

class TestNormalizeFeaturesExtra:
    def test_minmax_range(self):
        X = _X(20, 4)
        Xn = normalize_features(X, method="minmax")
        assert Xn.min() >= -1e-9 and Xn.max() <= 1.0 + 1e-9

    def test_zscore_mean_zero(self):
        X = _X(20, 4)
        Xn = normalize_features(X, method="zscore")
        assert np.abs(Xn.mean(axis=0)).max() < 1e-9

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_features(_X(), method="l2")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_features(np.array([1.0, 2.0]))

    def test_returns_float64(self):
        Xn = normalize_features(_X())
        assert Xn.dtype == np.float64


# ─── select_top_k ─────────────────────────────────────────────────────────────

class TestSelectTopKExtra:
    def test_returns_selection_result(self):
        s = _sel(5)
        assert isinstance(select_top_k(s, 3), SelectionResult)

    def test_n_selected_reduced(self):
        s = _sel(5)
        result = select_top_k(s, 2)
        assert result.n_selected == 2

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            select_top_k(_sel(5), 0)

    def test_k_exceeds_n_raises(self):
        with pytest.raises(ValueError):
            select_top_k(_sel(3), 10)

    def test_k_equals_n_ok(self):
        s = _sel(4)
        result = select_top_k(s, 4)
        assert result.n_selected == 4


# ─── apply_selection ──────────────────────────────────────────────────────────

class TestApplySelectionExtra:
    def test_output_shape(self):
        X = _X(10, 6)
        s = _sel(3)
        result = apply_selection(X, s)
        assert result.shape == (10, 3)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            apply_selection(np.array([1.0, 2.0]), _sel(1))

    def test_correct_columns_selected(self):
        X = np.eye(5)
        sel = SelectionResult(selected_indices=np.array([0, 2], dtype=np.int64),
                               n_selected=2, scores=np.ones(2))
        result = apply_selection(X, sel)
        assert result.shape == (5, 2)


# ─── batch_select ─────────────────────────────────────────────────────────────

class TestBatchSelectExtra:
    def test_returns_list(self):
        fs_list = [FeatureSet(features=np.arange(5, dtype=float))]
        sel = SelectionResult(selected_indices=np.array([0, 1], dtype=np.int64),
                               n_selected=2, scores=np.ones(2))
        assert isinstance(batch_select(fs_list, sel), list)

    def test_length_matches(self):
        fs_list = [FeatureSet(features=np.arange(5, dtype=float))] * 3
        sel = SelectionResult(selected_indices=np.array([0, 2], dtype=np.int64),
                               n_selected=2, scores=np.ones(2))
        result = batch_select(fs_list, sel)
        assert len(result) == 3

    def test_each_is_feature_set(self):
        fs_list = [FeatureSet(features=np.arange(4, dtype=float))]
        sel = SelectionResult(selected_indices=np.array([1, 3], dtype=np.int64),
                               n_selected=2, scores=np.ones(2))
        for fs in batch_select(fs_list, sel):
            assert isinstance(fs, FeatureSet)

    def test_feature_length_reduced(self):
        fs_list = [FeatureSet(features=np.arange(6, dtype=float))]
        sel = SelectionResult(selected_indices=np.array([0, 1, 2], dtype=np.int64),
                               n_selected=3, scores=np.ones(3))
        result = batch_select(fs_list, sel)
        assert len(result[0]) == 3

    def test_empty_list_returns_empty(self):
        sel = _sel(2)
        assert batch_select([], sel) == []
