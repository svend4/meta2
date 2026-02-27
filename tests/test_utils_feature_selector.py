"""Tests for puzzle_reconstruction.utils.feature_selector"""
import numpy as np
import pytest
from puzzle_reconstruction.utils.feature_selector import (
    FeatureSet, SelectionResult,
    variance_selection, correlation_selection, rank_features,
    pca_reduce, normalize_features, select_top_k, apply_selection,
    batch_select,
)

np.random.seed(42)


# ── FeatureSet ────────────────────────────────────────────────────────────────

def test_feature_set_1d_required():
    with pytest.raises(ValueError):
        FeatureSet(features=np.ones((3, 4)))


def test_feature_set_fragment_id_negative_raises():
    with pytest.raises(ValueError):
        FeatureSet(features=np.ones(5), fragment_id=-1)


def test_feature_set_labels_length_mismatch_raises():
    with pytest.raises(ValueError):
        FeatureSet(features=np.ones(3), labels=["a", "b"])


def test_feature_set_len():
    fs = FeatureSet(features=np.ones(7))
    assert len(fs) == 7


def test_feature_set_dtype_float32():
    fs = FeatureSet(features=np.array([1, 2, 3], dtype=np.int32))
    assert fs.features.dtype == np.float32


# ── SelectionResult ───────────────────────────────────────────────────────────

def test_selection_result_negative_n_selected_raises():
    with pytest.raises(ValueError):
        SelectionResult(selected_indices=np.array([]), n_selected=-1, scores=np.array([]))


def test_selection_result_len():
    sr = SelectionResult(selected_indices=np.array([0, 1]), n_selected=2, scores=np.zeros(2))
    assert len(sr) == 2


# ── variance_selection ────────────────────────────────────────────────────────

def test_variance_selection_basic():
    X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
    result = variance_selection(X, threshold=0.5)
    assert isinstance(result, SelectionResult)
    # Column 0 has zero variance; should be excluded
    assert 0 not in result.selected_indices
    assert 1 in result.selected_indices


def test_variance_selection_zero_threshold():
    X = np.random.rand(20, 5)
    result = variance_selection(X, threshold=0.0)
    assert result.n_selected == 5


def test_variance_selection_not_2d_raises():
    with pytest.raises(ValueError):
        variance_selection(np.ones(5))


def test_variance_selection_negative_threshold_raises():
    with pytest.raises(ValueError):
        variance_selection(np.ones((5, 3)), threshold=-0.1)


def test_variance_selection_scores_shape():
    X = np.random.rand(10, 4)
    result = variance_selection(X)
    assert result.scores.shape == (4,)


# ── correlation_selection ─────────────────────────────────────────────────────

def test_correlation_selection_removes_correlated():
    np.random.seed(0)
    x = np.random.rand(50)
    X = np.column_stack([x, x + 1e-10, np.random.rand(50)])  # col0 and col1 are highly correlated
    result = correlation_selection(X, max_corr=0.95)
    # At most 2 features kept (one from the correlated pair)
    assert result.n_selected <= 2


def test_correlation_selection_not_2d_raises():
    with pytest.raises(ValueError):
        correlation_selection(np.ones(5))


def test_correlation_selection_invalid_max_corr_raises():
    with pytest.raises(ValueError):
        correlation_selection(np.ones((5, 3)), max_corr=0.0)


def test_correlation_selection_max_corr_gt1_raises():
    with pytest.raises(ValueError):
        correlation_selection(np.ones((5, 3)), max_corr=1.5)


def test_correlation_selection_returns_indices_int64():
    X = np.random.rand(20, 4)
    result = correlation_selection(X)
    assert result.selected_indices.dtype == np.int64


# ── rank_features ─────────────────────────────────────────────────────────────

def test_rank_features_sorted_by_score():
    np.random.seed(1)
    X = np.random.rand(30, 5)
    y = X[:, 2] + 0.01 * np.random.rand(30)  # col 2 most correlated with y
    result = rank_features(X, y)
    assert result.selected_indices[0] == 2


def test_rank_features_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        rank_features(np.ones((5, 3)), np.ones(6))


def test_rank_features_not_2d_raises():
    with pytest.raises(ValueError):
        rank_features(np.ones(5), np.ones(5))


def test_rank_features_scores_nonneg():
    X = np.random.rand(20, 4)
    y = np.random.rand(20)
    result = rank_features(X, y)
    assert np.all(result.scores >= 0)


# ── pca_reduce ────────────────────────────────────────────────────────────────

def test_pca_reduce_output_shape():
    X = np.random.rand(30, 10)
    X_r, evr = pca_reduce(X, n_components=3)
    assert X_r.shape == (30, 3)
    assert evr.shape == (3,)


def test_pca_reduce_explained_variance_sum():
    X = np.random.rand(50, 8)
    _, evr = pca_reduce(X, n_components=4)
    assert 0.0 <= evr.sum() <= 1.0 + 1e-9


def test_pca_reduce_invalid_n_components_raises():
    with pytest.raises(ValueError):
        pca_reduce(np.random.rand(10, 5), n_components=0)


def test_pca_reduce_too_many_components_raises():
    with pytest.raises(ValueError):
        pca_reduce(np.random.rand(5, 3), n_components=10)


def test_pca_reduce_not_2d_raises():
    with pytest.raises(ValueError):
        pca_reduce(np.ones(5), n_components=1)


# ── normalize_features ────────────────────────────────────────────────────────

def test_normalize_features_minmax_range():
    X = np.random.rand(20, 5) * 100
    Xn = normalize_features(X, method="minmax")
    assert Xn.min() >= -1e-9
    assert Xn.max() <= 1.0 + 1e-9


def test_normalize_features_zscore_mean_std():
    X = np.random.rand(50, 4) * 10
    Xn = normalize_features(X, method="zscore")
    assert np.allclose(Xn.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Xn.std(axis=0), 1.0, atol=1e-9)


def test_normalize_features_invalid_method_raises():
    with pytest.raises(ValueError):
        normalize_features(np.ones((5, 3)), method="l2")


def test_normalize_features_not_2d_raises():
    with pytest.raises(ValueError):
        normalize_features(np.ones(5))


def test_normalize_features_returns_float64():
    X = np.ones((5, 3), dtype=np.int32)
    Xn = normalize_features(X)
    assert Xn.dtype == np.float64


# ── select_top_k ──────────────────────────────────────────────────────────────

def test_select_top_k_basic():
    result = SelectionResult(selected_indices=np.array([3, 1, 0, 2]),
                             n_selected=4, scores=np.array([0.9, 0.7, 0.5, 0.3]))
    top2 = select_top_k(result, 2)
    assert top2.n_selected == 2
    assert len(top2.selected_indices) == 2


def test_select_top_k_k_zero_raises():
    result = SelectionResult(selected_indices=np.array([0, 1]),
                             n_selected=2, scores=np.zeros(2))
    with pytest.raises(ValueError):
        select_top_k(result, 0)


def test_select_top_k_k_exceeds_raises():
    result = SelectionResult(selected_indices=np.array([0, 1]),
                             n_selected=2, scores=np.zeros(2))
    with pytest.raises(ValueError):
        select_top_k(result, 5)


# ── apply_selection ───────────────────────────────────────────────────────────

def test_apply_selection_shape():
    X = np.random.rand(20, 6)
    result = SelectionResult(selected_indices=np.array([0, 2, 4]),
                             n_selected=3, scores=np.zeros(3))
    Xs = apply_selection(X, result)
    assert Xs.shape == (20, 3)


def test_apply_selection_not_2d_raises():
    result = SelectionResult(selected_indices=np.array([0]),
                             n_selected=1, scores=np.zeros(1))
    with pytest.raises(ValueError):
        apply_selection(np.ones(5), result)


# ── batch_select ──────────────────────────────────────────────────────────────

def test_batch_select_applies_to_all():
    fsets = [FeatureSet(features=np.random.rand(6).astype(np.float32)) for _ in range(4)]
    result = SelectionResult(selected_indices=np.array([0, 2, 4]),
                             n_selected=3, scores=np.zeros(3))
    out = batch_select(fsets, result)
    assert len(out) == 4
    assert all(len(fs) == 3 for fs in out)


def test_batch_select_preserves_fragment_id():
    fsets = [FeatureSet(features=np.random.rand(5).astype(np.float32), fragment_id=i) for i in range(3)]
    result = SelectionResult(selected_indices=np.array([0, 1]),
                             n_selected=2, scores=np.zeros(2))
    out = batch_select(fsets, result)
    assert [fs.fragment_id for fs in out] == [0, 1, 2]
