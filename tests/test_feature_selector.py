"""Тесты для puzzle_reconstruction.utils.feature_selector."""
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


# ─── TestFeatureSet ───────────────────────────────────────────────────────────

class TestFeatureSet:
    def test_basic_creation(self):
        fs = FeatureSet(features=np.array([1.0, 2.0, 3.0]))
        assert len(fs) == 3

    def test_dtype_converted_to_float32(self):
        fs = FeatureSet(features=np.array([1, 2, 3], dtype=np.int32))
        assert fs.features.dtype == np.float32

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.ones((3, 3)))

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.ones(4), fragment_id=-1)

    def test_labels_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.ones(3), labels=["a", "b"])

    def test_labels_correct(self):
        fs = FeatureSet(features=np.ones(2), labels=["x", "y"])
        assert fs.labels == ["x", "y"]

    def test_len(self):
        fs = FeatureSet(features=np.ones(5))
        assert len(fs) == 5

    def test_fragment_id_default_zero(self):
        fs = FeatureSet(features=np.zeros(2))
        assert fs.fragment_id == 0


# ─── TestSelectionResult ──────────────────────────────────────────────────────

class TestSelectionResult:
    def _make(self, n=3):
        indices = np.arange(n, dtype=np.int64)
        scores = np.ones(n, dtype=np.float64)
        return SelectionResult(selected_indices=indices, n_selected=n, scores=scores)

    def test_basic_creation(self):
        sr = self._make(4)
        assert len(sr) == 4

    def test_indices_dtype_int64(self):
        sr = self._make(3)
        assert sr.selected_indices.dtype == np.int64

    def test_scores_dtype_float64(self):
        sr = self._make(3)
        assert sr.scores.dtype == np.float64

    def test_negative_n_selected_raises(self):
        with pytest.raises(ValueError):
            SelectionResult(
                selected_indices=np.array([], dtype=np.int64),
                n_selected=-1,
                scores=np.array([]),
            )

    def test_zero_n_selected_valid(self):
        sr = SelectionResult(
            selected_indices=np.array([], dtype=np.int64),
            n_selected=0,
            scores=np.array([]),
        )
        assert len(sr) == 0


# ─── TestVarianceSelection ────────────────────────────────────────────────────

class TestVarianceSelection:
    def _X(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((20, 5))

    def test_returns_selection_result(self):
        result = variance_selection(self._X(), threshold=0.0)
        assert isinstance(result, SelectionResult)

    def test_all_kept_at_zero_threshold(self):
        X = self._X()
        result = variance_selection(X, threshold=0.0)
        assert result.n_selected == X.shape[1]

    def test_constant_feature_removed(self):
        X = self._X()
        X[:, 2] = 5.0  # нулевая дисперсия
        result = variance_selection(X, threshold=1e-6)
        assert 2 not in result.selected_indices

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            variance_selection(self._X(), threshold=-0.1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            variance_selection(np.ones(10), threshold=0.0)

    def test_scores_length_matches_n_features(self):
        X = self._X()
        result = variance_selection(X, threshold=0.0)
        assert len(result.scores) == X.shape[1]


# ─── TestCorrelationSelection ─────────────────────────────────────────────────

class TestCorrelationSelection:
    def _X(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4))
        X[:, 3] = X[:, 0] * 0.99 + rng.standard_normal(30) * 0.01  # сильная корр.
        return X

    def test_returns_selection_result(self):
        result = correlation_selection(self._X(), max_corr=0.95)
        assert isinstance(result, SelectionResult)

    def test_correlated_feature_removed(self):
        X = self._X()
        result = correlation_selection(X, max_corr=0.95)
        # Хотя бы один из двух коррелированных признаков должен быть удалён
        assert result.n_selected < X.shape[1]

    def test_max_corr_one_keeps_all(self):
        X = self._X()
        result = correlation_selection(X, max_corr=1.0)
        assert result.n_selected == X.shape[1]

    def test_max_corr_out_of_range_raises(self):
        with pytest.raises(ValueError):
            correlation_selection(self._X(), max_corr=0.0)

    def test_max_corr_gt_one_raises(self):
        with pytest.raises(ValueError):
            correlation_selection(self._X(), max_corr=1.5)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            correlation_selection(np.ones(5), max_corr=0.9)

    def test_empty_features(self):
        X = np.zeros((10, 0))
        result = correlation_selection(X, max_corr=0.9)
        assert result.n_selected == 0


# ─── TestRankFeatures ─────────────────────────────────────────────────────────

class TestRankFeatures:
    def _Xy(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((40, 5))
        y = X[:, 0] * 2.0 + rng.standard_normal(40) * 0.1
        return X, y

    def test_returns_selection_result(self):
        X, y = self._Xy()
        result = rank_features(X, y)
        assert isinstance(result, SelectionResult)

    def test_most_correlated_first(self):
        X, y = self._Xy()
        result = rank_features(X, y)
        # Признак 0 коррелирован с y сильнее всего
        assert result.selected_indices[0] == 0

    def test_n_selected_equals_n_features(self):
        X, y = self._Xy()
        result = rank_features(X, y)
        assert result.n_selected == X.shape[1]

    def test_non_2d_X_raises(self):
        with pytest.raises(ValueError):
            rank_features(np.ones(5), np.ones(5))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rank_features(np.ones((10, 3)), np.ones(8))

    def test_scores_nonnegative(self):
        X, y = self._Xy()
        result = rank_features(X, y)
        assert (result.scores >= 0).all()


# ─── TestPcaReduce ────────────────────────────────────────────────────────────

class TestPcaReduce:
    def _X(self):
        rng = np.random.default_rng(1)
        return rng.standard_normal((50, 8))

    def test_output_shape(self):
        X_r, evr = pca_reduce(self._X(), n_components=3)
        assert X_r.shape == (50, 3)

    def test_evr_shape(self):
        _, evr = pca_reduce(self._X(), n_components=3)
        assert evr.shape == (3,)

    def test_evr_sums_le_one(self):
        _, evr = pca_reduce(self._X(), n_components=5)
        assert evr.sum() <= 1.0 + 1e-9

    def test_evr_nonnegative(self):
        _, evr = pca_reduce(self._X(), n_components=4)
        assert (evr >= 0).all()

    def test_n_components_zero_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(self._X(), n_components=0)

    def test_n_components_too_large_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(self._X(), n_components=100)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            pca_reduce(np.ones(10), n_components=2)

    def test_output_dtype_float64(self):
        X_r, _ = pca_reduce(self._X(), n_components=2)
        assert X_r.dtype == np.float64


# ─── TestNormalizeFeatures ────────────────────────────────────────────────────

class TestNormalizeFeatures:
    def _X(self):
        return np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    def test_minmax_range(self):
        X_n = normalize_features(self._X(), method="minmax")
        assert X_n.min() >= 0.0 - 1e-9
        assert X_n.max() <= 1.0 + 1e-9

    def test_minmax_min_zero(self):
        X_n = normalize_features(self._X(), method="minmax")
        assert X_n[:, 0].min() == pytest.approx(0.0)

    def test_minmax_max_one(self):
        X_n = normalize_features(self._X(), method="minmax")
        assert X_n[:, 0].max() == pytest.approx(1.0)

    def test_zscore_mean_zero(self):
        X_n = normalize_features(self._X(), method="zscore")
        np.testing.assert_allclose(X_n.mean(axis=0), 0.0, atol=1e-10)

    def test_zscore_std_one(self):
        X_n = normalize_features(self._X(), method="zscore")
        np.testing.assert_allclose(X_n.std(axis=0), 1.0, atol=1e-10)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_features(self._X(), method="unknown")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_features(np.ones(5), method="minmax")

    def test_output_dtype_float64(self):
        X_n = normalize_features(self._X(), method="minmax")
        assert X_n.dtype == np.float64


# ─── TestSelectTopK ───────────────────────────────────────────────────────────

class TestSelectTopK:
    def _sr(self):
        X = np.random.default_rng(3).standard_normal((20, 6))
        y = X[:, 0] + np.random.default_rng(3).standard_normal(20) * 0.1
        return rank_features(X, y)

    def test_returns_k_indices(self):
        sr = self._sr()
        top = select_top_k(sr, k=3)
        assert top.n_selected == 3
        assert len(top.selected_indices) == 3

    def test_k_one(self):
        sr = self._sr()
        top = select_top_k(sr, k=1)
        assert top.n_selected == 1

    def test_k_zero_raises(self):
        sr = self._sr()
        with pytest.raises(ValueError):
            select_top_k(sr, k=0)

    def test_k_too_large_raises(self):
        sr = self._sr()
        with pytest.raises(ValueError):
            select_top_k(sr, k=sr.n_selected + 1)

    def test_indices_dtype_int64(self):
        sr = self._sr()
        top = select_top_k(sr, k=2)
        assert top.selected_indices.dtype == np.int64


# ─── TestApplySelection ───────────────────────────────────────────────────────

class TestApplySelection:
    def test_output_shape(self):
        X = np.random.default_rng(0).standard_normal((10, 5))
        sr = variance_selection(X, threshold=0.0)
        out = apply_selection(X, sr)
        assert out.shape[1] == sr.n_selected
        assert out.shape[0] == X.shape[0]

    def test_non_2d_raises(self):
        sr = SelectionResult(
            selected_indices=np.array([0], dtype=np.int64),
            n_selected=1,
            scores=np.array([1.0]),
        )
        with pytest.raises(ValueError):
            apply_selection(np.ones(5), sr)

    def test_selects_correct_columns(self):
        X = np.eye(4)
        sr = SelectionResult(
            selected_indices=np.array([0, 2], dtype=np.int64),
            n_selected=2,
            scores=np.ones(2),
        )
        out = apply_selection(X, sr)
        np.testing.assert_array_equal(out, X[:, [0, 2]])


# ─── TestBatchSelect ──────────────────────────────────────────────────────────

class TestBatchSelect:
    def _sr(self):
        return SelectionResult(
            selected_indices=np.array([0, 2], dtype=np.int64),
            n_selected=2,
            scores=np.ones(2),
        )

    def test_returns_list(self):
        fsets = [FeatureSet(features=np.ones(4)) for _ in range(3)]
        out = batch_select(fsets, self._sr())
        assert isinstance(out, list)

    def test_correct_length(self):
        fsets = [FeatureSet(features=np.arange(4, dtype=float)) for _ in range(5)]
        out = batch_select(fsets, self._sr())
        assert len(out) == 5

    def test_features_selected(self):
        fsets = [FeatureSet(features=np.array([1.0, 2.0, 3.0, 4.0]))]
        out = batch_select(fsets, self._sr())
        np.testing.assert_array_equal(out[0].features, np.array([1.0, 3.0], dtype=np.float32))

    def test_empty_list(self):
        out = batch_select([], self._sr())
        assert out == []

    def test_fragment_id_preserved(self):
        fsets = [FeatureSet(features=np.ones(4), fragment_id=7)]
        out = batch_select(fsets, self._sr())
        assert out[0].fragment_id == 7
