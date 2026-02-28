"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.feature_selector
  - puzzle_reconstruction.utils.contour_sampler
  - puzzle_reconstruction.utils.spatial_index
"""
from __future__ import annotations

import numpy as np
import pytest

# ─── feature_selector ─────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.feature_selector import (
    FeatureSet,
    SelectionResult,
    apply_selection,
    batch_select,
    correlation_selection,
    normalize_features,
    pca_reduce,
    rank_features,
    select_top_k,
    variance_selection,
)

# ─── contour_sampler ──────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.contour_sampler import (
    SampledContour,
    SamplerConfig,
    batch_sample,
    normalize_contour,
    sample_contour,
    sample_corners,
    sample_curvature,
    sample_random,
    sample_uniform,
)

# ─── spatial_index ────────────────────────────────────────────────────────────
from puzzle_reconstruction.utils.spatial_index import (
    SpatialConfig,
    SpatialEntry,
    SpatialIndex,
    build_spatial_index,
    cluster_by_distance,
    pairwise_distances,
    query_knn,
    query_radius,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rng_matrix(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def _rect_contour(n: int = 20) -> np.ndarray:
    """Return a rectangular contour with n points, shape (n, 2)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(t) * 10 + 50, np.sin(t) * 5 + 50])
    return pts.astype(np.float64)


def _line_contour(n: int = 20) -> np.ndarray:
    """Return a simple line contour."""
    return np.column_stack([np.linspace(0, 100, n),
                             np.zeros(n)]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — feature_selector
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureSet:
    """FeatureSet dataclass invariants."""

    def test_features_converted_to_float32(self):
        fs = FeatureSet(features=np.array([1.0, 2.0, 3.0]))
        assert fs.features.dtype == np.float32

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(features=np.array([1.0]), fragment_id=-1)

    def test_len_equals_feature_count(self):
        fs = FeatureSet(features=np.array([1.0, 2.0, 3.0]))
        assert len(fs) == 3

    def test_label_mismatch_raises(self):
        with pytest.raises(ValueError):
            FeatureSet(
                features=np.array([1.0, 2.0]),
                labels=["a", "b", "c"],  # wrong length
            )


class TestVarianceSelection:
    """variance_selection invariants."""

    @pytest.mark.parametrize("n,d", [(20, 5), (50, 10), (100, 20)])
    def test_indices_subset_of_d(self, n, d):
        X = _rng_matrix(n, d)
        res = variance_selection(X)
        assert all(0 <= idx < d for idx in res.selected_indices)

    def test_zero_threshold_keeps_all_variable_features(self):
        X = _rng_matrix(20, 5)
        res = variance_selection(X, threshold=0.0)
        # All features are random and have variance > 0
        assert res.n_selected == 5

    def test_high_threshold_keeps_fewer_features(self):
        X = _rng_matrix(100, 10)
        variances = X.var(axis=0)
        max_var = float(variances.max())
        # threshold slightly below max: keeps exactly 1 feature (the one with max var)
        res_all = variance_selection(X, threshold=0.0)
        res_high = variance_selection(X, threshold=max_var * 0.99)
        assert res_high.n_selected <= res_all.n_selected

    def test_constant_feature_removed_with_threshold_gt_zero(self):
        X = np.random.randn(20, 3)
        X[:, 1] = 5.0  # constant column
        res = variance_selection(X, threshold=1e-6)
        assert 1 not in res.selected_indices

    def test_negative_threshold_raises(self):
        X = _rng_matrix(10, 5)
        with pytest.raises(ValueError):
            variance_selection(X, threshold=-0.1)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            variance_selection(np.zeros((10,)), threshold=0.0)

    def test_scores_length_equals_d(self):
        X = _rng_matrix(20, 7)
        res = variance_selection(X)
        assert len(res.scores) == 7


class TestCorrelationSelection:
    """correlation_selection invariants."""

    @pytest.mark.parametrize("n,d", [(30, 5), (50, 8)])
    def test_indices_within_range(self, n, d):
        X = _rng_matrix(n, d)
        res = correlation_selection(X, max_corr=0.95)
        assert all(0 <= idx < d for idx in res.selected_indices)

    def test_perfectly_correlated_features_removed(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal((50, 1))
        X = np.hstack([base, base * 2, rng.standard_normal((50, 1))])
        res = correlation_selection(X, max_corr=0.99)
        # columns 0 and 1 are perfectly correlated; one should be removed
        assert res.n_selected <= 2

    def test_max_corr_1_keeps_all(self):
        X = _rng_matrix(30, 5)
        res = correlation_selection(X, max_corr=1.0)
        assert res.n_selected == 5

    def test_invalid_max_corr_raises(self):
        X = _rng_matrix(20, 5)
        with pytest.raises(ValueError):
            correlation_selection(X, max_corr=0.0)
        with pytest.raises(ValueError):
            correlation_selection(X, max_corr=1.1)


class TestRankFeatures:
    """rank_features invariants."""

    @pytest.mark.parametrize("n,d", [(20, 5), (50, 8)])
    def test_n_selected_equals_d(self, n, d):
        X = _rng_matrix(n, d)
        y = np.random.randn(n)
        res = rank_features(X, y)
        assert res.n_selected == d

    def test_indices_are_permutation(self):
        n, d = 30, 6
        X = _rng_matrix(n, d)
        y = np.random.randn(n)
        res = rank_features(X, y)
        assert sorted(res.selected_indices.tolist()) == list(range(d))

    def test_scores_non_negative(self):
        n, d = 30, 5
        X = _rng_matrix(n, d)
        y = np.random.randn(n)
        res = rank_features(X, y)
        assert all(s >= 0.0 for s in res.scores)

    def test_shape_mismatch_raises(self):
        X = _rng_matrix(20, 5)
        y = np.random.randn(10)
        with pytest.raises(ValueError):
            rank_features(X, y)


class TestPcaReduce:
    """pca_reduce invariants."""

    @pytest.mark.parametrize("n,d,k", [
        (20, 5, 3), (50, 10, 5), (30, 8, 8),
    ])
    def test_output_shape(self, n, d, k):
        X = _rng_matrix(n, d)
        X_red, evr = pca_reduce(X, k)
        assert X_red.shape == (n, k)
        assert len(evr) == k

    @pytest.mark.parametrize("n,d,k", [(20, 5, 3), (50, 8, 4)])
    def test_evr_sum_le_one(self, n, d, k):
        X = _rng_matrix(n, d)
        _, evr = pca_reduce(X, k)
        assert evr.sum() <= 1.0 + 1e-7

    def test_evr_non_negative(self):
        X = _rng_matrix(30, 6)
        _, evr = pca_reduce(X, 4)
        assert all(e >= -1e-10 for e in evr)

    def test_n_components_too_large_raises(self):
        X = _rng_matrix(10, 5)
        with pytest.raises(ValueError):
            pca_reduce(X, 6)

    def test_n_components_zero_raises(self):
        X = _rng_matrix(10, 5)
        with pytest.raises(ValueError):
            pca_reduce(X, 0)


class TestNormalizeFeatures:
    """normalize_features invariants."""

    @pytest.mark.parametrize("n,d", [(20, 5), (50, 10)])
    def test_output_shape_preserved(self, n, d):
        X = _rng_matrix(n, d)
        Xn = normalize_features(X, "minmax")
        assert Xn.shape == (n, d)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_minmax_output_in_0_1(self, seed):
        X = _rng_matrix(30, 5, seed=seed)
        Xn = normalize_features(X, "minmax")
        assert float(Xn.min()) >= -1e-7
        assert float(Xn.max()) <= 1.0 + 1e-7

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_zscore_approximately_zero_mean(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((100, 5))
        Xn = normalize_features(X, "zscore")
        assert np.abs(Xn.mean(axis=0)).max() < 1e-9

    def test_unknown_method_raises(self):
        X = _rng_matrix(10, 3)
        with pytest.raises(ValueError):
            normalize_features(X, "unknown")

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_features(np.zeros(10), "minmax")


class TestSelectTopK:
    """select_top_k invariants."""

    def _make_result(self, n_selected: int) -> SelectionResult:
        indices = np.arange(n_selected, dtype=np.int64)
        scores = np.random.rand(n_selected)
        return SelectionResult(
            selected_indices=indices,
            n_selected=n_selected,
            scores=scores,
        )

    def test_output_n_selected_equals_k(self):
        res = self._make_result(10)
        top = select_top_k(res, 5)
        assert top.n_selected == 5
        assert len(top.selected_indices) == 5

    def test_k_equals_n_selected_unchanged(self):
        res = self._make_result(5)
        top = select_top_k(res, 5)
        assert top.n_selected == 5

    def test_k_zero_raises(self):
        res = self._make_result(5)
        with pytest.raises(ValueError):
            select_top_k(res, 0)

    def test_k_exceeds_n_raises(self):
        res = self._make_result(3)
        with pytest.raises(ValueError):
            select_top_k(res, 4)


class TestApplySelection:
    """apply_selection invariants."""

    def test_output_columns_match_selection(self):
        X = _rng_matrix(20, 10)
        res = variance_selection(X, threshold=0.0)
        Xs = apply_selection(X, res)
        assert Xs.shape == (20, res.n_selected)

    def test_not_2d_raises(self):
        res = SelectionResult(np.array([0, 1], dtype=np.int64), 2,
                              np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            apply_selection(np.zeros(10), res)


class TestBatchSelect:
    """batch_select invariants."""

    def test_length_preserved(self):
        fsets = [FeatureSet(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))] * 4
        res = SelectionResult(
            np.array([0, 2], dtype=np.int64), 2, np.array([0.9, 0.1])
        )
        out = batch_select(fsets, res)
        assert len(out) == 4

    def test_features_subset_selected(self):
        feats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fsets = [FeatureSet(feats)]
        res = SelectionResult(
            np.array([0, 4], dtype=np.int64), 2, np.array([0.9, 0.1])
        )
        out = batch_select(fsets, res)
        np.testing.assert_allclose(out[0].features, np.array([1.0, 5.0]),
                                   atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — contour_sampler
# ══════════════════════════════════════════════════════════════════════════════

class TestSamplerConfig:
    """SamplerConfig validation invariants."""

    def test_default_valid(self):
        cfg = SamplerConfig()
        assert cfg.n_points >= 2

    def test_n_points_below_2_raises(self):
        with pytest.raises(ValueError):
            SamplerConfig(n_points=1)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            SamplerConfig(strategy="invalid_strategy")

    def test_negative_corner_threshold_raises(self):
        with pytest.raises(ValueError):
            SamplerConfig(corner_threshold=-0.1)


class TestSampleUniform:
    """sample_uniform invariants."""

    @pytest.mark.parametrize("n", [2, 8, 16, 32])
    def test_returns_n_points(self, n):
        contour = _rect_contour(40)
        sc = sample_uniform(contour, n_points=n)
        assert sc.n_points == n

    def test_arc_lengths_non_decreasing(self):
        contour = _rect_contour(40)
        sc = sample_uniform(contour, n_points=16)
        for a, b in zip(sc.arc_lengths[:-1], sc.arc_lengths[1:]):
            assert a <= b + 1e-9

    def test_arc_lengths_start_at_zero(self):
        contour = _rect_contour(40)
        sc = sample_uniform(contour, n_points=16)
        assert abs(sc.arc_lengths[0]) < 1e-9

    def test_output_shape_is_n_2(self):
        contour = _rect_contour(40)
        sc = sample_uniform(contour, n_points=8)
        assert sc.points.shape == (8, 2)

    def test_strategy_field(self):
        contour = _rect_contour(40)
        sc = sample_uniform(contour, n_points=8)
        assert sc.strategy == "uniform"

    def test_n_source_equals_contour_len(self):
        contour = _rect_contour(40)
        sc = sample_uniform(contour, n_points=8)
        assert sc.n_source == len(contour)

    def test_n_points_below_2_raises(self):
        contour = _rect_contour(40)
        with pytest.raises(ValueError):
            sample_uniform(contour, n_points=1)

    def test_invalid_contour_raises(self):
        with pytest.raises(ValueError):
            sample_uniform(np.zeros((1, 2)), n_points=4)


class TestSampleRandom:
    """sample_random invariants."""

    @pytest.mark.parametrize("n", [2, 8, 16])
    def test_returns_n_points(self, n):
        contour = _rect_contour(40)
        sc = sample_random(contour, n_points=n)
        assert sc.n_points == n

    def test_strategy_field(self):
        contour = _rect_contour(40)
        sc = sample_random(contour, n_points=8)
        assert sc.strategy == "random"

    def test_reproducible_with_same_seed(self):
        contour = _rect_contour(40)
        sc1 = sample_random(contour, n_points=8, seed=42)
        sc2 = sample_random(contour, n_points=8, seed=42)
        np.testing.assert_array_equal(sc1.points, sc2.points)

    def test_different_seeds_may_differ(self):
        contour = _rect_contour(40)
        sc1 = sample_random(contour, n_points=8, seed=0)
        sc2 = sample_random(contour, n_points=8, seed=99)
        # Not required to differ, but tests that the seed is used
        # (just check that neither crashes)
        assert sc1.n_points == sc2.n_points


class TestSampleCurvature:
    """sample_curvature invariants."""

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_returns_n_points(self, n):
        contour = _rect_contour(40)
        sc = sample_curvature(contour, n_points=n)
        assert sc.n_points == n

    def test_strategy_field(self):
        contour = _rect_contour(40)
        sc = sample_curvature(contour, n_points=8)
        assert sc.strategy == "curvature"


class TestSampleCorners:
    """sample_corners invariants."""

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_returns_n_points(self, n):
        contour = _rect_contour(40)
        sc = sample_corners(contour, n_points=n)
        assert sc.n_points == n

    def test_strategy_field(self):
        contour = _rect_contour(40)
        sc = sample_corners(contour, n_points=8)
        assert sc.strategy == "corners"

    def test_negative_threshold_raises(self):
        contour = _rect_contour(40)
        with pytest.raises(ValueError):
            sample_corners(contour, n_points=8, corner_threshold=-0.1)


class TestSampleContour:
    """sample_contour (dispatch) invariants."""

    @pytest.mark.parametrize("strategy", ["uniform", "curvature", "random", "corners"])
    def test_dispatch_all_strategies(self, strategy):
        contour = _rect_contour(40)
        cfg = SamplerConfig(n_points=8, strategy=strategy)
        sc = sample_contour(contour, cfg)
        assert sc.n_points == 8
        assert sc.strategy == strategy

    def test_default_config_works(self):
        contour = _rect_contour(40)
        sc = sample_contour(contour)
        assert sc.n_points == SamplerConfig().n_points


class TestNormalizeContour:
    """normalize_contour invariants."""

    @pytest.mark.parametrize("n", [4, 10, 20, 40])
    def test_output_in_minus1_plus1(self, n):
        contour = _rect_contour(n)
        normed = normalize_contour(contour)
        assert float(normed.min()) >= -1.0 - 1e-7
        assert float(normed.max()) <= 1.0 + 1e-7

    @pytest.mark.parametrize("n", [4, 10, 40])
    def test_centroid_near_zero(self, n):
        contour = _rect_contour(n)
        normed = normalize_contour(contour)
        center = normed.mean(axis=0)
        assert abs(center[0]) < 1e-9
        assert abs(center[1]) < 1e-9

    def test_output_shape_preserved(self):
        contour = _rect_contour(20)
        normed = normalize_contour(contour)
        assert normed.shape == contour.shape

    def test_invalid_contour_raises(self):
        with pytest.raises(ValueError):
            normalize_contour(np.zeros((1, 2)))

    def test_degenerate_contour_unchanged(self):
        # All points at same location
        contour = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
                            dtype=np.float64)
        normed = normalize_contour(contour)
        # Should return near-zero (shifted but not scaled)
        assert np.allclose(normed, 0.0, atol=1e-9)

    def test_scale_independent(self):
        contour = _rect_contour(20)
        normed1 = normalize_contour(contour)
        normed2 = normalize_contour(contour * 100.0)
        np.testing.assert_allclose(normed1, normed2, atol=1e-9)


class TestBatchSample:
    """batch_sample invariants."""

    def test_length_preserved(self):
        contours = [_rect_contour(20) for _ in range(5)]
        cfg = SamplerConfig(n_points=8)
        out = batch_sample(contours, cfg)
        assert len(out) == 5

    def test_empty_list_returns_empty(self):
        assert batch_sample([]) == []

    def test_each_sampled_contour_has_correct_n_points(self):
        contours = [_rect_contour(20 + i * 5) for i in range(4)]
        cfg = SamplerConfig(n_points=8)
        out = batch_sample(contours, cfg)
        for sc in out:
            assert sc.n_points == 8


# ══════════════════════════════════════════════════════════════════════════════
# Tests — spatial_index
# ══════════════════════════════════════════════════════════════════════════════

class TestSpatialConfig:
    """SpatialConfig validation invariants."""

    def test_default_valid(self):
        cfg = SpatialConfig()
        assert cfg.cell_size > 0

    def test_zero_cell_size_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(cell_size=0.0)

    def test_negative_cell_size_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(cell_size=-1.0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(metric="L2")

    def test_negative_max_results_raises(self):
        with pytest.raises(ValueError):
            SpatialConfig(max_results=-1)


class TestSpatialEntry:
    """SpatialEntry validation invariants."""

    def test_valid_entry(self):
        e = SpatialEntry(item_id=0, position=np.array([1.0, 2.0]))
        assert e.item_id == 0
        assert e.position.shape == (2,)

    def test_negative_item_id_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=-1, position=np.array([1.0, 2.0]))

    def test_wrong_position_shape_raises(self):
        with pytest.raises(ValueError):
            SpatialEntry(item_id=0, position=np.array([1.0, 2.0, 3.0]))


class TestSpatialIndex:
    """SpatialIndex operation invariants."""

    def _build(self, n: int = 5, seed: int = 0) -> SpatialIndex:
        rng = np.random.default_rng(seed)
        positions = rng.uniform(0, 100, (n, 2))
        return build_spatial_index(positions)

    def test_size_after_build(self):
        idx = self._build(10)
        assert idx.size == 10
        assert len(idx) == 10

    def test_insert_increases_size(self):
        idx = SpatialIndex()
        assert idx.size == 0
        idx.insert(SpatialEntry(0, np.array([0.0, 0.0])))
        assert idx.size == 1

    def test_remove_decreases_size(self):
        idx = self._build(5)
        idx.remove(0)
        assert idx.size == 4

    def test_remove_nonexistent_returns_false(self):
        idx = self._build(3)
        assert not idx.remove(999)

    def test_contains(self):
        idx = self._build(5)
        assert 0 in idx
        assert 4 in idx
        assert 99 not in idx

    def test_clear_empties_index(self):
        idx = self._build(5)
        idx.clear()
        assert idx.size == 0

    def test_get_all_length(self):
        idx = self._build(7)
        assert len(idx.get_all()) == 7


class TestQueryRadius:
    """query_radius invariants."""

    @pytest.mark.parametrize("radius", [10.0, 20.0, 50.0])
    def test_all_results_within_radius(self, radius):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (20, 2))
        idx = build_spatial_index(positions)
        center = np.array([50.0, 50.0])
        results = query_radius(idx, center, radius)
        for dist, _ in results:
            assert dist <= radius + 1e-9

    def test_sorted_by_distance(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (30, 2))
        idx = build_spatial_index(positions)
        center = np.array([50.0, 50.0])
        results = query_radius(idx, center, 100.0)
        dists = [d for d, _ in results]
        assert dists == sorted(dists)

    def test_zero_radius_returns_only_collocated(self):
        positions = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        idx = build_spatial_index(positions)
        center = np.array([10.0, 10.0])
        results = query_radius(idx, center, 0.0)
        # Only the point at exactly center distance=0
        assert all(d <= 1e-9 for d, _ in results)

    def test_negative_radius_raises(self):
        idx = build_spatial_index(np.array([[0.0, 0.0], [1.0, 1.0]]))
        with pytest.raises(ValueError):
            idx.query_radius(np.array([0.0, 0.0]), -1.0)

    def test_large_radius_finds_all_points(self):
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [50.0, 50.0]])
        idx = build_spatial_index(positions)
        results = query_radius(idx, np.array([0.0, 0.0]), 1e9)
        assert len(results) == 3


class TestQueryKnn:
    """query_knn invariants."""

    @pytest.mark.parametrize("k,n", [(1, 10), (3, 10), (5, 10), (10, 10)])
    def test_returns_at_most_k_results(self, k, n):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (n, 2))
        idx = build_spatial_index(positions)
        center = np.array([50.0, 50.0])
        results = query_knn(idx, center, k)
        assert len(results) <= min(k, n)

    def test_returns_exactly_k_when_enough_points(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (20, 2))
        idx = build_spatial_index(positions)
        results = query_knn(idx, np.array([50.0, 50.0]), 5)
        assert len(results) == 5

    def test_sorted_by_distance(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (30, 2))
        idx = build_spatial_index(positions)
        results = query_knn(idx, np.array([50.0, 50.0]), 10)
        dists = [d for d, _ in results]
        assert dists == sorted(dists)

    def test_k_zero_raises(self):
        idx = build_spatial_index(np.array([[0.0, 0.0], [1.0, 1.0]]))
        with pytest.raises(ValueError):
            idx.query_knn(np.array([0.0, 0.0]), 0)

    def test_empty_index_returns_empty(self):
        idx = SpatialIndex()
        results = query_knn(idx, np.array([0.0, 0.0]), 5)
        assert results == []


class TestPairwiseDistances:
    """pairwise_distances invariants."""

    @pytest.mark.parametrize("n,metric", [
        (3, "euclidean"), (5, "manhattan"), (4, "chebyshev"),
        (10, "euclidean"), (8, "manhattan"),
    ])
    def test_matrix_shape(self, n, metric):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (n, 2))
        D = pairwise_distances(positions, metric)
        assert D.shape == (n, n)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_diagonal_is_zero(self, metric):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 100, (8, 2))
        D = pairwise_distances(positions, metric)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_symmetric(self, metric):
        rng = np.random.default_rng(2)
        positions = rng.uniform(0, 100, (8, 2))
        D = pairwise_distances(positions, metric)
        np.testing.assert_allclose(D, D.T, atol=1e-9)

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
    def test_non_negative(self, metric):
        rng = np.random.default_rng(3)
        positions = rng.uniform(0, 100, (8, 2))
        D = pairwise_distances(positions, metric)
        assert float(D.min()) >= -1e-9

    def test_invalid_positions_shape_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.zeros((5, 3)))

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            pairwise_distances(np.zeros((3, 2)), metric="L2")

    def test_euclidean_known_distance(self):
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(positions, "euclidean")
        assert abs(D[0, 1] - 5.0) < 1e-9

    def test_manhattan_known_distance(self):
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(positions, "manhattan")
        assert abs(D[0, 1] - 7.0) < 1e-9

    def test_chebyshev_known_distance(self):
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = pairwise_distances(positions, "chebyshev")
        assert abs(D[0, 1] - 4.0) < 1e-9


class TestBuildSpatialIndex:
    """build_spatial_index invariants."""

    def test_size_equals_n_positions(self):
        positions = np.random.rand(15, 2) * 100
        idx = build_spatial_index(positions)
        assert idx.size == 15

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            build_spatial_index(np.zeros((5, 3)))

    def test_payload_mismatch_raises(self):
        positions = np.zeros((5, 2))
        with pytest.raises(ValueError):
            build_spatial_index(positions, payloads=["a", "b"])


class TestClusterByDistance:
    """cluster_by_distance invariants."""

    @pytest.mark.parametrize("n,threshold", [
        (5, 10.0), (10, 20.0), (20, 5.0),
    ])
    def test_all_indices_present(self, n, threshold):
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 100, (n, 2))
        clusters = cluster_by_distance(positions, threshold)
        all_indices = sorted([i for c in clusters for i in c])
        assert all_indices == list(range(n))

    @pytest.mark.parametrize("n,threshold", [
        (5, 10.0), (10, 20.0), (8, 5.0),
    ])
    def test_clusters_are_disjoint(self, n, threshold):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 100, (n, 2))
        clusters = cluster_by_distance(positions, threshold)
        seen = set()
        for c in clusters:
            for idx in c:
                assert idx not in seen
                seen.add(idx)

    def test_empty_positions_returns_empty(self):
        result = cluster_by_distance(np.zeros((0, 2)), 10.0)
        assert result == []

    def test_large_threshold_gives_single_cluster(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        clusters = cluster_by_distance(positions, threshold=100.0)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_zero_threshold_gives_n_singletons(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        clusters = cluster_by_distance(positions, threshold=0.0)
        assert len(clusters) == 3

    def test_negative_threshold_raises(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            cluster_by_distance(positions, threshold=-1.0)
