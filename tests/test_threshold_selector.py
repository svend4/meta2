"""Тесты для puzzle_reconstruction.scoring.threshold_selector."""
import numpy as np
import pytest

from puzzle_reconstruction.scoring.threshold_selector import (
    ThresholdConfig,
    ThresholdResult,
    apply_threshold,
    batch_select_thresholds,
    select_adaptive_threshold,
    select_f1_threshold,
    select_fixed_threshold,
    select_otsu_threshold,
    select_percentile_threshold,
    select_threshold,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _uniform(n=100, low=0.0, high=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, n)


# ─── TestThresholdConfig ──────────────────────────────────────────────────────

class TestThresholdConfig:
    def test_defaults(self):
        cfg = ThresholdConfig()
        assert cfg.method == "percentile"
        assert cfg.fixed_value == 0.5
        assert cfg.percentile == 50.0
        assert cfg.n_bins == 256
        assert cfg.beta == 1.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(method="unknown")

    def test_negative_fixed_value_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(fixed_value=-0.1)

    def test_percentile_below_0_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(percentile=-1.0)

    def test_percentile_above_100_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(percentile=101.0)

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(n_bins=1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(beta=-1.0)

    def test_valid_methods(self):
        for m in ("fixed", "percentile", "otsu", "f1", "adaptive"):
            cfg = ThresholdConfig(method=m)
            assert cfg.method == m


# ─── TestThresholdResult ──────────────────────────────────────────────────────

class TestThresholdResult:
    def _make(self, threshold=0.5, n_above=30, n_below=70, n_total=100):
        return ThresholdResult(
            threshold=threshold,
            method="fixed",
            n_above=n_above,
            n_below=n_below,
            n_total=n_total,
        )

    def test_basic_construction(self):
        r = self._make()
        assert r.threshold == 0.5
        assert r.method == "fixed"

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=-0.1, method="fixed",
                            n_above=0, n_below=0, n_total=0)

    def test_negative_n_above_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="fixed",
                            n_above=-1, n_below=0, n_total=0)

    def test_negative_n_below_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="fixed",
                            n_above=0, n_below=-1, n_total=0)

    def test_negative_n_total_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="fixed",
                            n_above=0, n_below=0, n_total=-1)

    def test_acceptance_ratio_basic(self):
        r = self._make(n_above=30, n_total=100)
        assert abs(r.acceptance_ratio - 0.3) < 1e-9

    def test_rejection_ratio_basic(self):
        r = self._make(n_below=70, n_total=100)
        assert abs(r.rejection_ratio - 0.7) < 1e-9

    def test_acceptance_ratio_zero_total(self):
        r = ThresholdResult(threshold=0.0, method="fixed",
                            n_above=0, n_below=0, n_total=0)
        assert r.acceptance_ratio == 0.0

    def test_rejection_ratio_zero_total(self):
        r = ThresholdResult(threshold=0.0, method="fixed",
                            n_above=0, n_below=0, n_total=0)
        assert r.rejection_ratio == 0.0

    def test_ratios_sum_to_one(self):
        r = self._make(n_above=40, n_below=60, n_total=100)
        assert abs(r.acceptance_ratio + r.rejection_ratio - 1.0) < 1e-9


# ─── TestSelectFixedThreshold ─────────────────────────────────────────────────

class TestSelectFixedThreshold:
    def test_returns_threshold_result(self):
        r = select_fixed_threshold(_uniform(), 0.5)
        assert isinstance(r, ThresholdResult)

    def test_threshold_is_fixed_value(self):
        r = select_fixed_threshold(_uniform(), 0.3)
        assert abs(r.threshold - 0.3) < 1e-9

    def test_method_is_fixed(self):
        r = select_fixed_threshold(_uniform(), 0.5)
        assert r.method == "fixed"

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([]), 0.5)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(_uniform(), -0.1)

    def test_n_above_plus_n_below_equals_total(self):
        scores = _uniform(50)
        r = select_fixed_threshold(scores, 0.5)
        assert r.n_above + r.n_below == r.n_total

    def test_n_total_equals_len_scores(self):
        scores = _uniform(80)
        r = select_fixed_threshold(scores, 0.4)
        assert r.n_total == 80

    def test_zero_threshold_all_above(self):
        scores = _uniform(50, low=0.01, high=1.0)
        r = select_fixed_threshold(scores, 0.0)
        assert r.n_above == 50


# ─── TestSelectPercentileThreshold ────────────────────────────────────────────

class TestSelectPercentileThreshold:
    def test_returns_threshold_result(self):
        r = select_percentile_threshold(_uniform())
        assert isinstance(r, ThresholdResult)

    def test_method_is_percentile(self):
        r = select_percentile_threshold(_uniform())
        assert r.method == "percentile"

    def test_median_percentile(self):
        scores = np.arange(0.0, 1.01, 0.01)
        r = select_percentile_threshold(scores, 50.0)
        median = float(np.percentile(scores, 50.0))
        assert abs(r.threshold - median) < 1e-6

    def test_percentile_0_gives_min(self):
        scores = _uniform(100)
        r = select_percentile_threshold(scores, 0.0)
        assert abs(r.threshold - float(scores.min())) < 1e-9

    def test_percentile_100_gives_max(self):
        scores = _uniform(100)
        r = select_percentile_threshold(scores, 100.0)
        assert abs(r.threshold - float(scores.max())) < 1e-9

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(np.array([]))

    def test_percentile_below_0_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(_uniform(), -1.0)

    def test_percentile_above_100_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(_uniform(), 101.0)


# ─── TestSelectOtsuThreshold ──────────────────────────────────────────────────

class TestSelectOtsuThreshold:
    def test_returns_threshold_result(self):
        r = select_otsu_threshold(_uniform())
        assert isinstance(r, ThresholdResult)

    def test_method_is_otsu(self):
        r = select_otsu_threshold(_uniform())
        assert r.method == "otsu"

    def test_bimodal_separates_correctly(self):
        # Two clear clusters: ~0.1 and ~0.9
        low = np.random.default_rng(0).normal(0.1, 0.02, 100).clip(0, 1)
        high = np.random.default_rng(1).normal(0.9, 0.02, 100).clip(0, 1)
        scores = np.concatenate([low, high])
        r = select_otsu_threshold(scores)
        # Threshold should be between the clusters
        assert 0.3 < r.threshold < 0.7

    def test_constant_scores_no_crash(self):
        scores = np.full(50, 0.5)
        r = select_otsu_threshold(scores)
        assert isinstance(r, ThresholdResult)

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(np.array([]))

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(_uniform(), n_bins=1)

    def test_n_above_n_below_sum_correct(self):
        scores = _uniform(100)
        r = select_otsu_threshold(scores)
        assert r.n_above + r.n_below == r.n_total


# ─── TestSelectF1Threshold ────────────────────────────────────────────────────

class TestSelectF1Threshold:
    def _make_binary_data(self, seed=0):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(0, 1, 100)
        labels = (scores > 0.5).astype(int)
        # Add some noise
        noise_idx = rng.choice(100, 10, replace=False)
        labels[noise_idx] = 1 - labels[noise_idx]
        return scores, labels

    def test_returns_threshold_result(self):
        scores, labels = self._make_binary_data()
        r = select_f1_threshold(scores, labels)
        assert isinstance(r, ThresholdResult)

    def test_method_is_f1(self):
        scores, labels = self._make_binary_data()
        r = select_f1_threshold(scores, labels)
        assert r.method == "f1"

    def test_threshold_in_score_range(self):
        scores, labels = self._make_binary_data()
        r = select_f1_threshold(scores, labels)
        assert scores.min() <= r.threshold <= scores.max()

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([0.5, 0.6]),
                                np.array([1, 0, 1]))

    def test_beta_zero_raises(self):
        scores, labels = self._make_binary_data()
        with pytest.raises(ValueError):
            select_f1_threshold(scores, labels, beta=0.0)

    def test_n_candidates_lt_2_raises(self):
        scores, labels = self._make_binary_data()
        with pytest.raises(ValueError):
            select_f1_threshold(scores, labels, n_candidates=1)


# ─── TestSelectAdaptiveThreshold ─────────────────────────────────────────────

class TestSelectAdaptiveThreshold:
    def test_returns_threshold_result(self):
        r = select_adaptive_threshold(_uniform())
        assert isinstance(r, ThresholdResult)

    def test_method_is_adaptive(self):
        r = select_adaptive_threshold(_uniform())
        assert r.method == "adaptive"

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_adaptive_threshold(np.array([]))

    def test_threshold_in_range(self):
        scores = _uniform(200)
        r = select_adaptive_threshold(scores)
        assert scores.min() <= r.threshold <= scores.max()

    def test_n_above_n_below_sum_correct(self):
        r = select_adaptive_threshold(_uniform(100))
        assert r.n_above + r.n_below == r.n_total


# ─── TestSelectThreshold ──────────────────────────────────────────────────────

class TestSelectThreshold:
    def test_default_cfg_uses_percentile(self):
        r = select_threshold(_uniform())
        assert r.method == "percentile"

    def test_fixed_method(self):
        cfg = ThresholdConfig(method="fixed", fixed_value=0.3)
        r = select_threshold(_uniform(), cfg)
        assert r.method == "fixed"
        assert abs(r.threshold - 0.3) < 1e-9

    def test_percentile_method(self):
        cfg = ThresholdConfig(method="percentile", percentile=75.0)
        r = select_threshold(_uniform(), cfg)
        assert r.method == "percentile"

    def test_otsu_method(self):
        cfg = ThresholdConfig(method="otsu")
        r = select_threshold(_uniform(), cfg)
        assert r.method == "otsu"

    def test_adaptive_method(self):
        cfg = ThresholdConfig(method="adaptive")
        r = select_threshold(_uniform(), cfg)
        assert r.method == "adaptive"

    def test_f1_method_without_labels_raises(self):
        cfg = ThresholdConfig(method="f1")
        with pytest.raises(ValueError):
            select_threshold(_uniform(), cfg, labels=None)

    def test_f1_method_with_labels(self):
        scores = _uniform(100)
        labels = (scores > 0.5).astype(int)
        cfg = ThresholdConfig(method="f1")
        r = select_threshold(scores, cfg, labels=labels)
        assert r.method == "f1"


# ─── TestApplyThreshold ───────────────────────────────────────────────────────

class TestApplyThreshold:
    def test_returns_bool_array(self):
        scores = _uniform(50)
        r = select_fixed_threshold(scores, 0.5)
        mask = apply_threshold(scores, r)
        assert mask.dtype == bool

    def test_length_preserved(self):
        scores = _uniform(80)
        r = select_fixed_threshold(scores, 0.5)
        mask = apply_threshold(scores, r)
        assert len(mask) == 80

    def test_values_above_threshold_are_true(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(scores, 0.5)
        mask = apply_threshold(scores, r)
        assert mask[2] is np.bool_(True)
        assert mask[0] is np.bool_(False)

    def test_count_matches_n_above(self):
        scores = _uniform(100)
        r = select_fixed_threshold(scores, 0.5)
        mask = apply_threshold(scores, r)
        assert int(mask.sum()) == r.n_above


# ─── TestBatchSelectThresholds ────────────────────────────────────────────────

class TestBatchSelectThresholds:
    def test_returns_list(self):
        arrays = [_uniform(50), _uniform(50, seed=1)]
        results = batch_select_thresholds(arrays)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_threshold_result(self):
        arrays = [_uniform(30), _uniform(30, seed=2), _uniform(30, seed=3)]
        results = batch_select_thresholds(arrays)
        assert all(isinstance(r, ThresholdResult) for r in results)

    def test_empty_list(self):
        assert batch_select_thresholds([]) == []

    def test_custom_config_applied(self):
        cfg = ThresholdConfig(method="fixed", fixed_value=0.7)
        results = batch_select_thresholds([_uniform(50)], cfg)
        assert results[0].method == "fixed"
        assert abs(results[0].threshold - 0.7) < 1e-9
