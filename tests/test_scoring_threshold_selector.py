"""Тесты для puzzle_reconstruction/scoring/threshold_selector.py."""
import pytest
import numpy as np

from puzzle_reconstruction.scoring.threshold_selector import (
    ThresholdConfig,
    ThresholdResult,
    select_fixed_threshold,
    select_percentile_threshold,
    select_otsu_threshold,
    select_adaptive_threshold,
    select_threshold,
    apply_threshold,
    batch_select_thresholds,
)

SCORES_10 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


class TestThresholdConfig:
    def test_default_method_percentile(self):
        c = ThresholdConfig()
        assert c.method == "percentile"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(method="unknown")

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(percentile=110.0)

    def test_negative_fixed_value_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(fixed_value=-0.1)

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(n_bins=1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(beta=0.0)


class TestThresholdResult:
    def test_acceptance_ratio(self):
        r = ThresholdResult(threshold=0.5, method="fixed",
                             n_above=6, n_below=4, n_total=10)
        assert r.acceptance_ratio == pytest.approx(0.6)

    def test_rejection_ratio(self):
        r = ThresholdResult(threshold=0.5, method="fixed",
                             n_above=6, n_below=4, n_total=10)
        assert r.rejection_ratio == pytest.approx(0.4)

    def test_zero_total_acceptance_ratio(self):
        r = ThresholdResult(threshold=0.5, method="fixed",
                             n_above=0, n_below=0, n_total=0)
        assert r.acceptance_ratio == 0.0

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=-0.1, method="fixed",
                             n_above=0, n_below=0, n_total=0)


class TestSelectFixedThreshold:
    def test_basic(self):
        r = select_fixed_threshold(SCORES_10, value=0.5)
        assert r.threshold == pytest.approx(0.5)
        assert r.method == "fixed"

    def test_n_above_correct(self):
        r = select_fixed_threshold(SCORES_10, value=0.5)
        assert r.n_above == 6

    def test_n_above_plus_n_below_equals_total(self):
        r = select_fixed_threshold(SCORES_10, value=0.5)
        assert r.n_above + r.n_below == r.n_total

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([]))

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(SCORES_10, value=-0.1)


class TestSelectPercentileThreshold:
    def test_50th_percentile(self):
        r = select_percentile_threshold(SCORES_10, percentile=50.0)
        assert r.method == "percentile"
        assert 0.0 <= r.threshold <= 1.0

    def test_0th_percentile_min(self):
        r = select_percentile_threshold(SCORES_10, percentile=0.0)
        assert r.threshold == pytest.approx(SCORES_10.min())

    def test_100th_percentile_max(self):
        r = select_percentile_threshold(SCORES_10, percentile=100.0)
        assert r.threshold == pytest.approx(SCORES_10.max())

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(SCORES_10, percentile=101.0)


class TestSelectOtsuThreshold:
    def test_bimodal_splits_correctly(self):
        scores = np.array([0.1, 0.1, 0.12, 0.88, 0.9, 0.92])
        r = select_otsu_threshold(scores)
        assert r.method == "otsu"
        assert 0.2 <= r.threshold <= 0.8

    def test_constant_scores(self):
        scores = np.full(10, 0.5)
        r = select_otsu_threshold(scores)
        assert r.threshold == pytest.approx(0.5)

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(SCORES_10, n_bins=1)


class TestSelectAdaptiveThreshold:
    def test_returns_threshold_result(self):
        r = select_adaptive_threshold(SCORES_10)
        assert isinstance(r, ThresholdResult)
        assert r.method == "adaptive"

    def test_threshold_in_data_range(self):
        r = select_adaptive_threshold(SCORES_10)
        assert SCORES_10.min() <= r.threshold <= SCORES_10.max()


class TestSelectThreshold:
    def test_default_config(self):
        r = select_threshold(SCORES_10)
        assert isinstance(r, ThresholdResult)

    def test_fixed_method_via_config(self):
        cfg = ThresholdConfig(method="fixed", fixed_value=0.6)
        r = select_threshold(SCORES_10, cfg=cfg)
        assert r.threshold == pytest.approx(0.6)

    def test_f1_method_requires_labels(self):
        cfg = ThresholdConfig(method="f1")
        with pytest.raises(ValueError):
            select_threshold(SCORES_10, cfg=cfg, labels=None)

    def test_f1_method_with_labels(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        cfg = ThresholdConfig(method="f1")
        r = select_threshold(SCORES_10, cfg=cfg, labels=labels)
        assert isinstance(r, ThresholdResult)


class TestApplyThreshold:
    def test_returns_bool_array(self):
        r = select_fixed_threshold(SCORES_10, value=0.5)
        mask = apply_threshold(SCORES_10, r)
        assert mask.dtype == bool

    def test_mask_shape_matches_scores(self):
        r = select_fixed_threshold(SCORES_10, value=0.5)
        mask = apply_threshold(SCORES_10, r)
        assert mask.shape == SCORES_10.shape

    def test_above_threshold_are_true(self):
        r = select_fixed_threshold(SCORES_10, value=0.5)
        mask = apply_threshold(SCORES_10, r)
        assert np.all(SCORES_10[mask] >= 0.5)
        assert np.all(SCORES_10[~mask] < 0.5)


class TestBatchSelectThresholds:
    def test_returns_list_of_results(self):
        arrays = [SCORES_10, np.linspace(0, 1, 20)]
        results = batch_select_thresholds(arrays)
        assert len(results) == 2
        assert all(isinstance(r, ThresholdResult) for r in results)
