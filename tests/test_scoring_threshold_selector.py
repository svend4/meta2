"""Тесты для puzzle_reconstruction.scoring.threshold_selector."""
import pytest
import numpy as np
from puzzle_reconstruction.scoring.threshold_selector import (
    ThresholdConfig,
    ThresholdResult,
    apply_threshold,
    batch_select_thresholds,
    select_adaptive_threshold,
    select_fixed_threshold,
    select_f1_threshold,
    select_otsu_threshold,
    select_percentile_threshold,
    select_threshold,
)


# ─── TestThresholdConfig ──────────────────────────────────────────────────────

class TestThresholdConfig:
    def test_defaults(self):
        cfg = ThresholdConfig()
        assert cfg.method == "percentile"
        assert cfg.fixed_value == pytest.approx(0.5)
        assert cfg.percentile == pytest.approx(50.0)
        assert cfg.n_bins == 256
        assert cfg.beta == pytest.approx(1.0)

    def test_valid_methods(self):
        for m in ("fixed", "percentile", "otsu", "f1", "adaptive"):
            cfg = ThresholdConfig(method=m)
            assert cfg.method == m

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(method="unknown_method")

    def test_fixed_value_neg_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(fixed_value=-0.1)

    def test_percentile_below_zero_raises(self):
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


# ─── TestThresholdResult ──────────────────────────────────────────────────────

class TestThresholdResult:
    def _make(self, threshold=0.5, method="fixed",
              n_above=6, n_below=4, n_total=10) -> ThresholdResult:
        return ThresholdResult(
            threshold=threshold, method=method,
            n_above=n_above, n_below=n_below, n_total=n_total,
        )

    def test_acceptance_ratio(self):
        r = self._make(n_above=6, n_total=10)
        assert r.acceptance_ratio == pytest.approx(0.6)

    def test_rejection_ratio(self):
        r = self._make(n_below=4, n_total=10)
        assert r.rejection_ratio == pytest.approx(0.4)

    def test_acceptance_ratio_empty(self):
        r = self._make(n_above=0, n_below=0, n_total=0)
        assert r.acceptance_ratio == pytest.approx(0.0)

    def test_rejection_ratio_empty(self):
        r = self._make(n_above=0, n_below=0, n_total=0)
        assert r.rejection_ratio == pytest.approx(0.0)

    def test_threshold_neg_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=-0.1, method="fixed",
                            n_above=0, n_below=0, n_total=0)

    def test_n_above_neg_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="fixed",
                            n_above=-1, n_below=0, n_total=0)

    def test_n_below_neg_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="fixed",
                            n_above=0, n_below=-1, n_total=0)

    def test_n_total_neg_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="fixed",
                            n_above=0, n_below=0, n_total=-1)


# ─── TestSelectFixedThreshold ─────────────────────────────────────────────────

class TestSelectFixedThreshold:
    def test_returns_threshold_result(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(scores, 0.5)
        assert isinstance(r, ThresholdResult)

    def test_method_is_fixed(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(scores, 0.5)
        assert r.method == "fixed"

    def test_threshold_stored(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(scores, 0.7)
        assert r.threshold == pytest.approx(0.7)

    def test_n_above_correct(self):
        scores = np.array([0.3, 0.6, 0.8, 0.9, 0.1])
        r = select_fixed_threshold(scores, 0.5)
        assert r.n_above == 3  # 0.6, 0.8, 0.9

    def test_n_total_correct(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(scores, 0.5)
        assert r.n_total == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([]), 0.5)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([0.5]), -0.1)


# ─── TestSelectPercentileThreshold ───────────────────────────────────────────

class TestSelectPercentileThreshold:
    def test_returns_threshold_result(self):
        scores = np.linspace(0, 1, 100)
        r = select_percentile_threshold(scores, 50.0)
        assert isinstance(r, ThresholdResult)

    def test_method_is_percentile(self):
        r = select_percentile_threshold(np.array([0.1, 0.5, 0.9]))
        assert r.method == "percentile"

    def test_median_splits_half(self):
        scores = np.linspace(0, 1, 100)
        r = select_percentile_threshold(scores, 50.0)
        assert r.n_above == pytest.approx(r.n_total / 2, abs=1)

    def test_p100_selects_max(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_percentile_threshold(scores, 100.0)
        assert r.threshold == pytest.approx(0.9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(np.array([]))

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(np.array([0.5]), -1.0)


# ─── TestSelectOtsuThreshold ──────────────────────────────────────────────────

class TestSelectOtsuThreshold:
    def test_returns_threshold_result(self):
        scores = np.array([0.1, 0.15, 0.8, 0.85, 0.9])
        r = select_otsu_threshold(scores)
        assert isinstance(r, ThresholdResult)

    def test_method_is_otsu(self):
        scores = np.array([0.1, 0.15, 0.8, 0.85])
        r = select_otsu_threshold(scores)
        assert r.method == "otsu"

    def test_bimodal_separates(self):
        # Two clear clusters: low [0.1-0.2] and high [0.8-0.9]
        scores = np.array([0.1, 0.12, 0.15, 0.18, 0.8, 0.82, 0.85, 0.88])
        r = select_otsu_threshold(scores)
        assert 0.2 < r.threshold < 0.8

    def test_uniform_scores(self):
        scores = np.full(10, 0.5)
        r = select_otsu_threshold(scores)
        assert isinstance(r, ThresholdResult)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(np.array([]))

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(np.array([0.1, 0.5]), n_bins=1)


# ─── TestSelectAdaptiveThreshold ─────────────────────────────────────────────

class TestSelectAdaptiveThreshold:
    def test_returns_threshold_result(self):
        scores = np.linspace(0, 1, 20)
        r = select_adaptive_threshold(scores)
        assert isinstance(r, ThresholdResult)

    def test_method_is_adaptive(self):
        scores = np.linspace(0, 1, 20)
        r = select_adaptive_threshold(scores)
        assert r.method == "adaptive"

    def test_threshold_in_range(self):
        scores = np.linspace(0, 1, 20)
        r = select_adaptive_threshold(scores)
        assert scores.min() <= r.threshold <= scores.max()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_adaptive_threshold(np.array([]))


# ─── TestSelectThreshold ──────────────────────────────────────────────────────

class TestSelectThreshold:
    def test_default_cfg_percentile(self):
        scores = np.linspace(0, 1, 50)
        r = select_threshold(scores)
        assert r.method == "percentile"

    def test_fixed_cfg(self):
        scores = np.linspace(0, 1, 50)
        cfg = ThresholdConfig(method="fixed", fixed_value=0.7)
        r = select_threshold(scores, cfg)
        assert r.threshold == pytest.approx(0.7)

    def test_otsu_cfg(self):
        scores = np.array([0.1, 0.15, 0.8, 0.85])
        cfg = ThresholdConfig(method="otsu")
        r = select_threshold(scores, cfg)
        assert r.method == "otsu"

    def test_f1_requires_labels(self):
        scores = np.array([0.3, 0.7, 0.8])
        cfg = ThresholdConfig(method="f1")
        with pytest.raises(ValueError):
            select_threshold(scores, cfg, labels=None)

    def test_f1_with_labels(self):
        scores = np.array([0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 1, 1, 1])
        cfg = ThresholdConfig(method="f1")
        r = select_threshold(scores, cfg, labels=labels)
        assert isinstance(r, ThresholdResult)


# ─── TestApplyThreshold ───────────────────────────────────────────────────────

class TestApplyThreshold:
    def test_returns_bool_array(self):
        scores = np.array([0.3, 0.7, 0.9])
        result = ThresholdResult(threshold=0.5, method="fixed",
                                 n_above=2, n_below=1, n_total=3)
        mask = apply_threshold(scores, result)
        assert mask.dtype == bool

    def test_correct_mask(self):
        scores = np.array([0.3, 0.6, 0.8])
        result = ThresholdResult(threshold=0.5, method="fixed",
                                 n_above=2, n_below=1, n_total=3)
        mask = apply_threshold(scores, result)
        assert mask.tolist() == [False, True, True]

    def test_all_above(self):
        scores = np.array([0.8, 0.9, 1.0])
        result = ThresholdResult(threshold=0.5, method="fixed",
                                 n_above=3, n_below=0, n_total=3)
        mask = apply_threshold(scores, result)
        assert mask.all()

    def test_all_below(self):
        scores = np.array([0.1, 0.2, 0.3])
        result = ThresholdResult(threshold=0.5, method="fixed",
                                 n_above=0, n_below=3, n_total=3)
        mask = apply_threshold(scores, result)
        assert not mask.any()


# ─── TestBatchSelectThresholds ────────────────────────────────────────────────

class TestBatchSelectThresholds:
    def test_returns_list(self):
        arrays = [np.linspace(0, 1, 10), np.linspace(0, 1, 20)]
        results = batch_select_thresholds(arrays)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_threshold_result(self):
        arrays = [np.linspace(0, 1, 10)]
        results = batch_select_thresholds(arrays)
        assert isinstance(results[0], ThresholdResult)

    def test_empty_list(self):
        assert batch_select_thresholds([]) == []
