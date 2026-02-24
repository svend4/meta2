"""Extra tests for puzzle_reconstruction/scoring/threshold_selector.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.scoring.threshold_selector import (
    ThresholdConfig,
    ThresholdResult,
    select_fixed_threshold,
    select_percentile_threshold,
    select_otsu_threshold,
    select_f1_threshold,
    select_adaptive_threshold,
    select_threshold,
    apply_threshold,
    batch_select_thresholds,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _scores(n=20, low=0.0, high=1.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(low, high, n).astype(np.float64)


def _bimodal(n=50) -> np.ndarray:
    rng = np.random.default_rng(7)
    a = rng.normal(0.2, 0.05, n // 2)
    b = rng.normal(0.8, 0.05, n // 2)
    return np.clip(np.concatenate([a, b]), 0.0, 1.0)


def _result(t=0.5, method="fixed", n_above=5, n_below=5) -> ThresholdResult:
    return ThresholdResult(threshold=t, method=method,
                           n_above=n_above, n_below=n_below,
                           n_total=n_above + n_below)


# ─── ThresholdConfig ──────────────────────────────────────────────────────────

class TestThresholdConfigExtra:
    def test_default_method(self):
        assert ThresholdConfig().method == "percentile"

    def test_default_fixed_value(self):
        assert ThresholdConfig().fixed_value == pytest.approx(0.5)

    def test_default_percentile(self):
        assert ThresholdConfig().percentile == pytest.approx(50.0)

    def test_default_n_bins(self):
        assert ThresholdConfig().n_bins == 256

    def test_default_beta(self):
        assert ThresholdConfig().beta == pytest.approx(1.0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(method="unknown")

    def test_negative_fixed_value_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(fixed_value=-0.1)

    def test_percentile_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(percentile=101.0)

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(n_bins=1)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(beta=0.0)

    def test_valid_methods(self):
        for m in ("fixed", "percentile", "otsu", "f1", "adaptive"):
            cfg = ThresholdConfig(method=m)
            assert cfg.method == m


# ─── ThresholdResult ──────────────────────────────────────────────────────────

class TestThresholdResultExtra:
    def test_threshold_stored(self):
        r = _result(t=0.7)
        assert r.threshold == pytest.approx(0.7)

    def test_method_stored(self):
        r = _result(method="otsu")
        assert r.method == "otsu"

    def test_n_above_stored(self):
        r = _result(n_above=8)
        assert r.n_above == 8

    def test_n_below_stored(self):
        r = _result(n_below=3)
        assert r.n_below == 3

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=-0.1, method="x",
                            n_above=0, n_below=0, n_total=0)

    def test_negative_n_above_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="x",
                            n_above=-1, n_below=0, n_total=0)

    def test_negative_n_below_raises(self):
        with pytest.raises(ValueError):
            ThresholdResult(threshold=0.5, method="x",
                            n_above=0, n_below=-1, n_total=0)

    def test_acceptance_ratio_zero_when_no_total(self):
        r = ThresholdResult(threshold=0.5, method="x",
                            n_above=0, n_below=0, n_total=0)
        assert r.acceptance_ratio == pytest.approx(0.0)

    def test_acceptance_ratio_computed(self):
        r = _result(n_above=3, n_below=7)
        assert r.acceptance_ratio == pytest.approx(0.3)

    def test_rejection_ratio_zero_when_no_total(self):
        r = ThresholdResult(threshold=0.5, method="x",
                            n_above=0, n_below=0, n_total=0)
        assert r.rejection_ratio == pytest.approx(0.0)

    def test_rejection_ratio_computed(self):
        r = _result(n_above=3, n_below=7)
        assert r.rejection_ratio == pytest.approx(0.7)

    def test_acceptance_plus_rejection_one(self):
        r = _result(n_above=4, n_below=6)
        assert r.acceptance_ratio + r.rejection_ratio == pytest.approx(1.0)


# ─── select_fixed_threshold ───────────────────────────────────────────────────

class TestSelectFixedThresholdExtra:
    def test_returns_result(self):
        assert isinstance(select_fixed_threshold(_scores()), ThresholdResult)

    def test_method_is_fixed(self):
        r = select_fixed_threshold(_scores(), value=0.5)
        assert r.method == "fixed"

    def test_threshold_equals_value(self):
        r = select_fixed_threshold(_scores(), value=0.3)
        assert r.threshold == pytest.approx(0.3)

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([]))

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(_scores(), value=-0.1)

    def test_n_total_correct(self):
        s = _scores(10)
        r = select_fixed_threshold(s, 0.5)
        assert r.n_total == 10

    def test_n_above_plus_n_below_equals_n_total(self):
        s = _scores(20)
        r = select_fixed_threshold(s, 0.5)
        assert r.n_above + r.n_below == r.n_total


# ─── select_percentile_threshold ──────────────────────────────────────────────

class TestSelectPercentileThresholdExtra:
    def test_returns_result(self):
        assert isinstance(select_percentile_threshold(_scores()), ThresholdResult)

    def test_method_is_percentile(self):
        r = select_percentile_threshold(_scores())
        assert r.method == "percentile"

    def test_threshold_in_range(self):
        s = _scores()
        r = select_percentile_threshold(s, 50.0)
        assert s.min() <= r.threshold <= s.max()

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(np.array([]))

    def test_percentile_out_of_range_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(_scores(), 101.0)

    def test_p100_returns_max(self):
        s = np.array([0.1, 0.5, 0.9])
        r = select_percentile_threshold(s, 100.0)
        assert r.threshold == pytest.approx(float(s.max()))

    def test_p0_returns_min(self):
        s = np.array([0.1, 0.5, 0.9])
        r = select_percentile_threshold(s, 0.0)
        assert r.threshold == pytest.approx(float(s.min()))


# ─── select_otsu_threshold ────────────────────────────────────────────────────

class TestSelectOtsuThresholdExtra:
    def test_returns_result(self):
        assert isinstance(select_otsu_threshold(_bimodal()), ThresholdResult)

    def test_method_is_otsu(self):
        r = select_otsu_threshold(_bimodal())
        assert r.method == "otsu"

    def test_threshold_in_range(self):
        s = _bimodal()
        r = select_otsu_threshold(s)
        assert float(s.min()) <= r.threshold <= float(s.max())

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(np.array([]))

    def test_n_bins_lt_2_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(_scores(), n_bins=1)

    def test_uniform_scores_handled(self):
        s = np.full(20, 0.5)
        r = select_otsu_threshold(s)
        assert r.threshold == pytest.approx(0.5)

    def test_bimodal_threshold_between_modes(self):
        s = _bimodal()
        r = select_otsu_threshold(s)
        # Threshold should be between 0.2 and 0.8 for bimodal [0.2, 0.8] distribution
        assert 0.2 <= r.threshold <= 0.8


# ─── select_f1_threshold ──────────────────────────────────────────────────────

class TestSelectF1ThresholdExtra:
    def _sl(self, n=30):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, n)
        labels = (scores > 0.5).astype(int)
        return scores, labels

    def test_returns_result(self):
        s, l = self._sl()
        assert isinstance(select_f1_threshold(s, l), ThresholdResult)

    def test_method_is_f1(self):
        s, l = self._sl()
        assert select_f1_threshold(s, l).method == "f1"

    def test_threshold_in_range(self):
        s, l = self._sl()
        r = select_f1_threshold(s, l)
        assert float(s.min()) <= r.threshold <= float(s.max())

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([0.1, 0.9]),
                                np.array([0, 1, 0]))

    def test_beta_zero_raises(self):
        s, l = self._sl()
        with pytest.raises(ValueError):
            select_f1_threshold(s, l, beta=0.0)

    def test_n_candidates_lt_2_raises(self):
        s, l = self._sl()
        with pytest.raises(ValueError):
            select_f1_threshold(s, l, n_candidates=1)

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([]), np.array([]))


# ─── select_adaptive_threshold ────────────────────────────────────────────────

class TestSelectAdaptiveThresholdExtra:
    def test_returns_result(self):
        assert isinstance(select_adaptive_threshold(_scores()), ThresholdResult)

    def test_method_is_adaptive(self):
        r = select_adaptive_threshold(_scores())
        assert r.method == "adaptive"

    def test_threshold_in_range(self):
        s = _scores()
        r = select_adaptive_threshold(s)
        assert float(s.min()) <= r.threshold <= float(s.max())

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            select_adaptive_threshold(np.array([]))

    def test_single_value(self):
        s = np.array([0.6])
        r = select_adaptive_threshold(s)
        assert isinstance(r, ThresholdResult)


# ─── select_threshold ─────────────────────────────────────────────────────────

class TestSelectThresholdExtra:
    def test_fixed_method(self):
        cfg = ThresholdConfig(method="fixed", fixed_value=0.4)
        r = select_threshold(_scores(), cfg)
        assert r.method == "fixed"
        assert r.threshold == pytest.approx(0.4)

    def test_percentile_method(self):
        cfg = ThresholdConfig(method="percentile", percentile=75.0)
        r = select_threshold(_scores(), cfg)
        assert r.method == "percentile"

    def test_otsu_method(self):
        cfg = ThresholdConfig(method="otsu")
        r = select_threshold(_bimodal(), cfg)
        assert r.method == "otsu"

    def test_adaptive_method(self):
        cfg = ThresholdConfig(method="adaptive")
        r = select_threshold(_scores(), cfg)
        assert r.method == "adaptive"

    def test_f1_without_labels_raises(self):
        cfg = ThresholdConfig(method="f1")
        with pytest.raises(ValueError):
            select_threshold(_scores(), cfg, labels=None)

    def test_f1_with_labels_ok(self):
        s = _scores(20)
        l = (s > 0.5).astype(int)
        cfg = ThresholdConfig(method="f1")
        r = select_threshold(s, cfg, labels=l)
        assert r.method == "f1"

    def test_none_cfg_uses_defaults(self):
        r = select_threshold(_scores(), cfg=None)
        assert isinstance(r, ThresholdResult)


# ─── apply_threshold ──────────────────────────────────────────────────────────

class TestApplyThresholdExtra:
    def test_returns_bool_array(self):
        s = _scores(10)
        r = select_fixed_threshold(s, 0.5)
        mask = apply_threshold(s, r)
        assert mask.dtype == bool

    def test_shape_matches(self):
        s = _scores(15)
        r = select_fixed_threshold(s, 0.5)
        assert apply_threshold(s, r).shape == (15,)

    def test_all_true_when_threshold_zero(self):
        s = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(s, 0.0)
        assert apply_threshold(s, r).all()

    def test_all_false_when_threshold_max(self):
        s = np.array([0.1, 0.5, 0.9])
        r = select_fixed_threshold(s, 1.0)
        # Only values >= 1.0 pass: none in [0.1, 0.5, 0.9]
        mask = apply_threshold(s, r)
        assert not mask.any()

    def test_correct_count(self):
        s = np.array([0.2, 0.4, 0.6, 0.8])
        r = select_fixed_threshold(s, 0.5)
        assert int(apply_threshold(s, r).sum()) == 2


# ─── batch_select_thresholds ──────────────────────────────────────────────────

class TestBatchSelectThresholdsExtra:
    def test_returns_list(self):
        result = batch_select_thresholds([_scores()])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_select_thresholds([_scores(), _scores(15)])
        assert len(result) == 2

    def test_empty_batch(self):
        assert batch_select_thresholds([]) == []

    def test_each_element_is_result(self):
        for r in batch_select_thresholds([_scores()]):
            assert isinstance(r, ThresholdResult)

    def test_none_cfg(self):
        result = batch_select_thresholds([_scores()], cfg=None)
        assert len(result) == 1
