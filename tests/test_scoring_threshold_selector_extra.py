"""Extra tests for puzzle_reconstruction/scoring/threshold_selector.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── ThresholdConfig ────────────────────────────────────────────────────────

class TestThresholdConfigExtra:
    def test_defaults(self):
        cfg = ThresholdConfig()
        assert cfg.method == "percentile"
        assert cfg.fixed_value == pytest.approx(0.5)
        assert cfg.percentile == pytest.approx(50.0)
        assert cfg.n_bins == 256
        assert cfg.beta == pytest.approx(1.0)

    def test_valid_methods(self):
        for m in ("fixed", "percentile", "otsu", "f1", "adaptive"):
            ThresholdConfig(method=m)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(method="invalid")

    def test_negative_fixed_value_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(fixed_value=-0.1)

    def test_percentile_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(percentile=101.0)

    def test_negative_percentile_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(percentile=-1.0)

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(n_bins=1)

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError):
            ThresholdConfig(beta=0.0)


# ─── ThresholdResult ────────────────────────────────────────────────────────

class TestThresholdResultExtra:
    def test_acceptance_ratio(self):
        r = ThresholdResult(threshold=0.5, method="fixed",
                            n_above=7, n_below=3, n_total=10)
        assert r.acceptance_ratio == pytest.approx(0.7)

    def test_rejection_ratio(self):
        r = ThresholdResult(threshold=0.5, method="fixed",
                            n_above=7, n_below=3, n_total=10)
        assert r.rejection_ratio == pytest.approx(0.3)

    def test_zero_total(self):
        r = ThresholdResult(threshold=0.5, method="fixed",
                            n_above=0, n_below=0, n_total=0)
        assert r.acceptance_ratio == pytest.approx(0.0)
        assert r.rejection_ratio == pytest.approx(0.0)

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


# ─── select_fixed_threshold ─────────────────────────────────────────────────

class TestSelectFixedExtra:
    def test_basic(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        r = select_fixed_threshold(scores, 0.5)
        assert r.threshold == pytest.approx(0.5)
        assert r.method == "fixed"
        assert r.n_above == 3
        assert r.n_below == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([]), 0.5)

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            select_fixed_threshold(np.array([0.5]), -0.1)


# ─── select_percentile_threshold ────────────────────────────────────────────

class TestSelectPercentileExtra:
    def test_median(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        r = select_percentile_threshold(scores, 50.0)
        assert r.threshold == pytest.approx(0.5)
        assert r.method == "percentile"

    def test_0th_percentile(self):
        scores = np.array([0.2, 0.5, 0.8])
        r = select_percentile_threshold(scores, 0.0)
        assert r.threshold == pytest.approx(0.2)

    def test_100th_percentile(self):
        scores = np.array([0.2, 0.5, 0.8])
        r = select_percentile_threshold(scores, 100.0)
        assert r.threshold == pytest.approx(0.8)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(np.array([]), 50.0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            select_percentile_threshold(np.array([0.5]), 101.0)


# ─── select_otsu_threshold ──────────────────────────────────────────────────

class TestSelectOtsuExtra:
    def test_bimodal(self):
        low = np.full(50, 0.2)
        high = np.full(50, 0.8)
        scores = np.concatenate([low, high])
        r = select_otsu_threshold(scores)
        assert r.method == "otsu"
        # Threshold separates the two modes; n_above should capture the high group
        assert r.n_above == 50 and r.n_below == 50

    def test_uniform_value(self):
        scores = np.full(10, 0.5)
        r = select_otsu_threshold(scores)
        assert r.threshold == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(np.array([]))

    def test_n_bins_too_small_raises(self):
        with pytest.raises(ValueError):
            select_otsu_threshold(np.array([0.5]), n_bins=1)


# ─── select_f1_threshold ────────────────────────────────────────────────────

class TestSelectF1Extra:
    def test_basic(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        r = select_f1_threshold(scores, labels)
        assert r.method == "f1"
        assert 0.0 <= r.threshold <= 1.0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([0.5]), np.array([0, 1]))

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([0.5]), np.array([1]), beta=0.0)

    def test_n_candidates_too_small_raises(self):
        with pytest.raises(ValueError):
            select_f1_threshold(np.array([0.5]), np.array([1]),
                                n_candidates=1)


# ─── select_adaptive_threshold ──────────────────────────────────────────────

class TestSelectAdaptiveExtra:
    def test_basic(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        r = select_adaptive_threshold(scores)
        assert r.method == "adaptive"
        assert r.threshold >= 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_adaptive_threshold(np.array([]))


# ─── select_threshold (dispatch) ────────────────────────────────────────────

class TestSelectThresholdExtra:
    def test_default_config(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = select_threshold(scores)
        assert r.method == "percentile"

    def test_fixed(self):
        scores = np.array([0.1, 0.5, 0.9])
        cfg = ThresholdConfig(method="fixed", fixed_value=0.3)
        r = select_threshold(scores, cfg)
        assert r.threshold == pytest.approx(0.3)

    def test_f1_no_labels_raises(self):
        scores = np.array([0.5])
        cfg = ThresholdConfig(method="f1")
        with pytest.raises(ValueError):
            select_threshold(scores, cfg, labels=None)

    def test_f1_with_labels(self):
        scores = np.array([0.2, 0.8])
        labels = np.array([0, 1])
        cfg = ThresholdConfig(method="f1")
        r = select_threshold(scores, cfg, labels=labels)
        assert r.method == "f1"


# ─── apply_threshold ────────────────────────────────────────────────────────

class TestApplyThresholdExtra:
    def test_basic(self):
        scores = np.array([0.1, 0.5, 0.9])
        r = ThresholdResult(threshold=0.5, method="fixed",
                            n_above=2, n_below=1, n_total=3)
        mask = apply_threshold(scores, r)
        assert mask.sum() == 2


# ─── batch_select_thresholds ────────────────────────────────────────────────

class TestBatchSelectExtra:
    def test_length(self):
        arrays = [np.array([0.1, 0.9]), np.array([0.5])]
        results = batch_select_thresholds(arrays)
        assert len(results) == 2

    def test_custom_config(self):
        arrays = [np.array([0.1, 0.5, 0.9])]
        cfg = ThresholdConfig(method="fixed", fixed_value=0.3)
        results = batch_select_thresholds(arrays, cfg)
        assert results[0].threshold == pytest.approx(0.3)
