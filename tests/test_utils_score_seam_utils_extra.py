"""Extra tests for puzzle_reconstruction/utils/score_seam_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.score_seam_utils import (
    NormalizationRecord,
    ScoreCalibrationRecord,
    GradientRunRecord,
    OrientationHistogramRecord,
    SeamRunRecord,
    SeamScoreMatrix,
    AggregationRunRecord,
    make_normalization_record,
    make_seam_run_record,
)


# ─── NormalizationRecord ─────────────────────────────────────────────────────

class TestNormalizationRecordExtra:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NormalizationRecord(method="bad", n_scores=5)

    def test_negative_n_scores_raises(self):
        with pytest.raises(ValueError):
            NormalizationRecord(method="minmax", n_scores=-1)

    def test_value_range(self):
        r = NormalizationRecord(method="minmax", n_scores=10,
                                 min_val=0.2, max_val=0.8)
        assert r.value_range == pytest.approx(0.6)

    def test_valid_methods(self):
        for m in ("minmax", "zscore", "rank"):
            NormalizationRecord(method=m, n_scores=5)


# ─── ScoreCalibrationRecord ──────────────────────────────────────────────────

class TestScoreCalibrationRecordExtra:
    def test_is_identity_true(self):
        r = ScoreCalibrationRecord(n_reference=10, n_target=10)
        assert r.is_identity is True

    def test_is_identity_false(self):
        r = ScoreCalibrationRecord(n_reference=10, n_target=10,
                                    shift=0.1, scale=1.0)
        assert r.is_identity is False


# ─── GradientRunRecord ───────────────────────────────────────────────────────

class TestGradientRunRecordExtra:
    def test_negative_n_images_raises(self):
        with pytest.raises(ValueError):
            GradientRunRecord(n_images=-1, kernel="sobel", ksize=3)

    def test_zero_ksize_raises(self):
        with pytest.raises(ValueError):
            GradientRunRecord(n_images=5, kernel="sobel", ksize=0)

    def test_has_data_true(self):
        r = GradientRunRecord(n_images=5, kernel="sobel", ksize=3)
        assert r.has_data is True

    def test_has_data_false(self):
        r = GradientRunRecord(n_images=0, kernel="sobel", ksize=3)
        assert r.has_data is False


# ─── OrientationHistogramRecord ──────────────────────────────────────────────

class TestOrientationHistogramRecordExtra:
    def test_empty_histogram_ok(self):
        r = OrientationHistogramRecord(n_bins=8, histogram=[])
        assert r.dominant_bin is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            OrientationHistogramRecord(n_bins=8, histogram=[0.5, 0.5])

    def test_dominant_bin(self):
        hist = [0.1, 0.5, 0.3, 0.1]
        r = OrientationHistogramRecord(n_bins=4, histogram=hist)
        assert r.dominant_bin == 1

    def test_is_normalized_true(self):
        hist = [0.25, 0.25, 0.25, 0.25]
        r = OrientationHistogramRecord(n_bins=4, histogram=hist)
        assert r.is_normalized is True

    def test_is_normalized_false(self):
        hist = [1.0, 1.0, 1.0, 1.0]
        r = OrientationHistogramRecord(n_bins=4, histogram=hist)
        assert r.is_normalized is False


# ─── SeamRunRecord ────────────────────────────────────────────────────────────

class TestSeamRunRecordExtra:
    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            SeamRunRecord(n_pairs=-1)

    def test_mean_quality_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SeamRunRecord(n_pairs=5, mean_quality=1.5)

    def test_has_pairs_true(self):
        r = SeamRunRecord(n_pairs=5, mean_quality=0.8)
        assert r.has_pairs is True

    def test_has_pairs_false(self):
        r = SeamRunRecord(n_pairs=0)
        assert r.has_pairs is False


# ─── SeamScoreMatrix ──────────────────────────────────────────────────────────

class TestSeamScoreMatrixExtra:
    def test_get_existing(self):
        m = SeamScoreMatrix(n_fragments=5, scores={(0, 1): 0.9})
        assert m.get(0, 1) == pytest.approx(0.9)

    def test_get_mirror(self):
        m = SeamScoreMatrix(n_fragments=5, scores={(0, 1): 0.9})
        assert m.get(1, 0) == pytest.approx(0.9)

    def test_get_missing_default(self):
        m = SeamScoreMatrix(n_fragments=5)
        assert m.get(0, 1) == pytest.approx(0.0)

    def test_n_scored_pairs(self):
        m = SeamScoreMatrix(n_fragments=5,
                             scores={(0, 1): 0.5, (2, 3): 0.7})
        assert m.n_scored_pairs == 2


# ─── AggregationRunRecord ────────────────────────────────────────────────────

class TestAggregationRunRecordExtra:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            AggregationRunRecord(method="bad", n_items=5, n_channels=3)

    def test_valid_methods(self):
        for m in ("weighted_avg", "harmonic", "min", "max"):
            AggregationRunRecord(method=m, n_items=5, n_channels=3)

    def test_is_empty(self):
        r = AggregationRunRecord(method="min", n_items=0, n_channels=3)
        assert r.is_empty is True


# ─── make_normalization_record ────────────────────────────────────────────────

class TestMakeNormalizationRecordExtra:
    def test_returns_record(self):
        r = make_normalization_record("minmax", [0.1, 0.5, 0.9])
        assert isinstance(r, NormalizationRecord)

    def test_min_max_computed(self):
        r = make_normalization_record("minmax", [0.2, 0.8])
        assert r.min_val == pytest.approx(0.2) and r.max_val == pytest.approx(0.8)

    def test_empty_scores(self):
        r = make_normalization_record("zscore", [])
        assert r.n_scores == 0


# ─── make_seam_run_record ─────────────────────────────────────────────────────

class TestMakeSeamRunRecordExtra:
    def test_returns_record(self):
        r = make_seam_run_record([0.5, 0.7, 0.9])
        assert isinstance(r, SeamRunRecord)

    def test_empty_qualities(self):
        r = make_seam_run_record([])
        assert r.n_pairs == 0

    def test_stats_computed(self):
        r = make_seam_run_record([0.3, 0.7])
        assert r.mean_quality == pytest.approx(0.5)
        assert r.min_quality == pytest.approx(0.3)
        assert r.max_quality == pytest.approx(0.7)
