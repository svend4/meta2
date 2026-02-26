"""Tests for puzzle_reconstruction.utils.score_seam_utils."""
import pytest
import numpy as np

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

np.random.seed(13)


# ─── NormalizationRecord ─────────────────────────────────────────────────────

def test_norm_record_creation():
    r = NormalizationRecord(method="minmax", n_scores=10)
    assert r.method == "minmax"
    assert r.n_scores == 10
    assert r.min_val == pytest.approx(0.0)
    assert r.max_val == pytest.approx(1.0)


def test_norm_record_invalid_method():
    with pytest.raises(ValueError):
        NormalizationRecord(method="unknown", n_scores=5)


def test_norm_record_negative_n_scores():
    with pytest.raises(ValueError):
        NormalizationRecord(method="minmax", n_scores=-1)


def test_norm_record_value_range():
    r = NormalizationRecord(method="zscore", n_scores=5,
                             min_val=0.2, max_val=0.8)
    assert r.value_range == pytest.approx(0.6)


def test_norm_record_value_range_zero():
    r = NormalizationRecord(method="rank", n_scores=3,
                             min_val=0.5, max_val=0.5)
    assert r.value_range == pytest.approx(0.0)


def test_norm_record_label():
    r = NormalizationRecord(method="minmax", n_scores=5, label="test")
    assert r.label == "test"


# ─── ScoreCalibrationRecord ──────────────────────────────────────────────────

def test_calibration_record_default():
    r = ScoreCalibrationRecord(n_reference=100, n_target=50)
    assert r.shift == pytest.approx(0.0)
    assert r.scale == pytest.approx(1.0)
    assert r.is_identity is True


def test_calibration_record_non_identity():
    r = ScoreCalibrationRecord(n_reference=100, n_target=50,
                                shift=0.1, scale=1.5)
    assert r.is_identity is False


def test_calibration_record_fields():
    r = ScoreCalibrationRecord(n_reference=20, n_target=30,
                                shift=-0.2, scale=0.8, label="cal")
    assert r.n_reference == 20
    assert r.n_target == 30
    assert r.label == "cal"


# ─── GradientRunRecord ───────────────────────────────────────────────────────

def test_gradient_run_record_basic():
    r = GradientRunRecord(n_images=5, kernel="sobel", ksize=3)
    assert r.n_images == 5
    assert r.kernel == "sobel"
    assert r.ksize == 3
    assert r.has_data is True


def test_gradient_run_record_zero_images():
    r = GradientRunRecord(n_images=0, kernel="scharr", ksize=3)
    assert r.has_data is False


def test_gradient_run_record_invalid_n_images():
    with pytest.raises(ValueError):
        GradientRunRecord(n_images=-1, kernel="sobel", ksize=3)


def test_gradient_run_record_invalid_ksize():
    with pytest.raises(ValueError):
        GradientRunRecord(n_images=5, kernel="sobel", ksize=0)


def test_gradient_run_record_mean_energy():
    r = GradientRunRecord(n_images=3, kernel="laplacian", ksize=5,
                           mean_energy=42.0)
    assert r.mean_energy == pytest.approx(42.0)


# ─── OrientationHistogramRecord ──────────────────────────────────────────────

def test_orientation_histogram_basic():
    r = OrientationHistogramRecord(n_bins=8)
    assert r.n_bins == 8
    assert r.dominant_bin is None


def test_orientation_histogram_dominant_bin():
    hist = [0.1, 0.2, 0.5, 0.1, 0.05, 0.02, 0.01, 0.02]
    r = OrientationHistogramRecord(n_bins=8, histogram=hist)
    assert r.dominant_bin == 2


def test_orientation_histogram_invalid_length():
    with pytest.raises(ValueError):
        OrientationHistogramRecord(n_bins=8, histogram=[0.1, 0.2])


def test_orientation_histogram_is_normalized_true():
    hist = [0.125] * 8
    r = OrientationHistogramRecord(n_bins=8, histogram=hist)
    assert r.is_normalized is True


def test_orientation_histogram_is_normalized_false():
    hist = [0.5] * 8  # sums to 4
    r = OrientationHistogramRecord(n_bins=8, histogram=hist)
    assert r.is_normalized is False


def test_orientation_histogram_empty():
    r = OrientationHistogramRecord(n_bins=8)
    assert r.is_normalized is True  # empty → trivially normalized


# ─── SeamRunRecord ───────────────────────────────────────────────────────────

def test_seam_run_record_basic():
    r = SeamRunRecord(n_pairs=5, mean_quality=0.7)
    assert r.n_pairs == 5
    assert r.has_pairs is True


def test_seam_run_record_zero_pairs():
    r = SeamRunRecord(n_pairs=0)
    assert r.has_pairs is False


def test_seam_run_record_invalid_n_pairs():
    with pytest.raises(ValueError):
        SeamRunRecord(n_pairs=-1)


def test_seam_run_record_invalid_mean_quality():
    with pytest.raises(ValueError):
        SeamRunRecord(n_pairs=5, mean_quality=1.5)


def test_seam_run_record_quality_range():
    r = SeamRunRecord(n_pairs=10, mean_quality=0.5,
                      min_quality=0.1, max_quality=0.9)
    assert r.min_quality == pytest.approx(0.1)
    assert r.max_quality == pytest.approx(0.9)


# ─── SeamScoreMatrix ─────────────────────────────────────────────────────────

def test_seam_score_matrix_get_existing():
    m = SeamScoreMatrix(n_fragments=4, scores={(0, 1): 0.8, (2, 3): 0.6})
    assert m.get(0, 1) == pytest.approx(0.8)


def test_seam_score_matrix_get_mirror():
    m = SeamScoreMatrix(n_fragments=4, scores={(0, 1): 0.8})
    assert m.get(1, 0) == pytest.approx(0.8)


def test_seam_score_matrix_get_default():
    m = SeamScoreMatrix(n_fragments=4)
    assert m.get(0, 1, default=0.5) == pytest.approx(0.5)


def test_seam_score_matrix_n_scored_pairs():
    m = SeamScoreMatrix(n_fragments=5, scores={(0, 1): 0.9, (2, 4): 0.3})
    assert m.n_scored_pairs == 2


def test_seam_score_matrix_empty():
    m = SeamScoreMatrix(n_fragments=3)
    assert m.n_scored_pairs == 0


# ─── AggregationRunRecord ────────────────────────────────────────────────────

def test_aggregation_run_record_valid():
    r = AggregationRunRecord(method="weighted_avg", n_items=10, n_channels=3)
    assert r.method == "weighted_avg"
    assert r.is_empty is False


def test_aggregation_run_record_invalid_method():
    with pytest.raises(ValueError):
        AggregationRunRecord(method="bad_method", n_items=5, n_channels=2)


def test_aggregation_run_record_empty():
    r = AggregationRunRecord(method="harmonic", n_items=0, n_channels=2)
    assert r.is_empty is True


def test_aggregation_run_record_all_methods():
    for method in ("weighted_avg", "harmonic", "min", "max"):
        r = AggregationRunRecord(method=method, n_items=5, n_channels=2)
        assert r.method == method


# ─── make_normalization_record ───────────────────────────────────────────────

def test_make_normalization_record_basic():
    scores = [0.1, 0.5, 0.9]
    r = make_normalization_record("minmax", scores)
    assert r.n_scores == 3
    assert r.min_val == pytest.approx(0.1)
    assert r.max_val == pytest.approx(0.9)


def test_make_normalization_record_empty():
    r = make_normalization_record("rank", [])
    assert r.n_scores == 0
    assert r.min_val == pytest.approx(0.0)
    assert r.max_val == pytest.approx(1.0)


def test_make_normalization_record_label():
    r = make_normalization_record("zscore", [0.5, 0.7], label="myrun")
    assert r.label == "myrun"


# ─── make_seam_run_record ────────────────────────────────────────────────────

def test_make_seam_run_record_basic():
    r = make_seam_run_record([0.3, 0.6, 0.9])
    assert r.n_pairs == 3
    assert r.mean_quality == pytest.approx(0.6)
    assert r.min_quality == pytest.approx(0.3)
    assert r.max_quality == pytest.approx(0.9)


def test_make_seam_run_record_empty():
    r = make_seam_run_record([])
    assert r.n_pairs == 0


def test_make_seam_run_record_label():
    r = make_seam_run_record([0.5], label="batch1")
    assert r.label == "batch1"
