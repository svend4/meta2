"""Tests for puzzle_reconstruction.utils.matching_consistency_records."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.matching_consistency_records import (
    BoundaryMatchRecord,
    ColorMatchRecord,
    ConsistencyCheckRecord,
    ColorHistogramRecord,
    make_boundary_match_record,
    make_consistency_check_record,
)

np.random.seed(42)


# ── BoundaryMatchRecord ───────────────────────────────────────────────────────

def test_boundary_match_pair_key_sorted():
    rec = BoundaryMatchRecord(idx1=5, idx2=2, side1=0, side2=1,
                              hausdorff_score=0.8, chamfer_score=0.9,
                              frechet_score=0.7, total_score=0.8)
    assert rec.pair_key == (2, 5)


def test_boundary_match_pair_key_already_sorted():
    rec = BoundaryMatchRecord(idx1=1, idx2=3, side1=0, side2=1,
                              hausdorff_score=0.8, chamfer_score=0.9,
                              frechet_score=0.7, total_score=0.8)
    assert rec.pair_key == (1, 3)


def test_boundary_match_is_good_match_true():
    rec = BoundaryMatchRecord(idx1=0, idx2=1, side1=0, side2=1,
                              hausdorff_score=0.8, chamfer_score=0.8,
                              frechet_score=0.8, total_score=0.75)
    assert rec.is_good_match is True


def test_boundary_match_is_good_match_false():
    rec = BoundaryMatchRecord(idx1=0, idx2=1, side1=0, side2=1,
                              hausdorff_score=0.5, chamfer_score=0.5,
                              frechet_score=0.5, total_score=0.5)
    assert rec.is_good_match is False


def test_boundary_match_default_n_points():
    rec = BoundaryMatchRecord(idx1=0, idx2=1, side1=0, side2=1,
                              hausdorff_score=0.5, chamfer_score=0.5,
                              frechet_score=0.5, total_score=0.5)
    assert rec.n_points == 20


def test_make_boundary_match_record():
    rec = make_boundary_match_record(0, 2, 1, 0, 0.9, 0.85, 0.8, 0.85,
                                     n_points=30, max_dist=50.0)
    assert isinstance(rec, BoundaryMatchRecord)
    assert rec.idx1 == 0
    assert rec.idx2 == 2
    assert rec.n_points == 30
    assert rec.max_dist == pytest.approx(50.0)


# ── ColorMatchRecord ──────────────────────────────────────────────────────────

def test_color_match_is_compatible_true():
    rec = ColorMatchRecord(idx1=0, idx2=1, score=0.7,
                           hist_score=0.8, moment_score=0.7,
                           profile_score=0.6)
    assert rec.is_compatible is True


def test_color_match_is_compatible_false():
    rec = ColorMatchRecord(idx1=0, idx2=1, score=0.5,
                           hist_score=0.5, moment_score=0.5,
                           profile_score=0.5)
    assert rec.is_compatible is False


def test_color_match_default_colorspace():
    rec = ColorMatchRecord(idx1=0, idx2=1, score=0.7,
                           hist_score=0.8, moment_score=0.7,
                           profile_score=0.6)
    assert rec.colorspace == "hsv"


def test_color_match_default_metric():
    rec = ColorMatchRecord(idx1=0, idx2=1, score=0.7,
                           hist_score=0.8, moment_score=0.7,
                           profile_score=0.6)
    assert rec.metric == "bhatt"


# ── ConsistencyCheckRecord ────────────────────────────────────────────────────

def test_consistency_check_is_consistent_true():
    rec = ConsistencyCheckRecord(n_fragments=5, n_violations=0, score=0.95)
    assert rec.is_consistent is True


def test_consistency_check_is_consistent_false():
    rec = ConsistencyCheckRecord(n_fragments=5, n_violations=2, score=0.7)
    assert rec.is_consistent is False


def test_consistency_check_mean_score():
    rec = ConsistencyCheckRecord(
        n_fragments=5, n_violations=0, score=0.9,
        line_spacing_score=1.0, char_height_score=0.8,
        text_angle_score=0.6, margin_align_score=0.8
    )
    assert rec.mean_score == pytest.approx((1.0 + 0.8 + 0.6 + 0.8) / 4.0)


def test_make_consistency_check_record_default_scores():
    rec = make_consistency_check_record(10, 1, 0.85)
    assert rec.line_spacing_score == pytest.approx(1.0)
    assert rec.n_fragments == 10
    assert rec.n_violations == 1


def test_make_consistency_check_record_with_method_scores():
    ms = {
        "line_spacing": 0.9,
        "char_height": 0.8,
        "text_angle": 0.7,
        "margin_align": 0.6,
    }
    rec = make_consistency_check_record(8, 0, 0.9, method_scores=ms)
    assert rec.line_spacing_score == pytest.approx(0.9)
    assert rec.char_height_score == pytest.approx(0.8)
    assert rec.text_angle_score == pytest.approx(0.7)
    assert rec.margin_align_score == pytest.approx(0.6)


# ── ColorHistogramRecord ──────────────────────────────────────────────────────

def test_color_histogram_record_fields():
    rec = ColorHistogramRecord(bins=256, colorspace="hsv", n_channels=3,
                               histogram_length=768, min_value=0.0,
                               max_value=1.0, mean_value=0.5)
    assert rec.bins == 256
    assert rec.colorspace == "hsv"
    assert rec.n_channels == 3
    assert rec.histogram_length == 768
    assert rec.min_value == pytest.approx(0.0)
    assert rec.max_value == pytest.approx(1.0)
    assert rec.mean_value == pytest.approx(0.5)


def test_color_histogram_record_rgb():
    rec = ColorHistogramRecord(bins=64, colorspace="rgb", n_channels=3,
                               histogram_length=192, min_value=0.0,
                               max_value=255.0, mean_value=127.5)
    assert rec.colorspace == "rgb"
    assert rec.bins == 64


def test_boundary_match_total_score_boundary():
    rec = BoundaryMatchRecord(idx1=0, idx2=1, side1=0, side2=1,
                              hausdorff_score=0.7, chamfer_score=0.7,
                              frechet_score=0.7, total_score=0.7)
    assert rec.is_good_match is True


def test_consistency_check_all_default_scores_mean_one():
    rec = ConsistencyCheckRecord(n_fragments=3, n_violations=0, score=1.0)
    assert rec.mean_score == pytest.approx(1.0)
