"""Tests for puzzle_reconstruction.utils.illum_layout_records."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.illum_layout_records import (
    IllumNormRecord,
    ImageStatsRecord,
    KeypointRecord,
    LayoutCellRecord,
    make_illum_norm_record,
    make_layout_cell_record,
)

np.random.seed(42)


# ── IllumNormRecord ───────────────────────────────────────────────────────────

def test_illum_norm_was_bright_true():
    rec = IllumNormRecord(fragment_id=0, method="hist_eq",
                          original_mean=160.0, original_std=40.0)
    assert rec.was_bright is True


def test_illum_norm_was_bright_false():
    rec = IllumNormRecord(fragment_id=0, method="hist_eq",
                          original_mean=120.0, original_std=40.0)
    assert rec.was_bright is False


def test_illum_norm_was_dark_true():
    rec = IllumNormRecord(fragment_id=0, method="clahe",
                          original_mean=70.0, original_std=30.0)
    assert rec.was_dark is True


def test_illum_norm_was_dark_false():
    rec = IllumNormRecord(fragment_id=0, method="clahe",
                          original_mean=100.0, original_std=30.0)
    assert rec.was_dark is False


def test_illum_norm_contrast_ratio():
    rec = IllumNormRecord(fragment_id=1, method="linear",
                          original_mean=100.0, original_std=30.0,
                          target_std=60.0)
    assert rec.contrast_ratio == pytest.approx(60.0 / 30.0)


def test_illum_norm_contrast_ratio_zero_std():
    rec = IllumNormRecord(fragment_id=2, method="linear",
                          original_mean=100.0, original_std=0.0)
    assert rec.contrast_ratio == pytest.approx(0.0)


def test_make_illum_norm_record():
    rec = make_illum_norm_record(3, "hist_eq", 90.0, 25.0,
                                 target_mean=128.0, target_std=60.0)
    assert isinstance(rec, IllumNormRecord)
    assert rec.fragment_id == 3
    assert rec.original_mean == pytest.approx(90.0)
    assert rec.target_mean == pytest.approx(128.0)


# ── ImageStatsRecord ──────────────────────────────────────────────────────────

def test_image_stats_is_sharp_true():
    rec = ImageStatsRecord(fragment_id=0, mean=128.0, std=45.0,
                           entropy=5.0, sharpness=600.0, n_pixels=10000)
    assert rec.is_sharp is True


def test_image_stats_is_sharp_false():
    rec = ImageStatsRecord(fragment_id=0, mean=128.0, std=45.0,
                           entropy=5.0, sharpness=400.0, n_pixels=10000)
    assert rec.is_sharp is False


def test_image_stats_is_high_contrast_true():
    rec = ImageStatsRecord(fragment_id=1, mean=100.0, std=60.0,
                           entropy=4.5, sharpness=700.0, n_pixels=5000)
    assert rec.is_high_contrast is True


def test_image_stats_is_high_contrast_false():
    rec = ImageStatsRecord(fragment_id=1, mean=100.0, std=40.0,
                           entropy=4.5, sharpness=700.0, n_pixels=5000)
    assert rec.is_high_contrast is False


def test_image_stats_is_informative_true():
    rec = ImageStatsRecord(fragment_id=2, mean=110.0, std=50.0,
                           entropy=5.5, sharpness=800.0, n_pixels=8000)
    assert rec.is_informative is True


def test_image_stats_is_informative_false():
    rec = ImageStatsRecord(fragment_id=2, mean=110.0, std=50.0,
                           entropy=3.0, sharpness=800.0, n_pixels=8000)
    assert rec.is_informative is False


# ── KeypointRecord ────────────────────────────────────────────────────────────

def test_keypoint_is_well_textured_true():
    rec = KeypointRecord(fragment_id=0, n_keypoints=25)
    assert rec.is_well_textured is True


def test_keypoint_is_well_textured_false():
    rec = KeypointRecord(fragment_id=0, n_keypoints=10)
    assert rec.is_well_textured is False


def test_keypoint_has_good_match_true():
    rec = KeypointRecord(fragment_id=0, n_keypoints=30, match_score=0.7)
    assert rec.has_good_match is True


def test_keypoint_has_good_match_false():
    rec = KeypointRecord(fragment_id=0, n_keypoints=30, match_score=0.3)
    assert rec.has_good_match is False


def test_keypoint_default_detector():
    rec = KeypointRecord(fragment_id=5, n_keypoints=20)
    assert rec.detector == "orb"


# ── LayoutCellRecord ──────────────────────────────────────────────────────────

def test_layout_cell_area():
    rec = LayoutCellRecord(fragment_idx=0, x=10.0, y=20.0,
                           width=50.0, height=30.0)
    assert rec.area == pytest.approx(1500.0)


def test_layout_cell_center():
    rec = LayoutCellRecord(fragment_idx=1, x=0.0, y=0.0,
                           width=100.0, height=60.0)
    cx, cy = rec.center
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(30.0)


def test_layout_cell_is_rotated_true():
    rec = LayoutCellRecord(fragment_idx=2, x=0.0, y=0.0,
                           width=100.0, height=100.0, rotation=45.0)
    assert rec.is_rotated is True


def test_layout_cell_is_rotated_false_zero():
    rec = LayoutCellRecord(fragment_idx=3, x=0.0, y=0.0,
                           width=100.0, height=100.0, rotation=0.0)
    assert rec.is_rotated is False


def test_layout_cell_is_rotated_false_90():
    rec = LayoutCellRecord(fragment_idx=4, x=0.0, y=0.0,
                           width=100.0, height=100.0, rotation=90.0)
    assert rec.is_rotated is False


def test_make_layout_cell_record():
    rec = make_layout_cell_record(5, 10.0, 20.0, 80.0, 60.0,
                                  rotation=30.0, confidence=0.9)
    assert isinstance(rec, LayoutCellRecord)
    assert rec.fragment_idx == 5
    assert rec.confidence == pytest.approx(0.9)
    assert rec.rotation == pytest.approx(30.0)
