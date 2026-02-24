"""Extra tests for puzzle_reconstruction/utils/illum_layout_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.illum_layout_records import (
    IllumNormRecord,
    ImageStatsRecord,
    KeypointRecord,
    LayoutCellRecord,
    make_illum_norm_record,
    make_layout_cell_record,
)


# ─── IllumNormRecord ──────────────────────────────────────────────────────────

class TestIllumNormRecordExtra:
    def test_stores_fragment_id(self):
        r = IllumNormRecord(fragment_id=3, method="clahe",
                             original_mean=100.0, original_std=40.0)
        assert r.fragment_id == 3

    def test_was_bright_true(self):
        r = IllumNormRecord(fragment_id=0, method="norm",
                             original_mean=160.0, original_std=30.0)
        assert r.was_bright is True

    def test_was_bright_false(self):
        r = IllumNormRecord(fragment_id=0, method="norm",
                             original_mean=100.0, original_std=30.0)
        assert r.was_bright is False

    def test_was_dark_true(self):
        r = IllumNormRecord(fragment_id=0, method="norm",
                             original_mean=60.0, original_std=20.0)
        assert r.was_dark is True

    def test_was_dark_false(self):
        r = IllumNormRecord(fragment_id=0, method="norm",
                             original_mean=120.0, original_std=20.0)
        assert r.was_dark is False

    def test_contrast_ratio(self):
        r = IllumNormRecord(fragment_id=0, method="norm",
                             original_mean=100.0, original_std=30.0,
                             target_std=60.0)
        assert r.contrast_ratio == pytest.approx(2.0)

    def test_contrast_ratio_zero_std(self):
        r = IllumNormRecord(fragment_id=0, method="norm",
                             original_mean=100.0, original_std=0.0)
        assert r.contrast_ratio == pytest.approx(0.0)


# ─── ImageStatsRecord ─────────────────────────────────────────────────────────

class TestImageStatsRecordExtra:
    def test_stores_fragment_id(self):
        r = ImageStatsRecord(fragment_id=5, mean=128.0, std=50.0,
                              entropy=5.0, sharpness=600.0, n_pixels=1000)
        assert r.fragment_id == 5

    def test_is_sharp_true(self):
        r = ImageStatsRecord(fragment_id=0, mean=100.0, std=50.0,
                              entropy=5.0, sharpness=600.0, n_pixels=100)
        assert r.is_sharp is True

    def test_is_sharp_false(self):
        r = ImageStatsRecord(fragment_id=0, mean=100.0, std=50.0,
                              entropy=5.0, sharpness=100.0, n_pixels=100)
        assert r.is_sharp is False

    def test_is_high_contrast_true(self):
        r = ImageStatsRecord(fragment_id=0, mean=100.0, std=60.0,
                              entropy=5.0, sharpness=100.0, n_pixels=100)
        assert r.is_high_contrast is True

    def test_is_informative_true(self):
        r = ImageStatsRecord(fragment_id=0, mean=100.0, std=50.0,
                              entropy=5.0, sharpness=100.0, n_pixels=100)
        assert r.is_informative is True

    def test_is_informative_false(self):
        r = ImageStatsRecord(fragment_id=0, mean=100.0, std=50.0,
                              entropy=3.0, sharpness=100.0, n_pixels=100)
        assert r.is_informative is False


# ─── KeypointRecord ───────────────────────────────────────────────────────────

class TestKeypointRecordExtra:
    def test_stores_fragment_id(self):
        r = KeypointRecord(fragment_id=7, n_keypoints=30)
        assert r.fragment_id == 7

    def test_is_well_textured_true(self):
        r = KeypointRecord(fragment_id=0, n_keypoints=25)
        assert r.is_well_textured is True

    def test_is_well_textured_false(self):
        r = KeypointRecord(fragment_id=0, n_keypoints=10)
        assert r.is_well_textured is False

    def test_has_good_match_true(self):
        r = KeypointRecord(fragment_id=0, n_keypoints=10, match_score=0.8)
        assert r.has_good_match is True

    def test_has_good_match_false(self):
        r = KeypointRecord(fragment_id=0, n_keypoints=10, match_score=0.3)
        assert r.has_good_match is False


# ─── LayoutCellRecord ─────────────────────────────────────────────────────────

class TestLayoutCellRecordExtra:
    def test_stores_fragment_idx(self):
        r = LayoutCellRecord(fragment_idx=3, x=0.0, y=0.0,
                              width=10.0, height=5.0)
        assert r.fragment_idx == 3

    def test_area(self):
        r = LayoutCellRecord(fragment_idx=0, x=0.0, y=0.0,
                              width=8.0, height=4.0)
        assert r.area == pytest.approx(32.0)

    def test_center(self):
        r = LayoutCellRecord(fragment_idx=0, x=0.0, y=0.0,
                              width=10.0, height=4.0)
        cx, cy = r.center
        assert cx == pytest.approx(5.0) and cy == pytest.approx(2.0)

    def test_is_rotated_true(self):
        r = LayoutCellRecord(fragment_idx=0, x=0.0, y=0.0,
                              width=10.0, height=5.0, rotation=45.0)
        assert r.is_rotated is True

    def test_is_rotated_false_at_zero(self):
        r = LayoutCellRecord(fragment_idx=0, x=0.0, y=0.0,
                              width=10.0, height=5.0, rotation=0.0)
        assert r.is_rotated is False

    def test_is_rotated_false_at_90(self):
        r = LayoutCellRecord(fragment_idx=0, x=0.0, y=0.0,
                              width=10.0, height=5.0, rotation=90.0)
        assert r.is_rotated is False


# ─── make_illum_norm_record ───────────────────────────────────────────────────

class TestMakeIllumNormRecordExtra:
    def test_returns_record(self):
        r = make_illum_norm_record(0, "clahe", 100.0, 30.0)
        assert isinstance(r, IllumNormRecord)

    def test_values_stored(self):
        r = make_illum_norm_record(5, "norm", 80.0, 20.0,
                                    target_mean=128.0, target_std=60.0)
        assert r.fragment_id == 5 and r.original_mean == pytest.approx(80.0)


# ─── make_layout_cell_record ──────────────────────────────────────────────────

class TestMakeLayoutCellRecordExtra:
    def test_returns_record(self):
        r = make_layout_cell_record(0, 0.0, 0.0, 10.0, 5.0)
        assert isinstance(r, LayoutCellRecord)

    def test_values_stored(self):
        r = make_layout_cell_record(3, 5.0, 10.0, 20.0, 15.0,
                                     rotation=45.0, confidence=0.8)
        assert r.fragment_idx == 3 and r.rotation == pytest.approx(45.0)
        assert r.confidence == pytest.approx(0.8)
