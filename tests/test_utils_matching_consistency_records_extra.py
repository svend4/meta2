"""Extra tests for puzzle_reconstruction/utils/matching_consistency_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.matching_consistency_records import (
    BoundaryMatchRecord,
    ColorMatchRecord,
    ConsistencyCheckRecord,
    ColorHistogramRecord,
    make_boundary_match_record,
    make_consistency_check_record,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _boundary(idx1=0, idx2=1, side1=0, side2=1,
              hausdorff=0.8, chamfer=0.7, frechet=0.75, total=0.78) -> BoundaryMatchRecord:
    return BoundaryMatchRecord(idx1=idx1, idx2=idx2, side1=side1, side2=side2,
                               hausdorff_score=hausdorff, chamfer_score=chamfer,
                               frechet_score=frechet, total_score=total)


# ─── BoundaryMatchRecord ──────────────────────────────────────────────────────

class TestBoundaryMatchRecordExtra:
    def test_pair_key_ordered(self):
        r = _boundary(idx1=3, idx2=1)
        assert r.pair_key == (1, 3)

    def test_pair_key_same_when_ordered(self):
        r = _boundary(idx1=0, idx2=2)
        assert r.pair_key == (0, 2)

    def test_is_good_match_true(self):
        r = _boundary(total=0.8)
        assert r.is_good_match is True

    def test_is_good_match_false(self):
        r = _boundary(total=0.5)
        assert r.is_good_match is False

    def test_is_good_match_boundary(self):
        r = _boundary(total=0.7)
        assert r.is_good_match is True

    def test_stores_n_points(self):
        r = BoundaryMatchRecord(idx1=0, idx2=1, side1=0, side2=1,
                                hausdorff_score=0.8, chamfer_score=0.7,
                                frechet_score=0.75, total_score=0.78,
                                n_points=30)
        assert r.n_points == 30

    def test_default_n_points(self):
        assert _boundary().n_points == 20


# ─── ColorMatchRecord ─────────────────────────────────────────────────────────

class TestColorMatchRecordExtra:
    def test_is_compatible_true(self):
        r = ColorMatchRecord(idx1=0, idx2=1, score=0.7, hist_score=0.8,
                             moment_score=0.6, profile_score=0.75)
        assert r.is_compatible is True

    def test_is_compatible_false(self):
        r = ColorMatchRecord(idx1=0, idx2=1, score=0.4, hist_score=0.3,
                             moment_score=0.5, profile_score=0.4)
        assert r.is_compatible is False

    def test_default_colorspace(self):
        r = ColorMatchRecord(idx1=0, idx2=1, score=0.7, hist_score=0.8,
                             moment_score=0.6, profile_score=0.75)
        assert r.colorspace == "hsv"

    def test_default_metric(self):
        r = ColorMatchRecord(idx1=0, idx2=1, score=0.7, hist_score=0.8,
                             moment_score=0.6, profile_score=0.75)
        assert r.metric == "bhatt"

    def test_stores_scores(self):
        r = ColorMatchRecord(idx1=0, idx2=1, score=0.65, hist_score=0.8,
                             moment_score=0.6, profile_score=0.75)
        assert r.score == pytest.approx(0.65)


# ─── ConsistencyCheckRecord ───────────────────────────────────────────────────

class TestConsistencyCheckRecordExtra:
    def test_is_consistent_true(self):
        r = ConsistencyCheckRecord(n_fragments=3, n_violations=0, score=0.9)
        assert r.is_consistent is True

    def test_is_consistent_false(self):
        r = ConsistencyCheckRecord(n_fragments=3, n_violations=2, score=0.5)
        assert r.is_consistent is False

    def test_mean_score_all_ones(self):
        r = ConsistencyCheckRecord(n_fragments=3, n_violations=0, score=1.0,
                                   line_spacing_score=1.0, char_height_score=1.0,
                                   text_angle_score=1.0, margin_align_score=1.0)
        assert r.mean_score == pytest.approx(1.0)

    def test_mean_score_mixed(self):
        r = ConsistencyCheckRecord(n_fragments=2, n_violations=0, score=0.75,
                                   line_spacing_score=0.5, char_height_score=0.5,
                                   text_angle_score=1.0, margin_align_score=1.0)
        assert r.mean_score == pytest.approx(0.75)

    def test_default_sub_scores(self):
        r = ConsistencyCheckRecord(n_fragments=5, n_violations=0, score=0.8)
        assert r.line_spacing_score == pytest.approx(1.0)


# ─── ColorHistogramRecord ─────────────────────────────────────────────────────

class TestColorHistogramRecordExtra:
    def test_stores_bins(self):
        r = ColorHistogramRecord(bins=32, colorspace="rgb", n_channels=3,
                                 histogram_length=96, min_value=0.0,
                                 max_value=255.0, mean_value=127.0)
        assert r.bins == 32

    def test_stores_colorspace(self):
        r = ColorHistogramRecord(bins=16, colorspace="hsv", n_channels=3,
                                 histogram_length=48, min_value=0.0,
                                 max_value=1.0, mean_value=0.5)
        assert r.colorspace == "hsv"


# ─── make_boundary_match_record ───────────────────────────────────────────────

class TestMakeBoundaryMatchRecordExtra:
    def test_returns_record(self):
        r = make_boundary_match_record(0, 1, 0, 1, 0.8, 0.7, 0.75, 0.78)
        assert isinstance(r, BoundaryMatchRecord)

    def test_values_stored(self):
        r = make_boundary_match_record(2, 3, 1, 0, 0.9, 0.85, 0.88, 0.87)
        assert r.idx1 == 2 and r.idx2 == 3
        assert r.hausdorff_score == pytest.approx(0.9)

    def test_default_n_points(self):
        r = make_boundary_match_record(0, 1, 0, 1, 0.8, 0.7, 0.75, 0.78)
        assert r.n_points == 20

    def test_custom_max_dist(self):
        r = make_boundary_match_record(0, 1, 0, 1, 0.8, 0.7, 0.75, 0.78,
                                       max_dist=50.0)
        assert r.max_dist == pytest.approx(50.0)


# ─── make_consistency_check_record ────────────────────────────────────────────

class TestMakeConsistencyCheckRecordExtra:
    def test_returns_record(self):
        r = make_consistency_check_record(5, 0, 0.9)
        assert isinstance(r, ConsistencyCheckRecord)

    def test_method_scores_applied(self):
        ms = {"line_spacing": 0.8, "char_height": 0.6,
              "text_angle": 0.9, "margin_align": 0.7}
        r = make_consistency_check_record(3, 1, 0.75, ms)
        assert r.line_spacing_score == pytest.approx(0.8)
        assert r.char_height_score == pytest.approx(0.6)

    def test_none_method_scores_defaults(self):
        r = make_consistency_check_record(2, 0, 1.0)
        assert r.line_spacing_score == pytest.approx(1.0)

    def test_n_fragments_stored(self):
        r = make_consistency_check_record(7, 2, 0.6)
        assert r.n_fragments == 7
