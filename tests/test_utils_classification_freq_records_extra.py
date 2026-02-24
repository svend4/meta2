"""Extra tests for puzzle_reconstruction/utils/classification_freq_records.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.classification_freq_records import (
    FragmentClassifyRecord,
    FragmentMapRecord,
    FragmentValidationRecord,
    FreqDescriptorRecord,
    make_fragment_classify_record,
    make_freq_descriptor_record,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _classify(fid=0, ftype="corner", conf=0.9, has_text=False, text_lines=0) -> FragmentClassifyRecord:
    return FragmentClassifyRecord(
        fragment_id=fid,
        fragment_type=ftype,
        confidence=conf,
        has_text=has_text,
        text_lines=text_lines,
    )


def _map_rec(n_frags=4, n_zones=4, n_assigned=4, cw=100, ch=100) -> FragmentMapRecord:
    return FragmentMapRecord(
        n_fragments=n_frags, n_zones=n_zones,
        n_assigned=n_assigned, canvas_w=cw, canvas_h=ch,
    )


def _val_rec(fid=0, passed=True, n_issues=0, n_errors=0, n_warnings=0,
             width=50.0, height=50.0) -> FragmentValidationRecord:
    return FragmentValidationRecord(
        fragment_id=fid, passed=passed,
        n_issues=n_issues, n_errors=n_errors, n_warnings=n_warnings,
        width=width, height=height,
    )


def _freq_rec(fid=0, n_bands=8, centroid=0.2, entropy=1.5,
              dominant_band=2, hfr=0.3) -> FreqDescriptorRecord:
    return FreqDescriptorRecord(
        fragment_id=fid, n_bands=n_bands, centroid=centroid,
        entropy=entropy, dominant_band=dominant_band, high_freq_ratio=hfr,
    )


# ─── FragmentClassifyRecord ───────────────────────────────────────────────────

class TestFragmentClassifyRecordExtra:
    def test_stores_fragment_id(self):
        assert _classify(fid=5).fragment_id == 5

    def test_stores_type(self):
        assert _classify(ftype="edge").fragment_type == "edge"

    def test_stores_confidence(self):
        assert _classify(conf=0.75).confidence == pytest.approx(0.75)

    def test_is_corner_true(self):
        assert _classify(ftype="corner").is_corner is True

    def test_is_corner_false(self):
        assert _classify(ftype="edge").is_corner is False

    def test_is_edge_true(self):
        assert _classify(ftype="edge").is_edge is True

    def test_is_inner_true(self):
        assert _classify(ftype="inner").is_inner is True

    def test_is_inner_false(self):
        assert _classify(ftype="corner").is_inner is False

    def test_has_text_stored(self):
        assert _classify(has_text=True).has_text is True

    def test_text_lines_stored(self):
        assert _classify(text_lines=3).text_lines == 3


# ─── FragmentMapRecord ────────────────────────────────────────────────────────

class TestFragmentMapRecordExtra:
    def test_stores_n_fragments(self):
        assert _map_rec(n_frags=6).n_fragments == 6

    def test_stores_n_zones(self):
        assert _map_rec(n_zones=8).n_zones == 8

    def test_coverage_ratio_full(self):
        r = _map_rec(n_zones=4, n_assigned=4)
        assert r.coverage_ratio == pytest.approx(1.0)

    def test_coverage_ratio_zero_zones(self):
        r = _map_rec(n_zones=0, n_assigned=0)
        assert r.coverage_ratio == pytest.approx(0.0)

    def test_assignment_ratio_full(self):
        r = _map_rec(n_frags=4, n_assigned=4)
        assert r.assignment_ratio == pytest.approx(1.0)

    def test_assignment_ratio_zero_frags(self):
        r = _map_rec(n_frags=0, n_assigned=0)
        assert r.assignment_ratio == pytest.approx(0.0)

    def test_canvas_dims_stored(self):
        r = _map_rec(cw=200, ch=150)
        assert r.canvas_w == 200 and r.canvas_h == 150

    def test_coverage_capped_at_one(self):
        r = _map_rec(n_zones=2, n_assigned=5)
        assert r.coverage_ratio <= 1.0


# ─── FragmentValidationRecord ─────────────────────────────────────────────────

class TestFragmentValidationRecordExtra:
    def test_stores_passed(self):
        assert _val_rec(passed=False).passed is False

    def test_stores_n_issues(self):
        assert _val_rec(n_issues=2).n_issues == 2

    def test_stores_n_errors(self):
        assert _val_rec(n_errors=1).n_errors == 1

    def test_stores_n_warnings(self):
        assert _val_rec(n_warnings=3).n_warnings == 3

    def test_aspect_ratio_square(self):
        r = _val_rec(width=50.0, height=50.0)
        assert r.aspect_ratio == pytest.approx(1.0)

    def test_aspect_ratio_rect(self):
        r = _val_rec(width=30.0, height=60.0)
        assert r.aspect_ratio == pytest.approx(0.5)

    def test_aspect_ratio_zero_height(self):
        r = _val_rec(width=50.0, height=0.0)
        assert r.aspect_ratio == pytest.approx(0.0)

    def test_coverage_stored(self):
        r = FragmentValidationRecord(
            fragment_id=0, passed=True,
            n_issues=0, n_errors=0, n_warnings=0,
            width=10, height=10, coverage=0.8,
        )
        assert r.coverage == pytest.approx(0.8)


# ─── FreqDescriptorRecord ─────────────────────────────────────────────────────

class TestFreqDescriptorRecordExtra:
    def test_stores_n_bands(self):
        assert _freq_rec(n_bands=16).n_bands == 16

    def test_stores_centroid(self):
        assert _freq_rec(centroid=0.45).centroid == pytest.approx(0.45)

    def test_stores_entropy(self):
        assert _freq_rec(entropy=2.0).entropy == pytest.approx(2.0)

    def test_is_high_frequency_true(self):
        assert _freq_rec(hfr=0.6).is_high_frequency is True

    def test_is_high_frequency_false(self):
        assert _freq_rec(hfr=0.4).is_high_frequency is False

    def test_is_smooth_true(self):
        assert _freq_rec(centroid=0.2).is_smooth is True

    def test_is_smooth_false(self):
        assert _freq_rec(centroid=0.5).is_smooth is False

    def test_dominant_band_stored(self):
        assert _freq_rec(dominant_band=3).dominant_band == 3


# ─── make_fragment_classify_record ────────────────────────────────────────────

class TestMakeFragmentClassifyRecordExtra:
    def test_returns_record(self):
        r = make_fragment_classify_record(0, "corner", 0.9, False, 0, [])
        assert isinstance(r, FragmentClassifyRecord)

    def test_n_straight_sides_from_list(self):
        r = make_fragment_classify_record(0, "corner", 0.9, False, 0, [1, 2])
        assert r.n_straight_sides == 2

    def test_empty_sides(self):
        r = make_fragment_classify_record(0, "inner", 0.5, False, 0, [])
        assert r.n_straight_sides == 0

    def test_values_stored(self):
        r = make_fragment_classify_record(3, "edge", 0.7, True, 2, [1])
        assert r.fragment_id == 3
        assert r.fragment_type == "edge"
        assert r.has_text is True


# ─── make_freq_descriptor_record ──────────────────────────────────────────────

class TestMakeFreqDescriptorRecordExtra:
    def test_returns_record(self):
        r = make_freq_descriptor_record(0, 8, 0.2, 1.5, 2, 0.3)
        assert isinstance(r, FreqDescriptorRecord)

    def test_values_stored(self):
        r = make_freq_descriptor_record(5, 16, 0.4, 2.0, 3, 0.7)
        assert r.fragment_id == 5
        assert r.n_bands == 16
        assert r.centroid == pytest.approx(0.4)
        assert r.entropy == pytest.approx(2.0)
        assert r.dominant_band == 3
        assert r.high_freq_ratio == pytest.approx(0.7)

    def test_is_high_frequency(self):
        r = make_freq_descriptor_record(0, 8, 0.2, 1.0, 0, 0.8)
        assert r.is_high_frequency is True
