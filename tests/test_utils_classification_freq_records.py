"""Tests for puzzle_reconstruction.utils.classification_freq_records."""
import pytest
import numpy as np
from puzzle_reconstruction.utils.classification_freq_records import (
    FragmentClassifyRecord,
    FragmentMapRecord,
    FragmentValidationRecord,
    FreqDescriptorRecord,
    make_fragment_classify_record,
    make_freq_descriptor_record,
)

np.random.seed(42)


# ── 1. FragmentClassifyRecord basic ──────────────────────────────────────────
def test_classify_record_basic():
    r = FragmentClassifyRecord(
        fragment_id=1, fragment_type="corner", confidence=0.9,
        has_text=False, text_lines=0, n_straight_sides=2,
    )
    assert r.fragment_id == 1
    assert r.fragment_type == "corner"
    assert r.confidence == 0.9
    assert r.n_straight_sides == 2


# ── 2. FragmentClassifyRecord properties ─────────────────────────────────────
def test_classify_is_corner():
    r = FragmentClassifyRecord(1, "corner", 0.9, False, 0)
    assert r.is_corner is True
    assert r.is_edge is False
    assert r.is_inner is False


def test_classify_is_edge():
    r = FragmentClassifyRecord(2, "edge", 0.8, False, 0)
    assert r.is_edge is True
    assert r.is_corner is False


def test_classify_is_inner():
    r = FragmentClassifyRecord(3, "inner", 0.7, True, 2)
    assert r.is_inner is True
    assert r.is_corner is False


# ── 5. make_fragment_classify_record ─────────────────────────────────────────
def test_make_classify_record():
    r = make_fragment_classify_record(
        fragment_id=10, fragment_type="edge", confidence=0.85,
        has_text=True, text_lines=3, straight_sides=[0, 1],
    )
    assert r.fragment_id == 10
    assert r.fragment_type == "edge"
    assert r.n_straight_sides == 2  # len(straight_sides)
    assert r.has_text is True
    assert r.text_lines == 3


def test_make_classify_record_empty_sides():
    r = make_fragment_classify_record(0, "inner", 0.6, False, 0, [])
    assert r.n_straight_sides == 0


# ── 7. FragmentMapRecord ──────────────────────────────────────────────────────
def test_map_record_basic():
    rec = FragmentMapRecord(n_fragments=10, n_zones=8, n_assigned=6,
                             canvas_w=500, canvas_h=400)
    assert rec.n_fragments == 10
    assert rec.n_zones == 8


def test_map_record_coverage_ratio():
    rec = FragmentMapRecord(n_fragments=10, n_zones=8, n_assigned=6,
                             canvas_w=500, canvas_h=400)
    assert abs(rec.coverage_ratio - 6/8) < 1e-9


def test_map_record_coverage_zero_zones():
    rec = FragmentMapRecord(n_fragments=5, n_zones=0, n_assigned=0,
                             canvas_w=100, canvas_h=100)
    assert rec.coverage_ratio == 0.0


def test_map_record_assignment_ratio():
    rec = FragmentMapRecord(n_fragments=10, n_zones=8, n_assigned=6,
                             canvas_w=500, canvas_h=400)
    assert abs(rec.assignment_ratio - 0.6) < 1e-9


def test_map_record_assignment_ratio_zero_frags():
    rec = FragmentMapRecord(n_fragments=0, n_zones=5, n_assigned=0,
                             canvas_w=100, canvas_h=100)
    assert rec.assignment_ratio == 0.0


# ── 12. FragmentValidationRecord ─────────────────────────────────────────────
def test_validation_record_basic():
    v = FragmentValidationRecord(
        fragment_id=5, passed=True, n_issues=0, n_errors=0, n_warnings=0,
        width=100.0, height=80.0, coverage=0.9,
    )
    assert v.passed is True
    assert abs(v.aspect_ratio - 80.0/100.0) < 1e-9


def test_validation_record_zero_height():
    v = FragmentValidationRecord(1, False, 2, 1, 1, width=50.0, height=0.0)
    assert v.aspect_ratio == 0.0


# ── 14. FreqDescriptorRecord ──────────────────────────────────────────────────
def test_freq_descriptor_record_basic():
    r = FreqDescriptorRecord(
        fragment_id=1, n_bands=8, centroid=0.4, entropy=2.5,
        dominant_band=3, high_freq_ratio=0.6,
    )
    assert r.is_high_frequency is True
    assert r.is_smooth is False


def test_freq_descriptor_smooth():
    r = FreqDescriptorRecord(
        fragment_id=2, n_bands=8, centroid=0.2, entropy=1.0,
        dominant_band=0, high_freq_ratio=0.3,
    )
    assert r.is_smooth is True
    assert r.is_high_frequency is False


# ── 16. make_freq_descriptor_record ──────────────────────────────────────────
def test_make_freq_descriptor_record():
    r = make_freq_descriptor_record(
        fragment_id=7, n_bands=16, centroid=0.5, entropy=3.0,
        dominant_band=7, high_freq_ratio=0.55,
    )
    assert r.fragment_id == 7
    assert r.n_bands == 16
    assert r.dominant_band == 7
    assert r.high_freq_ratio == 0.55


# ── 17. coverage_ratio capped at 1 ───────────────────────────────────────────
def test_map_record_coverage_capped():
    rec = FragmentMapRecord(n_fragments=5, n_zones=3, n_assigned=5,
                             canvas_w=100, canvas_h=100)
    assert rec.coverage_ratio == 1.0
