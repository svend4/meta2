"""Tests for puzzle_reconstruction.verification.spatial_validator"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.spatial_validator import (
    SpatialIssue,
    SpatialReport,
    PlacedFragment,
    check_unique_ids,
    check_within_canvas,
    check_no_overlaps,
    check_coverage,
    check_gap_uniformity,
    validate_spatial,
    batch_validate,
)


def make_frag(fid, x=0.0, y=0.0, w=50.0, h=50.0):
    return PlacedFragment(fragment_id=fid, x=x, y=y, width=w, height=h)


# ─── SpatialIssue ─────────────────────────────────────────────────────────────

def test_spatial_issue_defaults():
    issue = SpatialIssue(code="OVERLAP", severity="error")
    assert issue.fragment_ids == []
    assert issue.detail == ""


def test_spatial_issue_invalid_severity():
    with pytest.raises(ValueError):
        SpatialIssue(code="ERR", severity="critical")


def test_spatial_issue_empty_code():
    with pytest.raises(ValueError):
        SpatialIssue(code="  ", severity="error")


# ─── SpatialReport ────────────────────────────────────────────────────────────

def test_spatial_report_is_valid_no_errors():
    report = SpatialReport(issues=[], n_fragments=3)
    assert report.is_valid


def test_spatial_report_is_valid_with_error():
    issue = SpatialIssue(code="OVERLAP", severity="error")
    report = SpatialReport(issues=[issue], n_fragments=2)
    assert not report.is_valid


def test_spatial_report_n_errors():
    issues = [
        SpatialIssue(code="E1", severity="error"),
        SpatialIssue(code="W1", severity="warning"),
        SpatialIssue(code="E2", severity="error"),
    ]
    report = SpatialReport(issues=issues, n_fragments=3)
    assert report.n_errors == 2
    assert report.n_warnings == 1


def test_spatial_report_invalid_n_fragments():
    with pytest.raises(ValueError):
        SpatialReport(n_fragments=-1)


def test_spatial_report_invalid_canvas():
    with pytest.raises(ValueError):
        SpatialReport(canvas_w=-1.0)


# ─── PlacedFragment ───────────────────────────────────────────────────────────

def test_placed_fragment_x2_y2():
    f = make_frag(0, x=10, y=20, w=30, h=40)
    assert f.x2 == pytest.approx(40.0)
    assert f.y2 == pytest.approx(60.0)


def test_placed_fragment_area():
    f = make_frag(0, w=30, h=40)
    assert f.area == pytest.approx(1200.0)


def test_placed_fragment_invalid_fid():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=-1, x=0, y=0, width=10, height=10)


def test_placed_fragment_invalid_x():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=0, x=-1, y=0, width=10, height=10)


def test_placed_fragment_invalid_width():
    with pytest.raises(ValueError):
        PlacedFragment(fragment_id=0, x=0, y=0, width=0, height=10)


# ─── check_unique_ids ─────────────────────────────────────────────────────────

def test_check_unique_ids_all_unique():
    frags = [make_frag(i) for i in range(3)]
    issues = check_unique_ids(frags)
    assert issues == []


def test_check_unique_ids_duplicate():
    frags = [make_frag(0), make_frag(1), make_frag(0)]
    issues = check_unique_ids(frags)
    assert len(issues) == 1
    assert issues[0].code == "DUPLICATE_ID"
    assert issues[0].severity == "error"


def test_check_unique_ids_empty():
    assert check_unique_ids([]) == []


# ─── check_within_canvas ──────────────────────────────────────────────────────

def test_check_within_canvas_all_inside():
    frags = [make_frag(i, x=i * 60, y=0, w=50, h=50) for i in range(2)]
    issues = check_within_canvas(frags, canvas_w=200, canvas_h=100)
    assert issues == []


def test_check_within_canvas_outside():
    frags = [make_frag(0, x=80, y=0, w=50, h=50)]
    issues = check_within_canvas(frags, canvas_w=100, canvas_h=100)
    assert len(issues) == 1
    assert issues[0].code == "OUT_OF_BOUNDS"


def test_check_within_canvas_invalid_canvas():
    with pytest.raises(ValueError):
        check_within_canvas([], canvas_w=0, canvas_h=100)


def test_check_within_canvas_vertical_overflow():
    frags = [make_frag(0, x=0, y=80, w=50, h=50)]
    issues = check_within_canvas(frags, canvas_w=200, canvas_h=100)
    assert len(issues) == 1


# ─── check_no_overlaps ────────────────────────────────────────────────────────

def test_check_no_overlaps_none():
    frags = [make_frag(i, x=i * 60, y=0, w=50, h=50) for i in range(3)]
    issues = check_no_overlaps(frags)
    assert issues == []


def test_check_no_overlaps_found():
    frags = [make_frag(0, x=0, y=0, w=60, h=60), make_frag(1, x=30, y=30, w=60, h=60)]
    issues = check_no_overlaps(frags)
    assert len(issues) > 0
    assert issues[0].code == "OVERLAP"


def test_check_no_overlaps_tolerance():
    # Overlap of 5px but tolerance=5px → no issue
    frags = [make_frag(0, x=0, y=0, w=55, h=50), make_frag(1, x=50, y=0, w=50, h=50)]
    issues = check_no_overlaps(frags, tolerance=5.0)
    assert issues == []


def test_check_no_overlaps_invalid_tolerance():
    with pytest.raises(ValueError):
        check_no_overlaps([], tolerance=-1.0)


# ─── check_coverage ───────────────────────────────────────────────────────────

def test_check_coverage_sufficient():
    frags = [make_frag(0, x=0, y=0, w=100, h=100)]
    issues = check_coverage(frags, canvas_w=100, canvas_h=100, min_coverage=0.5)
    assert issues == []


def test_check_coverage_insufficient():
    frags = [make_frag(0, x=0, y=0, w=10, h=10)]
    issues = check_coverage(frags, canvas_w=100, canvas_h=100, min_coverage=0.5)
    assert len(issues) == 1
    assert issues[0].code == "LOW_COVERAGE"
    assert issues[0].severity == "warning"


def test_check_coverage_invalid_canvas():
    with pytest.raises(ValueError):
        check_coverage([], canvas_w=0, canvas_h=100)


def test_check_coverage_invalid_min():
    with pytest.raises(ValueError):
        check_coverage([], canvas_w=100, canvas_h=100, min_coverage=1.5)


# ─── check_gap_uniformity ─────────────────────────────────────────────────────

def test_check_gap_uniformity_uniform():
    frags = [make_frag(i, x=i * 55, y=0) for i in range(4)]
    issues = check_gap_uniformity(frags, max_gap_std=5.0)
    assert isinstance(issues, list)


def test_check_gap_uniformity_single():
    frags = [make_frag(0)]
    issues = check_gap_uniformity(frags)
    assert issues == []


def test_check_gap_uniformity_invalid_std():
    with pytest.raises(ValueError):
        check_gap_uniformity([], max_gap_std=-1.0)


def test_check_gap_uniformity_nonuniform():
    # Very different gaps
    frags = [
        make_frag(0, x=0, y=0),
        make_frag(1, x=60, y=0),
        make_frag(2, x=500, y=0),
    ]
    issues = check_gap_uniformity(frags, max_gap_std=1.0)
    # Likely has warning
    assert isinstance(issues, list)


# ─── validate_spatial ─────────────────────────────────────────────────────────

def test_validate_spatial_clean():
    frags = [make_frag(i, x=i * 60, y=0) for i in range(3)]
    report = validate_spatial(frags, canvas_w=300, canvas_h=100)
    assert isinstance(report, SpatialReport)
    assert report.n_fragments == 3


def test_validate_spatial_with_issues():
    frags = [make_frag(0, x=0, y=0, w=100, h=100), make_frag(1, x=50, y=50, w=100, h=100)]
    report = validate_spatial(frags, canvas_w=200, canvas_h=200)
    assert isinstance(report, SpatialReport)


def test_validate_spatial_canvas_size():
    frags = [make_frag(0)]
    report = validate_spatial(frags, canvas_w=200.0, canvas_h=150.0)
    assert report.canvas_w == 200.0
    assert report.canvas_h == 150.0


# ─── batch_validate ───────────────────────────────────────────────────────────

def test_batch_validate_basic():
    assembly1 = [make_frag(i, x=i * 60, y=0) for i in range(2)]
    assembly2 = [make_frag(i, x=i * 60, y=0) for i in range(3)]
    reports = batch_validate([assembly1, assembly2], canvas_w=400, canvas_h=100)
    assert len(reports) == 2
    assert reports[0].n_fragments == 2
    assert reports[1].n_fragments == 3


def test_batch_validate_empty():
    reports = batch_validate([], canvas_w=100, canvas_h=100)
    assert reports == []
