"""Extra tests for puzzle_reconstruction/verification/spatial_validator.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pf(fid=0, x=0.0, y=0.0, w=50.0, h=50.0):
    return PlacedFragment(fragment_id=fid, x=x, y=y, width=w, height=h)


def _grid():
    """2x2 non-overlapping 50x50 on a 100x100 canvas."""
    return [_pf(0, 0, 0), _pf(1, 50, 0), _pf(2, 0, 50), _pf(3, 50, 50)]


# ─── SpatialIssue ────────────────────────────────────────────────────────────

class TestSpatialIssueExtra:
    def test_error(self):
        i = SpatialIssue(code="OVERLAP", severity="error")
        assert i.code == "OVERLAP"

    def test_warning(self):
        i = SpatialIssue(code="LOW_COV", severity="warning")
        assert i.severity == "warning"

    def test_info(self):
        i = SpatialIssue(code="NOTE", severity="info")
        assert i.severity == "info"

    def test_empty_code_raises(self):
        with pytest.raises(ValueError):
            SpatialIssue(code="", severity="error")

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            SpatialIssue(code="test", severity="fatal")


# ─── SpatialReport ───────────────────────────────────────────────────────────

class TestSpatialReportExtra:
    def test_valid(self):
        r = SpatialReport(n_fragments=4, canvas_w=100.0, canvas_h=100.0)
        assert r.is_valid is True
        assert r.n_errors == 0
        assert r.n_warnings == 0

    def test_with_error(self):
        i = SpatialIssue(code="OVERLAP", severity="error")
        r = SpatialReport(issues=[i], n_fragments=2)
        assert r.is_valid is False
        assert r.n_errors == 1

    def test_with_warning(self):
        i = SpatialIssue(code="LOW_COV", severity="warning")
        r = SpatialReport(issues=[i], n_fragments=2)
        assert r.is_valid is True
        assert r.n_warnings == 1

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            SpatialReport(n_fragments=-1)

    def test_negative_canvas_raises(self):
        with pytest.raises(ValueError):
            SpatialReport(canvas_w=-1.0)


# ─── PlacedFragment ─────────────────────────────────────────────────────────

class TestPlacedFragmentExtra:
    def test_properties(self):
        f = _pf(0, 10.0, 20.0, 30.0, 40.0)
        assert f.x2 == pytest.approx(40.0)
        assert f.y2 == pytest.approx(60.0)
        assert f.area == pytest.approx(1200.0)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=-1, x=0, y=0, width=10, height=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=-1, y=0, width=10, height=10)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, width=0, height=10)


# ─── check_unique_ids ────────────────────────────────────────────────────────

class TestCheckUniqueIdsExtra:
    def test_unique(self):
        assert check_unique_ids(_grid()) == []

    def test_duplicate(self):
        frags = [_pf(0, 0, 0), _pf(0, 50, 0)]
        issues = check_unique_ids(frags)
        assert len(issues) == 1
        assert issues[0].code == "DUPLICATE_ID"

    def test_empty(self):
        assert check_unique_ids([]) == []


# ─── check_within_canvas ────────────────────────────────────────────────────

class TestCheckWithinCanvasExtra:
    def test_inside(self):
        assert check_within_canvas(_grid(), 100, 100) == []

    def test_outside(self):
        issues = check_within_canvas([_pf(0, 80, 80)], 100, 100)
        assert len(issues) == 1
        assert issues[0].code == "OUT_OF_BOUNDS"

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            check_within_canvas([], 0, 100)

    def test_empty(self):
        assert check_within_canvas([], 100, 100) == []


# ─── check_no_overlaps ──────────────────────────────────────────────────────

class TestCheckNoOverlapsExtra:
    def test_no_overlap(self):
        assert check_no_overlaps(_grid()) == []

    def test_overlap(self):
        frags = [_pf(0, 0, 0, 100, 100), _pf(1, 50, 50, 100, 100)]
        issues = check_no_overlaps(frags)
        assert len(issues) == 1
        assert issues[0].code == "OVERLAP"

    def test_with_tolerance(self):
        frags = [_pf(0, 0, 0, 100, 100), _pf(1, 99, 0, 100, 100)]
        # Overlap is 1x100 px; tolerance=5 → 1 ≤ 5 → no violation
        issues = check_no_overlaps(frags, tolerance=5.0)
        assert issues == []

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            check_no_overlaps([], tolerance=-1.0)

    def test_empty(self):
        assert check_no_overlaps([]) == []


# ─── check_coverage ──────────────────────────────────────────────────────────

class TestCheckCoverageExtra:
    def test_full(self):
        assert check_coverage(_grid(), 100, 100) == []

    def test_low(self):
        issues = check_coverage([_pf(0, 0, 0, 10, 10)], 100, 100,
                                min_coverage=0.5)
        assert len(issues) == 1
        assert issues[0].code == "LOW_COVERAGE"

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            check_coverage([], 0, 100)

    def test_invalid_min_coverage_raises(self):
        with pytest.raises(ValueError):
            check_coverage([], 100, 100, min_coverage=1.5)


# ─── check_gap_uniformity ───────────────────────────────────────────────────

class TestCheckGapUniformityExtra:
    def test_uniform(self):
        # All same size, evenly spaced
        assert check_gap_uniformity(_grid()) == []

    def test_uneven(self):
        # Gaps: 0→1 is 10px, 0→2 is 190px, 1→2 is 190px → std >> 1
        frags = [_pf(0, 0, 0, 10, 10), _pf(1, 20, 0, 10, 10),
                 _pf(2, 0, 200, 10, 10)]
        issues = check_gap_uniformity(frags, max_gap_std=1.0)
        assert len(issues) == 1
        assert issues[0].code == "UNEVEN_GAPS"

    def test_single(self):
        assert check_gap_uniformity([_pf()]) == []

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            check_gap_uniformity([], max_gap_std=-1.0)


# ─── validate_spatial ────────────────────────────────────────────────────────

class TestValidateSpatialExtra:
    def test_clean(self):
        r = validate_spatial(_grid(), 100, 100)
        assert r.is_valid is True
        assert r.n_fragments == 4

    def test_with_overlap(self):
        frags = [_pf(0, 0, 0, 100, 100), _pf(1, 50, 50, 100, 100)]
        r = validate_spatial(frags, 200, 200)
        assert r.is_valid is False

    def test_empty(self):
        r = validate_spatial([], 100, 100)
        assert r.is_valid is True
        assert r.n_fragments == 0


# ─── batch_validate ──────────────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_empty(self):
        assert batch_validate([], 100, 100) == []

    def test_multiple(self):
        results = batch_validate([_grid(), [_pf()]], 100, 100)
        assert len(results) == 2
        assert all(isinstance(r, SpatialReport) for r in results)
