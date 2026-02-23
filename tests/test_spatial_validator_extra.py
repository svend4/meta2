"""Extra tests for puzzle_reconstruction.verification.spatial_validator."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _frag(fid, x=0.0, y=0.0, w=20.0, h=20.0):
    return PlacedFragment(fragment_id=fid, x=x, y=y, width=w, height=h)


def _grid(n, gap=5.0, w=20.0, h=20.0):
    return [_frag(i, x=i * (w + gap), y=0.0, w=w, h=h) for i in range(n)]


# ─── TestSpatialIssueExtra ──────────────────────────────────────────────────

class TestSpatialIssueExtra:
    def test_error_severity(self):
        si = SpatialIssue(code="ERR", severity="error")
        assert si.severity == "error"

    def test_warning_severity(self):
        si = SpatialIssue(code="W", severity="warning")
        assert si.severity == "warning"

    def test_info_severity(self):
        si = SpatialIssue(code="I", severity="info")
        assert si.severity == "info"

    def test_code_stored(self):
        si = SpatialIssue(code="OVERLAP", severity="error")
        assert si.code == "OVERLAP"

    def test_fragment_ids_default_empty(self):
        si = SpatialIssue(code="X", severity="info")
        assert si.fragment_ids == []

    def test_fragment_ids_stored(self):
        si = SpatialIssue(code="X", severity="error", fragment_ids=[3, 7])
        assert 3 in si.fragment_ids
        assert 7 in si.fragment_ids

    def test_detail_default_empty(self):
        si = SpatialIssue(code="X", severity="info")
        assert si.detail == ""

    def test_detail_stored(self):
        si = SpatialIssue(code="X", severity="info", detail="overlap by 5px")
        assert "5px" in si.detail

    def test_empty_code_raises(self):
        with pytest.raises(ValueError):
            SpatialIssue(code="", severity="error")

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            SpatialIssue(code="X", severity="fatal")


# ─── TestSpatialReportExtra ─────────────────────────────────────────────────

class TestSpatialReportExtra:
    def test_default_is_valid(self):
        r = SpatialReport()
        assert r.is_valid is True

    def test_default_n_fragments_zero(self):
        r = SpatialReport()
        assert r.n_fragments == 0

    def test_single_error_makes_invalid(self):
        r = SpatialReport(issues=[SpatialIssue(code="E", severity="error")])
        assert r.is_valid is False

    def test_info_only_valid(self):
        r = SpatialReport(issues=[SpatialIssue(code="I", severity="info")])
        assert r.is_valid is True

    def test_n_errors_count(self):
        issues = [SpatialIssue(code="E", severity="error")] * 3
        r = SpatialReport(issues=issues)
        assert r.n_errors == 3

    def test_n_warnings_count(self):
        issues = [SpatialIssue(code="W", severity="warning")] * 2
        r = SpatialReport(issues=issues)
        assert r.n_warnings == 2

    def test_mixed_issues(self):
        issues = [
            SpatialIssue(code="E", severity="error"),
            SpatialIssue(code="W", severity="warning"),
            SpatialIssue(code="I", severity="info"),
        ]
        r = SpatialReport(issues=issues)
        assert r.n_errors == 1
        assert r.n_warnings == 1
        assert r.is_valid is False

    def test_n_fragments_stored(self):
        r = SpatialReport(n_fragments=5)
        assert r.n_fragments == 5

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            SpatialReport(n_fragments=-1)

    def test_negative_canvas_w_raises(self):
        with pytest.raises(ValueError):
            SpatialReport(canvas_w=-1.0)


# ─── TestPlacedFragmentExtra ─────────────────────────────────────────────────

class TestPlacedFragmentExtra:
    def test_x2_property(self):
        f = _frag(0, x=10.0, y=0.0, w=30.0, h=20.0)
        assert f.x2 == pytest.approx(40.0)

    def test_y2_property(self):
        f = _frag(0, x=0.0, y=5.0, w=20.0, h=15.0)
        assert f.y2 == pytest.approx(20.0)

    def test_area_property(self):
        f = _frag(0, w=10.0, h=5.0)
        assert f.area == pytest.approx(50.0)

    def test_zero_x_y_ok(self):
        f = _frag(0, x=0.0, y=0.0)
        assert f.x == pytest.approx(0.0)
        assert f.y == pytest.approx(0.0)

    def test_large_coords(self):
        f = _frag(0, x=9999.0, y=8888.0, w=100.0, h=100.0)
        assert f.x2 == pytest.approx(10099.0)

    def test_fragment_id_stored(self):
        f = _frag(42)
        assert f.fragment_id == 42

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            _frag(-1)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=-1.0, y=0.0, width=10.0, height=10.0)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0.0, y=0.0, width=0.0, height=10.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0.0, y=0.0, width=10.0, height=0.0)


# ─── TestCheckUniqueIdsExtra ─────────────────────────────────────────────────

class TestCheckUniqueIdsExtra:
    def test_unique_ids_ok(self):
        frags = [_frag(i) for i in range(5)]
        assert check_unique_ids(frags) == []

    def test_duplicate_returns_error(self):
        frags = [_frag(0), _frag(1), _frag(0)]
        issues = check_unique_ids(frags)
        assert len(issues) == 1
        assert issues[0].severity == "error"

    def test_duplicate_code(self):
        frags = [_frag(0), _frag(0)]
        issues = check_unique_ids(frags)
        assert issues[0].code == "DUPLICATE_ID"

    def test_three_duplicates(self):
        frags = [_frag(0), _frag(0), _frag(1), _frag(1), _frag(2), _frag(2)]
        issues = check_unique_ids(frags)
        assert len(issues) == 3

    def test_empty_list_ok(self):
        assert check_unique_ids([]) == []

    def test_single_frag_ok(self):
        assert check_unique_ids([_frag(7)]) == []


# ─── TestCheckWithinCanvasExtra ──────────────────────────────────────────────

class TestCheckWithinCanvasExtra:
    def test_fits_exactly(self):
        frags = [_frag(0, x=80.0, y=80.0, w=20.0, h=20.0)]
        assert check_within_canvas(frags, canvas_w=100, canvas_h=100) == []

    def test_x_overflow(self):
        frags = [_frag(0, x=91.0, y=0.0, w=20.0, h=20.0)]
        issues = check_within_canvas(frags, canvas_w=100, canvas_h=100)
        assert len(issues) == 1
        assert issues[0].code == "OUT_OF_BOUNDS"

    def test_y_overflow(self):
        frags = [_frag(0, x=0.0, y=91.0, w=20.0, h=20.0)]
        issues = check_within_canvas(frags, canvas_w=100, canvas_h=100)
        assert len(issues) == 1

    def test_empty_fragments(self):
        assert check_within_canvas([], canvas_w=100, canvas_h=100) == []

    def test_zero_canvas_raises(self):
        with pytest.raises(ValueError):
            check_within_canvas([_frag(0)], canvas_w=0, canvas_h=100)

    def test_multiple_out_of_bounds(self):
        frags = [_frag(i, x=i * 200.0, y=0.0) for i in range(3)]
        issues = check_within_canvas(frags, canvas_w=100, canvas_h=50)
        assert len(issues) >= 2


# ─── TestCheckNoOverlapsExtra ────────────────────────────────────────────────

class TestCheckNoOverlapsExtra:
    def test_grid_no_overlap(self):
        frags = _grid(5, gap=5.0)
        assert check_no_overlaps(frags) == []

    def test_overlap_code(self):
        a = _frag(0, x=0.0, y=0.0, w=30.0, h=30.0)
        b = _frag(1, x=10.0, y=10.0, w=30.0, h=30.0)
        issues = check_no_overlaps([a, b])
        assert issues[0].code == "OVERLAP"

    def test_touching_ok(self):
        a = _frag(0, x=0.0, y=0.0, w=20.0, h=20.0)
        b = _frag(1, x=20.0, y=0.0, w=20.0, h=20.0)
        assert check_no_overlaps([a, b]) == []

    def test_tolerance_5_absorbs_small(self):
        a = _frag(0, x=0.0, y=0.0, w=22.0, h=20.0)
        b = _frag(1, x=20.0, y=0.0, w=20.0, h=20.0)
        issues = check_no_overlaps([a, b], tolerance=5.0)
        assert issues == []

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            check_no_overlaps(_grid(2), tolerance=-1.0)

    def test_single_frag_ok(self):
        assert check_no_overlaps([_frag(0)]) == []


# ─── TestCheckCoverageExtra ─────────────────────────────────────────────────

class TestCheckCoverageExtra:
    def test_full_coverage(self):
        frags = [_frag(0, x=0.0, y=0.0, w=100.0, h=100.0)]
        assert check_coverage(frags, canvas_w=100, canvas_h=100,
                               min_coverage=0.9) == []

    def test_low_coverage_code(self):
        frags = [_frag(0, x=0.0, y=0.0, w=1.0, h=1.0)]
        issues = check_coverage(frags, canvas_w=100, canvas_h=100,
                                 min_coverage=0.5)
        assert issues[0].code == "LOW_COVERAGE"

    def test_low_coverage_severity_warning(self):
        frags = [_frag(0, x=0.0, y=0.0, w=1.0, h=1.0)]
        issues = check_coverage(frags, canvas_w=100, canvas_h=100,
                                 min_coverage=0.5)
        assert issues[0].severity == "warning"

    def test_zero_min_coverage_ok(self):
        issues = check_coverage([], canvas_w=100, canvas_h=100, min_coverage=0.0)
        assert issues == []

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            check_coverage([_frag(0)], canvas_w=0, canvas_h=100)

    def test_min_coverage_above_1_raises(self):
        with pytest.raises(ValueError):
            check_coverage([_frag(0)], canvas_w=100, canvas_h=100, min_coverage=1.5)


# ─── TestCheckGapUniformityExtra ────────────────────────────────────────────

class TestCheckGapUniformityExtra:
    def test_two_frags_uniform(self):
        frags = _grid(2, gap=10.0)
        assert check_gap_uniformity(frags, max_gap_std=100.0) == []

    def test_highly_nonuniform_code(self):
        frags = [
            _frag(0, x=0, y=0, w=10, h=10),
            _frag(1, x=15, y=0, w=10, h=10),
            _frag(2, x=5000, y=0, w=10, h=10),
        ]
        issues = check_gap_uniformity(frags, max_gap_std=1.0)
        assert issues[0].code == "UNEVEN_GAPS"

    def test_empty_no_issue(self):
        assert check_gap_uniformity([]) == []

    def test_single_no_issue(self):
        assert check_gap_uniformity([_frag(0)]) == []

    def test_negative_max_gap_std_raises(self):
        with pytest.raises(ValueError):
            check_gap_uniformity(_grid(3), max_gap_std=-0.5)

    def test_large_max_gap_std_always_ok(self):
        frags = [
            _frag(0, x=0, y=0, w=10, h=10),
            _frag(1, x=100, y=0, w=10, h=10),
            _frag(2, x=10000, y=0, w=10, h=10),
        ]
        assert check_gap_uniformity(frags, max_gap_std=1e9) == []


# ─── TestValidateSpatialExtra ───────────────────────────────────────────────

class TestValidateSpatialExtra:
    def test_clean_valid(self):
        frags = _grid(3)
        r = validate_spatial(frags, canvas_w=500, canvas_h=100, min_coverage=0.0)
        assert r.is_valid is True

    def test_n_fragments_stored(self):
        frags = _grid(5)
        r = validate_spatial(frags, canvas_w=1000, canvas_h=100)
        assert r.n_fragments == 5

    def test_duplicate_invalid(self):
        frags = [_frag(0), _frag(0, x=50.0)]
        r = validate_spatial(frags, canvas_w=200, canvas_h=100, min_coverage=0.0)
        assert r.is_valid is False

    def test_oob_invalid(self):
        frags = [_frag(0, x=300.0, y=0.0, w=20.0, h=20.0)]
        r = validate_spatial(frags, canvas_w=100, canvas_h=100, min_coverage=0.0)
        assert r.is_valid is False

    def test_returns_spatial_report(self):
        r = validate_spatial([], canvas_w=100, canvas_h=100)
        assert isinstance(r, SpatialReport)

    def test_empty_assembly(self):
        r = validate_spatial([], canvas_w=100, canvas_h=100)
        assert r.n_fragments == 0

    def test_low_coverage_warning_not_error(self):
        frags = [_frag(0, x=0.0, y=0.0, w=1.0, h=1.0)]
        r = validate_spatial(frags, canvas_w=100, canvas_h=100, min_coverage=0.9)
        assert r.is_valid is True   # warning only


# ─── TestBatchValidateExtra ─────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_single_assembly(self):
        result = batch_validate([_grid(3)], canvas_w=500, canvas_h=100)
        assert len(result) == 1

    def test_all_reports(self):
        assemblies = [_grid(2), _grid(3), _grid(4)]
        result = batch_validate(assemblies, canvas_w=1000, canvas_h=100)
        for r in result:
            assert isinstance(r, SpatialReport)

    def test_empty_assemblies(self):
        assert batch_validate([], canvas_w=100, canvas_h=100) == []

    def test_length_matches(self):
        assemblies = [_grid(i + 1) for i in range(5)]
        result = batch_validate(assemblies, canvas_w=1000, canvas_h=100)
        assert len(result) == 5

    def test_n_fragments_per_report(self):
        assemblies = [_grid(2), _grid(3)]
        result = batch_validate(assemblies, canvas_w=1000, canvas_h=100)
        assert result[0].n_fragments == 2
        assert result[1].n_fragments == 3
