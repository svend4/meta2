"""Тесты для puzzle_reconstruction.verification.spatial_validator."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _frag(fid, x=0.0, y=0.0, w=20.0, h=20.0):
    return PlacedFragment(fragment_id=fid, x=x, y=y, width=w, height=h)


def _grid(n, gap=5.0, w=20.0, h=20.0):
    """n фрагментов в ряд без перекрытий."""
    return [_frag(i, x=i * (w + gap), y=0.0, w=w, h=h) for i in range(n)]


# ─── TestSpatialIssue ─────────────────────────────────────────────────────────

class TestSpatialIssue:
    def test_basic_creation(self):
        si = SpatialIssue(code="OVERLAP", severity="error")
        assert si.code == "OVERLAP"
        assert si.severity == "error"

    def test_empty_code_raises(self):
        with pytest.raises(ValueError):
            SpatialIssue(code="", severity="error")

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError):
            SpatialIssue(code="X", severity="critical")

    def test_valid_severities(self):
        for sev in ("error", "warning", "info"):
            si = SpatialIssue(code="C", severity=sev)
            assert si.severity == sev

    def test_fragment_ids_stored(self):
        si = SpatialIssue(code="X", severity="error", fragment_ids=[0, 1])
        assert si.fragment_ids == [0, 1]

    def test_detail_stored(self):
        si = SpatialIssue(code="X", severity="info", detail="desc")
        assert si.detail == "desc"


# ─── TestSpatialReport ────────────────────────────────────────────────────────

class TestSpatialReport:
    def test_defaults(self):
        r = SpatialReport()
        assert r.n_fragments == 0
        assert r.is_valid is True

    def test_negative_n_fragments_raises(self):
        with pytest.raises(ValueError):
            SpatialReport(n_fragments=-1)

    def test_negative_canvas_raises(self):
        with pytest.raises(ValueError):
            SpatialReport(canvas_w=-1.0)

    def test_n_errors_counted(self):
        issues = [
            SpatialIssue(code="A", severity="error"),
            SpatialIssue(code="B", severity="warning"),
        ]
        r = SpatialReport(issues=issues)
        assert r.n_errors == 1

    def test_n_warnings_counted(self):
        issues = [SpatialIssue(code="A", severity="warning")] * 3
        r = SpatialReport(issues=issues)
        assert r.n_warnings == 3

    def test_is_valid_false_with_error(self):
        issues = [SpatialIssue(code="ERR", severity="error")]
        r = SpatialReport(issues=issues)
        assert r.is_valid is False

    def test_is_valid_true_with_warning_only(self):
        issues = [SpatialIssue(code="WARN", severity="warning")]
        r = SpatialReport(issues=issues)
        assert r.is_valid is True


# ─── TestPlacedFragment ───────────────────────────────────────────────────────

class TestPlacedFragment:
    def test_basic_creation(self):
        f = _frag(0, 10, 20, 30, 40)
        assert f.fragment_id == 0

    def test_x2(self):
        f = _frag(0, 10, 0, 30, 10)
        assert f.x2 == pytest.approx(40.0)

    def test_y2(self):
        f = _frag(0, 0, 10, 10, 30)
        assert f.y2 == pytest.approx(40.0)

    def test_area(self):
        f = _frag(0, 0, 0, 10, 5)
        assert f.area == pytest.approx(50.0)

    def test_negative_id_raises(self):
        with pytest.raises(ValueError):
            _frag(-1)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=-1, y=0, width=10, height=10)

    def test_width_below_1_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, width=0.5, height=10)

    def test_height_below_1_raises(self):
        with pytest.raises(ValueError):
            PlacedFragment(fragment_id=0, x=0, y=0, width=10, height=0)


# ─── TestCheckUniqueIds ───────────────────────────────────────────────────────

class TestCheckUniqueIds:
    def test_no_duplicates_empty(self):
        frags = _grid(4)
        issues = check_unique_ids(frags)
        assert issues == []

    def test_duplicate_detected(self):
        frags = [_frag(0), _frag(1), _frag(0)]
        issues = check_unique_ids(frags)
        assert len(issues) == 1
        assert issues[0].code == "DUPLICATE_ID"
        assert issues[0].severity == "error"

    def test_empty_list(self):
        assert check_unique_ids([]) == []

    def test_single_fragment(self):
        assert check_unique_ids([_frag(0)]) == []

    def test_multiple_duplicates(self):
        frags = [_frag(0), _frag(0), _frag(1), _frag(1)]
        issues = check_unique_ids(frags)
        assert len(issues) == 2


# ─── TestCheckWithinCanvas ────────────────────────────────────────────────────

class TestCheckWithinCanvas:
    def test_all_inside(self):
        frags = _grid(3, gap=5)
        issues = check_within_canvas(frags, canvas_w=500, canvas_h=100)
        assert issues == []

    def test_out_of_bounds_detected(self):
        frags = [_frag(0, x=90, y=0, w=20, h=20)]
        issues = check_within_canvas(frags, canvas_w=100, canvas_h=100)
        assert len(issues) == 1
        assert issues[0].code == "OUT_OF_BOUNDS"

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            check_within_canvas([_frag(0)], canvas_w=0, canvas_h=100)

    def test_empty_fragments(self):
        issues = check_within_canvas([], canvas_w=100, canvas_h=100)
        assert issues == []

    def test_exactly_on_boundary_ok(self):
        frags = [_frag(0, x=80, y=80, w=20, h=20)]
        issues = check_within_canvas(frags, canvas_w=100, canvas_h=100)
        assert issues == []


# ─── TestCheckNoOverlaps ──────────────────────────────────────────────────────

class TestCheckNoOverlaps:
    def test_no_overlap_clean(self):
        frags = _grid(3, gap=5)
        issues = check_no_overlaps(frags)
        assert issues == []

    def test_overlap_detected(self):
        a = _frag(0, x=0, y=0, w=30, h=30)
        b = _frag(1, x=10, y=10, w=30, h=30)
        issues = check_no_overlaps([a, b])
        assert len(issues) == 1
        assert issues[0].code == "OVERLAP"

    def test_touching_not_overlap(self):
        a = _frag(0, x=0, y=0, w=20, h=20)   # x2=20
        b = _frag(1, x=20, y=0, w=20, h=20)  # x=20
        issues = check_no_overlaps([a, b])
        assert issues == []

    def test_tolerance(self):
        a = _frag(0, x=0, y=0, w=22, h=20)
        b = _frag(1, x=20, y=0, w=20, h=20)  # overlap_x=2
        issues_strict = check_no_overlaps([a, b], tolerance=0.0)
        issues_tolerant = check_no_overlaps([a, b], tolerance=5.0)
        assert len(issues_strict) == 1
        assert issues_tolerant == []

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            check_no_overlaps(_grid(2), tolerance=-1.0)

    def test_single_fragment(self):
        assert check_no_overlaps([_frag(0)]) == []


# ─── TestCheckCoverage ────────────────────────────────────────────────────────

class TestCheckCoverage:
    def test_sufficient_coverage(self):
        frags = [_frag(0, x=0, y=0, w=100, h=100)]
        issues = check_coverage(frags, canvas_w=100, canvas_h=100, min_coverage=0.9)
        assert issues == []

    def test_insufficient_coverage(self):
        frags = [_frag(0, x=0, y=0, w=10, h=10)]
        issues = check_coverage(frags, canvas_w=100, canvas_h=100, min_coverage=0.5)
        assert len(issues) == 1
        assert issues[0].code == "LOW_COVERAGE"
        assert issues[0].severity == "warning"

    def test_invalid_canvas_raises(self):
        with pytest.raises(ValueError):
            check_coverage([_frag(0)], canvas_w=0, canvas_h=100)

    def test_invalid_min_coverage_raises(self):
        with pytest.raises(ValueError):
            check_coverage([_frag(0)], canvas_w=100, canvas_h=100, min_coverage=1.5)

    def test_empty_fragments_low_coverage(self):
        issues = check_coverage([], canvas_w=100, canvas_h=100, min_coverage=0.1)
        assert len(issues) == 1


# ─── TestCheckGapUniformity ───────────────────────────────────────────────────

class TestCheckGapUniformity:
    def test_uniform_gaps_clean(self):
        frags = _grid(4, gap=5.0)
        issues = check_gap_uniformity(frags, max_gap_std=100.0)
        assert issues == []

    def test_highly_nonuniform_detected(self):
        frags = [
            _frag(0, x=0, y=0, w=10, h=10),
            _frag(1, x=15, y=0, w=10, h=10),
            _frag(2, x=1000, y=0, w=10, h=10),
        ]
        issues = check_gap_uniformity(frags, max_gap_std=5.0)
        assert len(issues) == 1
        assert issues[0].code == "UNEVEN_GAPS"

    def test_single_fragment_no_issue(self):
        assert check_gap_uniformity([_frag(0)]) == []

    def test_empty_list_no_issue(self):
        assert check_gap_uniformity([]) == []

    def test_negative_max_gap_std_raises(self):
        with pytest.raises(ValueError):
            check_gap_uniformity(_grid(3), max_gap_std=-1.0)


# ─── TestValidateSpatial ──────────────────────────────────────────────────────

class TestValidateSpatial:
    def test_returns_report(self):
        frags = _grid(3)
        r = validate_spatial(frags, canvas_w=500, canvas_h=100)
        assert isinstance(r, SpatialReport)

    def test_clean_assembly_valid(self):
        frags = _grid(3)
        r = validate_spatial(frags, canvas_w=500, canvas_h=100, min_coverage=0.0)
        assert r.is_valid is True

    def test_duplicate_id_makes_invalid(self):
        frags = [_frag(0), _frag(0, x=50)]
        r = validate_spatial(frags, canvas_w=200, canvas_h=100, min_coverage=0.0)
        assert r.is_valid is False

    def test_oob_makes_invalid(self):
        frags = [_frag(0, x=200, y=0, w=20, h=20)]
        r = validate_spatial(frags, canvas_w=100, canvas_h=100, min_coverage=0.0)
        assert r.is_valid is False

    def test_n_fragments_set(self):
        frags = _grid(4)
        r = validate_spatial(frags, canvas_w=500, canvas_h=100)
        assert r.n_fragments == 4

    def test_empty_assembly(self):
        r = validate_spatial([], canvas_w=100, canvas_h=100)
        assert isinstance(r, SpatialReport)


# ─── TestBatchValidate ────────────────────────────────────────────────────────

class TestBatchValidate:
    def test_returns_list(self):
        assemblies = [_grid(3), _grid(2)]
        result = batch_validate(assemblies, canvas_w=500, canvas_h=100)
        assert isinstance(result, list)

    def test_length_matches(self):
        assemblies = [_grid(2), _grid(3), _grid(4)]
        result = batch_validate(assemblies, canvas_w=500, canvas_h=100)
        assert len(result) == 3

    def test_all_reports(self):
        assemblies = [_grid(2)]
        result = batch_validate(assemblies, canvas_w=500, canvas_h=100)
        assert isinstance(result[0], SpatialReport)

    def test_empty_assemblies(self):
        result = batch_validate([], canvas_w=100, canvas_h=100)
        assert result == []
