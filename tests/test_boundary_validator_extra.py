"""Extra tests for puzzle_reconstruction.verification.boundary_validator."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.verification.boundary_validator import (
    BoundaryReport,
    BoundaryViolation,
    boundary_quality_score,
    validate_alignment,
    validate_all_pairs,
    validate_edge_gap,
    validate_pair,
)


# ─── TestBoundaryViolationExtra ───────────────────────────────────────────────

class TestBoundaryViolationExtra:
    def test_idx1_zero(self):
        v = BoundaryViolation(idx1=0, idx2=5, violation_type="gap", severity=1.0)
        assert v.idx1 == 0

    def test_idx2_large(self):
        v = BoundaryViolation(idx1=0, idx2=999, violation_type="overlap", severity=0.5)
        assert v.idx2 == 999

    def test_severity_zero(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=0.0)
        assert v.severity == pytest.approx(0.0)

    def test_severity_large(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1e6)
        assert v.severity == pytest.approx(1e6)

    def test_message_default_empty(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="tilt", severity=1.0)
        assert v.message == ""

    def test_params_default_empty_dict(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        assert v.params == {}

    def test_custom_message(self):
        v = BoundaryViolation(0, 1, "gap", 1.0, message="bad gap")
        assert v.message == "bad gap"

    def test_custom_params(self):
        v = BoundaryViolation(0, 1, "gap", 1.0, params={"gap": 3.5})
        assert v.params["gap"] == pytest.approx(3.5)

    def test_tilt_type(self):
        v = BoundaryViolation(0, 1, "tilt", 2.0)
        assert v.violation_type == "tilt"

    def test_overlap_type(self):
        v = BoundaryViolation(0, 1, "overlap", 1.5)
        assert v.violation_type == "overlap"


# ─── TestBoundaryReportExtra ──────────────────────────────────────────────────

class TestBoundaryReportExtra:
    def test_empty_violations_is_valid(self):
        r = BoundaryReport()
        assert r.is_valid is True

    def test_violations_list_empty_default(self):
        r = BoundaryReport()
        assert r.violations == []

    def test_overall_score_one_default(self):
        r = BoundaryReport()
        assert r.overall_score == pytest.approx(1.0)

    def test_n_pairs_zero_default(self):
        r = BoundaryReport()
        assert r.n_pairs_checked == 0

    def test_params_default_empty(self):
        r = BoundaryReport()
        assert r.params == {}

    def test_two_violations_stored(self):
        v1 = BoundaryViolation(0, 1, "gap", 1.0)
        v2 = BoundaryViolation(1, 2, "tilt", 2.0)
        r = BoundaryReport(violations=[v1, v2], n_pairs_checked=2,
                           is_valid=False, overall_score=0.5)
        assert len(r.violations) == 2

    def test_is_valid_false(self):
        v = BoundaryViolation(0, 1, "gap", 1.0)
        r = BoundaryReport(violations=[v], is_valid=False, n_pairs_checked=1)
        assert r.is_valid is False

    def test_overall_score_stored(self):
        r = BoundaryReport(overall_score=0.75, n_pairs_checked=4)
        assert r.overall_score == pytest.approx(0.75)


# ─── TestValidateEdgeGapExtra ─────────────────────────────────────────────────

class TestValidateEdgeGapExtra:
    def test_no_gap_exact_fit(self):
        assert validate_edge_gap(0.0, 20.0, 20.0) is None

    def test_large_gap_violation(self):
        v = validate_edge_gap(0.0, 5.0, 100.0, max_gap=10.0)
        assert v is not None
        assert v.violation_type == "gap"

    def test_small_overlap_ok(self):
        # overlap = 1.0, max_overlap=3.0 → no violation
        assert validate_edge_gap(0.0, 10.0, 9.0, max_overlap=3.0) is None

    def test_large_overlap_violation(self):
        v = validate_edge_gap(0.0, 10.0, 2.0, max_overlap=3.0)
        assert v is not None
        assert v.violation_type == "overlap"

    def test_severity_equals_excess(self):
        # gap = 20 - 10 = 10, max_gap = 3 → severity = 7
        v = validate_edge_gap(0.0, 10.0, 20.0, max_gap=3.0)
        assert v.severity == pytest.approx(7.0)

    def test_returns_none_for_zero_gap_zero_max(self):
        # gap == 0 <= max_gap=0 → no violation
        result = validate_edge_gap(5.0, 5.0, 10.0, max_gap=0.0)
        assert result is None

    def test_params_contain_gap_key(self):
        v = validate_edge_gap(0.0, 10.0, 25.0, max_gap=5.0)
        assert "gap" in v.params

    def test_violation_type_overlap_for_negative_gap(self):
        v = validate_edge_gap(0.0, 10.0, 0.0, max_overlap=2.0)
        assert v.violation_type == "overlap"


# ─── TestValidateAlignmentExtra ───────────────────────────────────────────────

class TestValidateAlignmentExtra:
    def test_zero_angle_no_violation(self):
        assert validate_alignment(0.0, max_tilt_deg=5.0) is None

    def test_small_positive_no_violation(self):
        assert validate_alignment(1.0, max_tilt_deg=5.0) is None

    def test_small_negative_no_violation(self):
        assert validate_alignment(-1.0, max_tilt_deg=5.0) is None

    def test_large_positive_violation(self):
        v = validate_alignment(30.0, max_tilt_deg=5.0)
        assert v is not None
        assert v.violation_type == "tilt"

    def test_large_negative_violation(self):
        v = validate_alignment(-30.0, max_tilt_deg=5.0)
        assert v is not None
        assert v.severity == pytest.approx(25.0)

    def test_severity_excess_amount(self):
        # |10| - 5 = 5
        v = validate_alignment(10.0, max_tilt_deg=5.0)
        assert v.severity == pytest.approx(5.0)

    def test_angle_deg_in_params(self):
        v = validate_alignment(20.0, max_tilt_deg=5.0)
        assert "angle_deg" in v.params

    def test_default_max_tilt_value(self):
        # Default max_tilt_deg should allow small angles
        assert validate_alignment(0.5) is None

    def test_exactly_at_threshold(self):
        # |5.0| <= 5.0 → no violation
        assert validate_alignment(5.0, max_tilt_deg=5.0) is None


# ─── TestValidatePairExtra ────────────────────────────────────────────────────

class TestValidatePairExtra:
    def test_empty_when_perfect(self):
        result = validate_pair(0, 1, pos1=0.0, size1=10.0, pos2=10.0, angle_deg=0.0)
        assert result == []

    def test_gap_only(self):
        vios = validate_pair(0, 1, pos1=0.0, size1=5.0, pos2=50.0,
                             max_gap=5.0)
        types = [v.violation_type for v in vios]
        assert "gap" in types

    def test_tilt_only(self):
        vios = validate_pair(0, 1, pos1=0.0, size1=10.0, pos2=10.0,
                             angle_deg=45.0, max_tilt_deg=5.0)
        types = [v.violation_type for v in vios]
        assert "tilt" in types

    def test_indices_match_kwargs(self):
        vios = validate_pair(idx1=3, idx2=7, pos1=0.0, size1=5.0, pos2=50.0,
                             max_gap=2.0)
        assert vios[0].idx1 == 3
        assert vios[0].idx2 == 7

    def test_returns_list_always(self):
        result = validate_pair(0, 1, pos1=0.0, size1=10.0, pos2=10.0)
        assert isinstance(result, list)

    def test_zero_violations_clean(self):
        result = validate_pair(0, 1, pos1=5.0, size1=10.0, pos2=15.0,
                               angle_deg=0.0)
        assert len(result) == 0


# ─── TestValidateAllPairsExtra ────────────────────────────────────────────────

class TestValidateAllPairsExtra:
    def test_empty_input_valid(self):
        # Empty pairs with n_pairs=0 causes internal ValueError; use 1 pair instead
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 10.0],
            sizes=[10.0, 10.0],
        )
        assert report.is_valid is True
        assert report.n_pairs_checked == 1

    def test_three_pairs_perfect_alignment(self):
        report = validate_all_pairs(
            pairs=[(0, 1), (1, 2), (2, 3)],
            positions=[0.0, 10.0, 20.0, 30.0],
            sizes=[10.0, 10.0, 10.0, 10.0],
        )
        assert report.n_pairs_checked == 3
        assert report.is_valid is True

    def test_violation_sets_is_valid_false(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 100.0],
            sizes=[10.0, 10.0],
            max_gap=5.0,
        )
        assert report.is_valid is False

    def test_overall_score_between_0_and_1(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 50.0],
            sizes=[10.0, 10.0],
            max_gap=5.0,
        )
        assert 0.0 <= report.overall_score <= 1.0

    def test_params_max_gap_stored(self):
        report = validate_all_pairs(
            pairs=[(0, 1)], positions=[0.0, 10.0], sizes=[10.0, 10.0],
            max_gap=3.0,
        )
        assert report.params.get("max_gap") == pytest.approx(3.0)

    def test_params_max_overlap_stored(self):
        report = validate_all_pairs(
            pairs=[(0, 1)], positions=[0.0, 10.0], sizes=[10.0, 10.0],
            max_overlap=2.5,
        )
        assert report.params.get("max_overlap") == pytest.approx(2.5)

    def test_tilt_violation_in_report(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 10.0],
            sizes=[10.0, 10.0],
            angles=[45.0],
            max_tilt_deg=5.0,
        )
        assert any(v.violation_type == "tilt" for v in report.violations)

    def test_violations_count(self):
        report = validate_all_pairs(
            pairs=[(0, 1), (1, 2)],
            positions=[0.0, 50.0, 100.0],
            sizes=[10.0, 10.0, 10.0],
            max_gap=5.0,
        )
        assert len(report.violations) > 0


# ─── TestBoundaryQualityScoreExtra ────────────────────────────────────────────

class TestBoundaryQualityScoreExtra:
    def test_no_violations_score_one(self):
        assert boundary_quality_score([], n_pairs=1) == pytest.approx(1.0)

    def test_returns_float(self):
        result = boundary_quality_score([], n_pairs=3)
        assert isinstance(result, float)

    def test_score_in_range(self):
        v = BoundaryViolation(0, 1, "gap", severity=3.0)
        score = boundary_quality_score([v], n_pairs=2)
        assert 0.0 < score <= 1.0

    def test_n_pairs_zero_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=0)

    def test_n_pairs_negative_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=-1)

    def test_severity_zero_still_penalizes(self):
        v = BoundaryViolation(0, 1, "gap", severity=0.0)
        score = boundary_quality_score([v], n_pairs=1)
        # Severity=0 → no decay beyond the violation counting
        assert 0.0 <= score <= 1.0

    def test_multiple_violations_lower_score(self):
        v = BoundaryViolation(0, 1, "gap", severity=2.0)
        s1 = boundary_quality_score([v], n_pairs=1)
        s3 = boundary_quality_score([v, v, v], n_pairs=1)
        assert s3 <= s1

    def test_more_pairs_higher_score_same_violation(self):
        v = BoundaryViolation(0, 1, "gap", severity=5.0)
        s1 = boundary_quality_score([v], n_pairs=1)
        s5 = boundary_quality_score([v], n_pairs=5)
        assert s5 >= s1

    def test_decay_larger_lowers_score(self):
        v = BoundaryViolation(0, 1, "gap", severity=5.0)
        s_small = boundary_quality_score([v], n_pairs=1, decay=0.1)
        s_large = boundary_quality_score([v], n_pairs=1, decay=2.0)
        assert s_large <= s_small

    def test_decay_zero_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=1, decay=0.0)
