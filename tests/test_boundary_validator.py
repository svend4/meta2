"""Tests for puzzle_reconstruction.verification.boundary_validator."""
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


# ─── BoundaryViolation ───────────────────────────────────────────────────────

class TestBoundaryViolation:
    def test_fields_stored(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=3.5)
        assert v.idx1 == 0
        assert v.idx2 == 1
        assert v.violation_type == "gap"
        assert v.severity == pytest.approx(3.5)
        assert v.message == ""
        assert v.params == {}

    def test_message_and_params(self):
        v = BoundaryViolation(
            idx1=2, idx2=3, violation_type="overlap",
            severity=1.0, message="too much", params={"key": 42},
        )
        assert v.message == "too much"
        assert v.params["key"] == 42

    def test_valid_violation_types(self):
        for vtype in ("gap", "overlap", "tilt"):
            v = BoundaryViolation(0, 1, vtype, 0.5)
            assert v.violation_type == vtype


# ─── BoundaryReport ──────────────────────────────────────────────────────────

class TestBoundaryReport:
    def test_defaults(self):
        r = BoundaryReport()
        assert r.violations == []
        assert r.n_pairs_checked == 0
        assert r.is_valid is True
        assert r.overall_score == pytest.approx(1.0)
        assert r.params == {}

    def test_custom_values(self):
        v = BoundaryViolation(0, 1, "gap", 2.0)
        r = BoundaryReport(
            violations=[v], n_pairs_checked=3,
            is_valid=False, overall_score=0.5,
        )
        assert len(r.violations) == 1
        assert r.n_pairs_checked == 3
        assert r.is_valid is False
        assert r.overall_score == pytest.approx(0.5)


# ─── validate_edge_gap ───────────────────────────────────────────────────────

class TestValidateEdgeGap:
    def test_no_violation_exact(self):
        # pos2 == pos1 + size1 → gap == 0 → no violation
        result = validate_edge_gap(pos1=0.0, size1=10.0, pos2=10.0)
        assert result is None

    def test_no_violation_within_tolerance(self):
        result = validate_edge_gap(pos1=0.0, size1=10.0, pos2=12.0, max_gap=5.0)
        assert result is None

    def test_gap_violation(self):
        # gap = 20 - 10 = 10 > max_gap=5
        v = validate_edge_gap(pos1=0.0, size1=10.0, pos2=20.0, max_gap=5.0)
        assert v is not None
        assert v.violation_type == "gap"
        assert v.severity == pytest.approx(5.0)

    def test_overlap_violation(self):
        # gap = 5 - 10 = -5 → overlap=5 > max_overlap=3
        v = validate_edge_gap(pos1=0.0, size1=10.0, pos2=5.0, max_overlap=3.0)
        assert v is not None
        assert v.violation_type == "overlap"
        assert v.severity == pytest.approx(2.0)

    def test_exactly_at_max_gap_no_violation(self):
        # gap == max_gap → no violation (not strictly greater)
        result = validate_edge_gap(pos1=0.0, size1=10.0, pos2=15.0, max_gap=5.0)
        assert result is None

    def test_negative_max_gap_raises(self):
        with pytest.raises(ValueError):
            validate_edge_gap(0.0, 10.0, 15.0, max_gap=-1.0)

    def test_negative_max_overlap_raises(self):
        with pytest.raises(ValueError):
            validate_edge_gap(0.0, 10.0, 5.0, max_overlap=-1.0)

    def test_returns_boundary_violation_type(self):
        v = validate_edge_gap(0.0, 10.0, 30.0, max_gap=5.0)
        assert isinstance(v, BoundaryViolation)

    def test_params_stored_in_gap_violation(self):
        v = validate_edge_gap(0.0, 10.0, 20.0, max_gap=5.0)
        assert "gap" in v.params


# ─── validate_alignment ──────────────────────────────────────────────────────

class TestValidateAlignment:
    def test_no_violation_zero(self):
        assert validate_alignment(0.0) is None

    def test_no_violation_within_tolerance(self):
        assert validate_alignment(1.5, max_tilt_deg=2.0) is None

    def test_tilt_violation(self):
        v = validate_alignment(5.0, max_tilt_deg=2.0)
        assert v is not None
        assert v.violation_type == "tilt"
        assert v.severity == pytest.approx(3.0)

    def test_negative_angle(self):
        v = validate_alignment(-5.0, max_tilt_deg=2.0)
        assert v is not None
        assert v.severity == pytest.approx(3.0)

    def test_exactly_at_threshold_no_violation(self):
        assert validate_alignment(2.0, max_tilt_deg=2.0) is None

    def test_zero_max_tilt_raises(self):
        with pytest.raises(ValueError):
            validate_alignment(0.0, max_tilt_deg=0.0)

    def test_negative_max_tilt_raises(self):
        with pytest.raises(ValueError):
            validate_alignment(0.0, max_tilt_deg=-1.0)

    def test_params_stored(self):
        v = validate_alignment(10.0, max_tilt_deg=2.0)
        assert "angle_deg" in v.params


# ─── validate_pair ───────────────────────────────────────────────────────────

class TestValidatePair:
    def test_no_violations(self):
        result = validate_pair(
            idx1=0, idx2=1,
            pos1=0.0, size1=10.0, pos2=10.0,
            angle_deg=0.0,
        )
        assert result == []

    def test_returns_list(self):
        result = validate_pair(0, 1, 0.0, 10.0, 30.0)
        assert isinstance(result, list)

    def test_gap_violation_detected(self):
        violations = validate_pair(
            idx1=0, idx2=1,
            pos1=0.0, size1=10.0, pos2=30.0,
            max_gap=5.0,
        )
        types = [v.violation_type for v in violations]
        assert "gap" in types

    def test_tilt_violation_detected(self):
        violations = validate_pair(
            idx1=0, idx2=1,
            pos1=0.0, size1=10.0, pos2=10.0,
            angle_deg=10.0, max_tilt_deg=2.0,
        )
        types = [v.violation_type for v in violations]
        assert "tilt" in types

    def test_both_violations(self):
        violations = validate_pair(
            idx1=0, idx2=1,
            pos1=0.0, size1=10.0, pos2=30.0,
            angle_deg=10.0,
            max_gap=5.0, max_tilt_deg=2.0,
        )
        assert len(violations) == 2

    def test_indices_set_on_violation(self):
        violations = validate_pair(
            idx1=4, idx2=7,
            pos1=0.0, size1=10.0, pos2=30.0,
            max_gap=5.0,
        )
        assert violations[0].idx1 == 4
        assert violations[0].idx2 == 7


# ─── validate_all_pairs ──────────────────────────────────────────────────────

class TestValidateAllPairs:
    def test_empty_pairs_returns_report(self):
        report = validate_all_pairs([], [], [])
        assert isinstance(report, BoundaryReport)
        assert report.n_pairs_checked == 0
        assert report.is_valid is True

    def test_single_pair_no_violation(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 10.0],
            sizes=[10.0, 10.0],
        )
        assert report.n_pairs_checked == 1
        assert len(report.violations) == 0
        assert report.is_valid is True

    def test_violation_detected(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 30.0],
            sizes=[10.0, 10.0],
            max_gap=5.0,
        )
        assert len(report.violations) > 0

    def test_is_valid_false_on_violation(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 30.0],
            sizes=[10.0, 10.0],
            max_gap=5.0,
        )
        assert report.is_valid is False

    def test_multiple_pairs(self):
        # 3 fragments in a row, perfectly aligned
        report = validate_all_pairs(
            pairs=[(0, 1), (1, 2)],
            positions=[0.0, 10.0, 20.0],
            sizes=[10.0, 10.0, 10.0],
        )
        assert report.n_pairs_checked == 2
        assert report.is_valid is True

    def test_angles_forwarded(self):
        # large tilt angle should produce violation
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 10.0],
            sizes=[10.0, 10.0],
            angles=[15.0],
            max_tilt_deg=2.0,
        )
        assert any(v.violation_type == "tilt" for v in report.violations)

    def test_overall_score_in_unit_interval(self):
        report = validate_all_pairs(
            pairs=[(0, 1)],
            positions=[0.0, 30.0],
            sizes=[10.0, 10.0],
            max_gap=5.0,
        )
        assert 0.0 < report.overall_score <= 1.0

    def test_params_stored_in_report(self):
        report = validate_all_pairs(
            [], [], [], max_gap=7.0, max_overlap=4.0,
        )
        assert report.params["max_gap"] == pytest.approx(7.0)
        assert report.params["max_overlap"] == pytest.approx(4.0)


# ─── boundary_quality_score ──────────────────────────────────────────────────

class TestBoundaryQualityScore:
    def test_no_violations_returns_one(self):
        assert boundary_quality_score([], n_pairs=5) == pytest.approx(1.0)

    def test_violations_reduce_score(self):
        v = BoundaryViolation(0, 1, "gap", severity=2.0)
        score = boundary_quality_score([v], n_pairs=1)
        assert score < 1.0

    def test_score_in_unit_interval(self):
        v = BoundaryViolation(0, 1, "overlap", severity=5.0)
        score = boundary_quality_score([v], n_pairs=1)
        assert 0.0 < score <= 1.0

    def test_n_pairs_less_than_1_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=0)

    def test_decay_zero_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=1, decay=0.0)

    def test_decay_negative_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=1, decay=-1.0)

    def test_higher_severity_lower_score(self):
        v_small = BoundaryViolation(0, 1, "gap", severity=1.0)
        v_large = BoundaryViolation(0, 1, "gap", severity=10.0)
        s_small = boundary_quality_score([v_small], n_pairs=1)
        s_large = boundary_quality_score([v_large], n_pairs=1)
        assert s_small > s_large

    def test_more_pairs_dilutes_severity(self):
        v = BoundaryViolation(0, 1, "gap", severity=5.0)
        s1 = boundary_quality_score([v], n_pairs=1)
        s10 = boundary_quality_score([v], n_pairs=10)
        assert s10 > s1
