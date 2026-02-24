"""Тесты для puzzle_reconstruction/verification/boundary_validator.py."""
import pytest
import numpy as np

from puzzle_reconstruction.verification.boundary_validator import (
    BoundaryViolation,
    BoundaryReport,
    validate_edge_gap,
    validate_alignment,
    validate_pair,
    validate_all_pairs,
    boundary_quality_score,
)


# ─── BoundaryViolation ────────────────────────────────────────────────────────

class TestBoundaryViolation:
    def test_creation(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=2.5)
        assert v.idx1 == 0
        assert v.idx2 == 1
        assert v.violation_type == "gap"
        assert v.severity == pytest.approx(2.5)
        assert v.message == ""
        assert v.params == {}

    def test_message_stored(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="overlap",
                              severity=1.0, message="Too much overlap")
        assert v.message == "Too much overlap"

    def test_params_stored(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="tilt",
                              severity=0.5, params={"angle": 5.0})
        assert v.params["angle"] == pytest.approx(5.0)

    def test_valid_types(self):
        for vtype in ("gap", "overlap", "tilt"):
            v = BoundaryViolation(idx1=0, idx2=1, violation_type=vtype, severity=0.0)
            assert v.violation_type == vtype

    def test_no_validation_on_creation(self):
        # No __post_init__ validation
        v = BoundaryViolation(idx1=-1, idx2=99, violation_type="unknown",
                              severity=-1.0)
        assert v.severity == pytest.approx(-1.0)


# ─── BoundaryReport ───────────────────────────────────────────────────────────

class TestBoundaryReport:
    def test_creation_defaults(self):
        r = BoundaryReport()
        assert r.violations == []
        assert r.n_pairs_checked == 0
        assert r.is_valid is True
        assert r.overall_score == pytest.approx(1.0)
        assert r.params == {}

    def test_violations_stored(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        r = BoundaryReport(violations=[v], n_pairs_checked=1)
        assert len(r.violations) == 1

    def test_is_valid_stored(self):
        r = BoundaryReport(is_valid=False)
        assert r.is_valid is False

    def test_overall_score_stored(self):
        r = BoundaryReport(overall_score=0.75)
        assert r.overall_score == pytest.approx(0.75)

    def test_params_stored(self):
        r = BoundaryReport(params={"max_gap": 5.0})
        assert r.params["max_gap"] == pytest.approx(5.0)


# ─── validate_edge_gap ────────────────────────────────────────────────────────

class TestValidateEdgeGap:
    def test_no_violation_exact_fit(self):
        # pos2 == pos1 + size1 → gap = 0, no violation
        result = validate_edge_gap(0.0, 10.0, 10.0, max_gap=5.0, max_overlap=3.0)
        assert result is None

    def test_gap_within_limit_returns_none(self):
        # gap = 3.0, max_gap = 5.0 → no violation
        result = validate_edge_gap(0.0, 10.0, 13.0, max_gap=5.0, max_overlap=3.0)
        assert result is None

    def test_gap_exceeds_limit(self):
        # gap = 8.0, max_gap = 5.0 → violation "gap"
        result = validate_edge_gap(0.0, 10.0, 18.0, max_gap=5.0, max_overlap=3.0)
        assert result is not None
        assert result.violation_type == "gap"

    def test_gap_violation_severity(self):
        # gap = 8, max_gap = 5 → severity = 3
        result = validate_edge_gap(0.0, 10.0, 18.0, max_gap=5.0, max_overlap=3.0)
        assert result.severity == pytest.approx(3.0)

    def test_overlap_within_limit_returns_none(self):
        # overlap = 2.0 (gap = -2), max_overlap = 3.0 → no violation
        result = validate_edge_gap(0.0, 10.0, 8.0, max_gap=5.0, max_overlap=3.0)
        assert result is None

    def test_overlap_exceeds_limit(self):
        # overlap = 5.0 (gap = -5), max_overlap = 3.0 → violation "overlap"
        result = validate_edge_gap(0.0, 10.0, 5.0, max_gap=5.0, max_overlap=3.0)
        assert result is not None
        assert result.violation_type == "overlap"

    def test_overlap_violation_severity(self):
        # overlap = 5.0, max_overlap = 3.0 → severity = 2.0
        result = validate_edge_gap(0.0, 10.0, 5.0, max_gap=5.0, max_overlap=3.0)
        assert result.severity == pytest.approx(2.0)

    def test_negative_max_gap_raises(self):
        with pytest.raises(ValueError):
            validate_edge_gap(0.0, 10.0, 10.0, max_gap=-1.0)

    def test_negative_max_overlap_raises(self):
        with pytest.raises(ValueError):
            validate_edge_gap(0.0, 10.0, 10.0, max_overlap=-1.0)

    def test_returns_boundary_violation(self):
        result = validate_edge_gap(0.0, 10.0, 20.0, max_gap=5.0)
        assert isinstance(result, BoundaryViolation)

    def test_zero_max_gap_any_positive_gap_violates(self):
        result = validate_edge_gap(0.0, 10.0, 10.1, max_gap=0.0)
        assert result is not None
        assert result.violation_type == "gap"

    def test_zero_max_overlap_any_overlap_violates(self):
        result = validate_edge_gap(0.0, 10.0, 9.9, max_overlap=0.0)
        assert result is not None
        assert result.violation_type == "overlap"


# ─── validate_alignment ───────────────────────────────────────────────────────

class TestValidateAlignment:
    def test_no_violation_zero_angle(self):
        result = validate_alignment(0.0, max_tilt_deg=2.0)
        assert result is None

    def test_no_violation_within_limit(self):
        result = validate_alignment(1.5, max_tilt_deg=2.0)
        assert result is None

    def test_violation_exceeds_limit(self):
        result = validate_alignment(3.0, max_tilt_deg=2.0)
        assert result is not None
        assert result.violation_type == "tilt"

    def test_severity_correct(self):
        # 3.0 - 2.0 = 1.0
        result = validate_alignment(3.0, max_tilt_deg=2.0)
        assert result.severity == pytest.approx(1.0)

    def test_negative_angle_absolute_value(self):
        result = validate_alignment(-3.0, max_tilt_deg=2.0)
        assert result is not None
        assert result.violation_type == "tilt"
        assert result.severity == pytest.approx(1.0)

    def test_zero_max_tilt_raises(self):
        with pytest.raises(ValueError, match="max_tilt_deg"):
            validate_alignment(0.0, max_tilt_deg=0.0)

    def test_negative_max_tilt_raises(self):
        with pytest.raises(ValueError):
            validate_alignment(0.0, max_tilt_deg=-1.0)

    def test_exact_limit_no_violation(self):
        result = validate_alignment(2.0, max_tilt_deg=2.0)
        assert result is None

    def test_returns_boundary_violation(self):
        result = validate_alignment(5.0, max_tilt_deg=2.0)
        assert isinstance(result, BoundaryViolation)

    def test_violation_contains_angle_params(self):
        result = validate_alignment(5.0, max_tilt_deg=2.0)
        assert "angle_deg" in result.params
        assert result.params["angle_deg"] == pytest.approx(5.0)


# ─── validate_pair ────────────────────────────────────────────────────────────

class TestValidatePair:
    def test_returns_list(self):
        result = validate_pair(0, 1, 0.0, 10.0, 10.0)
        assert isinstance(result, list)

    def test_no_violations_returns_empty(self):
        # Perfect fit, zero tilt
        result = validate_pair(0, 1, 0.0, 10.0, 10.0, angle_deg=0.0,
                               max_gap=5.0, max_overlap=3.0, max_tilt_deg=2.0)
        assert result == []

    def test_gap_violation_detected(self):
        # Large gap
        result = validate_pair(0, 1, 0.0, 10.0, 20.0, angle_deg=0.0, max_gap=5.0)
        assert len(result) >= 1
        types = [v.violation_type for v in result]
        assert "gap" in types

    def test_tilt_violation_detected(self):
        # Large tilt
        result = validate_pair(0, 1, 0.0, 10.0, 10.0, angle_deg=10.0, max_tilt_deg=2.0)
        types = [v.violation_type for v in result]
        assert "tilt" in types

    def test_both_violations_detected(self):
        # Both gap and tilt
        result = validate_pair(0, 1, 0.0, 10.0, 20.0, angle_deg=5.0,
                               max_gap=5.0, max_tilt_deg=2.0)
        types = [v.violation_type for v in result]
        assert "gap" in types
        assert "tilt" in types

    def test_idx1_idx2_stored_on_violations(self):
        result = validate_pair(3, 7, 0.0, 10.0, 20.0, max_gap=5.0)
        for v in result:
            assert v.idx1 == 3
            assert v.idx2 == 7

    def test_overlap_violation_detected(self):
        # Large overlap
        result = validate_pair(0, 1, 0.0, 10.0, 5.0, max_overlap=3.0)
        types = [v.violation_type for v in result]
        assert "overlap" in types

    def test_returns_boundary_violations(self):
        result = validate_pair(0, 1, 0.0, 10.0, 25.0)
        for v in result:
            assert isinstance(v, BoundaryViolation)


# ─── validate_all_pairs ───────────────────────────────────────────────────────

class TestValidateAllPairs:
    def test_returns_boundary_report(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes)
        assert isinstance(result, BoundaryReport)

    def test_n_pairs_checked(self):
        pairs = [(0, 1), (1, 2)]
        positions = [0.0, 10.0, 20.0]
        sizes = [10.0, 10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes)
        assert result.n_pairs_checked == 2

    def test_no_violations_is_valid(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes)
        assert result.is_valid is True

    def test_violations_collected(self):
        pairs = [(0, 1)]
        positions = [0.0, 30.0]  # gap = 20
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes, max_gap=5.0)
        assert len(result.violations) >= 1

    def test_invalid_when_violations(self):
        pairs = [(0, 1)]
        positions = [0.0, 30.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes, max_gap=5.0)
        assert result.is_valid is False

    def test_empty_pairs_returns_valid_report(self):
        # Empty pairs → no violations → valid report (n_pairs=0 allowed)
        result = validate_all_pairs([], [], [])
        assert isinstance(result, BoundaryReport)
        assert result.is_valid is True

    def test_angles_none_defaults_to_zero(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes, angles=None)
        assert isinstance(result, BoundaryReport)

    def test_custom_angles(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes,
                                    angles=[5.0], max_tilt_deg=2.0)
        types = [v.violation_type for v in result.violations]
        assert "tilt" in types

    def test_overall_score_in_0_1(self):
        pairs = [(0, 1), (1, 2)]
        positions = [0.0, 10.0, 20.0]
        sizes = [10.0, 10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes)
        assert 0.0 <= result.overall_score <= 1.0

    def test_params_stored(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes, max_gap=3.0)
        assert result.params["max_gap"] == pytest.approx(3.0)


# ─── boundary_quality_score ───────────────────────────────────────────────────

class TestBoundaryQualityScore:
    def test_no_violations_returns_1(self):
        score = boundary_quality_score([], n_pairs=1)
        assert score == pytest.approx(1.0)

    def test_n_pairs_less_than_1_raises(self):
        with pytest.raises(ValueError, match="n_pairs"):
            boundary_quality_score([], n_pairs=0)

    def test_decay_zero_raises(self):
        with pytest.raises(ValueError, match="decay"):
            boundary_quality_score([], n_pairs=1, decay=0.0)

    def test_decay_negative_raises(self):
        with pytest.raises(ValueError):
            boundary_quality_score([], n_pairs=1, decay=-1.0)

    def test_returns_float(self):
        score = boundary_quality_score([], n_pairs=1)
        assert isinstance(score, float)

    def test_with_violation_score_less_than_1(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=2.0)
        score = boundary_quality_score([v], n_pairs=1, decay=1.0)
        assert score < 1.0

    def test_score_in_0_1(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=5.0)
        score = boundary_quality_score([v], n_pairs=1, decay=1.0)
        assert 0.0 < score <= 1.0

    def test_higher_severity_lower_score(self):
        v_low = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        v_high = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=5.0)
        score_low = boundary_quality_score([v_low], n_pairs=1)
        score_high = boundary_quality_score([v_high], n_pairs=1)
        assert score_high < score_low

    def test_higher_decay_lower_score(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        score_low_decay = boundary_quality_score([v], n_pairs=1, decay=0.5)
        score_high_decay = boundary_quality_score([v], n_pairs=1, decay=2.0)
        assert score_high_decay < score_low_decay

    def test_known_value(self):
        # mean_severity = 2.0 / 1 = 2.0; decay=1.0 → exp(-2.0)
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=2.0)
        score = boundary_quality_score([v], n_pairs=1, decay=1.0)
        assert score == pytest.approx(np.exp(-2.0))

    def test_multiple_violations_averaged(self):
        # severity total = 1.0+3.0=4.0, n_pairs=2, mean=2.0 → exp(-2)
        v1 = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        v2 = BoundaryViolation(idx1=1, idx2=2, violation_type="gap", severity=3.0)
        score = boundary_quality_score([v1, v2], n_pairs=2, decay=1.0)
        assert score == pytest.approx(np.exp(-2.0))

    def test_n_pairs_1_valid(self):
        score = boundary_quality_score([], n_pairs=1)
        assert score == pytest.approx(1.0)
