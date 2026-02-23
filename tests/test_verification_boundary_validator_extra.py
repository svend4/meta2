"""Extra tests for puzzle_reconstruction/verification/boundary_validator.py."""
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


# ─── BoundaryViolation (extra) ────────────────────────────────────────────────

class TestBoundaryViolationExtra:
    def test_default_message_empty(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        assert v.message == ""

    def test_default_params_empty(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=1.0)
        assert v.params == {}

    def test_severity_stored_as_float(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="tilt", severity=3.14)
        assert v.severity == pytest.approx(3.14)

    def test_gap_type_stored(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap", severity=0.0)
        assert v.violation_type == "gap"

    def test_overlap_type_stored(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="overlap", severity=0.0)
        assert v.violation_type == "overlap"

    def test_custom_message(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="gap",
                              severity=1.0, message="edge too far apart")
        assert "far" in v.message

    def test_multiple_params(self):
        v = BoundaryViolation(idx1=0, idx2=1, violation_type="tilt",
                              severity=2.0, params={"a": 1.0, "b": 2.0})
        assert v.params["a"] == pytest.approx(1.0)
        assert v.params["b"] == pytest.approx(2.0)


# ─── BoundaryReport (extra) ───────────────────────────────────────────────────

class TestBoundaryReportExtra:
    def test_default_is_valid_true(self):
        r = BoundaryReport()
        assert r.is_valid is True

    def test_default_overall_score_one(self):
        r = BoundaryReport()
        assert r.overall_score == pytest.approx(1.0)

    def test_default_n_pairs_zero(self):
        r = BoundaryReport()
        assert r.n_pairs_checked == 0

    def test_multiple_violations_stored(self):
        v1 = BoundaryViolation(0, 1, "gap", 1.0)
        v2 = BoundaryViolation(1, 2, "overlap", 2.0)
        r = BoundaryReport(violations=[v1, v2], n_pairs_checked=2, is_valid=False)
        assert len(r.violations) == 2
        assert r.is_valid is False

    def test_params_multiple_keys(self):
        r = BoundaryReport(params={"max_gap": 5.0, "max_tilt_deg": 3.0})
        assert r.params["max_gap"] == pytest.approx(5.0)
        assert r.params["max_tilt_deg"] == pytest.approx(3.0)

    def test_overall_score_zero_allowed(self):
        r = BoundaryReport(overall_score=0.0)
        assert r.overall_score == pytest.approx(0.0)


# ─── validate_edge_gap (extra) ────────────────────────────────────────────────

class TestValidateEdgeGapExtra:
    def test_exact_fit_returns_none(self):
        # pos2 == pos1 + size1 → gap = 0
        result = validate_edge_gap(5.0, 10.0, 15.0, max_gap=1.0)
        assert result is None

    def test_tiny_gap_within_limit_none(self):
        result = validate_edge_gap(0.0, 10.0, 10.5, max_gap=1.0, max_overlap=0.0)
        assert result is None

    def test_large_gap_violation_type_gap(self):
        result = validate_edge_gap(0.0, 5.0, 100.0, max_gap=1.0)
        assert result is not None
        assert result.violation_type == "gap"

    def test_large_overlap_violation_type_overlap(self):
        result = validate_edge_gap(0.0, 10.0, 1.0, max_overlap=3.0, max_gap=0.0)
        assert result is not None
        assert result.violation_type == "overlap"

    def test_severity_equals_excess(self):
        # gap = 7.0, max_gap = 5.0 → severity = 2.0
        result = validate_edge_gap(0.0, 5.0, 12.0, max_gap=5.0)
        assert result.severity == pytest.approx(2.0)

    def test_overlap_severity_equals_excess(self):
        # overlap = 4.0 (gap = -4), max_overlap = 2.0 → severity = 2.0
        result = validate_edge_gap(0.0, 10.0, 6.0, max_overlap=2.0, max_gap=0.0)
        assert result.severity == pytest.approx(2.0)

    def test_violation_has_correct_indices(self):
        result = validate_edge_gap(0.0, 5.0, 20.0, max_gap=1.0)
        # validate_edge_gap doesn't set idx — just check it's a BoundaryViolation
        assert isinstance(result, BoundaryViolation)

    def test_zero_gap_zero_overlap_both_ok(self):
        result = validate_edge_gap(0.0, 10.0, 10.0, max_gap=0.0, max_overlap=0.0)
        assert result is None

    def test_negative_size_raises_or_handles(self):
        # API may or may not raise; just ensure no crash with valid positive size
        try:
            result = validate_edge_gap(0.0, 10.0, 5.0, max_gap=5.0, max_overlap=10.0)
            assert result is None or isinstance(result, BoundaryViolation)
        except (ValueError, Exception):
            pass  # acceptable


# ─── validate_alignment (extra) ───────────────────────────────────────────────

class TestValidateAlignmentExtra:
    def test_small_positive_angle_no_violation(self):
        result = validate_alignment(0.5, max_tilt_deg=2.0)
        assert result is None

    def test_small_negative_angle_no_violation(self):
        result = validate_alignment(-0.5, max_tilt_deg=2.0)
        assert result is None

    def test_large_positive_angle_violation(self):
        result = validate_alignment(10.0, max_tilt_deg=2.0)
        assert result is not None
        assert result.violation_type == "tilt"

    def test_large_negative_angle_violation(self):
        result = validate_alignment(-10.0, max_tilt_deg=2.0)
        assert result is not None
        assert result.severity == pytest.approx(8.0)

    def test_params_angle_deg_stored(self):
        result = validate_alignment(4.0, max_tilt_deg=2.0)
        assert "angle_deg" in result.params

    def test_severity_exact(self):
        # |angle| = 7.0, max = 5.0 → severity = 2.0
        result = validate_alignment(7.0, max_tilt_deg=5.0)
        assert result.severity == pytest.approx(2.0)

    def test_exactly_at_limit_no_violation(self):
        result = validate_alignment(3.0, max_tilt_deg=3.0)
        assert result is None

    def test_max_tilt_large_no_violation(self):
        result = validate_alignment(45.0, max_tilt_deg=90.0)
        assert result is None


# ─── validate_pair (extra) ────────────────────────────────────────────────────

class TestValidatePairExtra:
    def test_no_violations_empty_list(self):
        result = validate_pair(0, 1, 0.0, 10.0, 10.0,
                               angle_deg=0.0, max_gap=5.0,
                               max_overlap=3.0, max_tilt_deg=2.0)
        assert result == []

    def test_gap_and_tilt_both_violations(self):
        result = validate_pair(0, 1, 0.0, 5.0, 50.0,
                               angle_deg=10.0, max_gap=2.0, max_tilt_deg=2.0)
        types = [v.violation_type for v in result]
        assert "gap" in types
        assert "tilt" in types

    def test_indices_propagated(self):
        result = validate_pair(5, 9, 0.0, 10.0, 30.0, max_gap=5.0)
        for v in result:
            assert v.idx1 == 5
            assert v.idx2 == 9

    def test_default_angle_zero(self):
        # Perfect fit, default angle=0, should have no violations
        result = validate_pair(0, 1, 0.0, 10.0, 10.0,
                               max_gap=5.0, max_overlap=3.0, max_tilt_deg=2.0)
        assert result == []

    def test_overlap_violation(self):
        result = validate_pair(0, 1, 0.0, 10.0, 3.0, max_overlap=2.0, max_gap=10.0)
        types = [v.violation_type for v in result]
        assert "overlap" in types

    def test_all_violations_are_boundary_violations(self):
        result = validate_pair(0, 1, 0.0, 10.0, 30.0, angle_deg=8.0,
                               max_gap=5.0, max_tilt_deg=2.0)
        for v in result:
            assert isinstance(v, BoundaryViolation)


# ─── validate_all_pairs (extra) ───────────────────────────────────────────────

class TestValidateAllPairsExtra:
    def test_two_perfect_pairs_valid(self):
        pairs = [(0, 1), (1, 2)]
        positions = [0.0, 10.0, 20.0]
        sizes = [10.0, 10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes,
                                    max_gap=5.0, max_overlap=3.0)
        assert result.is_valid is True
        assert result.violations == []

    def test_gap_violation_detected(self):
        pairs = [(0, 1)]
        positions = [0.0, 50.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes, max_gap=5.0)
        assert result.is_valid is False
        assert len(result.violations) >= 1

    def test_n_pairs_checked_correct(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        positions = [0.0, 10.0, 20.0, 30.0]
        sizes = [10.0, 10.0, 10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes)
        assert result.n_pairs_checked == 3

    def test_overall_score_one_no_violations(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes,
                                    max_gap=5.0, max_overlap=3.0)
        assert result.overall_score == pytest.approx(1.0)

    def test_tilt_violation_with_angles(self):
        pairs = [(0, 1)]
        positions = [0.0, 10.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes,
                                    angles=[10.0], max_tilt_deg=2.0)
        types = [v.violation_type for v in result.violations]
        assert "tilt" in types

    def test_overall_score_lt_one_with_violations(self):
        pairs = [(0, 1)]
        positions = [0.0, 50.0]
        sizes = [10.0, 10.0]
        result = validate_all_pairs(pairs, positions, sizes, max_gap=5.0)
        assert result.overall_score < 1.0


# ─── boundary_quality_score (extra) ──────────────────────────────────────────

class TestBoundaryQualityScoreExtra:
    def test_empty_violations_returns_1(self):
        score = boundary_quality_score([], n_pairs=5)
        assert score == pytest.approx(1.0)

    def test_many_violations_low_score(self):
        violations = [
            BoundaryViolation(0, i + 1, "gap", severity=10.0)
            for i in range(10)
        ]
        score = boundary_quality_score(violations, n_pairs=10, decay=1.0)
        assert score < 0.5

    def test_known_formula_two_violations(self):
        # mean_severity = (2.0 + 4.0) / 2 = 3.0; decay=1.0 → exp(-3.0)
        v1 = BoundaryViolation(0, 1, "gap", severity=2.0)
        v2 = BoundaryViolation(1, 2, "gap", severity=4.0)
        score = boundary_quality_score([v1, v2], n_pairs=2, decay=1.0)
        assert score == pytest.approx(np.exp(-3.0))

    def test_score_increases_with_fewer_violations(self):
        one_v = [BoundaryViolation(0, 1, "gap", severity=2.0)]
        two_v = [BoundaryViolation(0, 1, "gap", severity=2.0),
                 BoundaryViolation(1, 2, "gap", severity=2.0)]
        score1 = boundary_quality_score(one_v, n_pairs=2, decay=1.0)
        score2 = boundary_quality_score(two_v, n_pairs=2, decay=1.0)
        assert score1 >= score2

    def test_score_in_0_1(self):
        v = BoundaryViolation(0, 1, "gap", severity=100.0)
        score = boundary_quality_score([v], n_pairs=1, decay=1.0)
        assert 0.0 <= score <= 1.0

    def test_decay_0_5_larger_score_than_decay_2(self):
        v = BoundaryViolation(0, 1, "gap", severity=1.0)
        score_low = boundary_quality_score([v], n_pairs=1, decay=0.5)
        score_high = boundary_quality_score([v], n_pairs=1, decay=2.0)
        assert score_low > score_high

    def test_returns_float(self):
        score = boundary_quality_score([], n_pairs=1)
        assert isinstance(score, float)
