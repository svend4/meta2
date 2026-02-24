"""Extra tests for puzzle_reconstruction/verification/fragment_validator.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.fragment_validator import (
    FragmentValidatorParams,
    ValidationIssue,
    ValidationResult,
    batch_validate,
    filter_valid,
    validate_aspect_ratio,
    validate_content_coverage,
    validate_contour,
    validate_dimensions,
    validate_fragment,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _black(h=32, w=32):
    return np.zeros((h, w), dtype=np.uint8)


def _bgr(h=32, w=32, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _square_contour(size=20.0):
    return np.array([
        [0.0, 0.0], [size, 0.0], [size, size], [0.0, size],
    ], dtype=np.float64)


# ─── ValidationIssue (extra) ─────────────────────────────────────────────────

class TestValidationIssueExtra:
    def test_code_stored(self):
        issue = ValidationIssue(code="too_small", message="Too small")
        assert issue.code == "too_small"

    def test_message_stored(self):
        issue = ValidationIssue(code="x", message="Hello")
        assert issue.message == "Hello"

    def test_default_severity_error(self):
        issue = ValidationIssue(code="x", message="y")
        assert issue.severity == "error"

    def test_warning_severity(self):
        issue = ValidationIssue(code="x", message="y", severity="warning")
        assert issue.severity == "warning"

    def test_custom_severity_error(self):
        issue = ValidationIssue(code="x", message="y", severity="error")
        assert issue.severity == "error"


# ─── ValidationResult (extra) ────────────────────────────────────────────────

class TestValidationResultExtra:
    def test_default_passed_true(self):
        assert ValidationResult().passed is True

    def test_default_fragment_idx(self):
        assert ValidationResult().fragment_idx == -1

    def test_default_issues_empty(self):
        assert ValidationResult().issues == []

    def test_default_metrics_empty(self):
        assert ValidationResult().metrics == {}

    def test_add_error_sets_failed(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("e", "err", severity="error"))
        assert r.passed is False

    def test_add_warning_keeps_passed(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("w", "warn", severity="warning"))
        assert r.passed is True

    def test_multiple_issues_accumulated(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("a", "A", severity="error"))
        r.add_issue(ValidationIssue("b", "B", severity="warning"))
        assert len(r.issues) == 2

    def test_fragment_idx_stored(self):
        r = ValidationResult(fragment_idx=99)
        assert r.fragment_idx == 99

    def test_metrics_stored(self):
        r = ValidationResult(metrics={"width": 32.0})
        assert r.metrics["width"] == pytest.approx(32.0)

    def test_two_errors_both_issues(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("a", "A"))
        r.add_issue(ValidationIssue("b", "B"))
        assert len(r.issues) == 2
        assert r.passed is False


# ─── FragmentValidatorParams (extra) ─────────────────────────────────────────

class TestFragmentValidatorParamsExtra:
    def test_min_width_ge_1(self):
        assert FragmentValidatorParams().min_width >= 1

    def test_min_height_ge_1(self):
        assert FragmentValidatorParams().min_height >= 1

    def test_min_aspect_ratio_in_range(self):
        ar = FragmentValidatorParams().min_aspect_ratio
        assert 0.0 <= ar <= 1.0

    def test_min_coverage_in_range(self):
        cov = FragmentValidatorParams().min_coverage
        assert 0.0 <= cov <= 1.0

    def test_min_contour_points_ge_3(self):
        assert FragmentValidatorParams().min_contour_points >= 3

    def test_min_width_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_width=0)

    def test_min_height_zero_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_height=0)

    def test_min_aspect_ratio_neg_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_aspect_ratio=-0.1)

    def test_min_aspect_ratio_gt_one_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_aspect_ratio=1.1)

    def test_min_coverage_neg_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_coverage=-0.1)

    def test_min_coverage_gt_one_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_coverage=1.5)

    def test_min_contour_points_lt_3_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_contour_points=2)

    def test_custom_values_stored(self):
        p = FragmentValidatorParams(min_width=64, min_height=64, min_coverage=0.3)
        assert p.min_width == 64
        assert p.min_height == 64
        assert p.min_coverage == pytest.approx(0.3)


# ─── validate_dimensions (extra) ─────────────────────────────────────────────

class TestValidateDimensionsExtra:
    def test_valid_passes(self):
        p = FragmentValidatorParams(min_width=16, min_height=16)
        r = validate_dimensions(_gray(32, 32), p)
        assert r.passed is True

    def test_too_narrow_fails(self):
        p = FragmentValidatorParams(min_width=50)
        r = validate_dimensions(_gray(32, 20), p)
        assert r.passed is False
        assert any(i.code == "too_narrow" for i in r.issues)

    def test_too_short_fails(self):
        p = FragmentValidatorParams(min_height=50)
        r = validate_dimensions(_gray(20, 32), p)
        assert r.passed is False
        assert any(i.code == "too_short" for i in r.issues)

    def test_too_wide_fails(self):
        p = FragmentValidatorParams(max_width=10)
        r = validate_dimensions(_gray(32, 32), p)
        assert any(i.code == "too_wide" for i in r.issues)

    def test_too_tall_fails(self):
        p = FragmentValidatorParams(max_height=10)
        r = validate_dimensions(_gray(32, 32), p)
        assert any(i.code == "too_tall" for i in r.issues)

    def test_metrics_width(self):
        r = validate_dimensions(_gray(20, 30))
        assert r.metrics["width"] == pytest.approx(30.0)

    def test_metrics_height(self):
        r = validate_dimensions(_gray(20, 30))
        assert r.metrics["height"] == pytest.approx(20.0)

    def test_metrics_area(self):
        r = validate_dimensions(_gray(20, 30))
        assert r.metrics["area"] == pytest.approx(600.0)

    def test_max_width_zero_no_upper_check(self):
        p = FragmentValidatorParams(max_width=0)
        r = validate_dimensions(_gray(32, 1000), p)
        assert not any(i.code == "too_wide" for i in r.issues)

    def test_bgr_accepted(self):
        r = validate_dimensions(_bgr(32, 32))
        assert "width" in r.metrics


# ─── validate_aspect_ratio (extra) ───────────────────────────────────────────

class TestValidateAspectRatioExtra:
    def test_square_passes(self):
        p = FragmentValidatorParams(min_aspect_ratio=0.1)
        r = validate_aspect_ratio(_gray(32, 32), p)
        assert r.passed is True

    def test_aspect_ratio_metric_square(self):
        r = validate_aspect_ratio(_gray(32, 32))
        assert r.metrics["aspect_ratio"] == pytest.approx(1.0)

    def test_extreme_fails(self):
        p = FragmentValidatorParams(min_aspect_ratio=0.5)
        r = validate_aspect_ratio(_gray(4, 100), p)
        assert r.passed is False
        assert any(i.code == "extreme_aspect_ratio" for i in r.issues)

    def test_metric_in_unit_interval(self):
        r = validate_aspect_ratio(_gray(10, 40))
        assert 0.0 <= r.metrics["aspect_ratio"] <= 1.0

    def test_tall_same_ar_as_wide(self):
        r_wide = validate_aspect_ratio(_gray(10, 40))
        r_tall = validate_aspect_ratio(_gray(40, 10))
        assert r_wide.metrics["aspect_ratio"] == pytest.approx(
            r_tall.metrics["aspect_ratio"])

    def test_returns_validation_result(self):
        assert isinstance(validate_aspect_ratio(_gray()), ValidationResult)


# ─── validate_content_coverage (extra) ───────────────────────────────────────

class TestValidateContentCoverageExtra:
    def test_full_gray_passes(self):
        p = FragmentValidatorParams(min_coverage=0.5)
        r = validate_content_coverage(_gray(value=200), p)
        assert r.passed is True

    def test_coverage_metric_near_one(self):
        r = validate_content_coverage(_gray(value=200))
        assert r.metrics["coverage"] == pytest.approx(1.0, abs=1e-3)

    def test_black_fails(self):
        p = FragmentValidatorParams(min_coverage=0.1)
        r = validate_content_coverage(_black(), p)
        assert r.passed is False
        assert any(i.code == "insufficient_coverage" for i in r.issues)

    def test_coverage_in_range(self):
        r = validate_content_coverage(_gray(value=100))
        assert 0.0 <= r.metrics["coverage"] <= 1.0

    def test_bgr_accepted(self):
        r = validate_content_coverage(_bgr(value=200), threshold=10)
        assert r.metrics["coverage"] > 0.0

    def test_content_pixels_full(self):
        img = _gray(32, 32, value=200)
        r = validate_content_coverage(img, threshold=10)
        assert r.metrics["content_pixels"] == pytest.approx(32 * 32, rel=1e-3)

    def test_custom_threshold_filters_low(self):
        img = np.full((32, 32), 5, dtype=np.uint8)
        r = validate_content_coverage(img, threshold=10)
        assert r.metrics["coverage"] == pytest.approx(0.0)


# ─── validate_contour (extra) ────────────────────────────────────────────────

class TestValidateContourExtra:
    def test_square_passes(self):
        p = FragmentValidatorParams(min_contour_area=10.0)
        r = validate_contour(_square_contour(20), p)
        assert r.passed is True

    def test_too_few_points_fails(self):
        p = FragmentValidatorParams(min_contour_points=5)
        c = _square_contour()  # 4 points
        r = validate_contour(c, p)
        assert r.passed is False
        assert any(i.code == "too_few_contour_points" for i in r.issues)

    def test_n_points_metric(self):
        r = validate_contour(_square_contour())
        assert r.metrics["n_points"] == pytest.approx(4.0)

    def test_area_metric_positive(self):
        r = validate_contour(_square_contour(20))
        assert r.metrics["contour_area"] > 0.0

    def test_small_area_fails(self):
        p = FragmentValidatorParams(min_contour_area=10000.0)
        r = validate_contour(_square_contour(5), p)
        assert r.passed is False

    def test_two_point_contour_fails(self):
        c = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        r = validate_contour(c)
        assert r.passed is False

    def test_degenerate_duplicate_points_fails(self):
        c = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
                     dtype=np.float64)
        r = validate_contour(c)
        assert r.passed is False

    def test_cv2_format_contour(self):
        c = _square_contour().reshape(-1, 1, 2)
        r = validate_contour(c)
        assert r.metrics["n_points"] == pytest.approx(4.0)

    def test_too_many_points_warning(self):
        p = FragmentValidatorParams(max_contour_points=3)
        c = _square_contour()  # 4 points
        r = validate_contour(c, p)
        warn_issues = [i for i in r.issues if i.code == "too_many_contour_points"]
        assert len(warn_issues) > 0
        assert warn_issues[0].severity == "warning"


# ─── validate_fragment (extra) ───────────────────────────────────────────────

class TestValidateFragmentExtra:
    def test_returns_validation_result(self):
        assert isinstance(validate_fragment(_gray()), ValidationResult)

    def test_fragment_idx_stored(self):
        r = validate_fragment(_gray(), fragment_idx=7)
        assert r.fragment_idx == 7

    def test_valid_fragment_passes(self):
        p = FragmentValidatorParams(min_width=16, min_height=16, min_coverage=0.05)
        r = validate_fragment(_gray(32, 32, 128), params=p)
        assert r.passed is True

    def test_too_narrow_fails(self):
        p = FragmentValidatorParams(min_width=100)
        r = validate_fragment(_gray(32, 32), params=p)
        assert r.passed is False

    def test_coverage_fail(self):
        p = FragmentValidatorParams(min_coverage=0.9)
        r = validate_fragment(_black(), params=p)
        assert any(i.code == "insufficient_coverage" for i in r.issues)

    def test_with_contour_checks_contour(self):
        p = FragmentValidatorParams(min_contour_area=10000.0)
        r = validate_fragment(_gray(), contour=_square_contour(5), params=p)
        assert any(i.code == "degenerate_contour" for i in r.issues)

    def test_without_contour_no_contour_issues(self):
        r = validate_fragment(_gray())
        contour_codes = {"too_few_contour_points", "too_many_contour_points",
                         "degenerate_contour"}
        assert not any(i.code in contour_codes for i in r.issues)

    def test_metrics_aggregated(self):
        r = validate_fragment(_gray(32, 40))
        assert "width" in r.metrics
        assert "height" in r.metrics
        assert "coverage" in r.metrics

    def test_default_params_ok(self):
        r = validate_fragment(_gray(32, 32))
        assert isinstance(r, ValidationResult)


# ─── batch_validate (extra) ──────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_empty_returns_empty(self):
        assert batch_validate([]) == []

    def test_length_preserved(self):
        imgs = [_gray()] * 4
        assert len(batch_validate(imgs)) == 4

    def test_fragment_idx_sequential(self):
        imgs = [_gray()] * 3
        results = batch_validate(imgs)
        assert [r.fragment_idx for r in results] == [0, 1, 2]

    def test_all_validation_results(self):
        for r in batch_validate([_gray(), _black(), _bgr()]):
            assert isinstance(r, ValidationResult)

    def test_contour_applied(self):
        p = FragmentValidatorParams(min_contour_area=10000.0)
        results = batch_validate([_gray(32, 32)],
                                  contours=[_square_contour(2)], params=p)
        assert results[0].passed is False

    def test_none_contours_skipped(self):
        results = batch_validate([_gray(), _gray()], contours=[None, None])
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_single_image(self):
        results = batch_validate([_gray()])
        assert len(results) == 1


# ─── filter_valid (extra) ────────────────────────────────────────────────────

class TestFilterValidExtra:
    def test_all_pass(self):
        results = [ValidationResult(fragment_idx=i, passed=True) for i in range(4)]
        assert filter_valid(results) == [0, 1, 2, 3]

    def test_none_pass(self):
        results = [ValidationResult(fragment_idx=i, passed=False) for i in range(3)]
        assert filter_valid(results) == []

    def test_mixed(self):
        results = [
            ValidationResult(fragment_idx=0, passed=True),
            ValidationResult(fragment_idx=1, passed=False),
            ValidationResult(fragment_idx=2, passed=True),
        ]
        assert filter_valid(results) == [0, 2]

    def test_empty_input(self):
        assert filter_valid([]) == []

    def test_uses_fragment_idx(self):
        results = [
            ValidationResult(fragment_idx=10, passed=True),
            ValidationResult(fragment_idx=20, passed=True),
        ]
        assert filter_valid(results) == [10, 20]

    def test_returns_list(self):
        assert isinstance(filter_valid([]), list)

    def test_single_pass(self):
        r = [ValidationResult(fragment_idx=5, passed=True)]
        assert filter_valid(r) == [5]

    def test_single_fail(self):
        r = [ValidationResult(fragment_idx=5, passed=False)]
        assert filter_valid(r) == []
