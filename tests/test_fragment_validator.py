"""Tests for puzzle_reconstruction.verification.fragment_validator."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _black(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _bgr(h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _square_contour(size: float = 20.0) -> np.ndarray:
    return np.array([
        [0.0, 0.0], [size, 0.0], [size, size], [0.0, size],
    ], dtype=np.float64)


def _collinear_contour() -> np.ndarray:
    """All points on the same line — degenerate."""
    return np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)


def _two_point_contour() -> np.ndarray:
    return np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)


# ─── ValidationIssue ─────────────────────────────────────────────────────────

class TestValidationIssue:
    def test_fields_stored(self):
        issue = ValidationIssue(code="too_small", message="Width too small")
        assert issue.code == "too_small"
        assert issue.message == "Width too small"
        assert issue.severity == "error"

    def test_warning_severity(self):
        issue = ValidationIssue(code="warn", message="Be careful", severity="warning")
        assert issue.severity == "warning"

    def test_custom_severity(self):
        issue = ValidationIssue(code="c", message="m", severity="error")
        assert issue.severity == "error"


# ─── ValidationResult ────────────────────────────────────────────────────────

class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult()
        assert r.fragment_idx == -1
        assert r.passed is True
        assert r.issues == []
        assert r.metrics == {}

    def test_add_error_sets_passed_false(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("e", "err", severity="error"))
        assert r.passed is False

    def test_add_warning_keeps_passed_true(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("w", "warn", severity="warning"))
        assert r.passed is True

    def test_multiple_issues_accumulated(self):
        r = ValidationResult()
        r.add_issue(ValidationIssue("a", "A"))
        r.add_issue(ValidationIssue("b", "B", severity="warning"))
        assert len(r.issues) == 2

    def test_fragment_idx_stored(self):
        r = ValidationResult(fragment_idx=7)
        assert r.fragment_idx == 7

    def test_metrics_stored(self):
        r = ValidationResult(metrics={"width": 100.0})
        assert r.metrics["width"] == pytest.approx(100.0)


# ─── FragmentValidatorParams ─────────────────────────────────────────────────

class TestFragmentValidatorParams:
    def test_defaults(self):
        p = FragmentValidatorParams()
        assert p.min_width >= 1
        assert p.min_height >= 1
        assert 0.0 <= p.min_aspect_ratio <= 1.0
        assert 0.0 <= p.min_coverage <= 1.0
        assert p.min_contour_points >= 3

    def test_min_width_less_than_1_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_width=0)

    def test_min_height_less_than_1_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_height=0)

    def test_min_aspect_ratio_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_aspect_ratio=-0.1)

    def test_min_aspect_ratio_greater_than_1_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_aspect_ratio=1.1)

    def test_min_coverage_negative_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_coverage=-0.1)

    def test_min_coverage_greater_than_1_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_coverage=1.5)

    def test_min_contour_points_less_than_3_raises(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_contour_points=2)

    def test_custom_values_stored(self):
        p = FragmentValidatorParams(
            min_width=32, min_height=32, max_width=512,
            min_aspect_ratio=0.1, min_coverage=0.2
        )
        assert p.min_width == 32
        assert p.max_width == 512
        assert p.min_coverage == pytest.approx(0.2)


# ─── validate_dimensions ─────────────────────────────────────────────────────

class TestValidateDimensions:
    def test_valid_image_passes(self):
        p = FragmentValidatorParams(min_width=16, min_height=16)
        r = validate_dimensions(_gray(32, 32), p)
        assert r.passed is True
        assert r.issues == []

    def test_too_narrow(self):
        p = FragmentValidatorParams(min_width=40)
        r = validate_dimensions(_gray(32, 20), p)
        assert r.passed is False
        codes = [i.code for i in r.issues]
        assert "too_narrow" in codes

    def test_too_short(self):
        p = FragmentValidatorParams(min_height=40)
        r = validate_dimensions(_gray(20, 32), p)
        assert r.passed is False
        codes = [i.code for i in r.issues]
        assert "too_short" in codes

    def test_too_wide(self):
        p = FragmentValidatorParams(max_width=10)
        r = validate_dimensions(_gray(32, 32), p)
        assert r.passed is False
        assert any(i.code == "too_wide" for i in r.issues)

    def test_too_tall(self):
        p = FragmentValidatorParams(max_height=10)
        r = validate_dimensions(_gray(32, 32), p)
        assert r.passed is False
        assert any(i.code == "too_tall" for i in r.issues)

    def test_metrics_populated(self):
        r = validate_dimensions(_gray(20, 30))
        assert r.metrics["width"] == pytest.approx(30.0)
        assert r.metrics["height"] == pytest.approx(20.0)
        assert r.metrics["area"] == pytest.approx(600.0)

    def test_max_width_zero_no_upper_check(self):
        p = FragmentValidatorParams(max_width=0)
        r = validate_dimensions(_gray(32, 1000), p)
        assert not any(i.code == "too_wide" for i in r.issues)

    def test_bgr_image_accepted(self):
        r = validate_dimensions(_bgr(32, 32))
        assert "width" in r.metrics


# ─── validate_aspect_ratio ───────────────────────────────────────────────────

class TestValidateAspectRatio:
    def test_square_image_passes(self):
        p = FragmentValidatorParams(min_aspect_ratio=0.1)
        r = validate_aspect_ratio(_gray(32, 32), p)
        assert r.passed is True
        assert r.metrics["aspect_ratio"] == pytest.approx(1.0)

    def test_extreme_aspect_ratio_fails(self):
        p = FragmentValidatorParams(min_aspect_ratio=0.5)
        r = validate_aspect_ratio(_gray(4, 100), p)
        assert r.passed is False
        assert any(i.code == "extreme_aspect_ratio" for i in r.issues)

    def test_metric_aspect_ratio_in_unit_interval(self):
        r = validate_aspect_ratio(_gray(10, 40))
        ar = r.metrics["aspect_ratio"]
        assert 0.0 <= ar <= 1.0

    def test_tall_image_same_ar_as_wide(self):
        r_wide = validate_aspect_ratio(_gray(10, 40))
        r_tall = validate_aspect_ratio(_gray(40, 10))
        assert r_wide.metrics["aspect_ratio"] == pytest.approx(
            r_tall.metrics["aspect_ratio"]
        )

    def test_default_params(self):
        r = validate_aspect_ratio(_gray(32, 32))
        assert isinstance(r, ValidationResult)


# ─── validate_content_coverage ───────────────────────────────────────────────

class TestValidateContentCoverage:
    def test_full_coverage_passes(self):
        p = FragmentValidatorParams(min_coverage=0.5)
        r = validate_content_coverage(_gray(value=200), p)
        assert r.passed is True
        assert r.metrics["coverage"] == pytest.approx(1.0, abs=1e-4)

    def test_black_image_fails(self):
        p = FragmentValidatorParams(min_coverage=0.1)
        r = validate_content_coverage(_black(32, 32), p)
        assert r.passed is False
        assert any(i.code == "insufficient_coverage" for i in r.issues)

    def test_coverage_metric_in_unit_interval(self):
        r = validate_content_coverage(_gray(value=100))
        assert 0.0 <= r.metrics["coverage"] <= 1.0

    def test_bgr_accepted(self):
        r = validate_content_coverage(_bgr(), threshold=10)
        assert r.metrics["coverage"] > 0.0

    def test_content_pixels_metric(self):
        img = _gray(32, 32, value=200)
        r = validate_content_coverage(img, threshold=10)
        assert r.metrics["content_pixels"] == pytest.approx(32 * 32, rel=1e-3)

    def test_custom_threshold(self):
        img = np.full((32, 32), 5, dtype=np.uint8)
        # threshold=10 → pixels with value 5 not counted
        r = validate_content_coverage(img, threshold=10)
        assert r.metrics["coverage"] == pytest.approx(0.0)


# ─── validate_contour ────────────────────────────────────────────────────────

class TestValidateContour:
    def test_valid_square_passes(self):
        p = FragmentValidatorParams(min_contour_area=10.0)
        r = validate_contour(_square_contour(20), p)
        assert r.passed is True

    def test_too_few_points_fails(self):
        p = FragmentValidatorParams(min_contour_points=4)
        c = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]], dtype=np.float64)
        r = validate_contour(c, p)
        assert r.passed is False
        assert any(i.code == "too_few_contour_points" for i in r.issues)

    def test_too_many_points_warning(self):
        p = FragmentValidatorParams(max_contour_points=3)
        c = _square_contour()  # 4 points
        r = validate_contour(c, p)
        # warning, not error
        assert any(i.code == "too_many_contour_points" for i in r.issues)
        warn_issues = [i for i in r.issues if i.code == "too_many_contour_points"]
        assert warn_issues[0].severity == "warning"

    def test_degenerate_area_fails(self):
        p = FragmentValidatorParams(min_contour_area=1000.0)
        r = validate_contour(_square_contour(5), p)
        assert r.passed is False
        assert any(i.code == "degenerate_contour" for i in r.issues)

    def test_n_points_metric(self):
        c = _square_contour()
        r = validate_contour(c)
        assert r.metrics["n_points"] == pytest.approx(4.0)

    def test_contour_area_metric_positive(self):
        r = validate_contour(_square_contour(20))
        assert r.metrics["contour_area"] > 0.0

    def test_two_point_contour_fails(self):
        r = validate_contour(_two_point_contour())
        assert r.passed is False

    def test_cv2_format_contour(self):
        c = _square_contour().reshape(-1, 1, 2)
        r = validate_contour(c)
        assert r.metrics["n_points"] == pytest.approx(4.0)

    def test_duplicate_points_degenerate(self):
        c = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
        r = validate_contour(c)
        assert r.passed is False


# ─── validate_fragment ───────────────────────────────────────────────────────

class TestValidateFragment:
    def test_valid_fragment_passes(self):
        p = FragmentValidatorParams(min_width=16, min_height=16, min_coverage=0.05)
        r = validate_fragment(_gray(32, 32, 128), params=p)
        assert r.passed is True

    def test_fragment_idx_stored(self):
        r = validate_fragment(_gray(), fragment_idx=5)
        assert r.fragment_idx == 5

    def test_includes_dimension_check(self):
        p = FragmentValidatorParams(min_width=64)
        r = validate_fragment(_gray(32, 32), params=p)
        assert r.passed is False
        codes = [i.code for i in r.issues]
        assert "too_narrow" in codes

    def test_includes_coverage_check(self):
        p = FragmentValidatorParams(min_coverage=0.9)
        # Black image → coverage fails
        r = validate_fragment(_black(32, 32), params=p)
        assert r.passed is False
        codes = [i.code for i in r.issues]
        assert "insufficient_coverage" in codes

    def test_with_contour_includes_contour_check(self):
        p = FragmentValidatorParams(min_contour_area=10000.0)
        r = validate_fragment(_gray(32, 32), contour=_square_contour(5), params=p)
        assert r.passed is False
        assert any(i.code == "degenerate_contour" for i in r.issues)

    def test_without_contour_no_contour_issues(self):
        r = validate_fragment(_gray(32, 32))
        contour_codes = {"too_few_contour_points", "too_many_contour_points", "degenerate_contour"}
        assert not any(i.code in contour_codes for i in r.issues)

    def test_metrics_aggregated(self):
        r = validate_fragment(_gray(32, 40))
        assert "width" in r.metrics
        assert "height" in r.metrics
        assert "aspect_ratio" in r.metrics
        assert "coverage" in r.metrics

    def test_default_params(self):
        r = validate_fragment(_gray(32, 32))
        assert isinstance(r, ValidationResult)


# ─── batch_validate ──────────────────────────────────────────────────────────

class TestBatchValidate:
    def test_empty_returns_empty(self):
        assert batch_validate([]) == []

    def test_length_preserved(self):
        imgs = [_gray()] * 5
        results = batch_validate(imgs)
        assert len(results) == 5

    def test_fragment_idx_sequential(self):
        imgs = [_gray()] * 3
        results = batch_validate(imgs)
        idxs = [r.fragment_idx for r in results]
        assert idxs == [0, 1, 2]

    def test_contours_applied(self):
        imgs = [_gray(32, 32)]
        p = FragmentValidatorParams(min_contour_area=10000.0)
        results = batch_validate(imgs, contours=[_square_contour(2)], params=p)
        assert results[0].passed is False

    def test_none_contours_skipped(self):
        imgs = [_gray(32, 32)] * 2
        results = batch_validate(imgs, contours=[None, None])
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_all_return_validation_result(self):
        imgs = [_gray(32, 32), _black(10, 10), _bgr(20, 20)]
        results = batch_validate(imgs)
        assert all(isinstance(r, ValidationResult) for r in results)


# ─── filter_valid ────────────────────────────────────────────────────────────

class TestFilterValid:
    def test_all_pass(self):
        results = [ValidationResult(fragment_idx=i, passed=True) for i in range(3)]
        assert filter_valid(results) == [0, 1, 2]

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
