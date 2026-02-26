"""Tests for puzzle_reconstruction.verification.fragment_validator"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.fragment_validator import (
    ValidationIssue,
    ValidationResult,
    FragmentValidatorParams,
    validate_dimensions,
    validate_aspect_ratio,
    validate_content_coverage,
    validate_contour,
    validate_fragment,
    batch_validate,
    filter_valid,
)


def make_img(h=32, w=32, fill=128, channels=3):
    if channels == 3:
        return np.full((h, w, 3), fill, dtype=np.uint8)
    return np.full((h, w), fill, dtype=np.uint8)


# ─── ValidationIssue ──────────────────────────────────────────────────────────

def test_validation_issue_defaults():
    issue = ValidationIssue(code="too_small", message="too small")
    assert issue.severity == "error"


def test_validation_issue_warning():
    issue = ValidationIssue(code="warn", message="caution", severity="warning")
    assert issue.severity == "warning"


# ─── ValidationResult ─────────────────────────────────────────────────────────

def test_validation_result_defaults():
    result = ValidationResult()
    assert result.passed is True
    assert result.issues == []
    assert result.fragment_idx == -1


def test_validation_result_add_error():
    result = ValidationResult()
    result.add_issue(ValidationIssue(code="err", message="error"))
    assert result.passed is False
    assert len(result.issues) == 1


def test_validation_result_add_warning_keeps_passed():
    result = ValidationResult()
    result.add_issue(ValidationIssue(code="warn", message="warning", severity="warning"))
    assert result.passed is True


# ─── FragmentValidatorParams ──────────────────────────────────────────────────

def test_params_defaults():
    p = FragmentValidatorParams()
    assert p.min_width == 16
    assert p.min_height == 16


def test_params_invalid_min_width():
    with pytest.raises(ValueError):
        FragmentValidatorParams(min_width=0)


def test_params_invalid_min_height():
    with pytest.raises(ValueError):
        FragmentValidatorParams(min_height=0)


def test_params_invalid_aspect_ratio():
    with pytest.raises(ValueError):
        FragmentValidatorParams(min_aspect_ratio=1.5)


def test_params_invalid_coverage():
    with pytest.raises(ValueError):
        FragmentValidatorParams(min_coverage=2.0)


def test_params_invalid_contour_points():
    with pytest.raises(ValueError):
        FragmentValidatorParams(min_contour_points=2)


# ─── validate_dimensions ──────────────────────────────────────────────────────

def test_validate_dimensions_pass():
    img = make_img(64, 64)
    result = validate_dimensions(img)
    assert result.passed
    assert result.metrics["width"] == 64.0
    assert result.metrics["height"] == 64.0


def test_validate_dimensions_too_narrow():
    img = make_img(32, 4)
    params = FragmentValidatorParams(min_width=16)
    result = validate_dimensions(img, params)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "too_narrow" in codes


def test_validate_dimensions_too_short():
    img = make_img(4, 32)
    params = FragmentValidatorParams(min_height=16)
    result = validate_dimensions(img, params)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "too_short" in codes


def test_validate_dimensions_too_wide():
    img = make_img(32, 200)
    params = FragmentValidatorParams(max_width=100)
    result = validate_dimensions(img, params)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "too_wide" in codes


def test_validate_dimensions_too_tall():
    img = make_img(200, 32)
    params = FragmentValidatorParams(max_height=100)
    result = validate_dimensions(img, params)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "too_tall" in codes


def test_validate_dimensions_area_metric():
    img = make_img(8, 16)
    result = validate_dimensions(img, FragmentValidatorParams(min_width=1, min_height=1))
    assert result.metrics["area"] == pytest.approx(128.0)


# ─── validate_aspect_ratio ────────────────────────────────────────────────────

def test_validate_aspect_ratio_square_passes():
    img = make_img(64, 64)
    result = validate_aspect_ratio(img)
    assert result.passed
    assert result.metrics["aspect_ratio"] == pytest.approx(1.0)


def test_validate_aspect_ratio_extreme_fails():
    img = make_img(4, 200)
    params = FragmentValidatorParams(min_aspect_ratio=0.1)
    result = validate_aspect_ratio(img, params)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "extreme_aspect_ratio" in codes


def test_validate_aspect_ratio_metric_range():
    img = make_img(32, 64)
    result = validate_aspect_ratio(img)
    ar = result.metrics["aspect_ratio"]
    assert 0.0 < ar <= 1.0


# ─── validate_content_coverage ────────────────────────────────────────────────

def test_validate_content_coverage_full():
    img = make_img(32, 32, fill=200)
    result = validate_content_coverage(img)
    assert result.passed
    assert result.metrics["coverage"] > 0.9


def test_validate_content_coverage_empty():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    params = FragmentValidatorParams(min_coverage=0.05)
    result = validate_content_coverage(img, params)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "insufficient_coverage" in codes


def test_validate_content_coverage_grayscale():
    img = np.full((32, 32), 200, dtype=np.uint8)
    result = validate_content_coverage(img)
    assert result.passed


def test_validate_content_coverage_threshold():
    img = np.full((32, 32, 3), 5, dtype=np.uint8)  # below threshold=10
    params = FragmentValidatorParams(min_coverage=0.5)
    result = validate_content_coverage(img, params, threshold=10)
    assert not result.passed


# ─── validate_contour ─────────────────────────────────────────────────────────

def test_validate_contour_valid():
    contour = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.int32)
    result = validate_contour(contour)
    assert result.passed
    assert result.metrics["n_points"] == 4.0


def test_validate_contour_too_few_points():
    contour = np.array([[0, 0], [10, 0]], dtype=np.int32)
    result = validate_contour(contour)
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "too_few_contour_points" in codes


def test_validate_contour_too_many_points():
    n = 1000
    contour = np.zeros((n, 2), dtype=np.int32)
    # make unique points
    contour[:, 0] = np.arange(n)
    params = FragmentValidatorParams(max_contour_points=100)
    result = validate_contour(contour, params)
    codes = [i.code for i in result.issues]
    assert "too_many_contour_points" in codes


def test_validate_contour_area_metric():
    contour = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    result = validate_contour(contour)
    assert result.metrics["contour_area"] > 0


# ─── validate_fragment ────────────────────────────────────────────────────────

def test_validate_fragment_basic_pass():
    img = make_img(64, 64, fill=200)
    result = validate_fragment(img, fragment_idx=0)
    assert result.fragment_idx == 0
    assert isinstance(result.passed, bool)


def test_validate_fragment_with_contour():
    img = make_img(64, 64, fill=200)
    contour = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=np.int32)
    result = validate_fragment(img, contour=contour, fragment_idx=1)
    assert "n_points" in result.metrics


def test_validate_fragment_collects_all_metrics():
    img = make_img(64, 64, fill=200)
    result = validate_fragment(img)
    assert "width" in result.metrics
    assert "height" in result.metrics
    assert "aspect_ratio" in result.metrics
    assert "coverage" in result.metrics


# ─── batch_validate ───────────────────────────────────────────────────────────

def test_batch_validate_basic():
    images = [make_img(32, 32, fill=200), make_img(64, 64, fill=150)]
    results = batch_validate(images)
    assert len(results) == 2
    assert results[0].fragment_idx == 0
    assert results[1].fragment_idx == 1


def test_batch_validate_with_contours():
    images = [make_img(64, 64, fill=200)]
    contour = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=np.int32)
    results = batch_validate(images, contours=[contour])
    assert len(results) == 1
    assert "n_points" in results[0].metrics


def test_batch_validate_empty():
    results = batch_validate([])
    assert results == []


# ─── filter_valid ─────────────────────────────────────────────────────────────

def test_filter_valid_all_pass():
    results = [
        ValidationResult(fragment_idx=0, passed=True),
        ValidationResult(fragment_idx=1, passed=True),
    ]
    indices = filter_valid(results)
    assert indices == [0, 1]


def test_filter_valid_some_fail():
    results = [
        ValidationResult(fragment_idx=0, passed=True),
        ValidationResult(fragment_idx=1, passed=False),
        ValidationResult(fragment_idx=2, passed=True),
    ]
    indices = filter_valid(results)
    assert indices == [0, 2]


def test_filter_valid_all_fail():
    results = [ValidationResult(fragment_idx=i, passed=False) for i in range(3)]
    assert filter_valid(results) == []


def test_filter_valid_empty():
    assert filter_valid([]) == []
