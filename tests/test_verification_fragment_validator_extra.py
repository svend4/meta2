"""Extra tests for puzzle_reconstruction/verification/fragment_validator.py."""
from __future__ import annotations

import numpy as np
import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _contour(n=20, radius=50.0):
    """Circular contour with n points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([
        100 + radius * np.cos(angles),
        100 + radius * np.sin(angles),
    ]).astype(np.float32)
    return pts


# ─── ValidationIssue ─────────────────────────────────────────────────────────

class TestValidationIssueExtra:
    def test_defaults(self):
        vi = ValidationIssue(code="test", message="msg")
        assert vi.severity == "error"

    def test_warning(self):
        vi = ValidationIssue(code="warn", message="msg", severity="warning")
        assert vi.severity == "warning"

    def test_fields(self):
        vi = ValidationIssue(code="bad", message="detail")
        assert vi.code == "bad"
        assert vi.message == "detail"


# ─── ValidationResult ────────────────────────────────────────────────────────

class TestValidationResultExtra:
    def test_defaults(self):
        vr = ValidationResult()
        assert vr.passed is True
        assert vr.issues == []
        assert vr.metrics == {}
        assert vr.fragment_idx == -1

    def test_add_error_issue(self):
        vr = ValidationResult()
        vr.add_issue(ValidationIssue(code="e", message="err", severity="error"))
        assert vr.passed is False
        assert len(vr.issues) == 1

    def test_add_warning_keeps_passed(self):
        vr = ValidationResult()
        vr.add_issue(ValidationIssue(code="w", message="warn", severity="warning"))
        assert vr.passed is True
        assert len(vr.issues) == 1

    def test_multiple_issues(self):
        vr = ValidationResult()
        vr.add_issue(ValidationIssue(code="w", message="warn", severity="warning"))
        vr.add_issue(ValidationIssue(code="e", message="err", severity="error"))
        assert vr.passed is False
        assert len(vr.issues) == 2


# ─── FragmentValidatorParams ─────────────────────────────────────────────────

class TestFragmentValidatorParamsExtra:
    def test_defaults(self):
        p = FragmentValidatorParams()
        assert p.min_width == 16
        assert p.min_height == 16
        assert p.max_width == 0
        assert p.max_height == 0
        assert p.min_aspect_ratio == 0.05
        assert p.min_coverage == 0.05
        assert p.min_contour_points == 3
        assert p.max_contour_points == 0
        assert p.min_contour_area == pytest.approx(10.0)

    def test_invalid_min_width(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_width=0)

    def test_invalid_min_height(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_height=0)

    def test_invalid_aspect_ratio_low(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_aspect_ratio=-0.1)

    def test_invalid_aspect_ratio_high(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_aspect_ratio=1.5)

    def test_invalid_coverage_low(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_coverage=-0.1)

    def test_invalid_coverage_high(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_coverage=1.5)

    def test_invalid_contour_points(self):
        with pytest.raises(ValueError):
            FragmentValidatorParams(min_contour_points=2)


# ─── validate_dimensions ─────────────────────────────────────────────────────

class TestValidateDimensionsExtra:
    def test_ok(self):
        r = validate_dimensions(_gray(64, 64))
        assert r.passed is True
        assert r.metrics["width"] == 64.0
        assert r.metrics["height"] == 64.0
        assert r.metrics["area"] == 64.0 * 64.0

    def test_too_narrow(self):
        r = validate_dimensions(_gray(64, 8))
        assert r.passed is False
        assert any(i.code == "too_narrow" for i in r.issues)

    def test_too_short(self):
        r = validate_dimensions(_gray(8, 64))
        assert r.passed is False
        assert any(i.code == "too_short" for i in r.issues)

    def test_too_wide(self):
        p = FragmentValidatorParams(max_width=50)
        r = validate_dimensions(_gray(64, 100), p)
        assert r.passed is False
        assert any(i.code == "too_wide" for i in r.issues)

    def test_too_tall(self):
        p = FragmentValidatorParams(max_height=50)
        r = validate_dimensions(_gray(100, 64), p)
        assert r.passed is False
        assert any(i.code == "too_tall" for i in r.issues)

    def test_bgr_image(self):
        r = validate_dimensions(_bgr(32, 32))
        assert r.passed is True
        assert r.metrics["width"] == 32.0

    def test_custom_params(self):
        p = FragmentValidatorParams(min_width=1, min_height=1)
        r = validate_dimensions(_gray(2, 2), p)
        assert r.passed is True


# ─── validate_aspect_ratio ───────────────────────────────────────────────────

class TestValidateAspectRatioExtra:
    def test_square(self):
        r = validate_aspect_ratio(_gray(64, 64))
        assert r.passed is True
        assert r.metrics["aspect_ratio"] == pytest.approx(1.0)

    def test_extreme_ratio(self):
        p = FragmentValidatorParams(min_width=1, min_height=1,
                                    min_aspect_ratio=0.5)
        r = validate_aspect_ratio(_gray(100, 1), p)
        assert r.passed is False
        assert any(i.code == "extreme_aspect_ratio" for i in r.issues)

    def test_rectangular(self):
        r = validate_aspect_ratio(_gray(64, 128))
        # min(64,128)/max(64,128) = 0.5
        assert r.metrics["aspect_ratio"] == pytest.approx(0.5)


# ─── validate_content_coverage ───────────────────────────────────────────────

class TestValidateContentCoverageExtra:
    def test_full_coverage(self):
        r = validate_content_coverage(_gray(32, 32, val=128))
        assert r.passed is True
        assert r.metrics["coverage"] > 0.9

    def test_empty_image(self):
        r = validate_content_coverage(_gray(32, 32, val=0))
        assert r.passed is False
        assert r.metrics["coverage"] == pytest.approx(0.0)

    def test_bgr_image(self):
        r = validate_content_coverage(_bgr(32, 32, val=128))
        assert r.passed is True

    def test_custom_threshold(self):
        r = validate_content_coverage(_gray(32, 32, val=5), threshold=10)
        assert r.metrics["coverage"] == pytest.approx(0.0)

    def test_content_pixels(self):
        r = validate_content_coverage(_gray(10, 10, val=128))
        assert r.metrics["content_pixels"] == 100.0


# ─── validate_contour ────────────────────────────────────────────────────────

class TestValidateContourExtra:
    def test_valid_contour(self):
        r = validate_contour(_contour(20, 50.0))
        assert r.passed is True
        assert r.metrics["n_points"] == 20.0
        assert r.metrics["contour_area"] > 0

    def test_too_few_points(self):
        pts = np.array([[0, 0], [10, 0]], dtype=np.float32)
        r = validate_contour(pts)
        assert r.passed is False
        assert any(i.code == "too_few_contour_points" for i in r.issues)

    def test_too_many_points_warning(self):
        p = FragmentValidatorParams(max_contour_points=5)
        r = validate_contour(_contour(10), p)
        assert any(i.code == "too_many_contour_points" for i in r.issues)
        # warning only, so still passes
        assert r.passed is True

    def test_degenerate_area(self):
        # Collinear points → area ≈ 0
        pts = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
        r = validate_contour(pts)
        assert r.passed is False
        assert any(i.code == "degenerate_contour" for i in r.issues)

    def test_duplicate_points(self):
        pts = np.array([[0, 0], [0, 0], [0, 0], [10, 10]], dtype=np.float32)
        r = validate_contour(pts)
        assert any(i.code == "degenerate_contour" for i in r.issues)

    def test_reshaped_contour(self):
        # (N, 1, 2) format (OpenCV style)
        pts = _contour(10).reshape(-1, 1, 2)
        r = validate_contour(pts)
        assert r.metrics["n_points"] == 10.0


# ─── validate_fragment ───────────────────────────────────────────────────────

class TestValidateFragmentExtra:
    def test_valid(self):
        r = validate_fragment(_gray(64, 64), fragment_idx=5)
        assert r.passed is True
        assert r.fragment_idx == 5
        assert "width" in r.metrics
        assert "aspect_ratio" in r.metrics
        assert "coverage" in r.metrics

    def test_with_contour(self):
        r = validate_fragment(_gray(64, 64), contour=_contour(20))
        assert "n_points" in r.metrics
        assert "contour_area" in r.metrics

    def test_no_contour(self):
        r = validate_fragment(_gray(64, 64))
        assert "n_points" not in r.metrics

    def test_failing_dimensions(self):
        r = validate_fragment(_gray(4, 4))
        assert r.passed is False

    def test_failing_coverage(self):
        r = validate_fragment(_gray(64, 64, val=0))
        assert r.passed is False


# ─── batch_validate ──────────────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_empty(self):
        assert batch_validate([]) == []

    def test_multiple(self):
        imgs = [_gray(64, 64), _gray(64, 64)]
        results = batch_validate(imgs)
        assert len(results) == 2
        assert results[0].fragment_idx == 0
        assert results[1].fragment_idx == 1

    def test_with_contours(self):
        imgs = [_gray(64, 64)]
        contours = [_contour(20)]
        results = batch_validate(imgs, contours=contours)
        assert "n_points" in results[0].metrics

    def test_none_contour(self):
        imgs = [_gray(64, 64)]
        contours = [None]
        results = batch_validate(imgs, contours=contours)
        assert "n_points" not in results[0].metrics


# ─── filter_valid ────────────────────────────────────────────────────────────

class TestFilterValidExtra:
    def test_empty(self):
        assert filter_valid([]) == []

    def test_all_pass(self):
        results = batch_validate([_gray(64, 64), _gray(64, 64)])
        valid_ids = filter_valid(results)
        assert valid_ids == [0, 1]

    def test_all_fail(self):
        results = batch_validate([_gray(4, 4), _gray(4, 4)])
        valid_ids = filter_valid(results)
        assert valid_ids == []

    def test_mixed(self):
        results = batch_validate([_gray(64, 64), _gray(4, 4)])
        valid_ids = filter_valid(results)
        assert valid_ids == [0]
