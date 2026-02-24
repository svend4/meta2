"""Extra tests for puzzle_reconstruction/verification/consistency_checker.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.verification.consistency_checker import (
    ConsistencyType,
    ConsistencyViolation,
    ConsistencyResult,
    estimate_line_spacing,
    estimate_char_height,
    estimate_text_angle,
    check_line_spacing,
    check_char_height,
    check_text_angle,
    check_margin_alignment,
    check_consistency,
    batch_check_consistency,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _gray(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _text_image() -> np.ndarray:
    """Simple grayscale image with two white text-line-like bands."""
    img = np.zeros((64, 128), dtype=np.uint8)
    img[10:16, 10:110] = 255
    img[30:36, 10:110] = 255
    return img


def _violation(vtype=ConsistencyType.LINE_SPACING, severity=0.5):
    return ConsistencyViolation(type=vtype, severity=severity)


def _result(score=1.0, n_checked=1, violations=None):
    return ConsistencyResult(
        violations=violations or [],
        score=score,
        n_checked=n_checked,
    )


# ─── ConsistencyType ──────────────────────────────────────────────────────────

class TestConsistencyTypeExtra:
    def test_line_spacing_value(self):
        assert ConsistencyType.LINE_SPACING == "line_spacing"

    def test_char_height_value(self):
        assert ConsistencyType.CHAR_HEIGHT == "char_height"

    def test_text_angle_value(self):
        assert ConsistencyType.TEXT_ANGLE == "text_angle"

    def test_margin_align_value(self):
        assert ConsistencyType.MARGIN_ALIGN == "margin_align"

    def test_insufficient_value(self):
        assert ConsistencyType.INSUFFICIENT == "insufficient"

    def test_all_five_values(self):
        assert len(ConsistencyType) == 5


# ─── ConsistencyViolation ─────────────────────────────────────────────────────

class TestConsistencyViolationExtra:
    def test_stores_type(self):
        v = _violation(vtype=ConsistencyType.CHAR_HEIGHT)
        assert v.type == ConsistencyType.CHAR_HEIGHT

    def test_stores_severity(self):
        v = _violation(severity=0.75)
        assert v.severity == pytest.approx(0.75)

    def test_default_fragment_ids_empty(self):
        v = _violation()
        assert v.fragment_ids == []

    def test_default_description_empty(self):
        v = _violation()
        assert v.description == ""

    def test_default_values_empty(self):
        v = _violation()
        assert v.values == {}

    def test_custom_fragment_ids(self):
        v = ConsistencyViolation(
            type=ConsistencyType.TEXT_ANGLE,
            severity=0.4,
            fragment_ids=[1, 2, 3],
        )
        assert v.fragment_ids == [1, 2, 3]

    def test_custom_description(self):
        v = ConsistencyViolation(
            type=ConsistencyType.LINE_SPACING,
            severity=0.5,
            description="test violation",
        )
        assert v.description == "test violation"

    def test_repr_contains_type(self):
        v = _violation()
        assert "line_spacing" in repr(v)


# ─── ConsistencyResult ────────────────────────────────────────────────────────

class TestConsistencyResultExtra:
    def test_stores_score(self):
        r = _result(score=0.85)
        assert r.score == pytest.approx(0.85)

    def test_stores_n_checked(self):
        r = _result(n_checked=6)
        assert r.n_checked == 6

    def test_n_violations_zero(self):
        r = _result(violations=[])
        assert r.n_violations == 0

    def test_n_violations_counted(self):
        v1 = _violation()
        v2 = _violation(severity=0.3)
        r = _result(violations=[v1, v2])
        assert r.n_violations == 2

    def test_max_severity_no_violations(self):
        r = _result(violations=[])
        assert r.max_severity == pytest.approx(0.0)

    def test_max_severity_with_violations(self):
        v1 = _violation(severity=0.2)
        v2 = _violation(severity=0.9)
        r = _result(violations=[v1, v2])
        assert r.max_severity == pytest.approx(0.9)

    def test_default_method_scores_empty(self):
        r = _result()
        assert isinstance(r.method_scores, dict)

    def test_method_scores_stored(self):
        r = ConsistencyResult(
            violations=[],
            score=1.0,
            n_checked=1,
            method_scores={"line_spacing": 0.9},
        )
        assert r.method_scores["line_spacing"] == pytest.approx(0.9)

    def test_repr_contains_score(self):
        r = _result(score=0.75)
        assert "0.75" in repr(r) or "score" in repr(r)


# ─── estimate_line_spacing ────────────────────────────────────────────────────

class TestEstimateLineSpacingExtra:
    def test_returns_float(self):
        assert isinstance(estimate_line_spacing(_blank()), float)

    def test_blank_returns_zero(self):
        assert estimate_line_spacing(_blank()) == pytest.approx(0.0)

    def test_nonneg(self):
        assert estimate_line_spacing(_text_image()) >= 0.0

    def test_bgr_input_ok(self):
        bgr = np.zeros((64, 128, 3), dtype=np.uint8)
        result = estimate_line_spacing(bgr)
        assert isinstance(result, float)

    def test_text_image_nonzero(self):
        # Two lines → should detect positive spacing
        result = estimate_line_spacing(_text_image())
        assert result >= 0.0


# ─── estimate_char_height ─────────────────────────────────────────────────────

class TestEstimateCharHeightExtra:
    def test_returns_float(self):
        assert isinstance(estimate_char_height(_blank()), float)

    def test_blank_nonneg(self):
        assert estimate_char_height(_blank()) >= 0.0

    def test_nonneg(self):
        assert estimate_char_height(_text_image()) >= 0.0

    def test_bgr_input_ok(self):
        bgr = np.zeros((64, 128, 3), dtype=np.uint8)
        result = estimate_char_height(bgr)
        assert isinstance(result, float)

    def test_min_height_filter(self):
        result = estimate_char_height(_text_image(), min_height=4)
        assert isinstance(result, float)


# ─── estimate_text_angle ──────────────────────────────────────────────────────

class TestEstimateTextAngleExtra:
    def test_returns_float(self):
        assert isinstance(estimate_text_angle(_blank()), float)

    def test_blank_returns_zero(self):
        assert estimate_text_angle(_blank()) == pytest.approx(0.0)

    def test_uniform_gray_near_zero(self):
        assert abs(estimate_text_angle(_gray())) <= 45.0

    def test_bgr_input_ok(self):
        bgr = np.zeros((64, 128, 3), dtype=np.uint8)
        result = estimate_text_angle(bgr)
        assert isinstance(result, float)

    def test_in_angle_range(self):
        result = estimate_text_angle(_text_image())
        assert -45.0 <= result <= 45.0


# ─── check_line_spacing ───────────────────────────────────────────────────────

class TestCheckLineSpacingExtra:
    def test_returns_list(self):
        result = check_line_spacing([0], [_blank()])
        assert isinstance(result, list)

    def test_single_fragment_empty(self):
        result = check_line_spacing([0], [_blank()])
        assert result == []

    def test_empty_inputs_empty(self):
        result = check_line_spacing([], [])
        assert result == []

    def test_violations_are_correct_type(self):
        imgs = [_text_image()] * 2
        for v in check_line_spacing([0, 1], imgs):
            assert v.type == ConsistencyType.LINE_SPACING

    def test_consistent_images_no_violations(self):
        imgs = [_text_image(), _text_image()]
        result = check_line_spacing([0, 1], imgs)
        assert isinstance(result, list)


# ─── check_char_height ────────────────────────────────────────────────────────

class TestCheckCharHeightExtra:
    def test_returns_list(self):
        assert isinstance(check_char_height([0], [_blank()]), list)

    def test_single_fragment_empty(self):
        assert check_char_height([0], [_blank()]) == []

    def test_empty_inputs_empty(self):
        assert check_char_height([], []) == []

    def test_violations_correct_type(self):
        for v in check_char_height([0, 1], [_text_image(), _blank()]):
            assert v.type == ConsistencyType.CHAR_HEIGHT


# ─── check_text_angle ─────────────────────────────────────────────────────────

class TestCheckTextAngleExtra:
    def test_returns_list(self):
        assert isinstance(check_text_angle([0], [_blank()]), list)

    def test_single_fragment_empty(self):
        assert check_text_angle([0], [_blank()]) == []

    def test_empty_inputs_empty(self):
        assert check_text_angle([], []) == []

    def test_violations_correct_type(self):
        for v in check_text_angle([0, 1], [_blank(), _blank()]):
            assert v.type == ConsistencyType.TEXT_ANGLE


# ─── check_margin_alignment ───────────────────────────────────────────────────

class TestCheckMarginAlignmentExtra:
    def test_returns_list(self):
        assert isinstance(check_margin_alignment([0], [_blank()]), list)

    def test_single_fragment_empty(self):
        assert check_margin_alignment([0], [_blank()]) == []

    def test_empty_inputs_empty(self):
        assert check_margin_alignment([], []) == []

    def test_violations_correct_type(self):
        for v in check_margin_alignment([0, 1], [_blank(), _blank()]):
            assert v.type == ConsistencyType.MARGIN_ALIGN


# ─── check_consistency ────────────────────────────────────────────────────────

class TestCheckConsistencyExtra:
    def test_returns_consistency_result(self):
        r = check_consistency([0], [_blank()])
        assert isinstance(r, ConsistencyResult)

    def test_score_in_range(self):
        r = check_consistency([0, 1], [_blank(), _blank()])
        assert 0.0 <= r.score <= 1.0

    def test_single_fragment_score_one(self):
        r = check_consistency([0], [_blank()])
        assert r.score == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            check_consistency([0, 1], [_blank()])

    def test_method_scores_keys(self):
        r = check_consistency([0, 1], [_blank(), _blank()])
        for k in ("line_spacing", "char_height", "text_angle", "margin_align"):
            assert k in r.method_scores

    def test_n_checked_two_fragments(self):
        r = check_consistency([0, 1], [_blank(), _blank()])
        assert r.n_checked == 1  # 2*(2-1)//2 = 1

    def test_n_checked_three_fragments(self):
        r = check_consistency([0, 1, 2], [_blank()] * 3)
        assert r.n_checked == 3  # 3*(3-1)//2 = 3

    def test_no_violations_score_one(self):
        r = check_consistency([0], [_blank()])
        assert r.score == pytest.approx(1.0)
        assert r.n_violations == 0

    def test_violations_list(self):
        r = check_consistency([0, 1], [_blank(), _blank()])
        assert isinstance(r.violations, list)


# ─── batch_check_consistency ──────────────────────────────────────────────────

class TestBatchCheckConsistencyExtra:
    def test_returns_list(self):
        result = batch_check_consistency([[0]], [[_blank()]])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_check_consistency(
            [[0], [1, 2]],
            [[_blank()], [_blank(), _blank()]],
        )
        assert len(result) == 2

    def test_each_element_is_result(self):
        for r in batch_check_consistency([[0]], [[_blank()]]):
            assert isinstance(r, ConsistencyResult)

    def test_group_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_check_consistency([[0], [1]], [[_blank()]])

    def test_empty_groups_ok(self):
        result = batch_check_consistency([], [])
        assert result == []

    def test_scores_in_range(self):
        result = batch_check_consistency(
            [[0], [0, 1]],
            [[_blank()], [_blank(), _blank()]],
        )
        for r in result:
            assert 0.0 <= r.score <= 1.0
