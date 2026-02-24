"""Extra tests for puzzle_reconstruction/verification/consistency_checker.py."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _blank(h=80, w=80):
    return np.zeros((h, w), dtype=np.uint8)


def _text_like():
    img = np.full((100, 100), 255, dtype=np.uint8)
    for y in [10, 25, 40, 55, 70]:
        img[y:y + 8, 5:95] = 30
    return img


# ─── ConsistencyType (extra) ─────────────────────────────────────────────────

class TestConsistencyTypeExtra:
    def test_five_members(self):
        assert len(ConsistencyType) == 5

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

    def test_is_str_enum(self):
        assert isinstance(ConsistencyType.LINE_SPACING, str)

    def test_all_distinct(self):
        vals = [ct.value for ct in ConsistencyType]
        assert len(vals) == len(set(vals))


# ─── ConsistencyViolation (extra) ────────────────────────────────────────────

class TestConsistencyViolationExtra:
    def test_type_stored(self):
        v = ConsistencyViolation(type=ConsistencyType.CHAR_HEIGHT, severity=0.3)
        assert v.type == ConsistencyType.CHAR_HEIGHT

    def test_severity_stored(self):
        v = ConsistencyViolation(type=ConsistencyType.TEXT_ANGLE, severity=0.75)
        assert v.severity == pytest.approx(0.75)

    def test_default_fragment_ids_empty(self):
        v = ConsistencyViolation(type=ConsistencyType.LINE_SPACING, severity=0.1)
        assert v.fragment_ids == []

    def test_default_description_empty(self):
        v = ConsistencyViolation(type=ConsistencyType.MARGIN_ALIGN, severity=0.2)
        assert v.description == ""

    def test_default_values_empty(self):
        v = ConsistencyViolation(type=ConsistencyType.TEXT_ANGLE, severity=0.5)
        assert v.values == {}

    def test_custom_fragment_ids(self):
        v = ConsistencyViolation(type=ConsistencyType.CHAR_HEIGHT, severity=0.3,
                                 fragment_ids=[5, 6, 7])
        assert v.fragment_ids == [5, 6, 7]

    def test_custom_values(self):
        v = ConsistencyViolation(type=ConsistencyType.LINE_SPACING, severity=0.4,
                                 values={1: 10.0, 2: 15.0})
        assert v.values[1] == pytest.approx(10.0)

    def test_repr_contains_type(self):
        v = ConsistencyViolation(type=ConsistencyType.LINE_SPACING, severity=0.5)
        assert "ConsistencyViolation" in repr(v)


# ─── ConsistencyResult (extra) ───────────────────────────────────────────────

class TestConsistencyResultExtra:
    def test_n_violations_zero(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=0)
        assert res.n_violations == 0

    def test_n_violations_positive(self):
        v = ConsistencyViolation(type=ConsistencyType.LINE_SPACING, severity=0.3)
        res = ConsistencyResult(violations=[v, v], score=0.7, n_checked=2)
        assert res.n_violations == 2

    def test_max_severity_empty_zero(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=0)
        assert res.max_severity == pytest.approx(0.0)

    def test_max_severity_multiple(self):
        v1 = ConsistencyViolation(type=ConsistencyType.CHAR_HEIGHT, severity=0.3)
        v2 = ConsistencyViolation(type=ConsistencyType.TEXT_ANGLE, severity=0.7)
        res = ConsistencyResult(violations=[v1, v2], score=0.6, n_checked=2)
        assert res.max_severity == pytest.approx(0.7)

    def test_score_stored(self):
        res = ConsistencyResult(violations=[], score=0.85, n_checked=5)
        assert res.score == pytest.approx(0.85)

    def test_n_checked_stored(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=7)
        assert res.n_checked == 7

    def test_method_scores_default_empty(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=0)
        assert res.method_scores == {}

    def test_repr_contains_score(self):
        res = ConsistencyResult(violations=[], score=0.6, n_checked=3)
        assert "score=" in repr(res)


# ─── estimate_line_spacing (extra) ───────────────────────────────────────────

class TestEstimateLineSpacingExtra:
    def test_returns_float(self):
        val = estimate_line_spacing(_blank())
        assert isinstance(val, float)

    def test_blank_zero(self):
        assert estimate_line_spacing(_blank()) == pytest.approx(0.0)

    def test_nonneg(self):
        assert estimate_line_spacing(_text_like()) >= 0.0

    def test_bgr_accepted(self):
        bgr = cv2.cvtColor(_blank(), cv2.COLOR_GRAY2BGR)
        val = estimate_line_spacing(bgr)
        assert isinstance(val, float)

    def test_small_image_ok(self):
        val = estimate_line_spacing(np.zeros((5, 5), dtype=np.uint8))
        assert isinstance(val, float)


# ─── estimate_char_height (extra) ────────────────────────────────────────────

class TestEstimateCharHeightExtra:
    def test_returns_float(self):
        val = estimate_char_height(_blank())
        assert isinstance(val, float)

    def test_blank_nonneg(self):
        assert estimate_char_height(_blank()) >= 0.0

    def test_nonneg(self):
        assert estimate_char_height(_text_like()) >= 0.0

    def test_bgr_accepted(self):
        bgr = cv2.cvtColor(_blank(), cv2.COLOR_GRAY2BGR)
        val = estimate_char_height(bgr)
        assert isinstance(val, float)

    def test_small_image_ok(self):
        val = estimate_char_height(np.zeros((4, 4), dtype=np.uint8))
        assert val >= 0.0


# ─── estimate_text_angle (extra) ─────────────────────────────────────────────

class TestEstimateTextAngleExtra:
    def test_returns_float(self):
        val = estimate_text_angle(_blank())
        assert isinstance(val, float)

    def test_blank_zero(self):
        assert estimate_text_angle(_blank()) == pytest.approx(0.0)

    def test_result_in_range(self):
        val = estimate_text_angle(_text_like())
        assert -45.0 <= val <= 45.0

    def test_bgr_accepted(self):
        bgr = cv2.cvtColor(_blank(), cv2.COLOR_GRAY2BGR)
        val = estimate_text_angle(bgr)
        assert isinstance(val, float)

    def test_horizontal_lines_near_zero(self):
        ang = estimate_text_angle(_text_like())
        assert abs(ang) < 10.0


# ─── check_line_spacing (extra) ──────────────────────────────────────────────

class TestCheckLineSpacingExtra:
    def test_empty_inputs_empty(self):
        assert check_line_spacing([], []) == []

    def test_single_empty(self):
        assert check_line_spacing([1], [_blank()]) == []

    def test_returns_list(self):
        result = check_line_spacing([1, 2], [_blank(), _blank()])
        assert isinstance(result, list)

    def test_identical_no_violation(self):
        img = _text_like()
        result = check_line_spacing([1, 2], [img, img], tol_ratio=0.1)
        assert result == []

    def test_violations_are_correct_type(self):
        img2 = np.full((100, 100), 255, dtype=np.uint8)
        img2[5:8, 5:95] = 30
        result = check_line_spacing([1, 2], [_text_like(), img2])
        for v in result:
            assert isinstance(v, ConsistencyViolation)
            assert v.type == ConsistencyType.LINE_SPACING


# ─── check_char_height (extra) ───────────────────────────────────────────────

class TestCheckCharHeightExtra:
    def test_empty_returns_empty(self):
        assert check_char_height([], []) == []

    def test_single_returns_empty(self):
        assert check_char_height([1], [_blank()]) == []

    def test_blank_no_violation(self):
        result = check_char_height([1, 2], [_blank(), _blank()])
        assert result == []

    def test_returns_list(self):
        result = check_char_height([1, 2], [_blank(), _blank()])
        assert isinstance(result, list)

    def test_violation_type(self):
        big_text = np.full((100, 100), 255, dtype=np.uint8)
        big_text[5:25, 5:95] = 30
        result = check_char_height([1, 2], [_text_like(), big_text])
        for v in result:
            assert v.type == ConsistencyType.CHAR_HEIGHT


# ─── check_text_angle (extra) ────────────────────────────────────────────────

class TestCheckTextAngleExtra:
    def test_empty_returns_empty(self):
        assert check_text_angle([], []) == []

    def test_single_returns_empty(self):
        assert check_text_angle([1], [_blank()]) == []

    def test_blank_no_violation(self):
        result = check_text_angle([1, 2], [_blank(), _blank()])
        assert result == []

    def test_returns_list(self):
        assert isinstance(check_text_angle([1, 2], [_blank(), _blank()]), list)

    def test_violation_type(self):
        h = np.zeros((80, 80), dtype=np.uint8)
        cv2.line(h, (5, 40), (75, 40), 255, 2)
        v = np.zeros((80, 80), dtype=np.uint8)
        cv2.line(v, (40, 5), (40, 75), 255, 2)
        result = check_text_angle([1, 2], [h, v], max_angle=5.0)
        for viol in result:
            assert viol.type == ConsistencyType.TEXT_ANGLE


# ─── check_margin_alignment (extra) ──────────────────────────────────────────

class TestCheckMarginAlignmentExtra:
    def test_empty_returns_empty(self):
        assert check_margin_alignment([], []) == []

    def test_single_returns_empty(self):
        assert check_margin_alignment([1], [_blank()]) == []

    def test_blank_no_violation(self):
        result = check_margin_alignment([1, 2], [_blank(), _blank()])
        assert result == []

    def test_returns_list(self):
        assert isinstance(check_margin_alignment([1, 2], [_blank(), _blank()]), list)

    def test_violation_type(self):
        left_text = np.full((80, 80), 255, dtype=np.uint8)
        left_text[:, 5:15] = 0
        right_text = np.full((80, 80), 255, dtype=np.uint8)
        right_text[:, 60:75] = 0
        result = check_margin_alignment([1, 2], [left_text, right_text], tol_px=5.0)
        for v in result:
            assert v.type == ConsistencyType.MARGIN_ALIGN


# ─── check_consistency (extra) ───────────────────────────────────────────────

class TestCheckConsistencyExtra:
    def test_returns_consistency_result(self):
        res = check_consistency([1, 2], [_blank(), _blank()])
        assert isinstance(res, ConsistencyResult)

    def test_score_in_range(self):
        res = check_consistency([1, 2], [_text_like(), _blank()])
        assert 0.0 <= res.score <= 1.0

    def test_identical_high_score(self):
        res = check_consistency([1, 2], [_blank(), _blank()])
        assert res.score == pytest.approx(1.0)

    def test_method_scores_four_keys(self):
        res = check_consistency([1, 2], [_blank(), _blank()])
        assert "line_spacing" in res.method_scores
        assert "char_height" in res.method_scores
        assert "text_angle" in res.method_scores
        assert "margin_align" in res.method_scores

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            check_consistency([1, 2, 3], [_blank(), _blank()])

    def test_empty_inputs_score_one(self):
        res = check_consistency([], [])
        assert res.score == pytest.approx(1.0)

    def test_n_checked_nonneg(self):
        res = check_consistency([1, 2], [_blank(), _blank()])
        assert res.n_checked >= 0

    def test_violations_is_list(self):
        res = check_consistency([1, 2], [_blank(), _blank()])
        assert isinstance(res.violations, list)


# ─── batch_check_consistency (extra) ─────────────────────────────────────────

class TestBatchCheckConsistencyExtra:
    def test_returns_list(self):
        results = batch_check_consistency(
            [[1, 2], [3, 4]],
            [[_blank(), _blank()], [_blank(), _blank()]],
        )
        assert isinstance(results, list)

    def test_length_matches(self):
        results = batch_check_consistency(
            [[1], [2], [3]],
            [[_blank()], [_blank()], [_blank()]],
        )
        assert len(results) == 3

    def test_each_is_consistency_result(self):
        results = batch_check_consistency([[1]], [[_blank()]])
        assert isinstance(results[0], ConsistencyResult)

    def test_empty_returns_empty(self):
        assert batch_check_consistency([], []) == []

    def test_mismatched_groups_raises(self):
        with pytest.raises(ValueError):
            batch_check_consistency([[1, 2]], [[_blank()], [_blank()]])

    def test_two_groups(self):
        results = batch_check_consistency(
            [[1, 2], [3, 4]],
            [[_blank(), _blank()], [_blank(), _blank()]],
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ConsistencyResult)
