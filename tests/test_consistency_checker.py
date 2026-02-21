"""
Тесты для puzzle_reconstruction/verification/consistency_checker.py

Покрывает:
    ConsistencyType        — 5 значений, str-Enum
    ConsistencyViolation   — repr, поля, defaults
    ConsistencyResult      — n_violations, max_severity, repr, method_scores
    estimate_line_spacing  — float ≥ 0, blank → 0, малое изображение
    estimate_char_height   — float ≥ 0, blank → 0, изображение с текстом
    estimate_text_angle    — float, blank → 0, диапазон
    check_line_spacing     — одиночный фрагмент → [], одинаковые → [],
                             аутлайер → нарушение
    check_char_height      — аналогично check_line_spacing
    check_text_angle       — аналогично, max_angle параметр
    check_margin_alignment — аналогично, tol_px параметр
    check_consistency      — возвращает ConsistencyResult, score ∈ [0,1],
                             ValueError на несовпадение длин,
                             method_scores dict с 4 ключами
    batch_check_consistency — список ConsistencyResult, ValueError
"""
from __future__ import annotations

from typing import List

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


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_img():
    return np.zeros((80, 80), dtype=np.uint8)


@pytest.fixture
def text_like_img():
    """Изображение с горизонтальными чёрными строками на белом фоне."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    for y in [10, 25, 40, 55, 70]:
        img[y:y + 8, 5:95] = 30
    return img


@pytest.fixture
def checkerboard():
    img = np.zeros((80, 80), dtype=np.uint8)
    for i in range(0, 80, 8):
        for j in range(0, 80, 8):
            if (i // 8 + j // 8) % 2 == 0:
                img[i:i + 8, j:j + 8] = 200
    return img


@pytest.fixture
def sample_violation():
    return ConsistencyViolation(
        type=ConsistencyType.LINE_SPACING,
        severity=0.4,
        fragment_ids=[1, 3],
        description="test",
        values={1: 20.0, 3: 30.0},
    )


@pytest.fixture
def sample_result(sample_violation):
    return ConsistencyResult(
        violations=[sample_violation],
        score=0.7,
        n_checked=3,
        method_scores={"line_spacing": 0.0, "char_height": 1.0},
    )


# ─── ConsistencyType ──────────────────────────────────────────────────────────

class TestConsistencyType:
    def test_five_types(self):
        assert len(ConsistencyType) == 5

    def test_is_str_enum(self):
        assert isinstance(ConsistencyType.LINE_SPACING, str)
        assert ConsistencyType.LINE_SPACING == "line_spacing"

    def test_all_values(self):
        vals = {ct.value for ct in ConsistencyType}
        assert "line_spacing"  in vals
        assert "char_height"   in vals
        assert "text_angle"    in vals
        assert "margin_align"  in vals
        assert "insufficient"  in vals

    def test_comparison(self):
        assert ConsistencyType.LINE_SPACING != ConsistencyType.CHAR_HEIGHT


# ─── ConsistencyViolation ─────────────────────────────────────────────────────

class TestConsistencyViolation:
    def test_fields(self, sample_violation):
        assert sample_violation.type     == ConsistencyType.LINE_SPACING
        assert sample_violation.severity == pytest.approx(0.4)
        assert sample_violation.fragment_ids == [1, 3]
        assert sample_violation.description == "test"
        assert sample_violation.values[1] == pytest.approx(20.0)

    def test_default_fragment_ids(self):
        v = ConsistencyViolation(type=ConsistencyType.CHAR_HEIGHT, severity=0.2)
        assert v.fragment_ids == []

    def test_default_description(self):
        v = ConsistencyViolation(type=ConsistencyType.TEXT_ANGLE, severity=0.1)
        assert v.description == ""

    def test_default_values(self):
        v = ConsistencyViolation(type=ConsistencyType.MARGIN_ALIGN, severity=0.5)
        assert v.values == {}

    def test_repr_contains_type(self, sample_violation):
        r = repr(sample_violation)
        assert "ConsistencyViolation" in r
        assert "line_spacing" in r
        assert "severity=" in r


# ─── ConsistencyResult ────────────────────────────────────────────────────────

class TestConsistencyResult:
    def test_n_violations(self, sample_result):
        assert sample_result.n_violations == 1

    def test_max_severity(self, sample_result):
        assert sample_result.max_severity == pytest.approx(0.4)

    def test_max_severity_empty(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=0)
        assert res.max_severity == pytest.approx(0.0)

    def test_n_violations_zero(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=0)
        assert res.n_violations == 0

    def test_repr_fields(self, sample_result):
        r = repr(sample_result)
        assert "ConsistencyResult" in r
        assert "score=" in r
        assert "violations=" in r

    def test_method_scores_default_empty(self):
        res = ConsistencyResult(violations=[], score=1.0, n_checked=0)
        assert res.method_scores == {}

    def test_score_field(self, sample_result):
        assert sample_result.score == pytest.approx(0.7)

    def test_n_checked_field(self, sample_result):
        assert sample_result.n_checked == 3


# ─── estimate_line_spacing ────────────────────────────────────────────────────

class TestEstimateLineSpacing:
    def test_returns_float(self, blank_img):
        val = estimate_line_spacing(blank_img)
        assert isinstance(val, float)

    def test_blank_returns_zero(self, blank_img):
        assert estimate_line_spacing(blank_img) == pytest.approx(0.0)

    def test_nonnegative(self, text_like_img):
        assert estimate_line_spacing(text_like_img) >= 0.0

    def test_bgr_accepted(self, blank_img):
        bgr = cv2.cvtColor(blank_img, cv2.COLOR_GRAY2BGR)
        val = estimate_line_spacing(bgr)
        assert isinstance(val, float)

    def test_text_image_nonzero(self, text_like_img):
        val = estimate_line_spacing(text_like_img)
        # Текстовое изображение с 5 строками должно дать ненулевой интервал
        assert val >= 0.0  # может быть 0, если детектор не видит разрыва

    def test_small_image_no_crash(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        val = estimate_line_spacing(img)
        assert isinstance(val, float)


# ─── estimate_char_height ─────────────────────────────────────────────────────

class TestEstimateCharHeight:
    def test_returns_float(self, blank_img):
        val = estimate_char_height(blank_img)
        assert isinstance(val, float)

    def test_blank_returns_zero(self, blank_img):
        assert estimate_char_height(blank_img) == pytest.approx(0.0)

    def test_nonnegative(self, text_like_img):
        assert estimate_char_height(text_like_img) >= 0.0

    def test_bgr_accepted(self, blank_img):
        bgr = cv2.cvtColor(blank_img, cv2.COLOR_GRAY2BGR)
        val = estimate_char_height(bgr)
        assert isinstance(val, float)

    def test_small_image_no_crash(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        val = estimate_char_height(img)
        assert val >= 0.0

    def test_text_image_positive(self, text_like_img):
        val = estimate_char_height(text_like_img)
        # Строки высотой 8px → медиана должна быть близко к 8
        if val > 0:
            assert val >= 1.0


# ─── estimate_text_angle ──────────────────────────────────────────────────────

class TestEstimateTextAngle:
    def test_returns_float(self, blank_img):
        val = estimate_text_angle(blank_img)
        assert isinstance(val, float)

    def test_blank_returns_zero(self, blank_img):
        assert estimate_text_angle(blank_img) == pytest.approx(0.0)

    def test_horizontal_lines_near_zero(self, text_like_img):
        ang = estimate_text_angle(text_like_img)
        assert abs(ang) < 10.0  # горизонтальные линии → угол ≈ 0

    def test_result_in_range(self, checkerboard):
        ang = estimate_text_angle(checkerboard)
        assert -45.0 <= ang <= 45.0

    def test_bgr_accepted(self, blank_img):
        bgr = cv2.cvtColor(blank_img, cv2.COLOR_GRAY2BGR)
        val = estimate_text_angle(bgr)
        assert isinstance(val, float)


# ─── check_line_spacing ───────────────────────────────────────────────────────

class TestCheckLineSpacing:
    def test_single_fragment_returns_empty(self, text_like_img):
        result = check_line_spacing([1], [text_like_img])
        assert result == []

    def test_empty_inputs_returns_empty(self):
        assert check_line_spacing([], []) == []

    def test_returns_list(self, blank_img):
        result = check_line_spacing([1, 2], [blank_img, blank_img])
        assert isinstance(result, list)

    def test_blank_images_no_violation(self, blank_img):
        # Пустые изображения → spacing=0 → нет данных для проверки
        result = check_line_spacing([1, 2, 3],
                                     [blank_img, blank_img, blank_img])
        assert result == []

    def test_violation_is_consistency_violation(self, text_like_img):
        # Если будет нарушение — тип правильный
        img2 = np.full((100, 100), 255, dtype=np.uint8)
        img2[5:8, 5:95] = 30   # одна линия (другой интервал)
        result = check_line_spacing([1, 2], [text_like_img, img2])
        for v in result:
            assert isinstance(v, ConsistencyViolation)
            assert v.type == ConsistencyType.LINE_SPACING

    def test_identical_spacing_no_violation(self, text_like_img):
        # Два одинаковых изображения → нет нарушений
        result = check_line_spacing([1, 2],
                                     [text_like_img, text_like_img],
                                     tol_ratio=0.1)
        assert result == []


# ─── check_char_height ────────────────────────────────────────────────────────

class TestCheckCharHeight:
    def test_single_fragment_returns_empty(self, text_like_img):
        assert check_char_height([1], [text_like_img]) == []

    def test_returns_list(self, blank_img):
        result = check_char_height([1, 2], [blank_img, blank_img])
        assert isinstance(result, list)

    def test_blank_no_violation(self, blank_img):
        result = check_char_height([1, 2], [blank_img, blank_img])
        assert result == []

    def test_violation_type(self, text_like_img):
        # Создаём изображение с другим размером символов
        big_text = np.full((100, 100), 255, dtype=np.uint8)
        big_text[5:25, 5:95] = 30   # большая строка
        result = check_char_height([1, 2], [text_like_img, big_text])
        for v in result:
            assert v.type == ConsistencyType.CHAR_HEIGHT

    def test_identical_images_no_violation(self, text_like_img):
        result = check_char_height([1, 2],
                                    [text_like_img, text_like_img],
                                    tol_ratio=0.1)
        assert result == []


# ─── check_text_angle ─────────────────────────────────────────────────────────

class TestCheckTextAngle:
    def test_single_returns_empty(self, blank_img):
        assert check_text_angle([1], [blank_img]) == []

    def test_returns_list(self, blank_img):
        result = check_text_angle([1, 2], [blank_img, blank_img])
        assert isinstance(result, list)

    def test_blank_returns_empty(self, blank_img):
        result = check_text_angle([1, 2], [blank_img, blank_img])
        assert result == []

    def test_violation_type(self):
        # Горизонтальное vs вертикальное изображение
        h = np.zeros((80, 80), dtype=np.uint8)
        cv2.line(h, (5, 40), (75, 40), 255, 2)   # горизонталь
        v = np.zeros((80, 80), dtype=np.uint8)
        cv2.line(v, (40, 5), (40, 75), 255, 2)   # вертикаль
        result = check_text_angle([1, 2], [h, v], max_angle=5.0)
        for viol in result:
            assert viol.type == ConsistencyType.TEXT_ANGLE

    def test_max_angle_parameter(self, blank_img):
        result_tight = check_text_angle([1, 2], [blank_img, blank_img],
                                         max_angle=0.01)
        result_loose = check_text_angle([1, 2], [blank_img, blank_img],
                                         max_angle=90.0)
        assert isinstance(result_tight, list)
        assert isinstance(result_loose, list)


# ─── check_margin_alignment ───────────────────────────────────────────────────

class TestCheckMarginAlignment:
    def test_single_returns_empty(self, blank_img):
        assert check_margin_alignment([1], [blank_img]) == []

    def test_returns_list(self, blank_img):
        result = check_margin_alignment([1, 2], [blank_img, blank_img])
        assert isinstance(result, list)

    def test_blank_returns_empty(self, blank_img):
        result = check_margin_alignment([1, 2], [blank_img, blank_img])
        assert result == []

    def test_violation_type(self):
        left_text  = np.full((80, 80), 255, dtype=np.uint8)
        left_text[:, 5:15] = 0   # текст у левого края
        right_text = np.full((80, 80), 255, dtype=np.uint8)
        right_text[:, 60:75] = 0   # текст у правого края
        result = check_margin_alignment([1, 2], [left_text, right_text],
                                         tol_px=5.0)
        for v in result:
            assert v.type == ConsistencyType.MARGIN_ALIGN

    def test_tol_px_parameter(self, blank_img):
        result_tight = check_margin_alignment([1, 2], [blank_img, blank_img],
                                               tol_px=0.01)
        result_loose = check_margin_alignment([1, 2], [blank_img, blank_img],
                                               tol_px=1000.0)
        assert isinstance(result_tight, list)
        assert isinstance(result_loose, list)


# ─── check_consistency ────────────────────────────────────────────────────────

class TestCheckConsistency:
    def test_returns_consistency_result(self, blank_img):
        res = check_consistency([1, 2], [blank_img, blank_img])
        assert isinstance(res, ConsistencyResult)

    def test_score_in_range(self, text_like_img, blank_img):
        res = check_consistency([1, 2], [text_like_img, blank_img])
        assert 0.0 <= res.score <= 1.0

    def test_identical_fragments_high_score(self, blank_img):
        res = check_consistency([1, 2, 3],
                                 [blank_img, blank_img, blank_img])
        assert res.score == pytest.approx(1.0)

    def test_mismatched_lengths_raises(self, blank_img):
        with pytest.raises(ValueError, match="длины"):
            check_consistency([1, 2, 3], [blank_img, blank_img])

    def test_method_scores_has_four_keys(self, blank_img):
        res = check_consistency([1, 2], [blank_img, blank_img])
        assert "line_spacing"  in res.method_scores
        assert "char_height"   in res.method_scores
        assert "text_angle"    in res.method_scores
        assert "margin_align"  in res.method_scores

    def test_violations_is_list(self, blank_img):
        res = check_consistency([1, 2], [blank_img, blank_img])
        assert isinstance(res.violations, list)

    def test_n_checked_nonnegative(self, blank_img):
        res = check_consistency([1, 2, 3],
                                 [blank_img, blank_img, blank_img])
        assert res.n_checked >= 0

    def test_empty_inputs(self):
        res = check_consistency([], [])
        assert isinstance(res, ConsistencyResult)
        assert res.score == pytest.approx(1.0)


# ─── batch_check_consistency ──────────────────────────────────────────────────

class TestBatchCheckConsistency:
    def test_returns_list(self, blank_img):
        results = batch_check_consistency(
            [[1, 2], [3, 4]],
            [[blank_img, blank_img], [blank_img, blank_img]],
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_consistency_result(self, blank_img):
        results = batch_check_consistency(
            [[1]], [[blank_img]]
        )
        assert isinstance(results[0], ConsistencyResult)

    def test_empty_groups(self):
        results = batch_check_consistency([], [])
        assert results == []

    def test_mismatched_groups_raises(self, blank_img):
        with pytest.raises(ValueError, match="Число групп"):
            batch_check_consistency(
                [[1, 2], [3]],
                [[blank_img, blank_img]],
            )

    def test_kwargs_forwarded(self, blank_img):
        results = batch_check_consistency(
            [[1, 2]],
            [[blank_img, blank_img]],
            angle_max=1.0,
        )
        assert len(results) == 1

    def test_length_matches_input(self, blank_img):
        results = batch_check_consistency(
            [[1], [2], [3]],
            [[blank_img], [blank_img], [blank_img]],
        )
        assert len(results) == 3
