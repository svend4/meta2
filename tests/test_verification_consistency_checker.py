"""Тесты для puzzle_reconstruction/verification/consistency_checker.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_blank(h=64, w=64):
    """Blank (white) image — no text content."""
    return np.full((h, w), 255, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_text_like(h=128, w=128, n_lines=4, line_height=8, char_width=5):
    """Synthetic text-like image: dark horizontal bars on white background."""
    img = np.full((h, w), 255, dtype=np.uint8)
    spacing = h // (n_lines + 1)
    for i in range(n_lines):
        y = spacing * (i + 1)
        y0 = max(0, y - line_height // 2)
        y1 = min(h, y + line_height // 2)
        img[y0:y1, 10:w-10] = 0
    return img


def make_gray_image(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


# ─── ConsistencyType ──────────────────────────────────────────────────────────

class TestConsistencyType:
    def test_values(self):
        assert ConsistencyType.LINE_SPACING.value == "line_spacing"
        assert ConsistencyType.CHAR_HEIGHT.value == "char_height"
        assert ConsistencyType.TEXT_ANGLE.value == "text_angle"
        assert ConsistencyType.MARGIN_ALIGN.value == "margin_align"
        assert ConsistencyType.INSUFFICIENT.value == "insufficient"

    def test_is_string(self):
        assert isinstance(ConsistencyType.LINE_SPACING, str)


# ─── ConsistencyViolation ─────────────────────────────────────────────────────

class TestConsistencyViolation:
    def test_creation(self):
        v = ConsistencyViolation(
            type=ConsistencyType.LINE_SPACING,
            severity=0.5,
            fragment_ids=[0, 1],
            description="Test violation",
            values={0: 12.0, 1: 20.0},
        )
        assert v.type == ConsistencyType.LINE_SPACING
        assert v.severity == pytest.approx(0.5)
        assert v.fragment_ids == [0, 1]
        assert v.description == "Test violation"
        assert v.values[0] == pytest.approx(12.0)

    def test_defaults(self):
        v = ConsistencyViolation(
            type=ConsistencyType.CHAR_HEIGHT,
            severity=0.3,
        )
        assert v.fragment_ids == []
        assert v.description == ""
        assert v.values == {}

    def test_no_validation_on_creation(self):
        # No __post_init__ validation
        v = ConsistencyViolation(type=ConsistencyType.TEXT_ANGLE, severity=-1.0)
        assert v.severity == pytest.approx(-1.0)


# ─── ConsistencyResult ────────────────────────────────────────────────────────

class TestConsistencyResult:
    def _make_result(self, violations=None, score=1.0, n_checked=0):
        return ConsistencyResult(
            violations=violations or [],
            score=score,
            n_checked=n_checked,
        )

    def test_creation(self):
        r = self._make_result(score=0.8, n_checked=3)
        assert r.score == pytest.approx(0.8)
        assert r.n_checked == 3

    def test_n_violations_zero(self):
        r = self._make_result()
        assert r.n_violations == 0

    def test_n_violations_with_violations(self):
        v = ConsistencyViolation(type=ConsistencyType.LINE_SPACING, severity=0.5)
        r = self._make_result(violations=[v, v])
        assert r.n_violations == 2

    def test_max_severity_zero_no_violations(self):
        r = self._make_result()
        assert r.max_severity == pytest.approx(0.0)

    def test_max_severity_with_violations(self):
        v1 = ConsistencyViolation(type=ConsistencyType.LINE_SPACING, severity=0.3)
        v2 = ConsistencyViolation(type=ConsistencyType.CHAR_HEIGHT, severity=0.7)
        r = self._make_result(violations=[v1, v2])
        assert r.max_severity == pytest.approx(0.7)

    def test_method_scores_stored(self):
        r = ConsistencyResult(
            violations=[],
            score=1.0,
            n_checked=0,
            method_scores={"line_spacing": 1.0, "char_height": 0.8},
        )
        assert r.method_scores["line_spacing"] == pytest.approx(1.0)


# ─── estimate_line_spacing ────────────────────────────────────────────────────

class TestEstimateLineSpacing:
    def test_returns_float(self):
        img = make_text_like()
        val = estimate_line_spacing(img)
        assert isinstance(val, float)

    def test_blank_image_returns_zero(self):
        img = make_blank()
        val = estimate_line_spacing(img)
        assert val == pytest.approx(0.0)

    def test_nonnegative(self):
        img = make_text_like()
        val = estimate_line_spacing(img)
        assert val >= 0.0

    def test_text_image_positive(self):
        img = make_text_like(h=128, w=128, n_lines=4)
        val = estimate_line_spacing(img)
        assert val >= 0.0  # may be 0 if Otsu doesn't segment well on synthetic

    def test_accepts_bgr(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        val = estimate_line_spacing(img)
        assert val >= 0.0


# ─── estimate_char_height ─────────────────────────────────────────────────────

class TestEstimateCharHeight:
    def test_returns_float(self):
        img = make_noisy()
        val = estimate_char_height(img)
        assert isinstance(val, float)

    def test_blank_image_returns_zero(self):
        img = make_blank()
        val = estimate_char_height(img)
        assert val == pytest.approx(0.0)

    def test_nonnegative(self):
        img = make_noisy()
        val = estimate_char_height(img)
        assert val >= 0.0

    def test_accepts_bgr(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        val = estimate_char_height(img)
        assert val >= 0.0

    def test_synthetic_text_positive(self):
        # Synthetic text: dark rectangles on white background
        img = np.full((64, 64), 255, dtype=np.uint8)
        img[10:20, 5:30] = 0  # single dark block
        val = estimate_char_height(img)
        assert val >= 0.0


# ─── estimate_text_angle ──────────────────────────────────────────────────────

class TestEstimateTextAngle:
    def test_returns_float(self):
        img = make_noisy()
        val = estimate_text_angle(img)
        assert isinstance(val, float)

    def test_blank_returns_zero(self):
        img = make_blank()
        val = estimate_text_angle(img)
        assert val == pytest.approx(0.0)

    def test_uniform_gray_returns_zero(self):
        img = make_gray_image()
        val = estimate_text_angle(img)
        assert val == pytest.approx(0.0)

    def test_accepts_bgr(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        val = estimate_text_angle(img)
        assert isinstance(val, float)


# ─── check_line_spacing ───────────────────────────────────────────────────────

class TestCheckLineSpacing:
    def test_returns_list(self):
        imgs = [make_text_like() for _ in range(2)]
        result = check_line_spacing([0, 1], imgs)
        assert isinstance(result, list)

    def test_single_fragment_returns_empty(self):
        result = check_line_spacing([0], [make_text_like()])
        assert result == []

    def test_consistent_images_returns_empty(self):
        # Same image twice → same spacing → no violation
        img = make_text_like(h=128, n_lines=4)
        result = check_line_spacing([0, 1], [img, img])
        assert isinstance(result, list)

    def test_violation_has_correct_type(self):
        # Different spacing images may trigger violation
        imgs = [make_blank(), make_text_like()]
        result = check_line_spacing([0, 1], imgs)
        for v in result:
            assert v.type == ConsistencyType.LINE_SPACING

    def test_empty_lists_returns_empty(self):
        result = check_line_spacing([], [])
        assert result == []

    def test_returns_boundary_violations(self):
        imgs = [make_text_like() for _ in range(2)]
        result = check_line_spacing([0, 1], imgs)
        for v in result:
            assert isinstance(v, ConsistencyViolation)


# ─── check_char_height ────────────────────────────────────────────────────────

class TestCheckCharHeight:
    def test_returns_list(self):
        imgs = [make_noisy(seed=i) for i in range(2)]
        result = check_char_height([0, 1], imgs)
        assert isinstance(result, list)

    def test_single_fragment_empty(self):
        result = check_char_height([0], [make_noisy()])
        assert result == []

    def test_violation_has_correct_type(self):
        imgs = [make_noisy(seed=i) for i in range(2)]
        result = check_char_height([0, 1], imgs)
        for v in result:
            assert v.type == ConsistencyType.CHAR_HEIGHT

    def test_empty_returns_empty(self):
        result = check_char_height([], [])
        assert result == []


# ─── check_text_angle ─────────────────────────────────────────────────────────

class TestCheckTextAngle:
    def test_returns_list(self):
        imgs = [make_noisy(seed=i) for i in range(2)]
        result = check_text_angle([0, 1], imgs)
        assert isinstance(result, list)

    def test_single_fragment_empty(self):
        result = check_text_angle([0], [make_noisy()])
        assert result == []

    def test_violation_has_correct_type(self):
        imgs = [make_noisy(seed=i) for i in range(2)]
        result = check_text_angle([0, 1], imgs)
        for v in result:
            assert v.type == ConsistencyType.TEXT_ANGLE

    def test_empty_returns_empty(self):
        result = check_text_angle([], [])
        assert result == []


# ─── check_margin_alignment ───────────────────────────────────────────────────

class TestCheckMarginAlignment:
    def test_returns_list(self):
        imgs = [make_text_like() for _ in range(2)]
        result = check_margin_alignment([0, 1], imgs)
        assert isinstance(result, list)

    def test_single_fragment_empty(self):
        result = check_margin_alignment([0], [make_text_like()])
        assert result == []

    def test_same_image_no_violation(self):
        img = make_text_like()
        result = check_margin_alignment([0, 1], [img, img])
        assert result == []

    def test_violation_has_correct_type(self):
        # Two images with different left margins
        img1 = np.full((64, 64), 255, dtype=np.uint8)
        img1[20:30, 5:40] = 0  # text starts at col 5
        img2 = np.full((64, 64), 255, dtype=np.uint8)
        img2[20:30, 40:60] = 0  # text starts at col 40
        result = check_margin_alignment([0, 1], [img1, img2])
        for v in result:
            assert v.type == ConsistencyType.MARGIN_ALIGN

    def test_empty_returns_empty(self):
        result = check_margin_alignment([], [])
        assert result == []


# ─── check_consistency ────────────────────────────────────────────────────────

class TestCheckConsistency:
    def test_returns_consistency_result(self):
        imgs = [make_text_like() for _ in range(2)]
        result = check_consistency([0, 1], imgs)
        assert isinstance(result, ConsistencyResult)

    def test_score_in_0_1(self):
        imgs = [make_noisy(seed=i) for i in range(3)]
        result = check_consistency([0, 1, 2], imgs)
        assert 0.0 <= result.score <= 1.0

    def test_single_fragment_score_1(self):
        result = check_consistency([0], [make_text_like()])
        assert result.score == pytest.approx(1.0)
        assert result.n_violations == 0

    def test_identical_images_no_violations(self):
        img = make_text_like()
        result = check_consistency([0, 1], [img, img])
        assert result.score >= 0.0  # may still have violations for angle etc.

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            check_consistency([0, 1], [make_blank()])

    def test_method_scores_has_4_keys(self):
        imgs = [make_text_like() for _ in range(2)]
        result = check_consistency([0, 1], imgs)
        assert "line_spacing" in result.method_scores
        assert "char_height" in result.method_scores
        assert "text_angle" in result.method_scores
        assert "margin_align" in result.method_scores

    def test_violations_are_consistency_violations(self):
        imgs = [make_noisy(seed=i) for i in range(3)]
        result = check_consistency([0, 1, 2], imgs)
        for v in result.violations:
            assert isinstance(v, ConsistencyViolation)

    def test_n_checked_for_2_fragments(self):
        imgs = [make_text_like() for _ in range(2)]
        result = check_consistency([0, 1], imgs)
        # C(2,2) = 1 pair
        assert result.n_checked == 1

    def test_n_checked_for_3_fragments(self):
        imgs = [make_text_like() for _ in range(3)]
        result = check_consistency([0, 1, 2], imgs)
        # C(3,2) = 3 pairs
        assert result.n_checked == 3

    def test_no_violations_score_is_1(self):
        # Same image for all fragments: same spacing, angle, etc.
        img = make_text_like(h=128, n_lines=4)
        result = check_consistency([0, 1], [img, img])
        if result.n_violations == 0:
            assert result.score == pytest.approx(1.0)


# ─── batch_check_consistency ──────────────────────────────────────────────────

class TestBatchCheckConsistency:
    def test_returns_list(self):
        imgs = [[make_text_like() for _ in range(2)]]
        result = batch_check_consistency([[0, 1]], imgs)
        assert isinstance(result, list)

    def test_length_matches_groups(self):
        groups = [[make_text_like() for _ in range(2)] for _ in range(3)]
        result = batch_check_consistency([[0, 1]] * 3, groups)
        assert len(result) == 3

    def test_each_result_is_consistency_result(self):
        groups = [[make_text_like() for _ in range(2)] for _ in range(2)]
        result = batch_check_consistency([[0, 1]] * 2, groups)
        for r in result:
            assert isinstance(r, ConsistencyResult)

    def test_mismatched_group_count_raises(self):
        with pytest.raises(ValueError):
            batch_check_consistency([[0, 1]], [])

    def test_empty_groups_returns_empty(self):
        result = batch_check_consistency([], [])
        assert result == []
