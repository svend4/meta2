"""Extra tests for puzzle_reconstruction/verification/layout_checker.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.verification.layout_checker import (
    LayoutCheckResult,
    LayoutViolation,
    LayoutViolationType,
    batch_check_layout,
    check_aspect_ratio,
    check_grid_alignment,
    check_layout,
    compute_bounding_box,
    detect_gaps,
    detect_overlaps,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _box(x, y, w, h):
    return (float(x), float(y), float(w), float(h))


def _row_positions(n: int = 3, gap: float = 0.0) -> dict:
    """N boxes in a row with optional gap between them."""
    pos = {}
    for i in range(n):
        pos[i] = _box(i * (50.0 + gap), 0.0, 50.0, 50.0)
    return pos


def _overlapping_positions() -> dict:
    return {
        0: _box(0.0, 0.0, 60.0, 50.0),
        1: _box(50.0, 0.0, 60.0, 50.0),   # 10px overlap
    }


# ─── LayoutViolationType (extra) ──────────────────────────────────────────────

class TestLayoutViolationTypeExtra:
    def test_overlap_value(self):
        assert LayoutViolationType.OVERLAP == "overlap"

    def test_gap_value(self):
        assert LayoutViolationType.GAP == "gap"

    def test_misalignment_value(self):
        assert LayoutViolationType.MISALIGNMENT == "misalignment"

    def test_aspect_ratio_value(self):
        assert LayoutViolationType.ASPECT_RATIO == "aspect_ratio"

    def test_boundary_value(self):
        assert LayoutViolationType.BOUNDARY == "boundary"

    def test_insufficient_value(self):
        assert LayoutViolationType.INSUFFICIENT == "insufficient"

    def test_all_members_are_strings(self):
        for member in LayoutViolationType:
            assert isinstance(member.value, str)


# ─── LayoutViolation (extra) ──────────────────────────────────────────────────

class TestLayoutViolationExtra:
    def test_type_stored(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5)
        assert v.type == LayoutViolationType.OVERLAP

    def test_severity_stored(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.3)
        assert v.severity == pytest.approx(0.3)

    def test_default_fragment_ids_empty(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.5)
        assert v.fragment_ids == []

    def test_default_description_empty(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.5)
        assert v.description == ""

    def test_default_values_empty(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.5)
        assert v.values == {}

    def test_custom_fragment_ids(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.8,
                            fragment_ids=[0, 1])
        assert v.fragment_ids == [0, 1]

    def test_custom_description(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5,
                            description="test")
        assert v.description == "test"

    def test_custom_values(self):
        v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5,
                            values={"area": 25.0})
        assert v.values["area"] == pytest.approx(25.0)


# ─── LayoutCheckResult (extra) ────────────────────────────────────────────────

class TestLayoutCheckResultExtra:
    def _make(self, n_violations: int = 0, score: float = 1.0) -> LayoutCheckResult:
        viols = [
            LayoutViolation(LayoutViolationType.OVERLAP, severity=0.5)
            for _ in range(n_violations)
        ]
        return LayoutCheckResult(violations=viols, score=score, n_checked=5)

    def test_n_violations_zero(self):
        r = self._make(0)
        assert r.n_violations == 0

    def test_n_violations_count(self):
        r = self._make(3)
        assert r.n_violations == 3

    def test_max_severity_empty(self):
        r = self._make(0)
        assert r.max_severity == pytest.approx(0.0)

    def test_max_severity_nonempty(self):
        viols = [
            LayoutViolation(LayoutViolationType.OVERLAP, severity=0.3),
            LayoutViolation(LayoutViolationType.GAP, severity=0.7),
        ]
        r = LayoutCheckResult(violations=viols, score=0.5, n_checked=5)
        assert r.max_severity == pytest.approx(0.7)

    def test_score_stored(self):
        r = self._make(score=0.8)
        assert r.score == pytest.approx(0.8)

    def test_n_checked_stored(self):
        r = LayoutCheckResult(violations=[], score=1.0, n_checked=10)
        assert r.n_checked == 10

    def test_bounding_box_default_none(self):
        r = LayoutCheckResult(violations=[], score=1.0, n_checked=0)
        assert r.bounding_box is None

    def test_method_scores_default_empty(self):
        r = LayoutCheckResult(violations=[], score=1.0, n_checked=0)
        assert r.method_scores == {}

    def test_method_scores_stored(self):
        ms = {"overlap": 1.0, "gap": 0.9}
        r = LayoutCheckResult(violations=[], score=1.0, n_checked=3,
                               method_scores=ms)
        assert r.method_scores["overlap"] == pytest.approx(1.0)


# ─── compute_bounding_box (extra) ─────────────────────────────────────────────

class TestComputeBoundingBoxExtra:
    def test_empty_returns_zeros(self):
        assert compute_bounding_box({}) == (0.0, 0.0, 0.0, 0.0)

    def test_single_box(self):
        bb = compute_bounding_box({0: _box(10, 20, 50, 30)})
        assert bb == pytest.approx((10.0, 20.0, 50.0, 30.0))

    def test_two_boxes_side_by_side(self):
        pos = {0: _box(0, 0, 50, 50), 1: _box(50, 0, 50, 50)}
        bb = compute_bounding_box(pos)
        assert bb[2] == pytest.approx(100.0)
        assert bb[3] == pytest.approx(50.0)

    def test_negative_coordinates(self):
        pos = {0: _box(-10, -20, 30, 40)}
        bb = compute_bounding_box(pos)
        assert bb[0] == pytest.approx(-10.0)
        assert bb[1] == pytest.approx(-20.0)

    def test_returns_4_tuple(self):
        bb = compute_bounding_box({0: _box(0, 0, 10, 10)})
        assert len(bb) == 4

    def test_stacked_boxes(self):
        pos = {0: _box(0, 0, 50, 50), 1: _box(0, 50, 50, 50)}
        bb = compute_bounding_box(pos)
        assert bb[3] == pytest.approx(100.0)


# ─── detect_overlaps (extra) ──────────────────────────────────────────────────

class TestDetectOverlapsExtra:
    def test_no_overlap_returns_empty(self):
        pos = _row_positions(3, gap=0.0)
        ids = list(pos.keys())
        result = detect_overlaps(ids, pos)
        assert result == []

    def test_overlapping_pair_detected(self):
        pos = _overlapping_positions()
        ids = [0, 1]
        result = detect_overlaps(ids, pos, min_overlap=5.0)
        assert len(result) >= 1

    def test_violation_type_is_overlap(self):
        pos = _overlapping_positions()
        result = detect_overlaps([0, 1], pos, min_overlap=5.0)
        for v in result:
            assert v.type == LayoutViolationType.OVERLAP

    def test_severity_in_0_1(self):
        pos = _overlapping_positions()
        result = detect_overlaps([0, 1], pos, min_overlap=5.0)
        for v in result:
            assert 0.0 <= v.severity <= 1.0

    def test_fragment_ids_in_violation(self):
        pos = _overlapping_positions()
        result = detect_overlaps([0, 1], pos, min_overlap=5.0)
        if result:
            assert 0 in result[0].fragment_ids
            assert 1 in result[0].fragment_ids

    def test_empty_ids_returns_empty(self):
        assert detect_overlaps([], {}) == []

    def test_single_fragment_returns_empty(self):
        pos = {0: _box(0, 0, 50, 50)}
        assert detect_overlaps([0], pos) == []

    def test_high_min_overlap_no_violation(self):
        pos = _overlapping_positions()
        result = detect_overlaps([0, 1], pos, min_overlap=999.0)
        assert result == []


# ─── detect_gaps (extra) ──────────────────────────────────────────────────────

class TestDetectGapsExtra:
    def test_perfect_row_no_gaps(self):
        pos = _row_positions(3, gap=0.0)
        result = detect_gaps(list(pos.keys()), pos, gap_tol=5.0)
        assert result == []

    def test_large_gap_detected(self):
        pos = {
            0: _box(0.0, 0.0, 50.0, 50.0),
            1: _box(150.0, 0.0, 50.0, 50.0),  # 100px gap
        }
        result = detect_gaps([0, 1], pos, expected_gap=0.0, gap_tol=5.0)
        assert len(result) >= 1

    def test_violation_type_is_gap(self):
        pos = {
            0: _box(0.0, 0.0, 50.0, 50.0),
            1: _box(150.0, 0.0, 50.0, 50.0),
        }
        result = detect_gaps([0, 1], pos, gap_tol=5.0)
        for v in result:
            assert v.type == LayoutViolationType.GAP

    def test_empty_ids_returns_empty(self):
        assert detect_gaps([], {}) == []

    def test_single_fragment_returns_empty(self):
        pos = {0: _box(0, 0, 50, 50)}
        assert detect_gaps([0], pos) == []

    def test_gap_within_tol_no_violation(self):
        pos = {
            0: _box(0.0, 0.0, 50.0, 50.0),
            1: _box(53.0, 0.0, 50.0, 50.0),  # 3px gap, tol=10
        }
        result = detect_gaps([0, 1], pos, gap_tol=10.0)
        assert result == []


# ─── check_grid_alignment (extra) ─────────────────────────────────────────────

class TestCheckGridAlignmentExtra:
    def test_aligned_grid_no_violations(self):
        pos = {
            0: _box(0, 0, 50, 50),
            1: _box(50, 0, 50, 50),
            2: _box(0, 50, 50, 50),
            3: _box(50, 50, 50, 50),
        }
        result = check_grid_alignment([0, 1, 2, 3], pos, tol_px=5.0)
        assert result == []

    def test_single_fragment_returns_empty(self):
        pos = {0: _box(0, 0, 50, 50)}
        assert check_grid_alignment([0], pos) == []

    def test_empty_returns_empty(self):
        assert check_grid_alignment([], {}) == []

    def test_violation_type_is_misalignment(self):
        pos = {
            0: _box(0, 0, 50, 50),
            1: _box(50, 0, 50, 50),
            2: _box(25, 50, 50, 50),    # badly misaligned
        }
        result = check_grid_alignment([0, 1, 2], pos, tol_px=2.0)
        for v in result:
            assert v.type == LayoutViolationType.MISALIGNMENT

    def test_severity_in_0_1(self):
        pos = {
            0: _box(0, 0, 50, 50),
            1: _box(50, 0, 50, 50),
            2: _box(999, 50, 50, 50),
        }
        result = check_grid_alignment([0, 1, 2], pos, tol_px=1.0)
        for v in result:
            assert 0.0 <= v.severity <= 1.0


# ─── check_aspect_ratio (extra) ───────────────────────────────────────────────

class TestCheckAspectRatioExtra:
    def test_no_expected_ratio_returns_empty(self):
        pos = {0: _box(0, 0, 100, 50)}
        result = check_aspect_ratio([0], pos, expected_ratio=None)
        assert result == []

    def test_exact_ratio_no_violation(self):
        pos = {0: _box(0, 0, 200, 100)}  # 2:1 ratio
        result = check_aspect_ratio([0], pos, expected_ratio=2.0, tol_ratio=0.1)
        assert result == []

    def test_wrong_ratio_violation(self):
        pos = {0: _box(0, 0, 300, 100)}  # 3:1 actual vs 1:1 expected
        result = check_aspect_ratio([0], pos, expected_ratio=1.0, tol_ratio=0.1)
        assert len(result) == 1

    def test_violation_type_is_aspect_ratio(self):
        pos = {0: _box(0, 0, 300, 100)}
        result = check_aspect_ratio([0], pos, expected_ratio=1.0, tol_ratio=0.1)
        if result:
            assert result[0].type == LayoutViolationType.ASPECT_RATIO

    def test_empty_ids_returns_empty(self):
        result = check_aspect_ratio([], {}, expected_ratio=1.0)
        assert result == []

    def test_severity_in_0_1(self):
        pos = {0: _box(0, 0, 500, 100)}
        result = check_aspect_ratio([0], pos, expected_ratio=1.0, tol_ratio=0.1)
        if result:
            assert 0.0 <= result[0].severity <= 1.0


# ─── check_layout (extra) ─────────────────────────────────────────────────────

class TestCheckLayoutExtra:
    def test_returns_layout_check_result(self):
        pos = _row_positions(3)
        result = check_layout([0, 1, 2], pos)
        assert isinstance(result, LayoutCheckResult)

    def test_single_fragment_score_1(self):
        pos = {0: _box(0, 0, 50, 50)}
        result = check_layout([0], pos)
        assert result.score == pytest.approx(1.0)

    def test_empty_fragments_score_1(self):
        result = check_layout([], {})
        assert result.score == pytest.approx(1.0)

    def test_perfect_grid_high_score(self):
        pos = _row_positions(3)
        result = check_layout([0, 1, 2], pos)
        assert result.score > 0.0

    def test_overlapping_lowers_score(self):
        pos = _overlapping_positions()
        perfect = _row_positions(2)
        r_bad = check_layout([0, 1], pos)
        r_good = check_layout([0, 1], perfect)
        assert r_bad.score <= r_good.score

    def test_n_checked_correct(self):
        pos = _row_positions(4)
        result = check_layout([0, 1, 2, 3], pos)
        assert result.n_checked == 4

    def test_bounding_box_present(self):
        pos = _row_positions(3)
        result = check_layout([0, 1, 2], pos)
        assert result.bounding_box is not None

    def test_method_scores_present(self):
        pos = _row_positions(3)
        result = check_layout([0, 1, 2], pos)
        assert "overlap" in result.method_scores
        assert "gap" in result.method_scores
        assert "alignment" in result.method_scores

    def test_score_in_0_1(self):
        pos = _overlapping_positions()
        result = check_layout([0, 1], pos)
        assert 0.0 <= result.score <= 1.0

    def test_aspect_ratio_with_expected(self):
        pos = {0: _box(0, 0, 200, 100), 1: _box(200, 0, 200, 100)}
        result = check_layout([0, 1], pos, expected_ratio=2.0, ratio_tol=0.1)
        assert isinstance(result, LayoutCheckResult)


# ─── batch_check_layout (extra) ───────────────────────────────────────────────

class TestBatchCheckLayoutExtra:
    def test_empty_groups_returns_empty(self):
        result = batch_check_layout([], [])
        assert result == []

    def test_single_group(self):
        pos = _row_positions(3)
        result = batch_check_layout([[0, 1, 2]], [pos])
        assert len(result) == 1
        assert isinstance(result[0], LayoutCheckResult)

    def test_multiple_groups(self):
        pos1 = _row_positions(3)
        pos2 = _row_positions(2)
        result = batch_check_layout([[0, 1, 2], [0, 1]], [pos1, pos2])
        assert len(result) == 2

    def test_mismatched_lengths_raises(self):
        pos = _row_positions(3)
        with pytest.raises(ValueError):
            batch_check_layout([[0, 1, 2]], [{}, {}])

    def test_kwargs_passed(self):
        pos = _overlapping_positions()
        result = batch_check_layout([[0, 1]], [pos], overlap_min=1000.0)
        assert result[0].n_violations == 0

    def test_all_results_are_layout_check_result(self):
        groups = [list(range(i)) for i in range(1, 4)]
        pos_groups = [_row_positions(i) for i in range(1, 4)]
        results = batch_check_layout(groups, pos_groups)
        for r in results:
            assert isinstance(r, LayoutCheckResult)
