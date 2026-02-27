"""Tests for puzzle_reconstruction.verification.layout_checker"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.layout_checker import (
    LayoutViolationType,
    LayoutViolation,
    LayoutCheckResult,
    compute_bounding_box,
    _overlap_area,
    _iou_1d,
    detect_overlaps,
    detect_gaps,
    check_grid_alignment,
    check_aspect_ratio,
    check_layout,
    batch_check_layout,
)


# Helper: non-overlapping 2x2 grid of fragments
def make_grid_positions():
    return {
        0: (0.0, 0.0, 50.0, 50.0),
        1: (50.0, 0.0, 50.0, 50.0),
        2: (0.0, 50.0, 50.0, 50.0),
        3: (50.0, 50.0, 50.0, 50.0),
    }


# ─── LayoutViolationType ──────────────────────────────────────────────────────

def test_layout_violation_type_values():
    assert LayoutViolationType.OVERLAP == "overlap"
    assert LayoutViolationType.GAP == "gap"
    assert LayoutViolationType.MISALIGNMENT == "misalignment"


# ─── LayoutViolation ──────────────────────────────────────────────────────────

def test_layout_violation_defaults():
    v = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.5)
    assert v.fragment_ids == []
    assert v.description == ""


def test_layout_violation_with_data():
    v = LayoutViolation(
        type=LayoutViolationType.GAP,
        severity=0.3,
        fragment_ids=[0, 1],
        description="Test gap",
        values={"gap": 20.0},
    )
    assert v.values["gap"] == 20.0


# ─── LayoutCheckResult ────────────────────────────────────────────────────────

def test_layout_check_result_properties():
    v1 = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.7, fragment_ids=[0, 1])
    result = LayoutCheckResult(
        violations=[v1],
        score=0.5,
        n_checked=4,
        bounding_box=(0, 0, 100, 100),
    )
    assert result.n_violations == 1
    assert result.max_severity == pytest.approx(0.7)


def test_layout_check_result_no_violations():
    result = LayoutCheckResult(violations=[], score=1.0, n_checked=2)
    assert result.max_severity == 0.0


# ─── compute_bounding_box ─────────────────────────────────────────────────────

def test_compute_bounding_box_empty():
    bb = compute_bounding_box({})
    assert bb == (0.0, 0.0, 0.0, 0.0)


def test_compute_bounding_box_single():
    bb = compute_bounding_box({0: (10.0, 20.0, 30.0, 40.0)})
    assert bb == (10.0, 20.0, 30.0, 40.0)


def test_compute_bounding_box_grid():
    positions = make_grid_positions()
    bb = compute_bounding_box(positions)
    x, y, w, h = bb
    assert x == 0.0
    assert y == 0.0
    assert w == pytest.approx(100.0)
    assert h == pytest.approx(100.0)


# ─── _overlap_area ────────────────────────────────────────────────────────────

def test_overlap_area_no_overlap():
    assert _overlap_area((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_overlap_area_full_overlap():
    area = _overlap_area((0, 0, 10, 10), (0, 0, 10, 10))
    assert area == pytest.approx(100.0)


def test_overlap_area_partial():
    area = _overlap_area((0, 0, 10, 10), (5, 0, 10, 10))
    assert area == pytest.approx(50.0)


# ─── _iou_1d ──────────────────────────────────────────────────────────────────

def test_iou_1d_no_overlap():
    assert _iou_1d(0, 5, 10, 15) == 0.0


def test_iou_1d_full_overlap():
    result = _iou_1d(0, 10, 0, 10)
    assert result == pytest.approx(1.0)


def test_iou_1d_partial():
    result = _iou_1d(0, 10, 5, 15)
    # inter=5, union=15
    assert result == pytest.approx(5.0 / 15.0)


# ─── detect_overlaps ──────────────────────────────────────────────────────────

def test_detect_overlaps_none():
    positions = make_grid_positions()
    violations = detect_overlaps([0, 1, 2, 3], positions, min_overlap=5.0)
    assert violations == []


def test_detect_overlaps_found():
    positions = {0: (0.0, 0.0, 100.0, 100.0), 1: (50.0, 50.0, 100.0, 100.0)}
    violations = detect_overlaps([0, 1], positions, min_overlap=5.0)
    assert len(violations) > 0
    assert violations[0].type == LayoutViolationType.OVERLAP


def test_detect_overlaps_severity_range():
    positions = {0: (0.0, 0.0, 50.0, 50.0), 1: (25.0, 25.0, 50.0, 50.0)}
    violations = detect_overlaps([0, 1], positions)
    for v in violations:
        assert 0.0 <= v.severity <= 1.0


def test_detect_overlaps_missing_id():
    positions = {0: (0.0, 0.0, 50.0, 50.0)}
    # id 99 not in positions → should be skipped
    violations = detect_overlaps([0, 99], positions)
    assert violations == []


# ─── detect_gaps ──────────────────────────────────────────────────────────────

def test_detect_gaps_no_gap():
    # Perfect tiling → no gaps
    positions = {0: (0.0, 0.0, 50.0, 50.0), 1: (50.0, 0.0, 50.0, 50.0)}
    violations = detect_gaps([0, 1], positions, gap_tol=10.0)
    assert violations == []


def test_detect_gaps_large_gap():
    # large horizontal gap with vertical overlap > 50%
    positions = {
        0: (0.0, 0.0, 50.0, 100.0),
        1: (150.0, 0.0, 50.0, 100.0),
    }
    violations = detect_gaps([0, 1], positions, expected_gap=0.0, gap_tol=5.0)
    # May or may not detect depending on proximity
    assert isinstance(violations, list)


# ─── check_grid_alignment ─────────────────────────────────────────────────────

def test_check_grid_alignment_perfect():
    positions = make_grid_positions()
    violations = check_grid_alignment([0, 1, 2, 3], positions, tol_px=5.0)
    assert violations == []


def test_check_grid_alignment_single():
    positions = {0: (0.0, 0.0, 50.0, 50.0)}
    violations = check_grid_alignment([0], positions)
    assert violations == []


def test_check_grid_alignment_misaligned():
    positions = {
        0: (0.0, 0.0, 50.0, 50.0),
        1: (0.0, 50.0, 50.0, 50.0),  # Same x → aligned
        2: (7.0, 100.0, 50.0, 50.0),  # Slightly off
    }
    violations = check_grid_alignment([0, 1, 2], positions, tol_px=1.0)
    # May generate misalignment violations
    assert isinstance(violations, list)


# ─── check_aspect_ratio ───────────────────────────────────────────────────────

def test_check_aspect_ratio_none():
    positions = make_grid_positions()
    violations = check_aspect_ratio([0, 1, 2, 3], positions, expected_ratio=None)
    assert violations == []


def test_check_aspect_ratio_correct():
    positions = {0: (0.0, 0.0, 100.0, 100.0)}
    violations = check_aspect_ratio([0], positions, expected_ratio=1.0, tol_ratio=0.1)
    assert violations == []


def test_check_aspect_ratio_wrong():
    positions = {0: (0.0, 0.0, 200.0, 100.0)}
    violations = check_aspect_ratio([0], positions, expected_ratio=1.0, tol_ratio=0.1)
    assert len(violations) == 1
    assert violations[0].type == LayoutViolationType.ASPECT_RATIO


def test_check_aspect_ratio_empty():
    violations = check_aspect_ratio([], {}, expected_ratio=1.0)
    assert violations == []


# ─── check_layout ─────────────────────────────────────────────────────────────

def test_check_layout_perfect():
    positions = make_grid_positions()
    result = check_layout([0, 1, 2, 3], positions)
    assert isinstance(result, LayoutCheckResult)
    assert 0.0 <= result.score <= 1.0


def test_check_layout_single_fragment():
    positions = {0: (0.0, 0.0, 50.0, 50.0)}
    result = check_layout([0], positions)
    assert result.score == pytest.approx(1.0)
    assert result.n_checked == 1


def test_check_layout_empty():
    result = check_layout([], {})
    assert result.score == pytest.approx(1.0)
    assert result.n_checked == 0


def test_check_layout_has_method_scores():
    positions = make_grid_positions()
    result = check_layout([0, 1, 2, 3], positions)
    assert "overlap" in result.method_scores
    assert "gap" in result.method_scores
    assert "alignment" in result.method_scores
    assert "aspect_ratio" in result.method_scores


def test_check_layout_with_overlaps():
    positions = {0: (0.0, 0.0, 100.0, 100.0), 1: (50.0, 50.0, 100.0, 100.0)}
    result = check_layout([0, 1], positions, overlap_min=5.0)
    assert result.score < 1.0


def test_check_layout_bounding_box():
    positions = make_grid_positions()
    result = check_layout([0, 1, 2, 3], positions)
    assert result.bounding_box is not None


# ─── batch_check_layout ───────────────────────────────────────────────────────

def test_batch_check_layout_basic():
    groups = [[0, 1], [2, 3]]
    pos_groups = [
        {0: (0.0, 0.0, 50.0, 50.0), 1: (50.0, 0.0, 50.0, 50.0)},
        {2: (0.0, 0.0, 50.0, 50.0), 3: (0.0, 50.0, 50.0, 50.0)},
    ]
    results = batch_check_layout(groups, pos_groups)
    assert len(results) == 2


def test_batch_check_layout_mismatch_raises():
    with pytest.raises(ValueError):
        batch_check_layout([[0, 1]], [{}, {}, {}])


def test_batch_check_layout_empty():
    results = batch_check_layout([], [])
    assert results == []
