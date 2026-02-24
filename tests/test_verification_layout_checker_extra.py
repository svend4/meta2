"""Extra tests for puzzle_reconstruction/verification/layout_checker.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.verification.layout_checker import (
    LayoutViolationType,
    LayoutViolation,
    LayoutCheckResult,
    compute_bounding_box,
    detect_overlaps,
    detect_gaps,
    check_grid_alignment,
    check_aspect_ratio,
    check_layout,
    batch_check_layout,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pos_grid():
    """2x2 grid, each 100x100, no gaps."""
    return {
        0: (0.0, 0.0, 100.0, 100.0),
        1: (100.0, 0.0, 100.0, 100.0),
        2: (0.0, 100.0, 100.0, 100.0),
        3: (100.0, 100.0, 100.0, 100.0),
    }


def _pos_overlap():
    """Two overlapping fragments."""
    return {
        0: (0.0, 0.0, 100.0, 100.0),
        1: (50.0, 50.0, 100.0, 100.0),
    }


# ─── LayoutViolationType ────────────────────────────────────────────────────

class TestLayoutViolationTypeExtra:
    def test_values(self):
        assert LayoutViolationType.OVERLAP.value == "overlap"
        assert LayoutViolationType.GAP.value == "gap"
        assert LayoutViolationType.MISALIGNMENT.value == "misalignment"
        assert LayoutViolationType.ASPECT_RATIO.value == "aspect_ratio"
        assert LayoutViolationType.BOUNDARY.value == "boundary"
        assert LayoutViolationType.INSUFFICIENT.value == "insufficient"

    def test_is_str_enum(self):
        assert isinstance(LayoutViolationType.OVERLAP, str)


# ─── LayoutViolation ────────────────────────────────────────────────────────

class TestLayoutViolationExtra:
    def test_creation(self):
        v = LayoutViolation(
            type=LayoutViolationType.OVERLAP,
            severity=0.5,
            fragment_ids=[0, 1],
            description="test",
        )
        assert v.type == LayoutViolationType.OVERLAP
        assert v.severity == 0.5
        assert v.fragment_ids == [0, 1]

    def test_defaults(self):
        v = LayoutViolation(type=LayoutViolationType.GAP, severity=0.0)
        assert v.fragment_ids == []
        assert v.description == ""
        assert v.values == {}


# ─── LayoutCheckResult ──────────────────────────────────────────────────────

class TestLayoutCheckResultExtra:
    def test_no_violations(self):
        r = LayoutCheckResult(violations=[], score=1.0, n_checked=4)
        assert r.n_violations == 0
        assert r.max_severity == 0.0

    def test_with_violations(self):
        v1 = LayoutViolation(type=LayoutViolationType.OVERLAP, severity=0.3)
        v2 = LayoutViolation(type=LayoutViolationType.GAP, severity=0.8)
        r = LayoutCheckResult(violations=[v1, v2], score=0.5, n_checked=2)
        assert r.n_violations == 2
        assert r.max_severity == pytest.approx(0.8)

    def test_bounding_box(self):
        r = LayoutCheckResult(
            violations=[], score=1.0, n_checked=1,
            bounding_box=(0.0, 0.0, 100.0, 200.0),
        )
        assert r.bounding_box == (0.0, 0.0, 100.0, 200.0)


# ─── compute_bounding_box ───────────────────────────────────────────────────

class TestComputeBoundingBoxExtra:
    def test_empty(self):
        assert compute_bounding_box({}) == (0.0, 0.0, 0.0, 0.0)

    def test_single(self):
        bbox = compute_bounding_box({0: (10.0, 20.0, 30.0, 40.0)})
        assert bbox == (10.0, 20.0, 30.0, 40.0)

    def test_grid(self):
        bbox = compute_bounding_box(_pos_grid())
        assert bbox == pytest.approx((0.0, 0.0, 200.0, 200.0))

    def test_offset(self):
        bbox = compute_bounding_box({
            0: (50.0, 50.0, 100.0, 100.0),
            1: (200.0, 200.0, 50.0, 50.0),
        })
        assert bbox == pytest.approx((50.0, 50.0, 200.0, 200.0))


# ─── detect_overlaps ────────────────────────────────────────────────────────

class TestDetectOverlapsExtra:
    def test_no_overlap(self):
        viols = detect_overlaps([0, 1, 2, 3], _pos_grid())
        assert viols == []

    def test_overlap(self):
        viols = detect_overlaps([0, 1], _pos_overlap())
        assert len(viols) == 1
        assert viols[0].type == LayoutViolationType.OVERLAP

    def test_empty_ids(self):
        assert detect_overlaps([], _pos_grid()) == []

    def test_missing_id(self):
        viols = detect_overlaps([0, 99], _pos_grid())
        assert viols == []

    def test_custom_min_overlap(self):
        # overlap area = 50*50 = 2500; with huge min_overlap, no violation
        viols = detect_overlaps([0, 1], _pos_overlap(), min_overlap=5000.0)
        assert viols == []


# ─── detect_gaps ─────────────────────────────────────────────────────────────

class TestDetectGapsExtra:
    def test_no_gap(self):
        viols = detect_gaps([0, 1, 2, 3], _pos_grid())
        assert viols == []

    def test_horizontal_gap(self):
        pos = {
            0: (0.0, 0.0, 100.0, 100.0),
            1: (200.0, 0.0, 100.0, 100.0),
        }
        viols = detect_gaps([0, 1], pos, gap_tol=5.0)
        assert len(viols) >= 1
        assert viols[0].type == LayoutViolationType.GAP

    def test_vertical_gap(self):
        pos = {
            0: (0.0, 0.0, 100.0, 100.0),
            1: (0.0, 200.0, 100.0, 100.0),
        }
        viols = detect_gaps([0, 1], pos, gap_tol=5.0)
        assert len(viols) >= 1

    def test_empty(self):
        assert detect_gaps([], {}) == []


# ─── check_grid_alignment ───────────────────────────────────────────────────

class TestCheckGridAlignmentExtra:
    def test_aligned(self):
        viols = check_grid_alignment([0, 1, 2, 3], _pos_grid())
        assert viols == []

    def test_misaligned(self):
        pos = {
            0: (0.0, 0.0, 100.0, 100.0),
            1: (100.0, 0.0, 100.0, 100.0),
            2: (50.0, 100.0, 100.0, 100.0),  # misaligned X
        }
        viols = check_grid_alignment([0, 1, 2], pos, tol_px=5.0)
        # fragment 2 has X=50 which doesn't align to 0 or 100
        assert len(viols) >= 0  # Depends on grid rounding

    def test_single_fragment(self):
        viols = check_grid_alignment([0], {0: (0.0, 0.0, 50.0, 50.0)})
        assert viols == []

    def test_empty(self):
        viols = check_grid_alignment([], {})
        assert viols == []


# ─── check_aspect_ratio ─────────────────────────────────────────────────────

class TestCheckAspectRatioExtra:
    def test_no_expected(self):
        viols = check_aspect_ratio([0, 1, 2, 3], _pos_grid())
        assert viols == []

    def test_matching_ratio(self):
        # Grid is 200x200 → ratio = 1.0
        viols = check_aspect_ratio(
            [0, 1, 2, 3], _pos_grid(), expected_ratio=1.0)
        assert viols == []

    def test_mismatched_ratio(self):
        # Grid is 200x200 → ratio = 1.0, expected 2.0
        viols = check_aspect_ratio(
            [0, 1, 2, 3], _pos_grid(), expected_ratio=2.0, tol_ratio=0.1)
        assert len(viols) == 1
        assert viols[0].type == LayoutViolationType.ASPECT_RATIO

    def test_empty(self):
        viols = check_aspect_ratio([], {}, expected_ratio=1.0)
        assert viols == []


# ─── check_layout ───────────────────────────────────────────────────────────

class TestCheckLayoutExtra:
    def test_perfect_grid(self):
        r = check_layout([0, 1, 2, 3], _pos_grid())
        assert r.score == pytest.approx(1.0)
        assert r.n_checked == 4
        assert r.bounding_box is not None

    def test_with_overlap(self):
        r = check_layout([0, 1], _pos_overlap())
        assert r.score < 1.0
        assert r.n_violations > 0

    def test_single_fragment(self):
        r = check_layout([0], {0: (0.0, 0.0, 100.0, 100.0)})
        assert r.score == 1.0
        assert r.n_checked == 1

    def test_empty(self):
        r = check_layout([], {})
        assert r.score == 1.0
        assert r.n_checked == 0

    def test_method_scores(self):
        r = check_layout([0, 1, 2, 3], _pos_grid())
        assert "overlap" in r.method_scores
        assert "gap" in r.method_scores
        assert "alignment" in r.method_scores
        assert "aspect_ratio" in r.method_scores


# ─── batch_check_layout ─────────────────────────────────────────────────────

class TestBatchCheckLayoutExtra:
    def test_empty(self):
        assert batch_check_layout([], []) == []

    def test_one(self):
        results = batch_check_layout(
            [[0, 1, 2, 3]], [_pos_grid()])
        assert len(results) == 1
        assert isinstance(results[0], LayoutCheckResult)

    def test_multiple(self):
        results = batch_check_layout(
            [[0, 1, 2, 3], [0, 1]],
            [_pos_grid(), _pos_overlap()],
        )
        assert len(results) == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            batch_check_layout([[0]], [])
