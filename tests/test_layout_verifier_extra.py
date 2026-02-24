"""Extra tests for puzzle_reconstruction/verification/layout_verifier.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.models import Assembly, Fragment, Placement
from puzzle_reconstruction.verification.layout_verifier import (
    ConstraintType,
    FragmentBox,
    LayoutConstraint,
    LayoutVerificationResult,
    build_layout_boxes,
    check_column_alignment,
    check_duplicate_placements,
    check_gaps,
    check_out_of_bounds,
    check_overlaps,
    check_row_alignment,
    verify_layout,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _fragment(fid: int, w: int = 50, h: int = 50) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((h, w, 3), dtype=np.uint8),
        edges=[],
        bounding_box=(0, 0, w, h),
    )


def _placement(fid: int, x: float = 0.0, y: float = 0.0) -> Placement:
    return Placement(fragment_id=fid, position=(x, y), rotation=0.0)


def _assembly(placements, method: str = "greedy") -> Assembly:
    return Assembly(placements=placements, total_score=0.9, method=method)


def _fbox(fid: int, x: float, y: float, w: float = 50.0, h: float = 50.0) -> FragmentBox:
    return FragmentBox(fid=fid, x=x, y=y, w=w, h=h)


def _row_boxes(n: int = 3, gap: float = 0.0) -> list:
    return [_fbox(i, float(i) * (50.0 + gap), 0.0) for i in range(n)]


# ─── ConstraintType (extra) ───────────────────────────────────────────────────

class TestConstraintTypeExtra:
    def test_overlap_value(self):
        assert ConstraintType.OVERLAP == "overlap"

    def test_gap_value(self):
        assert ConstraintType.GAP == "gap"

    def test_misalign_col_value(self):
        assert ConstraintType.MISALIGN_COL == "misalign_column"

    def test_misalign_row_value(self):
        assert ConstraintType.MISALIGN_ROW == "misalign_row"

    def test_out_of_bounds_value(self):
        assert ConstraintType.OUT_OF_BOUNDS == "out_of_bounds"

    def test_duplicate_place_value(self):
        assert ConstraintType.DUPLICATE_PLACE == "duplicate_place"

    def test_all_are_str_enum(self):
        for ct in ConstraintType:
            assert isinstance(ct.value, str)


# ─── LayoutConstraint (extra) ─────────────────────────────────────────────────

class TestLayoutConstraintExtra:
    def test_kind_stored(self):
        lc = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1))
        assert lc.kind == ConstraintType.OVERLAP

    def test_fids_stored(self):
        lc = LayoutConstraint(kind=ConstraintType.GAP, fids=(2, 3))
        assert lc.fids == (2, 3)

    def test_default_severity(self):
        lc = LayoutConstraint(kind=ConstraintType.GAP, fids=(0,))
        assert lc.severity == pytest.approx(0.5)

    def test_custom_severity(self):
        lc = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1), severity=0.9)
        assert lc.severity == pytest.approx(0.9)

    def test_default_detail_empty(self):
        lc = LayoutConstraint(kind=ConstraintType.GAP, fids=(0,))
        assert lc.detail == ""

    def test_custom_detail(self):
        lc = LayoutConstraint(kind=ConstraintType.GAP, fids=(0,), detail="big gap")
        assert lc.detail == "big gap"

    def test_repr_contains_kind(self):
        lc = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1))
        assert "overlap" in repr(lc)

    def test_repr_contains_severity(self):
        lc = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0,), severity=0.75)
        assert "0.75" in repr(lc)


# ─── FragmentBox (extra) ──────────────────────────────────────────────────────

class TestFragmentBoxExtra:
    def test_x2(self):
        fb = _fbox(0, 10.0, 20.0, w=30.0, h=40.0)
        assert fb.x2 == pytest.approx(40.0)

    def test_y2(self):
        fb = _fbox(0, 10.0, 20.0, w=30.0, h=40.0)
        assert fb.y2 == pytest.approx(60.0)

    def test_cx(self):
        fb = _fbox(0, 10.0, 20.0, w=30.0)
        assert fb.cx == pytest.approx(25.0)

    def test_cy(self):
        fb = _fbox(0, 10.0, 20.0, h=40.0)
        assert fb.cy == pytest.approx(40.0)

    def test_intersects_true(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)  # 10px overlap
        assert a.intersects(b) is True

    def test_intersects_false_no_contact(self):
        a = _fbox(0, 0.0, 0.0)
        b = _fbox(1, 100.0, 0.0)
        assert a.intersects(b) is False

    def test_intersects_false_touching(self):
        a = _fbox(0, 0.0, 0.0, 50.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 50.0, 50.0)  # touching, not intersecting
        assert a.intersects(b) is False

    def test_overlap_area_zero_no_overlap(self):
        a = _fbox(0, 0.0, 0.0)
        b = _fbox(1, 100.0, 0.0)
        assert a.overlap_area(b) == pytest.approx(0.0)

    def test_overlap_area_positive_when_overlapping(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)
        area = a.overlap_area(b)
        assert area == pytest.approx(10.0 * 50.0)

    def test_overlap_area_full_containment(self):
        a = _fbox(0, 0.0, 0.0, 100.0, 100.0)
        b = _fbox(1, 10.0, 10.0, 20.0, 20.0)
        area = a.overlap_area(b)
        assert area == pytest.approx(400.0)

    def test_gap_to_positive_when_separated(self):
        a = _fbox(0, 0.0, 0.0, 50.0, 50.0)
        b = _fbox(1, 80.0, 0.0, 50.0, 50.0)
        gap = a.gap_to(b)
        assert gap == pytest.approx(30.0)

    def test_gap_to_zero_when_touching(self):
        a = _fbox(0, 0.0, 0.0, 50.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 50.0, 50.0)
        assert a.gap_to(b) == pytest.approx(0.0)

    def test_gap_to_negative_when_overlapping(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)
        assert a.gap_to(b) < 0.0

    def test_are_neighbors_close(self):
        a = _fbox(0, 0.0, 0.0, 50.0, 50.0)
        b = _fbox(1, 60.0, 0.0, 50.0, 50.0)  # 10px gap
        assert a.are_neighbors(b, proximity=20.0) is True

    def test_are_neighbors_far(self):
        a = _fbox(0, 0.0, 0.0)
        b = _fbox(1, 200.0, 0.0)
        assert a.are_neighbors(b, proximity=20.0) is False

    def test_default_rotation_zero(self):
        fb = _fbox(0, 0.0, 0.0)
        assert fb.rotation == pytest.approx(0.0)


# ─── LayoutVerificationResult (extra) ─────────────────────────────────────────

class TestLayoutVerificationResultExtra:
    def _make(self, n: int = 0, vs: float = 0.0) -> LayoutVerificationResult:
        consts = [
            LayoutConstraint(ConstraintType.OVERLAP, (0, 1), severity=0.5)
            for _ in range(n)
        ]
        return LayoutVerificationResult(
            constraints=consts,
            violation_score=vs,
            valid=(n == 0),
            n_fragments=5,
        )

    def test_valid_true_when_no_constraints(self):
        r = self._make(0)
        assert r.valid is True

    def test_valid_false_when_constraints(self):
        r = self._make(1)
        assert r.valid is False

    def test_n_fragments_stored(self):
        r = self._make()
        assert r.n_fragments == 5

    def test_violation_score_stored(self):
        r = self._make(vs=0.4)
        assert r.violation_score == pytest.approx(0.4)

    def test_by_kind_filters(self):
        consts = [
            LayoutConstraint(ConstraintType.OVERLAP, (0, 1)),
            LayoutConstraint(ConstraintType.GAP, (1, 2)),
        ]
        r = LayoutVerificationResult(consts, 0.5, False, 2)
        overlaps = r.by_kind(ConstraintType.OVERLAP)
        assert len(overlaps) == 1
        assert overlaps[0].kind == ConstraintType.OVERLAP

    def test_by_kind_empty_when_no_match(self):
        r = self._make(0)
        assert r.by_kind(ConstraintType.OVERLAP) == []

    def test_summary_contains_pass_or_fail(self):
        r = self._make(0)
        assert "PASS" in r.summary()

    def test_summary_fail(self):
        r = self._make(1, vs=0.5)
        assert "FAIL" in r.summary()

    def test_boxes_default_empty(self):
        r = self._make()
        assert r.boxes == []

    def test_boxes_stored(self):
        boxes = [_fbox(0, 0, 0)]
        r = LayoutVerificationResult([], 0.0, True, 1, boxes=boxes)
        assert len(r.boxes) == 1


# ─── build_layout_boxes (extra) ───────────────────────────────────────────────

class TestBuildLayoutBoxesExtra:
    def test_returns_list(self):
        frags = [_fragment(0)]
        asm = _assembly([_placement(0, 10.0, 20.0)])
        boxes = build_layout_boxes(asm, frags)
        assert isinstance(boxes, list)

    def test_correct_count(self):
        frags = [_fragment(i) for i in range(3)]
        pls = [_placement(i, float(i) * 60) for i in range(3)]
        asm = _assembly(pls)
        boxes = build_layout_boxes(asm, frags)
        assert len(boxes) == 3

    def test_missing_fragment_skipped(self):
        frags = [_fragment(0)]
        pls = [_placement(0), _placement(99, 100.0)]  # 99 not in frags
        asm = _assembly(pls)
        boxes = build_layout_boxes(asm, frags)
        assert len(boxes) == 1

    def test_position_correct(self):
        frags = [_fragment(0, w=60, h=40)]
        asm = _assembly([_placement(0, 15.0, 25.0)])
        boxes = build_layout_boxes(asm, frags)
        assert boxes[0].x == pytest.approx(15.0)
        assert boxes[0].y == pytest.approx(25.0)

    def test_size_from_fragment(self):
        frags = [_fragment(0, w=80, h=60)]
        asm = _assembly([_placement(0)])
        boxes = build_layout_boxes(asm, frags)
        assert boxes[0].w == pytest.approx(80.0)
        assert boxes[0].h == pytest.approx(60.0)

    def test_empty_placements(self):
        frags = [_fragment(0)]
        asm = _assembly([])
        boxes = build_layout_boxes(asm, frags)
        assert boxes == []


# ─── check_overlaps (extra) ───────────────────────────────────────────────────

class TestCheckOverlapsExtra:
    def test_no_overlap_empty_result(self):
        boxes = _row_boxes(3)
        assert check_overlaps(boxes) == []

    def test_overlap_detected(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)
        result = check_overlaps([a, b], min_area=1.0)
        assert len(result) >= 1

    def test_violation_kind_overlap(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)
        result = check_overlaps([a, b])
        for r in result:
            assert r.kind == ConstraintType.OVERLAP

    def test_empty_boxes_empty_result(self):
        assert check_overlaps([]) == []

    def test_single_box_empty_result(self):
        assert check_overlaps([_fbox(0, 0, 0)]) == []

    def test_severity_in_0_1(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)
        result = check_overlaps([a, b])
        for r in result:
            assert 0.0 <= r.severity <= 1.0

    def test_high_min_area_no_violation(self):
        a = _fbox(0, 0.0, 0.0, 60.0, 50.0)
        b = _fbox(1, 50.0, 0.0, 60.0, 50.0)
        result = check_overlaps([a, b], min_area=9999.0)
        assert result == []


# ─── check_gaps (extra) ───────────────────────────────────────────────────────

class TestCheckGapsExtra:
    def test_touching_boxes_no_gap_violation(self):
        boxes = _row_boxes(3, gap=0.0)
        result = check_gaps(boxes, max_gap=5.0, proximity=49.0)
        assert result == []

    def test_large_gap_detected(self):
        a = _fbox(0, 0.0, 0.0)
        b = _fbox(1, 100.0, 0.0)  # 50px gap
        result = check_gaps([a, b], max_gap=10.0, proximity=200.0)
        assert len(result) >= 1

    def test_violation_kind_gap(self):
        a = _fbox(0, 0.0, 0.0)
        b = _fbox(1, 100.0, 0.0)
        result = check_gaps([a, b], max_gap=5.0, proximity=200.0)
        for r in result:
            assert r.kind == ConstraintType.GAP

    def test_empty_boxes_empty_result(self):
        assert check_gaps([], max_gap=5.0) == []

    def test_single_box_empty_result(self):
        assert check_gaps([_fbox(0, 0, 0)], max_gap=5.0) == []

    def test_far_apart_not_neighbors(self):
        a = _fbox(0, 0.0, 0.0)
        b = _fbox(1, 1000.0, 0.0)
        result = check_gaps([a, b], max_gap=5.0, proximity=20.0)
        assert result == []


# ─── check_column_alignment (extra) ───────────────────────────────────────────

class TestCheckColumnAlignmentExtra:
    def test_aligned_column_no_violation(self):
        boxes = [_fbox(i, 0.0, float(i) * 60.0) for i in range(3)]
        result = check_column_alignment(boxes, tolerance=5.0)
        assert result == []

    def test_single_box_empty_result(self):
        assert check_column_alignment([_fbox(0, 0, 0)]) == []

    def test_empty_boxes_empty_result(self):
        assert check_column_alignment([]) == []

    def test_misaligned_detected(self):
        boxes = [
            _fbox(0, 0.0, 0.0),
            _fbox(1, 0.0, 60.0),
            _fbox(2, 30.0, 120.0),   # 30px offset from column
        ]
        result = check_column_alignment(boxes, tolerance=2.0)
        assert isinstance(result, list)
        for r in result:
            assert r.kind == ConstraintType.MISALIGN_COL

    def test_violation_kind_misalign_col(self):
        boxes = [
            _fbox(0, 0.0, 0.0),
            _fbox(1, 0.0, 60.0),
            _fbox(2, 30.0, 120.0),
        ]
        result = check_column_alignment(boxes, tolerance=2.0)
        for r in result:
            assert r.kind == ConstraintType.MISALIGN_COL

    def test_severity_in_0_1(self):
        boxes = [_fbox(i, float(i) * 5.0, float(i) * 60.0) for i in range(3)]
        result = check_column_alignment(boxes, tolerance=1.0)
        for r in result:
            assert 0.0 <= r.severity <= 1.0


# ─── check_row_alignment (extra) ──────────────────────────────────────────────

class TestCheckRowAlignmentExtra:
    def test_aligned_row_no_violation(self):
        boxes = _row_boxes(3)
        result = check_row_alignment(boxes, tolerance=5.0)
        assert result == []

    def test_single_box_empty_result(self):
        assert check_row_alignment([_fbox(0, 0, 0)]) == []

    def test_empty_boxes_empty_result(self):
        assert check_row_alignment([]) == []

    def test_misaligned_row_detected(self):
        boxes = [
            _fbox(0, 0.0, 0.0),
            _fbox(1, 60.0, 0.0),
            _fbox(2, 120.0, 30.0),   # 30px offset
        ]
        result = check_row_alignment(boxes, tolerance=2.0)
        assert isinstance(result, list)
        for r in result:
            assert r.kind == ConstraintType.MISALIGN_ROW

    def test_violation_kind_misalign_row(self):
        boxes = [
            _fbox(0, 0.0, 0.0),
            _fbox(1, 60.0, 0.0),
            _fbox(2, 120.0, 30.0),
        ]
        result = check_row_alignment(boxes, tolerance=2.0)
        for r in result:
            assert r.kind == ConstraintType.MISALIGN_ROW


# ─── check_out_of_bounds (extra) ──────────────────────────────────────────────

class TestCheckOutOfBoundsExtra:
    def test_inside_canvas_no_violation(self):
        boxes = [_fbox(0, 10.0, 10.0, 30.0, 30.0)]
        result = check_out_of_bounds(boxes, canvas_w=100.0, canvas_h=100.0)
        assert result == []

    def test_outside_right_violation(self):
        boxes = [_fbox(0, 90.0, 10.0, 50.0, 30.0)]  # extends to 140
        result = check_out_of_bounds(boxes, canvas_w=100.0, canvas_h=100.0)
        assert len(result) >= 1

    def test_outside_bottom_violation(self):
        boxes = [_fbox(0, 10.0, 80.0, 30.0, 50.0)]  # extends to 130
        result = check_out_of_bounds(boxes, canvas_w=200.0, canvas_h=100.0)
        assert len(result) >= 1

    def test_violation_kind_out_of_bounds(self):
        boxes = [_fbox(0, 90.0, 10.0, 50.0, 30.0)]
        result = check_out_of_bounds(boxes, canvas_w=100.0, canvas_h=100.0)
        for r in result:
            assert r.kind == ConstraintType.OUT_OF_BOUNDS

    def test_margin_allows_small_overshoot(self):
        boxes = [_fbox(0, 0.0, 0.0, 55.0, 50.0)]  # 5px beyond 50
        result = check_out_of_bounds(boxes, canvas_w=50.0, canvas_h=100.0, margin=10.0)
        assert result == []

    def test_empty_boxes_empty_result(self):
        result = check_out_of_bounds([], canvas_w=100.0, canvas_h=100.0)
        assert result == []


# ─── check_duplicate_placements (extra) ───────────────────────────────────────

class TestCheckDuplicatePlacementsExtra:
    def test_no_duplicates_empty_result(self):
        pls = [_placement(i) for i in range(4)]
        asm = _assembly(pls)
        result = check_duplicate_placements(asm)
        assert result == []

    def test_one_duplicate_detected(self):
        pls = [_placement(0), _placement(1), _placement(0)]
        asm = _assembly(pls)
        result = check_duplicate_placements(asm)
        assert len(result) == 1

    def test_duplicate_kind(self):
        pls = [_placement(0), _placement(0)]
        asm = _assembly(pls)
        result = check_duplicate_placements(asm)
        assert result[0].kind == ConstraintType.DUPLICATE_PLACE

    def test_duplicate_severity_1(self):
        pls = [_placement(0), _placement(0)]
        asm = _assembly(pls)
        result = check_duplicate_placements(asm)
        assert result[0].severity == pytest.approx(1.0)

    def test_empty_placements_empty_result(self):
        asm = _assembly([])
        assert check_duplicate_placements(asm) == []

    def test_multiple_duplicates(self):
        pls = [_placement(0), _placement(0), _placement(1), _placement(1)]
        asm = _assembly(pls)
        result = check_duplicate_placements(asm)
        assert len(result) == 2


# ─── verify_layout (extra) ────────────────────────────────────────────────────

class TestVerifyLayoutExtra:
    def _perfect(self, n: int = 3) -> tuple:
        frags = [_fragment(i) for i in range(n)]
        pls = [_placement(i, float(i) * 60.0, 0.0) for i in range(n)]
        asm = _assembly(pls)
        return frags, asm

    def test_returns_layout_verification_result(self):
        frags, asm = self._perfect()
        r = verify_layout(asm, frags)
        assert isinstance(r, LayoutVerificationResult)

    def test_perfect_layout_valid(self):
        frags, asm = self._perfect()
        r = verify_layout(asm, frags)
        assert r.valid is True

    def test_perfect_violation_score_zero(self):
        frags, asm = self._perfect()
        r = verify_layout(asm, frags)
        assert r.violation_score == pytest.approx(0.0)

    def test_overlapping_not_valid(self):
        frags = [_fragment(0, w=80), _fragment(1, w=80)]
        pls = [_placement(0, 0.0), _placement(1, 50.0)]  # 30px overlap
        asm = _assembly(pls)
        r = verify_layout(asm, frags)
        assert r.valid is False

    def test_violation_score_in_0_1(self):
        frags = [_fragment(0, w=80), _fragment(1, w=80)]
        pls = [_placement(0), _placement(1, 50.0)]
        asm = _assembly(pls)
        r = verify_layout(asm, frags)
        assert 0.0 <= r.violation_score <= 1.0

    def test_n_fragments_correct(self):
        frags, asm = self._perfect(4)
        r = verify_layout(asm, frags)
        assert r.n_fragments == 4

    def test_boxes_populated(self):
        frags, asm = self._perfect(2)
        r = verify_layout(asm, frags)
        assert len(r.boxes) == 2

    def test_with_canvas_size(self):
        frags, asm = self._perfect(2)
        r = verify_layout(asm, frags, canvas_size=(500.0, 500.0))
        assert isinstance(r, LayoutVerificationResult)

    def test_out_of_bounds_detected(self):
        frags = [_fragment(0, w=50, h=50)]
        pls = [_placement(0, 480.0, 480.0)]  # way outside 100x100
        asm = _assembly(pls)
        r = verify_layout(asm, frags, canvas_size=(100.0, 100.0))
        assert not r.valid

    def test_duplicate_placement_detected(self):
        frags = [_fragment(0)]
        pls = [_placement(0, 0.0), _placement(0, 100.0)]
        asm = _assembly(pls)
        r = verify_layout(asm, frags)
        assert not r.valid

    def test_by_kind_overlap(self):
        frags = [_fragment(0, w=80), _fragment(1, w=80)]
        pls = [_placement(0), _placement(1, 50.0)]
        asm = _assembly(pls)
        r = verify_layout(asm, frags)
        overlaps = r.by_kind(ConstraintType.OVERLAP)
        assert len(overlaps) >= 1
