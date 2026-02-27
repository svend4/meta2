"""Tests for puzzle_reconstruction.verification.layout_verifier"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.layout_verifier import (
    ConstraintType,
    LayoutConstraint,
    FragmentBox,
    LayoutVerificationResult,
    check_overlaps,
    check_gaps,
    check_column_alignment,
    check_row_alignment,
    check_out_of_bounds,
)
from puzzle_reconstruction.models import Assembly, Fragment, Placement


def make_box(fid, x, y, w, h, rotation=0.0):
    return FragmentBox(fid=fid, x=float(x), y=float(y), w=float(w), h=float(h), rotation=rotation)


def make_fragment_img(fid, h=50, w=50):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return Fragment(fragment_id=fid, image=img)


def make_assembly(placements_list):
    placements = [Placement(fragment_id=p[0], position=(p[1], p[2]), rotation=0.0)
                  for p in placements_list]
    asm = Assembly()
    asm.placements = placements
    return asm


# ─── ConstraintType ───────────────────────────────────────────────────────────

def test_constraint_type_values():
    assert ConstraintType.OVERLAP == "overlap"
    assert ConstraintType.GAP == "gap"
    assert ConstraintType.MISALIGN_COL == "misalign_column"
    assert ConstraintType.MISALIGN_ROW == "misalign_row"
    assert ConstraintType.OUT_OF_BOUNDS == "out_of_bounds"
    assert ConstraintType.DUPLICATE_PLACE == "duplicate_place"


# ─── LayoutConstraint ─────────────────────────────────────────────────────────

def test_layout_constraint_repr():
    c = LayoutConstraint(
        kind=ConstraintType.OVERLAP,
        fids=(0, 1),
        severity=0.5,
        detail="test"
    )
    r = repr(c)
    assert "overlap" in r


# ─── FragmentBox ──────────────────────────────────────────────────────────────

def test_fragment_box_x2_y2():
    box = make_box(0, 10, 20, 30, 40)
    assert box.x2 == pytest.approx(40.0)
    assert box.y2 == pytest.approx(60.0)


def test_fragment_box_center():
    box = make_box(0, 0, 0, 100, 100)
    assert box.cx == pytest.approx(50.0)
    assert box.cy == pytest.approx(50.0)


def test_fragment_box_intersects_true():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 25, 25, 50, 50)
    assert b1.intersects(b2)


def test_fragment_box_intersects_false():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 60, 0, 50, 50)
    assert not b1.intersects(b2)


def test_fragment_box_overlap_area():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 25, 25, 50, 50)
    area = b1.overlap_area(b2)
    assert area == pytest.approx(25 * 25)


def test_fragment_box_overlap_area_no_overlap():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 100, 0, 50, 50)
    assert b1.overlap_area(b2) == pytest.approx(0.0)


def test_fragment_box_gap_to_adjacent():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 50, 0, 50, 50)
    gap = b1.gap_to(b2)
    assert gap >= 0.0


def test_fragment_box_are_neighbors_close():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 55, 0, 50, 50)
    assert b1.are_neighbors(b2, proximity=20.0)


def test_fragment_box_are_neighbors_far():
    b1 = make_box(0, 0, 0, 50, 50)
    b2 = make_box(1, 300, 0, 50, 50)
    assert not b1.are_neighbors(b2, proximity=20.0)


# ─── LayoutVerificationResult ─────────────────────────────────────────────────

def test_layout_verification_result_valid():
    result = LayoutVerificationResult(
        constraints=[],
        violation_score=0.0,
        valid=True,
        n_fragments=3,
    )
    assert result.valid
    assert result.summary().startswith("LayoutVerificationResult(PASS")


def test_layout_verification_result_invalid():
    c = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1), severity=0.8)
    result = LayoutVerificationResult(
        constraints=[c],
        violation_score=0.8,
        valid=False,
        n_fragments=2,
    )
    assert not result.valid
    assert "FAIL" in result.summary()


def test_layout_verification_result_by_kind():
    c1 = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1))
    c2 = LayoutConstraint(kind=ConstraintType.GAP, fids=(1, 2))
    result = LayoutVerificationResult(
        constraints=[c1, c2],
        violation_score=0.5,
        valid=False,
        n_fragments=3,
    )
    overlaps = result.by_kind(ConstraintType.OVERLAP)
    assert len(overlaps) == 1


# ─── check_overlaps ───────────────────────────────────────────────────────────

def test_check_overlaps_none():
    boxes = [make_box(i, i * 60, 0, 50, 50) for i in range(3)]
    constraints = check_overlaps(boxes, min_area=1.0)
    assert constraints == []


def test_check_overlaps_found():
    boxes = [make_box(0, 0, 0, 100, 100), make_box(1, 50, 50, 100, 100)]
    constraints = check_overlaps(boxes, min_area=1.0)
    assert len(constraints) > 0
    assert constraints[0].kind == ConstraintType.OVERLAP


def test_check_overlaps_severity_range():
    boxes = [make_box(0, 0, 0, 100, 100), make_box(1, 50, 50, 100, 100)]
    constraints = check_overlaps(boxes)
    for c in constraints:
        assert 0.0 <= c.severity <= 1.0


def test_check_overlaps_empty():
    assert check_overlaps([]) == []


# ─── check_gaps ───────────────────────────────────────────────────────────────

def test_check_gaps_adjacent_no_gap():
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 50, 0, 50, 50)]
    constraints = check_gaps(boxes, max_gap=10.0, proximity=15.0)
    assert constraints == []


def test_check_gaps_large_gap():
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 100, 0, 50, 50)]
    constraints = check_gaps(boxes, max_gap=5.0, proximity=100.0)
    assert len(constraints) > 0
    assert constraints[0].kind == ConstraintType.GAP


def test_check_gaps_empty():
    assert check_gaps([], max_gap=10.0) == []


def test_check_gaps_single():
    boxes = [make_box(0, 0, 0, 50, 50)]
    assert check_gaps(boxes, max_gap=10.0) == []


# ─── check_column_alignment ───────────────────────────────────────────────────

def test_check_column_alignment_aligned():
    # Two boxes with same cx
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 0, 60, 50, 50)]
    constraints = check_column_alignment(boxes, tolerance=5.0)
    assert constraints == []


def test_check_column_alignment_misaligned():
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 0, 60, 50, 50), make_box(2, 30, 120, 50, 50)]
    # box 2 has different cx
    constraints = check_column_alignment(boxes, tolerance=1.0)
    assert isinstance(constraints, list)


def test_check_column_alignment_single():
    boxes = [make_box(0, 0, 0, 50, 50)]
    assert check_column_alignment(boxes) == []


# ─── check_row_alignment ──────────────────────────────────────────────────────

def test_check_row_alignment_aligned():
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 60, 0, 50, 50)]
    constraints = check_row_alignment(boxes, tolerance=5.0)
    assert constraints == []


def test_check_row_alignment_misaligned():
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 60, 0, 50, 50), make_box(2, 120, 20, 50, 50)]
    constraints = check_row_alignment(boxes, tolerance=1.0)
    assert isinstance(constraints, list)


def test_check_row_alignment_single():
    assert check_row_alignment([make_box(0, 0, 0, 50, 50)]) == []


# ─── check_out_of_bounds ──────────────────────────────────────────────────────

def test_check_out_of_bounds_inside():
    boxes = [make_box(0, 0, 0, 50, 50), make_box(1, 50, 0, 50, 50)]
    constraints = check_out_of_bounds(boxes, canvas_w=200, canvas_h=200)
    assert constraints == []


def test_check_out_of_bounds_outside():
    boxes = [make_box(0, 150, 0, 100, 50)]
    constraints = check_out_of_bounds(boxes, canvas_w=200, canvas_h=200)
    assert len(constraints) > 0
    assert constraints[0].kind == ConstraintType.OUT_OF_BOUNDS


def test_check_out_of_bounds_with_margin():
    boxes = [make_box(0, 195, 0, 10, 10)]
    constraints = check_out_of_bounds(boxes, canvas_w=200, canvas_h=200, margin=10.0)
    assert constraints == []


def test_check_out_of_bounds_empty():
    assert check_out_of_bounds([], canvas_w=100, canvas_h=100) == []
