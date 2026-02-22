"""
Тесты для puzzle_reconstruction/verification/layout_verifier.py

Покрытие:
    ConstraintType   — все 6 значений, строковое представление
    LayoutConstraint — repr, severity ∈ [0,1]
    FragmentBox      — x2/y2/cx/cy, intersects (overlap/касание/нет),
                       overlap_area (нет/частичное/полное), gap_to,
                       are_neighbors, repr
    build_layout_boxes — корректность x/y/w/h, нет Fragment → пропуск,
                          пустая сборка
    check_overlaps   — нет перекрытий, частичное, полное (severity=1),
                       min_area фильтр
    check_gaps       — нет соседей, близкий сосед с малым зазором → OK,
                       близкий сосед с большим зазором → нарушение
    check_column_alignment — выровненные, один сдвинутый → нарушение,
                             единственный элемент → пусто
    check_row_alignment    — аналогично по Y
    check_out_of_bounds    — внутри → пусто, вылет → нарушение,
                             margin позволяет небольшой вылет
    check_duplicate_placements — нет дубликатов, один дубликат → нарушение
    verify_layout    — идеальное расположение, перекрытие → violation_score>0,
                       valid=True/False, n_fragments, boxes
    LayoutVerificationResult — by_kind, summary, valid
"""
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


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _fragment(fid: int, w: int = 50, h: int = 50) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=np.zeros((h, w, 3), dtype=np.uint8),
        edges=[],
        bounding_box=(0, 0, w, h),
    )


def _placement(fid: int, x: float, y: float, rot: float = 0.0) -> Placement:
    return Placement(fragment_id=fid, position=(x, y), rotation=rot)


def _box(fid: int, x: float, y: float, w: float = 50.0,
          h: float = 50.0) -> FragmentBox:
    return FragmentBox(fid=fid, x=x, y=y, w=w, h=h)


def _assembly(placements):
    return Assembly(placements=placements, total_score=0.8, method="test")


# ─── ConstraintType ───────────────────────────────────────────────────────────

class TestConstraintType:
    def test_all_six_values(self):
        assert len(ConstraintType) == 6

    def test_string_values(self):
        assert ConstraintType.OVERLAP.value         == "overlap"
        assert ConstraintType.GAP.value             == "gap"
        assert ConstraintType.MISALIGN_COL.value    == "misalign_column"
        assert ConstraintType.MISALIGN_ROW.value    == "misalign_row"
        assert ConstraintType.OUT_OF_BOUNDS.value   == "out_of_bounds"
        assert ConstraintType.DUPLICATE_PLACE.value == "duplicate_place"


# ─── LayoutConstraint ─────────────────────────────────────────────────────────

class TestLayoutConstraint:
    def test_repr_contains_kind(self):
        lc = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1),
                               severity=0.4)
        assert "overlap" in repr(lc)

    def test_repr_contains_fids(self):
        lc = LayoutConstraint(kind=ConstraintType.GAP, fids=(2, 3),
                               severity=0.3)
        assert "2" in repr(lc) and "3" in repr(lc)

    def test_severity_default(self):
        lc = LayoutConstraint(kind=ConstraintType.GAP, fids=(0,))
        assert 0.0 <= lc.severity <= 1.0

    def test_detail_stored(self):
        lc = LayoutConstraint(kind=ConstraintType.OVERLAP, fids=(0, 1),
                               detail="area=25")
        assert "25" in lc.detail


# ─── FragmentBox ──────────────────────────────────────────────────────────────

class TestFragmentBox:
    def test_x2(self):
        b = _box(0, 10.0, 20.0, 50.0, 30.0)
        assert b.x2 == pytest.approx(60.0)

    def test_y2(self):
        b = _box(0, 10.0, 20.0, 50.0, 30.0)
        assert b.y2 == pytest.approx(50.0)

    def test_cx(self):
        b = _box(0, 0.0, 0.0, 100.0, 60.0)
        assert b.cx == pytest.approx(50.0)

    def test_cy(self):
        b = _box(0, 0.0, 0.0, 100.0, 60.0)
        assert b.cy == pytest.approx(30.0)

    def test_intersects_overlapping(self):
        a = _box(0, 0.0,  0.0)   # [0,50]×[0,50]
        b = _box(1, 25.0, 25.0)  # [25,75]×[25,75]
        assert a.intersects(b)

    def test_intersects_touching_no_overlap(self):
        """Касание (только граница) = нет пересечения."""
        a = _box(0, 0.0,  0.0)   # [0,50]×[0,50]
        b = _box(1, 50.0, 0.0)   # [50,100]×[0,50]
        assert not a.intersects(b)

    def test_intersects_no_contact(self):
        a = _box(0, 0.0,  0.0)
        b = _box(1, 100.0, 0.0)
        assert not a.intersects(b)

    def test_overlap_area_none(self):
        a = _box(0, 0.0, 0.0)
        b = _box(1, 60.0, 0.0)
        assert a.overlap_area(b) == pytest.approx(0.0)

    def test_overlap_area_partial(self):
        a = _box(0, 0.0, 0.0, 50.0, 50.0)
        b = _box(1, 25.0, 0.0, 50.0, 50.0)
        area = a.overlap_area(b)
        assert area == pytest.approx(25.0 * 50.0)

    def test_overlap_area_full_containment(self):
        """a полностью содержит b."""
        a = _box(0, 0.0,  0.0,  100.0, 100.0)
        b = _box(1, 10.0, 10.0,  30.0,  30.0)
        area = a.overlap_area(b)
        assert area == pytest.approx(30.0 * 30.0)

    def test_gap_to_no_contact(self):
        a = _box(0, 0.0, 0.0, 50.0, 50.0)
        b = _box(1, 70.0, 0.0, 50.0, 50.0)
        assert a.gap_to(b) > 0

    def test_gap_to_touching(self):
        a = _box(0, 0.0, 0.0, 50.0, 50.0)
        b = _box(1, 50.0, 0.0, 50.0, 50.0)
        assert a.gap_to(b) == pytest.approx(0.0)

    def test_gap_to_overlapping_negative(self):
        a = _box(0, 0.0, 0.0, 50.0, 50.0)
        b = _box(1, 25.0, 0.0, 50.0, 50.0)
        assert a.gap_to(b) < 0.0

    def test_are_neighbors_close(self):
        a = _box(0, 0.0, 0.0, 50.0, 50.0)
        b = _box(1, 55.0, 0.0, 50.0, 50.0)  # зазор = 5
        assert a.are_neighbors(b, proximity=10.0)

    def test_are_neighbors_far(self):
        a = _box(0, 0.0,  0.0)
        b = _box(1, 200.0, 0.0)
        assert not a.are_neighbors(b, proximity=10.0)

    def test_repr_contains_fid(self):
        b = _box(42, 0.0, 0.0)
        assert "42" in repr(b)


# ─── build_layout_boxes ───────────────────────────────────────────────────────

class TestBuildLayoutBoxes:
    def test_correct_position(self):
        frags = [_fragment(0)]
        asm   = _assembly([_placement(0, 30.0, 40.0)])
        boxes = build_layout_boxes(asm, frags)
        assert boxes[0].x == pytest.approx(30.0)
        assert boxes[0].y == pytest.approx(40.0)

    def test_correct_size(self):
        frags = [_fragment(0, w=80, h=60)]
        asm   = _assembly([_placement(0, 0.0, 0.0)])
        boxes = build_layout_boxes(asm, frags)
        assert boxes[0].w == pytest.approx(80.0)
        assert boxes[0].h == pytest.approx(60.0)

    def test_missing_fragment_skipped(self):
        frags = [_fragment(0)]
        asm   = _assembly([_placement(0, 0.0, 0.0), _placement(99, 50.0, 0.0)])
        boxes = build_layout_boxes(asm, frags)
        assert len(boxes) == 1
        assert boxes[0].fid == 0

    def test_empty_assembly(self):
        asm   = _assembly([])
        boxes = build_layout_boxes(asm, [])
        assert boxes == []

    def test_rotation_stored(self):
        frags = [_fragment(0)]
        asm   = _assembly([_placement(0, 0.0, 0.0, rot=45.0)])
        boxes = build_layout_boxes(asm, frags)
        assert boxes[0].rotation == pytest.approx(45.0)

    def test_multiple_fragments(self):
        frags = [_fragment(i) for i in range(3)]
        asm   = _assembly([_placement(i, i * 60.0, 0.0) for i in range(3)])
        boxes = build_layout_boxes(asm, frags)
        assert len(boxes) == 3


# ─── check_overlaps ───────────────────────────────────────────────────────────

class TestCheckOverlaps:
    def test_no_overlap(self):
        boxes = [_box(0, 0.0, 0.0), _box(1, 60.0, 0.0)]
        assert check_overlaps(boxes) == []

    def test_partial_overlap(self):
        boxes = [_box(0, 0.0, 0.0), _box(1, 25.0, 0.0)]
        c     = check_overlaps(boxes, min_area=1.0)
        assert len(c) == 1
        assert c[0].kind == ConstraintType.OVERLAP

    def test_full_containment_severity_one(self):
        """Полное перекрытие → severity ≈ 1."""
        a = _box(0, 0.0,  0.0, 100.0, 100.0)
        b = _box(1, 10.0, 10.0, 30.0,  30.0)
        c = check_overlaps([a, b], min_area=1.0)
        assert len(c) == 1
        assert c[0].severity == pytest.approx(1.0)

    def test_min_area_filter(self):
        """Малое перекрытие игнорируется при большом min_area."""
        boxes = [_box(0, 0.0, 0.0), _box(1, 49.0, 0.0)]
        c1    = check_overlaps(boxes, min_area=1.0)
        c2    = check_overlaps(boxes, min_area=10000.0)
        assert len(c1) >= 1
        assert len(c2) == 0

    def test_fids_correct(self):
        boxes = [_box(5, 0.0, 0.0), _box(7, 25.0, 0.0)]
        c     = check_overlaps(boxes)
        assert set(c[0].fids) == {5, 7}

    def test_single_box_no_violations(self):
        assert check_overlaps([_box(0, 0.0, 0.0)]) == []

    def test_empty_boxes_no_violations(self):
        assert check_overlaps([]) == []


# ─── check_gaps ───────────────────────────────────────────────────────────────

class TestCheckGaps:
    def test_no_gap_touching(self):
        """Вплотную стоящие фрагменты — нет нарушения."""
        boxes = [_box(0, 0.0, 0.0), _box(1, 50.0, 0.0)]
        c     = check_gaps(boxes, max_gap=15.0, proximity=60.0)
        assert c == []

    def test_small_gap_no_violation(self):
        """Зазор меньше max_gap — нет нарушения."""
        boxes = [_box(0, 0.0, 0.0), _box(1, 55.0, 0.0)]  # зазор=5
        c     = check_gaps(boxes, max_gap=15.0, proximity=60.0)
        assert c == []

    def test_large_gap_violation(self):
        """Зазор > max_gap → нарушение."""
        boxes = [_box(0, 0.0, 0.0), _box(1, 80.0, 0.0)]  # зазор=30
        c     = check_gaps(boxes, max_gap=15.0, proximity=90.0)
        assert len(c) >= 1
        assert c[0].kind == ConstraintType.GAP

    def test_far_boxes_not_neighbors(self):
        """Очень далёкие коробки не считаются соседями."""
        boxes = [_box(0, 0.0, 0.0), _box(1, 500.0, 0.0)]
        c     = check_gaps(boxes, max_gap=5.0, proximity=50.0)
        assert c == []

    def test_single_box_no_violations(self):
        assert check_gaps([_box(0, 0.0, 0.0)]) == []


# ─── check_column_alignment ───────────────────────────────────────────────────

class TestCheckColumnAlignment:
    def test_aligned_column_no_violation(self):
        """Коробки с одинаковым cx — нет нарушения."""
        boxes = [_box(i, 0.0, i * 60.0) for i in range(3)]
        c     = check_column_alignment(boxes, tolerance=5.0)
        assert c == []

    def test_misaligned_column_violation(self):
        """Один фрагмент сдвинут на 30px по X → нарушение."""
        boxes = [
            _box(0, 0.0,   0.0),
            _box(1, 0.0,  60.0),
            _box(2, 30.0, 120.0),  # cx = 55, остальные cx = 25
        ]
        c = check_column_alignment(boxes, tolerance=5.0)
        assert len(c) >= 1
        assert any(cc.kind == ConstraintType.MISALIGN_COL for cc in c)

    def test_single_box_no_violation(self):
        assert check_column_alignment([_box(0, 0.0, 0.0)]) == []

    def test_empty_boxes_no_violation(self):
        assert check_column_alignment([]) == []

    def test_severity_in_range(self):
        boxes = [
            _box(0, 0.0,   0.0),
            _box(1, 0.0,  60.0),
            _box(2, 50.0, 120.0),
        ]
        c = check_column_alignment(boxes, tolerance=5.0)
        for cc in c:
            assert 0.0 <= cc.severity <= 1.0


# ─── check_row_alignment ──────────────────────────────────────────────────────

class TestCheckRowAlignment:
    def test_aligned_row_no_violation(self):
        boxes = [_box(i, i * 60.0, 0.0) for i in range(3)]
        c     = check_row_alignment(boxes, tolerance=5.0)
        assert c == []

    def test_misaligned_row_violation(self):
        boxes = [
            _box(0, 0.0,   0.0),
            _box(1, 60.0,  0.0),
            _box(2, 120.0, 30.0),  # cy = 55, остальные cy = 25
        ]
        c = check_row_alignment(boxes, tolerance=5.0)
        assert len(c) >= 1
        assert any(cc.kind == ConstraintType.MISALIGN_ROW for cc in c)

    def test_single_box_no_violation(self):
        assert check_row_alignment([_box(0, 0.0, 0.0)]) == []

    def test_empty_boxes_no_violation(self):
        assert check_row_alignment([]) == []


# ─── check_out_of_bounds ──────────────────────────────────────────────────────

class TestCheckOutOfBounds:
    def test_inside_no_violation(self):
        boxes = [_box(0, 10.0, 10.0, 50.0, 50.0)]
        c     = check_out_of_bounds(boxes, canvas_w=200.0, canvas_h=200.0)
        assert c == []

    def test_right_overshoot_violation(self):
        boxes = [_box(0, 170.0, 10.0, 50.0, 50.0)]  # x2=220 > 200
        c     = check_out_of_bounds(boxes, canvas_w=200.0, canvas_h=200.0)
        assert len(c) == 1
        assert c[0].kind == ConstraintType.OUT_OF_BOUNDS

    def test_top_overshoot_violation(self):
        boxes = [_box(0, 0.0, -10.0, 50.0, 50.0)]   # y=-10 < 0
        c     = check_out_of_bounds(boxes, canvas_w=200.0, canvas_h=200.0)
        assert len(c) == 1

    def test_margin_allows_small_overshoot(self):
        boxes = [_box(0, 195.0, 0.0, 10.0, 50.0)]   # x2=205, вылет=5
        c_no_margin  = check_out_of_bounds(boxes, 200.0, 200.0, margin=0.0)
        c_with_margin = check_out_of_bounds(boxes, 200.0, 200.0, margin=10.0)
        assert len(c_no_margin)   >= 1
        assert len(c_with_margin) == 0

    def test_fid_in_violation(self):
        boxes = [_box(7, 190.0, 10.0, 50.0, 50.0)]
        c     = check_out_of_bounds(boxes, 200.0, 200.0)
        assert c[0].fids == (7,)


# ─── check_duplicate_placements ───────────────────────────────────────────────

class TestCheckDuplicatePlacements:
    def test_no_duplicates_empty(self):
        asm = _assembly([_placement(0, 0, 0), _placement(1, 60, 0)])
        assert check_duplicate_placements(asm) == []

    def test_one_duplicate(self):
        asm = _assembly([
            _placement(0, 0, 0),
            _placement(0, 60, 0),  # дубликат fid=0
            _placement(1, 120, 0),
        ])
        c = check_duplicate_placements(asm)
        assert len(c) == 1
        assert c[0].kind == ConstraintType.DUPLICATE_PLACE
        assert c[0].severity == pytest.approx(1.0)
        assert 0 in c[0].fids

    def test_two_duplicates(self):
        asm = _assembly([
            _placement(0, 0,   0),
            _placement(0, 60,  0),
            _placement(1, 120, 0),
            _placement(1, 180, 0),
        ])
        c = check_duplicate_placements(asm)
        assert len(c) == 2


# ─── verify_layout ────────────────────────────────────────────────────────────

class TestVerifyLayout:
    def _perfect_layout(self):
        """3 фрагмента в ряд без перекрытий."""
        frags = [_fragment(i) for i in range(3)]
        asm   = _assembly([_placement(i, i * 60.0, 0.0) for i in range(3)])
        return frags, asm

    def test_perfect_layout_valid(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags, max_gap=20.0, proximity=70.0)
        assert r.valid

    def test_perfect_layout_no_constraints(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags, max_gap=20.0, proximity=70.0)
        assert len(r.constraints) == 0

    def test_perfect_layout_violation_score_zero(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags, max_gap=20.0, proximity=70.0)
        assert r.violation_score == pytest.approx(0.0)

    def test_overlap_makes_invalid(self):
        frags = [_fragment(0), _fragment(1)]
        asm   = _assembly([_placement(0, 0.0, 0.0), _placement(1, 10.0, 0.0)])
        r     = verify_layout(asm, frags)
        assert not r.valid
        assert r.violation_score > 0.0

    def test_n_fragments_correct(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags)
        assert r.n_fragments == 3

    def test_boxes_populated(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags)
        assert len(r.boxes) == 3

    def test_canvas_size_out_of_bounds(self):
        frags = [_fragment(0)]
        asm   = _assembly([_placement(0, 180.0, 0.0)])
        r     = verify_layout(asm, frags, canvas_size=(200.0, 200.0))
        # fid=0: x=180, w=50 → x2=230 > 200 → out_of_bounds
        assert not r.valid

    def test_canvas_size_none_no_bounds_check(self):
        frags = [_fragment(0)]
        asm   = _assembly([_placement(0, 1000.0, 0.0)])
        r     = verify_layout(asm, frags, canvas_size=None)
        oob = r.by_kind(ConstraintType.OUT_OF_BOUNDS)
        assert len(oob) == 0

    def test_by_kind(self):
        frags = [_fragment(0), _fragment(1)]
        asm   = _assembly([_placement(0, 0.0, 0.0), _placement(1, 10.0, 0.0)])
        r     = verify_layout(asm, frags)
        overlaps = r.by_kind(ConstraintType.OVERLAP)
        assert len(overlaps) >= 1

    def test_summary_contains_pass_or_fail(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags, max_gap=200.0, proximity=500.0)
        s = r.summary()
        assert "PASS" in s or "FAIL" in s

    def test_summary_contains_n_fragments(self):
        frags, asm = self._perfect_layout()
        r = verify_layout(asm, frags)
        assert "3" in r.summary()

    def test_duplicate_detected(self):
        frags = [_fragment(0)]
        asm   = _assembly([_placement(0, 0.0, 0.0), _placement(0, 60.0, 0.0)])
        r     = verify_layout(asm, frags)
        dups = r.by_kind(ConstraintType.DUPLICATE_PLACE)
        assert len(dups) >= 1

    def test_violation_score_in_range(self):
        frags = [_fragment(0), _fragment(1)]
        asm   = _assembly([_placement(0, 0.0, 0.0), _placement(1, 5.0, 0.0)])
        r     = verify_layout(asm, frags)
        assert 0.0 <= r.violation_score <= 1.0
