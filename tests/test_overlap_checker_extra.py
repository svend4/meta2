"""Extra tests for puzzle_reconstruction/verification/overlap_checker.py."""
import numpy as np
import pytest

from puzzle_reconstruction.verification.overlap_checker import (
    OverlapResult,
    check_all_overlaps,
    check_overlap_pair,
    find_conflicting_pairs,
    polygon_intersection_area,
    polygon_iou,
    polygon_union_area,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rect(x, y, w, h):
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


_A = _rect(0, 0, 30, 30)
_B = _rect(50, 50, 30, 30)
_P1 = _rect(0, 0, 30, 30)
_P2 = _rect(20, 20, 30, 30)   # 10×10 overlap
_SAME = _rect(10, 10, 40, 40)
_UNIT = _rect(0, 0, 5, 5)


# ─── TestOverlapResultExtra ───────────────────────────────────────────────────

class TestOverlapResultExtra:
    def _make(self, **kw):
        defaults = dict(idx1=2, idx2=5, intersection_area=50.0,
                        iou=0.25, has_overlap=True)
        defaults.update(kw)
        return OverlapResult(**defaults)

    def test_idx_values_stored(self):
        r = self._make(idx1=7, idx2=11)
        assert r.idx1 == 7
        assert r.idx2 == 11

    def test_large_intersection_area(self):
        r = self._make(intersection_area=10000.0)
        assert r.intersection_area == pytest.approx(10000.0)

    def test_iou_zero_stored(self):
        r = self._make(iou=0.0, has_overlap=False)
        assert r.iou == pytest.approx(0.0)

    def test_iou_one_stored(self):
        r = self._make(iou=1.0, has_overlap=True)
        assert r.iou == pytest.approx(1.0)

    def test_params_multiple_keys(self):
        r = self._make(params={"iou_thresh": 0.1, "method": "polygon", "idx": 3})
        assert r.params["idx"] == 3

    def test_has_overlap_true_stored(self):
        r = self._make(has_overlap=True)
        assert r.has_overlap is True

    def test_repr_is_string(self):
        assert isinstance(repr(self._make()), str)


# ─── TestPolygonIntersectionAreaExtra ────────────────────────────────────────

class TestPolygonIntersectionAreaExtra:
    def test_known_10x10_overlap(self):
        # P1=[0,0,30,30], P2=[20,20,30,30] → overlap region is ~10x10
        area = polygon_intersection_area(_P1, _P2)
        assert 80.0 <= area <= 150.0

    def test_touching_edge_small_area(self):
        left = _rect(0, 0, 20, 20)
        right = _rect(20, 0, 20, 20)  # shares edge only
        area = polygon_intersection_area(left, right)
        # Intersection at shared edge is much smaller than either polygon
        assert area < 0.15 * 400.0  # less than 15% of 20x20=400

    def test_large_polys(self):
        a = _rect(0, 0, 500, 500)
        b = _rect(250, 250, 500, 500)
        area = polygon_intersection_area(a, b)
        assert area == pytest.approx(62500.0, rel=0.05)

    def test_unit_poly_with_itself(self):
        area = polygon_intersection_area(_UNIT, _UNIT)
        assert area > 0.0

    def test_three_different_pairs(self):
        for p1, p2 in [(_A, _A), (_SAME, _SAME), (_P1, _P2)]:
            area = polygon_intersection_area(p1, p2)
            assert area >= 0.0


# ─── TestPolygonUnionAreaExtra ────────────────────────────────────────────────

class TestPolygonUnionAreaExtra:
    def test_known_union(self):
        # Two non-overlapping 30x30 squares → union ≈ twice the single area
        single = polygon_union_area(_A, _A)
        union = polygon_union_area(_A, _B)
        assert union == pytest.approx(2 * single, rel=0.05)

    def test_unit_with_itself(self):
        union = polygon_union_area(_UNIT, _UNIT)
        assert union > 0.0

    def test_partial_union_in_range(self):
        union = polygon_union_area(_P1, _P2)
        single = polygon_union_area(_P1, _P1)
        # union should be at least as large as one polygon
        assert union >= single

    def test_large_poly_with_itself(self):
        p = _rect(0, 0, 100, 100)
        union = polygon_union_area(p, p)
        assert union == pytest.approx(10000.0, rel=0.05)

    def test_union_gt_each_for_partial(self):
        u = polygon_union_area(_P1, _P2)
        a1 = polygon_union_area(_P1, _P1)
        a2 = polygon_union_area(_P2, _P2)
        assert u > max(a1, a2)


# ─── TestPolygonIouExtra ─────────────────────────────────────────────────────

class TestPolygonIouExtra:
    def test_partial_overlap_positive_iou(self):
        iou = polygon_iou(_P1, _P2)
        assert 0.0 < iou < 1.0

    def test_large_fully_inside(self):
        outer = _rect(0, 0, 100, 100)
        inner = _rect(25, 25, 50, 50)
        iou = polygon_iou(inner, outer)
        # IoU = 2500 / 10000 = 0.25
        assert iou == pytest.approx(0.25, rel=0.1)

    def test_same_rect_iou_one(self):
        p = _rect(5, 5, 20, 20)
        assert polygon_iou(p, p) == pytest.approx(1.0, rel=0.05)

    def test_symmetry_various(self):
        for p1, p2 in [(_P1, _P2), (_A, _SAME)]:
            assert polygon_iou(p1, p2) == pytest.approx(
                polygon_iou(p2, p1), rel=0.05
            )

    def test_very_small_overlap_low_iou(self):
        a = _rect(0, 0, 100, 100)
        b = _rect(99, 99, 100, 100)  # 1x1 overlap
        iou = polygon_iou(a, b)
        assert iou < 0.01


# ─── TestCheckOverlapPairExtra ────────────────────────────────────────────────

class TestCheckOverlapPairExtra:
    def test_large_polys_overlapping(self):
        a = _rect(0, 0, 200, 200)
        b = _rect(100, 100, 200, 200)
        r = check_overlap_pair(a, b, iou_thresh=0.05)
        assert r.has_overlap is True

    def test_thresh_0_partial_is_true(self):
        r = check_overlap_pair(_P1, _P2, iou_thresh=0.0)
        assert r.has_overlap is True

    def test_thresh_0_5_depends_on_iou(self):
        r = check_overlap_pair(_P1, _P2)
        iou = r.iou
        r2 = check_overlap_pair(_P1, _P2, iou_thresh=iou + 0.01)
        assert r2.has_overlap is False

    def test_identical_large_thresh_is_true(self):
        r = check_overlap_pair(_SAME, _SAME, iou_thresh=0.9)
        assert r.has_overlap is True

    def test_returns_overlap_result_type(self):
        r = check_overlap_pair(_A, _B)
        assert isinstance(r, OverlapResult)

    def test_iou_nonneg_various(self):
        for p1, p2 in [(_A, _B), (_P1, _P2), (_SAME, _SAME)]:
            r = check_overlap_pair(p1, p2)
            assert r.iou >= 0.0


# ─── TestCheckAllOverlapsExtra ───────────────────────────────────────────────

class TestCheckAllOverlapsExtra:
    def test_five_polys_ten_pairs(self):
        polys = [_rect(i * 40, 0, 30, 30) for i in range(5)]
        r = check_all_overlaps(polys)
        assert len(r) == 10

    def test_two_identical_has_overlap(self):
        r = check_all_overlaps([_SAME, _SAME], iou_thresh=0.1)
        assert r[0].has_overlap is True

    def test_all_non_overlapping_none_conflict(self):
        polys = [_rect(i * 50, 0, 30, 30) for i in range(4)]
        r = check_all_overlaps(polys, iou_thresh=0.05)
        assert all(not res.has_overlap for res in r)

    def test_idx1_always_less_than_idx2(self):
        polys = [_A, _B, _P1, _P2]
        for r in check_all_overlaps(polys):
            assert r.idx1 < r.idx2

    def test_results_have_iou_in_range(self):
        polys = [_A, _B, _P1, _P2, _SAME]
        for r in check_all_overlaps(polys):
            assert 0.0 <= r.iou <= 1.0


# ─── TestFindConflictingPairsExtra ───────────────────────────────────────────

class TestFindConflictingPairsExtra:
    def test_multiple_conflicts(self):
        polys = [_SAME, _SAME, _SAME]  # All overlap
        results = check_all_overlaps(polys, iou_thresh=0.05)
        conflicts = find_conflicting_pairs(results)
        assert len(conflicts) == 3

    def test_thresh_filter(self):
        polys = [_P1, _P2]
        results = check_all_overlaps(polys, iou_thresh=0.01)
        iou = results[0].iou
        conflicts_high = find_conflicting_pairs(results, iou_thresh=iou + 0.01)
        conflicts_low = find_conflicting_pairs(results, iou_thresh=iou - 0.01)
        assert len(conflicts_high) == 0
        assert len(conflicts_low) == 1

    def test_return_all_when_thresh_zero(self):
        polys = [_P1, _P2]
        results = check_all_overlaps(polys, iou_thresh=0.01)
        conflicts = find_conflicting_pairs(results, iou_thresh=0.0)
        assert len(conflicts) == 1

    def test_non_overlap_not_in_conflicts(self):
        polys = [_rect(0, 0, 10, 10), _rect(20, 0, 10, 10)]
        results = check_all_overlaps(polys)
        conflicts = find_conflicting_pairs(results)
        assert conflicts == []

    def test_large_batch_all_overlap(self):
        polys = [_SAME] * 4
        results = check_all_overlaps(polys, iou_thresh=0.05)
        conflicts = find_conflicting_pairs(results)
        assert len(conflicts) == 6  # C(4,2) = 6
