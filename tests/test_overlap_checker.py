"""Тесты для puzzle_reconstruction/verification/overlap_checker.py."""
import numpy as np
import pytest

from puzzle_reconstruction.verification.overlap_checker import (
    OverlapResult,
    polygon_intersection_area,
    polygon_union_area,
    polygon_iou,
    check_overlap_pair,
    check_all_overlaps,
    find_conflicting_pairs,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rect(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Прямоугольный полигон (4 вершины)."""
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


# Две непересекающихся квадрата
_POLY_A = _rect(0, 0, 30, 30)
_POLY_B = _rect(50, 50, 30, 30)

# Два совпадающих квадрата
_POLY_SAME = _rect(10, 10, 40, 40)

# Два частично перекрывающихся квадрата
_POLY_P1 = _rect(0, 0, 30, 30)
_POLY_P2 = _rect(20, 20, 30, 30)   # перекрытие 10×10 = 100 пикс²

# Единичный полигон
_POLY_UNIT = _rect(0, 0, 5, 5)


# ─── OverlapResult ────────────────────────────────────────────────────────────

class TestOverlapResult:
    def _make(self, **kw):
        defaults = dict(idx1=0, idx2=1, intersection_area=10.0,
                         iou=0.1, has_overlap=True)
        defaults.update(kw)
        return OverlapResult(**defaults)

    def test_fields(self):
        r = self._make()
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.intersection_area == pytest.approx(10.0)
        assert r.iou == pytest.approx(0.1)
        assert r.has_overlap is True

    def test_method_default_polygon(self):
        r = self._make()
        assert r.method == "polygon"

    def test_params_default_empty(self):
        r = self._make()
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        r = self._make(params={"iou_thresh": 0.05})
        assert r.params["iou_thresh"] == pytest.approx(0.05)

    def test_repr_contains_class(self):
        assert "OverlapResult" in repr(self._make())

    def test_repr_contains_iou(self):
        r = self._make(iou=0.4321)
        assert "0.43" in repr(r) or "iou" in repr(r).lower()

    def test_repr_contains_has_overlap(self):
        r = self._make(has_overlap=False)
        assert "False" in repr(r)

    def test_has_overlap_false(self):
        r = self._make(has_overlap=False)
        assert r.has_overlap is False

    def test_intersection_area_zero(self):
        r = self._make(intersection_area=0.0, iou=0.0, has_overlap=False)
        assert r.intersection_area == pytest.approx(0.0)


# ─── polygon_intersection_area ────────────────────────────────────────────────

class TestPolygonIntersectionArea:
    def test_returns_float(self):
        assert isinstance(polygon_intersection_area(_POLY_A, _POLY_B), float)

    def test_non_overlapping_zero(self):
        area = polygon_intersection_area(_POLY_A, _POLY_B)
        assert area == pytest.approx(0.0, abs=1.0)

    def test_identical_approx_polygon_area(self):
        area = polygon_intersection_area(_POLY_SAME, _POLY_SAME)
        # 40×40 = 1600 пикс²
        assert area == pytest.approx(1600.0, rel=0.05)

    def test_partial_overlap_positive(self):
        area = polygon_intersection_area(_POLY_P1, _POLY_P2)
        assert area > 0.0

    def test_partial_overlap_less_than_each(self):
        inter = polygon_intersection_area(_POLY_P1, _POLY_P2)
        area_p1 = polygon_intersection_area(_POLY_P1, _POLY_P1)
        assert inter < area_p1

    def test_nonneg(self):
        assert polygon_intersection_area(_POLY_A, _POLY_B) >= 0.0

    def test_commutative(self):
        a = polygon_intersection_area(_POLY_P1, _POLY_P2)
        b = polygon_intersection_area(_POLY_P2, _POLY_P1)
        assert a == pytest.approx(b, rel=0.05)

    def test_small_polys(self):
        a = _rect(0, 0, 5, 5)
        b = _rect(3, 3, 5, 5)
        area = polygon_intersection_area(a, b)
        assert area > 0.0
        assert area < 25.0

    def test_no_overlap_different_sides(self):
        left  = _rect(0, 0, 20, 40)
        right = _rect(30, 0, 20, 40)
        assert polygon_intersection_area(left, right) == pytest.approx(0.0, abs=1.0)

    def test_n1_form_input(self):
        poly1 = _POLY_SAME.reshape(-1, 1, 2)
        poly2 = _POLY_SAME.reshape(-1, 1, 2)
        area  = polygon_intersection_area(poly1, poly2)
        assert area == pytest.approx(1600.0, rel=0.05)


# ─── polygon_union_area ───────────────────────────────────────────────────────

class TestPolygonUnionArea:
    def test_returns_float(self):
        assert isinstance(polygon_union_area(_POLY_A, _POLY_B), float)

    def test_nonneg(self):
        assert polygon_union_area(_POLY_A, _POLY_B) >= 0.0

    def test_union_gte_intersection(self):
        union = polygon_union_area(_POLY_P1, _POLY_P2)
        inter = polygon_intersection_area(_POLY_P1, _POLY_P2)
        assert union >= inter

    def test_non_overlapping_approx_sum(self):
        union = polygon_union_area(_POLY_A, _POLY_B)
        area_a = polygon_union_area(_POLY_A, _POLY_A)
        area_b = polygon_union_area(_POLY_B, _POLY_B)
        # Для непересекающихся: union ≈ area_a + area_b
        assert union == pytest.approx(area_a + area_b, rel=0.05)

    def test_identical_approx_area(self):
        union = polygon_union_area(_POLY_SAME, _POLY_SAME)
        assert union == pytest.approx(1600.0, rel=0.05)

    def test_commutative(self):
        a = polygon_union_area(_POLY_P1, _POLY_P2)
        b = polygon_union_area(_POLY_P2, _POLY_P1)
        assert a == pytest.approx(b, rel=0.05)

    def test_partial_union_gt_each_area(self):
        union  = polygon_union_area(_POLY_P1, _POLY_P2)
        area1  = polygon_union_area(_POLY_P1, _POLY_P1)
        area2  = polygon_union_area(_POLY_P2, _POLY_P2)
        assert union > max(area1, area2)


# ─── polygon_iou ──────────────────────────────────────────────────────────────

class TestPolygonIou:
    def test_returns_float(self):
        assert isinstance(polygon_iou(_POLY_A, _POLY_B), float)

    def test_non_overlapping_zero(self):
        assert polygon_iou(_POLY_A, _POLY_B) == pytest.approx(0.0, abs=0.01)

    def test_identical_one(self):
        assert polygon_iou(_POLY_SAME, _POLY_SAME) == pytest.approx(1.0, rel=0.05)

    def test_in_range(self):
        v = polygon_iou(_POLY_P1, _POLY_P2)
        assert 0.0 <= v <= 1.0

    def test_partial_in_open_interval(self):
        v = polygon_iou(_POLY_P1, _POLY_P2)
        assert 0.0 < v < 1.0

    def test_commutative(self):
        a = polygon_iou(_POLY_P1, _POLY_P2)
        b = polygon_iou(_POLY_P2, _POLY_P1)
        assert a == pytest.approx(b, rel=0.05)

    def test_small_polys_nonneg(self):
        assert polygon_iou(_POLY_UNIT, _POLY_UNIT) >= 0.0

    def test_fully_inside(self):
        outer = _rect(0, 0, 40, 40)
        inner = _rect(10, 10, 10, 10)
        iou   = polygon_iou(inner, outer)
        # inner полностью внутри outer → IoU = area_inner / area_outer ≈ 100/1600
        assert 0.0 < iou < 1.0

    def test_far_apart_zero(self):
        a = _rect(0, 0, 10, 10)
        b = _rect(200, 200, 10, 10)
        assert polygon_iou(a, b) == pytest.approx(0.0, abs=0.01)


# ─── check_overlap_pair ───────────────────────────────────────────────────────

class TestCheckOverlapPair:
    def test_returns_result(self):
        r = check_overlap_pair(_POLY_A, _POLY_B)
        assert isinstance(r, OverlapResult)

    def test_non_overlapping_has_overlap_false(self):
        r = check_overlap_pair(_POLY_A, _POLY_B, iou_thresh=0.05)
        assert r.has_overlap is False

    def test_identical_has_overlap_true(self):
        r = check_overlap_pair(_POLY_SAME, _POLY_SAME, iou_thresh=0.05)
        assert r.has_overlap is True

    def test_idx_stored(self):
        r = check_overlap_pair(_POLY_P1, _POLY_P2, idx1=3, idx2=7)
        assert r.idx1 == 3
        assert r.idx2 == 7

    def test_params_iou_thresh_stored(self):
        r = check_overlap_pair(_POLY_A, _POLY_B, iou_thresh=0.10)
        assert r.params.get("iou_thresh") == pytest.approx(0.10)

    def test_partial_overlap_iou_positive(self):
        r = check_overlap_pair(_POLY_P1, _POLY_P2)
        assert r.iou > 0.0

    def test_intersection_area_nonneg(self):
        r = check_overlap_pair(_POLY_A, _POLY_B)
        assert r.intersection_area >= 0.0

    def test_method_is_polygon(self):
        r = check_overlap_pair(_POLY_A, _POLY_B)
        assert r.method == "polygon"

    def test_iou_in_range(self):
        r = check_overlap_pair(_POLY_P1, _POLY_P2)
        assert 0.0 <= r.iou <= 1.0

    def test_threshold_boundary_low(self):
        r = check_overlap_pair(_POLY_P1, _POLY_P2, iou_thresh=0.0)
        # Любое частичное перекрытие > 0.0 → has_overlap True
        assert r.has_overlap is True

    def test_threshold_boundary_high(self):
        r = check_overlap_pair(_POLY_P1, _POLY_P2, iou_thresh=1.0)
        # IoU < 1.0 → has_overlap False
        assert r.has_overlap is False


# ─── check_all_overlaps ───────────────────────────────────────────────────────

class TestCheckAllOverlaps:
    def test_empty_returns_empty(self):
        assert check_all_overlaps([]) == []

    def test_one_poly_returns_empty(self):
        assert check_all_overlaps([_POLY_A]) == []

    def test_two_polys_one_pair(self):
        r = check_all_overlaps([_POLY_A, _POLY_B])
        assert len(r) == 1
        assert isinstance(r[0], OverlapResult)

    def test_three_polys_three_pairs(self):
        r = check_all_overlaps([_POLY_A, _POLY_B, _POLY_SAME])
        assert len(r) == 3

    def test_four_polys_six_pairs(self):
        polys = [_POLY_A, _POLY_B, _POLY_P1, _POLY_P2]
        r     = check_all_overlaps(polys)
        assert len(r) == 6

    def test_each_is_result(self):
        for r in check_all_overlaps([_POLY_P1, _POLY_P2, _POLY_A]):
            assert isinstance(r, OverlapResult)

    def test_idx_ordered(self):
        results = check_all_overlaps([_POLY_A, _POLY_B, _POLY_P1])
        for r in results:
            assert r.idx1 < r.idx2

    def test_iou_in_range(self):
        for r in check_all_overlaps([_POLY_P1, _POLY_P2, _POLY_A]):
            assert 0.0 <= r.iou <= 1.0

    def test_non_overlapping_all_false(self):
        polys   = [_rect(0, 0, 10, 10), _rect(20, 0, 10, 10),
                   _rect(40, 0, 10, 10)]
        results = check_all_overlaps(polys, iou_thresh=0.05)
        for r in results:
            assert r.has_overlap is False

    def test_iou_thresh_forwarded(self):
        r = check_all_overlaps([_POLY_SAME, _POLY_SAME], iou_thresh=0.5)
        assert r[0].has_overlap is True


# ─── find_conflicting_pairs ───────────────────────────────────────────────────

class TestFindConflictingPairs:
    def _results(self):
        return check_all_overlaps(
            [_POLY_SAME, _POLY_SAME, _POLY_A, _POLY_B],
            iou_thresh=0.05,
        )

    def test_returns_list(self):
        assert isinstance(find_conflicting_pairs([]), list)

    def test_empty_input_empty_output(self):
        assert find_conflicting_pairs([]) == []

    def test_filters_has_overlap_true(self):
        results  = self._results()
        conflict = find_conflicting_pairs(results)
        for r in conflict:
            assert r.has_overlap is True

    def test_no_conflicts_empty(self):
        polys   = [_rect(0, 0, 10, 10), _rect(20, 0, 10, 10)]
        results = check_all_overlaps(polys, iou_thresh=0.05)
        assert find_conflicting_pairs(results) == []

    def test_all_conflict(self):
        polys    = [_POLY_SAME, _POLY_SAME]
        results  = check_all_overlaps(polys, iou_thresh=0.05)
        conflict = find_conflicting_pairs(results)
        assert len(conflict) == 1

    def test_iou_thresh_override(self):
        polys    = [_POLY_P1, _POLY_P2]
        results  = check_all_overlaps(polys, iou_thresh=0.05)
        # При thresh=1.0 ничего не конфликтует
        conflict = find_conflicting_pairs(results, iou_thresh=1.0)
        assert conflict == []

    def test_iou_thresh_zero_override(self):
        polys    = [_POLY_P1, _POLY_P2]
        results  = check_all_overlaps(polys, iou_thresh=0.05)
        conflict = find_conflicting_pairs(results, iou_thresh=0.0)
        assert len(conflict) == 1

    def test_each_is_result(self):
        polys   = [_POLY_SAME, _POLY_SAME, _POLY_A]
        results = check_all_overlaps(polys, iou_thresh=0.05)
        for r in find_conflicting_pairs(results):
            assert isinstance(r, OverlapResult)
