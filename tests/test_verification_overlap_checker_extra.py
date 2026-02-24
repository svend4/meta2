"""Extra tests for puzzle_reconstruction/verification/overlap_checker.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _square(x=0, y=0, size=100):
    """Square polygon at (x,y) with given size."""
    return np.array([
        [x, y], [x + size, y],
        [x + size, y + size], [x, y + size],
    ], dtype=np.float32)


def _triangle(x=0, y=0, size=100):
    return np.array([
        [x, y], [x + size, y], [x + size / 2, y + size],
    ], dtype=np.float32)


# ─── OverlapResult ───────────────────────────────────────────────────────────

class TestOverlapResultExtra:
    def test_creation(self):
        r = OverlapResult(
            idx1=0, idx2=1, intersection_area=50.0,
            iou=0.1, has_overlap=True,
        )
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.method == "polygon"

    def test_defaults(self):
        r = OverlapResult(idx1=0, idx2=1, intersection_area=0.0,
                          iou=0.0, has_overlap=False)
        assert r.method == "polygon"
        assert r.params == {}

    def test_repr(self):
        r = OverlapResult(idx1=0, idx2=1, intersection_area=50.0,
                          iou=0.25, has_overlap=True)
        s = repr(r)
        assert "(0,1)" in s
        assert "0.25" in s


# ─── polygon_intersection_area ───────────────────────────────────────────────

class TestPolygonIntersectionAreaExtra:
    def test_no_overlap(self):
        a = _square(0, 0, 50)
        b = _square(200, 200, 50)
        area = polygon_intersection_area(a, b)
        assert area == pytest.approx(0.0, abs=5.0)

    def test_full_overlap(self):
        a = _square(0, 0, 100)
        area = polygon_intersection_area(a, a)
        # Rasterized → approximately 100*100 = 10000
        assert area > 9000

    def test_partial_overlap(self):
        a = _square(0, 0, 100)
        b = _square(50, 50, 100)
        area = polygon_intersection_area(a, b)
        # Overlap region ≈ 50*50 = 2500
        assert 2000 < area < 3000

    def test_reshapes_correctly(self):
        a = _square(0, 0, 100).reshape(-1, 1, 2)
        b = _square(0, 0, 100)
        area = polygon_intersection_area(a, b)
        assert area > 9000


# ─── polygon_union_area ──────────────────────────────────────────────────────

class TestPolygonUnionAreaExtra:
    def test_no_overlap(self):
        a = _square(0, 0, 50)
        b = _square(200, 200, 50)
        area = polygon_union_area(a, b)
        # Two separate 50x50 squares ≈ 5000
        assert area > 4000

    def test_full_overlap(self):
        a = _square(0, 0, 100)
        area = polygon_union_area(a, a)
        assert area > 9000

    def test_partial(self):
        a = _square(0, 0, 100)
        b = _square(50, 50, 100)
        area = polygon_union_area(a, b)
        # Union ≈ 2*10000 - 2500 = 17500
        assert area > 15000


# ─── polygon_iou ─────────────────────────────────────────────────────────────

class TestPolygonIouExtra:
    def test_identical(self):
        a = _square(0, 0, 100)
        iou = polygon_iou(a, a)
        assert iou == pytest.approx(1.0, abs=0.05)

    def test_no_overlap(self):
        a = _square(0, 0, 50)
        b = _square(200, 200, 50)
        iou = polygon_iou(a, b)
        assert iou == pytest.approx(0.0, abs=0.01)

    def test_partial(self):
        a = _square(0, 0, 100)
        b = _square(50, 50, 100)
        iou = polygon_iou(a, b)
        assert 0.0 < iou < 1.0


# ─── check_overlap_pair ─────────────────────────────────────────────────────

class TestCheckOverlapPairExtra:
    def test_no_overlap(self):
        r = check_overlap_pair(_square(0, 0), _square(500, 500), 0, 1)
        assert r.has_overlap is False
        assert r.iou == pytest.approx(0.0, abs=0.01)

    def test_with_overlap(self):
        r = check_overlap_pair(_square(0, 0), _square(50, 50), 0, 1)
        assert r.has_overlap is True
        assert r.iou > 0.05

    def test_custom_threshold(self):
        r = check_overlap_pair(_square(0, 0), _square(50, 50), 0, 1,
                               iou_thresh=0.9)
        assert r.has_overlap is False

    def test_indices(self):
        r = check_overlap_pair(_square(), _square(), 5, 10)
        assert r.idx1 == 5
        assert r.idx2 == 10


# ─── check_all_overlaps ─────────────────────────────────────────────────────

class TestCheckAllOverlapsExtra:
    def test_empty(self):
        assert check_all_overlaps([]) == []

    def test_single(self):
        assert check_all_overlaps([_square()]) == []

    def test_two_no_overlap(self):
        results = check_all_overlaps([_square(0, 0), _square(500, 500)])
        assert len(results) == 1
        assert results[0].has_overlap is False

    def test_three(self):
        polys = [_square(0, 0), _square(50, 50), _square(500, 500)]
        results = check_all_overlaps(polys)
        assert len(results) == 3  # C(3,2)

    def test_with_triangles(self):
        results = check_all_overlaps([_triangle(0, 0), _triangle(50, 50)])
        assert len(results) == 1


# ─── find_conflicting_pairs ──────────────────────────────────────────────────

class TestFindConflictingPairsExtra:
    def test_no_conflicts(self):
        results = check_all_overlaps([_square(0, 0), _square(500, 500)])
        conflicts = find_conflicting_pairs(results)
        assert conflicts == []

    def test_with_conflicts(self):
        results = check_all_overlaps([_square(0, 0), _square(50, 50)])
        conflicts = find_conflicting_pairs(results)
        assert len(conflicts) == 1

    def test_custom_threshold(self):
        results = check_all_overlaps([_square(0, 0), _square(50, 50)])
        conflicts = find_conflicting_pairs(results, iou_thresh=0.99)
        assert conflicts == []

    def test_empty(self):
        assert find_conflicting_pairs([]) == []
