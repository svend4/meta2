"""Tests for puzzle_reconstruction.verification.overlap_checker"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

from puzzle_reconstruction.verification.overlap_checker import (
    OverlapResult,
    polygon_intersection_area,
    polygon_union_area,
    polygon_iou,
    check_overlap_pair,
    check_all_overlaps,
    find_conflicting_pairs,
)


def square(x, y, size):
    """Return square polygon as (4,2) array."""
    return np.array([
        [x, y], [x + size, y],
        [x + size, y + size], [x, y + size]
    ], dtype=np.float32)


# ─── OverlapResult ────────────────────────────────────────────────────────────

def test_overlap_result_repr():
    r = OverlapResult(idx1=0, idx2=1, intersection_area=100.0, iou=0.5, has_overlap=True)
    assert "0" in repr(r)
    assert "1" in repr(r)


def test_overlap_result_method():
    r = OverlapResult(idx1=0, idx2=1, intersection_area=0.0, iou=0.0, has_overlap=False)
    assert r.method == "polygon"


# ─── polygon_intersection_area ────────────────────────────────────────────────

def test_intersection_no_overlap():
    p1 = square(0, 0, 50)
    p2 = square(100, 100, 50)
    area = polygon_intersection_area(p1, p2)
    assert area == pytest.approx(0.0, abs=5.0)


def test_intersection_full_overlap():
    p1 = square(0, 0, 50)
    p2 = square(0, 0, 50)
    area = polygon_intersection_area(p1, p2)
    assert area > 1000  # Should be close to 2500 for 50x50


def test_intersection_partial():
    p1 = square(0, 0, 100)
    p2 = square(50, 0, 100)
    area = polygon_intersection_area(p1, p2)
    assert area > 0.0
    # Should be around 50*100 = 5000
    assert area < 10000


def test_intersection_touching_edge():
    # Touching polygons may return a small area due to edge-sharing in rasterisation
    p1 = square(0, 0, 50)
    p2 = square(50, 0, 50)
    area = polygon_intersection_area(p1, p2)
    assert area < 100.0  # negligible compared to polygon area 2500


# ─── polygon_union_area ───────────────────────────────────────────────────────

def test_union_no_overlap():
    p1 = square(0, 0, 50)
    p2 = square(100, 100, 50)
    area = polygon_union_area(p1, p2)
    # Should be 2 * 50^2 ≈ 5000
    assert area > 0.0


def test_union_full_overlap():
    p1 = square(0, 0, 50)
    p2 = square(0, 0, 50)
    area = polygon_union_area(p1, p2)
    # Should be same as single polygon area
    assert area > 0.0


def test_union_partial():
    p1 = square(0, 0, 100)
    p2 = square(50, 0, 100)
    area = polygon_union_area(p1, p2)
    inter = polygon_intersection_area(p1, p2)
    assert area > inter


# ─── polygon_iou ──────────────────────────────────────────────────────────────

def test_polygon_iou_no_overlap():
    p1 = square(0, 0, 50)
    p2 = square(100, 100, 50)
    iou = polygon_iou(p1, p2)
    assert iou == pytest.approx(0.0, abs=0.05)


def test_polygon_iou_full_overlap():
    p1 = square(0, 0, 50)
    p2 = square(0, 0, 50)
    iou = polygon_iou(p1, p2)
    assert iou == pytest.approx(1.0, abs=0.05)


def test_polygon_iou_range():
    p1 = square(0, 0, 100)
    p2 = square(50, 0, 100)
    iou = polygon_iou(p1, p2)
    assert 0.0 <= iou <= 1.0


def test_polygon_iou_symmetry():
    p1 = square(0, 0, 60)
    p2 = square(30, 0, 60)
    assert polygon_iou(p1, p2) == pytest.approx(polygon_iou(p2, p1), abs=0.01)


# ─── check_overlap_pair ───────────────────────────────────────────────────────

def test_check_overlap_pair_no_overlap():
    p1 = square(0, 0, 50)
    p2 = square(100, 100, 50)
    result = check_overlap_pair(p1, p2, idx1=0, idx2=1, iou_thresh=0.05)
    assert not result.has_overlap
    assert result.idx1 == 0
    assert result.idx2 == 1


def test_check_overlap_pair_has_overlap():
    p1 = square(0, 0, 100)
    p2 = square(50, 50, 100)
    result = check_overlap_pair(p1, p2, iou_thresh=0.01)
    assert result.has_overlap


def test_check_overlap_pair_result_type():
    p1 = square(0, 0, 50)
    p2 = square(25, 25, 50)
    result = check_overlap_pair(p1, p2)
    assert isinstance(result, OverlapResult)


def test_check_overlap_pair_params():
    p1 = square(0, 0, 50)
    p2 = square(25, 25, 50)
    result = check_overlap_pair(p1, p2, iou_thresh=0.1)
    assert result.params.get("iou_thresh") == 0.1


# ─── check_all_overlaps ───────────────────────────────────────────────────────

def test_check_all_overlaps_n_pairs():
    polygons = [square(i * 100, 0, 50) for i in range(4)]
    results = check_all_overlaps(polygons)
    # n*(n-1)/2 = 4*3/2 = 6
    assert len(results) == 6


def test_check_all_overlaps_no_overlaps():
    polygons = [square(i * 200, 0, 50) for i in range(3)]
    results = check_all_overlaps(polygons, iou_thresh=0.05)
    conflicts = [r for r in results if r.has_overlap]
    assert conflicts == []


def test_check_all_overlaps_with_overlap():
    polygons = [square(0, 0, 100), square(50, 50, 100), square(500, 0, 50)]
    results = check_all_overlaps(polygons, iou_thresh=0.01)
    conflicts = [r for r in results if r.has_overlap]
    assert len(conflicts) >= 1


def test_check_all_overlaps_indices():
    polygons = [square(i * 200, 0, 50) for i in range(3)]
    results = check_all_overlaps(polygons)
    pairs = [(r.idx1, r.idx2) for r in results]
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs


def test_check_all_overlaps_single():
    polygons = [square(0, 0, 50)]
    results = check_all_overlaps(polygons)
    assert results == []


def test_check_all_overlaps_empty():
    results = check_all_overlaps([])
    assert results == []


# ─── find_conflicting_pairs ───────────────────────────────────────────────────

def test_find_conflicting_pairs_uses_has_overlap():
    r1 = OverlapResult(idx1=0, idx2=1, intersection_area=100.0, iou=0.6, has_overlap=True)
    r2 = OverlapResult(idx1=1, idx2=2, intersection_area=0.0, iou=0.0, has_overlap=False)
    conflicts = find_conflicting_pairs([r1, r2])
    assert len(conflicts) == 1
    assert conflicts[0].idx1 == 0


def test_find_conflicting_pairs_custom_thresh():
    r1 = OverlapResult(idx1=0, idx2=1, intersection_area=100.0, iou=0.3, has_overlap=False)
    r2 = OverlapResult(idx1=1, idx2=2, intersection_area=0.0, iou=0.1, has_overlap=False)
    conflicts = find_conflicting_pairs([r1, r2], iou_thresh=0.2)
    assert len(conflicts) == 1


def test_find_conflicting_pairs_empty():
    assert find_conflicting_pairs([]) == []


def test_find_conflicting_pairs_all_pass():
    results = [
        OverlapResult(idx1=i, idx2=i+1, intersection_area=0.0, iou=0.0, has_overlap=False)
        for i in range(3)
    ]
    assert find_conflicting_pairs(results) == []
