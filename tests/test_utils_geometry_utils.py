"""Tests for puzzle_reconstruction.utils.geometry_utils."""
import pytest
from puzzle_reconstruction.utils.geometry_utils import (
    BoundingBox,
    OverlapSummary,
    GeometryComparisonRecord,
    bbox_from_points,
    summarize_overlaps,
    rank_geometry_comparisons,
)


# ── BoundingBox ───────────────────────────────────────────────────────────────

def test_bbox_area():
    bb = BoundingBox(x=0, y=0, width=5, height=3)
    assert bb.area == pytest.approx(15.0)


def test_bbox_area_zero():
    bb = BoundingBox(x=0, y=0, width=0, height=10)
    assert bb.area == 0.0


def test_bbox_aspect_ratio():
    bb = BoundingBox(x=0, y=0, width=10, height=5)
    assert bb.aspect_ratio == pytest.approx(2.0)


def test_bbox_aspect_ratio_zero_height():
    bb = BoundingBox(x=0, y=0, width=5, height=0)
    assert bb.aspect_ratio == 0.0


def test_bbox_center():
    bb = BoundingBox(x=0, y=0, width=10, height=6)
    cx, cy = bb.center
    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(3.0)


def test_bbox_invalid_width():
    with pytest.raises(ValueError):
        BoundingBox(x=0, y=0, width=-1, height=5)


def test_bbox_invalid_height():
    with pytest.raises(ValueError):
        BoundingBox(x=0, y=0, width=5, height=-1)


def test_bbox_iou_full_overlap():
    bb = BoundingBox(x=0, y=0, width=10, height=10)
    assert bb.iou(bb) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    b1 = BoundingBox(x=0, y=0, width=5, height=5)
    b2 = BoundingBox(x=100, y=100, width=5, height=5)
    assert b1.iou(b2) == 0.0


def test_bbox_iou_partial_overlap():
    b1 = BoundingBox(x=0, y=0, width=10, height=10)
    b2 = BoundingBox(x=5, y=5, width=10, height=10)
    iou = b1.iou(b2)
    assert 0.0 < iou < 1.0


def test_bbox_iou_in_01():
    b1 = BoundingBox(x=0, y=0, width=8, height=8)
    b2 = BoundingBox(x=4, y=4, width=8, height=8)
    iou = b1.iou(b2)
    assert 0.0 <= iou <= 1.0


# ── OverlapSummary ────────────────────────────────────────────────────────────

def test_overlap_summary_conflict_ratio():
    s = OverlapSummary(n_pairs=10, n_conflicting=4, mean_iou=0.1, max_iou=0.5)
    assert s.conflict_ratio == pytest.approx(0.4)


def test_overlap_summary_zero_pairs():
    s = OverlapSummary(n_pairs=0, n_conflicting=0, mean_iou=0.0, max_iou=0.0)
    assert s.conflict_ratio == 0.0


def test_overlap_summary_invalid_n_pairs():
    with pytest.raises(ValueError):
        OverlapSummary(n_pairs=-1, n_conflicting=0, mean_iou=0.0, max_iou=0.0)


def test_overlap_summary_invalid_mean_iou():
    with pytest.raises(ValueError):
        OverlapSummary(n_pairs=5, n_conflicting=0, mean_iou=1.5, max_iou=0.5)


def test_overlap_summary_invalid_max_iou():
    with pytest.raises(ValueError):
        OverlapSummary(n_pairs=5, n_conflicting=0, mean_iou=0.3, max_iou=-0.1)


# ── GeometryComparisonRecord ──────────────────────────────────────────────────

def test_geo_record_valid():
    r = GeometryComparisonRecord(idx1=0, idx2=1, aspect_score=0.8, area_score=0.6, total_score=0.7)
    assert r.total_score == pytest.approx(0.7)


def test_geo_record_invalid_idx():
    with pytest.raises(ValueError):
        GeometryComparisonRecord(idx1=-1, idx2=0, aspect_score=0.5, area_score=0.5, total_score=0.5)


def test_geo_record_invalid_score():
    with pytest.raises(ValueError):
        GeometryComparisonRecord(idx1=0, idx2=1, aspect_score=1.5, area_score=0.5, total_score=0.5)


# ── bbox_from_points ──────────────────────────────────────────────────────────

def test_bbox_from_points_basic():
    pts = [(0, 0), (10, 0), (10, 5), (0, 5)]
    bb = bbox_from_points(pts)
    assert bb.width == pytest.approx(10.0)
    assert bb.height == pytest.approx(5.0)
    assert bb.x == pytest.approx(0.0)
    assert bb.y == pytest.approx(0.0)


def test_bbox_from_points_single():
    bb = bbox_from_points([(3.0, 4.0)])
    assert bb.width == 0.0
    assert bb.height == 0.0
    assert bb.x == pytest.approx(3.0)
    assert bb.y == pytest.approx(4.0)


def test_bbox_from_points_returns_bbox():
    pts = [(1, 2), (3, 5)]
    assert isinstance(bbox_from_points(pts), BoundingBox)


def test_bbox_from_points_empty():
    with pytest.raises(ValueError):
        bbox_from_points([])


# ── summarize_overlaps ────────────────────────────────────────────────────────

def test_summarize_overlaps_empty():
    s = summarize_overlaps([])
    assert s.n_pairs == 0
    assert s.mean_iou == 0.0


def test_summarize_overlaps_basic():
    ious = [0.1, 0.3, 0.8, 0.0]
    s = summarize_overlaps(ious, iou_threshold=0.05)
    assert s.n_pairs == 4
    assert s.max_iou == pytest.approx(0.8)


def test_summarize_overlaps_conflicting():
    ious = [0.1, 0.2, 0.0]
    s = summarize_overlaps(ious, iou_threshold=0.05)
    assert s.n_conflicting == 2  # 0.1 and 0.2 exceed threshold


def test_summarize_overlaps_returns_summary():
    s = summarize_overlaps([0.5, 0.3])
    assert isinstance(s, OverlapSummary)


# ── rank_geometry_comparisons ─────────────────────────────────────────────────

def test_rank_geometry_comparisons_order():
    records = [
        GeometryComparisonRecord(0, 1, 0.5, 0.5, 0.3),
        GeometryComparisonRecord(1, 2, 0.8, 0.9, 0.9),
        GeometryComparisonRecord(2, 3, 0.6, 0.7, 0.6),
    ]
    ranked = rank_geometry_comparisons(records)
    assert ranked[0][1].total_score >= ranked[1][1].total_score


def test_rank_geometry_comparisons_ranks():
    records = [
        GeometryComparisonRecord(0, 1, 0.5, 0.5, 0.5),
        GeometryComparisonRecord(0, 2, 0.8, 0.8, 0.8),
    ]
    ranked = rank_geometry_comparisons(records)
    assert ranked[0][0] == 1  # rank starts at 1


def test_rank_geometry_comparisons_empty():
    assert rank_geometry_comparisons([]) == []


def test_rank_geometry_comparisons_single():
    records = [GeometryComparisonRecord(0, 1, 0.5, 0.5, 0.5)]
    ranked = rank_geometry_comparisons(records)
    assert len(ranked) == 1
    assert ranked[0][0] == 1
