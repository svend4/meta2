"""Extra tests for puzzle_reconstruction/utils/geometry_utils.py."""
from __future__ import annotations

import pytest

from puzzle_reconstruction.utils.geometry_utils import (
    BoundingBox,
    OverlapSummary,
    GeometryComparisonRecord,
    bbox_from_points,
    summarize_overlaps,
    rank_geometry_comparisons,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bbox(x=0.0, y=0.0, w=10.0, h=5.0) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=w, height=h)


def _gcr(idx1=0, idx2=1, total=0.7) -> GeometryComparisonRecord:
    return GeometryComparisonRecord(idx1=idx1, idx2=idx2,
                                     aspect_score=0.8, area_score=0.7,
                                     total_score=total)


# ─── BoundingBox ──────────────────────────────────────────────────────────────

class TestBoundingBoxExtra:
    def test_stores_dimensions(self):
        bb = _bbox(x=1.0, y=2.0, w=8.0, h=4.0)
        assert bb.x == pytest.approx(1.0) and bb.y == pytest.approx(2.0)
        assert bb.width == pytest.approx(8.0) and bb.height == pytest.approx(4.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            BoundingBox(x=0.0, y=0.0, width=-1.0, height=5.0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            BoundingBox(x=0.0, y=0.0, width=5.0, height=-1.0)

    def test_area(self):
        assert _bbox(w=4.0, h=3.0).area == pytest.approx(12.0)

    def test_aspect_ratio(self):
        assert _bbox(w=10.0, h=5.0).aspect_ratio == pytest.approx(2.0)

    def test_aspect_ratio_zero_height(self):
        bb = BoundingBox(x=0.0, y=0.0, width=5.0, height=0.0)
        assert bb.aspect_ratio == pytest.approx(0.0)

    def test_center(self):
        cx, cy = _bbox(x=0.0, y=0.0, w=10.0, h=4.0).center
        assert cx == pytest.approx(5.0) and cy == pytest.approx(2.0)

    def test_iou_identical(self):
        bb = _bbox()
        assert bb.iou(bb) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = _bbox(x=0.0, y=0.0, w=5.0, h=5.0)
        b = _bbox(x=100.0, y=100.0, w=5.0, h=5.0)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_in_range(self):
        a = _bbox(x=0.0, y=0.0, w=10.0, h=10.0)
        b = _bbox(x=5.0, y=5.0, w=10.0, h=10.0)
        assert 0.0 <= a.iou(b) <= 1.0


# ─── OverlapSummary ───────────────────────────────────────────────────────────

class TestOverlapSummaryExtra:
    def test_stores_n_pairs(self):
        s = OverlapSummary(n_pairs=5, n_conflicting=2, mean_iou=0.3, max_iou=0.7)
        assert s.n_pairs == 5

    def test_negative_n_pairs_raises(self):
        with pytest.raises(ValueError):
            OverlapSummary(n_pairs=-1, n_conflicting=0, mean_iou=0.0, max_iou=0.0)

    def test_mean_iou_out_of_range_raises(self):
        with pytest.raises(ValueError):
            OverlapSummary(n_pairs=1, n_conflicting=0, mean_iou=1.5, max_iou=0.5)

    def test_conflict_ratio(self):
        s = OverlapSummary(n_pairs=4, n_conflicting=2, mean_iou=0.3, max_iou=0.6)
        assert s.conflict_ratio == pytest.approx(0.5)

    def test_conflict_ratio_zero_pairs(self):
        s = OverlapSummary(n_pairs=0, n_conflicting=0, mean_iou=0.0, max_iou=0.0)
        assert s.conflict_ratio == pytest.approx(0.0)


# ─── GeometryComparisonRecord ─────────────────────────────────────────────────

class TestGeometryComparisonRecordExtra:
    def test_stores_indices(self):
        r = _gcr(idx1=2, idx2=5)
        assert r.idx1 == 2 and r.idx2 == 5

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            GeometryComparisonRecord(idx1=-1, idx2=0, aspect_score=0.5,
                                      area_score=0.5, total_score=0.5)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            GeometryComparisonRecord(idx1=0, idx2=1, aspect_score=1.5,
                                      area_score=0.5, total_score=0.5)

    def test_stores_total_score(self):
        assert _gcr(total=0.65).total_score == pytest.approx(0.65)


# ─── bbox_from_points ─────────────────────────────────────────────────────────

class TestBboxFromPointsExtra:
    def test_returns_bounding_box(self):
        pts = [(0.0, 0.0), (5.0, 3.0)]
        assert isinstance(bbox_from_points(pts), BoundingBox)

    def test_correct_dimensions(self):
        pts = [(1.0, 2.0), (5.0, 7.0)]
        bb = bbox_from_points(pts)
        assert bb.x == pytest.approx(1.0) and bb.y == pytest.approx(2.0)
        assert bb.width == pytest.approx(4.0) and bb.height == pytest.approx(5.0)

    def test_single_point(self):
        bb = bbox_from_points([(3.0, 4.0)])
        assert bb.width == pytest.approx(0.0) and bb.height == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bbox_from_points([])


# ─── summarize_overlaps ───────────────────────────────────────────────────────

class TestSummarizeOverlapsExtra:
    def test_returns_overlap_summary(self):
        assert isinstance(summarize_overlaps([0.1, 0.5, 0.9]), OverlapSummary)

    def test_empty_input(self):
        s = summarize_overlaps([])
        assert s.n_pairs == 0

    def test_conflicting_count(self):
        s = summarize_overlaps([0.01, 0.06, 0.03], iou_threshold=0.05)
        assert s.n_conflicting == 1

    def test_mean_iou(self):
        s = summarize_overlaps([0.2, 0.4, 0.6])
        assert s.mean_iou == pytest.approx(0.4)


# ─── rank_geometry_comparisons ────────────────────────────────────────────────

class TestRankGeometryComparisonsExtra:
    def test_returns_list(self):
        recs = [_gcr(total=0.5), _gcr(total=0.8)]
        result = rank_geometry_comparisons(recs)
        assert isinstance(result, list)

    def test_descending_order(self):
        recs = [_gcr(total=0.3), _gcr(total=0.9), _gcr(total=0.6)]
        ranked = rank_geometry_comparisons(recs)
        totals = [r.total_score for _, r in ranked]
        assert totals == sorted(totals, reverse=True)

    def test_rank_starts_at_one(self):
        recs = [_gcr()]
        assert rank_geometry_comparisons(recs)[0][0] == 1

    def test_empty_returns_empty(self):
        assert rank_geometry_comparisons([]) == []
