"""Extra tests for puzzle_reconstruction/utils/bbox_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.bbox_utils import (
    BBox,
    bbox_iou,
    bbox_intersection,
    bbox_union,
    expand_bbox,
    crop_image,
    bboxes_from_mask,
    merge_overlapping_bboxes,
    bbox_center,
    bbox_aspect_ratio,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _b(x=0, y=0, w=10, h=10) -> BBox:
    return BBox(x=x, y=y, w=w, h=h)


def _img(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


# ─── BBox ─────────────────────────────────────────────────────────────────────

class TestBBoxExtra:
    def test_stores_xywh(self):
        b = BBox(x=5, y=10, w=20, h=30)
        assert b.x == 5 and b.y == 10 and b.w == 20 and b.h == 30

    def test_zero_w_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=0, h=5)

    def test_zero_h_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=5, h=0)

    def test_negative_w_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=-1, h=5)

    def test_x2(self):
        b = _b(x=3, w=7)
        assert b.x2 == 10

    def test_y2(self):
        b = _b(y=4, h=6)
        assert b.y2 == 10

    def test_area(self):
        b = _b(w=5, h=8)
        assert b.area == 40

    def test_to_tuple(self):
        b = BBox(x=1, y=2, w=3, h=4)
        assert b.to_tuple() == (1, 2, 3, 4)


# ─── bbox_iou ─────────────────────────────────────────────────────────────────

class TestBboxIouExtra:
    def test_identical_boxes_iou_one(self):
        b = _b()
        assert bbox_iou(b, b) == pytest.approx(1.0)

    def test_non_overlapping_iou_zero(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=20, y=20, w=10, h=10)
        assert bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=5, y=5, w=10, h=10)
        iou = bbox_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_contained_box(self):
        outer = _b(x=0, y=0, w=20, h=20)
        inner = _b(x=5, y=5, w=5, h=5)
        iou = bbox_iou(outer, inner)
        assert iou == pytest.approx(25 / (400 + 25 - 25))

    def test_result_in_range(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=3, y=3, w=10, h=10)
        assert 0.0 <= bbox_iou(a, b) <= 1.0


# ─── bbox_intersection ────────────────────────────────────────────────────────

class TestBboxIntersectionExtra:
    def test_non_overlapping_returns_none(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=20, y=0, w=10, h=10)
        assert bbox_intersection(a, b) is None

    def test_touching_returns_none(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=10, y=0, w=10, h=10)
        assert bbox_intersection(a, b) is None

    def test_partial_overlap(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=5, y=5, w=10, h=10)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert inter.w == 5 and inter.h == 5

    def test_contained_box(self):
        outer = _b(x=0, y=0, w=20, h=20)
        inner = _b(x=5, y=5, w=5, h=5)
        inter = bbox_intersection(outer, inner)
        assert inter is not None
        assert inter.w == 5 and inter.h == 5

    def test_returns_bbox(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=5, y=5, w=10, h=10)
        inter = bbox_intersection(a, b)
        assert isinstance(inter, BBox)


# ─── bbox_union ───────────────────────────────────────────────────────────────

class TestBboxUnionExtra:
    def test_returns_bbox(self):
        assert isinstance(bbox_union(_b(), _b(x=20)), BBox)

    def test_identical_boxes(self):
        b = _b(x=5, y=5, w=10, h=10)
        u = bbox_union(b, b)
        assert u.to_tuple() == b.to_tuple()

    def test_union_contains_both(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=20, y=20, w=10, h=10)
        u = bbox_union(a, b)
        assert u.x == 0 and u.y == 0
        assert u.x2 == 30 and u.y2 == 30

    def test_area_ge_each(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=5, y=5, w=10, h=10)
        u = bbox_union(a, b)
        assert u.area >= a.area and u.area >= b.area


# ─── expand_bbox ──────────────────────────────────────────────────────────────

class TestExpandBboxExtra:
    def test_returns_bbox(self):
        assert isinstance(expand_bbox(_b(), 5), BBox)

    def test_zero_padding_unchanged(self):
        b = _b(x=5, y=5, w=10, h=10)
        e = expand_bbox(b, 0)
        assert e.to_tuple() == b.to_tuple()

    def test_padding_increases_size(self):
        b = _b(x=5, y=5, w=10, h=10)
        e = expand_bbox(b, 3)
        assert e.w > b.w and e.h > b.h

    def test_negative_padding_raises(self):
        with pytest.raises(ValueError):
            expand_bbox(_b(), -1)

    def test_clamped_by_max_w(self):
        b = _b(x=0, y=0, w=10, h=10)
        e = expand_bbox(b, 100, max_w=20)
        assert e.x2 <= 20

    def test_clamped_by_max_h(self):
        b = _b(x=0, y=0, w=10, h=10)
        e = expand_bbox(b, 100, max_h=20)
        assert e.y2 <= 20

    def test_left_edge_clamped_to_zero(self):
        b = _b(x=2, y=2, w=5, h=5)
        e = expand_bbox(b, 10)
        assert e.x >= 0 and e.y >= 0


# ─── crop_image ───────────────────────────────────────────────────────────────

class TestCropImageExtra:
    def test_returns_ndarray(self):
        img = _img()
        assert isinstance(crop_image(img, _b(0, 0, 10, 10)), np.ndarray)

    def test_correct_output_shape(self):
        img = _img(64, 64)
        out = crop_image(img, BBox(10, 5, 20, 15))
        assert out.shape == (15, 20)

    def test_values_preserved(self):
        img = np.arange(64, dtype=np.uint8).reshape(8, 8)
        out = crop_image(img, BBox(0, 0, 4, 4))
        np.testing.assert_array_equal(out, img[:4, :4])

    def test_bbox_outside_raises(self):
        img = _img(32, 32)
        with pytest.raises(ValueError):
            crop_image(img, BBox(100, 100, 10, 10))

    def test_3d_image_ok(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        out = crop_image(img, BBox(5, 5, 10, 10))
        assert out.shape == (10, 10, 3)


# ─── bboxes_from_mask ─────────────────────────────────────────────────────────

class TestBboxesFromMaskExtra:
    def _mask_with_blob(self, y=5, x=5, h=10, w=10) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255
        return mask

    def test_returns_list(self):
        mask = self._mask_with_blob()
        assert isinstance(bboxes_from_mask(mask), list)

    def test_detects_single_blob(self):
        mask = self._mask_with_blob()
        bboxes = bboxes_from_mask(mask)
        assert len(bboxes) == 1

    def test_bbox_contains_blob(self):
        mask = self._mask_with_blob(y=10, x=10, h=8, w=8)
        bboxes = bboxes_from_mask(mask)
        b = bboxes[0]
        assert b.x <= 10 and b.y <= 10
        assert b.x2 >= 18 and b.y2 >= 18

    def test_empty_mask_no_bboxes(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        assert bboxes_from_mask(mask) == []

    def test_min_area_filters_small(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:7, 5:7] = 255   # 4 pixels
        mask[20:30, 20:30] = 255  # 100 pixels
        bboxes = bboxes_from_mask(mask, min_area=10)
        assert len(bboxes) == 1

    def test_min_area_lt_1_raises(self):
        with pytest.raises(ValueError):
            bboxes_from_mask(np.zeros((10, 10), dtype=np.uint8), min_area=0)

    def test_sorted_by_area_descending(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:20, 0:20] = 255
        mask[40:45, 40:45] = 255
        bboxes = bboxes_from_mask(mask)
        areas = [b.area for b in bboxes]
        assert areas == sorted(areas, reverse=True)


# ─── merge_overlapping_bboxes ─────────────────────────────────────────────────

class TestMergeOverlappingBboxesExtra:
    def test_returns_list(self):
        assert isinstance(merge_overlapping_bboxes([_b()]), list)

    def test_non_overlapping_unchanged_count(self):
        a = _b(x=0, y=0, w=5, h=5)
        b = _b(x=50, y=50, w=5, h=5)
        result = merge_overlapping_bboxes([a, b], min_iou=0.1)
        assert len(result) == 2

    def test_overlapping_merged(self):
        a = _b(x=0, y=0, w=10, h=10)
        b = _b(x=5, y=5, w=10, h=10)
        result = merge_overlapping_bboxes([a, b])
        assert len(result) == 1

    def test_empty_list(self):
        assert merge_overlapping_bboxes([]) == []

    def test_single_bbox_unchanged(self):
        b = _b(x=5, y=5, w=10, h=10)
        result = merge_overlapping_bboxes([b])
        assert len(result) == 1

    def test_negative_min_iou_raises(self):
        with pytest.raises(ValueError):
            merge_overlapping_bboxes([_b()], min_iou=-0.1)


# ─── bbox_center ──────────────────────────────────────────────────────────────

class TestBboxCenterExtra:
    def test_returns_tuple(self):
        result = bbox_center(_b())
        assert isinstance(result, tuple) and len(result) == 2

    def test_square_center(self):
        b = BBox(x=0, y=0, w=10, h=10)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_rectangle_center(self):
        b = BBox(x=10, y=20, w=6, h=4)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(13.0)
        assert cy == pytest.approx(22.0)

    def test_offset_bbox(self):
        b = BBox(x=100, y=200, w=20, h=40)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(110.0)
        assert cy == pytest.approx(220.0)


# ─── bbox_aspect_ratio ────────────────────────────────────────────────────────

class TestBboxAspectRatioExtra:
    def test_returns_float(self):
        assert isinstance(bbox_aspect_ratio(_b()), float)

    def test_square_is_one(self):
        assert bbox_aspect_ratio(BBox(0, 0, 10, 10)) == pytest.approx(1.0)

    def test_wide_bbox(self):
        b = BBox(0, 0, w=20, h=5)
        ar = bbox_aspect_ratio(b)
        assert ar == pytest.approx(5 / 20)

    def test_tall_bbox(self):
        b = BBox(0, 0, w=5, h=20)
        ar = bbox_aspect_ratio(b)
        assert ar == pytest.approx(5 / 20)

    def test_result_in_range(self):
        b = BBox(0, 0, w=3, h=17)
        ar = bbox_aspect_ratio(b)
        assert 0.0 < ar <= 1.0
