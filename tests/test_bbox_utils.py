"""Tests for puzzle_reconstruction.utils.bbox_utils."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.bbox_utils import (
    BBox,
    bbox_aspect_ratio,
    bbox_center,
    bbox_intersection,
    bbox_iou,
    bbox_union,
    bboxes_from_mask,
    crop_image,
    expand_bbox,
    merge_overlapping_bboxes,
)


# ─── BBox ────────────────────────────────────────────────────────────────────

class TestBBox:
    def test_fields_stored(self):
        b = BBox(x=10, y=20, w=30, h=40)
        assert b.x == 10
        assert b.y == 20
        assert b.w == 30
        assert b.h == 40

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=0, h=10)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=-1, h=10)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=10, h=0)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            BBox(x=0, y=0, w=10, h=-5)

    def test_x2_property(self):
        b = BBox(x=5, y=0, w=10, h=1)
        assert b.x2 == 15

    def test_y2_property(self):
        b = BBox(x=0, y=3, w=1, h=7)
        assert b.y2 == 10

    def test_area_property(self):
        b = BBox(x=0, y=0, w=6, h=4)
        assert b.area == 24

    def test_to_tuple(self):
        b = BBox(x=1, y=2, w=3, h=4)
        assert b.to_tuple() == (1, 2, 3, 4)

    def test_negative_coordinates_allowed(self):
        b = BBox(x=-5, y=-3, w=10, h=8)
        assert b.x == -5

    def test_minimum_valid_bbox(self):
        b = BBox(x=0, y=0, w=1, h=1)
        assert b.area == 1


# ─── bbox_iou ────────────────────────────────────────────────────────────────

class TestBboxIou:
    def test_identical_boxes_iou_one(self):
        b = BBox(0, 0, 10, 10)
        assert bbox_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 10, 10)
        assert bbox_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 10, 10)
        # intersection = 5×10=50, union = 10×10+10×10-50 = 150
        assert bbox_iou(a, b) == pytest.approx(50.0 / 150.0)

    def test_symmetric(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        assert bbox_iou(a, b) == pytest.approx(bbox_iou(b, a))

    def test_touching_edges_iou_zero(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 10, 10)
        assert bbox_iou(a, b) == pytest.approx(0.0)

    def test_contained_box(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 10, 10)
        # iou = inner.area / outer.area = 100 / 400 = 0.25
        assert bbox_iou(outer, inner) == pytest.approx(0.25)

    def test_result_in_unit_interval(self):
        a = BBox(0, 0, 15, 15)
        b = BBox(10, 10, 15, 15)
        iou = bbox_iou(a, b)
        assert 0.0 <= iou <= 1.0


# ─── bbox_intersection ───────────────────────────────────────────────────────

class TestBboxIntersection:
    def test_overlapping_returns_bbox(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert isinstance(inter, BBox)

    def test_no_overlap_returns_none(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(10, 10, 5, 5)
        assert bbox_intersection(a, b) is None

    def test_touching_returns_none(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 10, 10)
        assert bbox_intersection(a, b) is None

    def test_correct_coordinates(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        inter = bbox_intersection(a, b)
        assert inter.x == 5
        assert inter.y == 5
        assert inter.w == 5
        assert inter.h == 5

    def test_contained_box(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 10, 10)
        inter = bbox_intersection(outer, inner)
        assert inter is not None
        assert inter.area == inner.area

    def test_symmetric(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(4, 4, 10, 10)
        inter_ab = bbox_intersection(a, b)
        inter_ba = bbox_intersection(b, a)
        assert inter_ab.to_tuple() == inter_ba.to_tuple()


# ─── bbox_union ──────────────────────────────────────────────────────────────

class TestBboxUnion:
    def test_returns_bbox(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        assert isinstance(bbox_union(a, b), BBox)

    def test_contains_both(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(15, 15, 5, 5)
        u = bbox_union(a, b)
        assert u.x <= min(a.x, b.x)
        assert u.y <= min(a.y, b.y)
        assert u.x2 >= max(a.x2, b.x2)
        assert u.y2 >= max(a.y2, b.y2)

    def test_identical_boxes(self):
        b = BBox(3, 4, 5, 6)
        u = bbox_union(b, b)
        assert u.to_tuple() == b.to_tuple()

    def test_correct_extent(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 30, 5, 5)
        u = bbox_union(a, b)
        assert u.x == 0
        assert u.y == 0
        assert u.x2 == 25
        assert u.y2 == 35

    def test_symmetric(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        assert bbox_union(a, b).to_tuple() == bbox_union(b, a).to_tuple()


# ─── expand_bbox ─────────────────────────────────────────────────────────────

class TestExpandBbox:
    def test_expands_by_padding(self):
        b = BBox(10, 10, 20, 20)
        e = expand_bbox(b, padding=5)
        assert e.x == 5
        assert e.y == 5
        assert e.w == 30
        assert e.h == 30

    def test_clamped_to_zero(self):
        b = BBox(2, 2, 5, 5)
        e = expand_bbox(b, padding=10)
        assert e.x == 0
        assert e.y == 0

    def test_clamped_to_max_dims(self):
        b = BBox(5, 5, 10, 10)
        e = expand_bbox(b, padding=20, max_h=20, max_w=20)
        assert e.x2 <= 20
        assert e.y2 <= 20

    def test_zero_padding_unchanged(self):
        b = BBox(10, 10, 20, 20)
        e = expand_bbox(b, padding=0)
        assert e.to_tuple() == b.to_tuple()

    def test_negative_padding_raises(self):
        b = BBox(5, 5, 10, 10)
        with pytest.raises(ValueError):
            expand_bbox(b, padding=-1)

    def test_min_width_one(self):
        b = BBox(5, 5, 10, 10)
        e = expand_bbox(b, padding=3, max_w=6, max_h=100)
        assert e.w >= 1


# ─── crop_image ──────────────────────────────────────────────────────────────

class TestCropImage:
    def _img(self) -> np.ndarray:
        img = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        return img

    def test_returns_array(self):
        assert isinstance(crop_image(self._img(), BBox(10, 10, 20, 20)), np.ndarray)

    def test_shape_matches_bbox(self):
        cropped = crop_image(self._img(), BBox(5, 5, 30, 25))
        assert cropped.shape == (25, 30)

    def test_values_match(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:20, 10:20] = 99
        cropped = crop_image(img, BBox(10, 10, 10, 10))
        assert np.all(cropped == 99)

    def test_bgr_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cropped = crop_image(img, BBox(5, 5, 20, 20))
        assert cropped.shape == (20, 20, 3)

    def test_out_of_bounds_raises(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            crop_image(img, BBox(20, 20, 5, 5))

    def test_clamped_to_image_boundary(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        cropped = crop_image(img, BBox(-5, -5, 20, 20))
        assert cropped.shape[0] >= 1
        assert cropped.shape[1] >= 1


# ─── bboxes_from_mask ────────────────────────────────────────────────────────

class TestBboxesFromMask:
    def _mask_two_blobs(self) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:20, 5:20] = 255
        mask[40:55, 40:55] = 255
        return mask

    def test_returns_list(self):
        assert isinstance(bboxes_from_mask(self._mask_two_blobs()), list)

    def test_two_blobs_two_bboxes(self):
        result = bboxes_from_mask(self._mask_two_blobs())
        assert len(result) == 2

    def test_blank_mask_empty(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        assert bboxes_from_mask(mask) == []

    def test_all_bboxes_are_bbox(self):
        result = bboxes_from_mask(self._mask_two_blobs())
        assert all(isinstance(b, BBox) for b in result)

    def test_sorted_by_area_descending(self):
        result = bboxes_from_mask(self._mask_two_blobs())
        if len(result) >= 2:
            assert result[0].area >= result[1].area

    def test_min_area_filters_small(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:10, 5:10] = 255  # 5×5=25 px
        mask[30:50, 30:50] = 255  # 20×20=400 px
        result = bboxes_from_mask(mask, min_area=100)
        assert len(result) == 1

    def test_min_area_zero_raises(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            bboxes_from_mask(mask, min_area=0)

    def test_bbox_coordinates_within_mask(self):
        mask = self._mask_two_blobs()
        h, w = mask.shape
        for b in bboxes_from_mask(mask):
            assert 0 <= b.x < w
            assert 0 <= b.y < h
            assert b.x2 <= w
            assert b.y2 <= h


# ─── merge_overlapping_bboxes ────────────────────────────────────────────────

class TestMergeOverlappingBboxes:
    def test_empty_list_returns_empty(self):
        assert merge_overlapping_bboxes([]) == []

    def test_single_box_unchanged(self):
        b = BBox(0, 0, 10, 10)
        result = merge_overlapping_bboxes([b])
        assert len(result) == 1
        assert result[0].to_tuple() == b.to_tuple()

    def test_non_overlapping_unchanged(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(20, 20, 5, 5)
        result = merge_overlapping_bboxes([a, b])
        assert len(result) == 2

    def test_overlapping_merged(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        result = merge_overlapping_bboxes([a, b])
        assert len(result) == 1

    def test_merged_contains_both(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        merged = merge_overlapping_bboxes([a, b])[0]
        assert merged.x <= 0
        assert merged.y <= 0
        assert merged.x2 >= 15
        assert merged.y2 >= 15

    def test_negative_min_iou_raises(self):
        with pytest.raises(ValueError):
            merge_overlapping_bboxes([BBox(0, 0, 5, 5)], min_iou=-0.1)

    def test_high_min_iou_keeps_separate(self):
        # boxes barely overlap — with high threshold not merged
        a = BBox(0, 0, 10, 10)
        b = BBox(9, 9, 10, 10)
        # IoU is tiny; min_iou=0.5 should keep them separate
        result = merge_overlapping_bboxes([a, b], min_iou=0.5)
        assert len(result) == 2


# ─── bbox_center ─────────────────────────────────────────────────────────────

class TestBboxCenter:
    def test_square_box(self):
        b = BBox(0, 0, 10, 10)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_offset_box(self):
        b = BBox(10, 20, 4, 6)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(12.0)
        assert cy == pytest.approx(23.0)

    def test_returns_tuple(self):
        b = BBox(0, 0, 5, 5)
        result = bbox_center(b)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─── bbox_aspect_ratio ───────────────────────────────────────────────────────

class TestBboxAspectRatio:
    def test_square_is_one(self):
        b = BBox(0, 0, 10, 10)
        assert bbox_aspect_ratio(b) == pytest.approx(1.0)

    def test_tall_box(self):
        b = BBox(0, 0, 5, 20)
        assert bbox_aspect_ratio(b) == pytest.approx(0.25)

    def test_wide_box(self):
        b = BBox(0, 0, 20, 5)
        assert bbox_aspect_ratio(b) == pytest.approx(0.25)

    def test_result_in_unit_interval(self):
        b = BBox(0, 0, 3, 17)
        ar = bbox_aspect_ratio(b)
        assert 0.0 < ar <= 1.0

    def test_minimum_one_pixel_side(self):
        b = BBox(0, 0, 1, 100)
        ar = bbox_aspect_ratio(b)
        assert ar == pytest.approx(0.01)
