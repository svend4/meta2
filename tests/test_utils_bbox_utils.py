"""Tests for puzzle_reconstruction/utils/bbox_utils.py"""
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


# ─── BBox ─────────────────────────────────────────────────────────────────────

class TestBBox:
    def test_basic_creation(self):
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
        b = BBox(x=5, y=3, w=10, h=8)
        assert b.x2 == 15

    def test_y2_property(self):
        b = BBox(x=5, y=3, w=10, h=8)
        assert b.y2 == 11

    def test_area_property(self):
        b = BBox(x=0, y=0, w=5, h=4)
        assert b.area == 20

    def test_to_tuple(self):
        b = BBox(x=1, y=2, w=3, h=4)
        assert b.to_tuple() == (1, 2, 3, 4)

    def test_negative_coords_allowed(self):
        b = BBox(x=-10, y=-5, w=20, h=15)
        assert b.x == -10

    def test_width_one_ok(self):
        b = BBox(x=0, y=0, w=1, h=1)
        assert b.area == 1


# ─── bbox_iou ─────────────────────────────────────────────────────────────────

class TestBboxIou:
    def test_identical_boxes(self):
        b = BBox(0, 0, 10, 10)
        assert bbox_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 10, 10)
        assert bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        iou = bbox_iou(a, b)
        # inter = 5*5 = 25, union = 100 + 100 - 25 = 175
        assert iou == pytest.approx(25.0 / 175.0)

    def test_contained_box(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 5, 5)
        iou = bbox_iou(outer, inner)
        # inter = 25, union = 400 + 25 - 25 = 400
        assert iou == pytest.approx(25.0 / 400.0)

    def test_iou_in_range(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(3, 3, 10, 10)
        iou = bbox_iou(a, b)
        assert 0.0 <= iou <= 1.0

    def test_touching_edge_no_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 10, 10)
        # They touch but don't overlap (ix2 == ix → 10 <= 10 is not strictly >)
        assert bbox_iou(a, b) == pytest.approx(0.0)


# ─── bbox_intersection ────────────────────────────────────────────────────────

class TestBboxIntersection:
    def test_no_intersection(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(10, 10, 5, 5)
        assert bbox_intersection(a, b) is None

    def test_partial_intersection(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        result = bbox_intersection(a, b)
        assert result is not None
        assert result.x == 5
        assert result.y == 5
        assert result.w == 5
        assert result.h == 5

    def test_identical_intersection(self):
        b = BBox(2, 3, 8, 6)
        result = bbox_intersection(b, b)
        assert result is not None
        assert result.w == 8
        assert result.h == 6

    def test_touching_edge_no_intersection(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(5, 0, 5, 5)
        assert bbox_intersection(a, b) is None

    def test_contained_intersection(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 5, 5)
        result = bbox_intersection(outer, inner)
        assert result is not None
        assert result.w == 5
        assert result.h == 5


# ─── bbox_union ───────────────────────────────────────────────────────────────

class TestBboxUnion:
    def test_identical_union(self):
        b = BBox(2, 3, 8, 6)
        result = bbox_union(b, b)
        assert result.x == 2
        assert result.y == 3
        assert result.w == 8
        assert result.h == 6

    def test_disjoint_union(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(10, 10, 5, 5)
        result = bbox_union(a, b)
        assert result.x == 0
        assert result.y == 0
        assert result.x2 == 15
        assert result.y2 == 15

    def test_union_area_ge_both(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        result = bbox_union(a, b)
        assert result.area >= a.area
        assert result.area >= b.area

    def test_contained_union(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 5, 5)
        result = bbox_union(outer, inner)
        assert result.w == 20
        assert result.h == 20


# ─── expand_bbox ──────────────────────────────────────────────────────────────

class TestExpandBbox:
    def test_zero_padding(self):
        b = BBox(5, 5, 10, 10)
        result = expand_bbox(b, 0)
        assert result.x == 5
        assert result.y == 5
        assert result.w == 10
        assert result.h == 10

    def test_expand_by_padding(self):
        b = BBox(10, 10, 5, 5)
        result = expand_bbox(b, 2)
        assert result.x == 8
        assert result.y == 8
        assert result.w == 9
        assert result.h == 9

    def test_clamp_to_zero(self):
        b = BBox(2, 2, 5, 5)
        result = expand_bbox(b, 10)
        assert result.x == 0
        assert result.y == 0

    def test_negative_padding_raises(self):
        b = BBox(5, 5, 10, 10)
        with pytest.raises(ValueError):
            expand_bbox(b, -1)

    def test_max_w_limits(self):
        b = BBox(5, 5, 10, 10)
        result = expand_bbox(b, 3, max_w=16)
        assert result.x2 <= 16

    def test_max_h_limits(self):
        b = BBox(5, 5, 10, 10)
        result = expand_bbox(b, 3, max_h=14)
        assert result.y2 <= 14

    def test_no_limit_when_zero(self):
        b = BBox(5, 5, 10, 10)
        result = expand_bbox(b, 5, max_w=0, max_h=0)
        assert result.x2 == 20
        assert result.y2 == 20

    def test_minimum_size_one(self):
        b = BBox(1, 1, 2, 2)
        result = expand_bbox(b, 0, max_w=1, max_h=1)
        assert result.w >= 1
        assert result.h >= 1


# ─── crop_image ───────────────────────────────────────────────────────────────

class TestCropImage:
    def test_basic_crop(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        b = BBox(10, 20, 30, 40)
        crop = crop_image(img, b)
        assert crop.shape == (40, 30)

    def test_bgr_crop(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b = BBox(5, 5, 20, 20)
        crop = crop_image(img, b)
        assert crop.shape == (20, 20, 3)

    def test_outside_image_raises(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        b = BBox(100, 100, 10, 10)
        with pytest.raises(ValueError):
            crop_image(img, b)

    def test_negative_x_clamps(self):
        img = np.ones((50, 50), dtype=np.uint8) * 100
        b = BBox(-5, -5, 20, 20)
        crop = crop_image(img, b)
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0

    def test_crop_values_preserved(self):
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        b = BBox(2, 3, 4, 2)
        crop = crop_image(img, b)
        np.testing.assert_array_equal(crop, img[3:5, 2:6])

    def test_partial_overlap_clamps(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        b = BBox(15, 15, 20, 20)  # extends beyond image
        crop = crop_image(img, b)
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0


# ─── bboxes_from_mask ─────────────────────────────────────────────────────────

class TestBboxesFromMask:
    def make_mask_with_two_blobs(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Large blob
        mask[10:50, 10:50] = 255
        # Small blob
        mask[60:70, 60:70] = 255
        return mask

    def test_returns_list(self):
        mask = self.make_mask_with_two_blobs()
        result = bboxes_from_mask(mask)
        assert isinstance(result, list)

    def test_two_components_found(self):
        mask = self.make_mask_with_two_blobs()
        result = bboxes_from_mask(mask)
        assert len(result) == 2

    def test_sorted_by_area_desc(self):
        mask = self.make_mask_with_two_blobs()
        result = bboxes_from_mask(mask)
        assert result[0].area >= result[1].area

    def test_all_bbox_instances(self):
        mask = self.make_mask_with_two_blobs()
        result = bboxes_from_mask(mask)
        assert all(isinstance(b, BBox) for b in result)

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = bboxes_from_mask(mask)
        assert result == []

    def test_min_area_filter(self):
        mask = self.make_mask_with_two_blobs()
        result = bboxes_from_mask(mask, min_area=500)
        assert len(result) == 1

    def test_min_area_zero_raises(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        with pytest.raises(ValueError):
            bboxes_from_mask(mask, min_area=0)

    def test_min_area_negative_raises(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        with pytest.raises(ValueError):
            bboxes_from_mask(mask, min_area=-1)

    def test_single_component(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[5:15, 5:25] = 255
        result = bboxes_from_mask(mask)
        assert len(result) == 1


# ─── merge_overlapping_bboxes ─────────────────────────────────────────────────

class TestMergeOverlappingBboxes:
    def test_empty_list(self):
        result = merge_overlapping_bboxes([])
        assert result == []

    def test_single_bbox(self):
        b = BBox(0, 0, 10, 10)
        result = merge_overlapping_bboxes([b])
        assert len(result) == 1

    def test_overlapping_boxes_merged(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        result = merge_overlapping_bboxes([a, b])
        assert len(result) == 1

    def test_disjoint_boxes_not_merged(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(20, 20, 5, 5)
        result = merge_overlapping_bboxes([a, b])
        assert len(result) == 2

    def test_negative_min_iou_raises(self):
        with pytest.raises(ValueError):
            merge_overlapping_bboxes([BBox(0, 0, 5, 5)], min_iou=-0.1)

    def test_high_min_iou_prevents_merge(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(8, 8, 10, 10)
        # small overlap → IoU < 0.9
        result = merge_overlapping_bboxes([a, b], min_iou=0.9)
        assert len(result) == 2

    def test_three_boxes_chain_merge(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 10, 10)
        c = BBox(10, 0, 10, 10)
        result = merge_overlapping_bboxes([a, b, c])
        assert len(result) == 1


# ─── bbox_center ──────────────────────────────────────────────────────────────

class TestBboxCenter:
    def test_square_center(self):
        b = BBox(0, 0, 10, 10)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_offset_box(self):
        b = BBox(4, 6, 8, 4)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(8.0)
        assert cy == pytest.approx(8.0)

    def test_unit_box(self):
        b = BBox(3, 7, 1, 1)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(3.5)
        assert cy == pytest.approx(7.5)

    def test_returns_tuple(self):
        b = BBox(0, 0, 4, 2)
        result = bbox_center(b)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─── bbox_aspect_ratio ────────────────────────────────────────────────────────

class TestBboxAspectRatio:
    def test_square_ratio_one(self):
        b = BBox(0, 0, 10, 10)
        assert bbox_aspect_ratio(b) == pytest.approx(1.0)

    def test_wide_box(self):
        b = BBox(0, 0, 20, 10)
        assert bbox_aspect_ratio(b) == pytest.approx(0.5)

    def test_tall_box(self):
        b = BBox(0, 0, 5, 20)
        assert bbox_aspect_ratio(b) == pytest.approx(0.25)

    def test_ratio_in_range(self):
        b = BBox(0, 0, 7, 13)
        ratio = bbox_aspect_ratio(b)
        assert 0.0 < ratio <= 1.0

    def test_ratio_symmetric(self):
        b1 = BBox(0, 0, 4, 8)
        b2 = BBox(0, 0, 8, 4)
        assert bbox_aspect_ratio(b1) == pytest.approx(bbox_aspect_ratio(b2))
