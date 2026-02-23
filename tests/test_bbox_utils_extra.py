"""Extra tests for puzzle_reconstruction.utils.bbox_utils."""
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


# ─── TestBBoxExtra ────────────────────────────────────────────────────────────

class TestBBoxExtra:
    def test_large_coords(self):
        b = BBox(x=1000, y=2000, w=500, h=300)
        assert b.x == 1000 and b.y == 2000

    def test_large_area(self):
        b = BBox(x=0, y=0, w=1000, h=1000)
        assert b.area == 1_000_000

    def test_x2_correct_for_large(self):
        b = BBox(x=100, y=200, w=50, h=75)
        assert b.x2 == 150
        assert b.y2 == 275

    def test_to_tuple_all_fields(self):
        b = BBox(x=3, y=7, w=11, h=13)
        assert b.to_tuple() == (3, 7, 11, 13)

    def test_1x1_bbox(self):
        b = BBox(x=5, y=5, w=1, h=1)
        assert b.area == 1

    def test_non_square_area(self):
        b = BBox(x=0, y=0, w=3, h=7)
        assert b.area == 21

    def test_float_coords(self):
        b = BBox(x=1.5, y=2.5, w=3.0, h=4.0)
        assert b.x == pytest.approx(1.5)


# ─── TestBboxIouExtra ─────────────────────────────────────────────────────────

class TestBboxIouExtra:
    def test_small_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(9, 9, 10, 10)
        iou = bbox_iou(a, b)
        assert 0.0 < iou < 0.1

    def test_many_combinations_in_range(self):
        boxes = [BBox(i * 5, 0, 10, 10) for i in range(5)]
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                iou = bbox_iou(boxes[i], boxes[j])
                assert 0.0 <= iou <= 1.0

    def test_full_containment_iou(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 10, 10)
        # iou = inner.area / outer.area = 100/400
        assert bbox_iou(outer, inner) == pytest.approx(0.25)

    def test_partial_overlap_in_range(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 10, 10)
        iou = bbox_iou(a, b)
        assert 0.0 < iou < 1.0


# ─── TestBboxIntersectionExtra ────────────────────────────────────────────────

class TestBboxIntersectionExtra:
    def test_partial_overlap_area(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 10, 10)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert inter.area == pytest.approx(50.0)

    def test_large_boxes_overlap(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 50, 100, 100)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert inter.area == pytest.approx(2500.0)

    def test_fully_inside_returns_inner(self):
        outer = BBox(0, 0, 30, 30)
        inner = BBox(5, 5, 10, 10)
        inter = bbox_intersection(outer, inner)
        assert inter is not None
        assert inter.area == inner.area

    def test_touching_on_corner_none(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(5, 5, 5, 5)
        assert bbox_intersection(a, b) is None

    def test_result_dimensions_correct(self):
        a = BBox(2, 3, 8, 7)
        b = BBox(6, 5, 8, 7)
        inter = bbox_intersection(a, b)
        assert inter is not None
        assert inter.x == 6
        assert inter.y == 5
        assert inter.w == pytest.approx(4.0)
        assert inter.h == pytest.approx(5.0)


# ─── TestBboxUnionExtra ───────────────────────────────────────────────────────

class TestBboxUnionExtra:
    def test_large_boxes_union(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(200, 200, 100, 100)
        u = bbox_union(a, b)
        assert u.x == 0 and u.y == 0
        assert u.x2 == 300 and u.y2 == 300

    def test_negative_coords_union(self):
        a = BBox(-10, -10, 20, 20)
        b = BBox(5, 5, 15, 15)
        u = bbox_union(a, b)
        assert u.x == -10
        assert u.y == -10

    def test_union_area_geq_both(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(3, 3, 10, 10)
        u = bbox_union(a, b)
        assert u.area >= max(a.area, b.area)

    def test_disjoint_union_area(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(10, 10, 5, 5)
        u = bbox_union(a, b)
        assert u.w == 15
        assert u.h == 15


# ─── TestExpandBboxExtra ──────────────────────────────────────────────────────

class TestExpandBboxExtra:
    def test_1_pixel_bbox_expands(self):
        b = BBox(10, 10, 1, 1)
        e = expand_bbox(b, padding=5)
        assert e.x == 5 and e.y == 5
        assert e.w == 11 and e.h == 11

    def test_large_image_bounds_ok(self):
        b = BBox(50, 50, 20, 20)
        e = expand_bbox(b, padding=10, max_w=1000, max_h=1000)
        assert e.x == 40 and e.y == 40
        assert e.w == 40 and e.h == 40

    def test_max_wh_exactly_at_boundary(self):
        b = BBox(0, 0, 10, 10)
        e = expand_bbox(b, padding=5, max_w=15, max_h=15)
        assert e.x2 <= 15
        assert e.y2 <= 15

    def test_padding_5_on_interior_bbox(self):
        b = BBox(20, 20, 10, 10)
        e = expand_bbox(b, padding=5, max_w=100, max_h=100)
        assert e.x == 15
        assert e.y == 15
        assert e.w == 20
        assert e.h == 20


# ─── TestCropImageExtra ───────────────────────────────────────────────────────

class TestCropImageExtra:
    def _img(self):
        return np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)

    def test_1x1_crop(self):
        img = np.ones((10, 10), dtype=np.uint8) * 42
        cropped = crop_image(img, BBox(5, 5, 1, 1))
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == 42

    def test_large_crop(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cropped = crop_image(img, BBox(0, 0, 150, 150))
        assert cropped.shape == (150, 150)

    def test_rgb_shape_correct(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cropped = crop_image(img, BBox(10, 10, 15, 20))
        assert cropped.shape == (20, 15, 3)

    def test_values_preserved(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[5:10, 5:10] = 77
        cropped = crop_image(img, BBox(5, 5, 5, 5))
        assert np.all(cropped == 77)

    def test_negative_corner_clamps(self):
        img = self._img()
        cropped = crop_image(img, BBox(-2, -2, 10, 10))
        assert cropped.shape[0] >= 1


# ─── TestBboxesFromMaskExtra ─────────────────────────────────────────────────

class TestBboxesFromMaskExtra:
    def test_single_blob(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:20, 5:20] = 255
        result = bboxes_from_mask(mask)
        assert len(result) == 1

    def test_large_mask(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:100, 10:100] = 255
        result = bboxes_from_mask(mask)
        assert len(result) == 1

    def test_bbox_width_height_positive(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:20, 5:20] = 255
        result = bboxes_from_mask(mask)
        assert result[0].w > 0 and result[0].h > 0

    def test_three_separate_blobs(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        mask[40:50, 40:50] = 255
        mask[70:80, 70:80] = 255
        result = bboxes_from_mask(mask)
        assert len(result) == 3

    def test_min_area_large_filters_all(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:10, 5:10] = 255  # small blob
        result = bboxes_from_mask(mask, min_area=10000)
        assert result == []


# ─── TestMergeOverlappingBboxesExtra ─────────────────────────────────────────

class TestMergeOverlappingBboxesExtra:
    def test_three_overlapping_to_one(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 10, 10)
        c = BBox(10, 0, 10, 10)
        result = merge_overlapping_bboxes([a, b, c])
        assert len(result) <= 3

    def test_three_separate(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(20, 20, 5, 5)
        c = BBox(40, 40, 5, 5)
        result = merge_overlapping_bboxes([a, b, c])
        assert len(result) == 3

    def test_all_same_bbox_merged(self):
        b = BBox(0, 0, 10, 10)
        result = merge_overlapping_bboxes([b, b, b])
        assert len(result) == 1

    def test_min_iou_zero_merges_overlapping(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        result = merge_overlapping_bboxes([a, b], min_iou=0.0)
        assert len(result) == 1

    def test_result_all_bbox_type(self):
        boxes = [BBox(i * 3, 0, 5, 5) for i in range(4)]
        result = merge_overlapping_bboxes(boxes)
        for b in result:
            assert isinstance(b, BBox)


# ─── TestBboxCenterExtra ──────────────────────────────────────────────────────

class TestBboxCenterExtra:
    def test_non_square_box(self):
        b = BBox(0, 0, 20, 10)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(10.0)
        assert cy == pytest.approx(5.0)

    def test_large_box(self):
        b = BBox(100, 200, 400, 300)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(300.0)
        assert cy == pytest.approx(350.0)

    def test_returns_floats(self):
        b = BBox(1, 1, 3, 3)
        cx, cy = bbox_center(b)
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    def test_negative_coords(self):
        b = BBox(-10, -20, 20, 40)
        cx, cy = bbox_center(b)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)


# ─── TestBboxAspectRatioExtra ─────────────────────────────────────────────────

class TestBboxAspectRatioExtra:
    def test_1x1_is_one(self):
        b = BBox(0, 0, 1, 1)
        assert bbox_aspect_ratio(b) == pytest.approx(1.0)

    def test_2_by_3_ratio(self):
        b = BBox(0, 0, 2, 3)
        assert bbox_aspect_ratio(b) == pytest.approx(2.0 / 3.0)

    def test_all_values_in_01(self):
        sizes = [(1, 100), (10, 50), (50, 50), (100, 1)]
        for w, h in sizes:
            ar = bbox_aspect_ratio(BBox(0, 0, w, h))
            assert 0.0 < ar <= 1.0

    def test_symmetric_tall_wide(self):
        tall = BBox(0, 0, 4, 8)
        wide = BBox(0, 0, 8, 4)
        # Both should give same aspect ratio (min/max)
        assert bbox_aspect_ratio(tall) == pytest.approx(bbox_aspect_ratio(wide))
