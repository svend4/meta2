"""Extra tests for puzzle_reconstruction/utils/segment_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.segment_utils import (
    SegmentConfig,
    RegionInfo,
    label_mask,
    region_info,
    all_regions,
    filter_regions,
    largest_region,
    mask_from_labels,
    mask_statistics,
    mask_bounding_box,
    extract_boundary,
)


# ─── SegmentConfig ────────────────────────────────────────────────────────────

class TestSegmentConfigExtra:
    def test_defaults(self):
        cfg = SegmentConfig()
        assert cfg.min_area == 50 and cfg.max_aspect_ratio == pytest.approx(10.0)

    def test_negative_min_area_raises(self):
        with pytest.raises(ValueError):
            SegmentConfig(min_area=-1)

    def test_zero_aspect_ratio_raises(self):
        with pytest.raises(ValueError):
            SegmentConfig(max_aspect_ratio=0.0)

    def test_negative_border_margin_raises(self):
        with pytest.raises(ValueError):
            SegmentConfig(border_margin=-1)


# ─── RegionInfo ───────────────────────────────────────────────────────────────

class TestRegionInfoExtra:
    def test_height_and_width(self):
        r = RegionInfo(label=1, area=100, bbox=(10, 20, 30, 50),
                       centroid=np.array([20.0, 35.0]), aspect_ratio=1.5)
        assert r.height == 20 and r.width == 30

    def test_to_dict_keys(self):
        r = RegionInfo(label=1, area=50, bbox=(0, 0, 10, 10),
                       centroid=np.array([5.0, 5.0]), aspect_ratio=1.0)
        d = r.to_dict()
        for k in ("label", "area", "bbox", "centroid", "aspect_ratio"):
            assert k in d


# ─── label_mask ───────────────────────────────────────────────────────────────

class TestLabelMaskExtra:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        labels, n = label_mask(mask)
        assert n == 0 and labels.max() == 0

    def test_single_region(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        labels, n = label_mask(mask)
        assert n == 1

    def test_two_separate_regions(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        mask[10:14, 10:14] = 1
        labels, n = label_mask(mask)
        assert n == 2


# ─── region_info ──────────────────────────────────────────────────────────────

class TestRegionInfoFuncExtra:
    def test_area_matches(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        labels, _ = label_mask(mask)
        ri = region_info(labels, 1)
        assert ri.area == 12  # 3 rows × 4 cols

    def test_zero_area_for_missing_label(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        labels, _ = label_mask(mask)
        ri = region_info(labels, 99)
        assert ri.area == 0


# ─── all_regions ──────────────────────────────────────────────────────────────

class TestAllRegionsExtra:
    def test_returns_list(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        labels, n = label_mask(mask)
        regions = all_regions(labels, n)
        assert len(regions) == n


# ─── filter_regions ───────────────────────────────────────────────────────────

class TestFilterRegionsExtra:
    def test_filters_small_regions(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[1:3, 1:3] = 1    # 4 pixels
        mask[10:20, 10:20] = 1  # 100 pixels
        labels, n = label_mask(mask)
        regions = all_regions(labels, n)
        cfg = SegmentConfig(min_area=10)
        filtered = filter_regions(regions, cfg)
        assert len(filtered) == 1 and filtered[0].area >= 10


# ─── largest_region ───────────────────────────────────────────────────────────

class TestLargestRegionExtra:
    def test_empty_returns_none(self):
        assert largest_region([]) is None

    def test_returns_largest(self):
        r1 = RegionInfo(label=1, area=10, bbox=(0, 0, 5, 5),
                         centroid=np.zeros(2), aspect_ratio=1.0)
        r2 = RegionInfo(label=2, area=100, bbox=(0, 0, 10, 10),
                         centroid=np.zeros(2), aspect_ratio=1.0)
        assert largest_region([r1, r2]).area == 100


# ─── mask_from_labels ─────────────────────────────────────────────────────────

class TestMaskFromLabelsExtra:
    def test_correct_fg(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[7:9, 7:9] = 1
        labels, _ = label_mask(mask)
        result = mask_from_labels(labels, [1])
        assert result.max() == 255
        assert result.sum() > 0


# ─── mask_statistics ──────────────────────────────────────────────────────────

class TestMaskStatisticsExtra:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        s = mask_statistics(mask)
        assert s["foreground_pixels"] == 0
        assert s["foreground_fraction"] == pytest.approx(0.0)

    def test_full_mask(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        s = mask_statistics(mask)
        assert s["foreground_pixels"] == 25
        assert s["foreground_fraction"] == pytest.approx(1.0)

    def test_total_pixels(self):
        mask = np.zeros((4, 6), dtype=np.uint8)
        s = mask_statistics(mask)
        assert s["total_pixels"] == 24


# ─── mask_bounding_box ────────────────────────────────────────────────────────

class TestMaskBoundingBoxExtra:
    def test_all_zero_returns_none(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_bounding_box(mask) is None

    def test_tight_bbox(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        y0, x0, y1, x1 = mask_bounding_box(mask)
        assert (y0, x0, y1, x1) == (2, 3, 5, 7)


# ─── extract_boundary ────────────────────────────────────────────────────────

class TestExtractBoundaryExtra:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        b = extract_boundary(mask)
        assert b.sum() == 0

    def test_single_pixel_is_boundary(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        b = extract_boundary(mask)
        assert b[2, 2] == 255

    def test_solid_block_has_boundary(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        b = extract_boundary(mask)
        # Interior should not be boundary
        assert b[4, 4] == 0
        # Edge should be boundary
        assert b[2, 2] == 255
