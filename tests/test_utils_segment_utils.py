"""Tests for puzzle_reconstruction.utils.segment_utils"""
import pytest
import numpy as np

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


# ── SegmentConfig ─────────────────────────────────────────────────────────────

def test_segment_config_defaults():
    cfg = SegmentConfig()
    assert cfg.min_area == 50
    assert cfg.max_aspect_ratio == 10.0
    assert cfg.border_margin == 2


def test_segment_config_negative_min_area_raises():
    with pytest.raises(ValueError):
        SegmentConfig(min_area=-1)


def test_segment_config_zero_max_aspect_raises():
    with pytest.raises(ValueError):
        SegmentConfig(max_aspect_ratio=0)


def test_segment_config_negative_max_aspect_raises():
    with pytest.raises(ValueError):
        SegmentConfig(max_aspect_ratio=-1.0)


def test_segment_config_negative_border_margin_raises():
    with pytest.raises(ValueError):
        SegmentConfig(border_margin=-1)


# ── label_mask ────────────────────────────────────────────────────────────────

def test_label_mask_single_region():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    labels, n = label_mask(mask)
    assert n == 1


def test_label_mask_two_regions():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:5, 1:5] = 255    # region 1
    mask[2:5, 15:19] = 255  # region 2
    labels, n = label_mask(mask)
    assert n == 2


def test_label_mask_empty_returns_zero_labels():
    mask = np.zeros((10, 10), dtype=np.uint8)
    labels, n = label_mask(mask)
    assert n == 0
    assert (labels == 0).all()


def test_label_mask_full_returns_one_label():
    mask = np.ones((5, 5), dtype=np.uint8) * 255
    labels, n = label_mask(mask)
    assert n == 1


def test_label_mask_output_shape():
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[1:4, 1:4] = 255
    labels, n = label_mask(mask)
    assert labels.shape == (7, 9)


def test_label_mask_background_is_zero():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255
    labels, n = label_mask(mask)
    assert labels[0, 0] == 0


def test_label_mask_bool_mask_works():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    labels, n = label_mask(mask)
    assert n == 1


# ── region_info ───────────────────────────────────────────────────────────────

def test_region_info_area():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 2:7] = 255
    labels, _ = label_mask(mask)
    ri = region_info(labels, 1)
    assert ri.area == 25


def test_region_info_bbox():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:8] = 255
    labels, _ = label_mask(mask)
    ri = region_info(labels, 1)
    y0, x0, y1, x1 = ri.bbox
    assert y0 == 2
    assert x0 == 3
    assert y1 == 5
    assert x1 == 8


def test_region_info_centroid_shape():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255
    labels, _ = label_mask(mask)
    ri = region_info(labels, 1)
    assert ri.centroid.shape == (2,)


def test_region_info_empty_label():
    labels = np.zeros((10, 10), dtype=np.int32)
    ri = region_info(labels, 99)
    assert ri.area == 0
    assert ri.bbox == (0, 0, 0, 0)


def test_region_info_height_width():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 3:8] = 255  # h=5, w=5
    labels, _ = label_mask(mask)
    ri = region_info(labels, 1)
    assert ri.height == 5
    assert ri.width == 5


def test_region_info_aspect_ratio():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:5, 0:20] = 255  # very wide: h=5, w=20 -> ratio=4
    labels, _ = label_mask(mask)
    ri = region_info(labels, 1)
    assert ri.aspect_ratio == pytest.approx(4.0)


def test_region_info_to_dict():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    labels, _ = label_mask(mask)
    ri = region_info(labels, 1)
    d = ri.to_dict()
    assert "label" in d
    assert "area" in d
    assert "bbox" in d
    assert "centroid" in d


# ── all_regions ───────────────────────────────────────────────────────────────

def test_all_regions_count():
    mask = np.zeros((10, 30), dtype=np.uint8)
    mask[1:4, 1:5] = 255
    mask[1:4, 10:14] = 255
    mask[1:4, 20:24] = 255
    labels, n = label_mask(mask)
    regions = all_regions(labels, n)
    assert len(regions) == 3


def test_all_regions_empty():
    labels = np.zeros((10, 10), dtype=np.int32)
    regions = all_regions(labels, 0)
    assert regions == []


# ── filter_regions ────────────────────────────────────────────────────────────

def test_filter_regions_by_area():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[1:3, 1:3] = 255     # small: 4 pixels
    mask[5:15, 5:20] = 255   # large: 150 pixels
    labels, n = label_mask(mask)
    regions = all_regions(labels, n)
    cfg = SegmentConfig(min_area=10, max_aspect_ratio=100.0)
    filtered = filter_regions(regions, cfg)
    assert len(filtered) == 1
    assert filtered[0].area >= 10


def test_filter_regions_by_aspect():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:20] = 255   # very wide sliver
    mask[5:15, 5:15] = 255  # square
    labels, n = label_mask(mask)
    regions = all_regions(labels, n)
    cfg = SegmentConfig(min_area=0, max_aspect_ratio=3.0)
    filtered = filter_regions(regions, cfg)
    # The square (ratio ~1) should pass; the sliver (ratio=10) may not
    assert all(r.aspect_ratio <= 3.0 for r in filtered)


def test_filter_regions_default_config():
    # Default min_area=50; use 10x10 region (100 pixels) to ensure it passes
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[2:12, 2:12] = 255   # 10x10 = 100 pixels > min_area=50
    labels, n = label_mask(mask)
    regions = all_regions(labels, n)
    filtered = filter_regions(regions)
    assert len(filtered) == 1


# ── largest_region ────────────────────────────────────────────────────────────

def test_largest_region_basic():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[1:3, 1:3] = 255     # small
    mask[5:15, 5:20] = 255   # large
    labels, n = label_mask(mask)
    regions = all_regions(labels, n)
    lr = largest_region(regions)
    assert lr is not None
    assert lr.area >= 4


def test_largest_region_empty():
    assert largest_region([]) is None


def test_largest_region_single():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:5, 1:5] = 255
    labels, n = label_mask(mask)
    regions = all_regions(labels, n)
    lr = largest_region(regions)
    assert lr is not None


# ── mask_from_labels ──────────────────────────────────────────────────────────

def test_mask_from_labels_basic():
    mask_in = np.zeros((10, 10), dtype=np.uint8)
    mask_in[1:4, 1:4] = 255
    mask_in[6:9, 6:9] = 255
    labels, n = label_mask(mask_in)
    mask_out = mask_from_labels(labels, [1])
    assert mask_out.dtype == np.uint8
    assert mask_out.max() == 255


def test_mask_from_labels_empty_keep_ids():
    labels = np.ones((5, 5), dtype=np.int32)
    mask_out = mask_from_labels(labels, [])
    assert (mask_out == 0).all()


def test_mask_from_labels_shape():
    labels = np.zeros((7, 9), dtype=np.int32)
    mask_out = mask_from_labels(labels, [])
    assert mask_out.shape == (7, 9)


# ── mask_statistics ───────────────────────────────────────────────────────────

def test_mask_statistics_full():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    stats = mask_statistics(mask)
    assert stats["foreground_pixels"] == 100
    assert stats["background_pixels"] == 0
    assert stats["foreground_fraction"] == pytest.approx(1.0)
    assert stats["total_pixels"] == 100


def test_mask_statistics_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    stats = mask_statistics(mask)
    assert stats["foreground_pixels"] == 0
    assert stats["foreground_fraction"] == pytest.approx(0.0)


def test_mask_statistics_partial():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 255
    stats = mask_statistics(mask)
    assert stats["foreground_fraction"] == pytest.approx(0.5)


# ── mask_bounding_box ─────────────────────────────────────────────────────────

def test_mask_bounding_box_basic():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 3:15] = 255
    bb = mask_bounding_box(mask)
    assert bb is not None
    y0, x0, y1, x1 = bb
    assert y0 == 5
    assert x0 == 3
    assert y1 == 10
    assert x1 == 15


def test_mask_bounding_box_empty_returns_none():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert mask_bounding_box(mask) is None


def test_mask_bounding_box_full_mask():
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    bb = mask_bounding_box(mask)
    assert bb == (0, 0, 8, 8)


# ── extract_boundary ──────────────────────────────────────────────────────────

def test_extract_boundary_shape():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    boundary = extract_boundary(mask)
    assert boundary.shape == (20, 20)
    assert boundary.dtype == np.uint8


def test_extract_boundary_values():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    boundary = extract_boundary(mask)
    # Boundary should be 0 or 255
    assert set(np.unique(boundary)).issubset({0, 255})


def test_extract_boundary_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    boundary = extract_boundary(mask)
    assert (boundary == 0).all()


def test_extract_boundary_interior_not_boundary():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, 3:17] = 255
    boundary = extract_boundary(mask)
    # Center pixel should NOT be boundary
    assert boundary[10, 10] == 0
