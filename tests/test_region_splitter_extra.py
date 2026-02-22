"""Extra tests for puzzle_reconstruction.algorithms.region_splitter
(supplementing test_region_splitter.py)."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.region_splitter import (
    RegionInfo,
    SplitResult,
    batch_find_regions,
    filter_regions,
    find_regions,
    largest_region,
    merge_small_regions,
    region_masks,
    split_mask_to_crops,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=32, w=32):
    return np.zeros((h, w), dtype=np.uint8)


def _two_blob_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 5:15] = 255    # 100 px
    mask[40:55, 40:55] = 255  # 225 px
    return mask


def _three_blob_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[2:7, 2:7] = 255      # 25 px
    mask[20:30, 20:30] = 255  # 100 px
    mask[40:55, 40:55] = 255  # 225 px
    return mask


def _make_region(label=1, area=50):
    return RegionInfo(
        label=label, area=area,
        bbox=(0, 0, 10, 10),
        centroid=(5.0, 5.0),
        mask=np.zeros((10, 10), dtype=np.uint8),
    )


# ─── RegionInfo extras ────────────────────────────────────────────────────────

class TestRegionInfoExtra2:
    def test_zero_label_valid(self):
        r = _make_region(label=0)
        assert r.label == 0

    def test_large_label(self):
        r = _make_region(label=9999)
        assert r.label == 9999

    def test_large_area(self):
        r = _make_region(area=100_000)
        assert r.area == 100_000
        assert len(r) == 100_000

    def test_centroid_stored(self):
        r = RegionInfo(label=1, area=10, bbox=(0, 0, 5, 5),
                       centroid=(2.5, 3.7), mask=np.zeros((5, 5), dtype=np.uint8))
        assert r.centroid[0] == pytest.approx(2.5)
        assert r.centroid[1] == pytest.approx(3.7)

    def test_bbox_stored(self):
        r = RegionInfo(label=1, area=10, bbox=(3, 4, 9, 11),
                       centroid=(6.0, 7.5), mask=np.zeros((6, 7), dtype=np.uint8))
        assert r.bbox == (3, 4, 9, 11)

    def test_mask_shape_stored(self):
        m = np.zeros((8, 12), dtype=np.uint8)
        r = RegionInfo(label=1, area=0, bbox=(0, 0, 8, 12),
                       centroid=(4.0, 6.0), mask=m)
        assert r.mask.shape == (8, 12)


# ─── SplitResult extras ───────────────────────────────────────────────────────

class TestSplitResultExtra2:
    def test_multiple_regions(self):
        lmap = np.zeros((16, 16), dtype=np.int32)
        regions = [_make_region(label=i + 1) for i in range(5)]
        sr = SplitResult(regions=regions, label_map=lmap, n_regions=5)
        assert sr.n_regions == 5
        assert len(sr) == 5

    def test_label_map_non_square(self):
        lmap = np.zeros((16, 48), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0)
        assert sr.label_map.shape == (16, 48)

    def test_params_stored(self):
        lmap = np.zeros((4, 4), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0,
                         params={"algo": "connected"})
        assert sr.params["algo"] == "connected"


# ─── find_regions extras ──────────────────────────────────────────────────────

class TestFindRegionsExtra2:
    def test_three_blobs_n_regions(self):
        result = find_regions(_three_blob_mask())
        assert result.n_regions == 3

    def test_single_pixel_blob(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[8, 8] = 255
        result = find_regions(mask)
        assert result.n_regions == 1
        assert result.regions[0].area == 1

    def test_four_corners(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0, 0] = 255
        mask[0, 31] = 255
        mask[31, 0] = 255
        mask[31, 31] = 255
        result = find_regions(mask)
        assert result.n_regions == 4

    def test_non_square_mask_non_2d_raises(self):
        with pytest.raises(ValueError):
            find_regions(np.zeros((4, 4, 1), dtype=np.uint8))

    def test_areas_sum_correct(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:10, 0:10] = 255   # 100 px
        mask[20:30, 20:30] = 255  # 100 px
        result = find_regions(mask)
        total = sum(r.area for r in result.regions)
        assert total == 200

    def test_label_map_contains_valid_labels(self):
        result = find_regions(_two_blob_mask())
        unique_labels = set(np.unique(result.label_map))
        assert 0 in unique_labels  # background
        for r in result.regions:
            assert r.label in unique_labels


# ─── filter_regions extras ────────────────────────────────────────────────────

class TestFilterRegionsExtra2:
    def test_filter_keeps_all_when_min_0(self):
        split = find_regions(_two_blob_mask())
        result = filter_regions(split, min_area=0)
        assert result.n_regions == split.n_regions

    def test_filter_exact_boundary_both_blobs(self):
        split = find_regions(_two_blob_mask())
        # min=100 → keep both (100 and 225)
        result = filter_regions(split, min_area=100)
        assert result.n_regions == 2

    def test_filter_to_zero(self):
        split = find_regions(_two_blob_mask())
        result = filter_regions(split, min_area=10000)
        assert result.n_regions == 0

    def test_filter_returns_split_result(self):
        split = find_regions(_two_blob_mask())
        result = filter_regions(split, min_area=50)
        assert isinstance(result, SplitResult)

    def test_max_area_exact_225_keeps_one(self):
        split = find_regions(_two_blob_mask())
        result = filter_regions(split, max_area=225)
        # both ≤ 225
        assert result.n_regions == 2

    def test_filter_three_blobs(self):
        split = find_regions(_three_blob_mask())
        result = filter_regions(split, min_area=50)
        # Only 100px and 225px blobs remain
        assert result.n_regions == 2


# ─── region_masks extras ──────────────────────────────────────────────────────

class TestRegionMasksExtra2:
    def test_shapes_match_original_mask(self):
        mask = _two_blob_mask()
        split = find_regions(mask)
        for m in region_masks(split):
            assert m.shape == mask.shape

    def test_dtype_uint8(self):
        split = find_regions(_two_blob_mask())
        for m in region_masks(split):
            assert m.dtype == np.uint8

    def test_non_overlapping(self):
        split = find_regions(_two_blob_mask())
        masks = region_masks(split)
        if len(masks) == 2:
            assert not np.logical_and(masks[0] > 0, masks[1] > 0).any()

    def test_union_covers_original(self):
        mask = _two_blob_mask()
        split = find_regions(mask)
        masks = region_masks(split)
        union = np.zeros_like(mask)
        for m in masks:
            union = np.maximum(union, m)
        assert np.all(union[mask > 0] > 0)

    def test_three_blobs_three_masks(self):
        split = find_regions(_three_blob_mask())
        masks = region_masks(split)
        assert len(masks) == 3


# ─── merge_small_regions extras ───────────────────────────────────────────────

class TestMergeSmallRegionsExtra2:
    def test_min_area_1_keeps_all(self):
        split = find_regions(_three_blob_mask())
        result = merge_small_regions(split, min_area=1)
        assert result.n_regions == split.n_regions

    def test_min_area_50_removes_25px_blob(self):
        split = find_regions(_three_blob_mask())
        result = merge_small_regions(split, min_area=50)
        # 25px removed → 2 remain
        assert result.n_regions == 2

    def test_all_blobs_removed(self):
        split = find_regions(_two_blob_mask())
        result = merge_small_regions(split, min_area=1000)
        assert result.n_regions == 0

    def test_returns_split_result(self):
        split = find_regions(_two_blob_mask())
        result = merge_small_regions(split, min_area=1)
        assert isinstance(result, SplitResult)


# ─── largest_region extras ────────────────────────────────────────────────────

class TestLargestRegionExtra2:
    def test_three_blobs_largest_is_225(self):
        split = find_regions(_three_blob_mask())
        lr = largest_region(split)
        assert lr is not None
        assert lr.area == 225

    def test_single_blob(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:14, 4:14] = 255  # 100 px
        split = find_regions(mask)
        lr = largest_region(split)
        assert lr is not None
        assert lr.area == 100

    def test_returns_region_info(self):
        split = find_regions(_two_blob_mask())
        lr = largest_region(split)
        assert isinstance(lr, RegionInfo)

    def test_area_is_max_of_all(self):
        split = find_regions(_three_blob_mask())
        lr = largest_region(split)
        max_area = max(r.area for r in split.regions)
        assert lr.area == max_area


# ─── split_mask_to_crops extras ───────────────────────────────────────────────

class TestSplitMaskToCropsExtra2:
    def test_bgr_crops_have_3_channels(self):
        img = np.ones((64, 64, 3), dtype=np.uint8)
        split = find_regions(_two_blob_mask())
        crops = split_mask_to_crops(img, split)
        for c in crops:
            assert c.ndim == 3
            assert c.shape[2] == 3

    def test_single_blob_one_crop(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:14, 4:14] = 255
        split = find_regions(mask)
        img = np.ones((32, 32), dtype=np.uint8)
        crops = split_mask_to_crops(img, split)
        assert len(crops) == 1

    def test_crops_within_image_bounds(self):
        img = np.ones((64, 64), dtype=np.uint8) * 100
        split = find_regions(_two_blob_mask())
        crops = split_mask_to_crops(img, split)
        for c in crops:
            assert c.shape[0] <= 64
            assert c.shape[1] <= 64

    def test_empty_split_no_crops(self):
        split = find_regions(_blank())
        img = np.zeros((32, 32), dtype=np.uint8)
        crops = split_mask_to_crops(img, split)
        assert crops == []


# ─── batch_find_regions extras ────────────────────────────────────────────────

class TestBatchFindRegionsExtra2:
    def test_three_masks_correct_counts(self):
        masks = [_blank(), _two_blob_mask(), _three_blob_mask()]
        results = batch_find_regions(masks)
        assert results[0].n_regions == 0
        assert results[1].n_regions == 2
        assert results[2].n_regions == 3

    def test_all_blank_masks(self):
        results = batch_find_regions([_blank()] * 4)
        assert all(r.n_regions == 0 for r in results)

    def test_all_two_blob_masks(self):
        results = batch_find_regions([_two_blob_mask()] * 3)
        assert all(r.n_regions == 2 for r in results)

    def test_single_mask(self):
        results = batch_find_regions([_two_blob_mask()])
        assert len(results) == 1
        assert results[0].n_regions == 2

    def test_non_square_masks(self):
        mask = np.zeros((32, 64), dtype=np.uint8)
        mask[5:15, 5:20] = 255
        results = batch_find_regions([mask])
        assert results[0].n_regions == 1
