"""Extra tests for puzzle_reconstruction.algorithms.region_splitter."""
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

def _blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _single_blob(h=32, w=32, y0=4, x0=4, size=10):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y0 + size, x0:x0 + size] = 255
    return mask


def _two_blobs():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 5:15] = 255    # blob 1: 100 px
    mask[40:55, 40:55] = 255  # blob 2: 225 px
    return mask


def _make_region(label=1, area=50):
    return RegionInfo(
        label=label, area=area,
        bbox=(0, 0, 10, 10),
        centroid=(5.0, 5.0),
        mask=np.zeros((10, 10), dtype=np.uint8),
    )


# ─── RegionInfo extras ────────────────────────────────────────────────────────

class TestRegionInfoExtra:
    def test_repr_is_string(self):
        r = _make_region()
        assert isinstance(repr(r), str)

    def test_centroid_stored(self):
        r = RegionInfo(label=1, area=10, bbox=(0, 0, 5, 5),
                       centroid=(2.5, 3.7), mask=np.zeros((5, 5), dtype=np.uint8))
        assert r.centroid == (2.5, 3.7)

    def test_bbox_stored(self):
        r = RegionInfo(label=1, area=10, bbox=(3, 4, 9, 11),
                       centroid=(6.0, 7.5), mask=np.zeros((7, 6), dtype=np.uint8))
        assert r.bbox == (3, 4, 9, 11)

    def test_large_area(self):
        r = RegionInfo(label=2, area=10000, bbox=(0, 0, 100, 100),
                       centroid=(50.0, 50.0), mask=np.zeros((100, 100), dtype=np.uint8))
        assert r.area == 10000

    def test_len_equals_area(self):
        for area in (0, 1, 55, 1000):
            r = _make_region(area=area)
            assert len(r) == area

    def test_mask_stored(self):
        m = np.full((4, 4), 255, dtype=np.uint8)
        r = RegionInfo(label=1, area=16, bbox=(0, 0, 4, 4),
                       centroid=(2.0, 2.0), mask=m)
        assert r.mask.shape == (4, 4)

    def test_params_custom_stored(self):
        r = RegionInfo(label=1, area=10, bbox=(0, 0, 5, 5),
                       centroid=(2.5, 2.5), mask=np.zeros((5, 5), dtype=np.uint8),
                       params={"tag": "blob"})
        assert r.params["tag"] == "blob"


# ─── SplitResult extras ───────────────────────────────────────────────────────

class TestSplitResultExtra:
    def test_repr_is_string(self):
        lmap = np.zeros((8, 8), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0)
        assert isinstance(repr(sr), str)

    def test_regions_stored(self):
        r1 = _make_region(label=1, area=30)
        r2 = _make_region(label=2, area=60)
        lmap = np.zeros((10, 10), dtype=np.int32)
        sr = SplitResult(regions=[r1, r2], label_map=lmap, n_regions=2)
        assert len(sr.regions) == 2

    def test_label_map_shape(self):
        lmap = np.zeros((16, 24), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0)
        assert sr.label_map.shape == (16, 24)

    def test_n_regions_stored(self):
        lmap = np.zeros((5, 5), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=7)
        assert sr.n_regions == 7
        assert len(sr) == 7

    def test_params_custom_stored(self):
        lmap = np.zeros((4, 4), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0,
                         params={"min_area": 10})
        assert sr.params["min_area"] == 10


# ─── find_regions extras ──────────────────────────────────────────────────────

class TestFindRegionsExtra:
    def test_non_square_mask(self):
        mask = np.zeros((32, 64), dtype=np.uint8)
        mask[5:15, 5:40] = 255
        result = find_regions(mask)
        assert result.n_regions == 1

    def test_all_white_mask(self):
        mask = np.full((32, 32), 255, dtype=np.uint8)
        result = find_regions(mask)
        assert result.n_regions == 1

    def test_single_blob_area_correct(self):
        mask = _single_blob(size=8)
        result = find_regions(mask)
        assert result.n_regions == 1
        assert result.regions[0].area == 64

    def test_small_1x1_blob(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3, 3] = 255
        result = find_regions(mask)
        assert result.n_regions == 1
        assert result.regions[0].area == 1

    def test_label_map_dtype(self):
        result = find_regions(_two_blobs())
        assert result.label_map.dtype == np.int32

    def test_three_blobs(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        mask[20:30, 20:30] = 255
        mask[50:60, 50:60] = 255
        result = find_regions(mask)
        assert result.n_regions == 3

    def test_min_area_via_filter(self):
        split = find_regions(_two_blobs())
        result = filter_regions(split, min_area=200)
        for r in result.regions:
            assert r.area >= 200

    def test_returns_split_result(self):
        assert isinstance(find_regions(_blank()), SplitResult)


# ─── filter_regions extras ────────────────────────────────────────────────────

class TestFilterRegionsExtra:
    def test_empty_split_returns_empty(self):
        empty = find_regions(_blank())
        result = filter_regions(empty)
        assert result.n_regions == 0

    def test_returns_split_result(self):
        split = find_regions(_two_blobs())
        assert isinstance(filter_regions(split), SplitResult)

    def test_min_area_equal_to_blob_keeps_it(self):
        split = find_regions(_two_blobs())
        # blob 1 has area=100, blob 2 has area=225
        result = filter_regions(split, min_area=100)
        assert result.n_regions == 2

    def test_min_area_just_above_small_blob_removes_it(self):
        split = find_regions(_two_blobs())
        result = filter_regions(split, min_area=101)
        assert result.n_regions == 1
        assert result.regions[0].area >= 101

    def test_max_area_filters_all(self):
        split = find_regions(_two_blobs())
        result = filter_regions(split, max_area=1)
        assert result.n_regions == 0

    def test_region_areas_within_bounds(self):
        split = find_regions(_two_blobs())
        result = filter_regions(split, min_area=50, max_area=300)
        for r in result.regions:
            assert 50 <= r.area <= 300


# ─── region_masks extras ──────────────────────────────────────────────────────

class TestRegionMasksExtra:
    def test_mask_shapes_match_original(self):
        mask = _two_blobs()
        split = find_regions(mask)
        for m in region_masks(split):
            assert m.shape == mask.shape

    def test_non_overlapping_masks(self):
        """For well-separated blobs, masks should not overlap."""
        split = find_regions(_two_blobs())
        masks = region_masks(split)
        if len(masks) == 2:
            overlap = np.logical_and(masks[0] > 0, masks[1] > 0)
            assert not overlap.any()

    def test_single_region_mask(self):
        mask = _single_blob()
        split = find_regions(mask)
        result = region_masks(split)
        assert len(result) == 1
        assert result[0].dtype == np.uint8

    def test_total_coverage(self):
        """Union of all region masks should cover original foreground."""
        mask = _two_blobs()
        split = find_regions(mask)
        masks = region_masks(split)
        union = np.zeros_like(mask)
        for m in masks:
            union = np.maximum(union, m)
        # Every foreground pixel in original should be covered
        assert np.all(union[mask > 0] > 0)


# ─── merge_small_regions extras ───────────────────────────────────────────────

class TestMergeSmallRegionsExtra:
    def test_returns_split_result(self):
        split = find_regions(_two_blobs())
        result = merge_small_regions(split, min_area=1)
        assert isinstance(result, SplitResult)

    def test_all_too_small_gives_zero(self):
        split = find_regions(_two_blobs())
        result = merge_small_regions(split, min_area=10000)
        assert result.n_regions == 0

    def test_min_area_exact_threshold(self):
        split = find_regions(_two_blobs())
        # blob1=100, blob2=225 → keep both if min_area=100
        result = merge_small_regions(split, min_area=100)
        assert result.n_regions == 2

    def test_large_min_area_removes_smaller_blob(self):
        split = find_regions(_two_blobs())
        result = merge_small_regions(split, min_area=150)
        assert result.n_regions == 1


# ─── largest_region extras ────────────────────────────────────────────────────

class TestLargestRegionExtra:
    def test_two_blobs_returns_largest(self):
        split = find_regions(_two_blobs())
        lr = largest_region(split)
        assert lr is not None
        assert lr.area == 225

    def test_returns_region_info(self):
        split = find_regions(_two_blobs())
        lr = largest_region(split)
        assert isinstance(lr, RegionInfo)

    def test_three_regions_largest_correct(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:5, 0:5] = 255   # 25 px
        mask[10:20, 10:20] = 255  # 100 px
        mask[30:50, 30:50] = 255  # 400 px
        split = find_regions(mask)
        lr = largest_region(split)
        assert lr.area == 400

    def test_single_region_returns_it(self):
        split = find_regions(_single_blob(size=12))
        lr = largest_region(split)
        assert lr is not None
        assert lr.area == 144


# ─── split_mask_to_crops extras ───────────────────────────────────────────────

class TestSplitMaskToCropsExtra:
    def test_non_square_gray_image(self):
        image = np.ones((64, 128), dtype=np.uint8) * 200
        split = find_regions(_two_blobs())
        # mask is 64x64, image is 64x128 → crops should work
        crops = split_mask_to_crops(image[:, :64], split)
        assert len(crops) == split.n_regions

    def test_single_blob_crop(self):
        mask = _single_blob(size=10)
        split = find_regions(mask)
        image = np.ones_like(mask) * 150
        crops = split_mask_to_crops(image, split)
        assert len(crops) == 1
        assert crops[0].dtype == np.uint8

    def test_bgr_crop_has_3_channels(self):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        split = find_regions(_two_blobs())
        crops = split_mask_to_crops(image, split)
        assert len(crops) == split.n_regions
        for c in crops:
            assert c.ndim == 3

    def test_crop_size_within_image(self):
        image = np.ones((64, 64), dtype=np.uint8)
        split = find_regions(_two_blobs())
        crops = split_mask_to_crops(image, split)
        for c in crops:
            assert c.shape[0] <= 64 and c.shape[1] <= 64


# ─── batch_find_regions extras ────────────────────────────────────────────────

class TestBatchFindRegionsExtra:
    def test_all_blank_all_zero(self):
        masks = [_blank() for _ in range(4)]
        results = batch_find_regions(masks)
        assert all(r.n_regions == 0 for r in results)

    def test_all_two_blobs(self):
        masks = [_two_blobs() for _ in range(3)]
        results = batch_find_regions(masks)
        assert all(r.n_regions == 2 for r in results)

    def test_heterogeneous(self):
        masks = [_blank(), _two_blobs(), _single_blob()]
        results = batch_find_regions(masks)
        assert results[0].n_regions == 0
        assert results[1].n_regions == 2
        assert results[2].n_regions == 1

    def test_all_split_results(self):
        masks = [_blank(), _two_blobs()]
        for r in batch_find_regions(masks):
            assert isinstance(r, SplitResult)

    def test_single_mask(self):
        results = batch_find_regions([_two_blobs()])
        assert len(results) == 1
        assert results[0].n_regions == 2
