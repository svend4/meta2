"""Tests for puzzle_reconstruction.algorithms.region_splitter."""
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

def _blank_mask(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _two_blobs():
    """Mask with two separated white blobs."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 5:15] = 255   # blob 1: 100 px
    mask[40:55, 40:55] = 255  # blob 2: 225 px
    return mask


def _make_region(label=1, area=100):
    mask = np.zeros((20, 20), dtype=np.uint8)
    return RegionInfo(
        label=label,
        area=area,
        bbox=(0, 0, 10, 10),
        centroid=(5.0, 5.0),
        mask=mask,
    )


# ─── TestRegionInfo ───────────────────────────────────────────────────────────

class TestRegionInfo:
    def test_valid_creation(self):
        r = _make_region(label=1, area=50)
        assert r.label == 1
        assert r.area == 50

    def test_negative_label_raises(self):
        with pytest.raises(ValueError, match="label"):
            RegionInfo(label=-1, area=10, bbox=(0, 0, 5, 5),
                       centroid=(2.5, 2.5), mask=np.zeros((5, 5), dtype=np.uint8))

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="area"):
            RegionInfo(label=1, area=-1, bbox=(0, 0, 5, 5),
                       centroid=(2.5, 2.5), mask=np.zeros((5, 5), dtype=np.uint8))

    def test_len_returns_area(self):
        r = _make_region(area=42)
        assert len(r) == 42

    def test_label_zero_allowed(self):
        r = RegionInfo(label=0, area=0, bbox=(0, 0, 0, 0),
                       centroid=(0.0, 0.0), mask=np.zeros((1, 1), dtype=np.uint8))
        assert r.label == 0

    def test_params_default_empty(self):
        r = _make_region()
        assert r.params == {}


# ─── TestSplitResult ──────────────────────────────────────────────────────────

class TestSplitResult:
    def test_valid_creation(self):
        lmap = np.zeros((10, 10), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0)
        assert sr.n_regions == 0

    def test_negative_n_regions_raises(self):
        lmap = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError, match="n_regions"):
            SplitResult(regions=[], label_map=lmap, n_regions=-1)

    def test_len_returns_n_regions(self):
        lmap = np.zeros((10, 10), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=3)
        assert len(sr) == 3

    def test_params_default_empty(self):
        lmap = np.zeros((10, 10), dtype=np.int32)
        sr = SplitResult(regions=[], label_map=lmap, n_regions=0)
        assert sr.params == {}


# ─── TestFindRegions ──────────────────────────────────────────────────────────

class TestFindRegions:
    def test_blank_mask_no_regions(self):
        result = find_regions(_blank_mask())
        assert result.n_regions == 0
        assert result.regions == []

    def test_two_blobs_detected(self):
        result = find_regions(_two_blobs())
        assert result.n_regions == 2

    def test_label_map_shape_matches_mask(self):
        mask = _two_blobs()
        result = find_regions(mask)
        assert result.label_map.shape == mask.shape

    def test_label_map_dtype_int32(self):
        result = find_regions(_two_blobs())
        assert result.label_map.dtype == np.int32

    def test_region_areas_positive(self):
        result = find_regions(_two_blobs())
        for r in result.regions:
            assert r.area > 0

    def test_non_2d_mask_raises(self):
        mask3d = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2-D"):
            find_regions(mask3d)

    def test_bool_mask_accepted(self):
        mask = _two_blobs().astype(bool)
        result = find_regions(mask)
        assert result.n_regions == 2

    def test_single_blob(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        result = find_regions(mask)
        assert result.n_regions == 1
        assert result.regions[0].label == 1


# ─── TestFilterRegions ────────────────────────────────────────────────────────

class TestFilterRegions:
    def setup_method(self):
        self.split = find_regions(_two_blobs())

    def test_no_filter_keeps_all(self):
        result = filter_regions(self.split)
        assert result.n_regions == self.split.n_regions

    def test_min_area_removes_small(self):
        result = filter_regions(self.split, min_area=200)
        assert result.n_regions < self.split.n_regions
        for r in result.regions:
            assert r.area >= 200

    def test_max_area_removes_large(self):
        result = filter_regions(self.split, max_area=150)
        for r in result.regions:
            assert r.area <= 150

    def test_negative_min_area_raises(self):
        with pytest.raises(ValueError, match="min_area"):
            filter_regions(self.split, min_area=-1)

    def test_max_area_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_area"):
            filter_regions(self.split, min_area=100, max_area=50)

    def test_filtered_labels_start_at_1(self):
        result = filter_regions(self.split, min_area=1)
        for i, r in enumerate(result.regions, start=1):
            assert r.label == i


# ─── TestRegionMasks ──────────────────────────────────────────────────────────

class TestRegionMasks:
    def test_returns_list_of_arrays(self):
        result = find_regions(_two_blobs())
        masks = region_masks(result)
        assert len(masks) == 2
        assert all(isinstance(m, np.ndarray) for m in masks)

    def test_masks_are_uint8(self):
        result = find_regions(_two_blobs())
        for m in region_masks(result):
            assert m.dtype == np.uint8

    def test_empty_split_returns_empty_list(self):
        result = find_regions(_blank_mask())
        assert region_masks(result) == []

    def test_mask_values_are_0_or_255(self):
        result = find_regions(_two_blobs())
        for m in region_masks(result):
            assert set(np.unique(m)).issubset({0, 255})


# ─── TestMergeSmallRegions ────────────────────────────────────────────────────

class TestMergeSmallRegions:
    def test_min_area_less_than_1_raises(self):
        result = find_regions(_two_blobs())
        with pytest.raises(ValueError, match="min_area"):
            merge_small_regions(result, min_area=0)

    def test_removes_small_regions(self):
        result = find_regions(_two_blobs())
        merged = merge_small_regions(result, min_area=150)
        assert merged.n_regions < result.n_regions

    def test_keeps_large_regions(self):
        result = find_regions(_two_blobs())
        merged = merge_small_regions(result, min_area=1)
        assert merged.n_regions == result.n_regions


# ─── TestLargestRegion ────────────────────────────────────────────────────────

class TestLargestRegion:
    def test_empty_split_returns_none(self):
        result = find_regions(_blank_mask())
        assert largest_region(result) is None

    def test_returns_largest(self):
        result = find_regions(_two_blobs())
        lr = largest_region(result)
        assert lr is not None
        for r in result.regions:
            assert lr.area >= r.area

    def test_single_region(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:20, 4:20] = 255
        result = find_regions(mask)
        lr = largest_region(result)
        assert lr is not None
        assert lr.area == result.regions[0].area


# ─── TestSplitMaskToCrops ────────────────────────────────────────────────────

class TestSplitMaskToCrops:
    def test_2d_image_crops(self):
        image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        split = find_regions(_two_blobs())
        crops = split_mask_to_crops(image, split)
        assert len(crops) == split.n_regions
        for crop in crops:
            assert crop.ndim == 2

    def test_3d_image_crops(self):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        split = find_regions(_two_blobs())
        crops = split_mask_to_crops(image, split)
        assert len(crops) == split.n_regions
        for crop in crops:
            assert crop.ndim == 3

    def test_non_2d_3d_image_raises(self):
        image = np.zeros((64, 64, 3, 1), dtype=np.uint8)
        split = find_regions(_two_blobs())
        with pytest.raises(ValueError, match="2-D или 3-D"):
            split_mask_to_crops(image, split)

    def test_empty_split_returns_empty_list(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        split = find_regions(_blank_mask(32, 32))
        crops = split_mask_to_crops(image, split)
        assert crops == []

    def test_crop_is_copy(self):
        image = np.ones((64, 64), dtype=np.uint8) * 100
        split = find_regions(_two_blobs())
        crops = split_mask_to_crops(image, split)
        if crops:
            crops[0][:] = 0
            assert image[5, 5] == 100  # original unmodified


# ─── TestBatchFindRegions ─────────────────────────────────────────────────────

class TestBatchFindRegions:
    def test_returns_list_of_split_results(self):
        masks = [_blank_mask(), _two_blobs(), _blank_mask()]
        results = batch_find_regions(masks)
        assert len(results) == 3
        assert all(isinstance(r, SplitResult) for r in results)

    def test_empty_list(self):
        assert batch_find_regions([]) == []

    def test_counts_match_expected(self):
        masks = [_blank_mask(), _two_blobs()]
        results = batch_find_regions(masks)
        assert results[0].n_regions == 0
        assert results[1].n_regions == 2
