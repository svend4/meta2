"""Тесты для puzzle_reconstruction.algorithms.region_splitter."""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.region_splitter import (
    RegionInfo,
    SplitResult,
    find_regions,
    filter_regions,
    region_masks,
    merge_small_regions,
    largest_region,
    split_mask_to_crops,
    batch_find_regions,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _two_blob_mask():
    """64×64 маска с двумя прямоугольными регионами."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:15, 5:15] = 255   # регион A  (10×10 = 100 px)
    mask[40:55, 40:55] = 255  # регион B  (15×15 = 225 px)
    return mask


# ─── TestRegionInfo ───────────────────────────────────────────────────────────

class TestRegionInfo:
    def _make(self, label=1, area=50):
        mask = np.zeros((10, 10), dtype=np.uint8)
        return RegionInfo(
            label=label, area=area,
            bbox=(0, 0, 10, 10),
            centroid=(5.0, 5.0),
            mask=mask,
        )

    def test_basic_creation(self):
        r = self._make()
        assert r.label == 1
        assert r.area == 50

    def test_len(self):
        r = self._make(area=42)
        assert len(r) == 42

    def test_negative_label_raises(self):
        with pytest.raises(ValueError):
            self._make(label=-1)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError):
            self._make(area=-1)

    def test_zero_area_valid(self):
        r = self._make(area=0)
        assert r.area == 0


# ─── TestSplitResult ──────────────────────────────────────────────────────────

class TestSplitResult:
    def _make(self, n=2):
        masks = np.zeros((16, 16), dtype=np.int32)
        regions = []
        for i in range(1, n + 1):
            regions.append(RegionInfo(
                label=i, area=10,
                bbox=(0, 0, 4, 4),
                centroid=(2.0, 2.0),
                mask=np.zeros((16, 16), dtype=np.uint8),
            ))
        return SplitResult(regions=regions, label_map=masks, n_regions=n)

    def test_len(self):
        sr = self._make(3)
        assert len(sr) == 3

    def test_negative_n_regions_raises(self):
        with pytest.raises(ValueError):
            SplitResult(
                regions=[], label_map=np.zeros((4, 4), dtype=np.int32), n_regions=-1
            )

    def test_zero_n_regions_valid(self):
        sr = SplitResult(
            regions=[], label_map=np.zeros((4, 4), dtype=np.int32), n_regions=0
        )
        assert len(sr) == 0


# ─── TestFindRegions ──────────────────────────────────────────────────────────

class TestFindRegions:
    def test_returns_split_result(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        assert isinstance(result, SplitResult)

    def test_two_blobs_found(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        assert result.n_regions == 2

    def test_empty_mask_no_regions(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = find_regions(mask)
        assert result.n_regions == 0

    def test_full_mask_one_region(self):
        mask = np.ones((16, 16), dtype=np.uint8) * 255
        result = find_regions(mask)
        assert result.n_regions == 1

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            find_regions(np.ones((4, 4, 3), dtype=np.uint8))

    def test_label_map_shape(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        assert result.label_map.shape == mask.shape

    def test_label_map_dtype_int32(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        assert result.label_map.dtype == np.int32

    def test_region_area_positive(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        for r in result.regions:
            assert r.area > 0


# ─── TestFilterRegions ────────────────────────────────────────────────────────

class TestFilterRegions:
    def test_min_area_removes_small(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        # A=100px, B=225px; keep only B
        filtered = filter_regions(result, min_area=150)
        assert filtered.n_regions == 1
        assert filtered.regions[0].area >= 150

    def test_max_area_removes_large(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        filtered = filter_regions(result, max_area=150)
        assert filtered.n_regions == 1
        assert filtered.regions[0].area <= 150

    def test_keep_all(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        filtered = filter_regions(result, min_area=0)
        assert filtered.n_regions == result.n_regions

    def test_negative_min_area_raises(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        with pytest.raises(ValueError):
            filter_regions(result, min_area=-1)

    def test_max_lt_min_raises(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        with pytest.raises(ValueError):
            filter_regions(result, min_area=100, max_area=50)


# ─── TestRegionMasks ──────────────────────────────────────────────────────────

class TestRegionMasks:
    def test_returns_list(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        masks = region_masks(result)
        assert isinstance(masks, list)

    def test_length_matches_n_regions(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        masks = region_masks(result)
        assert len(masks) == result.n_regions

    def test_each_mask_uint8(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        for m in region_masks(result):
            assert m.dtype == np.uint8

    def test_empty_result_empty_list(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        result = find_regions(mask)
        assert region_masks(result) == []


# ─── TestMergeSmallRegions ────────────────────────────────────────────────────

class TestMergeSmallRegions:
    def test_removes_small(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        merged = merge_small_regions(result, min_area=150)
        assert merged.n_regions == 1

    def test_keeps_all_if_min_area_1(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        merged = merge_small_regions(result, min_area=1)
        assert merged.n_regions == result.n_regions

    def test_min_area_zero_raises(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        with pytest.raises(ValueError):
            merge_small_regions(result, min_area=0)


# ─── TestLargestRegion ────────────────────────────────────────────────────────

class TestLargestRegion:
    def test_returns_largest(self):
        mask = _two_blob_mask()
        result = find_regions(mask)
        lr = largest_region(result)
        assert lr is not None
        assert lr.area == max(r.area for r in result.regions)

    def test_empty_returns_none(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        result = find_regions(mask)
        assert largest_region(result) is None

    def test_single_region(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        result = find_regions(mask)
        lr = largest_region(result)
        assert lr is not None
        assert lr.area > 0


# ─── TestSplitMaskToCrops ─────────────────────────────────────────────────────

class TestSplitMaskToCrops:
    def test_returns_list(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = _two_blob_mask()
        result = find_regions(mask)
        crops = split_mask_to_crops(img, result)
        assert isinstance(crops, list)

    def test_length_matches_n_regions(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = _two_blob_mask()
        result = find_regions(mask)
        crops = split_mask_to_crops(img, result)
        assert len(crops) == result.n_regions

    def test_color_image(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = _two_blob_mask()
        result = find_regions(mask)
        crops = split_mask_to_crops(img, result)
        for c in crops:
            assert c.ndim == 3

    def test_non_2d_3d_image_raises(self):
        img = np.ones((4, 4, 4, 4), dtype=np.uint8)
        mask = _two_blob_mask()
        result = find_regions(mask)
        with pytest.raises(ValueError):
            split_mask_to_crops(img, result)

    def test_empty_result_empty_crops(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        result = find_regions(mask)
        crops = split_mask_to_crops(img, result)
        assert crops == []


# ─── TestBatchFindRegions ─────────────────────────────────────────────────────

class TestBatchFindRegions:
    def test_returns_list(self):
        masks = [_two_blob_mask() for _ in range(3)]
        results = batch_find_regions(masks)
        assert isinstance(results, list)

    def test_correct_length(self):
        masks = [_two_blob_mask() for _ in range(4)]
        results = batch_find_regions(masks)
        assert len(results) == 4

    def test_empty_list(self):
        results = batch_find_regions([])
        assert results == []

    def test_each_split_result(self):
        masks = [_two_blob_mask(), np.zeros((16, 16), dtype=np.uint8)]
        results = batch_find_regions(masks)
        assert all(isinstance(r, SplitResult) for r in results)
