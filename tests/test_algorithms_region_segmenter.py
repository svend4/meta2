"""Тесты для puzzle_reconstruction.algorithms.region_segmenter."""
import pytest
import numpy as np
from puzzle_reconstruction.algorithms.region_segmenter import (
    RegionProps,
    SegmentationResult,
    label_connected,
    compute_region_props,
    filter_regions,
    merge_close_regions,
    region_adjacency,
    largest_region,
    regions_to_mask,
    batch_segment,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=64, w=64) -> np.ndarray:
    """All-black image: no regions."""
    return np.zeros((h, w), dtype=np.uint8)


def _white(h=64, w=64) -> np.ndarray:
    """All-white image: one big region."""
    return np.full((h, w), 255, dtype=np.uint8)


def _two_blobs() -> np.ndarray:
    """Two separate white rectangles in a black image."""
    img = np.zeros((64, 128), dtype=np.uint8)
    img[10:30, 10:40] = 255   # blob 1
    img[10:30, 80:110] = 255  # blob 2
    return img


def _rgb_two_blobs() -> np.ndarray:
    """BGR image with two white blobs."""
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[10:30, 10:40] = 255
    img[10:30, 80:110] = 255
    return img


def _segmented() -> SegmentationResult:
    return label_connected(_two_blobs(), threshold=128)


# ─── TestRegionProps ──────────────────────────────────────────────────────────

class TestRegionProps:
    def test_basic_fields(self):
        r = RegionProps(label=1, area=100, bbox=(5, 5, 20, 20),
                        centroid=(15.0, 15.0), aspect_ratio=1.0,
                        solidity=0.8, perimeter=40.0)
        assert r.label == 1
        assert r.area == 100
        assert r.bbox == (5, 5, 20, 20)

    def test_centroid_stored(self):
        r = RegionProps(label=2, area=50, bbox=(0, 0, 10, 5),
                        centroid=(5.0, 2.5), aspect_ratio=0.5,
                        solidity=1.0, perimeter=30.0)
        assert r.centroid == (5.0, 2.5)

    def test_aspect_ratio_stored(self):
        r = RegionProps(label=1, area=50, bbox=(0, 0, 20, 10),
                        centroid=(10.0, 5.0), aspect_ratio=0.5,
                        solidity=1.0, perimeter=60.0)
        assert r.aspect_ratio == pytest.approx(0.5)


# ─── TestSegmentationResult ───────────────────────────────────────────────────

class TestSegmentationResult:
    def test_basic_fields(self):
        r = _segmented()
        assert isinstance(r, SegmentationResult)
        assert isinstance(r.labels, np.ndarray)

    def test_labels_shape(self):
        r = label_connected(_two_blobs())
        assert r.labels.shape == (64, 128)

    def test_labels_dtype(self):
        r = _segmented()
        assert r.labels.dtype == np.int32

    def test_n_labels_two_blobs(self):
        r = _segmented()
        assert r.n_labels == 2

    def test_props_list(self):
        r = _segmented()
        assert isinstance(r.props, list)
        assert len(r.props) == 2

    def test_params_stored(self):
        r = label_connected(_two_blobs(), connectivity=8, threshold=128)
        assert r.params["connectivity"] == 8
        assert r.params["threshold"] == 128


# ─── TestLabelConnected ───────────────────────────────────────────────────────

class TestLabelConnected:
    def test_returns_segmentation_result(self):
        r = label_connected(_white())
        assert isinstance(r, SegmentationResult)

    def test_blank_no_regions(self):
        r = label_connected(_blank())
        assert r.n_labels == 0

    def test_white_one_region(self):
        r = label_connected(_white())
        assert r.n_labels == 1

    def test_two_blobs(self):
        r = label_connected(_two_blobs())
        assert r.n_labels == 2

    def test_connectivity_4_ok(self):
        r = label_connected(_white(), connectivity=4)
        assert isinstance(r, SegmentationResult)

    def test_connectivity_8_ok(self):
        r = label_connected(_white(), connectivity=8)
        assert isinstance(r, SegmentationResult)

    def test_connectivity_6_raises(self):
        with pytest.raises(ValueError):
            label_connected(_white(), connectivity=6)

    def test_connectivity_0_raises(self):
        with pytest.raises(ValueError):
            label_connected(_white(), connectivity=0)

    def test_threshold_0_ok(self):
        r = label_connected(_blank(), threshold=0)
        assert isinstance(r, SegmentationResult)

    def test_threshold_255_ok(self):
        r = label_connected(_white(), threshold=255)
        assert isinstance(r, SegmentationResult)

    def test_threshold_neg_raises(self):
        with pytest.raises(ValueError):
            label_connected(_white(), threshold=-1)

    def test_threshold_256_raises(self):
        with pytest.raises(ValueError):
            label_connected(_white(), threshold=256)

    def test_rgb_ok(self):
        r = label_connected(_rgb_two_blobs())
        assert r.n_labels == 2

    def test_labels_background_zero(self):
        r = label_connected(_two_blobs())
        # Background must have label 0
        assert r.labels.min() == 0

    def test_props_all_region_props(self):
        r = label_connected(_two_blobs())
        for p in r.props:
            assert isinstance(p, RegionProps)


# ─── TestComputeRegionProps ───────────────────────────────────────────────────

class TestComputeRegionProps:
    def test_returns_list(self):
        r = _segmented()
        result = compute_region_props(r)
        assert isinstance(result, list)

    def test_same_as_props(self):
        r = _segmented()
        result = compute_region_props(r)
        assert len(result) == len(r.props)

    def test_empty_result(self):
        r = label_connected(_blank())
        result = compute_region_props(r)
        assert result == []

    def test_all_region_props(self):
        r = _segmented()
        for p in compute_region_props(r):
            assert isinstance(p, RegionProps)


# ─── TestFilterRegions ────────────────────────────────────────────────────────

class TestFilterRegions:
    def test_returns_segmentation_result(self):
        r = _segmented()
        f = filter_regions(r, min_area=10)
        assert isinstance(f, SegmentationResult)

    def test_min_area_filters_small(self):
        r = _segmented()
        large_area = max(p.area for p in r.props)
        f = filter_regions(r, min_area=large_area)
        assert f.n_labels <= r.n_labels

    def test_max_area_filters_large(self):
        r = _segmented()
        small_area = min(p.area for p in r.props)
        f = filter_regions(r, max_area=small_area)
        assert all(p.area <= small_area for p in f.props)

    def test_no_filter_keeps_all(self):
        r = _segmented()
        f = filter_regions(r)
        assert f.n_labels == r.n_labels

    def test_min_gt_max_raises(self):
        r = _segmented()
        with pytest.raises(ValueError):
            filter_regions(r, min_area=500, max_area=100)

    def test_filtered_labels_zeroed(self):
        r = label_connected(_two_blobs())
        large = max(p.area for p in r.props)
        f = filter_regions(r, min_area=large + 1)
        assert f.n_labels == 0

    def test_min_aspect_ratio_filter(self):
        r = _segmented()
        f = filter_regions(r, min_aspect_ratio=0.5)
        for p in f.props:
            assert p.aspect_ratio >= 0.5


# ─── TestMergeCloseRegions ────────────────────────────────────────────────────

class TestMergeCloseRegions:
    def test_returns_segmentation_result(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=5.0)
        assert isinstance(m, SegmentationResult)

    def test_max_distance_neg_raises(self):
        r = _segmented()
        with pytest.raises(ValueError):
            merge_close_regions(r, max_distance=-1.0)

    def test_zero_distance_no_merge(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=0.0)
        assert m.n_labels == r.n_labels

    def test_huge_distance_merges_all(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=1000.0)
        assert m.n_labels <= r.n_labels

    def test_single_region_unchanged(self):
        r = label_connected(_white())
        m = merge_close_regions(r, max_distance=50.0)
        assert isinstance(m, SegmentationResult)


# ─── TestRegionAdjacency ──────────────────────────────────────────────────────

class TestRegionAdjacency:
    def test_returns_dict(self):
        r = _segmented()
        adj = region_adjacency(r)
        assert isinstance(adj, dict)

    def test_keys_are_labels(self):
        r = _segmented()
        adj = region_adjacency(r)
        for p in r.props:
            assert p.label in adj

    def test_dilation_ksize_zero_raises(self):
        r = _segmented()
        with pytest.raises(ValueError):
            region_adjacency(r, dilation_ksize=0)

    def test_no_regions_empty_dict(self):
        r = label_connected(_blank())
        adj = region_adjacency(r)
        assert adj == {}

    def test_far_blobs_not_adjacent(self):
        r = label_connected(_two_blobs(), threshold=128)
        adj = region_adjacency(r, dilation_ksize=3)
        # Blobs are far apart (>40 pixels), so they should not be adjacent
        for label, neighbors in adj.items():
            assert isinstance(neighbors, list)


# ─── TestLargestRegion ────────────────────────────────────────────────────────

class TestLargestRegion:
    def test_returns_region_props(self):
        r = _segmented()
        lr = largest_region(r)
        assert isinstance(lr, RegionProps)

    def test_empty_returns_none(self):
        r = label_connected(_blank())
        assert largest_region(r) is None

    def test_is_largest(self):
        r = _segmented()
        lr = largest_region(r)
        assert all(lr.area >= p.area for p in r.props)

    def test_single_region(self):
        r = label_connected(_white())
        lr = largest_region(r)
        assert lr is not None
        assert lr.area > 0


# ─── TestRegionsToMask ────────────────────────────────────────────────────────

class TestRegionsToMask:
    def test_returns_ndarray(self):
        r = _segmented()
        mask = regions_to_mask(r)
        assert isinstance(mask, np.ndarray)

    def test_mask_shape(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert mask.shape == (64, 128)

    def test_mask_dtype(self):
        r = _segmented()
        mask = regions_to_mask(r)
        assert mask.dtype == np.uint8

    def test_mask_values_binary(self):
        r = _segmented()
        mask = regions_to_mask(r)
        unique = set(mask.ravel().tolist())
        assert unique.issubset({0, 255})

    def test_subset_labels(self):
        r = _segmented()
        if r.n_labels >= 1:
            label = r.props[0].label
            mask = regions_to_mask(r, labels=[label])
            assert mask.sum() > 0

    def test_all_regions_none_labels(self):
        r = _segmented()
        mask_all = regions_to_mask(r, labels=None)
        mask_subset = regions_to_mask(r, labels=[p.label for p in r.props])
        np.testing.assert_array_equal(mask_all, mask_subset)

    def test_empty_regions_black_mask(self):
        r = label_connected(_blank())
        mask = regions_to_mask(r)
        assert mask.max() == 0


# ─── TestBatchSegment ─────────────────────────────────────────────────────────

class TestBatchSegment:
    def test_returns_list(self):
        imgs = [_white(), _blank()]
        result = batch_segment(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_two_blobs(), _white(), _blank()]
        assert len(batch_segment(imgs)) == 3

    def test_empty_list(self):
        assert batch_segment([]) == []

    def test_all_segmentation_results(self):
        imgs = [_white(), _two_blobs()]
        for r in batch_segment(imgs):
            assert isinstance(r, SegmentationResult)

    def test_connectivity_passed(self):
        imgs = [_white()]
        result = batch_segment(imgs, connectivity=4)
        assert result[0].params["connectivity"] == 4

    def test_threshold_passed(self):
        imgs = [_white()]
        result = batch_segment(imgs, threshold=200)
        assert result[0].params["threshold"] == 200
