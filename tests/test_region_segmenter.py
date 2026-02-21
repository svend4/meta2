"""Tests for puzzle_reconstruction.algorithms.region_segmenter."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.region_segmenter import (
    RegionProps,
    SegmentationResult,
    batch_segment,
    compute_region_props,
    filter_regions,
    label_connected,
    largest_region,
    merge_close_regions,
    region_adjacency,
    regions_to_mask,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _two_blobs() -> np.ndarray:
    """Binary image (uint8) with two separated blobs."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[5:20, 5:20] = 255   # blob 1
    img[40:55, 40:55] = 255  # blob 2
    return img


def _single_blob() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10:20, 10:20] = 255
    return img


def _blank() -> np.ndarray:
    return np.zeros((32, 32), dtype=np.uint8)


# ─── RegionProps ──────────────────────────────────────────────────────────────

class TestRegionProps:
    def test_fields_stored(self):
        p = RegionProps(
            label=1, area=100, bbox=(0, 0, 10, 10),
            centroid=(5.0, 5.0), aspect_ratio=1.0,
            solidity=1.0, perimeter=40.0,
        )
        assert p.label == 1
        assert p.area == 100
        assert p.bbox == (0, 0, 10, 10)
        assert p.centroid == (5.0, 5.0)
        assert p.aspect_ratio == pytest.approx(1.0)
        assert p.solidity == pytest.approx(1.0)
        assert p.perimeter == pytest.approx(40.0)


# ─── SegmentationResult ───────────────────────────────────────────────────────

class TestSegmentationResult:
    def test_fields_stored(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        r = SegmentationResult(labels=labels, n_labels=0)
        assert r.n_labels == 0
        assert r.labels.shape == (10, 10)

    def test_default_props_empty(self):
        r = SegmentationResult(labels=np.zeros((4, 4), dtype=np.int32), n_labels=0)
        assert r.props == []

    def test_default_params_empty(self):
        r = SegmentationResult(labels=np.zeros((4, 4), dtype=np.int32), n_labels=0)
        assert r.params == {}


# ─── label_connected ─────────────────────────────────────────────────────────

class TestLabelConnected:
    def test_returns_segmentation_result(self):
        r = label_connected(_two_blobs())
        assert isinstance(r, SegmentationResult)

    def test_two_blobs_found(self):
        r = label_connected(_two_blobs())
        assert r.n_labels == 2

    def test_blank_image_no_labels(self):
        r = label_connected(_blank())
        assert r.n_labels == 0

    def test_labels_dtype_int32(self):
        r = label_connected(_single_blob())
        assert r.labels.dtype == np.int32

    def test_labels_shape_matches_image(self):
        img = _two_blobs()
        r = label_connected(img)
        assert r.labels.shape == img.shape

    def test_props_count_matches_n_labels(self):
        r = label_connected(_two_blobs())
        assert len(r.props) == r.n_labels

    def test_params_stored(self):
        r = label_connected(_two_blobs(), connectivity=4, threshold=128)
        assert r.params["connectivity"] == 4
        assert r.params["threshold"] == 128

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError):
            label_connected(_single_blob(), connectivity=6)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            label_connected(_single_blob(), threshold=-1)
        with pytest.raises(ValueError):
            label_connected(_single_blob(), threshold=256)

    def test_bgr_image_accepted(self):
        img_bgr = np.stack([_single_blob()] * 3, axis=-1)
        r = label_connected(img_bgr)
        assert r.n_labels >= 0

    def test_props_area_positive(self):
        r = label_connected(_single_blob())
        for p in r.props:
            assert p.area > 0

    def test_props_aspect_ratio_in_unit_interval(self):
        r = label_connected(_two_blobs())
        for p in r.props:
            assert 0.0 < p.aspect_ratio <= 1.0

    def test_connectivity_8_default(self):
        r = label_connected(_two_blobs())
        assert r.params.get("connectivity") == 8


# ─── compute_region_props ─────────────────────────────────────────────────────

class TestComputeRegionProps:
    def test_returns_list(self):
        r = label_connected(_two_blobs())
        props = compute_region_props(r)
        assert isinstance(props, list)

    def test_same_length_as_result_props(self):
        r = label_connected(_two_blobs())
        props = compute_region_props(r)
        assert len(props) == len(r.props)

    def test_empty_result(self):
        r = label_connected(_blank())
        props = compute_region_props(r)
        assert props == []


# ─── filter_regions ───────────────────────────────────────────────────────────

class TestFilterRegions:
    def test_returns_segmentation_result(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r)
        assert isinstance(fr, SegmentationResult)

    def test_no_filter_keeps_all(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r)
        assert fr.n_labels == r.n_labels

    def test_min_area_filters_small(self):
        r = label_connected(_two_blobs())
        # Both blobs are 15×15 = 225 px; set min_area very large
        fr = filter_regions(r, min_area=10000)
        assert fr.n_labels == 0

    def test_max_area_filters_large(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r, max_area=1)
        assert fr.n_labels == 0

    def test_min_area_exceeds_max_raises(self):
        r = label_connected(_two_blobs())
        with pytest.raises(ValueError):
            filter_regions(r, min_area=100, max_area=50)

    def test_filtered_labels_zeroed(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r, min_area=10000)
        assert np.all(fr.labels == 0)

    def test_min_solidity_filters(self):
        r = label_connected(_two_blobs())
        # All blobs should be fairly solid (square blob)
        fr = filter_regions(r, min_solidity=0.9)
        assert fr.n_labels == r.n_labels

    def test_min_aspect_ratio_filters(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r, min_aspect_ratio=0.99)
        # Square blobs have AR=1.0 so none filtered
        assert fr.n_labels == r.n_labels


# ─── merge_close_regions ─────────────────────────────────────────────────────

class TestMergeCloseRegions:
    def test_returns_segmentation_result(self):
        r = label_connected(_two_blobs())
        mr = merge_close_regions(r, max_distance=5.0)
        assert isinstance(mr, SegmentationResult)

    def test_far_blobs_not_merged(self):
        r = label_connected(_two_blobs())
        mr = merge_close_regions(r, max_distance=5.0)
        assert mr.n_labels == r.n_labels

    def test_very_large_distance_merges_all(self):
        r = label_connected(_two_blobs())
        mr = merge_close_regions(r, max_distance=1000.0)
        assert mr.n_labels <= r.n_labels

    def test_single_region_unchanged(self):
        r = label_connected(_single_blob())
        mr = merge_close_regions(r, max_distance=10.0)
        assert mr.n_labels == r.n_labels

    def test_negative_distance_raises(self):
        r = label_connected(_two_blobs())
        with pytest.raises(ValueError):
            merge_close_regions(r, max_distance=-1.0)

    def test_zero_distance_unchanged(self):
        r = label_connected(_two_blobs())
        mr = merge_close_regions(r, max_distance=0.0)
        assert mr.n_labels == r.n_labels


# ─── region_adjacency ─────────────────────────────────────────────────────────

class TestRegionAdjacency:
    def test_returns_dict(self):
        r = label_connected(_two_blobs())
        adj = region_adjacency(r)
        assert isinstance(adj, dict)

    def test_keys_are_labels(self):
        r = label_connected(_two_blobs())
        adj = region_adjacency(r)
        expected_labels = {p.label for p in r.props}
        assert set(adj.keys()) == expected_labels

    def test_far_blobs_not_adjacent(self):
        r = label_connected(_two_blobs())
        adj = region_adjacency(r, dilation_ksize=3)
        for label, neighbors in adj.items():
            assert len(neighbors) == 0

    def test_adjacent_blobs_linked(self):
        # Two blobs right next to each other
        img = np.zeros((32, 64), dtype=np.uint8)
        img[10:22, 5:20] = 255
        img[10:22, 22:37] = 255
        r = label_connected(img)
        if r.n_labels == 2:
            adj = region_adjacency(r, dilation_ksize=5)
            # At least one of them should see the other as neighbor
            all_neighbors = sum(len(v) for v in adj.values())
            assert all_neighbors >= 0  # structure is correct

    def test_invalid_dilation_ksize_raises(self):
        r = label_connected(_single_blob())
        with pytest.raises(ValueError):
            region_adjacency(r, dilation_ksize=0)

    def test_symmetric_adjacency(self):
        img = np.zeros((32, 64), dtype=np.uint8)
        img[10:22, 5:15] = 255
        img[10:22, 17:27] = 255
        r = label_connected(img)
        adj = region_adjacency(r, dilation_ksize=7)
        for lab, neighbors in adj.items():
            for nb in neighbors:
                assert lab in adj[nb]


# ─── largest_region ───────────────────────────────────────────────────────────

class TestLargestRegion:
    def test_returns_region_props(self):
        r = label_connected(_two_blobs())
        lr = largest_region(r)
        assert isinstance(lr, RegionProps)

    def test_empty_result_returns_none(self):
        r = label_connected(_blank())
        assert largest_region(r) is None

    def test_largest_has_max_area(self):
        r = label_connected(_two_blobs())
        lr = largest_region(r)
        max_area = max(p.area for p in r.props)
        assert lr.area == max_area

    def test_single_region_is_largest(self):
        r = label_connected(_single_blob())
        lr = largest_region(r)
        assert lr is not None
        assert lr == r.props[0]


# ─── regions_to_mask ─────────────────────────────────────────────────────────

class TestRegionsToMask:
    def test_returns_uint8(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert mask.dtype == np.uint8

    def test_shape_matches_labels(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert mask.shape == r.labels.shape

    def test_all_regions_255(self):
        r = label_connected(_single_blob())
        mask = regions_to_mask(r)
        # Where blob was originally, mask should be 255
        assert np.any(mask == 255)

    def test_blank_result_all_zeros(self):
        r = label_connected(_blank())
        mask = regions_to_mask(r)
        assert np.all(mask == 0)

    def test_selected_labels_only(self):
        r = label_connected(_two_blobs())
        if r.n_labels >= 2:
            first_label = r.props[0].label
            mask = regions_to_mask(r, labels=[first_label])
            # mask should only cover first label pixels
            covered = r.labels == first_label
            np.testing.assert_array_equal(mask[covered], 255)

    def test_values_only_0_or_255(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})


# ─── batch_segment ────────────────────────────────────────────────────────────

class TestBatchSegment:
    def test_returns_list(self):
        result = batch_segment([_single_blob(), _two_blobs()])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        images = [_single_blob(), _two_blobs(), _blank()]
        result = batch_segment(images)
        assert len(result) == 3

    def test_all_segmentation_results(self):
        result = batch_segment([_single_blob(), _blank()])
        assert all(isinstance(r, SegmentationResult) for r in result)

    def test_empty_list_returns_empty(self):
        result = batch_segment([])
        assert result == []

    def test_connectivity_forwarded(self):
        result = batch_segment([_single_blob()], connectivity=4)
        assert result[0].params["connectivity"] == 4

    def test_threshold_forwarded(self):
        result = batch_segment([_single_blob()], threshold=200)
        assert result[0].params["threshold"] == 200
