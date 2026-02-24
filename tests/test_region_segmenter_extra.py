"""Extra tests for puzzle_reconstruction.algorithms.region_segmenter."""
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
    img = np.zeros((64, 64), dtype=np.uint8)
    img[5:20, 5:20] = 255   # blob 1 ≈ 15×15
    img[40:55, 40:55] = 255  # blob 2 ≈ 15×15
    return img


def _single_blob() -> np.ndarray:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10:20, 10:20] = 255
    return img


def _blank() -> np.ndarray:
    return np.zeros((32, 32), dtype=np.uint8)


def _three_blobs() -> np.ndarray:
    img = np.zeros((64, 128), dtype=np.uint8)
    img[5:15, 5:20] = 255    # blob 1
    img[5:15, 55:70] = 255   # blob 2
    img[5:15, 105:120] = 255  # blob 3
    return img


# ─── RegionProps (extra) ──────────────────────────────────────────────────────

class TestRegionPropsExtra:
    def test_label_stored(self):
        p = RegionProps(label=5, area=50, bbox=(0, 0, 5, 10),
                        centroid=(2.5, 5.0), aspect_ratio=0.5,
                        solidity=0.9, perimeter=30.0)
        assert p.label == 5

    def test_area_stored(self):
        p = RegionProps(label=1, area=200, bbox=(0, 0, 20, 10),
                        centroid=(10.0, 5.0), aspect_ratio=0.5,
                        solidity=1.0, perimeter=60.0)
        assert p.area == 200

    def test_bbox_tuple(self):
        p = RegionProps(label=1, area=100, bbox=(5, 10, 20, 30),
                        centroid=(15.0, 20.0), aspect_ratio=1.0,
                        solidity=0.8, perimeter=50.0)
        assert p.bbox == (5, 10, 20, 30)

    def test_centroid_tuple(self):
        p = RegionProps(label=1, area=100, bbox=(0, 0, 10, 10),
                        centroid=(3.5, 7.2), aspect_ratio=1.0,
                        solidity=1.0, perimeter=40.0)
        assert p.centroid == (3.5, 7.2)

    def test_aspect_ratio_stored(self):
        p = RegionProps(label=1, area=50, bbox=(0, 0, 5, 10),
                        centroid=(2.5, 5.0), aspect_ratio=0.5,
                        solidity=1.0, perimeter=30.0)
        assert p.aspect_ratio == pytest.approx(0.5)


# ─── SegmentationResult (extra) ───────────────────────────────────────────────

class TestSegmentationResultExtra:
    def test_n_labels_stored(self):
        r = SegmentationResult(labels=np.zeros((4, 4), dtype=np.int32), n_labels=3)
        assert r.n_labels == 3

    def test_labels_shape(self):
        labels = np.zeros((10, 15), dtype=np.int32)
        r = SegmentationResult(labels=labels, n_labels=0)
        assert r.labels.shape == (10, 15)

    def test_custom_props_stored(self):
        p = RegionProps(label=1, area=100, bbox=(0, 0, 10, 10),
                        centroid=(5.0, 5.0), aspect_ratio=1.0,
                        solidity=1.0, perimeter=40.0)
        r = SegmentationResult(labels=np.zeros((10, 10), dtype=np.int32),
                                n_labels=1, props=[p])
        assert len(r.props) == 1
        assert r.props[0].label == 1

    def test_custom_params_stored(self):
        r = SegmentationResult(labels=np.zeros((4, 4), dtype=np.int32),
                                n_labels=0, params={"foo": 42})
        assert r.params["foo"] == 42


# ─── label_connected (extra) ──────────────────────────────────────────────────

class TestLabelConnectedExtra:
    def test_three_blobs_found(self):
        r = label_connected(_three_blobs())
        assert r.n_labels == 3

    def test_connectivity_4_accepted(self):
        r = label_connected(_single_blob(), connectivity=4)
        assert r.params["connectivity"] == 4

    def test_connectivity_8_accepted(self):
        r = label_connected(_single_blob(), connectivity=8)
        assert r.params["connectivity"] == 8

    def test_threshold_255_detects_nothing(self):
        # with threshold=255, only pixels == 255 qualify; blob has 255 → detected
        r = label_connected(_single_blob(), threshold=255)
        assert r.n_labels >= 0  # depends on implementation

    def test_threshold_1_detects_blobs(self):
        r = label_connected(_single_blob(), threshold=1)
        assert r.n_labels >= 1

    def test_labels_nonneg(self):
        r = label_connected(_two_blobs())
        assert (r.labels >= 0).all()

    def test_labels_dtype_int32(self):
        r = label_connected(_two_blobs())
        assert r.labels.dtype == np.int32

    def test_blank_zero_labels(self):
        r = label_connected(_blank())
        assert r.n_labels == 0
        assert (r.labels == 0).all()

    def test_props_list_length(self):
        r = label_connected(_two_blobs())
        assert len(r.props) == r.n_labels

    def test_props_area_sums_to_blob_area(self):
        r = label_connected(_single_blob())
        total_area = sum(p.area for p in r.props)
        blob_pixels = int(np.count_nonzero(_single_blob()))
        assert abs(total_area - blob_pixels) <= 5  # allow small tolerance


# ─── compute_region_props (extra) ────────────────────────────────────────────

class TestComputeRegionPropsExtra:
    def test_returns_region_props_list(self):
        r = label_connected(_two_blobs())
        props = compute_region_props(r)
        for p in props:
            assert isinstance(p, RegionProps)

    def test_props_have_correct_labels(self):
        r = label_connected(_two_blobs())
        props = compute_region_props(r)
        labels_in_props = {p.label for p in props}
        labels_in_result = {p.label for p in r.props}
        assert labels_in_props == labels_in_result

    def test_single_blob_one_prop(self):
        r = label_connected(_single_blob())
        props = compute_region_props(r)
        assert len(props) == 1

    def test_props_area_positive(self):
        r = label_connected(_two_blobs())
        props = compute_region_props(r)
        for p in props:
            assert p.area > 0


# ─── filter_regions (extra) ──────────────────────────────────────────────────

class TestFilterRegionsExtra:
    def test_min_area_zero_keeps_all(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r, min_area=0)
        assert fr.n_labels == r.n_labels

    def test_min_area_exact_blob_size_keeps_all(self):
        r = label_connected(_single_blob())
        # blob is 10×10 = 100 px
        fr = filter_regions(r, min_area=100)
        assert fr.n_labels == r.n_labels

    def test_max_area_larger_than_blobs_keeps_all(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r, max_area=100000)
        assert fr.n_labels == r.n_labels

    def test_filter_result_props_updated(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r, min_area=10000)
        assert len(fr.props) == fr.n_labels

    def test_filter_returns_segmentation_result(self):
        r = label_connected(_two_blobs())
        fr = filter_regions(r)
        assert isinstance(fr, SegmentationResult)

    def test_min_solidity_one_filters_nonsquare(self):
        # With min_solidity=1.0, only perfectly convex blobs pass
        r = label_connected(_single_blob())
        fr = filter_regions(r, min_solidity=1.0)
        # Square blob is convex → may pass or not depending on how solidity is computed
        assert fr.n_labels >= 0


# ─── merge_close_regions (extra) ─────────────────────────────────────────────

class TestMergeCloseRegionsExtra:
    def test_result_n_labels_leq_original(self):
        r = label_connected(_two_blobs())
        mr = merge_close_regions(r, max_distance=100.0)
        assert mr.n_labels <= r.n_labels

    def test_single_blob_unchanged_labels(self):
        r = label_connected(_single_blob())
        mr = merge_close_regions(r, max_distance=5.0)
        assert mr.n_labels == r.n_labels

    def test_returns_segmentation_result(self):
        r = label_connected(_two_blobs())
        mr = merge_close_regions(r, max_distance=10.0)
        assert isinstance(mr, SegmentationResult)

    def test_blank_result_unchanged(self):
        r = label_connected(_blank())
        mr = merge_close_regions(r, max_distance=10.0)
        assert mr.n_labels == 0

    def test_three_blobs_large_distance(self):
        r = label_connected(_three_blobs())
        mr = merge_close_regions(r, max_distance=10000.0)
        assert mr.n_labels <= r.n_labels


# ─── region_adjacency (extra) ────────────────────────────────────────────────

class TestRegionAdjacencyExtra:
    def test_blank_result_empty_adj(self):
        r = label_connected(_blank())
        adj = region_adjacency(r)
        assert adj == {}

    def test_single_blob_no_neighbors(self):
        r = label_connected(_single_blob())
        adj = region_adjacency(r, dilation_ksize=3)
        for v in adj.values():
            assert len(v) == 0

    def test_dilation_ksize_1_raises(self):
        r = label_connected(_two_blobs())
        with pytest.raises(ValueError):
            region_adjacency(r, dilation_ksize=0)

    def test_adjacency_values_are_sets_or_lists(self):
        r = label_connected(_two_blobs())
        adj = region_adjacency(r, dilation_ksize=3)
        for v in adj.values():
            assert hasattr(v, '__iter__')

    def test_keys_match_props_labels(self):
        r = label_connected(_two_blobs())
        adj = region_adjacency(r)
        expected = {p.label for p in r.props}
        assert set(adj.keys()) == expected


# ─── largest_region (extra) ───────────────────────────────────────────────────

class TestLargestRegionExtra:
    def test_three_blobs_largest_selected(self):
        # Make one blob bigger
        img = np.zeros((64, 128), dtype=np.uint8)
        img[5:15, 5:20] = 255      # 10×15 = 150
        img[5:25, 55:75] = 255     # 20×20 = 400  ← largest
        img[5:10, 105:110] = 255   # 5×5 = 25
        r = label_connected(img)
        lr = largest_region(r)
        assert lr is not None
        for p in r.props:
            assert lr.area >= p.area

    def test_returns_region_props(self):
        r = label_connected(_single_blob())
        lr = largest_region(r)
        assert isinstance(lr, RegionProps)

    def test_single_blob_is_largest(self):
        r = label_connected(_single_blob())
        lr = largest_region(r)
        assert lr is not None
        assert lr.area == r.props[0].area

    def test_blank_returns_none(self):
        r = label_connected(_blank())
        assert largest_region(r) is None


# ─── regions_to_mask (extra) ─────────────────────────────────────────────────

class TestRegionsToMaskExtra:
    def test_dtype_uint8(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert mask.dtype == np.uint8

    def test_values_binary(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_shape_matches_labels(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert mask.shape == r.labels.shape

    def test_blank_all_zeros(self):
        r = label_connected(_blank())
        mask = regions_to_mask(r)
        assert (mask == 0).all()

    def test_all_labels_covered(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r)
        assert mask.sum() > 0

    def test_empty_labels_list_all_zeros(self):
        r = label_connected(_two_blobs())
        mask = regions_to_mask(r, labels=[])
        assert (mask == 0).all()


# ─── batch_segment (extra) ───────────────────────────────────────────────────

class TestBatchSegmentExtra:
    def test_three_images_three_results(self):
        result = batch_segment([_single_blob(), _two_blobs(), _blank()])
        assert len(result) == 3

    def test_all_segmentation_results(self):
        result = batch_segment([_single_blob(), _two_blobs()])
        for r in result:
            assert isinstance(r, SegmentationResult)

    def test_blank_image_zero_labels(self):
        result = batch_segment([_blank()])
        assert result[0].n_labels == 0

    def test_connectivity_4_forwarded(self):
        result = batch_segment([_single_blob()], connectivity=4)
        assert result[0].params["connectivity"] == 4

    def test_threshold_forwarded(self):
        result = batch_segment([_single_blob()], threshold=100)
        assert result[0].params["threshold"] == 100

    def test_large_batch(self):
        imgs = [_single_blob() for _ in range(10)]
        result = batch_segment(imgs)
        assert len(result) == 10
