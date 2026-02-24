"""Extra tests for puzzle_reconstruction.algorithms.region_segmenter."""
import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)

def _white(h=64, w=64):
    return np.full((h, w), 255, dtype=np.uint8)

def _two_blobs():
    img = np.zeros((64, 128), dtype=np.uint8)
    img[10:30, 10:40] = 255
    img[10:30, 80:110] = 255
    return img

def _rgb_white(h=32, w=32):
    return np.full((h, w, 3), 255, dtype=np.uint8)

def _segmented():
    return label_connected(_two_blobs(), threshold=128)


# ─── TestRegionPropsExtra ─────────────────────────────────────────────────────

class TestRegionPropsExtra:
    def test_label_zero_valid(self):
        r = RegionProps(label=0, area=0, bbox=(0, 0, 0, 0),
                        centroid=(0.0, 0.0), aspect_ratio=1.0,
                        solidity=0.0, perimeter=0.0)
        assert r.label == 0

    def test_label_stored(self):
        r = RegionProps(label=5, area=100, bbox=(1, 2, 10, 20),
                        centroid=(5.0, 10.0), aspect_ratio=0.5,
                        solidity=0.9, perimeter=50.0)
        assert r.label == 5

    def test_bbox_four_ints(self):
        r = RegionProps(label=1, area=50, bbox=(3, 7, 15, 25),
                        centroid=(10.0, 15.0), aspect_ratio=1.0,
                        solidity=1.0, perimeter=40.0)
        assert len(r.bbox) == 4

    def test_centroid_is_tuple(self):
        r = RegionProps(label=1, area=50, bbox=(0, 0, 10, 10),
                        centroid=(5.0, 5.0), aspect_ratio=1.0,
                        solidity=1.0, perimeter=40.0)
        assert isinstance(r.centroid, tuple)

    def test_area_stored(self):
        r = RegionProps(label=1, area=1234, bbox=(0, 0, 50, 50),
                        centroid=(25.0, 25.0), aspect_ratio=1.0,
                        solidity=0.9, perimeter=200.0)
        assert r.area == 1234

    def test_solidity_stored(self):
        r = RegionProps(label=1, area=50, bbox=(0, 0, 10, 10),
                        centroid=(5.0, 5.0), aspect_ratio=1.0,
                        solidity=0.75, perimeter=40.0)
        assert r.solidity == pytest.approx(0.75)


# ─── TestSegmentationResultExtra ─────────────────────────────────────────────

class TestSegmentationResultExtra:
    def test_n_labels_zero_on_blank(self):
        r = label_connected(_blank())
        assert r.n_labels == 0

    def test_n_labels_one_on_white(self):
        r = label_connected(_white())
        assert r.n_labels == 1

    def test_props_length_matches_n_labels(self):
        r = _segmented()
        assert len(r.props) == r.n_labels

    def test_threshold_in_params(self):
        r = label_connected(_white(), threshold=200)
        assert r.params["threshold"] == 200

    def test_connectivity_in_params(self):
        r = label_connected(_white(), connectivity=4)
        assert r.params["connectivity"] == 4


# ─── TestLabelConnectedExtra ─────────────────────────────────────────────────

class TestLabelConnectedExtra:
    def test_non_square_image(self):
        img = np.zeros((32, 128), dtype=np.uint8)
        img[5:20, 10:60] = 255
        r = label_connected(img, threshold=128)
        assert r.n_labels >= 1

    def test_threshold_128_grey_image(self):
        img = np.full((32, 32), 127, dtype=np.uint8)
        r = label_connected(img, threshold=128)
        assert r.n_labels == 0

    def test_threshold_127_grey_image(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        r = label_connected(img, threshold=127)
        assert r.n_labels == 1

    def test_rgb_one_blob(self):
        r = label_connected(_rgb_white())
        assert r.n_labels == 1

    def test_labels_non_negative(self):
        r = label_connected(_two_blobs())
        assert r.labels.min() >= 0

    def test_single_pixel_blob(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 8] = 255
        r = label_connected(img, threshold=128)
        assert r.n_labels == 1
        assert r.props[0].area == 1


# ─── TestComputeRegionPropsExtra ─────────────────────────────────────────────

class TestComputeRegionPropsExtra:
    def test_single_region_list_of_one(self):
        r = label_connected(_white())
        result = compute_region_props(r)
        assert len(result) == 1

    def test_areas_positive(self):
        r = _segmented()
        for p in compute_region_props(r):
            assert p.area > 0

    def test_two_blobs_two_props(self):
        r = _segmented()
        result = compute_region_props(r)
        assert len(result) == 2

    def test_centroids_within_image(self):
        r = _segmented()
        for p in compute_region_props(r):
            a, b = p.centroid
            # Centroid coordinates should fall within image dimensions
            assert 0 <= a <= 128
            assert 0 <= b <= 128


# ─── TestFilterRegionsExtra ───────────────────────────────────────────────────

class TestFilterRegionsExtra:
    def test_min_area_zero_keeps_all(self):
        r = _segmented()
        f = filter_regions(r, min_area=0)
        assert f.n_labels == r.n_labels

    def test_max_area_less_than_min_blob_filters_all(self):
        r = _segmented()
        min_area = min(p.area for p in r.props)
        # max_area strictly less than the smallest blob removes everything
        if min_area > 1:
            f = filter_regions(r, max_area=min_area - 1)
            assert f.n_labels == 0

    def test_min_equals_max_valid(self):
        r = _segmented()
        area = r.props[0].area
        f = filter_regions(r, min_area=area, max_area=area)
        assert isinstance(f, SegmentationResult)

    def test_min_area_large_keeps_none(self):
        r = _segmented()
        f = filter_regions(r, min_area=10000)
        assert f.n_labels == 0

    def test_returns_segmentation_result(self):
        r = _segmented()
        f = filter_regions(r, min_area=0, max_area=100000)
        assert isinstance(f, SegmentationResult)

    def test_min_aspect_ratio_filter(self):
        r = _segmented()
        f = filter_regions(r, min_aspect_ratio=0.1)
        assert isinstance(f, SegmentationResult)


# ─── TestMergeCloseRegionsExtra ───────────────────────────────────────────────

class TestMergeCloseRegionsExtra:
    def test_ksize_1_returns_result(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=5.0)
        assert isinstance(m, SegmentationResult)

    def test_empty_state_ok(self):
        r = label_connected(_blank())
        m = merge_close_regions(r, max_distance=5.0)
        assert m.n_labels == 0

    def test_n_labels_not_increases(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=5.0)
        assert m.n_labels <= r.n_labels

    def test_zero_distance_same_n_labels(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=0.0)
        assert m.n_labels == r.n_labels

    def test_labels_shape_preserved(self):
        r = _segmented()
        m = merge_close_regions(r, max_distance=5.0)
        assert m.labels.shape == r.labels.shape


# ─── TestRegionAdjacencyExtra ────────────────────────────────────────────────

class TestRegionAdjacencyExtra:
    def test_ksize_1_valid(self):
        r = _segmented()
        adj = region_adjacency(r, dilation_ksize=1)
        assert isinstance(adj, dict)

    def test_all_labels_in_keys(self):
        r = _segmented()
        adj = region_adjacency(r)
        labels = {p.label for p in r.props}
        assert labels == set(adj.keys())

    def test_adjacency_values_are_lists(self):
        r = _segmented()
        adj = region_adjacency(r)
        for val in adj.values():
            assert isinstance(val, list)

    def test_single_region_no_neighbors(self):
        r = label_connected(_white())
        adj = region_adjacency(r)
        for label, neighbors in adj.items():
            assert neighbors == []

    def test_ksize_negative_raises(self):
        r = _segmented()
        with pytest.raises(ValueError):
            region_adjacency(r, dilation_ksize=-1)


# ─── TestLargestRegionExtra ───────────────────────────────────────────────────

class TestLargestRegionExtra:
    def test_returns_largest_of_two(self):
        img = np.zeros((64, 128), dtype=np.uint8)
        img[5:30, 5:60] = 255   # large
        img[5:10, 90:95] = 255  # small
        r = label_connected(img, threshold=128)
        lr = largest_region(r)
        assert lr is not None
        assert all(lr.area >= p.area for p in r.props)

    def test_label_matches_props(self):
        r = _segmented()
        lr = largest_region(r)
        assert lr.label in {p.label for p in r.props}

    def test_area_positive(self):
        r = _segmented()
        lr = largest_region(r)
        assert lr.area > 0

    def test_centroid_valid(self):
        r = label_connected(_white())
        lr = largest_region(r)
        assert lr is not None
        cy, cx = lr.centroid
        assert isinstance(cy, float)
        assert isinstance(cx, float)


# ─── TestRegionsToMaskExtra ──────────────────────────────────────────────────

class TestRegionsToMaskExtra:
    def test_empty_label_list_black_mask(self):
        r = _segmented()
        mask = regions_to_mask(r, labels=[])
        assert mask.max() == 0

    def test_mask_has_white_for_blobs(self):
        r = _segmented()
        mask = regions_to_mask(r)
        assert mask.max() == 255

    def test_single_label_subset(self):
        r = _segmented()
        if r.n_labels >= 1:
            label = r.props[0].label
            mask = regions_to_mask(r, labels=[label])
            assert mask.sum() > 0

    def test_mask_non_negative(self):
        r = _segmented()
        mask = regions_to_mask(r)
        assert mask.min() >= 0

    def test_two_blob_mask_coverage(self):
        r = _segmented()
        mask_all = regions_to_mask(r)
        # Should have white pixels covering both blobs
        assert (mask_all > 0).sum() > 0


# ─── TestBatchSegmentExtra ────────────────────────────────────────────────────

class TestBatchSegmentExtra:
    def test_five_images(self):
        imgs = [_white()] * 3 + [_blank()] * 2
        results = batch_segment(imgs)
        assert len(results) == 5

    def test_rgb_images(self):
        imgs = [_rgb_white(32, 32), _rgb_white(32, 32)]
        results = batch_segment(imgs)
        for r in results:
            assert isinstance(r, SegmentationResult)

    def test_threshold_0_white_image(self):
        imgs = [_white()]
        results = batch_segment(imgs, threshold=0)
        assert results[0].n_labels >= 1

    def test_non_square_batch(self):
        imgs = [np.zeros((32, 64), dtype=np.uint8)]
        results = batch_segment(imgs)
        assert len(results) == 1

    def test_all_have_labels_array(self):
        imgs = [_two_blobs(), _white()]
        for r in batch_segment(imgs):
            assert hasattr(r, 'labels')
            assert isinstance(r.labels, np.ndarray)
