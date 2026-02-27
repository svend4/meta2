"""
Property-based tests for:
  - puzzle_reconstruction.utils.segment_utils
  - puzzle_reconstruction.utils.voting_utils
  - puzzle_reconstruction.utils.transform_utils

Verifies mathematical invariants:
- segment_utils:
    label_mask:          background pixels get label 0; labels in [0, n_labels]
    mask_statistics:     foreground_fraction in [0,1]; fg + bg = total
    mask_bounding_box:   None for empty mask; bbox pixels all foreground
    extract_boundary:    subset of foreground; boundary pixels touch background
    region_info:         area = count of pixels with that label
    filter_regions:      subset; every region satisfies constraints
    mask_from_labels:    only selected labels become foreground
- voting_utils:
    cast_pair_votes:     canonical form (min,max); weight accumulation
    aggregate_pair_votes: descending order; min_votes filter; normalize in [0,1]
    majority_vote:        winner has max count; None on empty
    weighted_vote:        equal weights → arithmetic mean; single element → value
    rank_fusion:          descending order; normalize in [0,1]
    cast_position_votes:  weight accumulation
- transform_utils:
    flip_image:          double flip = identity; shape preserved
    rotate_image:        0° rotation ≈ original; shape preserved
    scale_image:         sx=sy=1.0 → same size; shape predictable
    crop_region:         result shape = (h, w); full crop = original
    affine_from_params:  returns 2×3 matrix; identity params → approx I
    compose_affines:     single matrix → itself; two identities → identity
    batch_rotate:        same length; each has correct shape
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.segment_utils import (
    SegmentConfig,
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
from puzzle_reconstruction.utils.voting_utils import (
    VoteConfig,
    cast_pair_votes,
    aggregate_pair_votes,
    cast_position_votes,
    majority_vote,
    weighted_vote,
    rank_fusion,
    batch_vote,
)
from puzzle_reconstruction.utils.transform_utils import (
    rotate_image,
    flip_image,
    scale_image,
    crop_region,
    affine_from_params,
    compose_affines,
    apply_affine,
    apply_homography,
    batch_rotate,
)

RNG = np.random.default_rng(2027)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_uint8(h: int = 32, w: int = 32, channels: int = 1) -> np.ndarray:
    """Random uint8 image."""
    if channels == 1:
        return RNG.integers(0, 256, size=(h, w), dtype=np.uint8)
    return RNG.integers(0, 256, size=(h, w, channels), dtype=np.uint8)


def _rand_binary_mask(h: int = 16, w: int = 16, density: float = 0.5) -> np.ndarray:
    """Random binary mask (uint8, 0 or 255)."""
    raw = RNG.uniform(0, 1, size=(h, w))
    mask = (raw < density).astype(np.uint8) * 255
    return mask


def _checkerboard_mask(h: int = 8, w: int = 8) -> np.ndarray:
    """Checkerboard pattern as binary mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[::2, ::2] = 255
    m[1::2, 1::2] = 255
    return m


def _identity_affine() -> np.ndarray:
    """2×3 identity affine matrix."""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# segment_utils
# ═══════════════════════════════════════════════════════════════════════════════


class TestSegmentConfig:
    def test_default_valid(self):
        cfg = SegmentConfig()
        assert cfg.min_area >= 0
        assert cfg.max_aspect_ratio > 0
        assert cfg.border_margin >= 0

    @pytest.mark.parametrize("a", [-1, -10])
    def test_invalid_min_area(self, a):
        with pytest.raises(ValueError):
            SegmentConfig(min_area=a)

    @pytest.mark.parametrize("r", [0.0, -1.0])
    def test_invalid_aspect_ratio(self, r):
        with pytest.raises(ValueError):
            SegmentConfig(max_aspect_ratio=r)


class TestLabelMask:
    def test_empty_mask_no_labels(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        labels, n = label_mask(mask)
        assert n == 0
        np.testing.assert_array_equal(labels, np.zeros((8, 8), dtype=np.int32))

    def test_full_mask_one_label(self):
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        labels, n = label_mask(mask)
        assert n == 1
        assert (labels > 0).all()

    def test_labels_in_valid_range(self):
        mask = _rand_binary_mask(12, 12, density=0.6)
        labels, n = label_mask(mask)
        assert int(labels.min()) >= 0
        assert int(labels.max()) <= n

    def test_background_labeled_zero(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        labels, n = label_mask(mask)
        # Corners should be background = 0
        assert labels[0, 0] == 0
        assert labels[5, 5] == 0

    def test_isolated_pixels_separate_labels(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[0, 0] = 255
        mask[6, 6] = 255
        labels, n = label_mask(mask)
        assert n == 2
        assert labels[0, 0] != labels[6, 6]

    def test_single_pixel(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 255
        labels, n = label_mask(mask)
        assert n == 1
        assert labels[2, 2] == 1

    def test_output_shape(self):
        mask = _rand_binary_mask(10, 12)
        labels, _ = label_mask(mask)
        assert labels.shape == (10, 12)


class TestRegionInfo:
    def test_area_matches_pixel_count(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 255  # 3×4 = 12 pixels
        labels, n = label_mask(mask)
        info = region_info(labels, 1)
        assert info.area == 12

    def test_bbox_contains_all_pixels(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[3:7, 4:9] = 255
        labels, _ = label_mask(mask)
        info = region_info(labels, 1)
        y0, x0, y1, x1 = info.bbox
        assert y0 <= 3 and y1 >= 7
        assert x0 <= 4 and x1 >= 9

    def test_centroid_within_bbox(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[3:7, 4:9] = 255
        labels, _ = label_mask(mask)
        info = region_info(labels, 1)
        y0, x0, y1, x1 = info.bbox
        cy, cx = info.centroid
        assert y0 <= cy <= y1
        assert x0 <= cx <= x1

    def test_aspect_ratio_ge_one(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:10, 3:7] = 255
        labels, _ = label_mask(mask)
        info = region_info(labels, 1)
        assert info.aspect_ratio >= 1.0

    def test_to_dict_keys(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        labels, _ = label_mask(mask)
        d = region_info(labels, 1).to_dict()
        for key in ("label", "area", "bbox", "centroid", "aspect_ratio"):
            assert key in d


class TestFilterRegions:
    def test_result_subset(self):
        mask = _rand_binary_mask(20, 20, density=0.4)
        labels, n = label_mask(mask)
        regions = all_regions(labels, n)
        filtered = filter_regions(regions)
        assert set(r.label for r in filtered).issubset(
            set(r.label for r in regions))

    def test_min_area_respected(self):
        mask = _rand_binary_mask(20, 20, density=0.4)
        labels, n = label_mask(mask)
        regions = all_regions(labels, n)
        cfg = SegmentConfig(min_area=100)
        filtered = filter_regions(regions, cfg)
        assert all(r.area >= 100 for r in filtered)

    def test_aspect_ratio_respected(self):
        mask = _rand_binary_mask(20, 20, density=0.4)
        labels, n = label_mask(mask)
        regions = all_regions(labels, n)
        cfg = SegmentConfig(max_aspect_ratio=3.0)
        filtered = filter_regions(regions, cfg)
        assert all(r.aspect_ratio <= 3.0 for r in filtered)

    def test_empty_input(self):
        result = filter_regions([])
        assert result == []


class TestLargestRegion:
    def test_returns_none_for_empty(self):
        assert largest_region([]) is None

    def test_returns_max_area(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:3, 0:3] = 255   # 9 pixels
        mask[10:16, 10:16] = 255  # 36 pixels
        labels, n = label_mask(mask)
        regions = all_regions(labels, n)
        lg = largest_region(regions)
        assert lg is not None
        assert lg.area == max(r.area for r in regions)


class TestMaskFromLabels:
    def test_empty_keep_ids(self):
        mask = _rand_binary_mask(10, 10)
        labels, n = label_mask(mask)
        result = mask_from_labels(labels, [])
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_selected_labels_foreground(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[0:4, 0:4] = 255
        mask[8:12, 8:12] = 255
        labels, n = label_mask(mask)
        result = mask_from_labels(labels, [1])
        # Exactly label 1 becomes foreground
        assert int(result.max()) == 255
        # The region of label 1 should be 255
        label1_pixels = (labels == 1)
        assert (result[label1_pixels] == 255).all()

    def test_output_dtype_uint8(self):
        mask = _rand_binary_mask(10, 10)
        labels, n = label_mask(mask)
        result = mask_from_labels(labels, list(range(1, n + 1)))
        assert result.dtype == np.uint8


class TestMaskStatistics:
    def test_all_zero_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        stats = mask_statistics(mask)
        assert stats["foreground_pixels"] == 0
        assert stats["foreground_fraction"] == pytest.approx(0.0)
        assert stats["total_pixels"] == 100

    def test_all_ones_mask(self):
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        stats = mask_statistics(mask)
        assert stats["foreground_pixels"] == 100
        assert stats["foreground_fraction"] == pytest.approx(1.0)

    def test_fraction_in_range(self):
        for _ in range(20):
            mask = _rand_binary_mask(12, 12)
            stats = mask_statistics(mask)
            assert 0.0 <= stats["foreground_fraction"] <= 1.0

    def test_fg_plus_bg_equals_total(self):
        for _ in range(20):
            mask = _rand_binary_mask(10, 10)
            stats = mask_statistics(mask)
            assert stats["foreground_pixels"] + stats["background_pixels"] == stats["total_pixels"]

    def test_total_equals_size(self):
        for h, w in [(5, 7), (10, 10), (3, 20)]:
            mask = _rand_binary_mask(h, w)
            stats = mask_statistics(mask)
            assert stats["total_pixels"] == h * w


class TestMaskBoundingBox:
    def test_all_zero_returns_none(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        assert mask_bounding_box(mask) is None

    def test_bbox_contains_all_fg_pixels(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:7, 3:9] = 255
        bbox = mask_bounding_box(mask)
        assert bbox is not None
        y0, x0, y1, x1 = bbox
        ys, xs = np.where(mask > 0)
        assert y0 <= int(ys.min())
        assert y1 >= int(ys.max()) + 1
        assert x0 <= int(xs.min())
        assert x1 >= int(xs.max()) + 1

    def test_single_pixel(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3, 5] = 255
        bbox = mask_bounding_box(mask)
        assert bbox == (3, 5, 4, 6)

    def test_returns_tuple_of_four(self):
        mask = _rand_binary_mask(10, 10, density=0.5)
        bbox = mask_bounding_box(mask)
        if bbox is not None:
            assert len(bbox) == 4


class TestExtractBoundary:
    def test_empty_mask_empty_boundary(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        boundary = extract_boundary(mask)
        np.testing.assert_array_equal(boundary, np.zeros_like(mask))

    def test_boundary_subset_of_foreground(self):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        boundary = extract_boundary(mask)
        # Boundary must be a subset of foreground
        assert ((boundary > 0) & (mask == 0)).sum() == 0

    def test_all_pixels_mask_boundary_only_edge(self):
        mask = np.ones((8, 8), dtype=np.uint8) * 255
        boundary = extract_boundary(mask)
        # For a full mask, interior pixels are interior (not boundary)
        # only edge pixels should be boundary
        interior = boundary[1:-1, 1:-1]
        np.testing.assert_array_equal(interior, np.zeros_like(interior))

    def test_output_uint8(self):
        mask = _rand_binary_mask(10, 10)
        boundary = extract_boundary(mask)
        assert boundary.dtype == np.uint8

    def test_output_shape_matches(self):
        for h, w in [(8, 10), (15, 12)]:
            mask = _rand_binary_mask(h, w)
            boundary = extract_boundary(mask)
            assert boundary.shape == (h, w)


# ═══════════════════════════════════════════════════════════════════════════════
# voting_utils
# ═══════════════════════════════════════════════════════════════════════════════


class TestVoteConfig:
    def test_default_valid(self):
        cfg = VoteConfig()
        assert cfg.min_votes >= 1
        assert cfg.rrf_k > 0

    @pytest.mark.parametrize("mv", [0, -1])
    def test_invalid_min_votes(self, mv):
        with pytest.raises(ValueError):
            VoteConfig(min_votes=mv)

    @pytest.mark.parametrize("k", [0.0, -1.0])
    def test_invalid_rrf_k(self, k):
        with pytest.raises(ValueError):
            VoteConfig(rrf_k=k)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            VoteConfig(weights=[-0.1, 1.0])


class TestCastPairVotes:
    def test_canonical_form(self):
        """Votes are stored as (min, max) tuples."""
        pair_lists = [[(3, 1), (5, 2)], [(1, 3)]]
        votes = cast_pair_votes(pair_lists)
        # (3,1) and (1,3) should both map to (1,3)
        assert (1, 3) in votes

    def test_weight_accumulation(self):
        pair_lists = [[(0, 1)], [(0, 1)], [(0, 1)]]
        votes = cast_pair_votes(pair_lists)
        assert votes[(0, 1)] == pytest.approx(3.0)

    def test_weighted_accumulation(self):
        pair_lists = [[(0, 1)], [(0, 1)]]
        weights = [2.0, 3.0]
        votes = cast_pair_votes(pair_lists, weights=weights)
        assert votes[(0, 1)] == pytest.approx(5.0)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_pair_votes([[(0, 1)]], weights=[1.0, 2.0])

    def test_empty_input(self):
        votes = cast_pair_votes([])
        assert votes == {}

    def test_empty_pair_list(self):
        votes = cast_pair_votes([[]])
        assert votes == {}

    def test_all_values_positive(self):
        pair_lists = [[(i, j) for j in range(i + 1, 5)] for i in range(4)]
        votes = cast_pair_votes(pair_lists)
        assert all(v > 0 for v in votes.values())


class TestAggregatePairVotes:
    def test_descending_order(self):
        votes = {(0, 1): 3.0, (1, 2): 1.0, (0, 2): 2.0}
        result = aggregate_pair_votes(votes)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_min_votes_filter(self):
        votes = {(0, 1): 3.0, (1, 2): 1.0}
        cfg = VoteConfig(min_votes=2)
        result = aggregate_pair_votes(votes, cfg)
        pairs = [p for p, _ in result]
        assert (1, 2) not in pairs

    def test_normalize_in_range(self):
        votes = {(0, 1): 5.0, (1, 2): 3.0, (0, 2): 1.0}
        cfg = VoteConfig(normalize=True)
        result = aggregate_pair_votes(votes, cfg)
        scores = [s for _, s in result]
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_max_score_is_one_when_normalized(self):
        votes = {(0, 1): 5.0, (1, 2): 3.0}
        result = aggregate_pair_votes(votes)
        scores = [s for _, s in result]
        assert max(scores) == pytest.approx(1.0)

    def test_empty_votes(self):
        result = aggregate_pair_votes({})
        assert result == []


class TestCastPositionVotes:
    def test_weight_accumulation(self):
        pos_lists = [{0: 0.5, 1: 0.3}, {0: 0.4}]
        result = cast_position_votes(pos_lists)
        assert result[0] == pytest.approx(0.9)
        assert result[1] == pytest.approx(0.3)

    def test_weighted(self):
        pos_lists = [{0: 1.0}]
        result = cast_position_votes(pos_lists, weights=[3.0])
        assert result[0] == pytest.approx(3.0)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_position_votes([{0: 1.0}], weights=[1.0, 2.0])

    def test_empty_input(self):
        result = cast_position_votes([])
        assert result == {}


class TestMajorityVote:
    def test_returns_none_for_empty(self):
        assert majority_vote([]) is None

    def test_single_element(self):
        assert majority_vote(["a"]) == "a"

    def test_most_frequent(self):
        values = ["a", "b", "a", "c", "a"]
        assert majority_vote(values) == "a"

    def test_integers(self):
        values = [1, 2, 1, 3, 1, 2]
        assert majority_vote(values) == 1

    def test_all_same(self):
        values = [42] * 10
        assert majority_vote(values) == 42

    def test_uniform_distribution_returns_a_value(self):
        values = [1, 2, 3]
        result = majority_vote(values)
        assert result in {1, 2, 3}


class TestWeightedVote:
    def test_empty_input(self):
        assert weighted_vote([]) == pytest.approx(0.0)

    def test_equal_weights_equals_mean(self):
        for _ in range(20):
            values = RNG.uniform(-10, 10, size=10).tolist()
            weights = [1.0] * 10
            assert weighted_vote(values, weights) == pytest.approx(
                sum(values) / len(values), rel=1e-9)

    def test_single_element(self):
        assert weighted_vote([7.5]) == pytest.approx(7.5)
        assert weighted_vote([7.5], [1.0]) == pytest.approx(7.5)

    def test_weight_concentration(self):
        values = [0.0, 0.0, 100.0]
        weights = [0.001, 0.001, 100.0]
        result = weighted_vote(values, weights)
        assert result > 99.0

    def test_zero_weight_sum_returns_zero(self):
        assert weighted_vote([1.0, 2.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0, 2.0], [1.0])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0, 2.0], [-1.0, 1.0])

    def test_scaling_weights_same_result(self):
        values = [1.0, 2.0, 3.0]
        w = [0.1, 0.3, 0.6]
        r1 = weighted_vote(values, w)
        r2 = weighted_vote(values, [x * 10 for x in w])
        assert r1 == pytest.approx(r2, rel=1e-9)

    def test_result_in_value_range(self):
        for _ in range(20):
            values = RNG.uniform(-5, 5, size=8).tolist()
            weights = RNG.uniform(0.1, 1.0, size=8).tolist()
            result = weighted_vote(values, weights)
            assert min(values) - 1e-10 <= result <= max(values) + 1e-10


class TestRankFusion:
    def test_empty_input(self):
        result = rank_fusion([])
        assert result == []

    def test_descending_order(self):
        ranked_lists = [["a", "b", "c"], ["b", "a", "c"]]
        result = rank_fusion(ranked_lists)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_normalize_in_range(self):
        ranked_lists = [["a", "b", "c"], ["a", "c", "b"]]
        cfg = VoteConfig(normalize=True)
        result = rank_fusion(ranked_lists, cfg)
        for _, s in result:
            assert 0.0 <= s <= 1.0

    def test_top_item_is_max_score_one(self):
        ranked_lists = [["a", "b", "c"]]
        cfg = VoteConfig(normalize=True)
        result = rank_fusion(ranked_lists, cfg)
        if result:
            assert result[0][1] == pytest.approx(1.0)

    def test_all_elements_appear(self):
        ranked_lists = [["a", "b"], ["b", "c"]]
        result = rank_fusion(ranked_lists)
        items = [item for item, _ in result]
        assert set(items) == {"a", "b", "c"}

    def test_single_list_all_included(self):
        ranked_lists = [["x", "y", "z"]]
        result = rank_fusion(ranked_lists)
        items = {item for item, _ in result}
        assert items == {"x", "y", "z"}


class TestBatchVote:
    def test_same_length(self):
        batch = [
            [[(0, 1), (1, 2)], [(0, 2)]],
            [[(0, 3)], [(1, 3)]],
        ]
        result = batch_vote(batch)
        assert len(result) == 2

    def test_each_result_is_list(self):
        batch = [[[(0, 1)]], [[(2, 3)]]]
        result = batch_vote(batch)
        for r in result:
            assert isinstance(r, list)

    def test_empty_batch(self):
        result = batch_vote([])
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# transform_utils
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlipImage:
    def test_double_horizontal_flip_identity(self):
        for _ in range(5):
            img = _rand_uint8(16, 16)
            result = flip_image(flip_image(img, 1), 1)
            np.testing.assert_array_equal(result, img)

    def test_double_vertical_flip_identity(self):
        for _ in range(5):
            img = _rand_uint8(16, 16)
            result = flip_image(flip_image(img, 0), 0)
            np.testing.assert_array_equal(result, img)

    def test_double_both_flip_identity(self):
        for _ in range(5):
            img = _rand_uint8(16, 16)
            result = flip_image(flip_image(img, -1), -1)
            np.testing.assert_array_equal(result, img)

    def test_shape_preserved(self):
        for h, w in [(8, 12), (16, 16), (20, 10)]:
            img = _rand_uint8(h, w)
            for mode in [0, 1, -1]:
                assert flip_image(img, mode).shape == img.shape

    def test_dtype_preserved(self):
        img = _rand_uint8(10, 10)
        assert flip_image(img, 1).dtype == np.uint8

    def test_horizontal_flip_reverses_columns(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        flipped = flip_image(img, 1)
        np.testing.assert_array_equal(flipped, img[:, ::-1])

    def test_vertical_flip_reverses_rows(self):
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        flipped = flip_image(img, 0)
        np.testing.assert_array_equal(flipped, img[::-1, :])


class TestRotateImage:
    def test_zero_angle_near_identity(self):
        img = _rand_uint8(16, 16)
        result = rotate_image(img, 0.0)
        np.testing.assert_array_almost_equal(result, img)

    def test_shape_preserved(self):
        for h, w in [(10, 14), (20, 20)]:
            img = _rand_uint8(h, w)
            assert rotate_image(img, 45.0).shape == img.shape

    def test_dtype_preserved(self):
        img = _rand_uint8(12, 12)
        assert rotate_image(img, 30.0).dtype == np.uint8

    def test_360_near_identity(self):
        """Rotating by 360° should approximate the original."""
        img = _rand_uint8(16, 16)
        result = rotate_image(img, 360.0)
        diff = np.abs(result.astype(float) - img.astype(float))
        # Most pixels should be close; allow some border interpolation error
        assert float(diff.mean()) < 5.0

    def test_180_same_as_double_90(self):
        img = _rand_uint8(12, 12)
        r180 = rotate_image(img, 180.0)
        r90_twice = rotate_image(rotate_image(img, 90.0), 90.0)
        diff = np.abs(r180.astype(float) - r90_twice.astype(float))
        assert float(diff.mean()) < 2.0


class TestScaleImage:
    def test_scale_one_same_size(self):
        img = _rand_uint8(16, 16)
        result = scale_image(img, 1.0)
        assert result.shape == img.shape

    def test_scale_double_size(self):
        img = _rand_uint8(8, 10)
        result = scale_image(img, 2.0)
        assert result.shape == (16, 20)

    def test_scale_half_size(self):
        img = _rand_uint8(16, 20)
        result = scale_image(img, 0.5)
        assert result.shape == (8, 10)

    def test_asymmetric_scaling(self):
        img = _rand_uint8(10, 10)
        result = scale_image(img, sx=2.0, sy=0.5)
        assert result.shape == (5, 20)

    def test_dtype_preserved(self):
        img = _rand_uint8(10, 10)
        assert scale_image(img, 1.5).dtype == np.uint8

    def test_minimum_size_one(self):
        img = _rand_uint8(5, 5)
        result = scale_image(img, 0.001)
        assert result.shape[0] >= 1 and result.shape[1] >= 1


class TestCropRegion:
    def test_full_crop_equals_original(self):
        img = _rand_uint8(10, 12)
        h, w = img.shape
        result = crop_region(img, 0, 0, w, h)
        np.testing.assert_array_equal(result, img)

    def test_crop_shape_correct(self):
        img = _rand_uint8(20, 20)
        result = crop_region(img, 2, 3, 8, 6)
        assert result.shape == (6, 8)

    def test_clamp_prevents_out_of_bounds(self):
        img = _rand_uint8(10, 10)
        # Request region larger than image; should be clamped
        result = crop_region(img, -5, -5, 30, 30, clamp=True)
        assert result.shape == (10, 10)

    def test_empty_region_raises(self):
        img = _rand_uint8(10, 10)
        with pytest.raises(ValueError):
            crop_region(img, 5, 5, 0, 0, clamp=False)

    def test_single_pixel_crop(self):
        img = _rand_uint8(10, 10)
        result = crop_region(img, 3, 4, 1, 1)
        assert result.shape == (1, 1)
        assert result[0, 0] == img[4, 3]


class TestAffineFromParams:
    def test_returns_2x3(self):
        M = affine_from_params()
        assert M.shape == (2, 3)

    def test_dtype_float32(self):
        M = affine_from_params()
        assert M.dtype == np.float32

    def test_identity_params(self):
        """angle=0, tx=0, ty=0, sx=1 → approximately identity."""
        M = affine_from_params(angle=0.0, tx=0.0, ty=0.0, sx=1.0)
        expected = _identity_affine()
        np.testing.assert_array_almost_equal(M, expected, decimal=5)

    def test_translation_only(self):
        M = affine_from_params(tx=5.0, ty=3.0)
        assert M[0, 2] == pytest.approx(5.0, abs=1e-4)
        assert M[1, 2] == pytest.approx(3.0, abs=1e-4)

    def test_uniform_scale(self):
        M = affine_from_params(sx=2.0, sy=2.0)
        # Scale factor appears in M[0,0] and M[1,1]
        assert abs(M[0, 0]) == pytest.approx(2.0, abs=1e-4)
        assert abs(M[1, 1]) == pytest.approx(2.0, abs=1e-4)


class TestComposeAffines:
    def test_single_matrix_returns_same(self):
        M = affine_from_params(angle=30.0, tx=5.0)
        result = compose_affines([M])
        np.testing.assert_array_almost_equal(result, M, decimal=5)

    def test_two_identities(self):
        I = _identity_affine()
        result = compose_affines([I, I])
        np.testing.assert_array_almost_equal(result, I, decimal=5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_affines([])

    def test_returns_2x3(self):
        M1 = affine_from_params(tx=5.0)
        M2 = affine_from_params(ty=3.0)
        result = compose_affines([M1, M2])
        assert result.shape == (2, 3)

    def test_dtype_float32(self):
        result = compose_affines([_identity_affine()])
        assert result.dtype == np.float32


class TestBatchRotate:
    def test_same_length(self):
        images = [_rand_uint8(10, 10) for _ in range(5)]
        result = batch_rotate(images, angle=30.0)
        assert len(result) == 5

    def test_each_has_correct_shape(self):
        images = [_rand_uint8(10, 12)]
        result = batch_rotate(images, angle=45.0)
        assert result[0].shape == (10, 12)

    def test_empty_list(self):
        result = batch_rotate([], angle=90.0)
        assert result == []

    def test_zero_angle_near_identity(self):
        img = _rand_uint8(12, 12)
        result = batch_rotate([img], angle=0.0)
        np.testing.assert_array_almost_equal(result[0], img)
