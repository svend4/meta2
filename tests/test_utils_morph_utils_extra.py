"""Extra tests for puzzle_reconstruction/utils/morph_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.morph_utils import (
    MorphConfig,
    apply_erosion,
    apply_dilation,
    apply_opening,
    apply_closing,
    get_skeleton,
    label_regions,
    filter_regions_by_size,
    compute_region_stats,
    batch_morphology,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _binary(h=64, w=64) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[10:h-10, 10:w-10] = 255
    return img


def _two_blobs() -> np.ndarray:
    img = np.zeros((64, 128), dtype=np.uint8)
    img[10:30, 5:40] = 255
    img[10:30, 80:115] = 255
    return img


def _uniform(val=128) -> np.ndarray:
    return np.full((32, 32), val, dtype=np.uint8)


# ─── MorphConfig ──────────────────────────────────────────────────────────────

class TestMorphConfigExtra:
    def test_default_kernel_size(self):
        assert MorphConfig().kernel_size == 3

    def test_default_kernel_shape(self):
        assert MorphConfig().kernel_shape == "rect"

    def test_default_iterations(self):
        assert MorphConfig().iterations == 1

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_size=4)

    def test_zero_kernel_raises(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_size=0)

    def test_negative_iterations_raises(self):
        with pytest.raises(ValueError):
            MorphConfig(iterations=0)

    def test_invalid_kernel_shape_raises(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_shape="diamond")

    def test_valid_ellipse(self):
        cfg = MorphConfig(kernel_shape="ellipse")
        assert cfg.kernel_shape == "ellipse"

    def test_valid_cross(self):
        cfg = MorphConfig(kernel_shape="cross")
        assert cfg.kernel_shape == "cross"

    def test_build_kernel_returns_ndarray(self):
        cfg = MorphConfig()
        k = cfg.build_kernel()
        assert isinstance(k, np.ndarray)

    def test_build_kernel_shape(self):
        cfg = MorphConfig(kernel_size=5)
        k = cfg.build_kernel()
        assert k.shape == (5, 5)


# ─── apply_erosion ────────────────────────────────────────────────────────────

class TestApplyErosionExtra:
    def test_returns_ndarray(self):
        assert isinstance(apply_erosion(_binary()), np.ndarray)

    def test_shape_preserved(self):
        img = _binary(48, 64)
        assert apply_erosion(img).shape == (48, 64)

    def test_binary_erodes_border(self):
        img = _binary()
        out = apply_erosion(img)
        # Eroded image should have fewer white pixels
        assert out.sum() <= img.sum()

    def test_uniform_black_unchanged(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        out = apply_erosion(img)
        assert out.sum() == 0

    def test_none_cfg(self):
        out = apply_erosion(_binary(), cfg=None)
        assert isinstance(out, np.ndarray)

    def test_larger_kernel_erodes_more(self):
        img = _binary()
        cfg3 = MorphConfig(kernel_size=3)
        cfg5 = MorphConfig(kernel_size=5)
        assert apply_erosion(img, cfg5).sum() <= apply_erosion(img, cfg3).sum()


# ─── apply_dilation ───────────────────────────────────────────────────────────

class TestApplyDilationExtra:
    def test_returns_ndarray(self):
        assert isinstance(apply_dilation(_binary()), np.ndarray)

    def test_shape_preserved(self):
        img = _binary(48, 64)
        assert apply_dilation(img).shape == (48, 64)

    def test_dilation_expands(self):
        img = _binary()
        out = apply_dilation(img)
        assert out.sum() >= img.sum()

    def test_uniform_white_unchanged(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        out = apply_dilation(img)
        assert np.all(out == 255)

    def test_none_cfg(self):
        out = apply_dilation(_binary(), cfg=None)
        assert isinstance(out, np.ndarray)


# ─── apply_opening ────────────────────────────────────────────────────────────

class TestApplyOpeningExtra:
    def test_returns_ndarray(self):
        assert isinstance(apply_opening(_binary()), np.ndarray)

    def test_shape_preserved(self):
        img = _binary(48, 64)
        assert apply_opening(img).shape == (48, 64)

    def test_opening_le_original(self):
        img = _binary()
        out = apply_opening(img)
        assert out.sum() <= img.sum()

    def test_none_cfg(self):
        out = apply_opening(_binary(), cfg=None)
        assert isinstance(out, np.ndarray)


# ─── apply_closing ────────────────────────────────────────────────────────────

class TestApplyClosingExtra:
    def test_returns_ndarray(self):
        assert isinstance(apply_closing(_binary()), np.ndarray)

    def test_shape_preserved(self):
        img = _binary(48, 64)
        assert apply_closing(img).shape == (48, 64)

    def test_closing_ge_original(self):
        img = _binary()
        out = apply_closing(img)
        assert out.sum() >= img.sum()

    def test_none_cfg(self):
        out = apply_closing(_binary(), cfg=None)
        assert isinstance(out, np.ndarray)


# ─── get_skeleton ─────────────────────────────────────────────────────────────

class TestGetSkeletonExtra:
    def test_returns_ndarray(self):
        assert isinstance(get_skeleton(_binary()), np.ndarray)

    def test_shape_preserved(self):
        img = _binary(48, 48)
        assert get_skeleton(img).shape == (48, 48)

    def test_all_black_stays_black(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        out = get_skeleton(img)
        assert out.sum() == 0

    def test_skeleton_subset_of_original(self):
        img = _binary()
        out = get_skeleton(img)
        # Skeleton pixels should be within original white region
        assert np.all(out[img == 0] == 0)


# ─── label_regions ────────────────────────────────────────────────────────────

class TestLabelRegionsExtra:
    def test_returns_tuple(self):
        result = label_regions(_binary())
        assert isinstance(result, tuple) and len(result) == 2

    def test_count_and_labels(self):
        n, labels = label_regions(_binary())
        assert isinstance(n, int) and isinstance(labels, np.ndarray)

    def test_two_blobs_count(self):
        n, _ = label_regions(_two_blobs())
        assert n == 2

    def test_black_image_zero_regions(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        n, _ = label_regions(img)
        assert n == 0

    def test_labels_shape_matches_input(self):
        img = _binary(48, 64)
        _, labels = label_regions(img)
        assert labels.shape == (48, 64)


# ─── filter_regions_by_size ───────────────────────────────────────────────────

class TestFilterRegionsBySizeExtra:
    def test_returns_ndarray(self):
        out = filter_regions_by_size(_two_blobs())
        assert isinstance(out, np.ndarray)

    def test_shape_preserved(self):
        img = _two_blobs()
        out = filter_regions_by_size(img)
        assert out.shape == img.shape

    def test_min_area_removes_small(self):
        # Two blobs of similar size; use large min_area to remove all
        img = _two_blobs()
        out = filter_regions_by_size(img, min_area=100000)
        assert out.sum() == 0

    def test_max_area_removes_large(self):
        img = _two_blobs()
        out = filter_regions_by_size(img, max_area=1)
        assert out.sum() == 0

    def test_black_image_returns_black(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        out = filter_regions_by_size(img)
        assert out.sum() == 0


# ─── compute_region_stats ─────────────────────────────────────────────────────

class TestComputeRegionStatsExtra:
    def test_returns_list(self):
        assert isinstance(compute_region_stats(_binary()), list)

    def test_each_element_is_dict(self):
        stats = compute_region_stats(_binary())
        assert all(isinstance(s, dict) for s in stats)

    def test_keys_present(self):
        stats = compute_region_stats(_binary())
        expected = {"label", "area", "cx", "cy"}
        for s in stats:
            assert expected.issubset(s.keys())

    def test_two_blobs_returns_two(self):
        stats = compute_region_stats(_two_blobs())
        assert len(stats) == 2

    def test_black_image_empty_list(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        stats = compute_region_stats(img)
        assert len(stats) == 0

    def test_area_positive(self):
        for s in compute_region_stats(_binary()):
            assert s["area"] > 0


# ─── batch_morphology ─────────────────────────────────────────────────────────

class TestBatchMorphologyExtra:
    def test_returns_list(self):
        result = batch_morphology([_binary()])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_morphology([_binary(), _binary()])
        assert len(result) == 2

    def test_each_element_ndarray(self):
        for out in batch_morphology([_binary(), _two_blobs()]):
            assert isinstance(out, np.ndarray)

    def test_erosion_operation(self):
        result = batch_morphology([_binary()], operation="erosion")
        assert len(result) == 1

    def test_dilation_operation(self):
        result = batch_morphology([_binary()], operation="dilation")
        assert len(result) == 1

    def test_closing_operation(self):
        result = batch_morphology([_binary()], operation="closing")
        assert len(result) == 1

    def test_invalid_operation_raises(self):
        with pytest.raises(ValueError):
            batch_morphology([_binary()], operation="skeletonize")

    def test_none_cfg(self):
        result = batch_morphology([_binary()], cfg=None)
        assert len(result) == 1

    def test_shape_preserved(self):
        img = _binary(48, 64)
        result = batch_morphology([img])
        assert result[0].shape == (48, 64)
