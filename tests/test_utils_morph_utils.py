"""Тесты для puzzle_reconstruction/utils/morph_utils.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_binary(h=64, w=64):
    """White square on black background."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[10:h-10, 10:w-10] = 255
    return img


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_two_blobs():
    """Two separate white rectangles on black."""
    img = np.zeros((64, 128), dtype=np.uint8)
    img[10:30, 5:40] = 255    # left blob
    img[10:30, 80:115] = 255  # right blob
    return img


# ─── MorphConfig ──────────────────────────────────────────────────────────────

class TestMorphConfig:
    def test_defaults(self):
        cfg = MorphConfig()
        assert cfg.kernel_size == 3
        assert cfg.kernel_shape == "rect"
        assert cfg.iterations == 1

    def test_even_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size"):
            MorphConfig(kernel_size=4)

    def test_zero_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size"):
            MorphConfig(kernel_size=0)

    def test_negative_kernel_size_raises(self):
        with pytest.raises(ValueError, match="kernel_size"):
            MorphConfig(kernel_size=-1)

    def test_invalid_kernel_shape_raises(self):
        with pytest.raises(ValueError, match="kernel_shape"):
            MorphConfig(kernel_shape="diamond")

    def test_valid_shapes(self):
        for s in ("rect", "ellipse", "cross"):
            cfg = MorphConfig(kernel_shape=s)
            assert cfg.kernel_shape == s

    def test_zero_iterations_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            MorphConfig(iterations=0)

    def test_negative_iterations_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            MorphConfig(iterations=-1)

    def test_build_kernel_shape(self):
        cfg = MorphConfig(kernel_size=5)
        k = cfg.build_kernel()
        assert k.shape == (5, 5)

    def test_build_kernel_returns_ndarray(self):
        cfg = MorphConfig()
        k = cfg.build_kernel()
        assert isinstance(k, np.ndarray)

    def test_odd_kernel_size_valid(self):
        for s in (1, 3, 5, 7):
            cfg = MorphConfig(kernel_size=s)
            assert cfg.kernel_size == s


# ─── apply_erosion ────────────────────────────────────────────────────────────

class TestApplyErosion:
    def test_returns_ndarray(self):
        img = make_binary()
        result = apply_erosion(img)
        assert isinstance(result, np.ndarray)

    def test_same_shape_gray(self):
        img = make_binary(h=32, w=48)
        result = apply_erosion(img)
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        img = make_binary()
        result = apply_erosion(img)
        assert result.dtype == np.uint8

    def test_erosion_shrinks_foreground(self):
        img = make_binary()
        result = apply_erosion(img)
        assert result.sum() <= img.sum()

    def test_accepts_bgr(self):
        img = make_bgr()
        result = apply_erosion(img)
        assert result.shape == img.shape

    def test_all_black_unchanged(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = apply_erosion(img)
        np.testing.assert_array_equal(result, img)

    def test_larger_kernel_more_erosion(self):
        img = make_binary()
        r1 = apply_erosion(img, MorphConfig(kernel_size=3))
        r3 = apply_erosion(img, MorphConfig(kernel_size=7))
        assert r3.sum() <= r1.sum()


# ─── apply_dilation ───────────────────────────────────────────────────────────

class TestApplyDilation:
    def test_returns_ndarray(self):
        img = make_binary()
        result = apply_dilation(img)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self):
        img = make_binary(h=32, w=48)
        result = apply_dilation(img)
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        img = make_binary()
        assert apply_dilation(img).dtype == np.uint8

    def test_dilation_grows_foreground(self):
        img = make_binary()
        result = apply_dilation(img)
        assert result.sum() >= img.sum()

    def test_all_white_unchanged(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        result = apply_dilation(img)
        np.testing.assert_array_equal(result, img)

    def test_accepts_bgr(self):
        img = make_bgr()
        result = apply_dilation(img)
        assert result.shape == img.shape


# ─── apply_opening ────────────────────────────────────────────────────────────

class TestApplyOpening:
    def test_returns_ndarray(self):
        img = make_binary()
        assert isinstance(apply_opening(img), np.ndarray)

    def test_same_shape(self):
        img = make_binary(h=48, w=64)
        assert apply_opening(img).shape == img.shape

    def test_dtype_uint8(self):
        assert apply_opening(make_binary()).dtype == np.uint8

    def test_removes_small_noise(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[30:34, 30:34] = 255  # 4x4 square (small noise)
        img[10:50, 10:50] = 255  # big square
        cfg = MorphConfig(kernel_size=9)
        result = apply_opening(img, cfg)
        # Small noise outside large region should be gone or reduced
        assert result[31, 31] == 0 or result.sum() <= img.sum()

    def test_accepts_bgr(self):
        img = make_bgr()
        assert apply_opening(img).shape == img.shape


# ─── apply_closing ────────────────────────────────────────────────────────────

class TestApplyClosing:
    def test_returns_ndarray(self):
        img = make_binary()
        assert isinstance(apply_closing(img), np.ndarray)

    def test_same_shape(self):
        img = make_binary(h=48, w=64)
        assert apply_closing(img).shape == img.shape

    def test_dtype_uint8(self):
        assert apply_closing(make_binary()).dtype == np.uint8

    def test_fills_small_holes(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        img[30:33, 30:33] = 0  # small hole
        cfg = MorphConfig(kernel_size=7)
        result = apply_closing(img, cfg)
        # Closing should fill small holes → sum >= original
        assert result.sum() >= img.sum()

    def test_accepts_bgr(self):
        img = make_bgr()
        assert apply_closing(img).shape == img.shape


# ─── get_skeleton ─────────────────────────────────────────────────────────────

class TestGetSkeleton:
    def test_returns_ndarray(self):
        img = make_binary()
        result = get_skeleton(img)
        assert isinstance(result, np.ndarray)

    def test_same_spatial_shape(self):
        img = make_binary(h=32, w=48)
        result = get_skeleton(img)
        assert result.shape == (32, 48)

    def test_dtype_uint8(self):
        img = make_binary()
        assert get_skeleton(img).dtype == np.uint8

    def test_all_black_returns_all_black(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = get_skeleton(img)
        assert result.sum() == 0

    def test_binary_values(self):
        img = make_binary()
        result = get_skeleton(img)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_skeleton_subset_of_input(self):
        img = make_binary()
        skel = get_skeleton(img)
        # Skeleton pixels must be where image was foreground (after binarization)
        assert skel.sum() <= img.sum()

    def test_accepts_bgr(self):
        img = make_bgr()
        result = get_skeleton(img)
        assert result.ndim == 2


# ─── label_regions ────────────────────────────────────────────────────────────

class TestLabelRegions:
    def test_returns_tuple(self):
        img = make_two_blobs()
        result = label_regions(img)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_n_labels_two_blobs(self):
        img = make_two_blobs()
        n, _ = label_regions(img)
        assert n == 2

    def test_label_map_dtype(self):
        img = make_two_blobs()
        _, lmap = label_regions(img)
        assert lmap.dtype == np.int32

    def test_label_map_shape(self):
        img = make_two_blobs()
        _, lmap = label_regions(img)
        assert lmap.shape == img.shape

    def test_blank_image_zero_labels(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        n, _ = label_regions(img)
        assert n == 0

    def test_invalid_connectivity_raises(self):
        img = make_binary()
        with pytest.raises(ValueError):
            label_regions(img, connectivity=6)

    def test_connectivity_4_accepted(self):
        img = make_two_blobs()
        n, _ = label_regions(img, connectivity=4)
        assert n >= 1

    def test_connectivity_8_accepted(self):
        img = make_two_blobs()
        n, _ = label_regions(img, connectivity=8)
        assert n >= 1

    def test_n_labels_is_int(self):
        img = make_binary()
        n, _ = label_regions(img)
        assert isinstance(n, int)


# ─── filter_regions_by_size ───────────────────────────────────────────────────

class TestFilterRegionsBySize:
    def test_returns_ndarray(self):
        img = make_two_blobs()
        result = filter_regions_by_size(img)
        assert isinstance(result, np.ndarray)

    def test_same_shape(self):
        img = make_two_blobs()
        result = filter_regions_by_size(img)
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        img = make_two_blobs()
        assert filter_regions_by_size(img).dtype == np.uint8

    def test_negative_min_area_raises(self):
        img = make_binary()
        with pytest.raises(ValueError):
            filter_regions_by_size(img, min_area=-1)

    def test_large_min_area_removes_all(self):
        img = make_two_blobs()
        result = filter_regions_by_size(img, min_area=100_000)
        assert result.sum() == 0

    def test_zero_min_area_keeps_all(self):
        img = make_two_blobs()
        result = filter_regions_by_size(img, min_area=0)
        assert result.sum() > 0

    def test_max_area_filters_large(self):
        img = make_two_blobs()
        result = filter_regions_by_size(img, max_area=10)
        # Both blobs are large, so result should be mostly black
        assert result.sum() == 0

    def test_binary_output_values(self):
        img = make_two_blobs()
        result = filter_regions_by_size(img)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})


# ─── compute_region_stats ─────────────────────────────────────────────────────

class TestComputeRegionStats:
    def test_returns_list(self):
        img = make_two_blobs()
        result = compute_region_stats(img)
        assert isinstance(result, list)

    def test_two_blobs_two_entries(self):
        img = make_two_blobs()
        result = compute_region_stats(img)
        assert len(result) == 2

    def test_blank_returns_empty(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = compute_region_stats(img)
        assert result == []

    def test_keys_present(self):
        img = make_binary()
        result = compute_region_stats(img)
        assert len(result) >= 1
        keys = result[0].keys()
        for k in ("label", "area", "cx", "cy",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "aspect_ratio"):
            assert k in keys

    def test_area_positive(self):
        img = make_binary()
        result = compute_region_stats(img)
        for r in result:
            assert r["area"] > 0

    def test_aspect_ratio_positive(self):
        img = make_binary()
        result = compute_region_stats(img)
        for r in result:
            assert r["aspect_ratio"] >= 0.0

    def test_centroid_within_image(self):
        img = make_binary(h=64, w=64)
        result = compute_region_stats(img)
        for r in result:
            assert 0 <= r["cx"] <= 64
            assert 0 <= r["cy"] <= 64

    def test_accepts_bgr(self):
        img = make_bgr()
        # bgr uniform → likely one big region after binarization
        result = compute_region_stats(img)
        assert isinstance(result, list)


# ─── batch_morphology ─────────────────────────────────────────────────────────

class TestBatchMorphology:
    def test_empty_returns_empty(self):
        result = batch_morphology([])
        assert result == []

    def test_length_matches(self):
        images = [make_binary() for _ in range(4)]
        result = batch_morphology(images)
        assert len(result) == 4

    def test_returns_list(self):
        images = [make_binary()]
        result = batch_morphology(images)
        assert isinstance(result, list)

    def test_each_element_ndarray(self):
        images = [make_binary() for _ in range(3)]
        result = batch_morphology(images)
        for r in result:
            assert isinstance(r, np.ndarray)

    def test_operation_erosion(self):
        images = [make_binary()]
        result = batch_morphology(images, operation="erosion")
        assert result[0].sum() <= images[0].sum()

    def test_operation_dilation(self):
        images = [make_binary()]
        result = batch_morphology(images, operation="dilation")
        assert result[0].sum() >= images[0].sum()

    def test_operation_opening(self):
        images = [make_binary()]
        result = batch_morphology(images, operation="opening")
        assert isinstance(result[0], np.ndarray)

    def test_operation_closing(self):
        images = [make_binary()]
        result = batch_morphology(images, operation="closing")
        assert isinstance(result[0], np.ndarray)

    def test_unknown_operation_raises(self):
        with pytest.raises(ValueError):
            batch_morphology([make_binary()], operation="gradient")

    def test_cfg_passed_through(self):
        images = [make_binary()]
        cfg = MorphConfig(kernel_size=5)
        result = batch_morphology(images, operation="erosion", cfg=cfg)
        assert result[0].shape == images[0].shape
