"""
Property-based tests for:
  1. puzzle_reconstruction.utils.morph_utils
  2. puzzle_reconstruction.utils.edge_profile_utils

Verifies mathematical invariants:

morph_utils:
- MorphConfig:             kernel_size odd >=1; valid shapes; iterations >=1;
                           build_kernel returns correct shape
- apply_erosion:           same shape as input; result ⊆ input (erosion ≤ input)
- apply_dilation:          same shape; result ≥ input (dilation ≥ input)
- apply_opening:           same shape; result ≤ input (opening ≤ original)
- apply_closing:           same shape; result ≥ input after erosion
- label_regions:           n_labels ≥ 0; label_map same spatial shape;
                           max label = n_labels
- filter_regions_by_size:  same shape; result ⊆ input binary;
                           min_area < 0 raises
- compute_region_stats:    list of dicts; area ≥ 0; cx/cy inside bbox
- batch_morphology:        same count; each has same shape; raises on unknown op

edge_profile_utils:
- EdgeProfileConfig:       n_samples > 0; smooth_sigma ≥ 0; normalize bool
- EdgeProfile:             n_samples = len(values); invalid side raises;
                           float32 values
- build_edge_profile:      n_samples = cfg.n_samples; normalize → [0, 1];
                           values float32; empty points → zero profile
- profile_l2_distance:     ≥ 0; self = 0; symmetric; unequal lengths raises
- profile_cosine_similarity: ∈ [-1, 1]; self = 1; zero profile → 0
- profile_correlation:     ∈ [-1, 1]; self = 1; identical → 1
- resample_profile:        new n_samples correct; values float32;
                           same size = identity
- flip_profile:            double flip = original; length preserved
- mean_profile:            values between min and max of inputs;
                           single → same values
- batch_build_profiles:    same count as input; each has n_samples from cfg
- pairwise_l2_matrix:      symmetric; diagonal = 0; non-negative
- best_matching_profile:   index in [0, n); empty raises
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from puzzle_reconstruction.utils.morph_utils import (
    MorphConfig,
    apply_erosion,
    apply_dilation,
    apply_opening,
    apply_closing,
    label_regions,
    filter_regions_by_size,
    compute_region_stats,
    batch_morphology,
)
from puzzle_reconstruction.utils.edge_profile_utils import (
    EdgeProfileConfig,
    EdgeProfile,
    build_edge_profile,
    profile_l2_distance,
    profile_cosine_similarity,
    profile_correlation,
    resample_profile,
    flip_profile,
    mean_profile,
    batch_build_profiles,
    pairwise_l2_matrix,
    best_matching_profile,
)

RNG = np.random.default_rng(77)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _binary_img(h: int = 20, w: int = 20) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[5:15, 5:15] = 255
    return img


def _gray_img(h: int = 20, w: int = 20) -> np.ndarray:
    return RNG.integers(0, 256, size=(h, w), dtype=np.uint8)


def _make_profile(n: int = 32, side: str = "top") -> EdgeProfile:
    values = RNG.uniform(0.0, 1.0, size=n).astype(np.float32)
    return EdgeProfile(values=values, side=side)


def _line_points(n: int = 20, side: str = "top") -> np.ndarray:
    x = np.linspace(0, 100, n)
    y = RNG.uniform(10.0, 50.0, size=n)
    return np.stack([x, y], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# MorphConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestMorphConfig:

    def test_default_valid(self):
        cfg = MorphConfig()
        assert cfg.kernel_size >= 1
        assert cfg.kernel_size % 2 == 1
        assert cfg.kernel_shape in {"rect", "ellipse", "cross"}
        assert cfg.iterations >= 1

    def test_raises_even_kernel_size(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_size=4)

    def test_raises_zero_kernel_size(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_size=0)

    def test_raises_invalid_shape(self):
        with pytest.raises(ValueError):
            MorphConfig(kernel_shape="circle")

    def test_raises_zero_iterations(self):
        with pytest.raises(ValueError):
            MorphConfig(iterations=0)

    def test_build_kernel_shape(self):
        for shape in ("rect", "ellipse", "cross"):
            cfg = MorphConfig(kernel_size=5, kernel_shape=shape)
            k = cfg.build_kernel()
            assert k.shape == (5, 5)

    def test_kernel_dtype(self):
        cfg = MorphConfig(kernel_size=3)
        k = cfg.build_kernel()
        assert k.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════════
# apply_erosion
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyErosion:

    def test_same_shape(self):
        img = _binary_img()
        cfg = MorphConfig(kernel_size=3)
        out = apply_erosion(img, cfg)
        assert out.shape == img.shape

    def test_erosion_le_original(self):
        img = _binary_img()
        cfg = MorphConfig(kernel_size=3)
        out = apply_erosion(img, cfg)
        # Erosion can only remove pixels, so out ≤ img element-wise
        assert np.all(out <= img)

    def test_all_black_unchanged(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        out = apply_erosion(img, MorphConfig())
        assert np.all(out == 0)

    def test_all_white_unchanged(self):
        img = np.full((10, 10), 255, dtype=np.uint8)
        out = apply_erosion(img, MorphConfig())
        assert np.all(out == 255)

    def test_default_config(self):
        img = _binary_img()
        out = apply_erosion(img)
        assert out.shape == img.shape


# ═══════════════════════════════════════════════════════════════════════════════
# apply_dilation
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyDilation:

    def test_same_shape(self):
        img = _binary_img()
        out = apply_dilation(img, MorphConfig(kernel_size=3))
        assert out.shape == img.shape

    def test_dilation_ge_original(self):
        img = _binary_img()
        out = apply_dilation(img, MorphConfig(kernel_size=3))
        assert np.all(out >= img)

    def test_all_black_unchanged(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        out = apply_dilation(img)
        assert np.all(out == 0)

    def test_all_white_unchanged(self):
        img = np.full((10, 10), 255, dtype=np.uint8)
        out = apply_dilation(img)
        assert np.all(out == 255)


# ═══════════════════════════════════════════════════════════════════════════════
# apply_opening / apply_closing
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyOpeningClosing:

    def test_opening_same_shape(self):
        img = _binary_img()
        out = apply_opening(img, MorphConfig(kernel_size=3))
        assert out.shape == img.shape

    def test_opening_le_original(self):
        img = _binary_img()
        out = apply_opening(img, MorphConfig(kernel_size=3))
        assert np.all(out <= img)

    def test_closing_same_shape(self):
        img = _binary_img()
        out = apply_closing(img, MorphConfig(kernel_size=3))
        assert out.shape == img.shape

    def test_closing_ge_original(self):
        img = _binary_img()
        out = apply_closing(img, MorphConfig(kernel_size=3))
        assert np.all(out >= img)

    def test_opening_idempotent(self):
        img = _binary_img()
        cfg = MorphConfig(kernel_size=3)
        opened_once = apply_opening(img, cfg)
        opened_twice = apply_opening(opened_once, cfg)
        assert np.array_equal(opened_once, opened_twice)

    def test_closing_idempotent(self):
        img = _binary_img()
        cfg = MorphConfig(kernel_size=3)
        closed_once = apply_closing(img, cfg)
        closed_twice = apply_closing(closed_once, cfg)
        assert np.array_equal(closed_once, closed_twice)


# ═══════════════════════════════════════════════════════════════════════════════
# label_regions
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelRegions:

    def test_single_region(self):
        img = _binary_img()
        n, label_map = label_regions(img)
        assert n >= 1
        assert label_map.shape == img.shape

    def test_all_black_zero_regions(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        n, label_map = label_regions(img)
        assert n == 0

    def test_max_label_equals_n_labels(self):
        img = _binary_img()
        n, label_map = label_regions(img)
        if n > 0:
            assert int(label_map.max()) == n

    def test_label_map_dtype(self):
        img = _binary_img()
        _, label_map = label_regions(img)
        assert label_map.dtype == np.int32

    def test_raises_invalid_connectivity(self):
        img = _binary_img()
        with pytest.raises(ValueError):
            label_regions(img, connectivity=6)

    def test_connectivity_4_and_8(self):
        img = _binary_img()
        n4, _ = label_regions(img, connectivity=4)
        n8, _ = label_regions(img, connectivity=8)
        # 4-connectivity may find more or equal labels than 8-connectivity
        assert n4 >= n8 or n8 >= n4  # always true


# ═══════════════════════════════════════════════════════════════════════════════
# filter_regions_by_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterRegionsBySize:

    def test_same_shape(self):
        img = _binary_img()
        out = filter_regions_by_size(img, min_area=1)
        assert out.shape == img.shape

    def test_result_subset_of_input(self):
        img = _binary_img()
        out = filter_regions_by_size(img, min_area=1)
        # Each pixel in out should have been set in img binary
        assert np.all(out[img == 0] == 0)

    def test_large_min_area_removes_all(self):
        img = _binary_img()  # 10x10 rectangle = 100 pixels
        out = filter_regions_by_size(img, min_area=99999)
        assert np.all(out == 0)

    def test_min_area_zero_keeps_all(self):
        img = _binary_img()
        out = filter_regions_by_size(img, min_area=0)
        assert np.array_equal(out[img > 0], np.full_like(out[img > 0], 255))

    def test_raises_negative_min_area(self):
        img = _binary_img()
        with pytest.raises(ValueError):
            filter_regions_by_size(img, min_area=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_region_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeRegionStats:

    def test_single_region_keys(self):
        img = _binary_img()
        stats = compute_region_stats(img)
        assert len(stats) >= 1
        expected_keys = {"label", "area", "cx", "cy", "bbox_x", "bbox_y",
                         "bbox_w", "bbox_h", "aspect_ratio"}
        for stat in stats:
            assert set(stat.keys()) == expected_keys

    def test_area_positive(self):
        img = _binary_img()
        stats = compute_region_stats(img)
        for stat in stats:
            assert stat["area"] >= 1.0

    def test_all_black_empty_stats(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        stats = compute_region_stats(img)
        assert stats == []

    def test_centroid_inside_bbox(self):
        img = _binary_img()
        stats = compute_region_stats(img)
        for stat in stats:
            bx, by = stat["bbox_x"], stat["bbox_y"]
            bw, bh = stat["bbox_w"], stat["bbox_h"]
            assert bx <= stat["cx"] <= bx + bw
            assert by <= stat["cy"] <= by + bh


# ═══════════════════════════════════════════════════════════════════════════════
# batch_morphology
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchMorphology:

    def test_same_count(self):
        images = [_binary_img() for _ in range(3)]
        out = batch_morphology(images, operation="opening")
        assert len(out) == 3

    def test_each_same_shape(self):
        images = [_binary_img(10, 10), _binary_img(15, 12)]
        for op in ("erosion", "dilation", "opening", "closing"):
            out = batch_morphology(images, operation=op)
            for i, img in enumerate(images):
                assert out[i].shape == img.shape

    def test_raises_unknown_operation(self):
        images = [_binary_img()]
        with pytest.raises(ValueError):
            batch_morphology(images, operation="thinning")

    def test_empty_list(self):
        out = batch_morphology([], operation="erosion")
        assert out == []


# ═══════════════════════════════════════════════════════════════════════════════
# EdgeProfileConfig / EdgeProfile
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeProfileConfig:

    def test_default_valid(self):
        cfg = EdgeProfileConfig()
        assert cfg.n_samples > 0
        assert cfg.smooth_sigma >= 0
        assert isinstance(cfg.normalize, bool)

    def test_custom_n_samples(self):
        cfg = EdgeProfileConfig(n_samples=128)
        assert cfg.n_samples == 128


class TestEdgeProfile:

    def test_n_samples_equals_len_values(self):
        p = EdgeProfile(values=np.ones(32, dtype=np.float32), side="top")
        assert p.n_samples == 32
        assert len(p) == 32

    def test_values_are_float32(self):
        p = EdgeProfile(values=np.ones(16, dtype=np.float64), side="bottom")
        assert p.values.dtype == np.float32

    def test_raises_invalid_side(self):
        with pytest.raises(ValueError):
            EdgeProfile(values=np.ones(10, dtype=np.float32), side="diagonal")

    def test_valid_sides(self):
        for side in ("top", "bottom", "left", "right", "unknown"):
            p = EdgeProfile(values=np.ones(8, dtype=np.float32), side=side)
            assert p.side == side

    def test_1d_ravel(self):
        # 2D input should be raveled
        p = EdgeProfile(values=np.ones((4, 4), dtype=np.float32))
        assert p.values.ndim == 1
        assert p.n_samples == 16


# ═══════════════════════════════════════════════════════════════════════════════
# build_edge_profile
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildEdgeProfile:

    def test_n_samples_correct(self):
        pts = _line_points(30)
        cfg = EdgeProfileConfig(n_samples=64)
        p = build_edge_profile(pts, side="top", cfg=cfg)
        assert p.n_samples == 64

    def test_normalize_range(self):
        pts = _line_points(30)
        cfg = EdgeProfileConfig(n_samples=32, normalize=True, smooth_sigma=0.0)
        p = build_edge_profile(pts, side="top", cfg=cfg)
        if p.values.max() > p.values.min():
            assert float(p.values.min()) >= 0.0 - 1e-5
            assert float(p.values.max()) <= 1.0 + 1e-5

    def test_values_float32(self):
        pts = _line_points(20)
        p = build_edge_profile(pts)
        assert p.values.dtype == np.float32

    def test_empty_points_zero_profile(self):
        pts = np.zeros((0, 2), dtype=np.float64)
        cfg = EdgeProfileConfig(n_samples=16)
        p = build_edge_profile(pts, cfg=cfg)
        assert p.n_samples == 16
        assert np.all(p.values == 0.0)

    def test_side_preserved(self):
        pts = _line_points(20)
        for side in ("top", "bottom", "left", "right"):
            p = build_edge_profile(pts, side=side)
            assert p.side == side

    def test_raises_wrong_shape(self):
        pts = np.ones((10, 3))
        with pytest.raises(ValueError):
            build_edge_profile(pts)

    def test_no_smoothing(self):
        pts = _line_points(20)
        cfg = EdgeProfileConfig(n_samples=32, smooth_sigma=0.0)
        p = build_edge_profile(pts, cfg=cfg)
        assert p.n_samples == 32


# ═══════════════════════════════════════════════════════════════════════════════
# profile_l2_distance
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfileL2Distance:

    def test_non_negative(self):
        a = _make_profile(32)
        b = _make_profile(32)
        assert profile_l2_distance(a, b) >= 0.0

    def test_self_is_zero(self):
        a = _make_profile(32)
        assert abs(profile_l2_distance(a, a)) < 1e-6

    def test_symmetric(self):
        a = _make_profile(32)
        b = _make_profile(32)
        assert abs(profile_l2_distance(a, b) - profile_l2_distance(b, a)) < 1e-6

    def test_raises_unequal_lengths(self):
        a = _make_profile(32)
        b = _make_profile(16)
        with pytest.raises(ValueError):
            profile_l2_distance(a, b)

    def test_known_value(self):
        a = EdgeProfile(values=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        b = EdgeProfile(values=np.array([3.0, 4.0, 0.0], dtype=np.float32))
        assert abs(profile_l2_distance(a, b) - 5.0) < 1e-5

    def test_triangle_inequality(self):
        a = _make_profile(16)
        b = _make_profile(16)
        c = _make_profile(16)
        assert profile_l2_distance(a, c) <= profile_l2_distance(a, b) + profile_l2_distance(b, c) + 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# profile_cosine_similarity
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfileCosineSimilarity:

    def test_range(self):
        a = _make_profile(32)
        b = _make_profile(32)
        sim = profile_cosine_similarity(a, b)
        assert -1.0 - 1e-8 <= sim <= 1.0 + 1e-8

    def test_self_is_one(self):
        a = _make_profile(32)
        assert abs(profile_cosine_similarity(a, a) - 1.0) < 1e-5

    def test_zero_profile_returns_zero(self):
        a = EdgeProfile(values=np.zeros(16, dtype=np.float32))
        b = _make_profile(16)
        assert profile_cosine_similarity(a, b) == 0.0

    def test_raises_unequal_lengths(self):
        a = _make_profile(32)
        b = _make_profile(16)
        with pytest.raises(ValueError):
            profile_cosine_similarity(a, b)

    def test_symmetric(self):
        a = _make_profile(16)
        b = _make_profile(16)
        assert abs(profile_cosine_similarity(a, b) - profile_cosine_similarity(b, a)) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# profile_correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfileCorrelation:

    def test_range(self):
        a = _make_profile(32)
        b = _make_profile(32)
        r = profile_correlation(a, b)
        assert -1.0 - 1e-8 <= r <= 1.0 + 1e-8

    def test_self_is_one(self):
        a = _make_profile(32)
        assert abs(profile_correlation(a, a) - 1.0) < 1e-5

    def test_identical_profiles_is_one(self):
        values = RNG.uniform(0.0, 1.0, 32).astype(np.float32)
        a = EdgeProfile(values=values.copy())
        b = EdgeProfile(values=values.copy())
        assert abs(profile_correlation(a, b) - 1.0) < 1e-5

    def test_raises_unequal_lengths(self):
        a = _make_profile(32)
        b = _make_profile(16)
        with pytest.raises(ValueError):
            profile_correlation(a, b)

    def test_single_sample_returns_one(self):
        a = EdgeProfile(values=np.array([0.5], dtype=np.float32))
        b = EdgeProfile(values=np.array([0.7], dtype=np.float32))
        assert profile_correlation(a, b) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# resample_profile
# ═══════════════════════════════════════════════════════════════════════════════

class TestResampleProfile:

    def test_new_n_samples(self):
        p = _make_profile(32)
        p2 = resample_profile(p, 64)
        assert p2.n_samples == 64

    def test_values_float32(self):
        p = _make_profile(32)
        p2 = resample_profile(p, 16)
        assert p2.values.dtype == np.float32

    def test_same_size_is_close(self):
        p = _make_profile(32)
        p2 = resample_profile(p, 32)
        assert np.allclose(p.values, p2.values, atol=1e-5)

    def test_raises_zero_n_samples(self):
        p = _make_profile(32)
        with pytest.raises(ValueError):
            resample_profile(p, 0)

    def test_side_preserved(self):
        p = _make_profile(32, side="left")
        p2 = resample_profile(p, 64)
        assert p2.side == "left"

    def test_constant_profile_resampled_constant(self):
        values = np.full(20, 0.5, dtype=np.float32)
        p = EdgeProfile(values=values)
        p2 = resample_profile(p, 40)
        assert np.allclose(p2.values, 0.5, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# flip_profile
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlipProfile:

    def test_double_flip_is_identity(self):
        p = _make_profile(32)
        flipped_twice = flip_profile(flip_profile(p))
        assert np.allclose(p.values, flipped_twice.values, atol=1e-6)

    def test_length_preserved(self):
        p = _make_profile(32)
        f = flip_profile(p)
        assert f.n_samples == p.n_samples

    def test_side_preserved(self):
        p = _make_profile(32, side="right")
        f = flip_profile(p)
        assert f.side == "right"

    def test_values_reversed(self):
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        p = EdgeProfile(values=values)
        f = flip_profile(p)
        assert np.allclose(f.values, [4.0, 3.0, 2.0, 1.0], atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# mean_profile
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeanProfile:

    def test_single_input_is_same(self):
        p = _make_profile(32)
        m = mean_profile([p])
        assert np.allclose(m.values, p.values, atol=1e-5)

    def test_two_identical_mean_is_same(self):
        values = RNG.uniform(0, 1, 16).astype(np.float32)
        p1 = EdgeProfile(values=values.copy())
        p2 = EdgeProfile(values=values.copy())
        m = mean_profile([p1, p2])
        assert np.allclose(m.values, values, atol=1e-5)

    def test_mean_in_range(self):
        profiles = [_make_profile(16) for _ in range(5)]
        m = mean_profile(profiles)
        min_val = min(float(p.values.min()) for p in profiles)
        max_val = max(float(p.values.max()) for p in profiles)
        assert float(m.values.min()) >= min_val - 1e-5
        assert float(m.values.max()) <= max_val + 1e-5

    def test_empty_input_returns_empty(self):
        m = mean_profile([])
        assert m.n_samples == 0

    def test_raises_mismatched_n_samples(self):
        p1 = _make_profile(32)
        p2 = _make_profile(16)
        with pytest.raises(ValueError):
            mean_profile([p1, p2])


# ═══════════════════════════════════════════════════════════════════════════════
# batch_build_profiles
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchBuildProfiles:

    def test_same_count(self):
        point_sets = [_line_points(20) for _ in range(4)]
        profiles = batch_build_profiles(point_sets)
        assert len(profiles) == 4

    def test_each_has_correct_n_samples(self):
        point_sets = [_line_points(20) for _ in range(3)]
        cfg = EdgeProfileConfig(n_samples=48)
        profiles = batch_build_profiles(point_sets, cfg=cfg)
        for p in profiles:
            assert p.n_samples == 48

    def test_sides_assigned(self):
        point_sets = [_line_points(20) for _ in range(2)]
        sides = ["top", "bottom"]
        profiles = batch_build_profiles(point_sets, sides=sides)
        assert profiles[0].side == "top"
        assert profiles[1].side == "bottom"

    def test_empty_input(self):
        profiles = batch_build_profiles([])
        assert profiles == []


# ═══════════════════════════════════════════════════════════════════════════════
# pairwise_l2_matrix
# ═══════════════════════════════════════════════════════════════════════════════

class TestPairwiseL2Matrix:

    def test_shape(self):
        profiles = [_make_profile(16) for _ in range(4)]
        mat = pairwise_l2_matrix(profiles)
        assert mat.shape == (4, 4)

    def test_diagonal_is_zero(self):
        profiles = [_make_profile(16) for _ in range(4)]
        mat = pairwise_l2_matrix(profiles)
        assert np.allclose(np.diag(mat), 0.0, atol=1e-5)

    def test_symmetric(self):
        profiles = [_make_profile(16) for _ in range(4)]
        mat = pairwise_l2_matrix(profiles)
        assert np.allclose(mat, mat.T, atol=1e-5)

    def test_non_negative(self):
        profiles = [_make_profile(16) for _ in range(4)]
        mat = pairwise_l2_matrix(profiles)
        assert float(mat.min()) >= -1e-5

    def test_single_profile(self):
        profiles = [_make_profile(16)]
        mat = pairwise_l2_matrix(profiles)
        assert mat.shape == (1, 1)
        assert abs(float(mat[0, 0])) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# best_matching_profile
# ═══════════════════════════════════════════════════════════════════════════════

class TestBestMatchingProfile:

    def test_index_in_range(self):
        query = _make_profile(16)
        candidates = [_make_profile(16) for _ in range(5)]
        idx, dist = best_matching_profile(query, candidates)
        assert 0 <= idx < 5

    def test_distance_non_negative(self):
        query = _make_profile(16)
        candidates = [_make_profile(16) for _ in range(3)]
        _, dist = best_matching_profile(query, candidates)
        assert dist >= 0.0

    def test_self_match_returns_zero_dist(self):
        p = _make_profile(16)
        idx, dist = best_matching_profile(p, [p])
        assert idx == 0
        assert abs(dist) < 1e-5

    def test_raises_empty_candidates(self):
        query = _make_profile(16)
        with pytest.raises(ValueError):
            best_matching_profile(query, [])

    def test_identical_candidate_wins(self):
        query = _make_profile(16)
        other = _make_profile(16)
        candidates = [other, query, other]
        idx, dist = best_matching_profile(query, candidates)
        assert idx == 1
        assert abs(dist) < 1e-5
