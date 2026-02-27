"""Integration tests for algorithm descriptor modules.

Tests for:
- boundary_descriptor
- color_palette
- color_space
- contour_smoother
- edge_extractor
- edge_filter
- fourier_descriptor
- wavelet_descriptor
"""
from __future__ import annotations

import math
import pytest
import numpy as np

# ─── boundary_descriptor imports ──────────────────────────────────────────────
from puzzle_reconstruction.algorithms.boundary_descriptor import (
    DescriptorConfig,
    BoundaryDescriptor,
    compute_curvature,
    curvature_histogram,
    direction_histogram,
    chord_distribution,
    extract_descriptor,
    descriptor_similarity,
    batch_extract_descriptors,
)

# ─── color_palette imports ────────────────────────────────────────────────────
from puzzle_reconstruction.algorithms.color_palette import (
    ColorPaletteConfig,
    ColorPalette,
    extract_dominant_colors,
    palette_distance,
    compute_palette,
    batch_compute_palettes,
    rank_by_palette,
)

# ─── color_space imports ──────────────────────────────────────────────────────
from puzzle_reconstruction.algorithms.color_space import (
    ColorSpaceConfig,
    ColorHistogram,
    bgr_to_space,
    compute_channel_hist,
    compute_color_histogram,
    histogram_intersection,
    histogram_chi2,
    batch_compute_histograms,
)

# ─── contour_smoother imports ─────────────────────────────────────────────────
from puzzle_reconstruction.algorithms.contour_smoother import (
    SmootherConfig,
    SmoothedContour,
    smooth_gaussian,
    resample_contour,
    compute_arc_length,
    smooth_and_resample,
    align_contours,
    contour_similarity,
    batch_smooth,
)

# ─── edge_extractor imports ───────────────────────────────────────────────────
from puzzle_reconstruction.algorithms.edge_extractor import (
    EdgeSegment,
    FragmentEdges,
    detect_boundary,
    extract_edge_points,
    split_edge_by_side,
    compute_edge_length,
    simplify_edge,
    extract_fragment_edges,
    batch_extract_edges,
)

# ─── edge_filter imports ──────────────────────────────────────────────────────
from puzzle_reconstruction.algorithms.edge_filter import (
    EdgeFilterConfig,
    filter_by_score,
    filter_top_k,
    filter_compatible,
    deduplicate_pairs,
    apply_edge_filter,
    batch_filter_edges,
)
from puzzle_reconstruction.algorithms.edge_comparator import EdgeCompareResult

# ─── fourier_descriptor imports ───────────────────────────────────────────────
from puzzle_reconstruction.algorithms.fourier_descriptor import (
    FourierConfig,
    FourierDescriptor,
    compute_contour_centroid,
    complex_representation,
    compute_fd,
    fd_similarity,
    batch_compute_fd,
    rank_by_fd,
)

# ─── wavelet_descriptor imports ───────────────────────────────────────────────
from puzzle_reconstruction.algorithms.wavelet_descriptor import (
    WaveletDescriptor,
    compute_wavelet_descriptor,
    wavelet_similarity,
    wavelet_similarity_mirror,
    batch_wavelet_similarity,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_circle_contour(n: int = 64, r: float = 50.0, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Generate a closed circular contour of n points."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def make_square_contour(n: int = 40, side: float = 40.0) -> np.ndarray:
    """Generate a square contour with n points per side."""
    pts_per_side = n
    t = np.linspace(0, side, pts_per_side, endpoint=False)
    top    = np.stack([t,              np.zeros(pts_per_side)], axis=1)
    right  = np.stack([np.full(pts_per_side, side), t],         axis=1)
    bottom = np.stack([side - t,       np.full(pts_per_side, side)], axis=1)
    left   = np.stack([np.zeros(pts_per_side), side - t],       axis=1)
    return np.vstack([top, right, bottom, left])


def make_edge_compare_result(
    edge_id_a: int = 0,
    edge_id_b: int = 1,
    score: float = 0.7,
    css_sim: float = 0.7,
    ifs_sim: float = 0.7,
) -> EdgeCompareResult:
    return EdgeCompareResult(
        edge_id_a=edge_id_a,
        edge_id_b=edge_id_b,
        dtw_dist=1.0,
        css_sim=css_sim,
        fd_diff=0.1,
        ifs_sim=ifs_sim,
        score=score,
    )


# =============================================================================
# TestBoundaryDescriptor
# =============================================================================

class TestBoundaryDescriptor:

    def test_descriptor_config_defaults(self):
        cfg = DescriptorConfig()
        assert cfg.n_bins == 32
        assert cfg.smooth_sigma == 1.0
        assert cfg.normalize is True
        assert cfg.max_chord is None

    def test_descriptor_config_invalid_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            DescriptorConfig(n_bins=3)

    def test_descriptor_config_invalid_smooth_sigma(self):
        with pytest.raises(ValueError, match="smooth_sigma"):
            DescriptorConfig(smooth_sigma=-0.1)

    def test_descriptor_config_invalid_max_chord(self):
        with pytest.raises(ValueError, match="max_chord"):
            DescriptorConfig(max_chord=-1.0)

    def test_compute_curvature_shape(self):
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 100, (50, 2))
        curv = compute_curvature(pts, smooth_sigma=0.0)
        assert curv.shape == (50,)
        assert curv.dtype == np.float32

    def test_compute_curvature_circle_near_constant(self):
        # A circle has constant curvature 1/r
        pts = make_circle_contour(n=100, r=50.0)
        curv = compute_curvature(pts, smooth_sigma=0.5)
        # Std of curvature on circle should be small relative to mean
        assert curv.std() < abs(curv.mean()) * 2.0

    def test_compute_curvature_invalid_shape(self):
        with pytest.raises(ValueError):
            compute_curvature(np.array([[1, 2, 3]]))

    def test_compute_curvature_too_few_points(self):
        with pytest.raises(ValueError, match="3"):
            compute_curvature(np.array([[0, 0], [1, 1]]))

    def test_curvature_histogram_shape_and_sum(self):
        rng = np.random.default_rng(7)
        curv = rng.standard_normal(100).astype(np.float32)
        hist = curvature_histogram(curv, n_bins=16, normalize=True)
        assert hist.shape == (16,)
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_curvature_histogram_no_normalize(self):
        rng = np.random.default_rng(7)
        curv = rng.standard_normal(100).astype(np.float32)
        hist = curvature_histogram(curv, n_bins=16, normalize=False)
        assert hist.sum() == pytest.approx(100.0, abs=1.0)

    def test_direction_histogram_shape_and_sum(self):
        pts = make_circle_contour(n=64)
        hist = direction_histogram(pts, n_bins=32, normalize=True)
        assert hist.shape == (32,)
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_direction_histogram_circle_uniform(self):
        # Circle directions are roughly uniformly distributed
        pts = make_circle_contour(n=128)
        hist = direction_histogram(pts, n_bins=32, normalize=True)
        expected = 1.0 / 32
        assert np.max(np.abs(hist - expected)) < 0.05

    def test_chord_distribution_shape_and_sum(self):
        pts = make_circle_contour(n=50)
        hist = chord_distribution(pts, n_bins=16, normalize=True)
        assert hist.shape == (16,)
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_extract_descriptor_returns_correct_type(self):
        pts = make_circle_contour(n=64)
        desc = extract_descriptor(pts, fragment_id=1, edge_id=2)
        assert isinstance(desc, BoundaryDescriptor)
        assert desc.fragment_id == 1
        assert desc.edge_id == 2
        assert desc.length > 0

    def test_extract_descriptor_histogram_sizes(self):
        pts = make_circle_contour(n=64)
        cfg = DescriptorConfig(n_bins=16)
        desc = extract_descriptor(pts, cfg=cfg)
        assert len(desc.curvature_hist) == 16
        assert len(desc.direction_hist) == 16
        assert len(desc.chord_hist) == 16

    def test_extract_descriptor_invalid_points(self):
        with pytest.raises(ValueError):
            extract_descriptor(np.array([[0, 0], [1, 1]]))  # < 3 points

    def test_boundary_descriptor_feature_vector_shape(self):
        pts = make_circle_contour(n=64)
        cfg = DescriptorConfig(n_bins=16)
        desc = extract_descriptor(pts, cfg=cfg)
        fv = desc.feature_vector
        assert fv.shape == (48,)  # 3 * 16
        assert fv.dtype == np.float32

    def test_descriptor_similarity_self(self):
        pts = make_circle_contour(n=64)
        desc = extract_descriptor(pts)
        sim = descriptor_similarity(desc, desc)
        assert abs(sim - 1.0) < 1e-5

    def test_descriptor_similarity_different(self):
        pts_circle = make_circle_contour(n=64)
        pts_square = make_square_contour(n=16)
        desc_c = extract_descriptor(pts_circle)
        desc_s = extract_descriptor(pts_square)
        sim = descriptor_similarity(desc_c, desc_s)
        assert 0.0 <= sim <= 1.0

    def test_descriptor_similarity_range(self):
        rng = np.random.default_rng(99)
        pts1 = rng.uniform(0, 100, (50, 2))
        pts2 = rng.uniform(0, 100, (50, 2))
        d1 = extract_descriptor(pts1)
        d2 = extract_descriptor(pts2)
        sim = descriptor_similarity(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_batch_extract_descriptors_length(self):
        pts_list = [make_circle_contour(n=32 + i * 4) for i in range(5)]
        descs = batch_extract_descriptors(pts_list, fragment_id=3)
        assert len(descs) == 5
        for i, d in enumerate(descs):
            assert d.edge_id == i
            assert d.fragment_id == 3


# =============================================================================
# TestColorPalette
# =============================================================================

class TestColorPalette:

    def test_config_defaults(self):
        cfg = ColorPaletteConfig()
        assert cfg.n_colors == 8
        assert cfg.max_iter == 20
        assert cfg.tol == pytest.approx(1e-4)
        assert cfg.seed == 0

    def test_config_invalid_n_colors(self):
        with pytest.raises(ValueError, match="n_colors"):
            ColorPaletteConfig(n_colors=1)

    def test_config_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            ColorPaletteConfig(max_iter=0)

    def test_config_invalid_tol(self):
        with pytest.raises(ValueError, match="tol"):
            ColorPaletteConfig(tol=-0.1)

    def test_extract_dominant_colors_shape(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        colors, weights = extract_dominant_colors(img, n_colors=4, seed=0)
        assert colors.shape == (4, 3)
        assert weights.shape == (4,)
        assert colors.dtype == np.float32
        assert weights.dtype == np.float32

    def test_extract_dominant_colors_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        _, weights = extract_dominant_colors(img, n_colors=4, seed=0)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_extract_dominant_colors_grayscale(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (20, 20), dtype=np.uint8)
        colors, weights = extract_dominant_colors(img, n_colors=4, seed=0)
        assert colors.shape == (4, 1)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_extract_dominant_colors_reproducible(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        c1, w1 = extract_dominant_colors(img, n_colors=4, seed=7)
        c2, w2 = extract_dominant_colors(img, n_colors=4, seed=7)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(w1, w2)

    def test_palette_distance_identical_is_zero(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        p = compute_palette(img, fragment_id=0)
        dist = palette_distance(p, p)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_palette_distance_nonnegative(self):
        rng = np.random.default_rng(0)
        img1 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        p1 = compute_palette(img1, fragment_id=0)
        p2 = compute_palette(img2, fragment_id=1)
        dist = palette_distance(p1, p2)
        assert dist >= 0.0

    def test_palette_distance_different_n_colors_raises(self):
        cfg4 = ColorPaletteConfig(n_colors=4)
        cfg8 = ColorPaletteConfig(n_colors=8)
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        p4 = compute_palette(img, fragment_id=0, cfg=cfg4)
        p8 = compute_palette(img, fragment_id=0, cfg=cfg8)
        with pytest.raises(ValueError, match="n_colors"):
            palette_distance(p4, p8)

    def test_compute_palette_returns_correct_type(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        pal = compute_palette(img, fragment_id=5)
        assert isinstance(pal, ColorPalette)
        assert pal.fragment_id == 5

    def test_compute_palette_dominant_property(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        pal = compute_palette(img, fragment_id=0)
        dom = pal.dominant
        assert dom.shape == (3,)

    def test_batch_compute_palettes_length(self):
        rng = np.random.default_rng(5)
        imgs = [rng.integers(0, 256, (10, 10, 3), dtype=np.uint8) for _ in range(4)]
        palettes = batch_compute_palettes(imgs)
        assert len(palettes) == 4
        for i, p in enumerate(palettes):
            assert p.fragment_id == i

    def test_rank_by_palette_sorted_descending(self):
        rng = np.random.default_rng(3)
        imgs = [rng.integers(0, 256, (10, 10, 3), dtype=np.uint8) for _ in range(5)]
        palettes = batch_compute_palettes(imgs)
        query = palettes[0]
        candidates = palettes[1:]
        ranked = rank_by_palette(query, candidates)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_palette_self_top(self):
        rng = np.random.default_rng(3)
        imgs = [rng.integers(0, 256, (10, 10, 3), dtype=np.uint8) for _ in range(5)]
        palettes = batch_compute_palettes(imgs)
        query = palettes[0]
        # Include query itself in candidates
        ranked = rank_by_palette(query, palettes)
        top_idx, top_score = ranked[0]
        assert top_score == pytest.approx(1.0, abs=0.01)

    def test_rank_by_palette_custom_indices(self):
        rng = np.random.default_rng(3)
        imgs = [rng.integers(0, 256, (10, 10, 3), dtype=np.uint8) for _ in range(3)]
        palettes = batch_compute_palettes(imgs)
        ranked = rank_by_palette(palettes[0], palettes[1:], indices=[10, 20])
        assert len(ranked) == 2
        returned_indices = {r[0] for r in ranked}
        assert returned_indices == {10, 20}


# =============================================================================
# TestColorSpace
# =============================================================================

class TestColorSpace:

    def test_config_defaults(self):
        cfg = ColorSpaceConfig()
        assert cfg.target_space == "hsv"
        assert cfg.n_bins == 32
        assert cfg.normalize is True

    def test_config_invalid_space(self):
        with pytest.raises(ValueError, match="target_space"):
            ColorSpaceConfig(target_space="xyz")

    def test_config_invalid_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            ColorSpaceConfig(n_bins=2)

    def test_bgr_to_space_bgr(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        out = bgr_to_space(img, "bgr")
        np.testing.assert_array_equal(out, img)

    def test_bgr_to_space_hsv_shape(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        out = bgr_to_space(img, "hsv")
        assert out.shape == (10, 10, 3)
        assert out.dtype == np.uint8

    def test_bgr_to_space_gray_shape(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        out = bgr_to_space(img, "gray")
        assert out.shape == (10, 10)

    def test_bgr_to_space_lab_shape(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        out = bgr_to_space(img, "lab")
        assert out.shape == (10, 10, 3)

    def test_bgr_to_space_invalid(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            bgr_to_space(img, "yuv")

    def test_compute_channel_hist_shape(self):
        rng = np.random.default_rng(0)
        channel = rng.integers(0, 256, (20, 20), dtype=np.uint8)
        hist = compute_channel_hist(channel, n_bins=32, normalize=True)
        assert hist.shape == (32,)
        assert hist.dtype == np.float32

    def test_compute_channel_hist_sum_one(self):
        rng = np.random.default_rng(0)
        channel = rng.integers(0, 256, (20, 20), dtype=np.uint8)
        hist = compute_channel_hist(channel, n_bins=32, normalize=True)
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_compute_channel_hist_no_normalize(self):
        rng = np.random.default_rng(0)
        channel = rng.integers(0, 256, (20, 20), dtype=np.uint8)
        hist = compute_channel_hist(channel, n_bins=32, normalize=False)
        assert hist.sum() == pytest.approx(400.0, abs=1.0)

    def test_compute_color_histogram_hsv_dim(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        ch = compute_color_histogram(img, fragment_id=2)
        # HSV has 3 channels * 32 bins = 96
        assert ch.dim == 96
        assert ch.fragment_id == 2

    def test_compute_color_histogram_gray_dim(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        cfg = ColorSpaceConfig(target_space="gray", n_bins=16)
        ch = compute_color_histogram(img, cfg=cfg, fragment_id=0)
        assert ch.dim == 16

    def test_histogram_intersection_self_is_one(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        ch = compute_color_histogram(img)
        sim = histogram_intersection(ch, ch)
        assert abs(sim - 1.0) < 1e-5

    def test_histogram_intersection_range(self):
        rng = np.random.default_rng(0)
        img1 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        ch1 = compute_color_histogram(img1)
        ch2 = compute_color_histogram(img2)
        sim = histogram_intersection(ch1, ch2)
        assert 0.0 <= sim <= 1.0

    def test_histogram_chi2_self_is_one(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        ch = compute_color_histogram(img)
        sim = histogram_chi2(ch, ch)
        assert abs(sim - 1.0) < 1e-5

    def test_histogram_chi2_range(self):
        rng = np.random.default_rng(1)
        img1 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        ch1 = compute_color_histogram(img1)
        ch2 = compute_color_histogram(img2)
        sim = histogram_chi2(ch1, ch2)
        assert 0.0 <= sim <= 1.0

    def test_batch_compute_histograms_length(self):
        rng = np.random.default_rng(0)
        imgs = [rng.integers(0, 256, (10, 10, 3), dtype=np.uint8) for _ in range(5)]
        hists = batch_compute_histograms(imgs)
        assert len(hists) == 5
        for i, h in enumerate(hists):
            assert h.fragment_id == i


# =============================================================================
# TestContourSmoother
# =============================================================================

class TestContourSmoother:

    def test_smoother_config_defaults(self):
        cfg = SmootherConfig()
        assert cfg.sigma == 1.0
        assert cfg.n_points == 64
        assert cfg.closed is False

    def test_smoother_config_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            SmootherConfig(sigma=-1.0)

    def test_smoother_config_invalid_n_points(self):
        with pytest.raises(ValueError, match="n_points"):
            SmootherConfig(n_points=1)

    def test_compute_arc_length_straight_line(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]])
        length = compute_arc_length(pts)
        assert abs(length - 5.0) < 1e-10

    def test_compute_arc_length_closed(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        length_open = compute_arc_length(pts, closed=False)
        length_closed = compute_arc_length(pts, closed=True)
        assert abs(length_open - 3.0) < 1e-10
        assert abs(length_closed - 4.0) < 1e-10

    def test_compute_arc_length_single_point(self):
        pts = np.array([[5.0, 5.0]])
        length = compute_arc_length(pts)
        assert length == 0.0

    def test_smooth_gaussian_no_change_sigma_zero(self):
        pts = make_circle_contour(n=32)
        smoothed = smooth_gaussian(pts, sigma=0.0)
        np.testing.assert_array_almost_equal(smoothed, pts)

    def test_smooth_gaussian_shape(self):
        pts = make_circle_contour(n=50)
        smoothed = smooth_gaussian(pts, sigma=1.0)
        assert smoothed.shape == (50, 2)

    def test_smooth_gaussian_output_dtype_float64(self):
        pts = make_circle_contour(n=32)
        smoothed = smooth_gaussian(pts, sigma=1.0)
        assert smoothed.dtype == np.float64

    def test_resample_contour_target_shape(self):
        pts = make_circle_contour(n=100)
        resampled = resample_contour(pts, n_points=32)
        assert resampled.shape == (32, 2)

    def test_resample_contour_arc_length_preserved(self):
        pts = make_circle_contour(n=100, r=50.0)
        resampled = resample_contour(pts, n_points=50)
        original_len = compute_arc_length(pts)
        resampled_len = compute_arc_length(resampled)
        assert abs(original_len - resampled_len) < original_len * 0.05

    def test_smooth_and_resample_output_type(self):
        pts = make_circle_contour(n=64)
        result = smooth_and_resample(pts)
        assert isinstance(result, SmoothedContour)
        assert result.n_points == 64  # default n_points

    def test_smooth_and_resample_original_n(self):
        pts = make_circle_contour(n=80)
        result = smooth_and_resample(pts)
        assert result.original_n == 80

    def test_smooth_and_resample_method_gaussian(self):
        pts = make_circle_contour(n=64)
        cfg = SmootherConfig(sigma=2.0)
        result = smooth_and_resample(pts, cfg=cfg)
        assert result.method == "gaussian"

    def test_smooth_and_resample_method_none(self):
        pts = make_circle_contour(n=64)
        cfg = SmootherConfig(sigma=0.0)
        result = smooth_and_resample(pts, cfg=cfg)
        assert result.method == "none"

    def test_align_contours_same_returns_zero_shift(self):
        pts = make_circle_contour(n=32)
        c1, c2_aligned = align_contours(pts, pts)
        np.testing.assert_array_almost_equal(c1, c2_aligned)

    def test_align_contours_mismatched_length_raises(self):
        pts1 = make_circle_contour(n=32)
        pts2 = make_circle_contour(n=16)
        with pytest.raises(ValueError):
            align_contours(pts1, pts2)

    def test_contour_similarity_self_is_one(self):
        pts = make_circle_contour(n=64)
        sim = contour_similarity(pts, pts, metric="l2")
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_contour_similarity_range_l2(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_square_contour(n=20)
        sim = contour_similarity(pts1, pts2, metric="l2")
        assert 0.0 <= sim <= 1.0

    def test_contour_similarity_range_hausdorff(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_square_contour(n=20)
        sim = contour_similarity(pts1, pts2, metric="hausdorff")
        assert 0.0 <= sim <= 1.0

    def test_contour_similarity_invalid_metric(self):
        pts = make_circle_contour(n=32)
        with pytest.raises(ValueError):
            contour_similarity(pts, pts, metric="manhattan")

    def test_batch_smooth_length(self):
        pts_list = [make_circle_contour(n=32 + 8 * i) for i in range(4)]
        results = batch_smooth(pts_list)
        assert len(results) == 4

    def test_smoothed_contour_length_property(self):
        pts = make_circle_contour(n=64)
        result = smooth_and_resample(pts)
        assert result.length > 0.0

    def test_smoothed_contour_is_closed_false(self):
        pts = make_circle_contour(n=64)
        result = smooth_and_resample(pts, cfg=SmootherConfig(closed=False))
        assert result.is_closed is False


# =============================================================================
# TestEdgeExtractor
# =============================================================================

class TestEdgeExtractor:

    def _make_white_square_img(self, h=50, w=50):
        """Create an image with a bright square on dark background."""
        img = np.zeros((h, w), dtype=np.uint8)
        img[10:40, 10:40] = 200
        return img

    def test_edge_segment_valid_construction(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        seg = EdgeSegment(points=pts, side="top", length=1.0)
        assert seg.side == "top"
        assert len(seg) == 2

    def test_edge_segment_invalid_side(self):
        pts = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            EdgeSegment(points=pts, side="diagonal")

    def test_edge_segment_invalid_length(self):
        pts = np.zeros((2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            EdgeSegment(points=pts, side="top", length=-1.0)

    def test_fragment_edges_valid(self):
        seg = EdgeSegment(points=np.zeros((2, 2), dtype=np.float32), side="top")
        fe = FragmentEdges(segments=[seg], n_segments=1)
        assert len(fe) == 1

    def test_detect_boundary_shape(self):
        img = self._make_white_square_img()
        boundary = detect_boundary(img, threshold=10)
        assert boundary.shape == img.shape
        assert boundary.dtype == np.uint8

    def test_detect_boundary_has_edge_pixels(self):
        img = self._make_white_square_img()
        boundary = detect_boundary(img, threshold=10)
        assert boundary.max() == 255

    def test_detect_boundary_background_all_zero(self):
        # All-black image should produce no boundary
        img = np.zeros((30, 30), dtype=np.uint8)
        boundary = detect_boundary(img, threshold=10)
        assert boundary.max() == 0

    def test_detect_boundary_invalid_threshold(self):
        img = self._make_white_square_img()
        with pytest.raises(ValueError):
            detect_boundary(img, threshold=300)

    def test_extract_edge_points_shape(self):
        img = self._make_white_square_img()
        boundary = detect_boundary(img, threshold=10)
        pts = extract_edge_points(boundary)
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert pts.dtype == np.float32

    def test_extract_edge_points_nonempty(self):
        img = self._make_white_square_img()
        boundary = detect_boundary(img, threshold=10)
        pts = extract_edge_points(boundary)
        assert len(pts) > 0

    def test_extract_edge_points_empty_mask(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        pts = extract_edge_points(mask)
        assert pts.shape == (0, 2)

    def test_extract_edge_points_invalid_ndim(self):
        with pytest.raises(ValueError):
            extract_edge_points(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_split_edge_by_side_keys(self):
        pts = np.array([[5.0, 0.0], [0.0, 5.0], [49.0, 25.0], [25.0, 49.0]], dtype=np.float32)
        sides = split_edge_by_side(pts, (50, 50))
        assert set(sides.keys()) == {"top", "bottom", "left", "right"}

    def test_split_edge_by_side_total_count(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(1, 48, (100, 2)).astype(np.float32)
        sides = split_edge_by_side(pts, (50, 50))
        total = sum(len(v) for v in sides.values())
        assert total == 100

    def test_compute_edge_length_zero_for_single_point(self):
        pts = np.array([[5.0, 5.0]], dtype=np.float32)
        assert compute_edge_length(pts) == 0.0

    def test_compute_edge_length_known_value(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        assert abs(compute_edge_length(pts) - 5.0) < 1e-5

    def test_simplify_edge_reduces_points(self):
        pts = np.array([[float(i), 0.0] for i in range(20)], dtype=np.float32)
        simplified = simplify_edge(pts, epsilon=0.5)
        assert len(simplified) <= len(pts)

    def test_simplify_edge_invalid_epsilon(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            simplify_edge(pts, epsilon=-1.0)

    def test_extract_fragment_edges_returns_four_segments(self):
        img = self._make_white_square_img()
        fe = extract_fragment_edges(img, threshold=10)
        assert len(fe) == 4
        sides = {seg.side for seg in fe.segments}
        assert sides == {"top", "bottom", "left", "right"}

    def test_batch_extract_edges_length(self):
        imgs = [self._make_white_square_img() for _ in range(3)]
        results = batch_extract_edges(imgs, threshold=10)
        assert len(results) == 3
        for fe in results:
            assert isinstance(fe, FragmentEdges)


# =============================================================================
# TestEdgeFilter
# =============================================================================

class TestEdgeFilter:

    def _make_results(self):
        return [
            make_edge_compare_result(0, 1, score=0.8),
            make_edge_compare_result(0, 2, score=0.5),
            make_edge_compare_result(1, 2, score=0.65),
            make_edge_compare_result(2, 1, score=0.65),  # duplicate of (1,2)
            make_edge_compare_result(3, 4, score=0.3),
        ]

    def test_config_defaults(self):
        cfg = EdgeFilterConfig()
        assert cfg.min_score is None
        assert cfg.top_k is None
        assert cfg.deduplicate is True
        assert cfg.only_compatible is False

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError, match="min_score"):
            EdgeFilterConfig(min_score=1.5)

    def test_config_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            EdgeFilterConfig(top_k=0)

    def test_filter_by_score_threshold(self):
        results = self._make_results()
        filtered = filter_by_score(results, min_score=0.6)
        assert all(r.score >= 0.6 for r in filtered)
        assert len(filtered) == 3

    def test_filter_by_score_all_pass(self):
        results = self._make_results()
        filtered = filter_by_score(results, min_score=0.0)
        assert len(filtered) == len(results)

    def test_filter_by_score_none_pass(self):
        results = self._make_results()
        filtered = filter_by_score(results, min_score=1.0)
        # None have score >= 1.0 exactly
        assert len(filtered) == 0

    def test_filter_by_score_invalid(self):
        with pytest.raises(ValueError):
            filter_by_score([], min_score=1.5)

    def test_filter_top_k_returns_k(self):
        results = self._make_results()
        top = filter_top_k(results, k=2)
        assert len(top) == 2
        assert top[0].score >= top[1].score

    def test_filter_top_k_sorted_descending(self):
        results = self._make_results()
        top = filter_top_k(results, k=4)
        scores = [r.score for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_filter_top_k_less_than_k(self):
        results = self._make_results()
        top = filter_top_k(results, k=100)
        assert len(top) == len(results)

    def test_filter_top_k_invalid(self):
        with pytest.raises(ValueError):
            filter_top_k([], k=0)

    def test_filter_compatible_only_high_scores(self):
        results = self._make_results()
        compatible = filter_compatible(results)
        assert all(r.is_compatible for r in compatible)
        assert all(r.score >= 0.6 for r in compatible)

    def test_deduplicate_pairs_removes_mirror(self):
        results = self._make_results()
        deduped = deduplicate_pairs(results)
        keys = [r.pair_key for r in deduped]
        assert len(keys) == len(set(keys))

    def test_deduplicate_pairs_preserves_first(self):
        r1 = make_edge_compare_result(1, 2, score=0.8)
        r2 = make_edge_compare_result(2, 1, score=0.5)  # duplicate
        deduped = deduplicate_pairs([r1, r2])
        assert len(deduped) == 1
        assert deduped[0].score == pytest.approx(0.8)

    def test_apply_edge_filter_with_min_score(self):
        results = self._make_results()
        cfg = EdgeFilterConfig(min_score=0.6, deduplicate=False)
        filtered = apply_edge_filter(results, cfg)
        assert all(r.score >= 0.6 for r in filtered)

    def test_apply_edge_filter_with_top_k(self):
        results = self._make_results()
        cfg = EdgeFilterConfig(top_k=2, deduplicate=False)
        filtered = apply_edge_filter(results, cfg)
        assert len(filtered) == 2

    def test_apply_edge_filter_deduplicate(self):
        results = self._make_results()
        cfg = EdgeFilterConfig(deduplicate=True)
        filtered = apply_edge_filter(results, cfg)
        keys = [r.pair_key for r in filtered]
        assert len(keys) == len(set(keys))

    def test_apply_edge_filter_only_compatible(self):
        results = self._make_results()
        cfg = EdgeFilterConfig(only_compatible=True, deduplicate=False)
        filtered = apply_edge_filter(results, cfg)
        assert all(r.is_compatible for r in filtered)

    def test_batch_filter_edges_length(self):
        batch = [self._make_results() for _ in range(3)]
        cfg = EdgeFilterConfig(min_score=0.5, deduplicate=True)
        filtered_batches = batch_filter_edges(batch, cfg)
        assert len(filtered_batches) == 3

    def test_filter_empty_list(self):
        assert filter_by_score([], min_score=0.5) == []
        assert filter_top_k([], k=5) == []
        assert filter_compatible([]) == []
        assert deduplicate_pairs([]) == []


# =============================================================================
# TestFourierDescriptor
# =============================================================================

class TestFourierDescriptor:

    def test_config_defaults(self):
        cfg = FourierConfig()
        assert cfg.n_coeffs == 32
        assert cfg.normalize is True

    def test_config_invalid_n_coeffs(self):
        with pytest.raises(ValueError, match="n_coeffs"):
            FourierConfig(n_coeffs=3)

    def test_compute_contour_centroid_circle(self):
        pts = make_circle_contour(n=64, r=50.0, cx=10.0, cy=20.0)
        cx, cy = compute_contour_centroid(pts)
        assert abs(cx - 10.0) < 1.0
        assert abs(cy - 20.0) < 1.0

    def test_compute_contour_centroid_empty_raises(self):
        with pytest.raises(ValueError):
            compute_contour_centroid(np.zeros((0, 2)))

    def test_complex_representation_shape(self):
        pts = make_circle_contour(n=32)
        z = complex_representation(pts)
        assert z.shape == (32,)
        assert z.dtype == complex

    def test_complex_representation_values(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        z = complex_representation(pts)
        assert z[0] == pytest.approx(1.0 + 2.0j)
        assert z[1] == pytest.approx(3.0 + 4.0j)

    def test_compute_fd_output_type(self):
        pts = make_circle_contour(n=64)
        fd = compute_fd(pts, fragment_id=1, edge_id=2)
        assert isinstance(fd, FourierDescriptor)
        assert fd.fragment_id == 1
        assert fd.edge_id == 2

    def test_compute_fd_coefficient_shape(self):
        pts = make_circle_contour(n=64)
        cfg = FourierConfig(n_coeffs=16)
        fd = compute_fd(pts, cfg=cfg)
        assert fd.coefficients.shape == (32,)  # 2 * n_coeffs
        assert fd.dim == 32

    def test_compute_fd_magnitude_shape(self):
        pts = make_circle_contour(n=64)
        cfg = FourierConfig(n_coeffs=16)
        fd = compute_fd(pts, cfg=cfg)
        assert fd.magnitude.shape == (16,)

    def test_compute_fd_magnitude_nonnegative(self):
        pts = make_circle_contour(n=64)
        fd = compute_fd(pts)
        assert np.all(fd.magnitude >= 0)

    def test_compute_fd_too_few_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="4"):
            compute_fd(pts)

    def test_compute_fd_normalized_first_nonzero_coeff_one(self):
        pts = make_circle_contour(n=64)
        cfg = FourierConfig(n_coeffs=16, normalize=True)
        fd = compute_fd(pts, cfg=cfg)
        # For a centered circle the DC component (index 0) is ~0 after mean subtraction.
        # Normalization divides by the amplitude of the first nonzero coefficient,
        # so the dominant non-DC coefficient magnitude should equal 1.0.
        # For a circle, that is coefficient index 1.
        assert abs(fd.magnitude[1] - 1.0) < 0.05

    def test_fd_similarity_self_is_one(self):
        pts = make_circle_contour(n=64)
        fd = compute_fd(pts)
        sim = fd_similarity(fd, fd)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_fd_similarity_range(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_square_contour(n=20)
        fd1 = compute_fd(pts1)
        fd2 = compute_fd(pts2)
        sim = fd_similarity(fd1, fd2)
        assert 0.0 <= sim <= 1.0

    def test_fd_similarity_different_n_coeffs_raises(self):
        pts = make_circle_contour(n=64)
        fd1 = compute_fd(pts, cfg=FourierConfig(n_coeffs=8))
        fd2 = compute_fd(pts, cfg=FourierConfig(n_coeffs=16))
        with pytest.raises(ValueError, match="n_coeffs"):
            fd_similarity(fd1, fd2)

    def test_batch_compute_fd_length(self):
        pts_list = [make_circle_contour(n=32 + 8 * i) for i in range(5)]
        fds = batch_compute_fd(pts_list)
        assert len(fds) == 5
        for i, fd in enumerate(fds):
            assert fd.edge_id == i

    def test_rank_by_fd_sorted_descending(self):
        pts_query = make_circle_contour(n=64)
        pts_list = [make_circle_contour(n=32 + 8 * i) for i in range(4)]
        query = compute_fd(pts_query)
        candidates = batch_compute_fd(pts_list)
        ranked = rank_by_fd(query, candidates)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_fd_custom_indices(self):
        pts_query = make_circle_contour(n=64)
        pts_list = [make_circle_contour(n=64), make_square_contour(n=20)]
        query = compute_fd(pts_query)
        candidates = batch_compute_fd(pts_list)
        ranked = rank_by_fd(query, candidates, indices=[10, 20])
        assert {r[0] for r in ranked} == {10, 20}

    def test_fd_circle_vs_square_similarity_less_than_self(self):
        pts_circle = make_circle_contour(n=64)
        pts_square = make_square_contour(n=20)
        fd_circle = compute_fd(pts_circle)
        fd_square = compute_fd(pts_square)
        sim_self = fd_similarity(fd_circle, fd_circle)
        sim_cross = fd_similarity(fd_circle, fd_square)
        assert sim_self > sim_cross


# =============================================================================
# TestWaveletDescriptor
# =============================================================================

class TestWaveletDescriptor:

    def test_compute_wavelet_descriptor_returns_type(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts)
        assert isinstance(desc, WaveletDescriptor)

    def test_compute_wavelet_descriptor_coeffs_normalized(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts)
        norm = np.linalg.norm(desc.coeffs)
        assert abs(norm - 1.0) < 1e-6

    def test_compute_wavelet_descriptor_energy_sums_to_one(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts)
        if len(desc.energy_per_level) > 0:
            assert abs(desc.energy_per_level.sum() - 1.0) < 1e-6

    def test_compute_wavelet_descriptor_n_levels_stored(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts, n_levels=3)
        assert desc.n_levels == 3

    def test_compute_wavelet_descriptor_energy_per_level_length(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts, n_levels=3)
        assert len(desc.energy_per_level) == 3

    def test_compute_wavelet_descriptor_short_contour(self):
        pts = np.array([[0.0, 0.0]])  # less than 2 points
        desc = compute_wavelet_descriptor(pts, n_levels=4)
        assert len(desc.coeffs) == 2 ** 4

    def test_compute_wavelet_descriptor_custom_n_points(self):
        pts = make_circle_contour(n=100)
        desc = compute_wavelet_descriptor(pts, n_points=32, n_levels=3)
        assert len(desc.coeffs) > 0

    def test_compute_wavelet_descriptor_coeffs_dtype(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts)
        assert desc.coeffs.dtype in (np.float32, np.float64)

    def test_wavelet_similarity_self_is_one(self):
        pts = make_circle_contour(n=64)
        desc = compute_wavelet_descriptor(pts)
        sim = wavelet_similarity(desc, desc)
        assert abs(sim - 1.0) < 1e-5

    def test_wavelet_similarity_range(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_square_contour(n=20)
        d1 = compute_wavelet_descriptor(pts1)
        d2 = compute_wavelet_descriptor(pts2)
        sim = wavelet_similarity(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_wavelet_similarity_different_coeff_lengths(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_square_contour(n=20)
        d1 = compute_wavelet_descriptor(pts1, n_levels=4)
        d2 = compute_wavelet_descriptor(pts2, n_levels=2)
        # Should handle mismatched coefficient lengths
        sim = wavelet_similarity(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_wavelet_similarity_mirror_ge_direct(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_circle_contour(n=64)  # same shape, possibly mirrored
        d1 = compute_wavelet_descriptor(pts1)
        d2 = compute_wavelet_descriptor(pts2)
        sim_direct = wavelet_similarity(d1, d2)
        sim_mirror = wavelet_similarity_mirror(d1, d2)
        assert sim_mirror >= sim_direct - 1e-10

    def test_wavelet_similarity_mirror_range(self):
        pts1 = make_circle_contour(n=64)
        pts2 = make_square_contour(n=20)
        d1 = compute_wavelet_descriptor(pts1)
        d2 = compute_wavelet_descriptor(pts2)
        sim = wavelet_similarity_mirror(d1, d2)
        assert 0.0 <= sim <= 1.0

    def test_batch_wavelet_similarity_shape(self):
        pts_query = make_circle_contour(n=64)
        pts_list = [make_circle_contour(n=32 + 8 * i) for i in range(5)]
        query = compute_wavelet_descriptor(pts_query)
        candidates = [compute_wavelet_descriptor(p) for p in pts_list]
        scores = batch_wavelet_similarity(query, candidates)
        assert scores.shape == (5,)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_batch_wavelet_similarity_with_mirror(self):
        pts_query = make_circle_contour(n=64)
        pts_list = [make_circle_contour(n=64), make_square_contour(n=20)]
        query = compute_wavelet_descriptor(pts_query)
        candidates = [compute_wavelet_descriptor(p) for p in pts_list]
        scores = batch_wavelet_similarity(query, candidates, use_mirror=True)
        assert scores.shape == (2,)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_wavelet_circle_vs_square_self_comparison(self):
        pts_circle = make_circle_contour(n=64)
        pts_square = make_square_contour(n=20)
        dc = compute_wavelet_descriptor(pts_circle)
        ds = compute_wavelet_descriptor(pts_square)
        sim_self = wavelet_similarity(dc, dc)
        sim_cross = wavelet_similarity(dc, ds)
        # Self similarity should be highest
        assert sim_self >= sim_cross - 0.01
