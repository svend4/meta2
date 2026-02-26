"""
Extra edge-case tests for Phase 3 preprocessing modules.

Covers cases NOT addressed in the basic test files:
  - puzzle_reconstruction.preprocessing.tear_enhancer
  - puzzle_reconstruction.preprocessing.multiscale_segmenter
  - puzzle_reconstruction.preprocessing.illumination_equalizer
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.tear_enhancer import (
    TearEnhancerConfig,
    TearEnhancerResult,
    TearEdgeEnhancer,
    enhance_torn_edge,
)
from puzzle_reconstruction.preprocessing.multiscale_segmenter import (
    MultiscaleConfig,
    MultiscaleSegmentationResult,
    MultiscaleSegmenter,
    segment_multiscale,
)
from puzzle_reconstruction.preprocessing.illumination_equalizer import (
    IlluminationEqualizerConfig,
    IlluminationEqualizerResult,
    IlluminationEqualizer,
    equalize_fragments,
)


# =============================================================================
# Shared helpers
# =============================================================================

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _gray(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    return _rng(seed).integers(30, 225, (h, w), dtype=np.uint8)


def _rgb(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    return _rng(seed).integers(30, 225, (h, w, 3), dtype=np.uint8)


def _contour(n: int = 10, h: int = 32, w: int = 32) -> np.ndarray:
    """Simple rectangular contour with *n* points."""
    n = max(n, 4)
    side = n // 4
    pts = []
    for i in range(side):
        t = i / max(side - 1, 1)
        pts.append([int(w * 0.1 + t * w * 0.8), int(h * 0.1)])
    for i in range(side):
        t = i / max(side - 1, 1)
        pts.append([int(w * 0.9), int(h * 0.1 + t * h * 0.8)])
    for i in range(side):
        t = i / max(side - 1, 1)
        pts.append([int(w * 0.9 - t * w * 0.8), int(h * 0.9)])
    for i in range(side):
        t = i / max(side - 1, 1)
        pts.append([int(w * 0.1), int(h * 0.9 - t * h * 0.8)])
    return np.array(pts, dtype=np.float32)


# =============================================================================
# TearEdgeEnhancer – extra edge cases
# =============================================================================

class TestTearEnhancerEdgeCases:

    # --- Very small images ---------------------------------------------------

    def test_4x4_gray_image_does_not_crash(self):
        img = _gray(4, 4)
        contour = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float32)
        result = enhance_torn_edge(img, contour)
        assert result.enhanced_image.shape == (4, 4)

    def test_4x4_rgb_image_does_not_crash(self):
        img = _rgb(4, 4)
        contour = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float32)
        result = enhance_torn_edge(img, contour)
        assert result.enhanced_image.shape == (4, 4, 3)

    # --- Single-channel vs 3-channel consistency ----------------------------

    def test_gray_and_rgb_both_preserve_dtype(self):
        gray = _gray()
        rgb = _rgb()
        contour = _contour()
        r_g = enhance_torn_edge(gray, contour)
        r_c = enhance_torn_edge(rgb, contour)
        assert r_g.enhanced_image.dtype == gray.dtype
        assert r_c.enhanced_image.dtype == rgb.dtype

    def test_gray_output_is_2d(self):
        img = _gray()
        contour = _contour()
        result = enhance_torn_edge(img, contour)
        assert result.enhanced_image.ndim == 2

    def test_rgb_output_is_3d_with_3_channels(self):
        img = _rgb()
        contour = _contour()
        result = enhance_torn_edge(img, contour)
        assert result.enhanced_image.ndim == 3
        assert result.enhanced_image.shape[2] == 3

    # --- Contour with exactly 2 points --------------------------------------

    def test_contour_exactly_2_points_does_not_crash(self):
        img = _gray()
        contour = np.array([[5.0, 5.0], [25.0, 25.0]], dtype=np.float32)
        result = enhance_torn_edge(img, contour)
        assert result.enhanced_contour.ndim == 2
        assert result.enhanced_contour.shape[1] == 2

    def test_contour_2pts_supersample_factor_1_returns_same_points(self):
        img = _gray()
        contour = np.array([[5.0, 5.0], [25.0, 25.0]], dtype=np.float32)
        cfg = TearEnhancerConfig(supersample_factor=1)
        result = enhance_torn_edge(img, contour, config=cfg)
        assert result.enhanced_contour.shape[0] == 2

    # --- supersample_factor=4 -----------------------------------------------

    def test_supersample_factor_4_expands_contour(self):
        img = _gray()
        n_pts = 8
        contour = _contour(n=n_pts)
        cfg = TearEnhancerConfig(supersample_factor=4)
        result = enhance_torn_edge(img, contour, config=cfg)
        assert result.enhanced_contour.shape[0] > n_pts

    def test_supersample_factor_4_contour_dtype_float32(self):
        img = _gray()
        contour = _contour(n=8)
        cfg = TearEnhancerConfig(supersample_factor=4)
        result = enhance_torn_edge(img, contour, config=cfg)
        assert result.enhanced_contour.dtype == np.float32

    # --- denoise_radius=0 ---------------------------------------------------

    def test_denoise_radius_0_does_not_crash_gray(self):
        img = _gray()
        contour = _contour()
        cfg = TearEnhancerConfig(denoise_radius=0)
        result = enhance_torn_edge(img, contour, config=cfg)
        assert result.enhanced_image.shape == img.shape

    def test_denoise_radius_0_does_not_crash_rgb(self):
        img = _rgb()
        contour = _contour()
        cfg = TearEnhancerConfig(denoise_radius=0)
        result = enhance_torn_edge(img, contour, config=cfg)
        assert result.enhanced_image.shape == img.shape

    # --- Output is non-NaN --------------------------------------------------

    def test_no_nan_in_enhanced_image_gray(self):
        img = _gray()
        contour = _contour()
        result = enhance_torn_edge(img, contour)
        assert not np.any(np.isnan(result.enhanced_image.astype(float)))

    def test_no_nan_in_enhanced_image_rgb(self):
        img = _rgb()
        contour = _contour()
        result = enhance_torn_edge(img, contour)
        assert not np.any(np.isnan(result.enhanced_image.astype(float)))

    def test_no_nan_in_enhanced_contour(self):
        img = _gray()
        contour = _contour()
        result = enhance_torn_edge(img, contour)
        assert not np.any(np.isnan(result.enhanced_contour))

    def test_no_nan_in_sharpness_values(self):
        img = _gray()
        contour = _contour()
        result = enhance_torn_edge(img, contour)
        assert not np.isnan(result.sharpness_before)
        assert not np.isnan(result.sharpness_after)

    # --- Result types always correct ----------------------------------------

    def test_result_type_is_always_TearEnhancerResult(self):
        for seed in range(5):
            img = _gray(seed=seed)
            contour = _contour()
            result = enhance_torn_edge(img, contour)
            assert isinstance(result, TearEnhancerResult)

    # --- Chain: equalize → segment → enhance --------------------------------

    def test_chain_equalize_segment_enhance(self):
        """IlluminationEqualizer output feeds MultiscaleSegmenter, then TearEdgeEnhancer."""
        img = _rgb(64, 64)
        # Step 1: equalize
        eq_result = IlluminationEqualizer().equalize([img])
        eq_img = eq_result.images[0]

        # Step 2: segment (produces a mask, not directly used by enhance, but verifies no crash)
        seg_result = MultiscaleSegmenter().segment(eq_img)
        assert seg_result.mask.shape == eq_img.shape[:2]

        # Step 3: enhance (use the equalized image)
        contour = _contour(h=64, w=64)
        enh_result = enhance_torn_edge(eq_img, contour)
        assert enh_result.enhanced_image.shape == eq_img.shape


# =============================================================================
# MultiscaleSegmenter – extra edge cases
# =============================================================================

class TestMultiscaleSegmenterEdgeCases:

    # --- Very small images --------------------------------------------------

    def test_4x4_gray_image(self):
        img = np.full((4, 4), 128, dtype=np.uint8)
        result = MultiscaleSegmenter().segment(img)
        assert result.mask.shape == (4, 4)
        assert result.mask.dtype == bool

    def test_4x4_rgb_image(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = MultiscaleSegmenter().segment(img)
        assert result.mask.shape == (4, 4)

    # --- All-zeros image ----------------------------------------------------

    def test_all_zeros_image_returns_valid_result(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = MultiscaleSegmenter().segment(img)
        assert isinstance(result, MultiscaleSegmentationResult)
        assert result.mask.shape == (32, 32)
        assert result.confidence_map.shape == (32, 32)

    def test_all_zeros_confidence_map_range(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = MultiscaleSegmenter().segment(img)
        assert float(result.confidence_map.min()) >= 0.0
        assert float(result.confidence_map.max()) <= 1.0

    # --- Single non-zero pixel ----------------------------------------------

    def test_single_nonzero_pixel(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 16] = 255
        result = MultiscaleSegmenter().segment(img)
        assert result.mask.shape == (32, 32)

    # --- Single scale=[1.0] -------------------------------------------------

    def test_single_scale_1_works(self):
        img = _gray()
        cfg = MultiscaleConfig(scales=[1.0])
        result = MultiscaleSegmenter(cfg).segment(img)
        assert isinstance(result, MultiscaleSegmentationResult)
        assert result.n_scales_used == 1

    def test_single_scale_confidence_map_float32(self):
        img = _gray()
        cfg = MultiscaleConfig(scales=[1.0])
        result = MultiscaleSegmenter(cfg).segment(img)
        assert result.confidence_map.dtype == np.float32

    # --- Config with extreme scales=[1.0] + min_area=0 ----------------------

    def test_min_area_zero_keeps_all_components(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[5, 5] = 255   # single bright pixel
        cfg = MultiscaleConfig(scales=[1.0], min_area=0)
        result = MultiscaleSegmenter(cfg).segment(img)
        assert result.mask.shape == (32, 32)

    # --- Result types -------------------------------------------------------

    def test_mask_is_always_bool(self):
        for seed in range(3):
            img = _gray(seed=seed)
            result = MultiscaleSegmenter().segment(img)
            assert result.mask.dtype == bool

    def test_confidence_map_in_0_1(self):
        img = _gray()
        result = MultiscaleSegmenter().segment(img)
        assert float(result.confidence_map.min()) >= 0.0
        assert float(result.confidence_map.max()) <= 1.0 + 1e-6

    def test_n_scales_used_nonnegative(self):
        img = _gray()
        result = MultiscaleSegmenter().segment(img)
        assert result.n_scales_used >= 0

    def test_scales_list_preserved_in_result(self):
        cfg_scales = [1.0, 0.5]
        cfg = MultiscaleConfig(scales=cfg_scales)
        img = _gray()
        result = MultiscaleSegmenter(cfg).segment(img)
        assert result.scales == cfg_scales

    # --- No NaN in output ---------------------------------------------------

    def test_no_nan_confidence_map(self):
        img = _gray()
        result = MultiscaleSegmenter().segment(img)
        assert not np.any(np.isnan(result.confidence_map))

    # --- 3-channel vs 1-channel consistency ---------------------------------

    def test_gray_and_rgb_produce_same_spatial_shape(self):
        img_gray = _gray(48, 48)
        img_rgb = _rgb(48, 48)
        r_g = MultiscaleSegmenter().segment(img_gray)
        r_c = MultiscaleSegmenter().segment(img_rgb)
        assert r_g.mask.shape == (48, 48)
        assert r_c.mask.shape == (48, 48)

    # --- adaptive and triangle methods on tiny images -----------------------

    def test_adaptive_method_tiny_image(self):
        img = _gray(8, 8)
        cfg = MultiscaleConfig(scales=[1.0], method="adaptive", min_area=0)
        result = MultiscaleSegmenter(cfg).segment(img)
        assert result.mask.shape == (8, 8)

    def test_triangle_method_tiny_image(self):
        img = _gray(8, 8)
        cfg = MultiscaleConfig(scales=[1.0], method="triangle", min_area=0)
        result = MultiscaleSegmenter(cfg).segment(img)
        assert result.mask.shape == (8, 8)


# =============================================================================
# IlluminationEqualizer – extra edge cases
# =============================================================================

class TestIlluminationEqualizerEdgeCases:

    # --- Single image list --------------------------------------------------

    def test_equalize_single_image_returns_one_image(self):
        img = _gray()
        result = IlluminationEqualizer().equalize([img])
        assert len(result.images) == 1

    def test_equalize_single_image_shape_preserved(self):
        img = _gray()
        result = IlluminationEqualizer().equalize([img])
        assert result.images[0].shape == img.shape

    def test_equalize_single_rgb_image_unchanged_shape(self):
        img = _rgb()
        result = IlluminationEqualizer().equalize([img])
        assert result.images[0].shape == img.shape

    # --- reference_idx=0 vs reference_idx=1 give different results ----------

    def test_reference_idx_0_vs_1_differ(self):
        imgs = [_gray(seed=0), _gray(seed=1), _gray(seed=2)]
        cfg0 = IlluminationEqualizerConfig(method="histogram", reference_idx=0)
        cfg1 = IlluminationEqualizerConfig(method="histogram", reference_idx=1)
        r0 = IlluminationEqualizer(cfg0).equalize(imgs)
        r1 = IlluminationEqualizer(cfg1).equalize(imgs)
        # The non-reference images should differ between the two runs
        different = not np.array_equal(r0.images[1], r1.images[1])
        # At least one non-reference result should differ
        assert different or not np.array_equal(r0.images[0], r1.images[0])

    # --- Result types always correct ----------------------------------------

    def test_output_images_list_length_matches_input(self):
        imgs = [_gray(seed=i) for i in range(5)]
        result = IlluminationEqualizer().equalize(imgs)
        assert len(result.images) == 5

    def test_uniformity_scores_length_matches_input(self):
        imgs = [_gray(seed=i) for i in range(4)]
        result = IlluminationEqualizer().equalize(imgs)
        assert len(result.uniformity_scores) == 4

    def test_uniformity_scores_in_0_1(self):
        imgs = [_gray(seed=i) for i in range(3)]
        result = IlluminationEqualizer().equalize(imgs)
        for s in result.uniformity_scores:
            assert 0.0 <= s <= 1.0, f"Score out of range: {s}"

    def test_result_method_field_correct(self):
        imgs = [_gray()]
        for method in ("histogram", "retinex", "clahe"):
            cfg = IlluminationEqualizerConfig(method=method)
            result = IlluminationEqualizer(cfg).equalize(imgs)
            assert result.method == method

    # --- Very small images --------------------------------------------------

    def test_4x4_histogram_method(self):
        imgs = [np.full((4, 4), v, dtype=np.uint8) for v in [100, 150]]
        result = IlluminationEqualizer().equalize(imgs)
        assert result.images[0].shape == (4, 4)

    def test_4x4_retinex_method(self):
        imgs = [_gray(4, 4, seed=i) for i in range(2)]
        cfg = IlluminationEqualizerConfig(method="retinex")
        result = IlluminationEqualizer(cfg).equalize(imgs)
        assert result.images[0].shape == (4, 4)

    def test_4x4_clahe_method(self):
        imgs = [_gray(4, 4, seed=i) for i in range(2)]
        cfg = IlluminationEqualizerConfig(method="clahe")
        result = IlluminationEqualizer(cfg).equalize(imgs)
        assert result.images[0].shape == (4, 4)

    # --- equalize_fragments convenience wrapper ----------------------------

    def test_equalize_fragments_returns_list(self):
        imgs = [_gray(seed=i) for i in range(3)]
        out = equalize_fragments(imgs, method="histogram")
        assert isinstance(out, list)
        assert len(out) == 3

    def test_equalize_fragments_single_image(self):
        img = _gray()
        out = equalize_fragments([img])
        assert len(out) == 1
        assert out[0].shape == img.shape

    # --- No NaN in output ---------------------------------------------------

    def test_no_nan_in_histogram_output(self):
        imgs = [_gray(seed=i) for i in range(3)]
        result = IlluminationEqualizer().equalize(imgs)
        for img in result.images:
            assert not np.any(np.isnan(img.astype(float)))

    def test_no_nan_in_retinex_output(self):
        imgs = [_gray(seed=i) for i in range(2)]
        cfg = IlluminationEqualizerConfig(method="retinex")
        result = IlluminationEqualizer(cfg).equalize(imgs)
        for img in result.images:
            assert not np.any(np.isnan(img.astype(float)))

    # --- Chain: equalize → segment ------------------------------------------

    def test_chain_equalize_then_segment_no_crash(self):
        imgs = [_gray(32, 32, seed=i) for i in range(3)]
        eq_result = IlluminationEqualizer().equalize(imgs)
        for eq_img in eq_result.images:
            seg_result = MultiscaleSegmenter().segment(eq_img)
            assert seg_result.mask.shape == eq_img.shape

    # --- Config n_bins extreme value (via segmenter to exercise chain) ------
    # (IlluminationEqualizer does not have n_bins, but MultiscaleSegmenter
    #  indirectly depends on it via the histogram used in Otsu.)

    def test_equalize_retinex_rgb_output_uint8(self):
        imgs = [_rgb(32, 32, seed=i) for i in range(2)]
        cfg = IlluminationEqualizerConfig(method="retinex")
        result = IlluminationEqualizer(cfg).equalize(imgs)
        for img in result.images:
            assert img.dtype == np.uint8

    def test_multiscale_config_n_bins_extreme_otsu(self):
        """MultiscaleSegmenter with very small image (4×4) doesn't crash."""
        img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = segment_multiscale(img)
        assert isinstance(result, MultiscaleSegmentationResult)
