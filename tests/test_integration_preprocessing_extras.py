"""
Integration tests for puzzle_reconstruction preprocessing modules.

Covers:
    - adaptive_threshold
    - augment
    - background_remover
    - binarizer
    - channel_splitter
    - color_normalizer
    - contour_processor
    - contrast
    - contrast_enhancer
    - denoise
    - deskewer
    - document_cleaner
    - edge_detector
    - edge_enhancer
    - noise_reducer
    - morphology_ops
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)

def _color(h=100, w=100):
    """BGR uint8 image with random content."""
    return RNG.randint(0, 256, (h, w, 3), dtype=np.uint8)

def _gray(h=100, w=100):
    """Grayscale uint8 image with random content."""
    return RNG.randint(0, 256, (h, w), dtype=np.uint8)

def _binary(h=60, w=60):
    """Binary (0/255) uint8 image."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[20:40, 20:40] = 255
    return img

COLOR = _color()
GRAY = _gray()
BINARY = _binary()

# ===========================================================================
# 1. adaptive_threshold
# ===========================================================================

from puzzle_reconstruction.preprocessing.adaptive_threshold import (
    ThresholdParams,
    global_threshold,
    adaptive_mean,
    adaptive_gaussian,
    niblack_threshold,
    sauvola_threshold,
    bernsen_threshold,
    apply_threshold,
    batch_threshold,
)


class TestAdaptiveThreshold:

    def test_threshold_params_default(self):
        p = ThresholdParams()
        assert p.method == "otsu"

    def test_threshold_params_invalid_method(self):
        with pytest.raises(ValueError):
            ThresholdParams(method="unknown_method")

    def test_threshold_params_even_block(self):
        with pytest.raises(ValueError):
            ThresholdParams(block_size=10)

    def test_global_threshold_output_shape(self):
        result = global_threshold(GRAY)
        assert result.shape == GRAY.shape
        assert result.dtype == np.uint8

    def test_global_threshold_otsu(self):
        result = global_threshold(COLOR, use_otsu=True)
        assert result.shape == GRAY.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_adaptive_mean_output(self):
        result = adaptive_mean(GRAY, block_size=11)
        assert result.shape == GRAY.shape
        assert result.dtype == np.uint8

    def test_adaptive_gaussian_output(self):
        result = adaptive_gaussian(COLOR, block_size=11)
        assert result.shape == GRAY.shape

    def test_niblack_threshold_small_image(self):
        img = RNG.randint(0, 256, (20, 20), dtype=np.uint8)
        result = niblack_threshold(img, block_size=5)
        assert result.shape == img.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_sauvola_threshold_output(self):
        img = RNG.randint(0, 256, (20, 20), dtype=np.uint8)
        result = sauvola_threshold(img, block_size=5)
        assert result.shape == img.shape
        assert set(np.unique(result)).issubset({0, 255})

    def test_bernsen_threshold_output(self):
        img = RNG.randint(0, 256, (20, 20), dtype=np.uint8)
        result = bernsen_threshold(img, block_size=5)
        assert result.shape == img.shape

    def test_apply_threshold_dispatch(self):
        for method in ("otsu", "global", "adaptive_mean", "adaptive_gaussian", "bernsen"):
            p = ThresholdParams(method=method)
            out = apply_threshold(GRAY, p)
            assert out.shape == GRAY.shape

    def test_batch_threshold(self):
        imgs = [GRAY, _gray(50, 50)]
        results = batch_threshold(imgs, ThresholdParams(method="otsu"))
        assert len(results) == 2


# ===========================================================================
# 2. augment
# ===========================================================================

from puzzle_reconstruction.preprocessing.augment import (
    random_crop,
    random_rotate,
    add_gaussian_noise,
    add_salt_pepper,
    brightness_jitter,
    jpeg_compress,
    simulate_scan_noise,
    augment_batch,
)


class TestAugment:

    def test_random_crop_shape(self):
        out = random_crop(COLOR, rng=RNG)
        assert out.shape == COLOR.shape

    def test_random_rotate_same_size(self):
        out = random_rotate(COLOR, max_angle=10.0, rng=RNG)
        assert out.shape == COLOR.shape

    def test_random_rotate_expand(self):
        out = random_rotate(COLOR, max_angle=10.0, expand=True, rng=RNG)
        assert out.ndim == 3

    def test_add_gaussian_noise_dtype(self):
        out = add_gaussian_noise(COLOR, sigma=15.0, rng=RNG)
        assert out.dtype == np.uint8
        assert out.shape == COLOR.shape

    def test_add_gaussian_noise_zero_sigma_returns_input(self):
        out = add_gaussian_noise(COLOR, sigma=0.0, rng=RNG)
        np.testing.assert_array_equal(out, COLOR)

    def test_add_salt_pepper(self):
        out = add_salt_pepper(COLOR, amount=0.05, rng=RNG)
        assert out.shape == COLOR.shape
        assert out.dtype == np.uint8

    def test_brightness_jitter(self):
        out = brightness_jitter(COLOR, rng=RNG)
        assert out.shape == COLOR.shape
        assert out.dtype == np.uint8

    def test_jpeg_compress(self):
        out = jpeg_compress(COLOR, quality=50)
        assert out.shape == COLOR.shape
        assert out.dtype == np.uint8

    def test_simulate_scan_noise(self):
        out = simulate_scan_noise(COLOR, rng=RNG)
        assert out.shape == COLOR.shape
        assert out.dtype == np.uint8

    def test_augment_batch_length(self):
        imgs = [COLOR, _color(80, 80)]
        result = augment_batch(imgs, n_augments=2, seed=0)
        assert len(result) == len(imgs) * (1 + 2)


# ===========================================================================
# 3. background_remover
# ===========================================================================

from puzzle_reconstruction.preprocessing.background_remover import (
    BackgroundRemovalResult,
    remove_background_thresh,
    remove_background_edges,
    remove_background_grabcut,
    auto_remove_background,
    batch_remove_background,
)


class TestBackgroundRemover:

    def test_thresh_returns_result(self):
        r = remove_background_thresh(COLOR)
        assert isinstance(r, BackgroundRemovalResult)
        assert r.method == "thresh"

    def test_thresh_mask_shape(self):
        r = remove_background_thresh(GRAY)
        assert r.mask.shape == GRAY.shape
        assert r.foreground.shape == GRAY.shape

    def test_edges_method(self):
        r = remove_background_edges(COLOR)
        assert r.method == "edges"
        assert r.mask.shape == GRAY.shape

    def test_grabcut_method(self):
        r = remove_background_grabcut(COLOR, margin=5, n_iter=1)
        assert r.method == "grabcut"
        assert r.mask.shape == GRAY.shape

    def test_auto_remove_thresh(self):
        r = auto_remove_background(COLOR, method="thresh")
        assert r.method == "thresh"

    def test_auto_remove_invalid_method(self):
        with pytest.raises(ValueError):
            auto_remove_background(COLOR, method="invalid")

    def test_batch_remove(self):
        imgs = [COLOR, _color(80, 80)]
        results = batch_remove_background(imgs, method="thresh")
        assert len(results) == 2
        assert all(isinstance(r, BackgroundRemovalResult) for r in results)


# ===========================================================================
# 4. binarizer
# ===========================================================================

from puzzle_reconstruction.preprocessing.binarizer import (
    BinarizeResult,
    binarize_otsu,
    binarize_adaptive,
    binarize_sauvola,
    binarize_niblack,
    binarize_bernsen,
    auto_binarize,
    batch_binarize,
)


class TestBinarizer:

    def test_otsu_result_type(self):
        r = binarize_otsu(GRAY)
        assert isinstance(r, BinarizeResult)
        assert r.method == "otsu"

    def test_otsu_binary_values(self):
        r = binarize_otsu(COLOR)
        assert set(np.unique(r.binary)).issubset({0, 255})

    def test_otsu_foreground_ratio(self):
        r = binarize_otsu(GRAY)
        assert 0.0 <= r.foreground_ratio <= 1.0

    def test_adaptive_gaussian(self):
        r = binarize_adaptive(GRAY, adaptive_method="gaussian")
        assert r.method == "adaptive_gaussian"
        assert r.binary.shape == GRAY.shape

    def test_adaptive_mean(self):
        r = binarize_adaptive(GRAY, adaptive_method="mean")
        assert r.method == "adaptive_mean"

    def test_sauvola(self):
        r = binarize_sauvola(GRAY)
        assert r.method == "sauvola"
        assert r.binary.dtype == np.uint8

    def test_niblack(self):
        r = binarize_niblack(GRAY)
        assert r.method == "niblack"

    def test_bernsen(self):
        r = binarize_bernsen(GRAY)
        assert r.method == "bernsen"

    def test_auto_binarize(self):
        r = auto_binarize(GRAY)
        assert r.method in {"otsu", "sauvola"}

    def test_batch_binarize(self):
        results = batch_binarize([GRAY, COLOR], method="otsu")
        assert len(results) == 2

    def test_batch_binarize_invalid(self):
        with pytest.raises(ValueError):
            batch_binarize([GRAY], method="not_a_method")


# ===========================================================================
# 5. channel_splitter
# ===========================================================================

from puzzle_reconstruction.preprocessing.channel_splitter import (
    ChannelStats,
    split_channels,
    merge_channels,
    channel_statistics,
    equalize_channel,
    normalize_channel,
    channel_difference,
    apply_per_channel,
    batch_split,
)


class TestChannelSplitter:

    def test_split_color_three_channels(self):
        channels = split_channels(COLOR)
        assert len(channels) == 3
        assert all(c.ndim == 2 for c in channels)

    def test_split_gray_one_channel(self):
        channels = split_channels(GRAY)
        assert len(channels) == 1

    def test_merge_channels(self):
        channels = split_channels(COLOR)
        merged = merge_channels(channels)
        assert merged.shape == COLOR.shape

    def test_merge_single_channel(self):
        ch = GRAY[:, :, np.newaxis][:, :, 0]
        result = merge_channels([ch])
        assert result.ndim == 2

    def test_channel_statistics_type(self):
        stats = channel_statistics(GRAY)
        assert isinstance(stats, ChannelStats)
        assert stats.min_val <= stats.mean <= stats.max_val

    def test_equalize_channel_shape(self):
        out = equalize_channel(GRAY)
        assert out.shape == GRAY.shape
        assert out.dtype == np.uint8

    def test_normalize_channel_range(self):
        out = normalize_channel(GRAY, out_min=0.0, out_max=1.0)
        assert out.min() >= 0.0 - 1e-9
        assert out.max() <= 1.0 + 1e-9

    def test_channel_difference(self):
        ch1 = GRAY.copy()
        ch2 = GRAY.copy()
        diff = channel_difference(ch1, ch2)
        assert diff.min() >= 0.0
        np.testing.assert_allclose(diff, 0.0, atol=1e-9)

    def test_apply_per_channel(self):
        result = apply_per_channel(COLOR, lambda c: c)
        np.testing.assert_array_equal(result, COLOR)

    def test_batch_split(self):
        result = batch_split([COLOR, GRAY])
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 1


# ===========================================================================
# 6. color_normalizer
# ===========================================================================

from puzzle_reconstruction.preprocessing.color_normalizer import (
    NormConfig,
    NormResult,
    gamma_correction,
    equalize_histogram,
    apply_clahe,
    grey_world_balance,
    max_rgb_balance,
    minmax_normalize,
    normalize_image,
    batch_normalize,
)


class TestColorNormalizer:

    def test_norm_config_default(self):
        cfg = NormConfig()
        assert cfg.method == "clahe"

    def test_norm_config_invalid(self):
        with pytest.raises(ValueError):
            NormConfig(method="bogus")

    def test_gamma_correction_shape(self):
        out = gamma_correction(COLOR, gamma=1.0)
        assert out.shape == COLOR.shape
        assert out.dtype == np.uint8

    def test_equalize_histogram_gray(self):
        out = equalize_histogram(GRAY)
        assert out.shape == GRAY.shape

    def test_equalize_histogram_color(self):
        out = equalize_histogram(COLOR)
        assert out.shape == COLOR.shape

    def test_apply_clahe_gray(self):
        out = apply_clahe(GRAY)
        assert out.shape == GRAY.shape

    def test_grey_world_balance(self):
        out = grey_world_balance(COLOR)
        assert out.shape == COLOR.shape
        assert out.dtype == np.uint8

    def test_max_rgb_balance(self):
        out = max_rgb_balance(COLOR)
        assert out.shape == COLOR.shape

    def test_minmax_normalize(self):
        out = minmax_normalize(GRAY)
        assert out.dtype == np.uint8

    def test_normalize_image_result(self):
        r = normalize_image(COLOR)
        assert isinstance(r, NormResult)
        assert r.mean_before >= 0
        assert r.mean_after >= 0

    def test_batch_normalize(self):
        results = batch_normalize([COLOR, GRAY])
        assert len(results) == 2


# ===========================================================================
# 7. contour_processor
# ===========================================================================

from puzzle_reconstruction.preprocessing.contour_processor import (
    ContourConfig,
    ContourStats,
    ContourResult,
    resample_contour,
    smooth_contour,
    rdp_simplify,
    normalize_contour,
    contour_area,
    contour_perimeter,
    compute_contour_stats,
    process_contour,
    batch_process_contours,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                   [0.0, 0.5], [0.0, 0.0]], dtype=float)


class TestContourProcessor:

    def test_resample_contour_count(self):
        out = resample_contour(SQUARE, n_points=16)
        assert out.shape == (16, 2)

    def test_smooth_contour_shape(self):
        out = smooth_contour(SQUARE, sigma=1.0)
        assert out.shape == SQUARE.shape

    def test_smooth_contour_zero_sigma(self):
        out = smooth_contour(SQUARE, sigma=0.0)
        np.testing.assert_array_equal(out, SQUARE)

    def test_rdp_simplify_reduces(self):
        pts = np.array([[float(i), 0.0] for i in range(10)])
        out = rdp_simplify(pts, epsilon=0.5)
        assert out.shape[0] <= pts.shape[0]

    def test_normalize_contour_range(self):
        out = normalize_contour(SQUARE)
        assert out.min() >= -1.0 - 1e-9
        assert out.max() <= 1.0 + 1e-9

    def test_contour_area_positive(self):
        area = contour_area(SQUARE)
        assert area >= 0.0

    def test_contour_perimeter_positive(self):
        p = contour_perimeter(SQUARE)
        assert p > 0.0

    def test_compute_stats(self):
        stats = compute_contour_stats(SQUARE)
        assert isinstance(stats, ContourStats)
        assert stats.n_points == len(SQUARE)

    def test_process_contour(self):
        cfg = ContourConfig(n_points=16, smooth_sigma=0.5, rdp_epsilon=0.1)
        result = process_contour(SQUARE, fragment_id=1, cfg=cfg)
        assert isinstance(result, ContourResult)
        assert result.fragment_id == 1

    def test_batch_process_contours(self):
        contours = [SQUARE, SQUARE * 2.0]
        results = batch_process_contours(contours)
        assert len(results) == 2


# ===========================================================================
# 8. contrast
# ===========================================================================

from puzzle_reconstruction.preprocessing.contrast import (
    ContrastResult,
    measure_contrast,
    enhance_clahe,
    enhance_histeq,
    enhance_gamma,
    enhance_stretch,
    enhance_retinex,
    auto_enhance,
    batch_enhance,
)


class TestContrast:

    def test_measure_contrast_positive(self):
        val = measure_contrast(COLOR)
        assert val >= 0.0

    def test_enhance_clahe_result(self):
        r = enhance_clahe(GRAY)
        assert isinstance(r, ContrastResult)
        assert r.method == "clahe"
        assert r.enhanced.shape == GRAY.shape

    def test_enhance_histeq_color(self):
        r = enhance_histeq(COLOR)
        assert r.method == "histeq"
        assert r.enhanced.shape == COLOR.shape

    def test_enhance_gamma(self):
        r = enhance_gamma(GRAY, gamma=1.5)
        assert r.method == "gamma"

    def test_enhance_stretch(self):
        r = enhance_stretch(COLOR)
        assert r.method == "stretch"

    def test_enhance_retinex(self):
        r = enhance_retinex(GRAY, sigma=20.0)
        assert r.method == "retinex"

    def test_auto_enhance(self):
        r = auto_enhance(COLOR)
        assert r.method in {"clahe", "stretch", "gamma"}

    def test_batch_enhance(self):
        results = batch_enhance([COLOR, GRAY], method="histeq")
        assert len(results) == 2

    def test_improvement_property(self):
        r = enhance_clahe(GRAY)
        assert isinstance(r.improvement, float)

    def test_batch_enhance_invalid(self):
        with pytest.raises(ValueError):
            batch_enhance([GRAY], method="bad_method")


# ===========================================================================
# 9. contrast_enhancer
# ===========================================================================

from puzzle_reconstruction.preprocessing.contrast_enhancer import (
    EnhanceConfig,
    EnhanceResult,
    equalize_histogram as ce_equalize_histogram,
    stretch_contrast,
    apply_gamma,
    clahe_enhance,
    enhance_contrast,
    batch_enhance as ce_batch_enhance,
)


class TestContrastEnhancer:

    def test_enhance_config_default(self):
        cfg = EnhanceConfig()
        assert cfg.method == "equalize"

    def test_enhance_config_invalid(self):
        with pytest.raises(ValueError):
            EnhanceConfig(method="weird")

    def test_equalize_histogram_2d(self):
        cfg = EnhanceConfig(method="equalize")
        out = ce_equalize_histogram(GRAY, cfg)
        assert out.shape == GRAY.shape

    def test_stretch_contrast(self):
        cfg = EnhanceConfig(method="stretch")
        out = stretch_contrast(GRAY, cfg)
        assert out.shape == GRAY.shape

    def test_apply_gamma(self):
        cfg = EnhanceConfig(method="gamma", gamma=0.5)
        out = apply_gamma(GRAY, cfg)
        assert out.shape == GRAY.shape

    def test_clahe_enhance(self):
        cfg = EnhanceConfig(method="clahe", tile_size=8)
        out = clahe_enhance(GRAY, cfg)
        assert out.shape == GRAY.shape

    def test_enhance_contrast_result(self):
        r = enhance_contrast(GRAY)
        assert isinstance(r, EnhanceResult)
        assert r.method == "equalize"

    def test_enhance_contrast_color(self):
        cfg = EnhanceConfig(method="stretch")
        r = enhance_contrast(COLOR, cfg)
        assert r.image.shape == COLOR.shape

    def test_ce_batch_enhance(self):
        results = ce_batch_enhance([GRAY, COLOR])
        assert len(results) == 2

    def test_contrast_gain_property(self):
        r = enhance_contrast(GRAY)
        assert r.contrast_gain >= 0.0


# ===========================================================================
# 10. denoise
# ===========================================================================

from puzzle_reconstruction.preprocessing.denoise import (
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlmeans_denoise,
    auto_denoise,
    denoise_batch,
)


class TestDenoise:

    def test_gaussian_denoise_shape(self):
        out = gaussian_denoise(COLOR, sigma=1.5)
        assert out.shape == COLOR.shape

    def test_gaussian_denoise_zero_sigma(self):
        out = gaussian_denoise(COLOR, sigma=0.0)
        np.testing.assert_array_equal(out, COLOR)

    def test_median_denoise(self):
        out = median_denoise(COLOR, ksize=3)
        assert out.shape == COLOR.shape

    def test_bilateral_denoise(self):
        out = bilateral_denoise(COLOR, d=5)
        assert out.shape == COLOR.shape

    def test_nlmeans_denoise_gray(self):
        small = RNG.randint(0, 256, (50, 50), dtype=np.uint8)
        out = nlmeans_denoise(small)
        assert out.shape == small.shape

    def test_auto_denoise(self):
        out = auto_denoise(COLOR)
        assert out.shape == COLOR.shape

    def test_denoise_batch(self):
        results = denoise_batch([COLOR, GRAY], method="gaussian", sigma=1.0)
        assert len(results) == 2

    def test_denoise_batch_invalid(self):
        with pytest.raises(ValueError):
            denoise_batch([COLOR], method="not_a_method")


# ===========================================================================
# 11. deskewer
# ===========================================================================

from puzzle_reconstruction.preprocessing.deskewer import (
    DeskewResult,
    estimate_skew_projection,
    estimate_skew_hough,
    deskew_image,
    auto_deskew,
    batch_deskew,
)


class TestDeskewer:

    def test_estimate_skew_projection_returns_tuple(self):
        angle, conf = estimate_skew_projection(GRAY, n_angles=10)
        assert isinstance(angle, float)
        assert 0.0 <= conf <= 1.0

    def test_estimate_skew_hough(self):
        angle, conf = estimate_skew_hough(COLOR)
        assert isinstance(angle, float)
        assert 0.0 <= conf <= 1.0

    def test_deskew_image_shape(self):
        out = deskew_image(COLOR, angle=3.0)
        assert out.shape == COLOR.shape

    def test_deskew_image_zero_angle(self):
        out = deskew_image(GRAY, angle=0.0)
        assert out.shape == GRAY.shape

    def test_auto_deskew_projection(self):
        r = auto_deskew(COLOR, method="projection", n_angles=10)
        assert isinstance(r, DeskewResult)
        assert r.corrected.shape == COLOR.shape

    def test_auto_deskew_hough(self):
        r = auto_deskew(COLOR, method="hough")
        assert r.method == "hough"

    def test_auto_deskew_invalid_method(self):
        with pytest.raises(ValueError):
            auto_deskew(COLOR, method="magic")

    def test_batch_deskew(self):
        results = batch_deskew([COLOR, GRAY], method="projection", n_angles=10)
        assert len(results) == 2


# ===========================================================================
# 12. document_cleaner
# ===========================================================================

from puzzle_reconstruction.preprocessing.document_cleaner import (
    CleanResult,
    remove_shadow,
    remove_border_artifacts,
    normalize_illumination,
    remove_blobs,
    auto_clean,
    batch_clean,
)


class TestDocumentCleaner:

    def test_remove_shadow_shape(self):
        r = remove_shadow(COLOR)
        assert isinstance(r, CleanResult)
        assert r.cleaned.shape == COLOR.shape
        assert r.method == "shadow"

    def test_remove_border_artifacts(self):
        r = remove_border_artifacts(COLOR, border_px=5)
        assert r.method == "border"
        # Border pixels should be 255
        assert r.cleaned[0, 0, 0] == 255

    def test_normalize_illumination(self):
        r = normalize_illumination(GRAY, sigma=20.0)
        assert r.method == "illumination"
        assert r.cleaned.shape == GRAY.shape

    def test_remove_blobs(self):
        r = remove_blobs(GRAY, min_area=5, max_area=200)
        assert r.method == "blobs"
        assert isinstance(r.artifacts_removed, int)

    def test_auto_clean(self):
        r = auto_clean(COLOR)
        assert r.method == "auto"
        assert r.cleaned.shape == COLOR.shape

    def test_batch_clean(self):
        results = batch_clean([COLOR, GRAY], method="shadow")
        assert len(results) == 2

    def test_batch_clean_invalid(self):
        with pytest.raises(ValueError):
            batch_clean([COLOR], method="unknown_method")


# ===========================================================================
# 13. edge_detector
# ===========================================================================

from puzzle_reconstruction.preprocessing.edge_detector import (
    EdgeDetectionResult,
    detect_edges,
    adaptive_canny,
    sobel_edges,
    laplacian_edges,
    refine_edge_contour,
    edge_density,
    edge_orientation_hist,
)


class TestEdgeDetector:

    def test_detect_edges_adaptive_canny(self):
        r = detect_edges(COLOR, method="adaptive_canny")
        assert isinstance(r, EdgeDetectionResult)
        assert r.edge_map.shape == GRAY.shape

    def test_detect_edges_canny(self):
        r = detect_edges(GRAY, method="canny", threshold1=50, threshold2=150)
        assert r.edge_map.shape == GRAY.shape

    def test_detect_edges_sobel(self):
        r = detect_edges(COLOR, method="sobel")
        assert r.edge_map.shape == GRAY.shape

    def test_detect_edges_laplacian(self):
        r = detect_edges(GRAY, method="laplacian")
        assert r.edge_map.shape == GRAY.shape

    def test_adaptive_canny_density(self):
        r = adaptive_canny(GRAY)
        assert 0.0 <= r.density <= 1.0

    def test_sobel_edges_binary_values(self):
        r = sobel_edges(GRAY)
        assert set(np.unique(r.edge_map)).issubset({0, 255})

    def test_refine_edge_contour(self):
        edge_map = detect_edges(GRAY).edge_map
        refined = refine_edge_contour(edge_map, close_iter=1, dilate_iter=1)
        assert refined.shape == GRAY.shape

    def test_edge_density_value(self):
        d = edge_density(GRAY)
        assert 0.0 <= d <= 1.0

    def test_edge_orientation_hist_shape(self):
        edge_map = detect_edges(GRAY).edge_map
        hist = edge_orientation_hist(edge_map, n_bins=8)
        assert hist.shape == (8,)

    def test_edge_orientation_hist_normalized(self):
        edge_map = detect_edges(GRAY).edge_map
        hist = edge_orientation_hist(edge_map, n_bins=8, normalize=True)
        if hist.sum() > 0:
            np.testing.assert_allclose(hist.sum(), 1.0, atol=1e-6)


# ===========================================================================
# 14. edge_enhancer
# ===========================================================================

from puzzle_reconstruction.preprocessing.edge_enhancer import (
    EdgeEnhanceParams,
    unsharp_mask,
    laplacian_enhance,
    hybrid_enhance,
    gradient_scale_enhance,
    sharpness_measure,
    apply_edge_enhance,
    batch_edge_enhance,
)


class TestEdgeEnhancer:

    def test_params_default(self):
        p = EdgeEnhanceParams()
        assert p.method == "unsharp"

    def test_params_invalid_method(self):
        with pytest.raises(ValueError):
            EdgeEnhanceParams(method="mystery")

    def test_unsharp_mask_shape(self):
        out = unsharp_mask(GRAY)
        assert out.shape == GRAY.shape
        assert out.dtype == np.uint8

    def test_unsharp_mask_color(self):
        out = unsharp_mask(COLOR, strength=1.0)
        assert out.shape == COLOR.shape

    def test_laplacian_enhance(self):
        out = laplacian_enhance(GRAY, strength=0.5)
        assert out.shape == GRAY.shape

    def test_hybrid_enhance(self):
        out = hybrid_enhance(GRAY)
        assert out.shape == GRAY.shape

    def test_gradient_scale_enhance(self):
        out = gradient_scale_enhance(COLOR, strength=1.0)
        assert out.shape == COLOR.shape

    def test_sharpness_measure(self):
        val = sharpness_measure(GRAY)
        assert val >= 0.0

    def test_apply_edge_enhance_dispatch(self):
        for method in ("unsharp", "laplacian", "hybrid", "gradient_scale"):
            p = EdgeEnhanceParams(method=method, strength=1.0, blur_sigma=1.0, kernel_size=3)
            out = apply_edge_enhance(GRAY, p)
            assert out.shape == GRAY.shape

    def test_batch_edge_enhance(self):
        p = EdgeEnhanceParams(method="unsharp")
        results = batch_edge_enhance([COLOR, GRAY], p)
        assert len(results) == 2


# ===========================================================================
# 15. noise_reducer
# ===========================================================================

from puzzle_reconstruction.preprocessing.noise_reducer import (
    NoiseReductionResult,
    estimate_noise,
    gaussian_reduce,
    median_reduce,
    bilateral_reduce,
    auto_reduce,
    batch_reduce,
)


class TestNoiseReducer:

    def test_estimate_noise_non_negative(self):
        val = estimate_noise(COLOR)
        assert val >= 0.0

    def test_gaussian_reduce(self):
        r = gaussian_reduce(COLOR, ksize=5)
        assert isinstance(r, NoiseReductionResult)
        assert r.method == "gaussian"
        assert r.filtered.shape == COLOR.shape

    def test_median_reduce(self):
        r = median_reduce(GRAY, ksize=3)
        assert r.method == "median"
        assert r.filtered.shape == GRAY.shape

    def test_bilateral_reduce(self):
        r = bilateral_reduce(COLOR, d=5)
        assert r.method == "bilateral"
        assert r.filtered.shape == COLOR.shape

    def test_auto_reduce(self):
        r = auto_reduce(COLOR)
        assert r.method == "auto"
        assert r.filtered.shape == COLOR.shape

    def test_batch_reduce(self):
        results = batch_reduce([COLOR, GRAY], method="gaussian")
        assert len(results) == 2

    def test_batch_reduce_invalid(self):
        with pytest.raises(ValueError):
            batch_reduce([COLOR], method="unknown")

    def test_noise_estimates_non_negative(self):
        r = gaussian_reduce(COLOR)
        assert r.noise_estimate_before >= 0.0
        assert r.noise_estimate_after >= 0.0


# ===========================================================================
# 16. morphology_ops
# ===========================================================================

from puzzle_reconstruction.preprocessing.morphology_ops import (
    MorphParams,
    erode,
    dilate,
    open_morph,
    close_morph,
    tophat,
    blackhat,
    morphological_gradient,
    skeleton,
    remove_small_blobs,
    fill_holes,
    apply_morph,
    batch_morph,
)


class TestMorphologyOps:

    def test_morph_params_default(self):
        p = MorphParams()
        assert p.op == "open"

    def test_morph_params_invalid_op(self):
        with pytest.raises(ValueError):
            MorphParams(op="not_an_op")

    def test_erode_shape(self):
        out = erode(GRAY, ksize=3)
        assert out.shape == GRAY.shape

    def test_dilate_shape(self):
        out = dilate(GRAY, ksize=3)
        assert out.shape == GRAY.shape

    def test_open_morph(self):
        out = open_morph(BINARY, ksize=3)
        assert out.shape == BINARY.shape

    def test_close_morph(self):
        out = close_morph(BINARY, ksize=3)
        assert out.shape == BINARY.shape

    def test_tophat(self):
        out = tophat(GRAY, ksize=5)
        assert out.shape == GRAY.shape

    def test_blackhat(self):
        out = blackhat(GRAY, ksize=5)
        assert out.shape == GRAY.shape

    def test_morphological_gradient(self):
        out = morphological_gradient(GRAY, ksize=3)
        assert out.shape == GRAY.shape

    def test_skeleton(self):
        out = skeleton(BINARY)
        assert out.shape == BINARY.shape
        assert out.dtype == np.uint8

    def test_skeleton_raises_on_color(self):
        with pytest.raises(ValueError):
            skeleton(COLOR)

    def test_remove_small_blobs(self):
        out = remove_small_blobs(BINARY, min_area=10)
        assert out.shape == BINARY.shape

    def test_fill_holes(self):
        out = fill_holes(BINARY)
        assert out.shape == BINARY.shape

    def test_apply_morph_dispatch(self):
        for op in ("erode", "dilate", "open", "close", "tophat", "blackhat", "gradient"):
            p = MorphParams(op=op, ksize=3)
            out = apply_morph(GRAY, p)
            assert out.shape == GRAY.shape

    def test_batch_morph(self):
        results = batch_morph([BINARY, GRAY])
        assert len(results) == 2
