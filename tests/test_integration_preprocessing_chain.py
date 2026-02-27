"""
Integration tests for the preprocessing chain.

Covers:
  - PreprocessingChain.apply() on synthetic images
  - Pipeline.preprocess() producing List[Fragment]
  - Fragment attributes: mask, contour, edges
  - Parallel (n_workers=2) vs sequential (n_workers=1) preprocessing
  - Graceful degradation on degenerate images (1×1, all-black, all-white)
  - Pipeline timer stage data (profile)
  - segment → contour → orientation → color_norm → patch_sampler pipeline steps
  - Intermediate artefact dtype/shape correctness
  - edge_sharpener: valid SharpenerResult
  - gradient_analyzer: non-empty gradient profiles
"""
from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np
import pytest

# ─── Preprocessing chain ──────────────────────────────────────────────────────
from puzzle_reconstruction.preprocessing.chain import (
    PreprocessingChain,
    list_filters,
)

# ─── Pipeline + Config ────────────────────────────────────────────────────────
from puzzle_reconstruction.pipeline import Pipeline
from puzzle_reconstruction.config import Config

# ─── Models ───────────────────────────────────────────────────────────────────
from puzzle_reconstruction.models import Fragment

# ─── Preprocessing steps ──────────────────────────────────────────────────────
from puzzle_reconstruction.preprocessing.segmentation import segment_fragment
from puzzle_reconstruction.preprocessing.contour import extract_contour
from puzzle_reconstruction.preprocessing.orientation import (
    estimate_orientation,
    rotate_to_upright,
)
from puzzle_reconstruction.preprocessing.color_norm import normalize_color
from puzzle_reconstruction.preprocessing.patch_sampler import sample_patches
from puzzle_reconstruction.preprocessing.edge_sharpener import (
    sharpen_edges,
    sharpen_image,
    SharpenerConfig,
    SharpenerResult,
)
from puzzle_reconstruction.preprocessing.gradient_analyzer import (
    extract_gradient_profile,
    compute_gradient_map,
    GradientProfile,
)


# ─── Synthetic image helpers ──────────────────────────────────────────────────

def _make_image(h: int = 100, w: int = 100, seed: int = 0) -> np.ndarray:
    """Create a synthetic uint8 BGR image with a bright filled rectangle."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(h, w, 3) * 60 + 30).astype(np.uint8)
    # Bright rectangle occupies 60 % of area → easy to segment
    margin_y = h // 10
    margin_x = w // 10
    img[margin_y: h - margin_y, margin_x: w - margin_x] = 210
    return img


def _make_images(n: int, h: int = 100, w: int = 100) -> List[np.ndarray]:
    return [_make_image(h, w, seed=i) for i in range(n)]


def _make_pipeline(n_workers: int = 1) -> Pipeline:
    cfg = Config.default()
    return Pipeline(cfg, n_workers=n_workers)


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreprocessingChainIntegration
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPreprocessingChainIntegration:
    """Tests for the full pipeline preprocessing chain producing List[Fragment]."""

    # ── chain.run (pipeline.preprocess) produces List[Fragment] ───────────────

    def test_preprocess_2_images_no_exception(self):
        pipe = _make_pipeline()
        images = _make_images(2)
        frags = pipe.preprocess(images)
        assert isinstance(frags, list)

    def test_preprocess_2_images_returns_fragments(self):
        pipe = _make_pipeline()
        images = _make_images(2)
        frags = pipe.preprocess(images)
        assert len(frags) >= 1
        for f in frags:
            assert isinstance(f, Fragment)

    def test_preprocess_4_images_no_exception(self):
        pipe = _make_pipeline()
        images = _make_images(4)
        frags = pipe.preprocess(images)
        assert isinstance(frags, list)

    def test_preprocess_4_images_returns_fragments(self):
        pipe = _make_pipeline()
        images = _make_images(4)
        frags = pipe.preprocess(images)
        assert len(frags) >= 2
        for f in frags:
            assert isinstance(f, Fragment)

    def test_preprocess_9_images_no_exception(self):
        pipe = _make_pipeline()
        images = _make_images(9)
        frags = pipe.preprocess(images)
        assert isinstance(frags, list)

    def test_preprocess_9_images_returns_fragments(self):
        pipe = _make_pipeline()
        images = _make_images(9)
        frags = pipe.preprocess(images)
        assert len(frags) >= 4
        for f in frags:
            assert isinstance(f, Fragment)

    # ── Fragment attributes ────────────────────────────────────────────────────

    def test_fragment_has_mask(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert f.mask is not None

    def test_fragment_mask_is_ndarray(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert isinstance(f.mask, np.ndarray)

    def test_fragment_mask_2d(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert f.mask.ndim == 2

    def test_fragment_has_contour(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert f.contour is not None

    def test_fragment_contour_is_ndarray(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert isinstance(f.contour, np.ndarray)

    def test_fragment_contour_2d_shape(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert f.contour.ndim == 2
            assert f.contour.shape[1] == 2

    def test_fragment_contour_min_4_points(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert len(f.contour) >= 4

    def test_fragment_has_edges(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert f.edges is not None

    def test_fragment_edges_list(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert isinstance(f.edges, list)

    def test_fragment_edges_not_empty(self):
        pipe = _make_pipeline()
        frags = pipe.preprocess(_make_images(2))
        for f in frags:
            assert len(f.edges) > 0

    # ── Parallel (n_workers=2) vs sequential (n_workers=1) ────────────────────

    def test_parallel_same_count_as_sequential(self):
        images = _make_images(4, h=100, w=100)
        pipe_seq = _make_pipeline(n_workers=1)
        pipe_par = _make_pipeline(n_workers=2)
        frags_seq = pipe_seq.preprocess(images)
        frags_par = pipe_par.preprocess(images)
        assert len(frags_par) == len(frags_seq)

    def test_parallel_all_fragments_valid(self):
        images = _make_images(4)
        pipe = _make_pipeline(n_workers=2)
        frags = pipe.preprocess(images)
        for f in frags:
            assert isinstance(f, Fragment)
            assert f.mask is not None
            assert f.contour is not None
            assert f.edges is not None

    def test_parallel_fragment_ids_unique(self):
        images = _make_images(4)
        pipe = _make_pipeline(n_workers=2)
        frags = pipe.preprocess(images)
        ids = [f.fragment_id for f in frags]
        assert len(ids) == len(set(ids))

    # ── Graceful degradation on bad images ────────────────────────────────────

    def test_graceful_1x1_image(self):
        """1×1 image should not raise; fragment may or may not be produced."""
        pipe = _make_pipeline()
        tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = pipe.preprocess([tiny])
        assert isinstance(result, list)

    def test_graceful_all_black_image(self):
        """All-black 100×100 image should not raise."""
        pipe = _make_pipeline()
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipe.preprocess([black])
        assert isinstance(result, list)

    def test_graceful_all_white_image(self):
        """All-white 100×100 image should not raise."""
        pipe = _make_pipeline()
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = pipe.preprocess([white])
        assert isinstance(result, list)

    def test_graceful_mixed_good_and_bad_images(self):
        """Mix of good and bad images: at least the good ones survive."""
        pipe = _make_pipeline()
        good = [_make_image(100, 100, seed=i) for i in range(2)]
        bad = [
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.full((100, 100, 3), 0, dtype=np.uint8),
        ]
        result = pipe.preprocess(good + bad)
        assert isinstance(result, list)
        # Good images should produce at least 1 fragment
        assert len(result) >= 1

    # ── Profile timing ─────────────────────────────────────────────────────────

    def test_timer_has_preprocessing_stage(self):
        pipe = _make_pipeline()
        pipe.preprocess(_make_images(2))
        # PipelineTimer._stages dict is populated after a context-managed measure()
        # called internally by Pipeline.run(); preprocess() itself does not wrap
        # in a timer — but Pipeline.run() does. Use run() for profile testing.
        cfg = Config.default()
        pipe2 = Pipeline(cfg, n_workers=1)
        images = _make_images(2)
        result = pipe2.run(images)
        assert hasattr(result, "timer")
        assert hasattr(result.timer, "_stages")

    def test_timer_stages_not_empty(self):
        cfg = Config.default()
        pipe = Pipeline(cfg, n_workers=1)
        images = _make_images(2)
        result = pipe.run(images)
        assert len(result.timer._stages) > 0

    def test_timer_stages_contain_preprocessing(self):
        cfg = Config.default()
        pipe = Pipeline(cfg, n_workers=1)
        result = pipe.run(_make_images(2))
        # Russian stage name used internally
        stage_keys = result.timer._stages.keys()
        assert any("препроц" in k for k in stage_keys)

    def test_timer_stage_values_positive(self):
        cfg = Config.default()
        pipe = Pipeline(cfg, n_workers=1)
        result = pipe.run(_make_images(2))
        for stage, elapsed in result.timer._stages.items():
            assert elapsed >= 0.0, f"Stage '{stage}' has negative time: {elapsed}"

    def test_pipeline_result_summary_string(self):
        cfg = Config.default()
        pipe = Pipeline(cfg, n_workers=1)
        result = pipe.run(_make_images(2))
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreprocessingStepsIntegration
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestPreprocessingStepsIntegration:
    """Tests for individual preprocessing step functions."""

    # ── segment ───────────────────────────────────────────────────────────────

    def test_segment_otsu_returns_uint8_mask(self):
        img = _make_image(100, 100, seed=1)
        mask = segment_fragment(img, method="otsu")
        assert mask.dtype == np.uint8

    def test_segment_otsu_binary_values(self):
        img = _make_image(100, 100, seed=1)
        mask = segment_fragment(img, method="otsu")
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})

    def test_segment_otsu_shape_matches_image(self):
        img = _make_image(100, 100, seed=1)
        mask = segment_fragment(img, method="otsu")
        assert mask.shape == img.shape[:2]

    def test_segment_adaptive_returns_mask(self):
        img = _make_image(100, 100, seed=2)
        mask = segment_fragment(img, method="adaptive")
        assert mask.shape == img.shape[:2]
        assert np.any(mask > 0)

    def test_segment_200x200_mask_shape(self):
        img = _make_image(200, 200, seed=3)
        mask = segment_fragment(img)
        assert mask.shape == (200, 200)

    # ── contour ───────────────────────────────────────────────────────────────

    def test_contour_from_segment_is_2d(self):
        img = _make_image(100, 100, seed=4)
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert contour.ndim == 2
        assert contour.shape[1] == 2

    def test_contour_has_min_4_points(self):
        img = _make_image(100, 100, seed=4)
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert len(contour) >= 4

    def test_contour_values_within_image_bounds(self):
        img = _make_image(100, 100, seed=5)
        h, w = img.shape[:2]
        mask = segment_fragment(img)
        contour = extract_contour(mask)
        assert np.all(contour[:, 0] >= 0) and np.all(contour[:, 0] <= w)
        assert np.all(contour[:, 1] >= 0) and np.all(contour[:, 1] <= h)

    # ── orientation ───────────────────────────────────────────────────────────

    def test_orientation_returns_float(self):
        img = _make_image(100, 100, seed=6)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_rotate_to_upright_preserves_dtype(self):
        img = _make_image(100, 100, seed=7)
        angle = estimate_orientation(img)
        rotated = rotate_to_upright(img, angle)
        assert rotated.dtype == np.uint8

    def test_rotate_to_upright_preserves_shape_3channel(self):
        img = _make_image(100, 100, seed=7)
        angle = estimate_orientation(img)
        rotated = rotate_to_upright(img, angle)
        assert rotated.ndim == 3
        assert rotated.shape[2] == 3

    def test_rotate_zero_angle_returns_same_shape(self):
        img = _make_image(100, 100, seed=8)
        rotated = rotate_to_upright(img, 0.0)
        assert rotated.shape == img.shape

    # ── color_norm ────────────────────────────────────────────────────────────

    def test_color_normalize_returns_uint8(self):
        img = _make_image(100, 100, seed=9)
        norm = normalize_color(img)
        assert norm.dtype == np.uint8

    def test_color_normalize_preserves_shape(self):
        img = _make_image(100, 100, seed=9)
        norm = normalize_color(img)
        assert norm.shape == img.shape

    def test_color_normalize_values_in_range(self):
        img = _make_image(100, 100, seed=10)
        norm = normalize_color(img)
        assert int(norm.min()) >= 0
        assert int(norm.max()) <= 255

    # ── patch_sampler ─────────────────────────────────────────────────────────

    def test_patch_sampler_returns_result(self):
        img = _make_image(100, 100, seed=11)
        result = sample_patches(img)
        assert result is not None

    def test_patch_sampler_n_patches_positive(self):
        img = _make_image(100, 100, seed=11)
        result = sample_patches(img)
        assert result.n_patches >= 1

    def test_patch_sampler_image_shape_recorded(self):
        img = _make_image(100, 100, seed=11)
        result = sample_patches(img)
        assert result.image_shape[:2] == (100, 100)

    # ── edge_sharpener ────────────────────────────────────────────────────────

    def test_sharpen_image_returns_sharpener_result(self):
        img = _make_image(100, 100, seed=12)
        result = sharpen_image(img)
        assert isinstance(result, SharpenerResult)

    def test_sharpen_image_output_uint8(self):
        img = _make_image(100, 100, seed=12)
        result = sharpen_image(img)
        assert result.image.dtype == np.uint8

    def test_sharpen_image_preserves_shape(self):
        img = _make_image(100, 100, seed=12)
        result = sharpen_image(img)
        assert result.image.shape == img.shape

    def test_sharpen_edges_not_empty(self):
        img = _make_image(100, 100, seed=13)
        sharpened = sharpen_edges(img)
        assert sharpened is not None
        assert sharpened.size > 0

    def test_sharpen_edges_dtype_uint8(self):
        img = _make_image(100, 100, seed=13)
        sharpened = sharpen_edges(img)
        assert sharpened.dtype == np.uint8

    def test_sharpen_edges_same_shape(self):
        img = _make_image(100, 100, seed=13)
        sharpened = sharpen_edges(img)
        assert sharpened.shape == img.shape

    def test_sharpen_with_laplacian_method(self):
        img = _make_image(100, 100, seed=14)
        cfg = SharpenerConfig(method="laplacian", strength=0.5, ksize=3)
        result = sharpen_image(img, cfg)
        assert result.image.dtype == np.uint8
        assert result.method == "laplacian"

    def test_sharpen_with_high_pass_method(self):
        img = _make_image(100, 100, seed=14)
        cfg = SharpenerConfig(method="high_pass", strength=1.0, sigma=2.0)
        result = sharpen_image(img, cfg)
        assert result.image.dtype == np.uint8
        assert result.method == "high_pass"

    # ── gradient_analyzer ─────────────────────────────────────────────────────

    def test_gradient_analyzer_returns_profile(self):
        img = _make_image(100, 100, seed=15)
        profile = extract_gradient_profile(img, fragment_id=0)
        assert isinstance(profile, GradientProfile)

    def test_gradient_profile_mean_magnitude_nonneg(self):
        img = _make_image(100, 100, seed=15)
        profile = extract_gradient_profile(img, fragment_id=0)
        assert profile.mean_magnitude >= 0.0

    def test_gradient_profile_orientation_hist_not_empty(self):
        img = _make_image(100, 100, seed=16)
        profile = extract_gradient_profile(img, fragment_id=0)
        assert profile.orientation_hist is not None
        assert len(profile.orientation_hist) > 0

    def test_gradient_map_magnitude_not_all_zero(self):
        img = _make_image(100, 100, seed=17)
        gmap = compute_gradient_map(img)
        assert gmap.magnitude.max() > 0.0

    def test_gradient_map_shape_matches_image(self):
        img = _make_image(100, 100, seed=17)
        gmap = compute_gradient_map(img)
        assert gmap.magnitude.shape == img.shape[:2]
        assert gmap.angle.shape == img.shape[:2]

    def test_gradient_profile_energy_positive_on_nonflat_image(self):
        img = _make_image(100, 100, seed=18)
        profile = extract_gradient_profile(img, fragment_id=0)
        assert profile.energy > 0.0

    def test_gradient_profile_dominant_angle_in_range(self):
        img = _make_image(100, 100, seed=19)
        profile = extract_gradient_profile(img, fragment_id=0)
        assert 0.0 <= profile.dominant_angle < 180.0

    # ── PreprocessingChain.apply ───────────────────────────────────────────────

    def test_chain_apply_empty_filters_passthrough(self):
        img = _make_image(100, 100, seed=20)
        chain = PreprocessingChain(filters=[])
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_chain_apply_denoise_returns_image(self):
        img = _make_image(100, 100, seed=21)
        chain = PreprocessingChain(filters=["denoise"])
        result = chain.apply(img)
        assert result is not None
        assert result.dtype == np.uint8

    def test_chain_apply_contrast_returns_image(self):
        img = _make_image(100, 100, seed=22)
        chain = PreprocessingChain(filters=["contrast"])
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_chain_apply_multiple_filters_no_exception(self):
        img = _make_image(100, 100, seed=23)
        chain = PreprocessingChain(filters=["denoise", "contrast", "sharpen"])
        result = chain.apply(img)
        assert result is not None

    def test_chain_apply_auto_enhance_passthrough(self):
        img = _make_image(100, 100, seed=24)
        chain = PreprocessingChain(filters=[], auto_enhance=True)
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_chain_is_empty_when_no_filters(self):
        chain = PreprocessingChain(filters=[])
        assert chain.is_empty() is True

    def test_chain_not_empty_with_filters(self):
        chain = PreprocessingChain(filters=["denoise"])
        assert chain.is_empty() is False

    def test_chain_apply_unknown_filter_skips_gracefully(self):
        img = _make_image(100, 100, seed=25)
        chain = PreprocessingChain(filters=["nonexistent_filter_xyz"])
        result = chain.apply(img)
        # Unknown filter should be silently skipped, returning original image
        assert result is not None
        assert result.shape == img.shape

    def test_list_filters_returns_nonempty_list(self):
        filters = list_filters()
        assert isinstance(filters, list)
        assert len(filters) > 0

    def test_list_filters_contains_quality_assessor(self):
        filters = list_filters()
        assert "quality_assessor" in filters

    def test_list_filters_contains_denoise(self):
        filters = list_filters()
        assert "denoise" in filters

    # ── Intermediate artefact dtype/shape correctness ─────────────────────────

    def test_segment_contour_orientation_color_norm_pipeline(self):
        """Full manual pipeline: segment → contour → orientation → color_norm."""
        img = _make_image(200, 200, seed=30)
        # Step 1: color_norm
        norm = normalize_color(img)
        assert norm.dtype == np.uint8
        assert norm.shape == (200, 200, 3)
        # Step 2: segment
        mask = segment_fragment(norm)
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        # Step 3: orientation + rotate
        angle = estimate_orientation(norm, mask)
        rotated = rotate_to_upright(norm, angle)
        assert rotated.dtype == np.uint8
        assert rotated.ndim == 3
        # Step 4: contour
        contour = extract_contour(mask)
        assert contour.ndim == 2
        assert contour.shape[1] == 2
        assert len(contour) >= 4

    def test_patch_sampler_after_color_norm(self):
        """patch_sampler works on color-normalized image."""
        img = _make_image(100, 100, seed=31)
        norm = normalize_color(img)
        result = sample_patches(norm)
        assert result.n_patches >= 1
        assert result.image_shape[:2] == (100, 100)

    def test_gradient_then_sharpen_consistent_shapes(self):
        """gradient_analyze then sharpen_image produce consistent shapes."""
        img = _make_image(100, 100, seed=32)
        # Gradient
        profile = extract_gradient_profile(img, fragment_id=0)
        assert profile.orientation_hist is not None
        # Sharpen
        sharpened = sharpen_image(img)
        assert sharpened.image.shape == img.shape
