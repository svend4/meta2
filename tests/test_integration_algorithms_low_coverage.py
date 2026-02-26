"""Comprehensive integration tests for low-coverage algorithm modules.

Covers:
    1. puzzle_reconstruction.algorithms.fragment_quality
    2. puzzle_reconstruction.algorithms.patch_aligner
    3. puzzle_reconstruction.algorithms.patch_matcher
    4. puzzle_reconstruction.algorithms.region_scorer
    5. puzzle_reconstruction.algorithms.seam_evaluator
    6. puzzle_reconstruction.assembly.overlap_resolver
    7. puzzle_reconstruction.assembly.position_estimator
    8. puzzle_reconstruction.assembly.rl_agent
    9. puzzle_reconstruction.scoring.evidence_aggregator
   10. puzzle_reconstruction.preprocessing.chain
"""
from __future__ import annotations

import pytest
import numpy as np

# ─── Helpers ────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _gray(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _mask(h=64, w=64, fill=True):
    m = np.zeros((h, w), dtype=np.uint8)
    if fill:
        m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    return m


# ============================================================
# 1. fragment_quality
# ============================================================

class TestFragmentQuality:
    def _import(self):
        from puzzle_reconstruction.algorithms.fragment_quality import (
            QualityConfig, QualityReport,
            measure_blur, measure_contrast, measure_mask_coverage,
            measure_edge_sharpness, assess_fragment, rank_fragments, batch_assess,
        )
        return (QualityConfig, QualityReport,
                measure_blur, measure_contrast, measure_mask_coverage,
                measure_edge_sharpness, assess_fragment, rank_fragments, batch_assess)

    def test_quality_config_defaults(self):
        (QualityConfig, *_) = self._import()
        cfg = QualityConfig()
        assert cfg.w_blur == 1.0
        assert cfg.w_contrast == 1.0
        assert cfg.total_weight == 4.0

    def test_quality_config_invalid_weight(self):
        (QualityConfig, *_) = self._import()
        with pytest.raises(ValueError):
            QualityConfig(w_blur=-1.0)

    def test_quality_config_all_zero_weights(self):
        (QualityConfig, *_) = self._import()
        with pytest.raises(ValueError):
            QualityConfig(w_blur=0, w_contrast=0, w_coverage=0, w_sharpness=0)

    def test_quality_config_invalid_blur_ref(self):
        (QualityConfig, *_) = self._import()
        with pytest.raises(ValueError):
            QualityConfig(blur_ref=0.0)

    def test_quality_config_invalid_grad_ref(self):
        (QualityConfig, *_) = self._import()
        with pytest.raises(ValueError):
            QualityConfig(grad_ref=-1.0)

    def test_measure_blur_gray(self):
        (_, _, measure_blur, *_) = self._import()
        img = _gray(64, 64, seed=10)
        score = measure_blur(img)
        assert 0.0 <= score <= 1.0

    def test_measure_blur_bgr(self):
        (_, _, measure_blur, *_) = self._import()
        img = _bgr(64, 64, seed=11)
        score = measure_blur(img)
        assert 0.0 <= score <= 1.0

    def test_measure_blur_invalid_ref(self):
        (_, _, measure_blur, *_) = self._import()
        with pytest.raises(ValueError):
            measure_blur(_gray(), ref=0.0)

    def test_measure_blur_sharp_vs_blurry(self):
        (_, _, measure_blur, *_) = self._import()
        rng = np.random.default_rng(99)
        sharp = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        blurry = np.full((64, 64), 128, dtype=np.uint8)
        assert measure_blur(sharp) >= measure_blur(blurry)

    def test_measure_contrast_gray(self):
        (_, _, _, measure_contrast, *_) = self._import()
        img = _gray(64, 64, seed=20)
        score = measure_contrast(img)
        assert 0.0 <= score <= 1.0

    def test_measure_contrast_with_mask(self):
        (_, _, _, measure_contrast, *_) = self._import()
        img = _gray(64, 64, seed=21)
        mask = _mask(64, 64)
        score = measure_contrast(img, mask=mask)
        assert 0.0 <= score <= 1.0

    def test_measure_contrast_uniform(self):
        (_, _, _, measure_contrast, *_) = self._import()
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert measure_contrast(img) == 0.0

    def test_measure_contrast_empty_mask(self):
        (_, _, _, measure_contrast, *_) = self._import()
        img = _gray(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert measure_contrast(img, mask=mask) == 0.0

    def test_measure_mask_coverage_full(self):
        (_, _, _, _, measure_mask_coverage, *_) = self._import()
        mask = np.full((64, 64), 255, dtype=np.uint8)
        assert measure_mask_coverage(mask) == pytest.approx(1.0)

    def test_measure_mask_coverage_empty(self):
        (_, _, _, _, measure_mask_coverage, *_) = self._import()
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert measure_mask_coverage(mask) == pytest.approx(0.0)

    def test_measure_mask_coverage_half(self):
        (_, _, _, _, measure_mask_coverage, *_) = self._import()
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :] = 255
        assert measure_mask_coverage(mask) == pytest.approx(0.5)

    def test_measure_mask_coverage_invalid_ndim(self):
        (_, _, _, _, measure_mask_coverage, *_) = self._import()
        with pytest.raises(ValueError):
            measure_mask_coverage(np.zeros((4, 4, 2), dtype=np.uint8))

    def test_measure_edge_sharpness_basic(self):
        (_, _, _, _, _, measure_edge_sharpness, *_) = self._import()
        img = _bgr(64, 64, seed=30)
        score = measure_edge_sharpness(img)
        assert 0.0 <= score <= 1.0

    def test_measure_edge_sharpness_with_mask(self):
        (_, _, _, _, _, measure_edge_sharpness, *_) = self._import()
        img = _gray(64, 64, seed=31)
        mask = _mask(64, 64)
        score = measure_edge_sharpness(img, mask=mask)
        assert 0.0 <= score <= 1.0

    def test_measure_edge_sharpness_empty_mask(self):
        (_, _, _, _, _, measure_edge_sharpness, *_) = self._import()
        img = _gray(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        assert measure_edge_sharpness(img, mask=mask) == 0.0

    def test_measure_edge_sharpness_invalid_ref(self):
        (_, _, _, _, _, measure_edge_sharpness, *_) = self._import()
        with pytest.raises(ValueError):
            measure_edge_sharpness(_gray(), ref=0.0)

    def test_assess_fragment_returns_report(self):
        (QualityConfig, QualityReport, _, _, _, _, assess_fragment, *_) = self._import()
        img = _bgr(64, 64, seed=40)
        report = assess_fragment(img)
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.score <= 1.0

    def test_assess_fragment_with_mask(self):
        (_, _, _, _, _, _, assess_fragment, *_) = self._import()
        img = _bgr(64, 64, seed=41)
        mask = _mask(64, 64)
        report = assess_fragment(img, mask=mask, fragment_id=5)
        assert report.fragment_id == 5
        assert 0.0 <= report.coverage <= 1.0

    def test_assess_fragment_custom_config(self):
        (QualityConfig, _, _, _, _, _, assess_fragment, *_) = self._import()
        cfg = QualityConfig(w_blur=2.0, w_contrast=0.5)
        img = _bgr(64, 64, seed=42)
        report = assess_fragment(img, cfg=cfg)
        assert 0.0 <= report.score <= 1.0

    def test_quality_report_is_usable(self):
        (_, _, _, _, _, _, assess_fragment, *_) = self._import()
        rng = np.random.default_rng(50)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        report = assess_fragment(img, fragment_id=0)
        assert isinstance(report.is_usable, bool)

    def test_quality_report_summary(self):
        (_, _, _, _, _, _, assess_fragment, *_) = self._import()
        report = assess_fragment(_bgr(seed=51))
        s = report.summary()
        assert "QualityReport" in s

    def test_rank_fragments_sorted(self):
        (_, _, _, _, _, _, assess_fragment, rank_fragments, _) = self._import()
        imgs = [_bgr(seed=i) for i in range(5)]
        reports = [assess_fragment(img, fragment_id=i) for i, img in enumerate(imgs)]
        ranked = rank_fragments(reports)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_fragments_empty(self):
        (_, _, _, _, _, _, _, rank_fragments, _) = self._import()
        assert rank_fragments([]) == []

    def test_rank_fragments_with_indices(self):
        (_, _, _, _, _, _, assess_fragment, rank_fragments, _) = self._import()
        imgs = [_bgr(seed=i) for i in range(3)]
        reports = [assess_fragment(img, fragment_id=i) for i, img in enumerate(imgs)]
        indices = [10, 20, 30]
        ranked = rank_fragments(reports, indices=indices)
        fids = [fid for fid, _ in ranked]
        assert set(fids) == {10, 20, 30}

    def test_batch_assess_returns_list(self):
        (_, _, _, _, _, _, _, _, batch_assess) = self._import()
        imgs = [_bgr(seed=i) for i in range(4)]
        reports = batch_assess(imgs)
        assert len(reports) == 4
        for i, r in enumerate(reports):
            assert r.fragment_id == i

    def test_batch_assess_with_masks(self):
        (_, _, _, _, _, _, _, _, batch_assess) = self._import()
        imgs = [_bgr(seed=i) for i in range(3)]
        masks = [_mask() for _ in range(3)]
        reports = batch_assess(imgs, masks=masks)
        assert len(reports) == 3


# ============================================================
# 2. patch_aligner
# ============================================================

class TestPatchAligner:
    def _import(self):
        from puzzle_reconstruction.algorithms.patch_aligner import (
            AlignConfig, AlignResult,
            phase_correlate, ncc_score, align_patches,
            refine_alignment, batch_align,
        )
        return (AlignConfig, AlignResult,
                phase_correlate, ncc_score, align_patches,
                refine_alignment, batch_align)

    def test_align_config_defaults(self):
        (AlignConfig, *_) = self._import()
        cfg = AlignConfig()
        assert cfg.method == "combined"
        assert cfg.max_shift == 20.0

    def test_align_config_invalid_method(self):
        (AlignConfig, *_) = self._import()
        with pytest.raises(ValueError):
            AlignConfig(method="invalid")

    def test_align_config_invalid_max_shift(self):
        (AlignConfig, *_) = self._import()
        with pytest.raises(ValueError):
            AlignConfig(max_shift=0.0)

    def test_align_config_invalid_upsample(self):
        (AlignConfig, *_) = self._import()
        with pytest.raises(ValueError):
            AlignConfig(upsample_factor=0)

    def test_align_config_invalid_ncc_threshold(self):
        (AlignConfig, *_) = self._import()
        with pytest.raises(ValueError):
            AlignConfig(ncc_threshold=1.5)

    def test_align_config_invalid_refine_radius(self):
        (AlignConfig, *_) = self._import()
        with pytest.raises(ValueError):
            AlignConfig(refine_radius=-1)

    def test_phase_correlate_same_patch(self):
        (_, _, phase_correlate, *_) = self._import()
        patch = _gray(32, 32, seed=60).astype(np.float32)
        dy, dx = phase_correlate(patch, patch)
        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_phase_correlate_shape_mismatch(self):
        (_, _, phase_correlate, *_) = self._import()
        with pytest.raises(ValueError):
            phase_correlate(_gray(32, 32), _gray(16, 16))

    def test_phase_correlate_upsample(self):
        (_, _, phase_correlate, *_) = self._import()
        patch = _gray(32, 32, seed=61).astype(np.float32)
        dy, dx = phase_correlate(patch, patch, upsample_factor=2)
        assert isinstance(dy, float)

    def test_ncc_score_identical(self):
        (_, _, _, ncc_score, *_) = self._import()
        patch = _gray(16, 16, seed=70).astype(np.float32)
        score = ncc_score(patch, patch)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_ncc_score_uniform(self):
        (_, _, _, ncc_score, *_) = self._import()
        p = np.full((16, 16), 128.0, dtype=np.float32)
        score = ncc_score(p, p)
        assert score == 0.0

    def test_ncc_score_range(self):
        (_, _, _, ncc_score, *_) = self._import()
        a = _gray(16, 16, seed=71).astype(np.float32)
        b = _gray(16, 16, seed=72).astype(np.float32)
        score = ncc_score(a, b)
        assert -1.0 <= score <= 1.0

    def test_align_patches_phase(self):
        (AlignConfig, AlignResult, _, _, align_patches, *_) = self._import()
        cfg = AlignConfig(method="phase")
        a = _gray(32, 32, seed=80).astype(np.float32)
        b = _gray(32, 32, seed=81).astype(np.float32)
        result = align_patches(a, b, cfg=cfg)
        assert isinstance(result, AlignResult)
        assert -1.0 <= result.ncc <= 1.0

    def test_align_patches_ncc(self):
        (AlignConfig, AlignResult, _, _, align_patches, *_) = self._import()
        cfg = AlignConfig(method="ncc")
        a = _gray(32, 32, seed=82)
        result = align_patches(a, a, cfg=cfg)
        assert isinstance(result, AlignResult)

    def test_align_patches_combined(self):
        (AlignConfig, AlignResult, _, _, align_patches, *_) = self._import()
        cfg = AlignConfig(method="combined")
        a = _gray(32, 32, seed=83)
        b = _gray(32, 32, seed=84)
        result = align_patches(a, b, cfg=cfg)
        assert isinstance(result, AlignResult)

    def test_align_result_shift_magnitude(self):
        (AlignConfig, _, _, _, align_patches, *_) = self._import()
        a = _gray(32, 32, seed=85)
        result = align_patches(a, a)
        assert result.shift_magnitude >= 0.0

    def test_align_result_success(self):
        (AlignConfig, _, _, _, align_patches, *_) = self._import()
        a = _gray(32, 32, seed=86)
        result = align_patches(a, a)
        assert isinstance(result.success, bool)

    def test_refine_alignment(self):
        (AlignConfig, _, _, _, align_patches, refine_alignment, _) = self._import()
        a = _gray(32, 32, seed=90)
        b = _gray(32, 32, seed=91)
        initial = align_patches(a, b)
        # refine_alignment takes (patch_a, patch_b, initial_shift: Tuple[float, float])
        refined = refine_alignment(a, b, initial.shift)
        assert refined is not None

    def test_batch_align(self):
        (AlignConfig, AlignResult, _, _, _, _, batch_align) = self._import()
        pairs = [(_gray(seed=i), _gray(seed=i + 10)) for i in range(3)]
        results = batch_align(pairs)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, AlignResult)

    def test_batch_align_empty(self):
        (_, _, _, _, _, _, batch_align) = self._import()
        results = batch_align([])
        assert results == []

    def test_align_config_phase_method(self):
        (AlignConfig, *_) = self._import()
        cfg = AlignConfig(method="phase")
        assert cfg.method == "phase"

    def test_align_config_ncc_method(self):
        (AlignConfig, *_) = self._import()
        cfg = AlignConfig(method="ncc")
        assert cfg.method == "ncc"


# ============================================================
# 3. patch_matcher
# ============================================================

class TestPatchMatcher:
    def _import(self):
        from puzzle_reconstruction.algorithms.patch_matcher import (
            PatchConfig, PatchMatch,
            extract_patch, ncc_score, ssd_score, sad_score,
            match_patch_in_image, find_matches, top_matches, batch_patch_match,
        )
        return (PatchConfig, PatchMatch,
                extract_patch, ncc_score, ssd_score, sad_score,
                match_patch_in_image, find_matches, top_matches, batch_patch_match)

    def test_patch_config_defaults(self):
        (PatchConfig, *_) = self._import()
        cfg = PatchConfig()
        assert cfg.patch_size == 17
        assert cfg.method == "ncc"

    def test_patch_config_even_size(self):
        (PatchConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PatchConfig(patch_size=16)

    def test_patch_config_small_size(self):
        (PatchConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PatchConfig(patch_size=1)

    def test_patch_config_invalid_method(self):
        (PatchConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PatchConfig(method="invalid")

    def test_patch_config_invalid_stride(self):
        (PatchConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PatchConfig(stride=0)

    def test_patch_config_invalid_max_matches(self):
        (PatchConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PatchConfig(max_matches=0)

    def test_extract_patch_basic(self):
        (_, _, extract_patch, *_) = self._import()
        img = _gray(64, 64, seed=100)
        patch = extract_patch(img, 0, 0, 17)
        assert patch.shape == (17, 17)

    def test_extract_patch_bounds(self):
        (_, _, extract_patch, *_) = self._import()
        img = _gray(64, 64)
        with pytest.raises(ValueError):
            extract_patch(img, 60, 60, 17)  # out of bounds

    def test_extract_patch_color(self):
        (_, _, extract_patch, *_) = self._import()
        img = _bgr(64, 64, seed=101)
        patch = extract_patch(img, 0, 0, 9)
        assert patch.shape == (9, 9)

    def test_ncc_score_identical(self):
        (_, _, _, ncc_score, *_) = self._import()
        rng = np.random.default_rng(110)
        p = rng.random((16, 16)).astype(np.float32)
        assert ncc_score(p, p) == pytest.approx(1.0, abs=1e-5)

    def test_ncc_score_shape_mismatch(self):
        (_, _, _, ncc_score, *_) = self._import()
        with pytest.raises(ValueError):
            ncc_score(np.zeros((8, 8)), np.zeros((16, 16)))

    def test_ssd_score_identical(self):
        (_, _, _, _, ssd_score, *_) = self._import()
        p = _gray(8, 8, seed=120).astype(np.float32)
        assert ssd_score(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_sad_score_identical(self):
        (_, _, _, _, _, sad_score, *_) = self._import()
        p = _gray(8, 8, seed=130).astype(np.float32)
        assert sad_score(p, p) == pytest.approx(0.0, abs=1e-5)

    def test_sad_score_positive(self):
        (_, _, _, _, _, sad_score, *_) = self._import()
        a = _gray(8, 8, seed=131).astype(np.float32)
        b = _gray(8, 8, seed=132).astype(np.float32)
        assert sad_score(a, b) >= 0.0

    def test_match_patch_in_image_ncc(self):
        (_, _, _, _, _, _, match_patch_in_image, *_) = self._import()
        img = _gray(64, 64, seed=140)
        tmpl = img[10:27, 10:27].astype(np.float32)
        r, c, score = match_patch_in_image(tmpl, img, method="ncc")
        assert 0 <= r < 64
        assert 0 <= c < 64
        assert score >= -1.0 and score <= 1.0 + 1e-9

    def test_match_patch_in_image_ssd(self):
        (_, _, _, _, _, _, match_patch_in_image, *_) = self._import()
        img = _gray(32, 32, seed=141)
        tmpl = img[0:9, 0:9].astype(np.float32)
        r, c, score = match_patch_in_image(tmpl, img, method="ssd")
        assert score >= 0.0

    def test_match_patch_invalid_method(self):
        (_, _, _, _, _, _, match_patch_in_image, *_) = self._import()
        img = _gray(32, 32)
        tmpl = img[0:9, 0:9]
        with pytest.raises(ValueError):
            match_patch_in_image(tmpl, img, method="bad")

    def test_find_matches_returns_list(self):
        (PatchConfig, PatchMatch, _, _, _, _, _, find_matches, *_) = self._import()
        img1 = _gray(64, 64, seed=150)
        img2 = _gray(64, 64, seed=151)
        cfg = PatchConfig(patch_size=9, stride=8)
        matches = find_matches(img1, img2, cfg=cfg)
        assert isinstance(matches, list)
        for m in matches:
            assert isinstance(m, PatchMatch)

    def test_patch_match_src_dst_pos(self):
        (PatchConfig, PatchMatch, _, _, _, _, _, find_matches, *_) = self._import()
        img1 = _gray(64, 64, seed=152)
        img2 = _gray(64, 64, seed=153)
        cfg = PatchConfig(patch_size=9, stride=16)
        matches = find_matches(img1, img2, cfg=cfg)
        if matches:
            m = matches[0]
            assert m.src_pos == (m.row1, m.col1)
            assert m.dst_pos == (m.row2, m.col2)

    def test_batch_patch_match(self):
        (PatchConfig, _, _, _, _, _, _, _, _, batch_patch_match) = self._import()
        pairs = [(_gray(seed=i), _gray(seed=i + 5)) for i in range(3)]
        cfg = PatchConfig(patch_size=9, stride=16)
        results = batch_patch_match(pairs, cfg=cfg)
        assert len(results) == 3


# ============================================================
# 4. region_scorer
# ============================================================

class TestRegionScorer:
    def _import(self):
        from puzzle_reconstruction.algorithms.region_scorer import (
            RegionScorerConfig, RegionScore,
            color_similarity, texture_similarity, shape_similarity,
            boundary_proximity, score_region_pair, batch_score_regions,
            rank_region_pairs,
        )
        return (RegionScorerConfig, RegionScore,
                color_similarity, texture_similarity, shape_similarity,
                boundary_proximity, score_region_pair, batch_score_regions,
                rank_region_pairs)

    def test_config_defaults(self):
        (RegionScorerConfig, *_) = self._import()
        cfg = RegionScorerConfig()
        assert cfg.w_color == pytest.approx(0.35)
        assert cfg.total_weight > 0

    def test_config_invalid_weight(self):
        (RegionScorerConfig, *_) = self._import()
        with pytest.raises(ValueError):
            RegionScorerConfig(w_color=-0.1)

    def test_config_invalid_max_distance(self):
        (RegionScorerConfig, *_) = self._import()
        with pytest.raises(ValueError):
            RegionScorerConfig(max_distance=0.0)

    def test_color_similarity_identical(self):
        (_, _, color_similarity, *_) = self._import()
        img = _bgr(32, 32, seed=200)
        score = color_similarity(img, img)
        assert score == pytest.approx(1.0)

    def test_color_similarity_range(self):
        (_, _, color_similarity, *_) = self._import()
        a = _bgr(32, 32, seed=201)
        b = _bgr(32, 32, seed=202)
        score = color_similarity(a, b)
        assert 0.0 <= score <= 1.0

    def test_texture_similarity_identical(self):
        (_, _, _, texture_similarity, *_) = self._import()
        img = _bgr(32, 32, seed=210)
        score = texture_similarity(img, img)
        assert score == pytest.approx(1.0)

    def test_texture_similarity_range(self):
        (_, _, _, texture_similarity, *_) = self._import()
        a = _bgr(32, 32, seed=211)
        b = _bgr(32, 32, seed=212)
        score = texture_similarity(a, b)
        assert 0.0 <= score <= 1.0

    def test_shape_similarity_identical_bbox(self):
        (_, _, _, _, shape_similarity, *_) = self._import()
        bbox = (0, 0, 50, 30)
        score = shape_similarity(bbox, bbox)
        assert score == pytest.approx(1.0)

    def test_shape_similarity_different(self):
        (_, _, _, _, shape_similarity, *_) = self._import()
        score = shape_similarity((0, 0, 100, 10), (0, 0, 10, 100))
        assert 0.0 <= score <= 1.0

    def test_boundary_proximity_close(self):
        (_, _, _, _, _, boundary_proximity, *_) = self._import()
        p1 = (10, 10)
        p2 = (12, 12)
        score = boundary_proximity(p1, p2, max_distance=100.0)
        assert score > 0.9

    def test_boundary_proximity_far(self):
        (_, _, _, _, _, boundary_proximity, *_) = self._import()
        p1 = (0, 0)
        p2 = (100, 100)
        score = boundary_proximity(p1, p2, max_distance=100.0)
        assert 0.0 <= score <= 1.0

    def test_score_region_pair_basic(self):
        (_, RegionScore, _, _, _, _, score_region_pair, *_) = self._import()
        a = _bgr(32, 32, seed=220)
        b = _bgr(32, 32, seed=221)
        bbox_a = (0, 0, 32, 32)
        bbox_b = (32, 0, 32, 32)
        result = score_region_pair(a, bbox_a, b, bbox_b)
        assert isinstance(result, RegionScore)
        assert 0.0 <= result.score <= 1.0

    def test_score_region_pair_with_config(self):
        (RegionScorerConfig, _, _, _, _, _, score_region_pair, *_) = self._import()
        cfg = RegionScorerConfig(w_color=0.5, w_texture=0.5, w_shape=0.0, w_boundary=0.0)
        a = _bgr(32, 32, seed=222)
        b = _bgr(32, 32, seed=223)
        bbox_a = (0, 0, 32, 32)
        bbox_b = (32, 0, 32, 32)
        result = score_region_pair(a, bbox_a, b, bbox_b, cfg=cfg)
        assert 0.0 <= result.score <= 1.0

    def test_batch_score_regions(self):
        (_, RegionScore, _, _, _, _, _, batch_score_regions, _) = self._import()
        bbox = (0, 0, 32, 32)
        pairs = [(_bgr(seed=i), bbox, _bgr(seed=i + 5), bbox) for i in range(3)]
        results = batch_score_regions(pairs)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, RegionScore)

    def test_rank_region_pairs_sorted(self):
        (_, _, _, _, _, _, score_region_pair, batch_score_regions, rank_region_pairs) = self._import()
        bbox = (0, 0, 32, 32)
        pairs = [(_bgr(seed=i), bbox, _bgr(seed=i + 10), bbox) for i in range(4)]
        results = batch_score_regions(pairs)
        # rank_region_pairs returns List[Tuple[int, float]]
        ranked = rank_region_pairs(results)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_region_score_is_confident(self):
        (_, _, _, _, _, _, score_region_pair, *_) = self._import()
        a = _bgr(seed=230)
        bbox = (0, 0, 64, 64)
        result = score_region_pair(a, bbox, a, bbox)
        assert result.score >= 0.0


# ============================================================
# 5. seam_evaluator
# ============================================================

class TestSeamEvaluator:
    def _import(self):
        from puzzle_reconstruction.algorithms.seam_evaluator import (
            SeamConfig, SeamScore,
            extract_seam_strip, color_continuity, gradient_continuity,
            texture_continuity, evaluate_seam, batch_evaluate_seams, rank_seams,
        )
        return (SeamConfig, SeamScore,
                extract_seam_strip, color_continuity, gradient_continuity,
                texture_continuity, evaluate_seam, batch_evaluate_seams, rank_seams)

    def test_seam_config_defaults(self):
        (SeamConfig, *_) = self._import()
        cfg = SeamConfig()
        assert cfg.blend_width == 8
        assert cfg.total_weight > 0

    def test_seam_config_invalid_blend_width(self):
        (SeamConfig, *_) = self._import()
        with pytest.raises(ValueError):
            SeamConfig(blend_width=0)

    def test_seam_config_invalid_weight(self):
        (SeamConfig, *_) = self._import()
        with pytest.raises(ValueError):
            SeamConfig(w_color=-0.1)

    def test_extract_seam_strip_right(self):
        # side: 0=top, 1=right, 2=bottom, 3=left
        (_, _, extract_seam_strip, *_) = self._import()
        img = _bgr(64, 64, seed=300)
        strip = extract_seam_strip(img, side=1, width=8)
        assert strip.shape[1] == 8
        assert strip.shape[0] == 64

    def test_extract_seam_strip_left(self):
        (_, _, extract_seam_strip, *_) = self._import()
        img = _bgr(64, 64, seed=301)
        strip = extract_seam_strip(img, side=3, width=8)
        assert strip.shape[1] == 8

    def test_extract_seam_strip_top(self):
        (_, _, extract_seam_strip, *_) = self._import()
        img = _bgr(64, 64, seed=302)
        strip = extract_seam_strip(img, side=0, width=8)
        assert strip.shape[0] == 8

    def test_extract_seam_strip_bottom(self):
        (_, _, extract_seam_strip, *_) = self._import()
        img = _bgr(64, 64, seed=303)
        strip = extract_seam_strip(img, side=2, width=8)
        assert strip.shape[0] == 8

    def test_color_continuity_identical(self):
        (_, _, _, color_continuity, *_) = self._import()
        strip = _bgr(64, 8, seed=310)
        score = color_continuity(strip, strip)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_color_continuity_range(self):
        (_, _, _, color_continuity, *_) = self._import()
        a = _bgr(64, 8, seed=311)
        b = _bgr(64, 8, seed=312)
        score = color_continuity(a, b)
        assert 0.0 <= score <= 1.0

    def test_gradient_continuity_identical(self):
        (_, _, _, _, gradient_continuity, *_) = self._import()
        strip = _bgr(64, 8, seed=320)
        score = gradient_continuity(strip, strip)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_gradient_continuity_range(self):
        (_, _, _, _, gradient_continuity, *_) = self._import()
        a = _bgr(64, 8, seed=321)
        b = _bgr(64, 8, seed=322)
        score = gradient_continuity(a, b)
        assert 0.0 <= score <= 1.0

    def test_texture_continuity_range(self):
        (_, _, _, _, _, texture_continuity, *_) = self._import()
        a = _bgr(64, 8, seed=330)
        b = _bgr(64, 8, seed=331)
        score = texture_continuity(a, b)
        assert 0.0 <= score <= 1.0

    def test_evaluate_seam_basic(self):
        # evaluate_seam(img_a, side_a, img_b, side_b) — sides: 0=top,1=right,2=bottom,3=left
        (_, SeamScore, _, _, _, _, evaluate_seam, *_) = self._import()
        a = _bgr(64, 64, seed=340)
        b = _bgr(64, 64, seed=341)
        result = evaluate_seam(a, 1, b, 3)
        assert isinstance(result, SeamScore)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_seam_with_config(self):
        (SeamConfig, _, _, _, _, _, evaluate_seam, *_) = self._import()
        cfg = SeamConfig(w_color=1.0, w_gradient=0.0, w_texture=0.0, blend_width=4)
        a = _bgr(64, 64, seed=342)
        b = _bgr(64, 64, seed=343)
        result = evaluate_seam(a, 2, b, 0, cfg=cfg)
        assert 0.0 <= result.score <= 1.0

    def test_evaluate_seam_identical_images(self):
        (_, _, _, _, _, _, evaluate_seam, *_) = self._import()
        img = _bgr(64, 64, seed=344)
        result = evaluate_seam(img, 1, img, 1)
        assert result.score > 0.5

    def test_batch_evaluate_seams(self):
        (_, SeamScore, _, _, _, _, _, batch_evaluate_seams, _) = self._import()
        pairs = [(_bgr(seed=i), 1, _bgr(seed=i + 5), 3) for i in range(3)]
        results = batch_evaluate_seams(pairs)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SeamScore)

    def test_rank_seams(self):
        # rank_seams returns List[Tuple[int, float]]
        (_, _, _, _, _, _, _, batch_evaluate_seams, rank_seams) = self._import()
        pairs = [(_bgr(seed=i), 1, _bgr(seed=i + 5), 3) for i in range(4)]
        results = batch_evaluate_seams(pairs)
        ranked = rank_seams(results)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


# ============================================================
# 6. overlap_resolver
# ============================================================

class TestOverlapResolver:
    def _import(self):
        from puzzle_reconstruction.assembly.overlap_resolver import (
            ResolveConfig, BBox, Overlap, ResolveResult,
            compute_overlap, detect_overlaps, resolve_overlaps,
            compute_total_overlap, overlap_ratio,
        )
        return (ResolveConfig, BBox, Overlap, ResolveResult,
                compute_overlap, detect_overlaps, resolve_overlaps,
                compute_total_overlap, overlap_ratio)

    def _bbox(self, fid, x, y, w, h):
        from puzzle_reconstruction.assembly.overlap_resolver import BBox
        return BBox(fragment_id=fid, x=x, y=y, w=w, h=h)

    def _boxes_dict(self, specs):
        # specs = list of (fid, x, y, w, h)
        return {fid: self._bbox(fid, x, y, w, h) for fid, x, y, w, h in specs}

    def test_bbox_creation(self):
        b = self._bbox(1, 10, 20, 50, 30)
        assert b.x == 10
        assert b.w == 50
        assert b.fragment_id == 1

    def test_bbox_invalid_size(self):
        (_, BBox, *_) = self._import()
        with pytest.raises((ValueError, Exception)):
            BBox(fragment_id=0, x=0, y=0, w=0, h=10)

    def test_compute_overlap_no_overlap(self):
        (_, BBox, Overlap, _, compute_overlap, *_) = self._import()
        a = self._bbox(0, 0, 0, 10, 10)
        b = self._bbox(1, 20, 20, 10, 10)
        ov = compute_overlap(a, b)
        assert isinstance(ov, Overlap)

    def test_compute_overlap_full(self):
        (_, BBox, _, _, compute_overlap, *_) = self._import()
        a = self._bbox(0, 0, 0, 10, 10)
        ov = compute_overlap(a, a)
        assert ov.area >= 0

    def test_detect_overlaps_no_overlap(self):
        (_, BBox, _, _, _, detect_overlaps, *_) = self._import()
        boxes = self._boxes_dict([(i, i * 20, 0, 10, 10) for i in range(3)])
        overlaps = detect_overlaps(boxes)
        assert isinstance(overlaps, list)
        assert overlaps == []

    def test_detect_overlaps_with_overlap(self):
        (_, BBox, _, _, _, detect_overlaps, *_) = self._import()
        boxes = self._boxes_dict([(0, 0, 0, 20, 20), (1, 10, 10, 20, 20)])
        overlaps = detect_overlaps(boxes)
        assert len(overlaps) >= 1  # should have an overlap

    def test_resolve_overlaps_returns_result(self):
        (ResolveConfig, BBox, _, ResolveResult, _, _, resolve_overlaps, *_) = self._import()
        boxes = self._boxes_dict([(i, i * 15, 0, 10, 10) for i in range(3)])
        result = resolve_overlaps(boxes)
        assert isinstance(result, ResolveResult)

    def test_resolve_overlaps_with_config(self):
        (ResolveConfig, BBox, _, ResolveResult, _, _, resolve_overlaps, *_) = self._import()
        cfg = ResolveConfig()
        boxes = self._boxes_dict([(0, 0, 0, 10, 10), (1, 5, 5, 10, 10)])
        result = resolve_overlaps(boxes, cfg=cfg)
        assert isinstance(result, ResolveResult)

    def test_compute_total_overlap(self):
        (_, BBox, _, _, _, _, _, compute_total_overlap, _) = self._import()
        boxes = self._boxes_dict([(i, i * 5, 0, 10, 10) for i in range(3)])
        total = compute_total_overlap(boxes)
        assert total >= 0.0

    def test_overlap_ratio_no_overlap(self):
        (_, BBox, _, _, _, _, _, _, overlap_ratio) = self._import()
        # Boxes far apart
        boxes = self._boxes_dict([(0, 0, 0, 10, 10), (1, 20, 0, 10, 10)])
        ratio = overlap_ratio(boxes)
        assert ratio == pytest.approx(0.0)

    def test_overlap_ratio_with_overlap(self):
        (_, BBox, _, _, _, _, _, _, overlap_ratio) = self._import()
        # Boxes overlapping
        boxes = self._boxes_dict([(0, 0, 0, 10, 10), (1, 5, 0, 10, 10)])
        ratio = overlap_ratio(boxes)
        assert 0.0 <= ratio <= 1.0

    def test_bbox_right_bottom(self):
        b = self._bbox(0, 5, 10, 20, 30)
        assert b.x + b.w == 25
        assert b.y + b.h == 40

    def test_resolve_config_defaults(self):
        (ResolveConfig, *_) = self._import()
        cfg = ResolveConfig()
        assert cfg is not None

    def test_detect_overlaps_empty(self):
        (_, _, _, _, _, detect_overlaps, *_) = self._import()
        assert detect_overlaps({}) == []

    def test_resolve_overlaps_empty(self):
        (ResolveConfig, _, _, ResolveResult, _, _, resolve_overlaps, *_) = self._import()
        result = resolve_overlaps({})
        assert isinstance(result, ResolveResult)


# ============================================================
# 7. position_estimator
# ============================================================

class TestPositionEstimator:
    def _import(self):
        from puzzle_reconstruction.assembly.position_estimator import (
            PositionConfig, FragmentPosition, PositionEstimate,
            snap_to_grid, estimate_grid_positions, refine_positions,
            generate_position_candidates, batch_estimate_positions,
        )
        return (PositionConfig, FragmentPosition, PositionEstimate,
                snap_to_grid, estimate_grid_positions, refine_positions,
                generate_position_candidates, batch_estimate_positions)

    def test_position_config_defaults(self):
        (PositionConfig, *_) = self._import()
        cfg = PositionConfig()
        assert cfg.canvas_w == 512
        assert cfg.canvas_h == 512

    def test_position_config_invalid_canvas(self):
        (PositionConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PositionConfig(canvas_w=0)

    def test_position_config_invalid_padding(self):
        (PositionConfig, *_) = self._import()
        with pytest.raises(ValueError):
            PositionConfig(padding=-1)

    def test_fragment_position_creation(self):
        (_, FragmentPosition, *_) = self._import()
        fp = FragmentPosition(fragment_id=1, x=10, y=20, confidence=0.8)
        assert fp.coords == (10, 20)

    def test_fragment_position_invalid_id(self):
        (_, FragmentPosition, *_) = self._import()
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=-1, x=0, y=0)

    def test_fragment_position_invalid_confidence(self):
        (_, FragmentPosition, *_) = self._import()
        with pytest.raises(ValueError):
            FragmentPosition(fragment_id=0, x=0, y=0, confidence=1.5)

    def test_snap_to_grid_basic(self):
        (_, _, _, snap_to_grid, *_) = self._import()
        sx, sy = snap_to_grid(7, 13, 8)
        assert sx % 8 == 0
        assert sy % 8 == 0

    def test_snap_to_grid_exact(self):
        (_, _, _, snap_to_grid, *_) = self._import()
        sx, sy = snap_to_grid(16, 24, 8)
        assert sx == 16
        assert sy == 24

    def test_snap_to_grid_invalid(self):
        (_, _, _, snap_to_grid, *_) = self._import()
        with pytest.raises(ValueError):
            snap_to_grid(10, 10, 0)

    def test_snap_to_grid_negative_coords(self):
        (_, _, _, snap_to_grid, *_) = self._import()
        sx, sy = snap_to_grid(-5, -5, 8)
        assert sx >= 0
        assert sy >= 0

    def test_estimate_grid_positions_basic(self):
        (PositionConfig, _, PositionEstimate, _, estimate_grid_positions, *_) = self._import()
        ids = [0, 1, 2, 3]
        result = estimate_grid_positions(ids, frag_w=64, frag_h=64)
        assert isinstance(result, PositionEstimate)
        assert result.n_fragments == 4

    def test_estimate_grid_positions_empty(self):
        (_, _, PositionEstimate, _, estimate_grid_positions, *_) = self._import()
        result = estimate_grid_positions([], frag_w=64, frag_h=64)
        assert result.n_fragments == 0

    def test_estimate_grid_positions_snap(self):
        (PositionConfig, _, _, _, estimate_grid_positions, *_) = self._import()
        cfg = PositionConfig(snap_grid=True, snap_size=8)
        result = estimate_grid_positions([0, 1, 2], frag_w=60, frag_h=60, cfg=cfg)
        for pos in result.positions:
            assert pos.x % 8 == 0 or pos.x == 0
            assert pos.y % 8 == 0 or pos.y == 0

    def test_estimate_grid_by_id(self):
        (_, _, _, _, estimate_grid_positions, *_) = self._import()
        result = estimate_grid_positions([5, 10, 15], frag_w=32, frag_h=32)
        by_id = result.by_id
        assert 5 in by_id
        assert 10 in by_id

    def test_refine_positions(self):
        (_, _, _, _, estimate_grid_positions, refine_positions, *_) = self._import()
        base = estimate_grid_positions([0, 1, 2], frag_w=32, frag_h=32)
        offsets = [(5, 5), (-3, 2), (0, 0)]
        refined = refine_positions(base, offsets)
        assert refined.n_fragments == 3
        assert all(p.method == "refined" for p in refined.positions)

    def test_refine_positions_with_confidences(self):
        (_, _, _, _, estimate_grid_positions, refine_positions, *_) = self._import()
        base = estimate_grid_positions([0, 1], frag_w=32, frag_h=32)
        offsets = [(0, 0), (0, 0)]
        confidences = [0.8, 0.6]
        refined = refine_positions(base, offsets, confidences=confidences)
        assert refined.positions[0].confidence == pytest.approx(0.8)

    def test_generate_position_candidates(self):
        (_, _, _, _, _, _, generate_position_candidates, _) = self._import()
        cands = generate_position_candidates(50, 50, radius=2, step=1)
        assert len(cands) > 0
        assert all(cx >= 0 and cy >= 0 for cx, cy in cands)

    def test_generate_position_candidates_zero_radius(self):
        (_, _, _, _, _, _, generate_position_candidates, _) = self._import()
        cands = generate_position_candidates(10, 10, radius=0)
        assert (10, 10) in cands

    def test_batch_estimate_positions(self):
        (_, _, _, _, _, _, _, batch_estimate_positions) = self._import()
        id_lists = [[0, 1], [2, 3], [4]]
        results = batch_estimate_positions(id_lists, frag_w=32, frag_h=32)
        assert len(results) == 3
        assert results[0].n_fragments == 2


# ============================================================
# 8. rl_agent
# ============================================================

class TestRLAgent:
    def _import(self):
        from puzzle_reconstruction.assembly.rl_agent import (
            RLConfig, RLEpisode, TabularPolicy, RLAssembler, rl_assemble,
        )
        return (RLConfig, RLEpisode, TabularPolicy, RLAssembler, rl_assemble)

    def _compat_matrix(self, n=5, seed=42):
        rng = np.random.default_rng(seed)
        m = rng.random((n, n))
        np.fill_diagonal(m, 0)
        return m

    def test_rl_config_defaults(self):
        (RLConfig, *_) = self._import()
        cfg = RLConfig()
        assert cfg.n_episodes == 200
        assert cfg.gamma == 0.95

    def test_tabular_policy_action_probs_sum_to_one(self):
        (_, _, TabularPolicy, *_) = self._import()
        policy = TabularPolicy(n_fragments=5)
        probs = policy.action_probs(0, [1, 2, 3, 4])
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_tabular_policy_action_probs_empty(self):
        (_, _, TabularPolicy, *_) = self._import()
        policy = TabularPolicy(n_fragments=5)
        probs = policy.action_probs(0, [])
        assert len(probs) == 0

    def test_tabular_policy_best_action(self):
        (_, _, TabularPolicy, *_) = self._import()
        policy = TabularPolicy(n_fragments=5)
        action = policy.best_action(0, [1, 2, 3])
        assert action in [1, 2, 3]

    def test_tabular_policy_update(self):
        (_, _, TabularPolicy, *_) = self._import()
        policy = TabularPolicy(n_fragments=5)
        before = policy.theta[0, 1]
        policy.update(0, 1, G=1.0, lr=0.01)
        assert policy.theta[0, 1] != before

    def test_tabular_policy_temperature(self):
        (_, _, TabularPolicy, *_) = self._import()
        policy = TabularPolicy(n_fragments=3, temperature=0.1)
        probs = policy.action_probs(0, [1, 2])
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_rl_assembler_train(self):
        (RLConfig, _, _, RLAssembler, _) = self._import()
        cfg = RLConfig(n_episodes=5, random_seed=0)
        assembler = RLAssembler(config=cfg)
        compat = self._compat_matrix(n=4)
        episodes = assembler.train(compat, n_fragments=4)
        assert len(episodes) == 5

    def test_rl_episode_structure(self):
        (RLConfig, RLEpisode, _, RLAssembler, _) = self._import()
        cfg = RLConfig(n_episodes=3, random_seed=1)
        assembler = RLAssembler(config=cfg)
        compat = self._compat_matrix(n=4)
        episodes = assembler.train(compat, n_fragments=4)
        for ep in episodes:
            assert isinstance(ep, RLEpisode)
            assert ep.n_steps > 0
            assert len(ep.actions) == ep.n_steps

    def test_rl_assembler_assemble(self):
        (RLConfig, _, _, RLAssembler, _) = self._import()
        cfg = RLConfig(n_episodes=10, random_seed=2)
        assembler = RLAssembler(config=cfg)
        fragments = list(range(5))
        compat = self._compat_matrix(n=5)
        order, score = assembler.assemble(fragments, compat)
        assert len(order) == len(fragments)
        assert isinstance(score, float)

    def test_compute_returns_single(self):
        (_, _, _, RLAssembler, _) = self._import()
        returns = RLAssembler._compute_returns([1.0], gamma=0.9)
        assert returns[0] == pytest.approx(1.0)

    def test_compute_returns_discounted(self):
        (_, _, _, RLAssembler, _) = self._import()
        returns = RLAssembler._compute_returns([1.0, 1.0], gamma=0.5)
        assert returns[0] == pytest.approx(1.5)
        assert returns[1] == pytest.approx(1.0)

    def test_score_placement_empty(self):
        (_, _, _, RLAssembler, _) = self._import()
        compat = self._compat_matrix(n=4)
        assert RLAssembler._score_placement([], compat) == 0.0

    def test_score_placement_single(self):
        (_, _, _, RLAssembler, _) = self._import()
        compat = self._compat_matrix(n=4)
        assert RLAssembler._score_placement([2], compat) == 0.0

    def test_score_placement_sequence(self):
        (_, _, _, RLAssembler, _) = self._import()
        compat = np.ones((3, 3))
        score = RLAssembler._score_placement([0, 1, 2], compat)
        assert score == pytest.approx(2.0)

    def test_rl_assemble_convenience(self):
        (RLConfig, _, _, _, rl_assemble) = self._import()
        cfg = RLConfig(n_episodes=5, random_seed=3)
        fragments = list(range(3))
        compat = self._compat_matrix(n=3)
        order, score = rl_assemble(fragments, compat, config=cfg)
        assert len(order) == 3

    def test_rl_assemble_default_config(self):
        (_, _, _, _, rl_assemble) = self._import()
        fragments = list(range(3))
        compat = self._compat_matrix(n=3)
        order, score = rl_assemble(fragments, compat)
        assert isinstance(score, float)


# ============================================================
# 9. evidence_aggregator
# ============================================================

class TestEvidenceAggregator:
    def _import(self):
        from puzzle_reconstruction.scoring.evidence_aggregator import (
            EvidenceConfig, EvidenceScore,
            weight_evidence, threshold_evidence, compute_confidence,
            aggregate_evidence, rank_by_evidence, batch_aggregate,
        )
        return (EvidenceConfig, EvidenceScore,
                weight_evidence, threshold_evidence, compute_confidence,
                aggregate_evidence, rank_by_evidence, batch_aggregate)

    def test_evidence_config_defaults(self):
        (EvidenceConfig, *_) = self._import()
        cfg = EvidenceConfig()
        assert cfg.min_threshold == 0.0
        assert cfg.confidence_threshold == 0.5

    def test_evidence_config_invalid_threshold(self):
        (EvidenceConfig, *_) = self._import()
        with pytest.raises(ValueError):
            EvidenceConfig(min_threshold=-0.1)

    def test_evidence_config_invalid_conf_threshold(self):
        (EvidenceConfig, *_) = self._import()
        with pytest.raises(ValueError):
            EvidenceConfig(confidence_threshold=1.5)

    def test_evidence_config_negative_weight(self):
        (EvidenceConfig, *_) = self._import()
        with pytest.raises(ValueError):
            EvidenceConfig(weights={"color": -1.0})

    def test_weight_evidence_basic(self):
        (_, _, weight_evidence, *_) = self._import()
        scores = {"color": 0.8, "texture": 0.6}
        weights = {"color": 2.0, "texture": 1.0}
        result = weight_evidence(scores, weights)
        assert result["color"] == pytest.approx(1.6)
        assert result["texture"] == pytest.approx(0.6)

    def test_weight_evidence_missing_weight(self):
        (_, _, weight_evidence, *_) = self._import()
        scores = {"color": 0.5}
        result = weight_evidence(scores, {})
        assert result["color"] == pytest.approx(0.5)  # default weight 1.0

    def test_weight_evidence_invalid_score(self):
        (_, _, weight_evidence, *_) = self._import()
        with pytest.raises(ValueError):
            weight_evidence({"color": 1.5}, {})

    def test_threshold_evidence_basic(self):
        (_, _, _, threshold_evidence, *_) = self._import()
        scores = {"color": 0.8, "texture": 0.3}
        result = threshold_evidence(scores, min_threshold=0.5)
        assert result["color"] == pytest.approx(0.8)
        assert result["texture"] == 0.0

    def test_threshold_evidence_zero_threshold(self):
        (_, _, _, threshold_evidence, *_) = self._import()
        scores = {"a": 0.1, "b": 0.9}
        result = threshold_evidence(scores, min_threshold=0.0)
        assert result == scores

    def test_threshold_evidence_invalid(self):
        (_, _, _, threshold_evidence, *_) = self._import()
        with pytest.raises(ValueError):
            threshold_evidence({}, min_threshold=1.5)

    def test_compute_confidence_empty(self):
        (_, _, _, _, compute_confidence, *_) = self._import()
        assert compute_confidence({}, {}) == 0.0

    def test_compute_confidence_single(self):
        (_, _, _, _, compute_confidence, *_) = self._import()
        conf = compute_confidence({"color": 0.6}, {"color": 1.0})
        assert conf == pytest.approx(0.6)

    def test_compute_confidence_weighted(self):
        (_, _, _, _, compute_confidence, *_) = self._import()
        ws = {"a": 2.0, "b": 1.0}
        conf = compute_confidence({"a": 1.0, "b": 0.0}, ws)
        # (2.0*1 + 1.0*0) / (2+1) but weighted_scores = {a: w*s, b: w*s}
        # actually weighted_scores already have weights applied, so total / total_weight
        assert 0.0 <= conf <= 1.0

    def test_aggregate_evidence_basic(self):
        (_, EvidenceScore, _, _, _, aggregate_evidence, *_) = self._import()
        scores = {"color": 0.8, "texture": 0.6}
        result = aggregate_evidence(scores, pair_id=(0, 1))
        assert isinstance(result, EvidenceScore)
        assert 0.0 <= result.confidence <= 1.0
        assert result.pair_id == (0, 1)

    def test_aggregate_evidence_with_config(self):
        (EvidenceConfig, _, _, _, _, aggregate_evidence, *_) = self._import()
        cfg = EvidenceConfig(weights={"color": 2.0}, min_threshold=0.3)
        result = aggregate_evidence({"color": 0.9, "texture": 0.1}, cfg=cfg)
        assert result.n_channels >= 1

    def test_aggregate_evidence_require_all_fail(self):
        (EvidenceConfig, _, _, _, _, aggregate_evidence, *_) = self._import()
        cfg = EvidenceConfig(weights={"color": 1.0, "geometry": 1.0}, require_all=True)
        with pytest.raises(ValueError):
            aggregate_evidence({"color": 0.8}, cfg=cfg)

    def test_evidence_score_is_confident(self):
        (_, EvidenceScore, *_) = self._import()
        es = EvidenceScore(pair_id=(0, 1), confidence=0.8, n_channels=1)
        assert es.is_confident is True

    def test_evidence_score_not_confident(self):
        (_, EvidenceScore, *_) = self._import()
        es = EvidenceScore(pair_id=(0, 1), confidence=0.3, n_channels=1)
        assert es.is_confident is False

    def test_evidence_score_dominant_channel(self):
        (_, EvidenceScore, *_) = self._import()
        es = EvidenceScore(
            pair_id=(0, 1), confidence=0.7, n_channels=2,
            weighted_scores={"color": 0.8, "texture": 0.4}
        )
        assert es.dominant_channel == "color"

    def test_evidence_score_no_dominant_channel(self):
        (_, EvidenceScore, *_) = self._import()
        es = EvidenceScore(pair_id=(0, 1), confidence=0.0, n_channels=0)
        assert es.dominant_channel is None

    def test_rank_by_evidence(self):
        (_, EvidenceScore, _, _, _, _, rank_by_evidence, _) = self._import()
        scores = [
            EvidenceScore(pair_id=(0, i), confidence=float(i) / 10, n_channels=1)
            for i in range(5)
        ]
        ranked = rank_by_evidence(scores)
        confs = [e.confidence for e in ranked]
        assert confs == sorted(confs, reverse=True)

    def test_batch_aggregate_basic(self):
        (_, EvidenceScore, _, _, _, _, _, batch_aggregate) = self._import()
        batch = [{"color": 0.8}, {"texture": 0.5}, {"color": 0.3, "texture": 0.7}]
        results = batch_aggregate(batch)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EvidenceScore)

    def test_batch_aggregate_with_pair_ids(self):
        (_, _, _, _, _, _, _, batch_aggregate) = self._import()
        batch = [{"color": 0.5}, {"color": 0.7}]
        pair_ids = [(0, 1), (1, 2)]
        results = batch_aggregate(batch, pair_ids=pair_ids)
        assert results[0].pair_id == (0, 1)

    def test_batch_aggregate_mismatched_pair_ids(self):
        (_, _, _, _, _, _, _, batch_aggregate) = self._import()
        with pytest.raises(ValueError):
            batch_aggregate([{"a": 0.5}], pair_ids=[(0, 1), (1, 2)])


# ============================================================
# 10. preprocessing chain
# ============================================================

class TestPreprocessingChain:
    def _import(self):
        from puzzle_reconstruction.preprocessing.chain import (
            PreprocessingChain, list_filters, _extract_image,
        )
        return (PreprocessingChain, list_filters, _extract_image)

    def test_list_filters(self):
        (_, list_filters, _) = self._import()
        filters = list_filters()
        assert isinstance(filters, list)
        assert len(filters) > 0
        assert "quality_assessor" in filters

    def test_chain_is_empty_default(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain()
        assert chain.is_empty()

    def test_chain_not_empty_with_filters(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=["denoise"])
        assert not chain.is_empty()

    def test_chain_auto_enhance(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(auto_enhance=True)
        assert not chain.is_empty()

    def test_chain_apply_none_input(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain()
        result = chain.apply(None)
        assert result is None

    def test_chain_apply_empty_image(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain()
        empty = np.array([], dtype=np.uint8).reshape(0, 0)
        result = chain.apply(empty)
        assert result is None

    def test_chain_apply_passthrough(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=[])
        img = _bgr(64, 64, seed=400)
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_chain_apply_unknown_filter(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=["nonexistent_filter_xyz"])
        img = _bgr(64, 64, seed=401)
        result = chain.apply(img)
        assert result is not None  # unknown filters are skipped

    def test_chain_quality_gate_disabled(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=["quality_assessor"], quality_threshold=0.0)
        img = _bgr(64, 64, seed=402)
        result = chain.apply(img)
        assert result is not None

    def test_chain_apply_multiple_unknown_filters(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=["filter_a", "filter_b", "filter_c"])
        img = _bgr(64, 64, seed=403)
        result = chain.apply(img)
        assert result is not None
        assert result.shape == img.shape

    def test_extract_image_ndarray(self):
        (_, _, _extract_image) = self._import()
        img = _bgr(32, 32, seed=410)
        result = _extract_image(img)
        assert result is img

    def test_extract_image_none(self):
        (_, _, _extract_image) = self._import()
        result = _extract_image(None)
        assert result is None

    def test_extract_image_object_with_image_attr(self):
        (_, _, _extract_image) = self._import()
        class Wrapper:
            image = _bgr(32, 32, seed=411)
        result = _extract_image(Wrapper())
        assert isinstance(result, np.ndarray)

    def test_extract_image_object_with_result_attr(self):
        (_, _, _extract_image) = self._import()
        class Wrapper:
            result = _bgr(32, 32, seed=412)
        result = _extract_image(Wrapper())
        assert isinstance(result, np.ndarray)

    def test_chain_apply_grayscale(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=[])
        img = _gray(64, 64, seed=420)
        result = chain.apply(img)
        assert result is not None

    def test_chain_apply_small_image(self):
        (PreprocessingChain, *_) = self._import()
        chain = PreprocessingChain(filters=[])
        img = _bgr(8, 8, seed=421)
        result = chain.apply(img)
        assert result is not None
