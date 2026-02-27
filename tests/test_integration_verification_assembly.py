"""
Integration tests for the verification sub-package:
  - assembly_scorer
  - fragment_validator
  - completeness_checker
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Imports from assembly_scorer ────────────────────────────────────────────
from puzzle_reconstruction.verification.assembly_scorer import (
    AssemblyScoreReport,
    AssemblyScorerParams,
    ScoreComponent,
    compare_assemblies,
    compute_assembly_score,
    rank_assemblies,
    score_coverage,
    score_geometry,
    score_seam_quality,
    score_uniqueness,
)

# ── Imports from fragment_validator ─────────────────────────────────────────
from puzzle_reconstruction.verification.fragment_validator import (
    FragmentValidatorParams,
    ValidationResult,
    batch_validate,
    filter_valid,
    validate_aspect_ratio,
    validate_content_coverage,
    validate_contour,
    validate_dimensions,
    validate_fragment,
)

# ── Imports from completeness_checker ───────────────────────────────────────
from puzzle_reconstruction.verification.completeness_checker import (
    CompletenessReport,
    batch_check_coverage,
    check_fragment_coverage,
    check_spatial_coverage,
    completeness_score,
    find_missing_fragments,
    generate_completeness_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gray_image(h: int = 64, w: int = 64, fill: int = 128) -> np.ndarray:
    """Return a single-channel uint8 image filled with *fill*."""
    return np.full((h, w), fill, dtype=np.uint8)


def _color_image(h: int = 64, w: int = 64, fill: int = 128) -> np.ndarray:
    """Return a 3-channel uint8 BGR image filled with *fill*."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _square_contour(size: int = 40) -> np.ndarray:
    """Return a simple square contour as (N, 2) int32 array."""
    pts = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]],
        dtype=np.int32,
    )
    return pts


def _binary_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a fully-filled binary uint8 mask."""
    return np.ones((h, w), dtype=np.uint8) * 255


# ═════════════════════════════════════════════════════════════════════════════
# TestAssemblyScorer
# ═════════════════════════════════════════════════════════════════════════════

class TestAssemblyScorer:

    def test_score_geometry_returns_float_in_unit_interval(self):
        result = score_geometry(overlap_ratio=0.1, gap_ratio=0.1)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_geometry_perfect(self):
        result = score_geometry(overlap_ratio=0.0, gap_ratio=0.0, alignment_score=1.0)
        assert result == pytest.approx(1.0)

    def test_score_coverage_returns_float_in_unit_interval(self):
        result = score_coverage(n_placed=5, n_total=10)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_coverage_full(self):
        result = score_coverage(n_placed=10, n_total=10)
        assert result == pytest.approx(1.0)

    def test_score_seam_quality_returns_float_in_unit_interval(self):
        result = score_seam_quality([0.8, 0.9, 0.7])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_seam_quality_empty_list_returns_one(self):
        """Empty seam list should be handled gracefully → 1.0."""
        result = score_seam_quality([])
        assert result == pytest.approx(1.0)

    def test_score_uniqueness_returns_float_in_unit_interval(self):
        result = score_uniqueness(n_fragments=10, n_duplicates=2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_uniqueness_no_duplicates(self):
        result = score_uniqueness(n_fragments=5, n_duplicates=0)
        assert result == pytest.approx(1.0)

    def test_compute_assembly_score_returns_assembly_score_report(self):
        report = compute_assembly_score(n_placed=8, n_total=10)
        assert isinstance(report, AssemblyScoreReport)

    def test_assembly_score_report_has_total_score(self):
        report = compute_assembly_score(n_placed=5, n_total=10)
        assert hasattr(report, "total_score")
        assert 0.0 <= report.total_score <= 1.0

    def test_compare_assemblies_returns_int(self):
        report_a = compute_assembly_score(n_placed=10, n_total=10)
        report_b = compute_assembly_score(n_placed=5, n_total=10)
        result = compare_assemblies(report_a, report_b)
        assert result in (-1, 0, 1)

    def test_compare_assemblies_better_first(self):
        report_a = compute_assembly_score(n_placed=10, n_total=10)
        report_b = compute_assembly_score(
            n_placed=5, n_total=10, overlap_ratio=0.5
        )
        assert compare_assemblies(report_a, report_b) == 1

    def test_rank_assemblies_returns_list_in_descending_order(self):
        reports = [
            compute_assembly_score(n_placed=i, n_total=10)
            for i in [3, 7, 10, 1]
        ]
        ranked = rank_assemblies(reports)
        assert isinstance(ranked, list)
        scores = [r.total_score for _, r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_assemblies_rank_numbers_start_at_one(self):
        reports = [compute_assembly_score(n_placed=5, n_total=10)]
        ranked = rank_assemblies(reports)
        assert ranked[0][0] == 1

    def test_assembly_scorer_params_is_constructable(self):
        params = AssemblyScorerParams(
            w_geometry=0.4,
            w_coverage=0.3,
            w_seam=0.2,
            w_uniqueness=0.1,
        )
        assert params.w_geometry == pytest.approx(0.4)


# ═════════════════════════════════════════════════════════════════════════════
# TestFragmentValidator
# ═════════════════════════════════════════════════════════════════════════════

class TestFragmentValidator:

    def test_validate_dimensions_returns_validation_result(self):
        img = _gray_image()
        result = validate_dimensions(img)
        assert isinstance(result, ValidationResult)

    def test_validate_dimensions_valid_image_passes(self):
        img = _gray_image(64, 64)
        result = validate_dimensions(img)
        assert result.passed is True

    def test_validate_aspect_ratio_returns_validation_result(self):
        img = _gray_image()
        result = validate_aspect_ratio(img)
        assert isinstance(result, ValidationResult)

    def test_validate_aspect_ratio_square_image(self):
        img = _gray_image(64, 64)
        result = validate_aspect_ratio(img)
        assert result.passed is True
        assert result.metrics["aspect_ratio"] == pytest.approx(1.0)

    def test_validate_content_coverage_returns_validation_result(self):
        img = _gray_image()
        result = validate_content_coverage(img)
        assert isinstance(result, ValidationResult)

    def test_validate_content_coverage_bright_image_passes(self):
        img = _gray_image(fill=200)
        result = validate_content_coverage(img)
        assert result.passed is True

    def test_validate_fragment_returns_validation_result(self):
        img = _color_image()
        result = validate_fragment(img)
        assert isinstance(result, ValidationResult)

    def test_validate_fragment_with_contour(self):
        img = _color_image(64, 64)
        contour = _square_contour(40)
        result = validate_fragment(img, contour=contour)
        assert isinstance(result, ValidationResult)

    def test_batch_validate_returns_list_of_validation_result(self):
        images = [_gray_image() for _ in range(3)]
        results = batch_validate(images)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_batch_validate_empty_list_returns_empty(self):
        results = batch_validate([])
        assert results == []

    def test_filter_valid_returns_list_of_int(self):
        images = [_gray_image(64, 64), _gray_image(64, 64)]
        results = batch_validate(images)
        valid_indices = filter_valid(results)
        assert isinstance(valid_indices, list)

    def test_filter_valid_all_pass(self):
        images = [_gray_image(64, 64) for _ in range(4)]
        results = batch_validate(images)
        valid_indices = filter_valid(results)
        # All should pass – indices are fragment_idx values (0..3)
        assert len(valid_indices) == 4

    def test_tiny_fragment_handled_gracefully(self):
        """A 1x1 image must not crash the validator."""
        img = _gray_image(h=1, w=1, fill=200)
        params = FragmentValidatorParams(min_width=1, min_height=1)
        result = validate_fragment(img, params=params)
        assert isinstance(result, ValidationResult)

    def test_validation_result_has_is_valid_attribute(self):
        """ValidationResult should expose 'passed' (used as is_valid)."""
        result = ValidationResult()
        assert hasattr(result, "passed")

    def test_validate_contour_on_simple_contour(self):
        contour = _square_contour(50)
        result = validate_contour(contour)
        assert isinstance(result, ValidationResult)
        assert "n_points" in result.metrics
        assert "contour_area" in result.metrics

    def test_validate_contour_square_passes(self):
        contour = _square_contour(50)
        result = validate_contour(contour)
        assert result.passed is True


# ═════════════════════════════════════════════════════════════════════════════
# TestCompletenessChecker
# ═════════════════════════════════════════════════════════════════════════════

class TestCompletenessChecker:

    def test_check_fragment_coverage_returns_float(self):
        result = check_fragment_coverage([0, 1, 2], [0, 1, 2, 3, 4])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_find_missing_fragments_returns_list(self):
        missing = find_missing_fragments([0, 1], [0, 1, 2, 3])
        assert isinstance(missing, list)
        assert missing == [2, 3]

    def test_completeness_score_returns_float_in_unit_interval(self):
        score = completeness_score(n_placed=7, n_total=10, pixel_coverage=0.8)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_generate_completeness_report_returns_completeness_report(self):
        report = generate_completeness_report(
            placed_ids=[0, 1, 2], all_ids=[0, 1, 2, 3, 4]
        )
        assert isinstance(report, CompletenessReport)

    def test_batch_check_coverage_returns_list(self):
        all_ids = [0, 1, 2, 3, 4]
        placed_sets = [[0, 1], [0, 1, 2], [0, 1, 2, 3, 4]]
        results = batch_check_coverage(placed_sets, all_ids)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_completeness_report_has_coverage_ratio(self):
        report = generate_completeness_report(
            placed_ids=[0, 1], all_ids=[0, 1, 2]
        )
        assert hasattr(report, "fragment_coverage")
        # fragment_coverage is the coverage ratio
        assert 0.0 <= report.fragment_coverage <= 1.0

    def test_all_fragments_present_coverage_ratio_is_one(self):
        all_ids = [0, 1, 2, 3]
        report = generate_completeness_report(
            placed_ids=all_ids, all_ids=all_ids
        )
        assert report.fragment_coverage == pytest.approx(1.0)

    def test_missing_fragments_coverage_ratio_less_than_one(self):
        all_ids = [0, 1, 2, 3, 4]
        report = generate_completeness_report(
            placed_ids=[0, 1], all_ids=all_ids
        )
        assert report.fragment_coverage < 1.0

    def test_check_spatial_coverage_returns_float(self):
        masks = [_binary_mask(32, 32), _binary_mask(32, 32)]
        result = check_spatial_coverage(masks, target_shape=(64, 64))
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_empty_placed_list_returns_zero_score(self):
        score = completeness_score(n_placed=0, n_total=5, pixel_coverage=0.0)
        assert score == pytest.approx(0.0)


# ═════════════════════════════════════════════════════════════════════════════
# TestVerificationIntegration
# ═════════════════════════════════════════════════════════════════════════════

class TestVerificationIntegration:

    def test_fragment_validator_assembly_scorer_pipeline(self):
        """Validate fragments then score an assembly built from valid ones."""
        images = [_color_image(64, 64) for _ in range(5)]
        results = batch_validate(images)
        valid_idx = filter_valid(results)

        n_placed = len(valid_idx)
        n_total = len(images)
        report = compute_assembly_score(n_placed=n_placed, n_total=n_total)

        assert isinstance(report, AssemblyScoreReport)
        assert report.n_fragments == n_placed

    def test_batch_validate_filter_valid_gives_valid_indices(self):
        # Mix valid (64x64) and invalid-size (1x1 with default min_width=16) images
        valid_imgs = [_color_image(64, 64) for _ in range(3)]
        tiny_imgs = [_color_image(1, 1) for _ in range(2)]
        all_images = valid_imgs + tiny_imgs

        results = batch_validate(all_images)
        valid_idx = filter_valid(results)

        # The valid indices should only include the first 3
        # (tiny images fail min_width / min_height)
        assert all(i < 3 for i in valid_idx)

    def test_completeness_checker_assembly_scorer_chained(self):
        """Pipeline: completeness report feeds n_placed into assembly scorer."""
        all_ids = list(range(10))
        placed_ids = list(range(7))

        comp_report = generate_completeness_report(placed_ids, all_ids)
        asm_report = compute_assembly_score(
            n_placed=comp_report.n_placed,
            n_total=comp_report.n_total,
        )

        assert isinstance(asm_report, AssemblyScoreReport)
        assert asm_report.total_score <= 1.0

    def test_rank_assemblies_on_multiple_assemblies(self):
        params_list = [
            dict(n_placed=10, n_total=10, overlap_ratio=0.0),
            dict(n_placed=8, n_total=10, overlap_ratio=0.1),
            dict(n_placed=5, n_total=10, overlap_ratio=0.3),
        ]
        reports = [compute_assembly_score(**p) for p in params_list]
        ranked = rank_assemblies(reports)

        assert len(ranked) == 3
        scores = [r.total_score for _, r in ranked]
        assert scores[0] >= scores[1] >= scores[2]

    def test_compare_assemblies_on_two_assemblies(self):
        report_a = compute_assembly_score(
            n_placed=10, n_total=10, seam_scores=[0.9, 0.95]
        )
        report_b = compute_assembly_score(
            n_placed=4, n_total=10, seam_scores=[0.3, 0.2],
            overlap_ratio=0.4
        )
        cmp = compare_assemblies(report_a, report_b)
        assert cmp == 1  # report_a should be better


# ═════════════════════════════════════════════════════════════════════════════
# TestVerificationEdgeCases
# ═════════════════════════════════════════════════════════════════════════════

class TestVerificationEdgeCases:

    def test_single_fragment_assembly_scores(self):
        """Assembly with a single placed fragment should not crash."""
        report = compute_assembly_score(n_placed=1, n_total=1)
        assert isinstance(report, AssemblyScoreReport)
        assert 0.0 <= report.total_score <= 1.0

    def test_score_seam_quality_single_seam(self):
        result = score_seam_quality([0.75])
        assert result == pytest.approx(0.75)

    def test_score_uniqueness_all_unique(self):
        result = score_uniqueness(n_fragments=10, n_duplicates=0)
        assert result == pytest.approx(1.0)

    def test_completeness_with_no_placed_fragments(self):
        all_ids = [0, 1, 2, 3, 4]
        report = generate_completeness_report(
            placed_ids=[], all_ids=all_ids
        )
        assert report.fragment_coverage == pytest.approx(0.0)
        assert report.n_placed == 0

    def test_fragment_validator_with_tiny_images(self):
        """Tiny images should not raise exceptions."""
        img = _gray_image(h=2, w=2, fill=200)
        params = FragmentValidatorParams(min_width=1, min_height=1)
        result = validate_fragment(img, params=params)
        assert isinstance(result, ValidationResult)

    def test_score_geometry_max_penalty(self):
        """Full overlap and full gap → score is 0 (if alignment 1)."""
        result = score_geometry(overlap_ratio=1.0, gap_ratio=1.0, alignment_score=1.0)
        assert result == pytest.approx(0.0)

    def test_score_coverage_zero_placed(self):
        result = score_coverage(n_placed=0, n_total=5)
        assert result == pytest.approx(0.0)

    def test_rank_assemblies_empty_list(self):
        ranked = rank_assemblies([])
        assert ranked == []

    def test_completeness_score_full(self):
        """When all fragments placed and full pixel coverage → score is 1."""
        score = completeness_score(n_placed=5, n_total=5, pixel_coverage=1.0)
        assert score == pytest.approx(1.0)

    def test_batch_check_coverage_single_set(self):
        all_ids = [0, 1, 2]
        results = batch_check_coverage([[0, 1, 2]], all_ids)
        assert len(results) == 1
        assert results[0] == pytest.approx(1.0)
