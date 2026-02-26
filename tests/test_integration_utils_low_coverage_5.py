"""Integration tests for under-tested utility modules (batch 5).

Modules covered:
    1.  puzzle_reconstruction.utils.region_score_utils
    2.  puzzle_reconstruction.utils.rotation_hist_utils
    3.  puzzle_reconstruction.utils.rotation_score_utils
    4.  puzzle_reconstruction.utils.rotation_utils
    5.  puzzle_reconstruction.utils.sampling_utils
    6.  puzzle_reconstruction.utils.score_aggregator
    7.  puzzle_reconstruction.utils.score_matrix_utils
    8.  puzzle_reconstruction.utils.score_norm_utils
    9.  puzzle_reconstruction.utils.score_seam_utils
    10. puzzle_reconstruction.utils.scoring_pipeline_utils
    11. puzzle_reconstruction.utils.segment_utils
    12. puzzle_reconstruction.utils.seq_gap_utils
    13. puzzle_reconstruction.utils.sequence_utils
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

from puzzle_reconstruction.utils.region_score_utils import (
    RegionScore,
    RegionScoreConfig,
    batch_evaluate_regions,
    evaluate_region,
    filter_by_score,
    mask_perimeter,
    normalize_scores,
    rank_regions,
    region_compactness,
    region_extent,
    region_score_stats,
    score_region,
    top_k_regions,
)

from puzzle_reconstruction.utils.rotation_hist_utils import (
    HistogramDistanceConfig,
    HistogramDistanceEntry,
    HistogramDistanceSummary,
    RotationAnalysisConfig,
    RotationAnalysisEntry,
    RotationAnalysisSummary,
    batch_summarise_histogram_distance_entries,
    batch_summarise_rotation_analysis,
    best_histogram_distance_entry,
    best_rotation_entry,
    compare_histogram_distance_summaries,
    compare_rotation_summaries,
    filter_histogram_by_fragment,
    filter_histogram_by_max_distance,
    filter_histogram_by_metric,
    filter_rotation_by_angle_range,
    filter_rotation_by_confidence,
    filter_rotation_by_method,
    histogram_distance_stats,
    make_histogram_distance_entry,
    make_rotation_analysis_entry,
    rotation_angle_stats,
    summarise_histogram_distance_entries,
    summarise_rotation_analysis,
    top_k_closest_histogram_pairs,
    top_k_rotation_entries,
)

from puzzle_reconstruction.utils.rotation_score_utils import (
    RotationScoreConfig,
    RotationScoreEntry,
    aggregate_angles,
    angle_agreement,
    batch_make_entries,
    best_entry,
    filter_by_angle_range,
    filter_by_confidence,
    filter_by_method,
    group_by_method,
    make_entry,
    rank_by_confidence,
    rotation_score_stats,
    top_k_entries,
)

from puzzle_reconstruction.utils.rotation_utils import (
    RotationConfig,
    angle_difference,
    angles_to_matrix,
    batch_rotate_images,
    estimate_rotation,
    nearest_discrete,
    normalize_angle,
    rotate_image_angle,
    rotate_points_angle,
)

from puzzle_reconstruction.utils.sampling_utils import (
    SamplingConfig,
    acceptance_probability,
    batch_uniform_sample,
    sample_angle,
    sample_permutation,
    sample_position,
    sample_positions_grid,
    sample_swap_pair,
    uniform_sample,
    weighted_sample,
)

from puzzle_reconstruction.utils.score_aggregator import (
    AggregationMethod,
    AggregationResult,
    ScoreVector,
    aggregate_matrix,
    aggregate_pair,
    batch_aggregate,
    geometric_mean,
    harmonic_mean,
    top_k_pairs,
    weighted_sum,
)

from puzzle_reconstruction.utils.score_matrix_utils import (
    MatrixStats,
    RankEntry,
    ScoreMatrixConfig,
    apply_intra_fragment_mask,
    batch_matrix_stats,
    filter_by_threshold,
    intra_fragment_mask,
    matrix_stats,
    normalize_rows,
    symmetrize,
    threshold_matrix,
    top_k_indices,
    top_k_per_row,
    zero_diagonal,
)

from puzzle_reconstruction.utils.score_norm_utils import (
    ScoreNormConfig,
    ScoreNormEntry,
    ScoreNormSummary,
    batch_summarise_norm,
    compare_norm_summaries,
    entries_from_scores,
    filter_by_normalized_range,
    filter_by_original_range,
    make_norm_entry,
    norm_entry_stats,
    summarise_norm,
    top_k_norm_entries,
)

from puzzle_reconstruction.utils.score_seam_utils import (
    AggregationRunRecord,
    GradientRunRecord,
    NormalizationRecord,
    OrientationHistogramRecord,
    ScoreCalibrationRecord,
    SeamRunRecord,
    SeamScoreMatrix,
    make_normalization_record,
    make_seam_run_record,
)

from puzzle_reconstruction.utils.scoring_pipeline_utils import (
    BoundaryScoreRecord,
    PatchComparisonRecord,
    PipelineReport,
    StageResult,
    build_pipeline_report,
    rank_stage_results,
    summarize_boundary_scores,
)

from puzzle_reconstruction.utils.segment_utils import (
    RegionInfo,
    SegmentConfig,
    all_regions,
    extract_boundary,
    filter_regions,
    label_mask,
    largest_region,
    mask_bounding_box,
    mask_from_labels,
    mask_statistics,
    region_info,
)

from puzzle_reconstruction.utils.seq_gap_utils import (
    GapScoreConfig,
    GapScoreEntry,
    GapScoreSummary,
    SequenceScoreConfig,
    SequenceScoreEntry,
    SequenceScoreSummary,
    batch_summarise_gap_score_entries,
    batch_summarise_sequence_score_entries,
    best_gap_entry,
    best_sequence_entry,
    compare_gap_summaries,
    compare_sequence_summaries,
    filter_gap_by_category,
    filter_gap_by_max_distance,
    filter_overlapping_gaps,
    filter_sequence_by_algorithm,
    filter_sequence_by_min_score,
    filter_full_sequences,
    gap_score_stats,
    make_gap_score_entry,
    make_sequence_score_entry,
    sequence_score_stats,
    summarise_gap_score_entries,
    summarise_sequence_score_entries,
    top_k_closest_gaps,
    top_k_sequence_entries,
)

from puzzle_reconstruction.utils.sequence_utils import (
    SequenceConfig,
    align_sequences,
    batch_rank,
    invert_sequence,
    kendall_tau_distance,
    longest_increasing,
    normalize_sequence,
    rank_sequence,
    segment_by_threshold,
    sliding_scores,
)

RNG = np.random.default_rng(42)


# ==============================================================================
# 1. region_score_utils
# ==============================================================================

class TestRegionScoreUtils:

    def _solid_square_mask(self, size: int = 10) -> np.ndarray:
        m = np.zeros((size, size), dtype=np.uint8)
        m[1:-1, 1:-1] = 255
        return m

    def test_region_compactness_circle_approx(self):
        # For a circle, compactness should be close to 1
        area = 100
        perim = 2 * math.pi * math.sqrt(area / math.pi)
        c = region_compactness(area, perim)
        assert 0.9 <= c <= 1.0

    def test_region_compactness_zero_perimeter(self):
        assert region_compactness(100, 0.0) == 0.0

    def test_region_compactness_clamped_at_one(self):
        # Very large area / tiny perimeter can exceed 1 mathematically
        c = region_compactness(10_000, 1.0)
        assert c <= 1.0

    def test_region_extent_full_bbox(self):
        bbox = (0, 0, 10, 10)
        ext = region_extent(100, bbox)
        assert ext == pytest.approx(1.0)

    def test_region_extent_partial(self):
        bbox = (0, 0, 10, 20)  # 200 px bbox, 100 px area
        ext = region_extent(100, bbox)
        assert ext == pytest.approx(0.5)

    def test_region_extent_zero_bbox(self):
        ext = region_extent(0, (5, 5, 5, 5))
        assert 0.0 <= ext <= 1.0

    def test_mask_perimeter_solid_square(self):
        m = np.ones((5, 5), dtype=np.uint8)
        p = mask_perimeter(m)
        assert p > 0

    def test_mask_perimeter_empty_mask(self):
        m = np.zeros((5, 5), dtype=np.uint8)
        assert mask_perimeter(m) == 0.0

    def test_score_region_in_range(self):
        mask = self._solid_square_mask(10)
        area = int(mask.sum() / 255)
        bbox = (0, 0, 10, 10)
        sc = score_region(area, bbox, mask)
        assert 0.0 <= sc <= 1.0

    def test_score_region_custom_config(self):
        cfg = RegionScoreConfig(w_area=1.0, w_compactness=0.0, w_extent=0.0)
        mask = self._solid_square_mask(8)
        area = int((mask > 0).sum())
        sc = score_region(area, (0, 0, 8, 8), mask, cfg)
        assert 0.0 <= sc <= 1.0

    def test_evaluate_region_returns_region_score(self):
        mask = self._solid_square_mask(10)
        area = int((mask > 0).sum())
        rs = evaluate_region(1, area, (0, 0, 10, 10), mask)
        assert isinstance(rs, RegionScore)
        assert rs.label == 1
        assert 0.0 <= rs.score <= 1.0

    def test_filter_by_score_threshold(self):
        scores = [
            RegionScore(label=i, area=100, compactness=0.5, extent=0.5, score=i * 0.1)
            for i in range(11)
        ]
        filtered = filter_by_score(scores, threshold=0.5)
        assert all(r.score >= 0.5 for r in filtered)

    def test_rank_regions_descending(self):
        rng = np.random.default_rng(0)
        scores = [
            RegionScore(label=i, area=10, compactness=0.5, extent=0.5, score=float(v))
            for i, v in enumerate(rng.random(5))
        ]
        ranked = rank_regions(scores)
        sc_vals = [r.score for r in ranked]
        assert sc_vals == sorted(sc_vals, reverse=True)

    def test_rank_regions_ascending(self):
        scores = [
            RegionScore(label=i, area=10, compactness=0.5, extent=0.5, score=i * 0.1)
            for i in range(5)
        ]
        ranked = rank_regions(scores, reverse=False)
        sc_vals = [r.score for r in ranked]
        assert sc_vals == sorted(sc_vals)

    def test_top_k_regions(self):
        scores = [
            RegionScore(label=i, area=10, compactness=0.5, extent=0.5, score=i * 0.1)
            for i in range(10)
        ]
        top = top_k_regions(scores, k=3)
        assert len(top) == 3
        assert top[0].score >= top[1].score >= top[2].score

    def test_region_score_stats_empty(self):
        stats = region_score_stats([])
        assert stats["n"] == 0

    def test_region_score_stats_non_empty(self):
        scores = [
            RegionScore(label=i, area=100, compactness=0.5, extent=0.5, score=i * 0.25)
            for i in range(4)
        ]
        stats = region_score_stats(scores)
        assert stats["n"] == 4
        assert stats["total_area"] == 400

    def test_batch_evaluate_regions(self):
        mask = self._solid_square_mask(10)
        area = int((mask > 0).sum())
        regions = [
            {"label": i, "area": area, "bbox": (0, 0, 10, 10), "mask": mask}
            for i in range(3)
        ]
        results = batch_evaluate_regions(regions)
        assert len(results) == 3

    def test_normalize_scores_range(self):
        scores = [
            RegionScore(label=i, area=100, compactness=0.5, extent=0.5, score=i * 0.1)
            for i in range(5)
        ]
        normed = normalize_scores(scores)
        sc_vals = [r.score for r in normed]
        assert min(sc_vals) >= 0.0
        assert max(sc_vals) <= 1.0

    def test_normalize_scores_empty(self):
        assert normalize_scores([]) == []

    def test_normalize_scores_uniform(self):
        scores = [
            RegionScore(label=i, area=10, compactness=0.5, extent=0.5, score=0.5)
            for i in range(3)
        ]
        normed = normalize_scores(scores)
        assert all(r.score == pytest.approx(1.0) for r in normed)


# ==============================================================================
# 2. rotation_hist_utils
# ==============================================================================

class TestRotationHistUtils:

    def _make_entries(self, n: int = 5) -> list:
        rng = np.random.default_rng(7)
        return [
            make_rotation_analysis_entry(
                fragment_id=i,
                angle_deg=float(rng.uniform(0, 360)),
                confidence=float(rng.uniform(0, 1)),
                method="procrustes" if i % 2 == 0 else "phase",
            )
            for i in range(n)
        ]

    def test_make_rotation_analysis_entry(self):
        e = make_rotation_analysis_entry(0, 45.0, 0.9, method="phase", extra=1)
        assert e.fragment_id == 0
        assert e.angle_deg == 45.0
        assert e.confidence == 0.9
        assert e.params["extra"] == 1

    def test_summarise_rotation_analysis_empty(self):
        s = summarise_rotation_analysis([])
        assert s.n_entries == 0

    def test_summarise_rotation_analysis_basic(self):
        entries = self._make_entries(6)
        s = summarise_rotation_analysis(entries)
        assert s.n_entries == 6
        assert s.std_angle_deg >= 0.0

    def test_filter_rotation_by_confidence(self):
        entries = self._make_entries(10)
        filtered = filter_rotation_by_confidence(entries, min_confidence=0.5)
        assert all(e.confidence >= 0.5 for e in filtered)

    def test_filter_rotation_by_method(self):
        entries = self._make_entries(10)
        filtered = filter_rotation_by_method(entries, "phase")
        assert all(e.method == "phase" for e in filtered)

    def test_filter_rotation_by_angle_range(self):
        entries = self._make_entries(20)
        filtered = filter_rotation_by_angle_range(entries, 90.0, 270.0)
        assert all(90.0 <= e.angle_deg <= 270.0 for e in filtered)

    def test_top_k_rotation_entries(self):
        entries = self._make_entries(10)
        top = top_k_rotation_entries(entries, k=3)
        assert len(top) == 3
        confs = [e.confidence for e in top]
        assert confs == sorted(confs, reverse=True)

    def test_best_rotation_entry(self):
        entries = self._make_entries(5)
        best = best_rotation_entry(entries)
        assert best is not None
        assert best.confidence == max(e.confidence for e in entries)

    def test_best_rotation_entry_empty(self):
        assert best_rotation_entry([]) is None

    def test_rotation_angle_stats(self):
        entries = self._make_entries(5)
        stats = rotation_angle_stats(entries)
        assert "min" in stats and "max" in stats
        assert stats["count"] == 5

    def test_compare_rotation_summaries(self):
        e1 = self._make_entries(3)
        e2 = self._make_entries(4)
        s1 = summarise_rotation_analysis(e1)
        s2 = summarise_rotation_analysis(e2)
        cmp = compare_rotation_summaries(s1, s2)
        assert "delta_mean_angle_deg" in cmp

    def test_batch_summarise_rotation_analysis(self):
        groups = [self._make_entries(3), self._make_entries(4)]
        summaries = batch_summarise_rotation_analysis(groups)
        assert len(summaries) == 2

    # Histogram distance entries
    def _make_hist_entries(self, n: int = 5) -> list:
        rng = np.random.default_rng(11)
        return [
            make_histogram_distance_entry(
                frag_a=i, frag_b=i + 1,
                distance=float(rng.uniform(0, 2)),
                metric="emd" if i % 2 == 0 else "chi2",
            )
            for i in range(n)
        ]

    def test_make_histogram_distance_entry(self):
        e = make_histogram_distance_entry(0, 1, 0.5, metric="emd", k=3)
        assert e.frag_a == 0
        assert e.frag_b == 1
        assert e.distance == 0.5
        assert e.params["k"] == 3

    def test_summarise_histogram_distance_entries_empty(self):
        s = summarise_histogram_distance_entries([])
        assert s.n_pairs == 0

    def test_summarise_histogram_distance_entries(self):
        entries = self._make_hist_entries(5)
        s = summarise_histogram_distance_entries(entries)
        assert s.n_pairs == 5
        assert s.min_distance <= s.mean_distance <= s.max_distance

    def test_filter_histogram_by_max_distance(self):
        entries = self._make_hist_entries(10)
        filtered = filter_histogram_by_max_distance(entries, 1.0)
        assert all(e.distance <= 1.0 for e in filtered)

    def test_filter_histogram_by_metric(self):
        entries = self._make_hist_entries(10)
        filtered = filter_histogram_by_metric(entries, "emd")
        assert all(e.metric == "emd" for e in filtered)

    def test_filter_histogram_by_fragment(self):
        entries = self._make_hist_entries(10)
        filtered = filter_histogram_by_fragment(entries, 0)
        assert all(e.frag_a == 0 or e.frag_b == 0 for e in filtered)

    def test_top_k_closest_histogram_pairs(self):
        entries = self._make_hist_entries(10)
        top = top_k_closest_histogram_pairs(entries, k=3)
        assert len(top) == 3
        dists = [e.distance for e in top]
        assert dists == sorted(dists)

    def test_best_histogram_distance_entry(self):
        entries = self._make_hist_entries(5)
        best = best_histogram_distance_entry(entries)
        assert best is not None
        assert best.distance == min(e.distance for e in entries)

    def test_histogram_distance_stats(self):
        entries = self._make_hist_entries(5)
        stats = histogram_distance_stats(entries)
        assert stats["count"] == 5

    def test_compare_histogram_distance_summaries(self):
        e1 = self._make_hist_entries(3)
        e2 = self._make_hist_entries(6)
        s1 = summarise_histogram_distance_entries(e1)
        s2 = summarise_histogram_distance_entries(e2)
        cmp = compare_histogram_distance_summaries(s1, s2)
        assert "delta_n_pairs" in cmp

    def test_batch_summarise_histogram_distance_entries(self):
        groups = [self._make_hist_entries(3), self._make_hist_entries(4)]
        summaries = batch_summarise_histogram_distance_entries(groups)
        assert len(summaries) == 2


# ==============================================================================
# 3. rotation_score_utils
# ==============================================================================

class TestRotationScoreUtils:

    def _make_entries(self):
        return batch_make_entries(
            image_ids=[0, 1, 2, 3],
            angles_deg=[0.0, 45.0, 90.0, 135.0],
            confidences=[0.9, 0.5, 0.7, 0.3],
            methods=["pca", "moments", "pca", "gradient"],
        )

    def test_make_entry_valid(self):
        e = make_entry(0, 45.0, 0.8, "pca")
        assert e.image_id == 0
        assert e.angle_deg == 45.0
        assert e.confidence == 0.8

    def test_make_entry_invalid_confidence(self):
        with pytest.raises(ValueError):
            make_entry(0, 0.0, 1.5, "pca")

    def test_make_entry_negative_image_id(self):
        with pytest.raises(ValueError):
            make_entry(-1, 0.0, 0.5, "pca")

    def test_filter_by_confidence(self):
        entries = self._make_entries()
        filtered = filter_by_confidence(entries, 0.6)
        assert all(e.confidence >= 0.6 for e in filtered)

    def test_filter_by_method(self):
        entries = self._make_entries()
        filtered = filter_by_method(entries, "pca")
        assert all(e.method == "pca" for e in filtered)
        assert len(filtered) == 2

    def test_filter_by_angle_range(self):
        entries = self._make_entries()
        filtered = filter_by_angle_range(entries, 0.0, 90.0)
        assert all(0.0 <= e.angle_deg <= 90.0 for e in filtered)

    def test_filter_by_angle_range_invalid(self):
        entries = self._make_entries()
        with pytest.raises(ValueError):
            filter_by_angle_range(entries, 90.0, 0.0)

    def test_rank_by_confidence_descending(self):
        entries = self._make_entries()
        ranked = rank_by_confidence(entries)
        confs = [e.confidence for e in ranked]
        assert confs == sorted(confs, reverse=True)

    def test_best_entry_no_filter(self):
        entries = self._make_entries()
        best = best_entry(entries)
        assert best is not None
        assert best.confidence == max(e.confidence for e in entries)

    def test_best_entry_with_preferred_method(self):
        entries = self._make_entries()
        cfg = RotationScoreConfig(preferred_method="moments")
        best = best_entry(entries, cfg)
        assert best is not None
        assert best.method == "moments"

    def test_best_entry_empty(self):
        cfg = RotationScoreConfig(min_confidence=0.99)
        result = best_entry(self._make_entries(), cfg)
        assert result is None

    def test_aggregate_angles_weighted(self):
        entries = self._make_entries()
        agg = aggregate_angles(entries)
        assert isinstance(agg, float)

    def test_aggregate_angles_explicit_weights(self):
        entries = self._make_entries()
        agg = aggregate_angles(entries, weights=[1.0, 1.0, 1.0, 1.0])
        expected = sum(e.angle_deg for e in entries) / len(entries)
        assert agg == pytest.approx(expected)

    def test_rotation_score_stats(self):
        entries = self._make_entries()
        stats = rotation_score_stats(entries)
        assert stats["n"] == 4
        assert "mean_angle" in stats

    def test_rotation_score_stats_empty(self):
        stats = rotation_score_stats([])
        assert stats["n"] == 0

    def test_angle_agreement_all_same(self):
        entries = [make_entry(i, 45.0, 0.5, "pca") for i in range(4)]
        agree = angle_agreement(entries, tolerance_deg=1.0)
        assert agree == pytest.approx(1.0)

    def test_angle_agreement_all_different(self):
        entries = [
            make_entry(0, 0.0, 0.5, "pca"),
            make_entry(1, 180.0, 0.5, "pca"),
        ]
        agree = angle_agreement(entries, tolerance_deg=1.0)
        assert agree == pytest.approx(0.0)

    def test_batch_make_entries_length_mismatch(self):
        with pytest.raises(ValueError):
            batch_make_entries([0, 1], [0.0], [0.5, 0.5], ["pca", "pca"])

    def test_top_k_entries(self):
        entries = self._make_entries()
        top = top_k_entries(entries, k=2)
        assert len(top) == 2
        assert top[0].confidence >= top[1].confidence

    def test_group_by_method(self):
        entries = self._make_entries()
        groups = group_by_method(entries)
        assert "pca" in groups
        assert len(groups["pca"]) == 2


# ==============================================================================
# 4. rotation_utils
# ==============================================================================

class TestRotationUtils:

    def test_rotate_image_angle_shape_2d(self):
        img = np.ones((20, 30), dtype=np.uint8) * 128
        rotated = rotate_image_angle(img, 45.0)
        assert rotated.ndim == 2

    def test_rotate_image_angle_shape_3d(self):
        img = np.ones((20, 30, 3), dtype=np.uint8) * 128
        rotated = rotate_image_angle(img, 90.0)
        assert rotated.ndim == 3

    def test_rotate_image_angle_no_expand(self):
        cfg = RotationConfig(expand=False)
        img = np.zeros((20, 20), dtype=np.uint8)
        rotated = rotate_image_angle(img, 45.0, cfg)
        assert rotated.shape == (20, 20)

    def test_rotate_image_invalid_ndim(self):
        img = np.ones((4, 4, 3, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            rotate_image_angle(img, 10.0)

    def test_rotate_points_angle_basic(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        rotated = rotate_points_angle(pts, math.pi / 2)
        assert rotated.shape == (3, 2)

    def test_rotate_points_angle_360_is_identity(self):
        pts = np.array([[3.0, 4.0], [1.0, 2.0]])
        rotated = rotate_points_angle(pts, 2 * math.pi)
        np.testing.assert_allclose(rotated, pts, atol=1e-10)

    def test_rotate_points_angle_invalid_shape(self):
        with pytest.raises(ValueError):
            rotate_points_angle(np.array([1.0, 2.0, 3.0]), 0.5)

    def test_normalize_angle_full_range(self):
        a = normalize_angle(3 * math.pi)
        assert 0.0 <= a < 2 * math.pi

    def test_normalize_angle_half_range(self):
        a = normalize_angle(3 * math.pi, half_range=True)
        assert -math.pi < a <= math.pi

    def test_normalize_angle_zero(self):
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_angle_difference_same_angle(self):
        assert angle_difference(1.0, 1.0) == pytest.approx(0.0)

    def test_angle_difference_opposite(self):
        d = angle_difference(0.0, math.pi)
        assert d == pytest.approx(math.pi)

    def test_angle_difference_nonnegative(self):
        d = angle_difference(0.1, 5.9)
        assert d >= 0.0
        assert d <= math.pi

    def test_nearest_discrete_exact(self):
        candidates = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        nd = nearest_discrete(0.0, candidates)
        assert nd == 0.0

    def test_nearest_discrete_picks_closest(self):
        candidates = [0.0, math.pi]
        nd = nearest_discrete(0.1, candidates)
        assert nd == 0.0

    def test_nearest_discrete_empty_raises(self):
        with pytest.raises(ValueError):
            nearest_discrete(0.5, [])

    def test_angles_to_matrix_shape(self):
        angles = np.array([0.0, math.pi / 2, math.pi])
        mats = angles_to_matrix(angles)
        assert mats.shape == (3, 2, 2)

    def test_angles_to_matrix_identity_at_zero(self):
        mats = angles_to_matrix(np.array([0.0]))
        np.testing.assert_allclose(mats[0], np.eye(2), atol=1e-12)

    def test_angles_to_matrix_invalid_ndim(self):
        with pytest.raises(ValueError):
            angles_to_matrix(np.array([[0.0, 1.0]]))

    def test_batch_rotate_images_length_mismatch(self):
        imgs = [np.zeros((10, 10), dtype=np.uint8)]
        with pytest.raises(ValueError):
            batch_rotate_images(imgs, [0.0, 90.0])

    def test_batch_rotate_images_correct(self):
        imgs = [np.zeros((10, 10), dtype=np.uint8) for _ in range(3)]
        rotated = batch_rotate_images(imgs, [0.0, 90.0, 180.0])
        assert len(rotated) == 3

    def test_estimate_rotation_identity(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        angle = estimate_rotation(pts, pts)
        assert abs(angle) < 1e-10

    def test_estimate_rotation_90_deg(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        # Rotate pts by 90 degrees CCW
        rotated = rotate_points_angle(pts, math.pi / 2)
        angle = estimate_rotation(pts, rotated)
        assert abs(angle - math.pi / 2) < 1e-9

    def test_estimate_rotation_invalid(self):
        with pytest.raises(ValueError):
            estimate_rotation(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))


# ==============================================================================
# 5. sampling_utils
# ==============================================================================

class TestSamplingUtils:

    def test_uniform_sample_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = uniform_sample(2.0, 5.0, rng)
            assert 2.0 <= v <= 5.0

    def test_uniform_sample_equal_bounds(self):
        v = uniform_sample(3.0, 3.0)
        assert v == 3.0

    def test_uniform_sample_invalid(self):
        with pytest.raises(ValueError):
            uniform_sample(5.0, 2.0)

    def test_sample_angle_returns_valid(self):
        cfg = SamplingConfig(angles_deg=[0.0, 90.0, 180.0, 270.0])
        rng = np.random.default_rng(1)
        allowed_rads = {math.radians(d) for d in cfg.angles_deg}
        for _ in range(20):
            a = sample_angle(cfg, rng)
            assert any(abs(a - r) < 1e-10 for r in allowed_rads)

    def test_sample_position_in_bounds(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            x, y = sample_position(100, 200, rng)
            assert 0.0 <= x < 100.0
            assert 0.0 <= y < 200.0

    def test_sample_position_invalid_width(self):
        with pytest.raises(ValueError):
            sample_position(0, 10)

    def test_sample_positions_grid_count(self):
        cfg = SamplingConfig(grid_step=10)
        positions = sample_positions_grid(100, 50, cfg)
        expected = (100 // 10) * (50 // 10)
        assert len(positions) == expected

    def test_sample_positions_grid_values(self):
        cfg = SamplingConfig(grid_step=5)
        positions = sample_positions_grid(20, 10, cfg)
        assert all(0 <= x < 20 and 0 <= y < 10 for x, y in positions)

    def test_sample_permutation_length(self):
        rng = np.random.default_rng(3)
        perm = sample_permutation(8, rng)
        assert len(perm) == 8
        assert sorted(perm) == list(range(8))

    def test_sample_permutation_invalid(self):
        with pytest.raises(ValueError):
            sample_permutation(0)

    def test_weighted_sample_deterministic(self):
        rng = np.random.default_rng(4)
        weights = np.array([0.0, 0.0, 1.0])
        idx = weighted_sample(weights, rng)
        assert idx == 2

    def test_weighted_sample_zero_sum_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([0.0, 0.0, 0.0]))

    def test_weighted_sample_negative_raises(self):
        with pytest.raises(ValueError):
            weighted_sample(np.array([1.0, -0.5, 0.5]))

    def test_acceptance_probability_improvement(self):
        p = acceptance_probability(-1.0, 1.0)
        assert p == 1.0

    def test_acceptance_probability_worsening(self):
        p = acceptance_probability(1.0, 1.0)
        assert 0.0 < p < 1.0
        assert p == pytest.approx(math.exp(-1.0))

    def test_acceptance_probability_invalid_temp(self):
        with pytest.raises(ValueError):
            acceptance_probability(1.0, 0.0)

    def test_sample_swap_pair_distinct(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            i, j = sample_swap_pair(10, rng)
            assert i != j
            assert 0 <= i < 10 and 0 <= j < 10

    def test_sample_swap_pair_invalid(self):
        with pytest.raises(ValueError):
            sample_swap_pair(1)

    def test_batch_uniform_sample_shape(self):
        rng = np.random.default_rng(6)
        arr = batch_uniform_sample(0.0, 1.0, 50, rng)
        assert arr.shape == (50,)
        assert np.all((arr >= 0.0) & (arr <= 1.0))

    def test_batch_uniform_sample_constant(self):
        arr = batch_uniform_sample(3.0, 3.0, 5)
        np.testing.assert_array_equal(arr, np.full(5, 3.0))


# ==============================================================================
# 6. score_aggregator
# ==============================================================================

class TestScoreAggregator:

    def _sv(self, a=0, b=1, ch=None, wt=None):
        ch = ch or {"color": 0.8, "texture": 0.6}
        return ScoreVector(idx_a=a, idx_b=b, channels=ch, weights=wt or {})

    def test_score_vector_properties(self):
        sv = self._sv()
        assert sv.n_channels == 2
        assert sv.pair_key == (0, 1)
        assert sv.mean_score == pytest.approx(0.7)
        assert sv.max_score == 0.8
        assert sv.min_score == 0.6

    def test_score_vector_invalid_channel(self):
        with pytest.raises(ValueError):
            ScoreVector(idx_a=0, idx_b=1, channels={"c": 1.5})

    def test_weighted_sum_equal_weights(self):
        ch = {"a": 0.4, "b": 0.6}
        s = weighted_sum(ch)
        assert s == pytest.approx(0.5, abs=0.01)

    def test_weighted_sum_custom_weights(self):
        ch = {"a": 0.0, "b": 1.0}
        wt = {"a": 0.0, "b": 1.0}
        s = weighted_sum(ch, wt)
        assert s == pytest.approx(1.0)

    def test_weighted_sum_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_sum({})

    def test_harmonic_mean_basic(self):
        ch = {"a": 1.0, "b": 1.0}
        assert harmonic_mean(ch) == pytest.approx(1.0)

    def test_harmonic_mean_empty_raises(self):
        with pytest.raises(ValueError):
            harmonic_mean({})

    def test_geometric_mean_basic(self):
        ch = {"a": 1.0, "b": 1.0}
        assert geometric_mean(ch) == pytest.approx(1.0)

    def test_geometric_mean_zero_channel(self):
        ch = {"a": 0.0, "b": 1.0}
        gm = geometric_mean(ch)
        assert 0.0 <= gm <= 1.0

    def test_aggregate_pair_weighted(self):
        sv = self._sv()
        sc = aggregate_pair(sv, AggregationMethod.WEIGHTED)
        assert 0.0 <= sc <= 1.0

    def test_aggregate_pair_harmonic(self):
        sv = self._sv()
        sc = aggregate_pair(sv, AggregationMethod.HARMONIC)
        assert 0.0 <= sc <= 1.0

    def test_aggregate_pair_geometric(self):
        sv = self._sv()
        sc = aggregate_pair(sv, AggregationMethod.GEOMETRIC)
        assert 0.0 <= sc <= 1.0

    def test_aggregate_pair_min(self):
        sv = self._sv()
        sc = aggregate_pair(sv, AggregationMethod.MIN)
        assert sc == pytest.approx(0.6)

    def test_aggregate_pair_max(self):
        sv = self._sv()
        sc = aggregate_pair(sv, AggregationMethod.MAX)
        assert sc == pytest.approx(0.8)

    def test_aggregate_matrix_shape(self):
        vectors = [self._sv(0, 1), self._sv(1, 2), self._sv(0, 2)]
        mat = aggregate_matrix(vectors, n_fragments=3)
        assert mat.shape == (3, 3)

    def test_aggregate_matrix_symmetry(self):
        vectors = [self._sv(0, 1), self._sv(1, 2)]
        mat = aggregate_matrix(vectors, n_fragments=3)
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_batch_aggregate_basic(self):
        vectors = [self._sv(0, 1), self._sv(1, 2), self._sv(0, 2)]
        result = batch_aggregate(vectors)
        assert isinstance(result, AggregationResult)
        assert result.n_pairs == 3

    def test_batch_aggregate_empty(self):
        result = batch_aggregate([])
        assert result.n_pairs == 0
        assert result.top_pair is None

    def test_top_k_pairs(self):
        vectors = [
            self._sv(0, 1, ch={"c": 0.9}),
            self._sv(1, 2, ch={"c": 0.5}),
            self._sv(0, 2, ch={"c": 0.7}),
        ]
        result = batch_aggregate(vectors)
        top = top_k_pairs(result, k=2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]

    def test_aggregation_result_get_score(self):
        vectors = [self._sv(0, 1, ch={"c": 0.8})]
        result = batch_aggregate(vectors)
        sc = result.get_score(0, 1)
        assert sc is not None
        assert 0.0 <= sc <= 1.0


# ==============================================================================
# 7. score_matrix_utils
# ==============================================================================

class TestScoreMatrixUtils:

    def _matrix(self, n: int = 4) -> np.ndarray:
        rng = np.random.default_rng(99)
        m = rng.random((n, n)).astype(np.float32)
        np.fill_diagonal(m, 0.0)
        return m

    def test_zero_diagonal(self):
        m = self._matrix()
        zd = zero_diagonal(m)
        assert np.all(np.diag(zd) == 0.0)

    def test_zero_diagonal_off_unchanged(self):
        m = self._matrix()
        original_off = m.copy()
        np.fill_diagonal(original_off, 0.0)
        zd = zero_diagonal(m)
        np.fill_diagonal(zd, 0.0)
        np.testing.assert_array_almost_equal(zd, original_off)

    def test_symmetrize(self):
        m = self._matrix()
        sym = symmetrize(m)
        np.testing.assert_array_almost_equal(sym, sym.T)

    def test_threshold_matrix(self):
        # threshold_matrix zeros entries <= threshold
        m = np.array([[0.1, 0.5], [0.3, 0.8]], dtype=float)
        t = threshold_matrix(m, 0.4)
        assert t[0, 0] == 0.0   # 0.1 <= 0.4 → zeroed
        assert t[1, 0] == 0.0   # 0.3 <= 0.4 → zeroed
        assert t[0, 1] == 0.5   # 0.5 > 0.4 → kept
        assert t[1, 1] == 0.8   # 0.8 > 0.4 → kept

    def test_normalize_rows_sums_to_one(self):
        m = self._matrix(3)
        nr = normalize_rows(m)
        row_sums = nr.sum(axis=1)
        # Rows that were all-zero remain zero; others sum to 1
        for i, rs in enumerate(row_sums):
            if m[i].sum() > 1e-10:
                assert rs == pytest.approx(1.0, abs=1e-6)

    def test_top_k_indices_basic(self):
        row = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        idx = top_k_indices(row, k=2)
        assert list(idx) == [3, 1]

    def test_top_k_indices_zero_k(self):
        row = np.array([0.5, 0.3])
        idx = top_k_indices(row, k=0)
        assert len(idx) == 0

    def test_matrix_stats_basic(self):
        m = self._matrix(5)
        stats = matrix_stats(m)
        assert isinstance(stats, MatrixStats)
        assert stats.n_edges == 5
        assert 0.0 <= stats.sparsity <= 1.0

    def test_matrix_stats_top_pair(self):
        m = np.zeros((4, 4), dtype=float)
        m[1, 2] = 0.99
        m[2, 1] = 0.99
        stats = matrix_stats(m)
        assert set(stats.top_pair) == {1, 2}

    def test_top_k_per_row_shape(self):
        m = self._matrix(4)
        result = top_k_per_row(m, k=2)
        assert len(result) == 4
        for entries in result:
            assert len(entries) <= 2

    def test_top_k_per_row_exclude_self(self):
        m = self._matrix(4)
        result = top_k_per_row(m, k=3, exclude_self=True)
        for i, entries in enumerate(result):
            assert all(e.idx != i for e in entries)

    def test_filter_by_threshold(self):
        m = np.array([[0.0, 0.8], [0.3, 0.0]], dtype=float)
        filtered, pairs = filter_by_threshold(m, threshold=0.5)
        assert all(s > 0.5 for _, _, s in pairs)

    def test_intra_fragment_mask(self):
        mask = intra_fragment_mask([2, 3])
        # 5x5 mask; first 2x2 and last 3x3 blocks should be True
        assert mask[0, 1]  # within fragment 1
        assert mask[2, 4]  # within fragment 2
        assert not mask[0, 2]  # cross-fragment

    def test_apply_intra_fragment_mask(self):
        m = np.ones((5, 5), dtype=float)
        result = apply_intra_fragment_mask(m, [2, 3])
        # intra positions should be zeroed
        assert result[0, 0] == 0.0
        assert result[2, 2] == 0.0

    def test_batch_matrix_stats(self):
        matrices = [self._matrix(3), self._matrix(4)]
        stats_list = batch_matrix_stats(matrices)
        assert len(stats_list) == 2
        assert stats_list[0].n_edges == 3


# ==============================================================================
# 8. score_norm_utils
# ==============================================================================

class TestScoreNormUtils:

    def test_score_norm_config_invalid_method(self):
        with pytest.raises(ValueError):
            ScoreNormConfig(method="unknown")

    def test_score_norm_config_invalid_range(self):
        with pytest.raises(ValueError):
            ScoreNormConfig(feature_range=(1.0, 0.0))

    def test_make_norm_entry(self):
        e = make_norm_entry(0, 0.3, 0.7, method="minmax")
        assert e.idx == 0
        assert e.delta == pytest.approx(0.4)

    def test_make_norm_entry_invalid_idx(self):
        with pytest.raises(ValueError):
            make_norm_entry(-1, 0.3, 0.7)

    def test_entries_from_scores(self):
        orig = [0.1, 0.5, 0.9]
        norm = [0.0, 0.5, 1.0]
        entries = entries_from_scores(orig, norm)
        assert len(entries) == 3
        assert entries[0].original_score == 0.1

    def test_entries_from_scores_length_mismatch(self):
        with pytest.raises(ValueError):
            entries_from_scores([0.1, 0.5], [0.0])

    def test_summarise_norm_basic(self):
        entries = entries_from_scores([0.2, 0.8], [0.0, 1.0])
        s = summarise_norm(entries)
        assert s.n_total == 2
        assert s.original_min == pytest.approx(0.2)
        assert s.normalized_max == pytest.approx(1.0)

    def test_summarise_norm_empty(self):
        s = summarise_norm([])
        assert s.n_total == 0

    def test_filter_by_normalized_range(self):
        entries = entries_from_scores([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
        filtered = filter_by_normalized_range(entries, 0.4, 0.6)
        assert all(0.4 <= e.normalized_score <= 0.6 for e in filtered)

    def test_filter_by_original_range(self):
        entries = entries_from_scores([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])
        filtered = filter_by_original_range(entries, 0.4, 1.0)
        assert all(e.original_score >= 0.4 for e in filtered)

    def test_top_k_norm_entries(self):
        entries = entries_from_scores(list(range(10)), list(range(10)))
        top = top_k_norm_entries(entries, k=3)
        assert len(top) == 3
        assert top[0].normalized_score >= top[1].normalized_score

    def test_top_k_norm_entries_invalid_k(self):
        entries = entries_from_scores([0.5], [0.5])
        with pytest.raises(ValueError):
            top_k_norm_entries(entries, k=0)

    def test_norm_entry_stats(self):
        entries = entries_from_scores([0.0, 1.0], [0.0, 1.0])
        stats = norm_entry_stats(entries)
        assert stats["n"] == 2
        assert stats["mean_original"] == pytest.approx(0.5)

    def test_norm_entry_stats_empty(self):
        stats = norm_entry_stats([])
        assert stats["n"] == 0

    def test_compare_norm_summaries(self):
        s1 = summarise_norm(entries_from_scores([0.0, 1.0], [0.0, 1.0]))
        s2 = summarise_norm(entries_from_scores([0.2, 0.8], [0.1, 0.9]))
        cmp = compare_norm_summaries(s1, s2)
        assert "original_min_delta" in cmp

    def test_batch_summarise_norm(self):
        pairs = [([0.0, 1.0], [0.0, 1.0]), ([0.5, 0.5], [0.5, 0.5])]
        summaries = batch_summarise_norm(pairs)
        assert len(summaries) == 2


# ==============================================================================
# 9. score_seam_utils
# ==============================================================================

class TestScoreSeamUtils:

    def test_normalization_record_valid(self):
        rec = NormalizationRecord(method="minmax", n_scores=10, min_val=0.0, max_val=1.0)
        assert rec.value_range == pytest.approx(1.0)

    def test_normalization_record_invalid_method(self):
        with pytest.raises(ValueError):
            NormalizationRecord(method="bogus", n_scores=5)

    def test_normalization_record_invalid_n_scores(self):
        with pytest.raises(ValueError):
            NormalizationRecord(method="minmax", n_scores=-1)

    def test_score_calibration_record_identity(self):
        rec = ScoreCalibrationRecord(n_reference=100, n_target=50)
        assert rec.is_identity

    def test_score_calibration_record_non_identity(self):
        rec = ScoreCalibrationRecord(n_reference=100, n_target=50, shift=0.1)
        assert not rec.is_identity

    def test_gradient_run_record_has_data(self):
        rec = GradientRunRecord(n_images=5, kernel="sobel", ksize=3)
        assert rec.has_data

    def test_gradient_run_record_invalid(self):
        with pytest.raises(ValueError):
            GradientRunRecord(n_images=-1, kernel="sobel", ksize=3)

    def test_orientation_histogram_dominant_bin(self):
        hist = [0.1, 0.6, 0.3]
        rec = OrientationHistogramRecord(n_bins=3, histogram=hist)
        assert rec.dominant_bin == 1

    def test_orientation_histogram_empty(self):
        rec = OrientationHistogramRecord(n_bins=4)
        assert rec.dominant_bin is None
        assert rec.is_normalized

    def test_orientation_histogram_invalid_length(self):
        with pytest.raises(ValueError):
            OrientationHistogramRecord(n_bins=3, histogram=[0.5, 0.5])

    def test_seam_run_record_has_pairs(self):
        rec = SeamRunRecord(n_pairs=3, mean_quality=0.7)
        assert rec.has_pairs

    def test_seam_run_record_invalid_quality(self):
        with pytest.raises(ValueError):
            SeamRunRecord(n_pairs=1, mean_quality=1.5)

    def test_seam_score_matrix_get(self):
        sm = SeamScoreMatrix(n_fragments=3, scores={(0, 1): 0.8, (1, 2): 0.6})
        assert sm.get(0, 1) == 0.8
        assert sm.get(1, 0) == 0.8  # mirror
        assert sm.get(0, 2) == 0.0  # default

    def test_seam_score_matrix_n_scored_pairs(self):
        sm = SeamScoreMatrix(n_fragments=4, scores={(0, 1): 0.5, (2, 3): 0.7})
        assert sm.n_scored_pairs == 2

    def test_aggregation_run_record_invalid_method(self):
        with pytest.raises(ValueError):
            AggregationRunRecord(method="bad", n_items=5, n_channels=3)

    def test_aggregation_run_record_is_empty(self):
        rec = AggregationRunRecord(method="weighted_avg", n_items=0, n_channels=2)
        assert rec.is_empty

    def test_make_normalization_record(self):
        scores = [0.1, 0.5, 0.9]
        rec = make_normalization_record("zscore", scores, label="test")
        assert rec.n_scores == 3
        assert rec.min_val == pytest.approx(0.1)
        assert rec.max_val == pytest.approx(0.9)

    def test_make_seam_run_record(self):
        qualities = [0.4, 0.6, 0.8]
        rec = make_seam_run_record(qualities, label="batch1")
        assert rec.n_pairs == 3
        assert rec.mean_quality == pytest.approx(0.6)

    def test_make_seam_run_record_empty(self):
        rec = make_seam_run_record([])
        assert rec.n_pairs == 0


# ==============================================================================
# 10. scoring_pipeline_utils
# ==============================================================================

class TestScoringPipelineUtils:

    def test_stage_result_valid(self):
        sr = StageResult(stage_name="color", score=0.8, weight=2.0)
        assert sr.weighted_score == pytest.approx(1.6)

    def test_stage_result_invalid_score(self):
        with pytest.raises(ValueError):
            StageResult(stage_name="x", score=1.5)

    def test_stage_result_invalid_weight(self):
        with pytest.raises(ValueError):
            StageResult(stage_name="x", score=0.5, weight=-1.0)

    def test_stage_result_empty_name(self):
        with pytest.raises(ValueError):
            StageResult(stage_name="", score=0.5)

    def test_pipeline_report_add_and_score(self):
        report = PipelineReport()
        report.add_stage(StageResult("color", 0.8, 1.0))
        report.add_stage(StageResult("texture", 0.6, 1.0))
        assert report.n_stages == 2
        assert report.weighted_score == pytest.approx(0.7)

    def test_pipeline_report_empty(self):
        report = PipelineReport()
        assert report.weighted_score == 0.0
        assert report.total_weight == 0.0

    def test_pipeline_report_stage_by_name(self):
        report = PipelineReport()
        report.add_stage(StageResult("color", 0.9))
        found = report.stage_by_name("color")
        assert found is not None
        assert found.score == 0.9

    def test_pipeline_report_stage_by_name_missing(self):
        report = PipelineReport()
        assert report.stage_by_name("missing") is None

    def test_pipeline_report_to_dict(self):
        report = PipelineReport()
        report.add_stage(StageResult("edge", 0.5, 0.5))
        d = report.to_dict()
        assert d["n_stages"] == 1
        assert "stages" in d

    def test_build_pipeline_report(self):
        stages = [StageResult("a", 0.3), StageResult("b", 0.7)]
        report = build_pipeline_report(stages)
        assert report.n_stages == 2

    def test_boundary_score_record_valid(self):
        rec = BoundaryScoreRecord(0, 1, 0.8, 0.7, 0.6, 0.7)
        assert rec.total_score == 0.7

    def test_boundary_score_record_invalid_idx(self):
        with pytest.raises(ValueError):
            BoundaryScoreRecord(-1, 0, 0.5, 0.5, 0.5, 0.5)

    def test_summarize_boundary_scores_empty(self):
        s = summarize_boundary_scores([])
        assert s["n_pairs"] == 0

    def test_summarize_boundary_scores(self):
        records = [
            BoundaryScoreRecord(0, 1, 0.8, 0.7, 0.6, 0.7),
            BoundaryScoreRecord(1, 2, 0.9, 0.8, 0.7, 0.8),
        ]
        s = summarize_boundary_scores(records)
        assert s["n_pairs"] == 2
        assert s["mean_total"] == pytest.approx(0.75)

    def test_rank_stage_results(self):
        stages = [StageResult("a", 0.3), StageResult("b", 0.9), StageResult("c", 0.6)]
        ranked = rank_stage_results(stages)
        ranks, results = zip(*ranked)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert list(ranks) == [1, 2, 3]

    def test_patch_comparison_record_valid(self):
        rec = PatchComparisonRecord(row=0, col=1, method="ncc", value=0.85)
        assert rec.value == 0.85

    def test_patch_comparison_record_invalid(self):
        with pytest.raises(ValueError):
            PatchComparisonRecord(row=-1, col=0, method="ncc", value=0.5)


# ==============================================================================
# 11. segment_utils
# ==============================================================================

class TestSegmentUtils:

    def _simple_mask(self) -> np.ndarray:
        m = np.zeros((20, 20), dtype=np.uint8)
        m[5:15, 5:15] = 255  # single 10x10 block
        return m

    def test_label_mask_single_component(self):
        labels, n = label_mask(self._simple_mask())
        assert n == 1
        assert labels[10, 10] == 1

    def test_label_mask_two_components(self):
        m = np.zeros((20, 30), dtype=np.uint8)
        m[2:8, 2:8] = 255
        m[2:8, 22:28] = 255
        labels, n = label_mask(m)
        assert n == 2

    def test_label_mask_empty(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        labels, n = label_mask(m)
        assert n == 0

    def test_region_info_basic(self):
        labels, _ = label_mask(self._simple_mask())
        ri = region_info(labels, 1)
        assert ri.area == 100
        assert ri.label == 1

    def test_region_info_height_width(self):
        labels, _ = label_mask(self._simple_mask())
        ri = region_info(labels, 1)
        assert ri.height == 10
        assert ri.width == 10

    def test_region_info_nonexistent_label(self):
        labels, _ = label_mask(self._simple_mask())
        ri = region_info(labels, 99)
        assert ri.area == 0

    def test_all_regions_count(self):
        m = np.zeros((20, 30), dtype=np.uint8)
        m[2:8, 2:8] = 255
        m[12:18, 22:28] = 255
        labels, n = label_mask(m)
        regions = all_regions(labels, n)
        assert len(regions) == 2

    def test_filter_regions_min_area(self):
        m = np.zeros((20, 30), dtype=np.uint8)
        m[0:2, 0:2] = 255  # tiny
        m[5:15, 5:15] = 255  # large
        labels, n = label_mask(m)
        regions = all_regions(labels, n)
        cfg = SegmentConfig(min_area=50)
        filtered = filter_regions(regions, cfg)
        assert all(r.area >= 50 for r in filtered)

    def test_largest_region(self):
        m = np.zeros((30, 30), dtype=np.uint8)
        m[0:5, 0:5] = 255
        m[10:25, 10:25] = 255
        labels, n = label_mask(m)
        regions = all_regions(labels, n)
        largest = largest_region(regions)
        assert largest is not None
        assert largest.area == max(r.area for r in regions)

    def test_largest_region_empty(self):
        assert largest_region([]) is None

    def test_mask_from_labels(self):
        labels, n = label_mask(self._simple_mask())
        result = mask_from_labels(labels, [1])
        assert int(result.max()) == 255
        assert int(result[0, 0]) == 0

    def test_mask_statistics(self):
        m = self._simple_mask()
        stats = mask_statistics(m)
        assert stats["foreground_pixels"] == 100
        assert stats["total_pixels"] == 400
        assert stats["foreground_fraction"] == pytest.approx(0.25)

    def test_mask_bounding_box(self):
        m = self._simple_mask()
        bb = mask_bounding_box(m)
        assert bb == (5, 5, 15, 15)

    def test_mask_bounding_box_empty(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        assert mask_bounding_box(m) is None

    def test_extract_boundary_shape(self):
        m = self._simple_mask()
        boundary = extract_boundary(m)
        assert boundary.shape == m.shape
        assert boundary.dtype == np.uint8

    def test_segment_config_invalid(self):
        with pytest.raises(ValueError):
            SegmentConfig(min_area=-1)


# ==============================================================================
# 12. seq_gap_utils
# ==============================================================================

class TestSeqGapUtils:

    def _seq_entries(self, n: int = 5) -> list:
        return [
            make_sequence_score_entry(
                seq_id=i, order=list(range(4)), total_score=float(i) * 0.2,
                n_fragments=4, algorithm="greedy" if i % 2 == 0 else "beam",
            )
            for i in range(n)
        ]

    def test_make_sequence_score_entry(self):
        e = make_sequence_score_entry(0, [0, 1, 2], 0.75, 3)
        assert e.seq_id == 0
        assert e.total_score == 0.75

    def test_summarise_sequence_score_entries_empty(self):
        s = summarise_sequence_score_entries([])
        assert s.n_entries == 0

    def test_summarise_sequence_score_entries_basic(self):
        entries = self._seq_entries(5)
        s = summarise_sequence_score_entries(entries)
        assert s.n_entries == 5
        assert s.min_score == 0.0
        assert s.max_score == pytest.approx(0.8)

    def test_filter_full_sequences(self):
        entries = self._seq_entries(4)
        entries[1] = make_sequence_score_entry(1, [0, 1], 0.5, 2, is_full=False)
        full = filter_full_sequences(entries)
        assert all(e.is_full for e in full)

    def test_filter_sequence_by_min_score(self):
        entries = self._seq_entries(5)
        filtered = filter_sequence_by_min_score(entries, 0.4)
        assert all(e.total_score >= 0.4 for e in filtered)

    def test_filter_sequence_by_algorithm(self):
        entries = self._seq_entries(6)
        filtered = filter_sequence_by_algorithm(entries, "beam")
        assert all(e.algorithm == "beam" for e in filtered)

    def test_top_k_sequence_entries(self):
        entries = self._seq_entries(5)
        top = top_k_sequence_entries(entries, k=2)
        assert len(top) == 2
        assert top[0].total_score >= top[1].total_score

    def test_best_sequence_entry(self):
        entries = self._seq_entries(5)
        best = best_sequence_entry(entries)
        assert best is not None
        assert best.total_score == max(e.total_score for e in entries)

    def test_sequence_score_stats(self):
        entries = self._seq_entries(4)
        stats = sequence_score_stats(entries)
        assert stats["count"] == 4.0

    def test_compare_sequence_summaries(self):
        s1 = summarise_sequence_score_entries(self._seq_entries(3))
        s2 = summarise_sequence_score_entries(self._seq_entries(5))
        cmp = compare_sequence_summaries(s1, s2)
        assert "mean_score_delta" in cmp

    def test_batch_summarise_sequence_score_entries(self):
        groups = [self._seq_entries(2), self._seq_entries(3)]
        summaries = batch_summarise_sequence_score_entries(groups)
        assert len(summaries) == 2

    # Gap entries
    def _gap_entries(self, n: int = 5) -> list:
        categories = ["near", "far", "overlap", "touching", "near"]
        rng = np.random.default_rng(22)
        return [
            make_gap_score_entry(
                id1=i, id2=i + 1,
                gap_x=float(rng.uniform(-5, 5)),
                gap_y=float(rng.uniform(-5, 5)),
                distance=float(rng.uniform(0, 20)),
                category=categories[i % len(categories)],
            )
            for i in range(n)
        ]

    def test_make_gap_score_entry(self):
        e = make_gap_score_entry(0, 1, 2.0, 3.0, 5.0, "near")
        assert e.distance == 5.0
        assert e.category == "near"

    def test_summarise_gap_score_entries_empty(self):
        s = summarise_gap_score_entries([])
        assert s.n_entries == 0

    def test_summarise_gap_score_entries(self):
        entries = self._gap_entries(5)
        s = summarise_gap_score_entries(entries)
        assert s.n_entries == 5
        assert s.min_distance <= s.mean_distance <= s.max_distance

    def test_filter_overlapping_gaps(self):
        entries = self._gap_entries(5)
        overlaps = filter_overlapping_gaps(entries)
        assert all(e.category == "overlap" for e in overlaps)

    def test_filter_gap_by_category(self):
        entries = self._gap_entries(5)
        near = filter_gap_by_category(entries, "near")
        assert all(e.category == "near" for e in near)

    def test_filter_gap_by_max_distance(self):
        entries = self._gap_entries(10)
        filtered = filter_gap_by_max_distance(entries, 10.0)
        assert all(e.distance <= 10.0 for e in filtered)

    def test_top_k_closest_gaps(self):
        entries = self._gap_entries(5)
        top = top_k_closest_gaps(entries, k=2)
        assert len(top) == 2
        assert top[0].distance <= top[1].distance

    def test_best_gap_entry(self):
        entries = self._gap_entries(5)
        best = best_gap_entry(entries)
        assert best is not None
        assert best.distance == min(e.distance for e in entries)

    def test_gap_score_stats(self):
        entries = self._gap_entries(4)
        stats = gap_score_stats(entries)
        assert stats["count"] == 4.0

    def test_compare_gap_summaries(self):
        s1 = summarise_gap_score_entries(self._gap_entries(3))
        s2 = summarise_gap_score_entries(self._gap_entries(5))
        cmp = compare_gap_summaries(s1, s2)
        assert "mean_distance_delta" in cmp

    def test_batch_summarise_gap_score_entries(self):
        groups = [self._gap_entries(2), self._gap_entries(4)]
        summaries = batch_summarise_gap_score_entries(groups)
        assert len(summaries) == 2


# ==============================================================================
# 13. sequence_utils
# ==============================================================================

class TestSequenceUtils:

    def test_rank_sequence_basic(self):
        seq = np.array([3.0, 1.0, 2.0])
        ranks = rank_sequence(seq)
        assert list(ranks) == [3.0, 1.0, 2.0]

    def test_rank_sequence_ties(self):
        seq = np.array([1.0, 1.0, 3.0])
        ranks = rank_sequence(seq)
        assert ranks[0] == pytest.approx(ranks[1])  # tied at 1.5
        assert ranks[2] == 3.0

    def test_rank_sequence_empty_raises(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([]))

    def test_rank_sequence_non1d_raises(self):
        with pytest.raises(ValueError):
            rank_sequence(np.array([[1, 2], [3, 4]]))

    def test_normalize_sequence_range(self):
        seq = np.array([1.0, 3.0, 5.0, 7.0])
        norm = normalize_sequence(seq)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_normalize_sequence_constant(self):
        seq = np.array([5.0, 5.0, 5.0])
        norm = normalize_sequence(seq)
        np.testing.assert_array_equal(norm, np.zeros(3))

    def test_normalize_sequence_empty_raises(self):
        with pytest.raises(ValueError):
            normalize_sequence(np.array([]))

    def test_invert_sequence(self):
        seq = np.array([0.0, 0.5, 1.0])
        inv = invert_sequence(seq)
        np.testing.assert_allclose(inv, [1.0, 0.5, 0.0])

    def test_invert_sequence_empty_raises(self):
        with pytest.raises(ValueError):
            invert_sequence(np.array([]))

    def test_sliding_scores_mean(self):
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg = SequenceConfig(window=3, agg="mean")
        result = sliding_scores(seq, cfg)
        assert len(result) == len(seq)
        assert result[2] == pytest.approx(3.0)

    def test_sliding_scores_max(self):
        seq = np.array([1.0, 3.0, 2.0])
        cfg = SequenceConfig(window=3, agg="max")
        result = sliding_scores(seq, cfg)
        assert result[1] == pytest.approx(3.0)

    def test_sliding_scores_sum(self):
        seq = np.ones(5)
        cfg = SequenceConfig(window=3, agg="sum")
        result = sliding_scores(seq, cfg)
        # center should be 3.0
        assert result[2] == pytest.approx(3.0)

    def test_align_sequences_same_length(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        aa, bb = align_sequences(a, b)
        np.testing.assert_array_equal(aa, a)
        np.testing.assert_array_equal(bb, b)

    def test_align_sequences_different_lengths(self):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 5.0, 6.0, 7.0])
        aa, bb = align_sequences(a, b)
        assert len(aa) == 4
        assert len(bb) == 4

    def test_align_sequences_custom_target_len(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        aa, bb = align_sequences(a, b, target_len=5)
        assert len(aa) == 5
        assert len(bb) == 5

    def test_kendall_tau_distance_identical(self):
        perm = np.array([0, 1, 2, 3])
        assert kendall_tau_distance(perm, perm) == 0

    def test_kendall_tau_distance_reversed(self):
        perm_a = np.array([0, 1, 2, 3])
        perm_b = np.array([3, 2, 1, 0])
        d = kendall_tau_distance(perm_a, perm_b)
        assert d == 6  # n*(n-1)/2 for n=4

    def test_kendall_tau_distance_length_mismatch(self):
        with pytest.raises(ValueError):
            kendall_tau_distance(np.array([0, 1]), np.array([0, 1, 2]))

    def test_longest_increasing_basic(self):
        seq = np.array([3.0, 1.0, 2.0, 4.0])
        assert longest_increasing(seq) == 3  # 1, 2, 4

    def test_longest_increasing_all_same(self):
        seq = np.array([5.0, 5.0, 5.0])
        assert longest_increasing(seq) == 1

    def test_longest_increasing_empty(self):
        assert longest_increasing(np.array([])) == 0

    def test_segment_by_threshold_basic(self):
        seq = np.array([0.0, 0.8, 0.9, 0.1, 0.7, 0.8])
        cfg = SequenceConfig(threshold=0.5)
        segs = segment_by_threshold(seq, cfg)
        assert len(segs) >= 1
        for start, end in segs:
            assert all(seq[i] >= 0.5 for i in range(start, end + 1))

    def test_segment_by_threshold_no_segments(self):
        seq = np.array([0.1, 0.2, 0.3])
        cfg = SequenceConfig(threshold=0.9)
        segs = segment_by_threshold(seq, cfg)
        assert segs == []

    def test_batch_rank_basic(self):
        seqs = [np.array([3.0, 1.0, 2.0]), np.array([5.0, 6.0, 4.0])]
        ranked = batch_rank(seqs)
        assert len(ranked) == 2
        assert list(ranked[0]) == [3.0, 1.0, 2.0]

    def test_batch_rank_empty_raises(self):
        with pytest.raises(ValueError):
            batch_rank([])

    def test_sequence_config_invalid_window(self):
        with pytest.raises(ValueError):
            SequenceConfig(window=0)

    def test_sequence_config_invalid_agg(self):
        with pytest.raises(ValueError):
            SequenceConfig(agg="median")
