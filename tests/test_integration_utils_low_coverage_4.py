"""Integration tests for under-tested utility modules batch 4.

Modules under test:
    1. puzzle_reconstruction.utils.overlap_score_utils
    2. puzzle_reconstruction.utils.pair_score_utils
    3. puzzle_reconstruction.utils.patch_score_utils
    4. puzzle_reconstruction.utils.patch_utils
    5. puzzle_reconstruction.utils.path_plan_utils
    6. puzzle_reconstruction.utils.placement_metrics_utils
    7. puzzle_reconstruction.utils.placement_score_utils
    8. puzzle_reconstruction.utils.polygon_ops_utils
    9. puzzle_reconstruction.utils.position_tracking_utils
    10. puzzle_reconstruction.utils.quality_score_utils
    11. puzzle_reconstruction.utils.rank_result_utils
    12. puzzle_reconstruction.utils.ranking_layout_utils
    13. puzzle_reconstruction.utils.ranking_validation_utils
"""
import math
import pytest
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

from puzzle_reconstruction.utils.overlap_score_utils import (
    OverlapScoreConfig,
    OverlapScoreEntry,
    OverlapSummary,
    make_overlap_entry,
    summarise_overlaps,
    filter_significant_overlaps,
    filter_by_area,
    top_k_overlaps,
    overlap_stats,
    penalty_score,
    batch_make_overlap_entries,
    group_by_fragment,
)

from puzzle_reconstruction.utils.pair_score_utils import (
    PairScoreConfig,
    PairScoreEntry,
    PairScoreSummary,
    make_pair_score_entry,
    entries_from_pair_results,
    summarise_pair_scores,
    filter_strong_pair_matches,
    filter_weak_pair_matches,
    filter_pair_by_score_range,
    filter_pair_by_channel,
    filter_pair_by_dominant_channel,
    top_k_pair_entries,
    best_pair_entry,
    pair_score_stats,
    compare_pair_summaries,
    batch_summarise_pair_scores,
)

from puzzle_reconstruction.utils.patch_score_utils import (
    PatchScoreConfig,
    PatchScoreEntry,
    PatchScoreSummary,
    make_patch_entry,
    entries_from_patch_matches,
    summarise_patch_scores,
    filter_good_patch_scores,
    filter_poor_patch_scores,
    filter_patch_by_score_range,
    filter_by_side_pair,
    filter_by_ncc_range,
    top_k_patch_entries,
    best_patch_entry,
    patch_score_stats,
    compare_patch_summaries,
    batch_summarise_patch_scores,
)

from puzzle_reconstruction.utils.patch_utils import (
    PatchConfig,
    extract_patch,
    extract_patches,
    normalize_patch,
    patch_ssd,
    patch_ncc,
    patch_mse,
    compare_patches,
    batch_compare,
)

from puzzle_reconstruction.utils.path_plan_utils import (
    PathPlanConfig,
    PathPlanEntry,
    PathPlanSummary,
    make_path_entry,
    entries_from_path_results,
    summarise_path_entries,
    filter_found_paths,
    filter_not_found_paths,
    filter_path_by_cost_range,
    filter_path_by_max_hops,
    top_k_shortest_paths,
    cheapest_path_entry,
    path_cost_stats,
    compare_path_summaries,
    batch_summarise_path_entries,
    AssemblyPlanConfig,
    AssemblyPlanEntry,
    AssemblyPlanSummary,
    make_assembly_plan_entry,
    summarise_assembly_plans,
    filter_full_coverage_plans,
    filter_assembly_plans_by_coverage,
    filter_assembly_plans_by_score,
    filter_assembly_plans_by_strategy,
    top_k_assembly_plan_entries,
    best_assembly_plan_entry,
    assembly_plan_stats,
    compare_assembly_plan_summaries,
    batch_summarise_assembly_plans,
)

from puzzle_reconstruction.utils.placement_metrics_utils import (
    PlacementMetricsConfig,
    PlacementMetrics,
    placement_density,
    bbox_of_contour,
    bbox_area,
    bbox_intersection_area,
    compute_coverage,
    compute_pairwise_overlap,
    quality_score,
    assess_placement,
    compare_metrics,
    best_of,
    normalize_metrics,
    batch_quality_scores,
)

from puzzle_reconstruction.utils.placement_score_utils import (
    PlacementScoreConfig,
    PlacementScoreEntry,
    PlacementSummary,
    make_placement_entry,
    entries_from_history,
    summarise_placement,
    filter_positive_steps,
    filter_by_min_score,
    top_k_steps,
    rank_fragments,
    placement_score_stats,
    compare_placements,
    batch_summarise,
)

from puzzle_reconstruction.utils.polygon_ops_utils import (
    PolygonOpsConfig,
    PolygonOverlapResult,
    PolygonStats,
    signed_area,
    polygon_area,
    polygon_perimeter,
    polygon_centroid,
    polygon_bounding_box,
    polygon_stats,
    point_in_polygon,
    polygon_overlap,
    remove_collinear,
    ensure_ccw,
    ensure_cw,
    polygon_similarity,
    batch_polygon_stats,
    batch_polygon_overlap,
)

from puzzle_reconstruction.utils.position_tracking_utils import (
    PositionQualityRecord,
    PositionQualitySummary,
    make_position_quality_record,
    summarise_position_quality,
    filter_by_placement_rate,
    filter_by_method,
    top_k_position_records,
    best_position_record,
    position_quality_stats,
    AssemblyHistoryEntry,
    AssemblyHistorySummary,
    make_assembly_history_entry,
    summarise_assembly_history,
    filter_converged,
    filter_by_min_best_score,
    top_k_assembly_entries,
    best_assembly_entry,
    assembly_score_stats,
    compare_assembly_summaries,
    batch_summarise_assembly_history,
)

from puzzle_reconstruction.utils.quality_score_utils import (
    QualityScoreConfig,
    QualityScoreEntry,
    QualitySummary,
    make_quality_entry,
    entries_from_reports,
    summarise_quality,
    filter_acceptable,
    filter_rejected,
    filter_by_overall,
    filter_by_blur,
    top_k_quality_entries,
    quality_score_stats,
    compare_quality,
    batch_summarise_quality,
)

from puzzle_reconstruction.utils.rank_result_utils import (
    RankResultConfig,
    RankResultEntry,
    RankResultSummary,
    make_rank_result_entry,
    entries_from_ranked_pairs,
    summarise_rank_results,
    filter_high_rank_entries,
    filter_low_rank_entries,
    filter_by_rank_position,
    filter_rank_by_score_range,
    filter_rank_by_dominant_channel,
    top_k_rank_entries,
    best_rank_entry,
    rerank_entries,
    rank_result_stats,
    compare_rank_summaries,
    batch_summarise_rank_results,
)

from puzzle_reconstruction.utils.ranking_layout_utils import (
    GlobalRankingConfig,
    GlobalRankingEntry,
    GlobalRankingSummary,
    make_global_ranking_entry,
    summarise_global_ranking_entries,
    filter_ranking_by_min_score,
    filter_ranking_by_fragment,
    filter_ranking_by_top_k,
    top_k_ranking_entries,
    best_ranking_entry,
    ranking_score_stats,
    compare_global_ranking_summaries,
    batch_summarise_global_ranking_entries,
    LayoutScoringConfig,
    LayoutScoringEntry,
    LayoutScoringSummary,
    make_layout_scoring_entry,
    summarise_layout_scoring_entries,
    filter_layout_by_min_score,
    filter_layout_by_quality,
    filter_layout_by_max_overlap,
    top_k_layout_entries,
    best_layout_entry,
    layout_score_stats,
    compare_layout_scoring_summaries,
    batch_summarise_layout_scoring_entries,
)

from puzzle_reconstruction.utils.ranking_validation_utils import (
    RankingRunRecord,
    CandidateSummary,
    ScoreVectorRecord,
    ValidationRunRecord,
    BoundaryCheckSummary,
    PaletteComparisonRecord,
    PaletteRankingRecord,
    make_ranking_record,
    make_validation_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _make_square_poly(cx: float, cy: float, half: float) -> np.ndarray:
    """Return a CCW square polygon centred at (cx, cy)."""
    return np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=float)


def _make_patch_entries(n: int = 5, good: bool = True) -> list:
    score_base = 0.6 if good else 0.3
    entries = []
    for i in range(n):
        entries.append(make_patch_entry(
            pair_id=i,
            idx1=i,
            idx2=i + 1,
            side1=0,
            side2=1,
            ncc=0.5,
            ssd=0.2,
            ssim=0.7 if good else 0.3,
            total_score=min(1.0, score_base + i * 0.05),
        ))
    return entries


def _make_overlap_entries(n: int = 5) -> list:
    return [
        make_overlap_entry(i, i + 1, iou=0.1 * (i + 1), overlap_area=float(i + 1) * 5)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. overlap_score_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlapScoreUtils:

    def test_config_defaults(self):
        cfg = OverlapScoreConfig()
        assert cfg.iou_threshold == 0.05
        assert cfg.area_threshold == 1.0
        assert cfg.penalise_self_overlap is False

    def test_config_custom_values(self):
        cfg = OverlapScoreConfig(iou_threshold=0.2, area_threshold=5.0)
        assert cfg.iou_threshold == 0.2
        assert cfg.area_threshold == 5.0

    def test_config_invalid_iou(self):
        with pytest.raises(ValueError):
            OverlapScoreConfig(iou_threshold=1.5)

    def test_config_invalid_area(self):
        with pytest.raises(ValueError):
            OverlapScoreConfig(area_threshold=-1.0)

    def test_entry_creation(self):
        e = OverlapScoreEntry(idx1=0, idx2=1, iou=0.3, overlap_area=10.0, penalty=0.3)
        assert e.pair == (0, 1)
        assert e.iou == 0.3
        assert e.overlap_area == 10.0

    def test_entry_negative_idx_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=-1, idx2=0, iou=0.1, overlap_area=5.0)

    def test_entry_invalid_iou_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=0, idx2=1, iou=1.5, overlap_area=5.0)

    def test_entry_invalid_penalty_raises(self):
        with pytest.raises(ValueError):
            OverlapScoreEntry(idx1=0, idx2=1, iou=0.5, overlap_area=5.0, penalty=2.0)

    def test_make_overlap_entry_significant(self):
        cfg = OverlapScoreConfig(iou_threshold=0.05, area_threshold=1.0)
        e = make_overlap_entry(0, 1, 0.3, 10.0, cfg=cfg)
        assert e.penalty == pytest.approx(0.3)

    def test_make_overlap_entry_not_significant(self):
        cfg = OverlapScoreConfig(iou_threshold=0.5)
        e = make_overlap_entry(0, 1, 0.1, 10.0, cfg=cfg)
        assert e.penalty == 0.0

    def test_make_overlap_entry_with_meta(self):
        e = make_overlap_entry(0, 1, 0.2, 5.0, meta={"key": "value"})
        assert e.meta == {"key": "value"}

    def test_summarise_overlaps_empty(self):
        s = summarise_overlaps([])
        assert s.n_overlaps == 0
        assert s.is_valid is True
        assert s.total_area == 0.0

    def test_summarise_overlaps_with_entries(self):
        entries = _make_overlap_entries(4)
        s = summarise_overlaps(entries)
        assert s.n_overlaps > 0
        assert s.total_area > 0.0
        assert s.max_iou > 0.0

    def test_summarise_overlaps_all_below_threshold(self):
        cfg = OverlapScoreConfig(iou_threshold=0.9)
        entries = _make_overlap_entries(5)
        s = summarise_overlaps(entries, cfg=cfg)
        assert s.is_valid is True
        assert s.n_overlaps == 0

    def test_filter_significant_overlaps(self):
        entries = _make_overlap_entries(5)
        filtered = filter_significant_overlaps(entries, iou_threshold=0.3)
        assert all(e.iou >= 0.3 for e in filtered)

    def test_filter_by_area(self):
        entries = _make_overlap_entries(5)
        filtered = filter_by_area(entries, min_area=10.0)
        assert all(e.overlap_area >= 10.0 for e in filtered)

    def test_top_k_overlaps(self):
        entries = _make_overlap_entries(5)
        top = top_k_overlaps(entries, k=2)
        assert len(top) == 2
        assert top[0].iou >= top[1].iou

    def test_top_k_overlaps_fewer_than_k(self):
        entries = _make_overlap_entries(2)
        top = top_k_overlaps(entries, k=10)
        assert len(top) == 2

    def test_overlap_stats_empty(self):
        stats = overlap_stats([])
        assert stats["n"] == 0
        assert stats["mean_iou"] == 0.0

    def test_overlap_stats_with_entries(self):
        entries = _make_overlap_entries(5)
        stats = overlap_stats(entries)
        assert stats["n"] == 5
        assert stats["max_iou"] > stats["min_iou"]
        assert stats["total_area"] > 0.0

    def test_penalty_score_empty(self):
        assert penalty_score([]) == 0.0

    def test_penalty_score_nonzero(self):
        entries = [make_overlap_entry(0, 1, 0.3, 10.0)]
        s = penalty_score(entries)
        assert 0.0 <= s <= 1.0

    def test_batch_make_overlap_entries(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        ious = [0.1, 0.2, 0.3]
        areas = [5.0, 10.0, 15.0]
        entries = batch_make_overlap_entries(pairs, ious, areas)
        assert len(entries) == 3
        assert entries[2].iou == pytest.approx(0.3)

    def test_batch_make_overlap_entries_length_mismatch(self):
        with pytest.raises(ValueError):
            batch_make_overlap_entries([(0, 1)], [0.1, 0.2], [5.0])

    def test_group_by_fragment(self):
        entries = _make_overlap_entries(4)
        groups = group_by_fragment(entries)
        assert isinstance(groups, dict)
        for idx, grp in groups.items():
            assert all(e.idx1 == idx for e in grp)

    def test_entry_repr(self):
        e = make_overlap_entry(0, 1, 0.3, 10.0)
        r = repr(e)
        assert "OverlapScoreEntry" in r

    def test_summary_repr(self):
        s = summarise_overlaps(_make_overlap_entries(3))
        r = repr(s)
        assert "OverlapSummary" in r

