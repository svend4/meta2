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

