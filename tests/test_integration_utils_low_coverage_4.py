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


# ─────────────────────────────────────────────────────────────────────────────
# 2. pair_score_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPairScoreUtils:

    def test_config_defaults(self):
        cfg = PairScoreConfig()
        assert cfg.good_threshold == 0.7
        assert cfg.poor_threshold == 0.3

    def test_config_invalid_good_threshold(self):
        with pytest.raises(ValueError):
            PairScoreConfig(good_threshold=1.5)

    def test_config_invalid_poor_threshold(self):
        with pytest.raises(ValueError):
            PairScoreConfig(poor_threshold=-0.1)

    def test_make_entry_basic(self):
        e = make_pair_score_entry(0, 1, 0.8)
        assert e.frag_i == 0
        assert e.frag_j == 1
        assert e.score == pytest.approx(0.8)

    def test_entry_pair_key_ordering(self):
        e = make_pair_score_entry(5, 2, 0.5)
        assert e.pair_key == (2, 5)

    def test_entry_dominant_channel(self):
        e = make_pair_score_entry(0, 1, 0.7, channels={"a": 0.3, "b": 0.9, "c": 0.1})
        assert e.dominant_channel == "b"

    def test_entry_no_channels_dominant_none(self):
        e = make_pair_score_entry(0, 1, 0.5)
        assert e.dominant_channel is None

    def test_entry_is_strong_match_true(self):
        e = make_pair_score_entry(0, 1, 0.75)
        assert e.is_strong_match is True

    def test_entry_is_strong_match_false(self):
        e = make_pair_score_entry(0, 1, 0.5)
        assert e.is_strong_match is False

    def test_entries_from_pair_results(self):
        pairs = [(0, 1), (1, 2), (2, 3)]
        scores = [0.5, 0.7, 0.9]
        entries = entries_from_pair_results(pairs, scores)
        assert len(entries) == 3
        assert entries[2].score == pytest.approx(0.9)

    def test_entries_from_pair_results_length_mismatch(self):
        with pytest.raises(ValueError):
            entries_from_pair_results([(0, 1)], [0.5, 0.7])

    def test_entries_from_pair_results_with_channels(self):
        pairs = [(0, 1), (1, 2)]
        scores = [0.6, 0.8]
        chs = [{"ch1": 0.5}, {"ch1": 0.9}]
        entries = entries_from_pair_results(pairs, scores, channel_lists=chs)
        assert entries[1].channels["ch1"] == pytest.approx(0.9)

    def test_summarise_empty(self):
        s = summarise_pair_scores([])
        assert s.n_entries == 0
        assert s.mean_score == 0.0

    def test_summarise_pair_scores(self):
        pairs = [(i, i+1) for i in range(10)]
        scores = [i * 0.1 for i in range(10)]
        entries = entries_from_pair_results(pairs, scores)
        s = summarise_pair_scores(entries)
        assert s.n_entries == 10
        assert s.min_score == pytest.approx(0.0)
        assert s.max_score == pytest.approx(0.9)

    def test_summarise_channel_means(self):
        chs = [{"r": 0.4, "g": 0.6}] * 5
        pairs = [(i, i+1) for i in range(5)]
        entries = entries_from_pair_results(pairs, [0.5]*5, channel_lists=chs)
        s = summarise_pair_scores(entries)
        assert s.channel_means["r"] == pytest.approx(0.4)

    def test_filter_strong(self):
        entries = [make_pair_score_entry(i, i+1, 0.5 + i*0.1) for i in range(5)]
        strong = filter_strong_pair_matches(entries, threshold=0.7)
        assert all(e.score >= 0.7 for e in strong)

    def test_filter_weak(self):
        entries = [make_pair_score_entry(i, i+1, 0.1 * i) for i in range(5)]
        weak = filter_weak_pair_matches(entries, threshold=0.3)
        assert all(e.score < 0.3 for e in weak)

    def test_filter_by_score_range(self):
        entries = [make_pair_score_entry(i, i+1, 0.2 * i) for i in range(6)]
        result = filter_pair_by_score_range(entries, lo=0.2, hi=0.6)
        assert all(0.2 <= e.score <= 0.6 for e in result)

    def test_filter_by_channel(self):
        chs = [{"edge": 0.8}, {"edge": 0.2}, {"edge": 0.9}]
        entries = [make_pair_score_entry(i, i+1, 0.5, channels=chs[i]) for i in range(3)]
        result = filter_pair_by_channel(entries, "edge", min_val=0.7)
        assert len(result) == 2

    def test_filter_by_dominant_channel(self):
        chs = [{"r": 0.9, "g": 0.1}, {"r": 0.1, "g": 0.9}]
        entries = [make_pair_score_entry(i, i+1, 0.5, channels=chs[i]) for i in range(2)]
        result = filter_pair_by_dominant_channel(entries, "r")
        assert len(result) == 1
        assert result[0].frag_i == 0

    def test_top_k_pair_entries(self):
        entries = [make_pair_score_entry(i, i+1, float(i)/10) for i in range(10)]
        top = top_k_pair_entries(entries, k=3)
        assert len(top) == 3
        assert top[0].score >= top[1].score

    def test_best_pair_entry(self):
        entries = [make_pair_score_entry(i, i+1, float(i)/10) for i in range(5)]
        best = best_pair_entry(entries)
        assert best is not None
        assert best.score == pytest.approx(0.4)

    def test_best_pair_entry_empty(self):
        assert best_pair_entry([]) is None

    def test_pair_score_stats_empty(self):
        s = pair_score_stats([])
        assert s["count"] == 0

    def test_pair_score_stats(self):
        entries = [make_pair_score_entry(i, i+1, float(i)/10) for i in range(5)]
        stats = pair_score_stats(entries)
        assert stats["count"] == 5
        assert "n_strong" in stats

    def test_compare_pair_summaries(self):
        e1 = [make_pair_score_entry(i, i+1, 0.8) for i in range(5)]
        e2 = [make_pair_score_entry(i, i+1, 0.4) for i in range(5)]
        s1 = summarise_pair_scores(e1)
        s2 = summarise_pair_scores(e2)
        d = compare_pair_summaries(s1, s2)
        assert d["d_mean_score"] > 0

    def test_batch_summarise_pair_scores(self):
        groups = [
            [make_pair_score_entry(i, i+1, 0.5) for i in range(3)],
            [make_pair_score_entry(i, i+1, 0.8) for i in range(3)],
        ]
        results = batch_summarise_pair_scores(groups)
        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. patch_score_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchScoreUtils:

    def test_config_defaults(self):
        cfg = PatchScoreConfig()
        assert cfg.min_score == 0.0
        assert cfg.max_pairs == 1000
        assert cfg.method == "total"

    def test_config_invalid_method(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(method="unknown")

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(min_score=-0.1)

    def test_config_invalid_max_pairs(self):
        with pytest.raises(ValueError):
            PatchScoreConfig(max_pairs=0)

    def test_make_patch_entry(self):
        e = make_patch_entry(0, 0, 1, 0, 1, 0.5, 0.3, 0.7, 0.8)
        assert e.pair == (0, 1)
        assert e.total_score == pytest.approx(0.8)
        assert e.is_good is True

    def test_make_patch_entry_poor(self):
        e = make_patch_entry(0, 0, 1, 0, 1, 0.2, 0.5, 0.3, 0.4)
        assert e.is_good is False

    def test_make_patch_entry_negative_pair_id(self):
        with pytest.raises(ValueError):
            make_patch_entry(-1, 0, 1, 0, 1, 0.5, 0.3, 0.7, 0.8)

    def test_make_patch_entry_invalid_ssd(self):
        with pytest.raises(ValueError):
            make_patch_entry(0, 0, 1, 0, 1, 0.5, 1.5, 0.7, 0.8)

    def test_make_patch_entry_invalid_total_score(self):
        with pytest.raises(ValueError):
            make_patch_entry(0, 0, 1, 0, 1, 0.5, 0.3, 0.7, 1.5)

    def test_entries_from_patch_matches(self):
        class PM:
            idx1=0; idx2=1; side1=0; side2=1
            ncc=0.5; ssd=0.3; ssim=0.7; total_score=0.8

        entries = entries_from_patch_matches([PM(), PM()])
        assert len(entries) == 2
        assert entries[0].rank in (1, 2)

    def test_summarise_patch_scores_empty(self):
        s = summarise_patch_scores([])
        assert s.n_total == 0
        assert s.mean_total == 0.0

    def test_summarise_patch_scores(self):
        entries = _make_patch_entries(5, good=True)
        s = summarise_patch_scores(entries)
        assert s.n_total == 5
        assert s.n_good > 0
        assert s.mean_total > 0.0

    def test_summarise_patch_n_good_n_poor(self):
        good = _make_patch_entries(3, good=True)
        poor = _make_patch_entries(2, good=False)
        s = summarise_patch_scores(good + poor)
        assert s.n_good + s.n_poor == 5

    def test_filter_good_patch_scores(self):
        entries = _make_patch_entries(5, good=True) + _make_patch_entries(3, good=False)
        good = filter_good_patch_scores(entries)
        assert all(e.is_good for e in good)

    def test_filter_poor_patch_scores(self):
        entries = _make_patch_entries(5, good=True) + _make_patch_entries(3, good=False)
        poor = filter_poor_patch_scores(entries)
        assert all(not e.is_good for e in poor)

    def test_filter_patch_by_score_range(self):
        entries = _make_patch_entries(8)
        filtered = filter_patch_by_score_range(entries, lo=0.6, hi=0.8)
        assert all(0.6 <= e.total_score <= 0.8 for e in filtered)

    def test_filter_by_side_pair(self):
        e1 = make_patch_entry(0, 0, 1, 0, 1, 0.5, 0.3, 0.7, 0.8)
        e2 = make_patch_entry(1, 0, 1, 2, 3, 0.5, 0.3, 0.7, 0.7)
        result = filter_by_side_pair([e1, e2], side1=0, side2=1)
        assert len(result) == 1

    def test_filter_by_ncc_range(self):
        entries = [make_patch_entry(i, 0, 1, 0, 1, float(i-2)/4, 0.3, 0.7, 0.8)
                   for i in range(5)]
        result = filter_by_ncc_range(entries, lo=0.0, hi=0.5)
        assert all(0.0 <= e.ncc <= 0.5 for e in result)

    def test_top_k_patch_entries(self):
        entries = _make_patch_entries(8)
        top = top_k_patch_entries(entries, k=3)
        assert len(top) == 3
        assert top[0].total_score >= top[1].total_score

    def test_top_k_patch_entries_zero(self):
        entries = _make_patch_entries(3)
        assert top_k_patch_entries(entries, k=0) == []

    def test_best_patch_entry(self):
        entries = _make_patch_entries(5)
        best = best_patch_entry(entries)
        assert best is not None
        assert best.total_score == max(e.total_score for e in entries)

    def test_best_patch_entry_empty(self):
        assert best_patch_entry([]) is None

    def test_patch_score_stats_empty(self):
        s = patch_score_stats([])
        assert s["n"] == 0

    def test_patch_score_stats(self):
        entries = _make_patch_entries(5)
        s = patch_score_stats(entries)
        assert s["n"] == 5
        assert s["min"] <= s["max"]

    def test_compare_patch_summaries(self):
        s1 = summarise_patch_scores(_make_patch_entries(3, good=True))
        s2 = summarise_patch_scores(_make_patch_entries(3, good=False))
        d = compare_patch_summaries(s1, s2)
        assert d["a_better"] is True

    def test_batch_summarise_patch_scores(self):
        groups = [_make_patch_entries(3), _make_patch_entries(4)]
        results = batch_summarise_patch_scores(groups)
        assert len(results) == 2
        assert results[0].n_total == 3
        assert results[1].n_total == 4


# ─────────────────────────────────────────────────────────────────────────────
# 4. patch_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchUtils:

    def test_patch_config_defaults(self):
        cfg = PatchConfig()
        assert cfg.patch_h == 32
        assert cfg.patch_w == 32
        assert cfg.pad_value == 0

    def test_patch_config_invalid_h(self):
        with pytest.raises(ValueError):
            PatchConfig(patch_h=0)

    def test_patch_config_invalid_w(self):
        with pytest.raises(ValueError):
            PatchConfig(patch_w=-1)

    def test_patch_config_invalid_pad(self):
        with pytest.raises(ValueError):
            PatchConfig(pad_value=300)

    def test_patch_config_invalid_norm_mode(self):
        with pytest.raises(ValueError):
            PatchConfig(norm_mode="invalid")

    def test_extract_patch_basic(self):
        img = RNG.integers(0, 255, (100, 100), dtype=np.uint8)
        cfg = PatchConfig(patch_h=16, patch_w=16)
        patch = extract_patch(img, 50, 50, cfg)
        assert patch.shape == (16, 16)

    def test_extract_patch_color(self):
        img = RNG.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        patch = extract_patch(img, 50, 50, cfg)
        assert patch.shape == (8, 8, 3)

    def test_extract_patch_near_border(self):
        img = RNG.integers(0, 255, (50, 50), dtype=np.uint8)
        cfg = PatchConfig(patch_h=16, patch_w=16, pad_value=128)
        patch = extract_patch(img, 0, 0, cfg)
        assert patch.shape == (16, 16)

    def test_extract_patch_with_normalize(self):
        img = RNG.integers(10, 200, (50, 50), dtype=np.uint8)
        cfg = PatchConfig(patch_h=8, patch_w=8, normalize=True, norm_mode="minmax")
        patch = extract_patch(img, 25, 25, cfg)
        assert float(patch.min()) >= 0.0
        assert float(patch.max()) <= 1.0

    def test_extract_patch_invalid_ndim(self):
        img = RNG.integers(0, 255, (10, 10, 3, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_patch(img, 5, 5)

    def test_extract_patches_multiple(self):
        img = RNG.integers(0, 255, (100, 100), dtype=np.uint8)
        centers = [(10, 10), (50, 50), (80, 80)]
        patches = extract_patches(img, centers)
        assert len(patches) == 3

    def test_normalize_patch_minmax(self):
        patch = np.array([[0.0, 128.0], [255.0, 64.0]])
        result = normalize_patch(patch, mode="minmax")
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_normalize_patch_zscore(self):
        patch = RNG.random((8, 8)).astype(np.float32) * 100.0
        result = normalize_patch(patch, mode="zscore")
        assert abs(float(result.mean())) < 0.01

    def test_normalize_patch_constant(self):
        patch = np.ones((4, 4)) * 5.0
        result = normalize_patch(patch, mode="minmax")
        assert float(result.max()) == pytest.approx(0.0)

    def test_normalize_patch_invalid_mode(self):
        with pytest.raises(ValueError):
            normalize_patch(np.ones((4, 4)), mode="bad")

    def test_patch_ssd_identical(self):
        p = RNG.random((8, 8)).astype(np.float32)
        assert patch_ssd(p, p) == pytest.approx(0.0)

    def test_patch_ssd_different(self):
        a = np.zeros((8, 8), dtype=np.float32)
        b = np.ones((8, 8), dtype=np.float32)
        assert patch_ssd(a, b) == pytest.approx(64.0)

    def test_patch_ssd_shape_mismatch(self):
        with pytest.raises(ValueError):
            patch_ssd(np.ones((4, 4)), np.ones((4, 5)))

    def test_patch_ncc_identical(self):
        p = RNG.random((8, 8)).astype(np.float32)
        ncc = patch_ncc(p, p)
        assert ncc == pytest.approx(1.0, abs=1e-5)

    def test_patch_ncc_opposite(self):
        p = RNG.random((8, 8)).astype(np.float32)
        ncc = patch_ncc(p, -p)
        assert ncc == pytest.approx(-1.0, abs=1e-5)

    def test_patch_ncc_constant(self):
        a = np.ones((8, 8), dtype=np.float32)
        b = RNG.random((8, 8)).astype(np.float32)
        ncc = patch_ncc(a, b)
        assert ncc == pytest.approx(0.0)

    def test_patch_ncc_shape_mismatch(self):
        with pytest.raises(ValueError):
            patch_ncc(np.ones((4, 4)), np.ones((4, 5)))

    def test_patch_mse_identical(self):
        p = RNG.random((8, 8)).astype(np.float32)
        assert patch_mse(p, p) == pytest.approx(0.0)

    def test_patch_mse_shape_mismatch(self):
        with pytest.raises(ValueError):
            patch_mse(np.ones((4, 4)), np.ones((5, 4)))

    def test_compare_patches_ncc(self):
        p = RNG.random((8, 8)).astype(np.float32)
        val = compare_patches(p, p, method="ncc")
        assert val == pytest.approx(1.0, abs=1e-5)

    def test_compare_patches_ssd(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        val = compare_patches(a, b, method="ssd")
        assert val == pytest.approx(16.0)

    def test_compare_patches_mse(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        val = compare_patches(a, b, method="mse")
        assert val == pytest.approx(1.0)

    def test_compare_patches_invalid_method(self):
        p = np.ones((4, 4))
        with pytest.raises(ValueError):
            compare_patches(p, p, method="bad")

    def test_batch_compare_ncc(self):
        pairs = [(RNG.random((8, 8)), RNG.random((8, 8))) for _ in range(4)]
        results = batch_compare(pairs, method="ncc")
        assert len(results) == 4
        assert all(-1.0 <= r <= 1.0 for r in results)

    def test_batch_compare_invalid_method(self):
        pairs = [(np.ones((4, 4)), np.ones((4, 4)))]
        with pytest.raises(ValueError):
            batch_compare(pairs, method="invalid")


# ─────────────────────────────────────────────────────────────────────────────
# 5. path_plan_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPathPlanUtils:

    def _make_entries(self, n=5, all_found=True):
        results = []
        for i in range(n):
            found = True if all_found else (i % 2 == 0)
            results.append((0, i + 1, list(range(i + 2)), float(i + 1), found))
        return entries_from_path_results(results)

    def test_make_path_entry(self):
        e = make_path_entry(0, 3, [0, 1, 2, 3], 3.0, True)
        assert e.hops == 3
        assert e.found is True

    def test_path_entry_empty_path(self):
        e = make_path_entry(0, 1, [], 0.0, False)
        assert e.hops == 0

    def test_entries_from_path_results(self):
        entries = self._make_entries(5)
        assert len(entries) == 5

    def test_summarise_empty(self):
        s = summarise_path_entries([])
        assert s.n_entries == 0
        assert s.found_rate == 0.0

    def test_summarise_all_found(self):
        entries = self._make_entries(4, all_found=True)
        s = summarise_path_entries(entries)
        assert s.n_found == 4
        assert s.found_rate == pytest.approx(1.0)

    def test_summarise_mixed_found(self):
        entries = self._make_entries(4, all_found=False)
        s = summarise_path_entries(entries)
        assert s.n_not_found > 0
        assert 0.0 < s.found_rate < 1.0

    def test_filter_found_paths(self):
        entries = self._make_entries(5, all_found=False)
        found = filter_found_paths(entries)
        assert all(e.found for e in found)

    def test_filter_not_found_paths(self):
        entries = self._make_entries(5, all_found=False)
        not_found = filter_not_found_paths(entries)
        assert all(not e.found for e in not_found)

    def test_filter_path_by_cost_range(self):
        entries = self._make_entries(6)
        result = filter_path_by_cost_range(entries, lo=2.0, hi=4.0)
        assert all(2.0 <= e.cost <= 4.0 for e in result)

    def test_filter_path_by_max_hops(self):
        entries = self._make_entries(5)
        result = filter_path_by_max_hops(entries, max_hops=3)
        assert all(e.hops <= 3 for e in result)

    def test_top_k_shortest_paths(self):
        entries = self._make_entries(6)
        top = top_k_shortest_paths(entries, k=3)
        assert len(top) == 3
        assert top[0].cost <= top[1].cost

    def test_cheapest_path_entry(self):
        entries = self._make_entries(5)
        cheapest = cheapest_path_entry(entries)
        assert cheapest is not None
        assert cheapest.cost == min(e.cost for e in entries if e.found)

    def test_cheapest_path_entry_none_found(self):
        e = make_path_entry(0, 1, [], 0.0, False)
        assert cheapest_path_entry([e]) is None

    def test_path_cost_stats_empty(self):
        s = path_cost_stats([])
        assert s["count"] == 0

    def test_path_cost_stats(self):
        entries = self._make_entries(5)
        s = path_cost_stats(entries)
        assert s["count"] == 5
        assert s["min"] <= s["max"]

    def test_compare_path_summaries(self):
        s1 = summarise_path_entries(self._make_entries(4))
        e2 = [make_path_entry(0, 1, [0, 1], 10.0, True)]
        s2 = summarise_path_entries(e2)
        d = compare_path_summaries(s1, s2)
        assert "mean_cost_delta" in d

    def test_batch_summarise_path_entries(self):
        groups = [self._make_entries(3), self._make_entries(4)]
        results = batch_summarise_path_entries(groups)
        assert len(results) == 2

    # AssemblyPlan utilities

    def _make_plan_entries(self, n=4, strategy="greedy"):
        return [
            make_assembly_plan_entry(i, 10, 8, 0.8, 0.7, strategy, list(range(8)))
            for i in range(n)
        ]

    def test_make_assembly_plan_entry(self):
        e = make_assembly_plan_entry(0, 10, 9, 0.9, 0.85, "greedy", [1, 2, 3])
        assert e.plan_id == 0
        assert e.coverage == pytest.approx(0.9)

    def test_summarise_assembly_plans_empty(self):
        s = summarise_assembly_plans([])
        assert s.n_plans == 0

    def test_summarise_assembly_plans(self):
        entries = self._make_plan_entries(4)
        s = summarise_assembly_plans(entries)
        assert s.n_plans == 4
        assert s.strategy == "greedy"

    def test_summarise_assembly_plans_mixed_strategy(self):
        e1 = make_assembly_plan_entry(0, 10, 8, 0.8, 0.7, "greedy", [])
        e2 = make_assembly_plan_entry(1, 10, 8, 0.8, 0.7, "beam", [])
        s = summarise_assembly_plans([e1, e2])
        assert s.strategy == "mixed"

    def test_filter_full_coverage_plans(self):
        e1 = make_assembly_plan_entry(0, 10, 10, 1.0, 0.9, "greedy", [])
        e2 = make_assembly_plan_entry(1, 10, 5, 0.5, 0.7, "greedy", [])
        result = filter_full_coverage_plans([e1, e2])
        assert len(result) == 1
        assert result[0].plan_id == 0

    def test_filter_assembly_plans_by_coverage(self):
        entries = self._make_plan_entries(4)
        result = filter_assembly_plans_by_coverage(entries, min_coverage=0.9)
        assert all(e.coverage >= 0.9 for e in result)

    def test_filter_assembly_plans_by_score(self):
        entries = self._make_plan_entries(4)
        result = filter_assembly_plans_by_score(entries, min_score=0.8)
        assert all(e.mean_score >= 0.8 for e in result)

    def test_filter_assembly_plans_by_strategy(self):
        e1 = make_assembly_plan_entry(0, 10, 8, 0.8, 0.7, "greedy", [])
        e2 = make_assembly_plan_entry(1, 10, 8, 0.8, 0.7, "beam", [])
        result = filter_assembly_plans_by_strategy([e1, e2], "greedy")
        assert len(result) == 1

    def test_top_k_assembly_plan_entries(self):
        entries = [
            make_assembly_plan_entry(i, 10, 8, float(i)/10, float(i)/10, "greedy", [])
            for i in range(5)
        ]
        top = top_k_assembly_plan_entries(entries, k=2)
        assert len(top) == 2

    def test_best_assembly_plan_entry_empty(self):
        assert best_assembly_plan_entry([]) is None

    def test_best_assembly_plan_entry(self):
        entries = self._make_plan_entries(3)
        best = best_assembly_plan_entry(entries)
        assert best is not None

    def test_assembly_plan_stats(self):
        entries = self._make_plan_entries(3)
        s = assembly_plan_stats(entries)
        assert s["count"] == 3.0
        assert "mean_coverage" in s

    def test_compare_assembly_plan_summaries(self):
        e1 = [make_assembly_plan_entry(0, 10, 9, 0.9, 0.9, "greedy", [])]
        e2 = [make_assembly_plan_entry(0, 10, 5, 0.5, 0.5, "greedy", [])]
        s1 = summarise_assembly_plans(e1)
        s2 = summarise_assembly_plans(e2)
        d = compare_assembly_plan_summaries(s1, s2)
        assert d["mean_coverage_delta"] > 0

    def test_batch_summarise_assembly_plans(self):
        groups = [self._make_plan_entries(2), self._make_plan_entries(3)]
        results = batch_summarise_assembly_plans(groups)
        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. placement_metrics_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPlacementMetricsUtils:

    def test_placement_density_basic(self):
        assert placement_density(5, 10) == pytest.approx(0.5)

    def test_placement_density_all_placed(self):
        assert placement_density(10, 10) == pytest.approx(1.0)

    def test_placement_density_zero_total(self):
        assert placement_density(0, 0) == 0.0

    def test_placement_density_negative_raises(self):
        with pytest.raises(ValueError):
            placement_density(-1, 10)

    def test_placement_density_clamped(self):
        # n_placed > n_total → clamped to 1.0
        assert placement_density(15, 10) == pytest.approx(1.0)

    def test_bbox_of_contour_square(self):
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        bbox = bbox_of_contour(contour, position=(5.0, 5.0))
        assert bbox == pytest.approx((5.0, 5.0, 15.0, 15.0))

    def test_bbox_of_contour_empty(self):
        bbox = bbox_of_contour(np.empty((0, 2)), position=(3.0, 4.0))
        assert bbox == (3.0, 4.0, 3.0, 4.0)

    def test_bbox_area_basic(self):
        area = bbox_area((0.0, 0.0, 10.0, 5.0))
        assert area == pytest.approx(50.0)

    def test_bbox_area_zero(self):
        assert bbox_area((0.0, 0.0, 0.0, 5.0)) == 0.0

    def test_bbox_intersection_area_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        area = bbox_intersection_area(a, b)
        assert area == pytest.approx(25.0)

    def test_bbox_intersection_area_no_overlap(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (10.0, 10.0, 20.0, 20.0)
        assert bbox_intersection_area(a, b) == 0.0

    def test_compute_coverage_no_fragments(self):
        assert compute_coverage([], [], (100, 100)) == 0.0

    def test_compute_coverage_single_fragment(self):
        contour = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=float)
        cov = compute_coverage([(0.0, 0.0)], [contour], (100, 100))
        assert 0.0 < cov <= 1.0

    def test_compute_pairwise_overlap_no_overlap(self):
        c1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        c2 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=float)
        ovlp = compute_pairwise_overlap([(0.0, 0.0), (0.0, 0.0)], [c1, c2])
        assert ovlp == 0.0

    def test_compute_pairwise_overlap_with_overlap(self):
        c1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        c2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        ovlp = compute_pairwise_overlap([(0.0, 0.0), (0.0, 0.0)], [c1, c2])
        assert ovlp > 0.0

    def test_quality_score_perfect(self):
        qs = quality_score(1.0, 1.0, 0.0)
        assert qs == pytest.approx(1.0)

    def test_quality_score_zero_weights(self):
        qs = quality_score(1.0, 1.0, 0.0, w_density=0.0, w_coverage=0.0, w_overlap=0.0)
        assert qs == 0.0

    def test_quality_score_clamped(self):
        qs = quality_score(2.0, 2.0, 0.0)
        assert qs <= 1.0

    def test_assess_placement(self):
        c = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
        positions = [(0.0, 0.0), (30.0, 30.0)]
        contours = [c, c]
        m = assess_placement(positions, contours, n_total=5, canvas_size=(100, 100))
        assert isinstance(m, PlacementMetrics)
        assert 0.0 <= m.quality_score <= 1.0

    def test_compare_metrics(self):
        c = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        m1 = assess_placement([(0.0, 0.0)], [c], n_total=5)
        m2 = assess_placement([(5.0, 5.0)], [c], n_total=5)
        result = compare_metrics(m1, m2)
        assert "better" in result
        assert result["better"] in ("a", "b")

    def test_best_of(self):
        c = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        mlist = [
            assess_placement([(0.0, 0.0)], [c], n_total=5),
            assess_placement([(0.0, 0.0)], [c], n_total=2),
        ]
        idx = best_of(mlist)
        assert idx in (0, 1)

    def test_best_of_empty(self):
        with pytest.raises(ValueError):
            best_of([])

    def test_normalize_metrics(self):
        c = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        m1 = PlacementMetrics(3, 5, 0.6, 0.5, 0.0, quality_score=0.4)
        m2 = PlacementMetrics(4, 5, 0.8, 0.7, 0.0, quality_score=0.8)
        normed = normalize_metrics([m1, m2])
        assert len(normed) == 2
        assert normed[0].quality_score == pytest.approx(0.0)
        assert normed[1].quality_score == pytest.approx(1.0)

    def test_normalize_metrics_empty(self):
        assert normalize_metrics([]) == []

    def test_normalize_metrics_constant(self):
        m = PlacementMetrics(3, 5, 0.6, 0.5, 0.0, quality_score=0.5)
        normed = normalize_metrics([m, m])
        assert all(n.quality_score == pytest.approx(1.0) for n in normed)

    def test_batch_quality_scores(self):
        m1 = PlacementMetrics(3, 5, 0.6, 0.5, 0.0, quality_score=0.4)
        m2 = PlacementMetrics(4, 5, 0.8, 0.7, 0.0, quality_score=0.8)
        scores = batch_quality_scores([m1, m2])
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)


# ─────────────────────────────────────────────────────────────────────────────
# 7. placement_score_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPlacementScoreUtils:

    def test_config_defaults(self):
        cfg = PlacementScoreConfig()
        assert cfg.min_score == 0.0
        assert cfg.coverage_weight == 0.5

    def test_config_invalid_min_score(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(min_score=1.5)

    def test_config_invalid_coverage_weight(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(coverage_weight=-0.1)

    def test_config_invalid_overlap_penalty_weight(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(overlap_penalty_weight=2.0)

    def test_make_placement_entry(self):
        e = make_placement_entry(0, 3, 0.1, 0.1, (5.0, 10.0))
        assert e.step == 0
        assert e.fragment_idx == 3
        assert e.position == (5.0, 10.0)

    def test_placement_entry_negative_step_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreEntry(step=-1, fragment_idx=0, score_delta=0.1, cumulative_score=0.1)

    def test_placement_entry_negative_fragment_idx_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreEntry(step=0, fragment_idx=-1, score_delta=0.1, cumulative_score=0.1)

    def test_placement_entry_repr(self):
        e = make_placement_entry(0, 1, 0.2, 0.2)
        assert "PlacementScoreEntry" in repr(e)

    def test_entries_from_history_basic(self):
        history = [
            {"step": 0, "idx": 1, "score_delta": 0.1},
            {"step": 1, "idx": 2, "score_delta": 0.2},
        ]
        entries = entries_from_history(history)
        assert len(entries) == 2
        assert entries[1].cumulative_score == pytest.approx(0.3)

    def test_entries_from_history_with_position(self):
        history = [{"step": 0, "idx": 1, "score_delta": 0.3, "position": (5.0, 7.0)}]
        entries = entries_from_history(history)
        assert entries[0].position == (5.0, 7.0)

    def test_summarise_placement_empty(self):
        s = summarise_placement([])
        assert s.n_placed == 0
        assert s.final_score == 0.0

    def test_summarise_placement(self):
        history = [{"step": i, "idx": i, "score_delta": 0.1} for i in range(5)]
        entries = entries_from_history(history)
        s = summarise_placement(entries)
        assert s.n_placed == 5
        assert s.final_score == pytest.approx(0.5)

    def test_placement_summary_repr(self):
        history = [{"step": 0, "idx": 0, "score_delta": 0.1}]
        entries = entries_from_history(history)
        s = summarise_placement(entries)
        assert "PlacementSummary" in repr(s)

    def test_filter_positive_steps(self):
        history = [
            {"step": 0, "idx": 0, "score_delta": 0.2},
            {"step": 1, "idx": 1, "score_delta": -0.1},
            {"step": 2, "idx": 2, "score_delta": 0.3},
        ]
        entries = entries_from_history(history)
        pos = filter_positive_steps(entries)
        assert all(e.score_delta > 0 for e in pos)

    def test_filter_by_min_score(self):
        history = [{"step": i, "idx": i, "score_delta": 0.1} for i in range(5)]
        entries = entries_from_history(history)
        result = filter_by_min_score(entries, min_score=0.3)
        assert all(e.cumulative_score >= 0.3 for e in result)

    def test_top_k_steps(self):
        history = [{"step": i, "idx": i, "score_delta": float(i) * 0.1} for i in range(6)]
        entries = entries_from_history(history)
        top = top_k_steps(entries, k=3)
        assert len(top) == 3
        assert top[0].score_delta >= top[1].score_delta

    def test_rank_fragments(self):
        history = [{"step": i, "idx": i, "score_delta": float(i) * 0.1} for i in range(4)]
        entries = entries_from_history(history)
        ranked = rank_fragments(entries)
        assert len(ranked) == 4
        assert ranked[0][1] >= ranked[-1][1]

    def test_placement_score_stats_empty(self):
        s = placement_score_stats([])
        assert s["n"] == 0

    def test_placement_score_stats(self):
        history = [{"step": i, "idx": i, "score_delta": (0.1 if i % 2 == 0 else -0.05)}
                   for i in range(6)]
        entries = entries_from_history(history)
        stats = placement_score_stats(entries)
        assert stats["n"] == 6
        assert stats["n_positive"] > 0
        assert stats["n_negative"] > 0

    def test_compare_placements(self):
        h1 = [{"step": i, "idx": i, "score_delta": 0.2} for i in range(4)]
        h2 = [{"step": i, "idx": i, "score_delta": 0.1} for i in range(4)]
        s1 = summarise_placement(entries_from_history(h1))
        s2 = summarise_placement(entries_from_history(h2))
        d = compare_placements(s1, s2)
        assert d["better"] in ("a", "b", "tie")

    def test_batch_summarise(self):
        histories = [
            [{"step": i, "idx": i, "score_delta": 0.1} for i in range(3)],
            [{"step": i, "idx": i, "score_delta": 0.2} for i in range(4)],
        ]
        results = batch_summarise(histories)
        assert len(results) == 2
        assert results[0].n_placed == 3
        assert results[1].n_placed == 4


# ─────────────────────────────────────────────────────────────────────────────
# 8. polygon_ops_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPolygonOpsUtils:

    def test_config_defaults(self):
        cfg = PolygonOpsConfig()
        assert cfg.clip_epsilon == pytest.approx(1e-9)
        assert cfg.n_samples == 64

    def test_config_invalid_clip_epsilon(self):
        with pytest.raises(ValueError):
            PolygonOpsConfig(clip_epsilon=-1.0)

    def test_config_invalid_n_samples(self):
        with pytest.raises(ValueError):
            PolygonOpsConfig(n_samples=0)

    def test_signed_area_ccw(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        area = signed_area(sq)
        assert area == pytest.approx(100.0)

    def test_signed_area_cw(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)[::-1]
        area = signed_area(sq)
        assert area == pytest.approx(-100.0)

    def test_signed_area_degenerate(self):
        assert signed_area(np.array([[0, 0], [1, 1]])) == 0.0

    def test_polygon_area_positive(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        assert polygon_area(sq) == pytest.approx(100.0)

    def test_polygon_perimeter_square(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        assert polygon_perimeter(sq) == pytest.approx(40.0)

    def test_polygon_perimeter_less_than_2(self):
        assert polygon_perimeter(np.array([[0, 0]])) == 0.0

    def test_polygon_centroid_square(self):
        sq = _make_square_poly(3.0, 7.0, 5.0)
        c = polygon_centroid(sq)
        assert c == pytest.approx([3.0, 7.0], abs=1e-6)

    def test_polygon_centroid_degenerate(self):
        pts = np.array([[0, 0], [1, 0]])
        c = polygon_centroid(pts)
        assert c == pytest.approx([0.5, 0.0])

    def test_polygon_bounding_box(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        bb = polygon_bounding_box(sq)
        assert bb == pytest.approx((-5.0, -5.0, 5.0, 5.0))

    def test_polygon_bounding_box_empty(self):
        bb = polygon_bounding_box(np.empty((0, 2)))
        assert bb == (0.0, 0.0, 0.0, 0.0)

    def test_polygon_stats(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        s = polygon_stats(sq)
        assert s.n_vertices == 4
        assert s.area == pytest.approx(100.0)
        assert s.perimeter == pytest.approx(40.0)
        assert s.aspect_ratio == pytest.approx(1.0, abs=0.01)
        assert s.compactness > 0.0

    def test_polygon_stats_to_dict(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        d = polygon_stats(sq).to_dict()
        assert "area" in d
        assert "n_vertices" in d

    def test_point_in_polygon_inside(self):
        sq = _make_square_poly(0.0, 0.0, 10.0)
        assert point_in_polygon(np.array([0.0, 0.0]), sq) is True

    def test_point_in_polygon_outside(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        assert point_in_polygon(np.array([20.0, 20.0]), sq) is False

    def test_point_in_polygon_degenerate(self):
        assert point_in_polygon(np.array([0.0, 0.0]), np.array([[0, 0], [1, 1]])) is False

    def test_polygon_overlap_overlapping(self):
        p1 = _make_square_poly(0.0, 0.0, 5.0)
        p2 = _make_square_poly(3.0, 3.0, 5.0)
        result = polygon_overlap(p1, p2)
        assert result.overlap is True
        assert result.iou > 0.0

    def test_polygon_overlap_not_overlapping(self):
        p1 = _make_square_poly(0.0, 0.0, 5.0)
        p2 = _make_square_poly(50.0, 50.0, 5.0)
        result = polygon_overlap(p1, p2)
        assert result.overlap is False
        assert result.iou == pytest.approx(0.0)

    def test_polygon_overlap_to_dict(self):
        p1 = _make_square_poly(0.0, 0.0, 5.0)
        p2 = _make_square_poly(3.0, 3.0, 5.0)
        d = polygon_overlap(p1, p2).to_dict()
        assert "iou" in d

    def test_remove_collinear_no_collinear(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        result = remove_collinear(sq)
        assert len(result) == 4

    def test_remove_collinear_with_collinear(self):
        pts = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        result = remove_collinear(pts)
        assert len(result) < 5

    def test_ensure_ccw_already_ccw(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        result = ensure_ccw(sq)
        assert signed_area(result) > 0

    def test_ensure_cw(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        result = ensure_cw(sq)
        assert signed_area(result) < 0

    def test_polygon_similarity_identical(self):
        sq = _make_square_poly(0.0, 0.0, 5.0)
        s = polygon_similarity(sq, sq)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_polygon_similarity_non_overlapping(self):
        p1 = _make_square_poly(0.0, 0.0, 5.0)
        p2 = _make_square_poly(50.0, 50.0, 5.0)
        s = polygon_similarity(p1, p2)
        assert s == pytest.approx(0.0)

    def test_batch_polygon_stats(self):
        polys = [_make_square_poly(float(i), float(i), 5.0) for i in range(4)]
        results = batch_polygon_stats(polys)
        assert len(results) == 4

    def test_batch_polygon_overlap(self):
        p1s = [_make_square_poly(float(i), float(i), 5.0) for i in range(3)]
        p2s = [_make_square_poly(float(i)+2, float(i)+2, 5.0) for i in range(3)]
        results = batch_polygon_overlap(p1s, p2s)
        assert len(results) == 3

    def test_batch_polygon_overlap_length_mismatch(self):
        p1s = [_make_square_poly(0.0, 0.0, 5.0)]
        p2s = [_make_square_poly(0.0, 0.0, 5.0), _make_square_poly(1.0, 1.0, 5.0)]
        with pytest.raises(ValueError):
            batch_polygon_overlap(p1s, p2s)


# ─────────────────────────────────────────────────────────────────────────────
# 9. position_tracking_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionTrackingUtils:

    def _make_records(self, n=5, method="ransac"):
        return [
            make_position_quality_record(i, 10, 8, 0.85, 0.7 + i * 0.02, method)
            for i in range(n)
        ]

    def test_make_position_quality_record(self):
        r = make_position_quality_record(0, 10, 8, 0.9, 0.8, "ransac")
        assert r.run_id == 0
        assert r.n_fragments == 10
        assert r.method == "ransac"

    def test_placement_rate(self):
        r = make_position_quality_record(0, 10, 8, 0.9, 0.8, "ransac")
        assert r.placement_rate == pytest.approx(0.8)

    def test_placement_rate_zero_fragments(self):
        r = make_position_quality_record(0, 0, 0, 0.9, 0.8, "ransac")
        assert r.placement_rate == 0.0

    def test_position_quality_record_with_params(self):
        r = make_position_quality_record(0, 10, 8, 0.9, 0.8, "ransac", threshold=5.0)
        assert r.params["threshold"] == 5.0

    def test_summarise_position_quality_empty(self):
        s = summarise_position_quality([])
        assert s.n_runs == 0
        assert s.best_run_id is None

    def test_summarise_position_quality(self):
        records = self._make_records(5)
        s = summarise_position_quality(records)
        assert s.n_runs == 5
        assert s.total_fragments == 50
        assert s.best_run_id is not None
        assert s.worst_run_id is not None

    def test_summarise_position_quality_best_worst(self):
        records = self._make_records(5)
        s = summarise_position_quality(records)
        # best has highest coverage
        coverages = {r.run_id: r.canvas_coverage for r in records}
        assert coverages[s.best_run_id] >= coverages[s.worst_run_id]

    def test_filter_by_placement_rate(self):
        records = self._make_records(5)
        result = filter_by_placement_rate(records, min_rate=0.8)
        assert all(r.placement_rate >= 0.8 for r in result)

    def test_filter_by_method(self):
        r1s = self._make_records(3, method="ransac")
        r2s = self._make_records(2, method="icp")
        result = filter_by_method(r1s + r2s, method="ransac")
        assert len(result) == 3

    def test_top_k_position_records(self):
        records = self._make_records(6)
        top = top_k_position_records(records, k=3)
        assert len(top) == 3
        assert top[0].canvas_coverage >= top[1].canvas_coverage

    def test_best_position_record(self):
        records = self._make_records(5)
        best = best_position_record(records)
        assert best is not None

    def test_best_position_record_empty(self):
        assert best_position_record([]) is None

    def test_position_quality_stats_empty(self):
        s = position_quality_stats([])
        assert s["count"] == 0

    def test_position_quality_stats(self):
        records = self._make_records(5)
        s = position_quality_stats(records)
        assert s["count"] == 5
        assert s["min"] <= s["max"]

    # AssemblyHistoryEntry tests

    def _make_history_entries(self, n=5):
        return [
            make_assembly_history_entry(i, 100, 0.7 + i * 0.05, i % 2 == 0, 50 if i % 2 == 0 else None, "simulated_annealing")
            for i in range(n)
        ]

    def test_make_assembly_history_entry(self):
        e = make_assembly_history_entry(0, 100, 0.9, True, 50, "sa")
        assert e.run_id == 0
        assert e.converged is True
        assert e.method == "sa"

    def test_summarise_assembly_history_empty(self):
        s = summarise_assembly_history([])
        assert s.n_runs == 0
        assert s.best_run_id is None

    def test_summarise_assembly_history(self):
        entries = self._make_history_entries(4)
        s = summarise_assembly_history(entries)
        assert s.n_runs == 4
        assert 0.0 <= s.convergence_rate <= 1.0

    def test_filter_converged(self):
        entries = self._make_history_entries(5)
        conv = filter_converged(entries)
        assert all(e.converged for e in conv)

    def test_filter_by_min_best_score(self):
        entries = self._make_history_entries(5)
        result = filter_by_min_best_score(entries, min_score=0.8)
        assert all(e.best_score >= 0.8 for e in result)

    def test_top_k_assembly_entries(self):
        entries = self._make_history_entries(6)
        top = top_k_assembly_entries(entries, k=3)
        assert len(top) == 3
        assert top[0].best_score >= top[1].best_score

    def test_best_assembly_entry(self):
        entries = self._make_history_entries(5)
        best = best_assembly_entry(entries)
        assert best is not None

    def test_best_assembly_entry_empty(self):
        assert best_assembly_entry([]) is None

    def test_assembly_score_stats_empty(self):
        s = assembly_score_stats([])
        assert s["count"] == 0

    def test_assembly_score_stats(self):
        entries = self._make_history_entries(5)
        s = assembly_score_stats(entries)
        assert s["count"] == 5

    def test_compare_assembly_summaries(self):
        s1 = summarise_assembly_history(self._make_history_entries(4))
        s2 = summarise_assembly_history(self._make_history_entries(4))
        d = compare_assembly_summaries(s1, s2)
        assert "delta_mean_best_score" in d

    def test_batch_summarise_assembly_history(self):
        groups = [self._make_history_entries(3), self._make_history_entries(4)]
        results = batch_summarise_assembly_history(groups)
        assert len(results) == 2

