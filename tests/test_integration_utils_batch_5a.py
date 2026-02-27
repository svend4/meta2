"""Integration tests for utils batch 5a.

Covers:
  1. puzzle_reconstruction.utils.path_plan_utils
  2. puzzle_reconstruction.utils.placement_metrics_utils
  3. puzzle_reconstruction.utils.placement_score_utils
  4. puzzle_reconstruction.utils.polygon_ops_utils
  5. puzzle_reconstruction.utils.position_tracking_utils
"""
import math
import pytest
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
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
)
from puzzle_reconstruction.utils.position_tracking_utils import (
    PositionQualityRecord,
    PositionQualitySummary,
    make_position_quality_record,
    summarise_position_quality,
    filter_by_placement_rate,
    filter_by_method,
    top_k_position_records,
)


# ===========================================================================
# 1. path_plan_utils
# ===========================================================================

class TestPathPlanUtils:

    def _sample_entries(self):
        raw = [
            (0, 3, [0, 1, 2, 3], 3.5, True),
            (1, 4, [1, 4], 1.2, True),
            (2, 5, [2, 3, 4, 5], 4.0, True),
            (0, 5, [], 0.0, False),
            (3, 5, [3, 5], 0.8, True),
        ]
        return entries_from_path_results(raw)

    def test_make_path_entry_hops(self):
        e = make_path_entry(0, 3, [0, 1, 2, 3], 2.5, True)
        assert e.hops == 3
        assert e.start == 0 and e.end == 3

    def test_entries_from_path_results_length(self):
        entries = self._sample_entries()
        assert len(entries) == 5

    def test_path_entry_not_found_hops_zero(self):
        e = make_path_entry(0, 5, [], 0.0, False)
        assert e.hops == 0
        assert not e.found

    def test_summarise_path_entries_counts(self):
        entries = self._sample_entries()
        s = summarise_path_entries(entries)
        assert s.n_entries == 5
        assert s.n_found == 4
        assert s.n_not_found == 1

    def test_summarise_path_entries_found_rate(self):
        entries = self._sample_entries()
        s = summarise_path_entries(entries)
        assert abs(s.found_rate - 0.8) < 1e-9

    def test_summarise_empty_entries(self):
        s = summarise_path_entries([])
        assert s.n_entries == 0
        assert s.found_rate == 0.0

    def test_filter_found_and_not_found(self):
        entries = self._sample_entries()
        found = filter_found_paths(entries)
        not_found = filter_not_found_paths(entries)
        assert len(found) == 4
        assert len(not_found) == 1
        assert all(e.found for e in found)
        assert all(not e.found for e in not_found)

    def test_filter_by_cost_range(self):
        entries = self._sample_entries()
        cheap = filter_path_by_cost_range(entries, lo=0.0, hi=2.0)
        assert all(e.cost <= 2.0 for e in cheap)
        assert len(cheap) == 2  # costs 1.2 and 0.8

    def test_filter_by_max_hops(self):
        entries = self._sample_entries()
        short = filter_path_by_max_hops(entries, max_hops=2)
        assert all(e.hops <= 2 for e in short)

    def test_top_k_shortest_paths(self):
        entries = self._sample_entries()
        top2 = top_k_shortest_paths(entries, k=2)
        assert len(top2) == 2
        assert top2[0].cost <= top2[1].cost

    def test_cheapest_path_entry(self):
        entries = self._sample_entries()
        cheapest = cheapest_path_entry(entries)
        assert cheapest is not None
        assert cheapest.cost == min(e.cost for e in entries if e.found)

    def test_path_cost_stats_keys(self):
        entries = self._sample_entries()
        stats = path_cost_stats(entries)
        assert set(stats.keys()) == {"count", "mean", "std", "min", "max"}
        assert stats["count"] == 4.0

    def test_compare_path_summaries(self):
        entries = self._sample_entries()
        s = summarise_path_entries(entries)
        diff = compare_path_summaries(s, s)
        assert diff["found_rate_delta"] == 0.0
        assert diff["mean_cost_delta"] == 0.0


# ===========================================================================
# 2. placement_metrics_utils
# ===========================================================================

class TestPlacementMetricsUtils:

    def _square_contour(self, size=50.0):
        return np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=float)

    def test_placement_density_full(self):
        assert placement_density(10, 10) == 1.0

    def test_placement_density_partial(self):
        assert abs(placement_density(4, 8) - 0.5) < 1e-9

    def test_placement_density_zero_total(self):
        assert placement_density(0, 0) == 0.0

    def test_placement_density_raises_negative(self):
        with pytest.raises(ValueError):
            placement_density(-1, 5)

    def test_bbox_of_contour(self):
        c = self._square_contour(100.0)
        bb = bbox_of_contour(c, position=(10.0, 20.0))
        assert bb == (10.0, 20.0, 110.0, 120.0)

    def test_bbox_area(self):
        assert bbox_area((0.0, 0.0, 10.0, 5.0)) == 50.0

    def test_bbox_intersection_area_overlapping(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        assert bbox_intersection_area(a, b) == 25.0

    def test_bbox_intersection_area_non_overlapping(self):
        a = (0.0, 0.0, 5.0, 5.0)
        b = (10.0, 10.0, 20.0, 20.0)
        assert bbox_intersection_area(a, b) == 0.0

    def test_compute_coverage_returns_float_in_01(self):
        positions = [(0.0, 0.0), (100.0, 100.0)]
        contours = [self._square_contour(50.0), self._square_contour(50.0)]
        cov = compute_coverage(positions, contours, canvas_size=(512, 512))
        assert 0.0 <= cov <= 1.0

    def test_compute_coverage_empty(self):
        cov = compute_coverage([], [], canvas_size=(512, 512))
        assert cov == 0.0

    def test_compute_pairwise_overlap_no_overlap(self):
        positions = [(0.0, 0.0), (200.0, 200.0)]
        contours = [self._square_contour(50.0), self._square_contour(50.0)]
        ovlp = compute_pairwise_overlap(positions, contours)
        assert ovlp == 0.0

    def test_quality_score_perfect(self):
        qs = quality_score(1.0, 1.0, 0.0)
        assert abs(qs - 1.0) < 1e-6

    def test_assess_placement_returns_metrics(self):
        positions = [(10.0, 10.0), (100.0, 100.0)]
        contours = [self._square_contour(40.0), self._square_contour(40.0)]
        m = assess_placement(positions, contours, n_total=4)
        assert isinstance(m, PlacementMetrics)
        assert m.n_placed == 2
        assert m.n_total == 4
        assert 0.0 <= m.quality_score <= 1.0


# ===========================================================================
# 3. placement_score_utils
# ===========================================================================

class TestPlacementScoreUtils:

    def _make_entries(self):
        deltas = [0.1, 0.2, -0.05, 0.3, 0.15]
        entries = []
        cum = 0.0
        for i, d in enumerate(deltas):
            cum += d
            entries.append(make_placement_entry(i, i, d, cum, position=(float(i), 0.0)))
        return entries

    def test_config_defaults(self):
        cfg = PlacementScoreConfig()
        assert cfg.min_score == 0.0
        assert cfg.coverage_weight == 0.5

    def test_config_invalid_min_score_raises(self):
        with pytest.raises(ValueError):
            PlacementScoreConfig(min_score=1.5)

    def test_make_placement_entry_values(self):
        e = make_placement_entry(2, 7, 0.3, 0.9, position=(5.0, 3.0))
        assert e.step == 2
        assert e.fragment_idx == 7
        assert abs(e.score_delta - 0.3) < 1e-9

    def test_make_placement_entry_negative_step_raises(self):
        with pytest.raises(ValueError):
            make_placement_entry(-1, 0, 0.1, 0.1)

    def test_entries_from_history(self):
        history = [
            {"step": 0, "idx": 3, "score_delta": 0.2},
            {"step": 1, "idx": 5, "score_delta": 0.3},
        ]
        entries = entries_from_history(history)
        assert len(entries) == 2
        assert abs(entries[1].cumulative_score - 0.5) < 1e-9

    def test_summarise_placement_empty(self):
        s = summarise_placement([])
        assert s.n_placed == 0
        assert s.final_score == 0.0

    def test_summarise_placement_values(self):
        entries = self._make_entries()
        s = summarise_placement(entries)
        assert s.n_placed == 5
        assert abs(s.final_score - entries[-1].cumulative_score) < 1e-9

    def test_filter_positive_steps(self):
        entries = self._make_entries()
        pos = filter_positive_steps(entries)
        assert all(e.score_delta > 0.0 for e in pos)
        assert len(pos) == 4  # one negative delta (-0.05)

    def test_filter_by_min_score(self):
        entries = self._make_entries()
        filtered = filter_by_min_score(entries, min_score=0.5)
        assert all(e.cumulative_score >= 0.5 for e in filtered)

    def test_top_k_steps(self):
        entries = self._make_entries()
        top2 = top_k_steps(entries, k=2)
        assert len(top2) == 2
        assert top2[0].score_delta >= top2[1].score_delta

    def test_rank_fragments(self):
        entries = self._make_entries()
        ranked = rank_fragments(entries)
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_placement_score_stats_empty(self):
        stats = placement_score_stats([])
        assert stats["n"] == 0

    def test_placement_score_stats_keys(self):
        entries = self._make_entries()
        stats = placement_score_stats(entries)
        expected_keys = {"n", "final_score", "mean_delta", "std_delta",
                         "max_delta", "min_delta", "n_positive", "n_negative"}
        assert set(stats.keys()) == expected_keys


# ===========================================================================
# 4. polygon_ops_utils
# ===========================================================================

class TestPolygonOpsUtils:

    def _unit_square(self):
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def _triangle(self):
        return np.array([[0, 0], [4, 0], [0, 3]], dtype=float)

    def test_config_defaults(self):
        cfg = PolygonOpsConfig()
        assert cfg.clip_epsilon >= 0
        assert cfg.n_samples >= 1

    def test_config_invalid_raises(self):
        with pytest.raises(ValueError):
            PolygonOpsConfig(clip_epsilon=-1.0)

    def test_signed_area_ccw(self):
        sq = self._unit_square()
        a = signed_area(sq)
        assert a > 0  # CCW

    def test_polygon_area_unit_square(self):
        assert abs(polygon_area(self._unit_square()) - 1.0) < 1e-9

    def test_polygon_area_triangle(self):
        area = polygon_area(self._triangle())
        assert abs(area - 6.0) < 1e-9  # 0.5 * 4 * 3

    def test_polygon_perimeter_unit_square(self):
        p = polygon_perimeter(self._unit_square())
        assert abs(p - 4.0) < 1e-9

    def test_polygon_centroid_unit_square(self):
        c = polygon_centroid(self._unit_square())
        assert np.allclose(c, [0.5, 0.5], atol=1e-6)

    def test_polygon_bounding_box(self):
        bb = polygon_bounding_box(self._triangle())
        assert bb == (0.0, 0.0, 4.0, 3.0)

    def test_polygon_stats_returns_correct_type(self):
        stats = polygon_stats(self._unit_square())
        assert isinstance(stats, PolygonStats)
        assert stats.n_vertices == 4
        assert abs(stats.area - 1.0) < 1e-9

    def test_point_in_polygon_inside(self):
        sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        assert point_in_polygon(np.array([5.0, 5.0]), sq)

    def test_point_in_polygon_outside(self):
        sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        assert not point_in_polygon(np.array([20.0, 20.0]), sq)

    def test_polygon_overlap_overlapping(self):
        sq1 = self._unit_square()
        sq2 = sq1 + 0.5  # shifted 0.5 in both axes
        result = polygon_overlap(sq1, sq2)
        assert isinstance(result, PolygonOverlapResult)
        assert result.overlap
        assert result.iou > 0.0

    def test_polygon_overlap_non_overlapping(self):
        sq1 = self._unit_square()
        sq2 = sq1 + 10.0
        result = polygon_overlap(sq1, sq2)
        assert not result.overlap
        assert result.iou == 0.0

    def test_remove_collinear_square_unchanged(self):
        sq = self._unit_square()
        simplified = remove_collinear(sq, epsilon=1e-6)
        assert len(simplified) == 4

    def test_remove_collinear_removes_midpoints(self):
        # Square with midpoints on edges (collinear)
        poly = np.array([
            [0, 0], [5, 0], [10, 0],
            [10, 5], [10, 10],
            [5, 10], [0, 10],
            [0, 5],
        ], dtype=float)
        simplified = remove_collinear(poly, epsilon=1e-6)
        assert len(simplified) <= 4


# ===========================================================================
# 5. position_tracking_utils
# ===========================================================================

class TestPositionTrackingUtils:

    def _make_records(self):
        records = []
        coverages = rng.uniform(0.3, 0.9, size=6).tolist()
        methods = ["greedy", "greedy", "annealing", "annealing", "greedy", "annealing"]
        for i, (cov, method) in enumerate(zip(coverages, methods)):
            records.append(make_position_quality_record(
                run_id=i,
                n_fragments=20,
                n_placed=rng.integers(10, 20),
                mean_confidence=rng.uniform(0.5, 1.0),
                canvas_coverage=cov,
                method=method,
            ))
        return records

    def test_make_record_placement_rate(self):
        r = make_position_quality_record(0, 20, 15, 0.8, 0.7, "greedy")
        assert abs(r.placement_rate - 0.75) < 1e-9

    def test_make_record_zero_fragments(self):
        r = make_position_quality_record(0, 0, 0, 0.5, 0.0, "test")
        assert r.placement_rate == 0.0

    def test_summarise_empty(self):
        s = summarise_position_quality([])
        assert s.n_runs == 0
        assert s.best_run_id is None
        assert s.worst_run_id is None

    def test_summarise_n_runs(self):
        records = self._make_records()
        s = summarise_position_quality(records)
        assert s.n_runs == 6

    def test_summarise_total_fragments(self):
        records = self._make_records()
        s = summarise_position_quality(records)
        assert s.total_fragments == sum(r.n_fragments for r in records)

    def test_summarise_best_worst_run_ids(self):
        records = self._make_records()
        s = summarise_position_quality(records)
        best = max(records, key=lambda r: r.canvas_coverage)
        worst = min(records, key=lambda r: r.canvas_coverage)
        assert s.best_run_id == best.run_id
        assert s.worst_run_id == worst.run_id

    def test_filter_by_placement_rate(self):
        records = self._make_records()
        filtered = filter_by_placement_rate(records, min_rate=0.7)
        assert all(r.placement_rate >= 0.7 for r in filtered)

    def test_filter_by_method(self):
        records = self._make_records()
        greedy_records = filter_by_method(records, "greedy")
        assert all(r.method == "greedy" for r in greedy_records)
        assert len(greedy_records) == 3

    def test_top_k_position_records(self):
        records = self._make_records()
        top3 = top_k_position_records(records, k=3)
        assert len(top3) == 3
        coverages = [r.canvas_coverage for r in top3]
        assert coverages == sorted(coverages, reverse=True)

    def test_record_with_params(self):
        r = make_position_quality_record(
            run_id=99, n_fragments=10, n_placed=8,
            mean_confidence=0.9, canvas_coverage=0.75,
            method="custom", lr=0.01, epochs=50,
        )
        assert r.params["lr"] == 0.01
        assert r.params["epochs"] == 50

    def test_mean_coverage_in_summary(self):
        records = self._make_records()
        s = summarise_position_quality(records)
        expected = sum(r.canvas_coverage for r in records) / len(records)
        assert abs(s.mean_coverage - expected) < 1e-9
