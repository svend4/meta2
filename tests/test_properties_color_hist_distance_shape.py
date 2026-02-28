"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.color_hist_utils
  - puzzle_reconstruction.utils.distance_shape_utils
  - puzzle_reconstruction.utils.geometry_utils

color_hist_utils:
    make_color_hist_entry:     score = (intersection + chi2) / 2
    summarise_color_hist:      n_entries = len; min <= mean <= max
    filter_good_hist_entries:  all .score >= threshold
    filter_poor_hist_entries:  all .score < threshold
    top_k_hist_entries:        len <= k; sorted descending
    color_hist_stats:          count = n; max >= mean >= min

distance_shape_utils:
    DistanceMatrixRecord:      n = matrix.shape[0]; max_value = mat.max()
    SimilarityPair:            similarity ∈ [0, 1]; is_high = (sim >= 0.5)
    ContourMatchRecord:        cost >= 0; similarity ∈ [0, 1]; is_match = (sim >= 0.5)
    MetricsRunRecord:          precision/recall/f1 ∈ [0, 1]; is_perfect iff all = 1
    EvidenceAggregationRecord: confidence ∈ [0, 1]; is_confident = (conf >= 0.5)

geometry_utils:
    BoundingBox.area:          >= 0; = width * height
    BoundingBox.iou:           ∈ [0, 1]; self-iou = 1; commutative; no-overlap = 0
    BoundingBox.aspect_ratio:  >= 0
    bbox_from_points:          contains all input points
    summarize_overlaps:        n_pairs = len; n_conflicting <= n_pairs; mean ∈ [0, 1]
    GeometryComparisonRecord:  all scores ∈ [0, 1]
    rank_geometry_comparisons: len = len(records); ranks = 1..n; sorted descending
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.color_hist_utils import (
    ColorHistConfig,
    ColorHistEntry,
    ColorHistSummary,
    make_color_hist_entry,
    entries_from_comparisons,
    summarise_color_hist,
    filter_good_hist_entries,
    filter_poor_hist_entries,
    top_k_hist_entries,
    color_hist_stats,
)
from puzzle_reconstruction.utils.distance_shape_utils import (
    DistanceMatrixRecord,
    SimilarityPair,
    DistanceBatchResult,
    ContourMatchRecord,
    MetricsRunRecord,
    EvidenceAggregationRecord,
    make_distance_record,
    make_contour_match,
)
from puzzle_reconstruction.utils.geometry_utils import (
    BoundingBox,
    OverlapSummary,
    GeometryComparisonRecord,
    bbox_from_points,
    summarize_overlaps,
    rank_geometry_comparisons,
)

RNG = np.random.default_rng(9999)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_hist_entry(seed: int = 0) -> ColorHistEntry:
    rng = np.random.default_rng(seed)
    return make_color_hist_entry(
        frag_i=0, frag_j=1,
        intersection=float(rng.uniform(0, 1)),
        chi2=float(rng.uniform(0, 1)),
    )


def _make_hist_entries(n: int, seed: int = 0) -> List[ColorHistEntry]:
    return [_rand_hist_entry(seed + i) for i in range(n)]


def _rand_bbox(seed: int = 0) -> BoundingBox:
    rng = np.random.default_rng(seed)
    return BoundingBox(
        x=float(rng.uniform(0, 50)),
        y=float(rng.uniform(0, 50)),
        width=float(rng.uniform(1, 30)),
        height=float(rng.uniform(1, 30)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# color_hist_utils — score, summarise, filters
# ═══════════════════════════════════════════════════════════════════════════════

class TestColorHistEntry:
    """ColorHistEntry.score and make_color_hist_entry invariants."""

    @pytest.mark.parametrize("inter,chi2", [
        (0.6, 0.4), (0.0, 1.0), (1.0, 0.0), (0.5, 0.5),
    ])
    def test_score_is_mean_of_intersection_chi2(self, inter: float, chi2: float) -> None:
        e = make_color_hist_entry(0, 1, intersection=inter, chi2=chi2)
        assert e.score == pytest.approx((inter + chi2) / 2.0)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_score_in_range(self, seed: int) -> None:
        e = _rand_hist_entry(seed)
        # intersection and chi2 are in [0,1], so score in [0,1]
        assert 0.0 <= e.score <= 1.0


class TestColorHistSummarise:
    """summarise_color_hist and filter invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_n_entries(self, n: int, seed: int) -> None:
        entries = _make_hist_entries(n, seed)
        s = summarise_color_hist(entries)
        assert s.n_entries == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_hist_entries(8, seed)
        s = summarise_color_hist(entries)
        assert s.min_score <= s.mean_score + 1e-9
        assert s.mean_score <= s.max_score + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_good_all_above_threshold(self, seed: int) -> None:
        entries = _make_hist_entries(15, seed)
        good = filter_good_hist_entries(entries, threshold=0.7)
        for e in good:
            assert e.score >= 0.7

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_poor_all_below_threshold(self, seed: int) -> None:
        entries = _make_hist_entries(15, seed)
        poor = filter_poor_hist_entries(entries, threshold=0.3)
        for e in poor:
            assert e.score < 0.3

    @pytest.mark.parametrize("k", [2, 4, 6])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_hist_entries(10, seed=k)
        top = top_k_hist_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("k", [3, 5])
    def test_top_k_sorted_descending(self, k: int) -> None:
        entries = _make_hist_entries(10, seed=k + 1)
        top = top_k_hist_entries(entries, k)
        scores = [e.score for e in top]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_color_hist_stats_count(self, n: int, seed: int) -> None:
        entries = _make_hist_entries(n, seed)
        stats = color_hist_stats(entries)
        assert stats["count"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_color_hist_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_hist_entries(10, seed)
        stats = color_hist_stats(entries)
        assert stats["min"] <= stats["mean"] + 1e-9
        assert stats["mean"] <= stats["max"] + 1e-9

    @pytest.mark.parametrize("n", [3, 5])
    def test_entries_from_comparisons_length(self, n: int) -> None:
        pairs = [(i, i + 1) for i in range(n)]
        inters = [0.5] * n
        chi2s = [0.5] * n
        entries = entries_from_comparisons(pairs, inters, chi2s)
        assert len(entries) == n


# ═══════════════════════════════════════════════════════════════════════════════
# distance_shape_utils — DistanceMatrixRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistanceMatrixRecord:
    """DistanceMatrixRecord: n, max_value, min_offdiag invariants."""

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_n_equals_matrix_size(self, n: int) -> None:
        mat = np.eye(n) * 0.5
        rec = make_distance_record("test", "euclidean", mat)
        assert rec.n == n

    @pytest.mark.parametrize("n", [3, 4])
    def test_max_value(self, n: int) -> None:
        rng = np.random.default_rng(n)
        mat = rng.uniform(0, 1, (n, n))
        rec = make_distance_record("test", "l2", mat)
        assert rec.max_value == pytest.approx(float(mat.max()))

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError):
            DistanceMatrixRecord("test", "l2", np.ones((3, 4)))

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_min_offdiag_excludes_diagonal(self, n: int) -> None:
        mat = np.eye(n) * 100.0  # diagonal = 100
        # Set off-diagonal to small values
        for i in range(n):
            for j in range(n):
                if i != j:
                    mat[i, j] = 0.1
        rec = make_distance_record("test", "l2", mat)
        assert rec.min_offdiag == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# distance_shape_utils — SimilarityPair, ContourMatchRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimilarityPair:
    """SimilarityPair: similarity ∈ [0, 1]; is_high invariant."""

    @pytest.mark.parametrize("sim,expected_high", [
        (0.7, True), (0.5, True), (0.49, False), (0.0, False), (1.0, True),
    ])
    def test_is_high(self, sim: float, expected_high: bool) -> None:
        sp = SimilarityPair(i=0, j=1, similarity=sim)
        assert sp.is_high is expected_high

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError):
            SimilarityPair(i=-1, j=0, similarity=0.5)

    def test_similarity_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            SimilarityPair(i=0, j=1, similarity=1.5)
        with pytest.raises(ValueError):
            SimilarityPair(i=0, j=1, similarity=-0.1)


class TestContourMatchRecord:
    """ContourMatchRecord: cost >= 0; similarity ∈ [0, 1]; is_match invariant."""

    @pytest.mark.parametrize("sim,expected_match", [
        (0.6, True), (0.5, True), (0.4, False), (0.0, False),
    ])
    def test_is_match(self, sim: float, expected_match: bool) -> None:
        rec = make_contour_match(0, 1, cost=1.0, n_corr=10, similarity=sim)
        assert rec.is_match is expected_match

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourMatchRecord(0, 1, cost=-0.1, n_correspondences=5)

    @pytest.mark.parametrize("cost,sim", [(0.5, 0.7), (0.0, 1.0), (10.0, 0.3)])
    def test_valid_creation(self, cost: float, sim: float) -> None:
        rec = make_contour_match(0, 1, cost=cost, n_corr=5, similarity=sim)
        assert rec.cost >= 0.0
        assert 0.0 <= rec.similarity <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# distance_shape_utils — MetricsRunRecord, EvidenceAggregationRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsRunRecord:
    """MetricsRunRecord: precision/recall/f1 ∈ [0, 1]; is_perfect."""

    @pytest.mark.parametrize("p,r,f1,expected_perfect", [
        (1.0, 1.0, 1.0, True), (0.9, 0.8, 0.85, False), (0.0, 0.0, 0.0, False),
    ])
    def test_is_perfect(self, p: float, r: float, f1: float,
                        expected_perfect: bool) -> None:
        rec = MetricsRunRecord("run1", precision=p, recall=r, f1=f1,
                               n_fragments=5)
        assert rec.is_perfect is expected_perfect

    def test_out_of_range_precision_raises(self) -> None:
        with pytest.raises(ValueError):
            MetricsRunRecord("run1", precision=1.1, recall=0.5, f1=0.5,
                             n_fragments=5)

    def test_out_of_range_f1_raises(self) -> None:
        with pytest.raises(ValueError):
            MetricsRunRecord("run1", precision=0.8, recall=0.8, f1=-0.1,
                             n_fragments=5)


class TestEvidenceAggregationRecord:
    """EvidenceAggregationRecord: confidence ∈ [0, 1]; is_confident."""

    @pytest.mark.parametrize("conf,expected_confident", [
        (0.8, True), (0.5, True), (0.4, False), (0.0, False),
    ])
    def test_is_confident(self, conf: float, expected_confident: bool) -> None:
        rec = EvidenceAggregationRecord(
            step=0, pair_id=(0, 1), n_channels=3, confidence=conf
        )
        assert rec.is_confident is expected_confident

    def test_out_of_range_confidence_raises(self) -> None:
        with pytest.raises(ValueError):
            EvidenceAggregationRecord(step=0, pair_id=(0, 1),
                                      n_channels=3, confidence=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# geometry_utils — BoundingBox
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundingBox:
    """BoundingBox: area, iou, aspect_ratio, center."""

    @pytest.mark.parametrize("w,h", [(5.0, 3.0), (0.0, 10.0), (7.0, 0.0)])
    def test_area_nonneg(self, w: float, h: float) -> None:
        bb = BoundingBox(0, 0, w, h)
        assert bb.area >= 0.0

    @pytest.mark.parametrize("w,h", [(5.0, 3.0), (4.0, 8.0)])
    def test_area_equals_width_times_height(self, w: float, h: float) -> None:
        bb = BoundingBox(0, 0, w, h)
        assert bb.area == pytest.approx(w * h)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_self_iou_is_one(self, seed: int) -> None:
        bb = _rand_bbox(seed)
        assert bb.iou(bb) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_iou_commutative(self, seed: int) -> None:
        a = _rand_bbox(seed)
        b = _rand_bbox(seed + 10)
        assert a.iou(b) == pytest.approx(b.iou(a))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_iou_in_range(self, seed: int) -> None:
        a = _rand_bbox(seed)
        b = _rand_bbox(seed + 20)
        iou = a.iou(b)
        assert 0.0 <= iou <= 1.0 + 1e-9

    def test_non_overlapping_iou_zero(self) -> None:
        a = BoundingBox(0, 0, 5, 5)
        b = BoundingBox(100, 100, 5, 5)
        assert a.iou(b) == pytest.approx(0.0)

    @pytest.mark.parametrize("w,h", [(4.0, 2.0), (0.0, 5.0)])
    def test_aspect_ratio_nonneg(self, w: float, h: float) -> None:
        bb = BoundingBox(0, 0, w, h)
        assert bb.aspect_ratio >= 0.0

    def test_negative_width_raises(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(0, 0, -1.0, 5.0)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_center(self, seed: int) -> None:
        bb = _rand_bbox(seed)
        cx, cy = bb.center
        assert cx == pytest.approx(bb.x + bb.width / 2.0)
        assert cy == pytest.approx(bb.y + bb.height / 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# geometry_utils — bbox_from_points, summarize_overlaps, rank_geometry_comparisons
# ═══════════════════════════════════════════════════════════════════════════════

class TestBboxFromPoints:
    """bbox_from_points: contains all input points."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_all_points_within_bbox(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = [(float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)))
               for _ in range(8)]
        bb = bbox_from_points(pts)
        for x, y in pts:
            assert bb.x - 1e-9 <= x <= bb.x + bb.width + 1e-9
            assert bb.y - 1e-9 <= y <= bb.y + bb.height + 1e-9

    def test_single_point_zero_area(self) -> None:
        bb = bbox_from_points([(3.0, 4.0)])
        assert bb.area == pytest.approx(0.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            bbox_from_points([])


class TestSummarizeOverlaps:
    """summarize_overlaps: n_pairs, n_conflicting, mean_iou ∈ [0, 1]."""

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_n_pairs(self, n: int) -> None:
        iou_values = [0.1 * i for i in range(n)]
        s = summarize_overlaps(iou_values)
        assert s.n_pairs == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_n_conflicting_leq_n_pairs(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        iou_values = [float(rng.uniform(0, 1)) for _ in range(10)]
        s = summarize_overlaps(iou_values)
        assert 0 <= s.n_conflicting <= s.n_pairs

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_mean_iou_in_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        iou_values = [float(rng.uniform(0, 1)) for _ in range(10)]
        s = summarize_overlaps(iou_values)
        assert 0.0 <= s.mean_iou <= 1.0

    def test_empty_iou_values(self) -> None:
        s = summarize_overlaps([])
        assert s.n_pairs == 0
        assert s.mean_iou == 0.0


class TestRankGeometryComparisons:
    """rank_geometry_comparisons: ranks, sorted order."""

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_length_preserved(self, n: int) -> None:
        records = [
            GeometryComparisonRecord(i, i + 1, 0.5, 0.5, 0.5)
            for i in range(n)
        ]
        ranked = rank_geometry_comparisons(records)
        assert len(ranked) == n

    @pytest.mark.parametrize("n", [4, 6])
    def test_ranks_are_1_to_n(self, n: int) -> None:
        rng = np.random.default_rng(n)
        records = [
            GeometryComparisonRecord(
                i, i + 1,
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
            )
            for i in range(n)
        ]
        ranked = rank_geometry_comparisons(records)
        ranks = [r for r, _ in ranked]
        assert ranks == list(range(1, n + 1))

    @pytest.mark.parametrize("n", [4, 6])
    def test_sorted_by_total_score_descending(self, n: int) -> None:
        rng = np.random.default_rng(n + 100)
        records = [
            GeometryComparisonRecord(
                i, i + 1,
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
            )
            for i in range(n)
        ]
        ranked = rank_geometry_comparisons(records)
        scores = [r.total_score for _, r in ranked]
        assert scores == sorted(scores, reverse=True)
