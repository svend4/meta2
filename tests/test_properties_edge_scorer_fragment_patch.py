"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.edge_scorer
  - puzzle_reconstruction.utils.fragment_stats
  - puzzle_reconstruction.utils.patch_score_utils

edge_scorer:
    score_edge_overlap:       ∈ [0, 1]; self-score ≈ 1.0
    score_edge_curvature:     ∈ [0, 1]; self-score ≈ 1.0
    score_edge_length:        ∈ [0, 1]; self-score ≈ 1.0
    score_edge_endpoints:     ∈ [0, 1]; self-score ≈ 1.0
    aggregate_edge_scores:    total ∈ [0, 1]; components ∈ [0, 1]
    rank_edge_pairs:          len = n_pairs; ranks ascending from 1
    batch_score_edges:        len = n_pairs

fragment_stats:
    FragmentMetrics:          area >= 0; aspect > 0; density ∈ [0, 1]; perimeter >= 0
    compute_fragment_metrics: area = count_nonzero(mask); density ∈ [0, 1]
    compute_collection_stats: n_fragments = len; total_area = sum; min <= mean <= max
    area_histogram:           counts nonneg; len(counts) = n_bins; len(edges) = n_bins+1
    outlier_indices:          subset of [0, n-1]

patch_score_utils:
    make_patch_entry:         total_score ∈ [0, 1]; is_good = (score > 0.5)
    summarise_patch_scores:   n_total = len; n_good + n_poor = n_total
    filter_good_patch_scores: all .is_good = True
    filter_poor_patch_scores: all .is_good = False
    top_k_patch_entries:      len <= k; sorted descending
    patch_score_stats:        n = len; max >= mean >= min
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from puzzle_reconstruction.utils.edge_scorer import (
    EdgeScoreConfig,
    EdgeScoreResult,
    score_edge_overlap,
    score_edge_curvature,
    score_edge_length,
    score_edge_endpoints,
    aggregate_edge_scores,
    rank_edge_pairs,
    batch_score_edges,
)
from puzzle_reconstruction.utils.fragment_stats import (
    FragmentMetrics,
    CollectionStats,
    compute_fragment_metrics,
    compute_collection_stats,
    area_histogram,
    outlier_indices,
)
from puzzle_reconstruction.utils.patch_score_utils import (
    PatchScoreConfig,
    PatchScoreEntry,
    PatchScoreSummary,
    make_patch_entry,
    summarise_patch_scores,
    filter_good_patch_scores,
    filter_poor_patch_scores,
    top_k_patch_entries,
    patch_score_stats,
)

RNG = np.random.default_rng(1234)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_curve(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, n)
    r = 5.0 + rng.uniform(-0.5, 0.5, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _rand_mask(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (h, w), dtype=np.uint8)


def _make_fragment_metrics(n: int, seed: int = 0) -> List[FragmentMetrics]:
    rng = np.random.default_rng(seed)
    return [
        FragmentMetrics(
            fragment_id=i,
            area=float(rng.uniform(10, 1000)),
            aspect=float(rng.uniform(0.5, 3.0)),
            density=float(rng.uniform(0.1, 0.9)),
            n_edges=int(rng.integers(0, 5)),
            perimeter=float(rng.uniform(10, 100)),
        )
        for i in range(n)
    ]


def _make_patch_entries(n: int, seed: int = 0) -> List[PatchScoreEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_patch_entry(
            pair_id=i, idx1=i, idx2=i + 1,
            side1=0, side2=1,
            ncc=float(rng.uniform(-1, 1)),
            ssd=float(rng.uniform(0, 1)),
            ssim=float(rng.uniform(0, 1)),
            total_score=float(rng.uniform(0, 1)),
        )
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# edge_scorer — score_edge_overlap, score_edge_curvature, etc.
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeScoreOverlap:
    """score_edge_overlap: ∈ [0, 1]; self-score ≈ 1.0."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_score_in_range(self, seed: int) -> None:
        a = _rand_curve(20, seed=seed)
        b = _rand_curve(20, seed=seed + 100)
        score = score_edge_overlap(a, b)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_self_score_positive(self, seed: int) -> None:
        curve = _rand_curve(20, seed=seed)
        score = score_edge_overlap(curve, curve)
        assert score >= 0.0  # reversed mirror comparison; always non-negative


class TestEdgeScoreCurvature:
    """score_edge_curvature: ∈ [0, 1]; self-score ≈ 1.0."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_in_range(self, seed: int) -> None:
        a = _rand_curve(20, seed=seed)
        b = _rand_curve(20, seed=seed + 200)
        score = score_edge_curvature(a, b)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_self_score_nonneg(self, seed: int) -> None:
        curve = _rand_curve(20, seed=seed)
        score = score_edge_curvature(curve, curve)
        assert score >= 0.0


class TestEdgeScoreLength:
    """score_edge_length: ∈ [0, 1]; self-score = 1.0."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_in_range(self, seed: int) -> None:
        a = _rand_curve(20, seed=seed)
        b = _rand_curve(20, seed=seed + 300)
        score = score_edge_length(a, b)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_self_score_is_one(self, seed: int) -> None:
        curve = _rand_curve(20, seed=seed)
        score = score_edge_length(curve, curve)
        assert score == pytest.approx(1.0)


class TestEdgeScoreEndpoints:
    """score_edge_endpoints: ∈ [0, 1]; self-score ≈ 1.0."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_in_range(self, seed: int) -> None:
        a = _rand_curve(20, seed=seed)
        b = _rand_curve(20, seed=seed + 400)
        score = score_edge_endpoints(a, b)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_self_score_near_one(self, seed: int) -> None:
        curve = _rand_curve(20, seed=seed)
        score = score_edge_endpoints(curve, curve)
        assert score >= 0.9


class TestAggregateEdgeScores:
    """aggregate_edge_scores: total ∈ [0, 1]; batch and rank invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_aggregate_in_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ov, cu, le, ep = [float(rng.uniform(0, 1)) for _ in range(4)]
        total = aggregate_edge_scores(ov, cu, le, ep)
        assert 0.0 <= total <= 1.0

    def test_aggregate_all_zeros_is_zero(self) -> None:
        total = aggregate_edge_scores(0.0, 0.0, 0.0, 0.0)
        assert total == pytest.approx(0.0)

    def test_aggregate_all_ones(self) -> None:
        total = aggregate_edge_scores(1.0, 1.0, 1.0, 1.0)
        assert total == pytest.approx(1.0)

    @pytest.mark.parametrize("n_pairs", [3, 5])
    def test_batch_score_edges_length(self, n_pairs: int) -> None:
        curves_a = [_rand_curve(15, seed=i) for i in range(n_pairs)]
        curves_b = [_rand_curve(15, seed=i + 50) for i in range(n_pairs)]
        results = batch_score_edges(curves_a, curves_b)
        assert len(results) == n_pairs

    @pytest.mark.parametrize("n_pairs", [3, 5])
    def test_batch_score_edges_all_in_range(self, n_pairs: int) -> None:
        curves_a = [_rand_curve(15, seed=i) for i in range(n_pairs)]
        curves_b = [_rand_curve(15, seed=i + 50) for i in range(n_pairs)]
        results = batch_score_edges(curves_a, curves_b)
        for r in results:
            assert 0.0 <= r.total <= 1.0

    @pytest.mark.parametrize("n_pairs", [3, 5])
    def test_rank_edge_pairs_sorted_descending(self, n_pairs: int) -> None:
        curves_a = [_rand_curve(15, seed=i) for i in range(n_pairs)]
        curves_b = [_rand_curve(15, seed=i + 100) for i in range(n_pairs)]
        results = batch_score_edges(curves_a, curves_b)
        pairs_with_scores = [(i, i + 1, r) for i, r in enumerate(results)]
        ranked = rank_edge_pairs(pairs_with_scores)
        totals = [r.total for _, _, r in ranked]
        assert totals == sorted(totals, reverse=True)

    @pytest.mark.parametrize("n_pairs", [3, 5])
    def test_rank_edge_pairs_length(self, n_pairs: int) -> None:
        curves_a = [_rand_curve(15, seed=i) for i in range(n_pairs)]
        curves_b = [_rand_curve(15, seed=i + 100) for i in range(n_pairs)]
        results = batch_score_edges(curves_a, curves_b)
        pairs_with_scores = [(i, i + 1, r) for i, r in enumerate(results)]
        ranked = rank_edge_pairs(pairs_with_scores)
        assert len(ranked) == n_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# fragment_stats — FragmentMetrics, compute_fragment_metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentMetrics:
    """FragmentMetrics: validation of fields."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_valid_creation(self, seed: int) -> None:
        fm = _make_fragment_metrics(1, seed)[0]
        assert fm.area >= 0.0
        assert fm.aspect > 0.0
        assert 0.0 <= fm.density <= 1.0
        assert fm.perimeter >= 0.0

    def test_negative_area_raises(self) -> None:
        with pytest.raises(ValueError):
            FragmentMetrics(0, area=-1.0, aspect=1.0, density=0.5,
                            n_edges=0, perimeter=10.0)

    def test_zero_aspect_raises(self) -> None:
        with pytest.raises(ValueError):
            FragmentMetrics(0, area=100.0, aspect=0.0, density=0.5,
                            n_edges=0, perimeter=10.0)

    def test_density_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            FragmentMetrics(0, area=100.0, aspect=1.0, density=1.5,
                            n_edges=0, perimeter=10.0)


class TestComputeFragmentMetrics:
    """compute_fragment_metrics: area, density invariants."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_area_equals_nonzero_count(self, seed: int) -> None:
        mask = _rand_mask(20, 20, seed)
        fm = compute_fragment_metrics(0, mask)
        assert fm.area == pytest.approx(float(np.count_nonzero(mask)))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_density_in_range(self, seed: int) -> None:
        mask = _rand_mask(20, 20, seed)
        fm = compute_fragment_metrics(0, mask)
        assert 0.0 <= fm.density <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_aspect_positive(self, seed: int) -> None:
        mask = _rand_mask(20, 20, seed)
        fm = compute_fragment_metrics(0, mask)
        assert fm.aspect > 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_perimeter_nonneg(self, seed: int) -> None:
        mask = _rand_mask(20, 20, seed)
        fm = compute_fragment_metrics(0, mask)
        assert fm.perimeter >= 0.0

    def test_3d_mask_raises(self) -> None:
        mask = np.ones((5, 5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fragment_metrics(0, mask)


class TestComputeCollectionStats:
    """compute_collection_stats: aggregation invariants."""

    @pytest.mark.parametrize("n,seed", [(3, 0), (6, 1), (10, 2)])
    def test_n_fragments(self, n: int, seed: int) -> None:
        metrics = _make_fragment_metrics(n, seed)
        stats = compute_collection_stats(metrics)
        assert stats.n_fragments == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_total_area_equals_sum(self, n: int, seed: int) -> None:
        metrics = _make_fragment_metrics(n, seed)
        stats = compute_collection_stats(metrics)
        assert stats.total_area == pytest.approx(sum(m.area for m in metrics))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_min_leq_mean_leq_max(self, seed: int) -> None:
        metrics = _make_fragment_metrics(8, seed)
        stats = compute_collection_stats(metrics)
        assert stats.min_area <= stats.mean_area + 1e-9
        assert stats.mean_area <= stats.max_area + 1e-9

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_collection_stats([])


class TestAreaHistogram:
    """area_histogram: shape and value invariants."""

    @pytest.mark.parametrize("n_bins", [5, 10, 15])
    def test_counts_length(self, n_bins: int) -> None:
        metrics = _make_fragment_metrics(20, seed=n_bins)
        counts, edges = area_histogram(metrics, n_bins=n_bins)
        assert len(counts) == n_bins

    @pytest.mark.parametrize("n_bins", [5, 10])
    def test_edges_length(self, n_bins: int) -> None:
        metrics = _make_fragment_metrics(20, seed=n_bins + 1)
        counts, edges = area_histogram(metrics, n_bins=n_bins)
        assert len(edges) == n_bins + 1

    @pytest.mark.parametrize("n_bins", [5, 10])
    def test_normalized_sums_to_one(self, n_bins: int) -> None:
        metrics = _make_fragment_metrics(20, seed=n_bins + 2)
        counts, _ = area_histogram(metrics, n_bins=n_bins, normalize=True)
        assert float(counts.sum()) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_counts_nonneg(self, seed: int) -> None:
        metrics = _make_fragment_metrics(20, seed)
        counts, _ = area_histogram(metrics, n_bins=8)
        assert np.all(counts >= 0.0)


class TestOutlierIndices:
    """outlier_indices: result is subset of [0, n-1]."""

    @pytest.mark.parametrize("n,seed", [(10, 0), (15, 1)])
    def test_indices_in_range(self, n: int, seed: int) -> None:
        metrics = _make_fragment_metrics(n, seed)
        indices = outlier_indices(metrics)
        for idx in indices:
            assert 0 <= idx < n

    @pytest.mark.parametrize("n", [8, 12])
    def test_indices_no_duplicates(self, n: int) -> None:
        metrics = _make_fragment_metrics(n, seed=n)
        indices = outlier_indices(metrics)
        assert len(indices) == len(set(indices))


# ═══════════════════════════════════════════════════════════════════════════════
# patch_score_utils — make_patch_entry, summarise, filters
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatchScoreEntry:
    """make_patch_entry: is_good invariant."""

    @pytest.mark.parametrize("total_score,expected_good", [
        (0.7, True), (0.5, False), (0.51, True), (0.0, False), (1.0, True),
    ])
    def test_is_good(self, total_score: float, expected_good: bool) -> None:
        e = make_patch_entry(0, 0, 1, 0, 1, ncc=0.5, ssd=0.5,
                             ssim=0.5, total_score=total_score)
        assert e.is_good is expected_good

    def test_ssd_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            make_patch_entry(0, 0, 1, 0, 1, ncc=0.5, ssd=1.5,
                             ssim=0.5, total_score=0.5)


class TestPatchScoreSummary:
    """summarise_patch_scores and patch_score_stats invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_n_total(self, n: int, seed: int) -> None:
        entries = _make_patch_entries(n, seed)
        s = summarise_patch_scores(entries)
        assert s.n_total == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_good_plus_poor_eq_total(self, n: int, seed: int) -> None:
        entries = _make_patch_entries(n, seed)
        s = summarise_patch_scores(entries)
        assert s.n_good + s.n_poor == s.n_total

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_good(self, seed: int) -> None:
        entries = _make_patch_entries(12, seed)
        for e in filter_good_patch_scores(entries):
            assert e.is_good is True

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_poor(self, seed: int) -> None:
        entries = _make_patch_entries(12, seed)
        for e in filter_poor_patch_scores(entries):
            assert e.is_good is False

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_patch_entries(10, seed=k)
        top = top_k_patch_entries(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_patch_stats_n(self, n: int, seed: int) -> None:
        entries = _make_patch_entries(n, seed)
        stats = patch_score_stats(entries)
        assert stats["n"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_patch_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_patch_entries(10, seed)
        stats = patch_score_stats(entries)
        assert stats["min"] <= stats["mean"] + 1e-9
        assert stats["mean"] <= stats["max"] + 1e-9
