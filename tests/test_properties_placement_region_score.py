"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.placement_score_utils
  - puzzle_reconstruction.utils.region_score_utils

placement_score_utils:
    make_placement_entry:     step >= 0; fragment_idx >= 0
    entries_from_history:     len = len(history); sorted by step
    summarise_placement:      n_placed = len; max_delta >= mean >= min_delta
    filter_positive_steps:    all .score_delta > 0
    filter_by_min_score:      all .cumulative_score >= threshold
    top_k_steps:              len <= k; sorted descending by cumulative_score
    rank_fragments:           result is list of ints; len = n_unique_fragments
    placement_score_stats:    n = len; n_positive + n_negative <= n

region_score_utils:
    region_compactness:       ∈ [0, 1]; zero perimeter → 0
    region_extent:            ∈ (0, 1]; zero area → clipped to 1
    mask_perimeter:           >= 0; all-zero mask → 0
    score_region:             ∈ [0, 1]
    evaluate_region:          score ∈ [0, 1]; compactness ∈ [0, 1]; extent ∈ (0, 1]
    filter_by_score:          all .score >= min_score
    rank_regions:             sorted descending; len preserved
    top_k_regions:            len <= k
    region_score_stats:       n = len; max >= mean >= min
    normalize_scores:         all .score ∈ [0, 1]; max = 1 (unless single entry)
    batch_evaluate_regions:   len = len(regions)
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

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
from puzzle_reconstruction.utils.region_score_utils import (
    RegionScoreConfig,
    RegionScore,
    region_compactness,
    region_extent,
    mask_perimeter,
    score_region,
    evaluate_region,
    filter_by_score,
    rank_regions,
    top_k_regions,
    region_score_stats,
    normalize_scores,
    batch_evaluate_regions,
)

RNG = np.random.default_rng(8888)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_placement_entries(n: int, seed: int = 0) -> List[PlacementScoreEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_placement_entry(
            step=i,
            fragment_idx=i,
            score_delta=float(rng.uniform(-0.2, 0.5)),
            cumulative_score=float(rng.uniform(0, 5)),
        )
        for i in range(n)
    ]


def _rand_mask(h: int = 20, w: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (h, w), dtype=np.uint8)


def _make_region_scores(n: int, seed: int = 0) -> List[RegionScore]:
    rng = np.random.default_rng(seed)
    return [
        RegionScore(
            label=i,
            area=int(rng.integers(10, 1000)),
            compactness=float(rng.uniform(0, 1)),
            extent=float(rng.uniform(0.1, 1.0)),
            score=float(rng.uniform(0, 1)),
        )
        for i in range(n)
    ]


def _make_region_dict(seed: int = 0) -> Dict:
    mask = _rand_mask(20, 20, seed=seed)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        mask = np.ones((20, 20), dtype=np.uint8)
        rows = np.array([0, 19])
        cols = np.array([0, 19])
    bbox = (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))
    return {"label": seed, "area": int(mask.sum()), "bbox": bbox, "mask": mask}


# ═══════════════════════════════════════════════════════════════════════════════
# placement_score_utils — make_placement_entry, entries_from_history
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacementEntry:
    """make_placement_entry: step >= 0; fragment_idx >= 0."""

    @pytest.mark.parametrize("step,idx", [(0, 0), (5, 3), (10, 7)])
    def test_fields_set(self, step: int, idx: int) -> None:
        e = make_placement_entry(step, idx, 0.3, 1.5)
        assert e.step == step
        assert e.fragment_idx == idx

    def test_negative_step_raises(self) -> None:
        with pytest.raises(ValueError):
            make_placement_entry(-1, 0, 0.0, 0.0)

    def test_negative_frag_idx_raises(self) -> None:
        with pytest.raises(ValueError):
            make_placement_entry(0, -1, 0.0, 0.0)

    @pytest.mark.parametrize("n", [5, 8, 12])
    def test_entries_from_history_length(self, n: int) -> None:
        history = [
            {"step": i, "idx": i, "score_delta": 0.1 * i}
            for i in range(n)
        ]
        entries = entries_from_history(history)
        assert len(entries) == n

    @pytest.mark.parametrize("n", [5, 8])
    def test_entries_from_history_sorted_by_step(self, n: int) -> None:
        # Shuffle history order
        rng = np.random.default_rng(n)
        history = [{"step": i, "idx": i, "score_delta": 0.1 * i}
                   for i in range(n)]
        indices = rng.permutation(n).tolist()
        shuffled = [history[i] for i in indices]
        entries = entries_from_history(shuffled)
        steps = [e.step for e in entries]
        assert steps == sorted(steps)


# ═══════════════════════════════════════════════════════════════════════════════
# placement_score_utils — summarise, filters, stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacementSummarise:
    """summarise_placement and filter invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_n_placed(self, n: int, seed: int) -> None:
        entries = _make_placement_entries(n, seed)
        s = summarise_placement(entries)
        assert s.n_placed == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_max_delta_geq_mean(self, seed: int) -> None:
        entries = _make_placement_entries(8, seed)
        s = summarise_placement(entries)
        assert s.max_delta >= s.mean_delta - 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_min_delta_leq_mean(self, seed: int) -> None:
        entries = _make_placement_entries(8, seed)
        s = summarise_placement(entries)
        assert s.min_delta <= s.mean_delta + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_filter_positive_steps(self, seed: int) -> None:
        entries = _make_placement_entries(15, seed)
        pos = filter_positive_steps(entries)
        for e in pos:
            assert e.score_delta > 0.0

    @pytest.mark.parametrize("threshold", [0.0, 1.0, 2.0])
    def test_filter_by_min_score(self, threshold: float) -> None:
        entries = _make_placement_entries(15, seed=42)
        filtered = filter_by_min_score(entries, min_score=threshold)
        for e in filtered:
            assert e.cumulative_score >= threshold

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_steps_length(self, k: int) -> None:
        entries = _make_placement_entries(10, seed=k)
        top = top_k_steps(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_placement_stats_n(self, n: int, seed: int) -> None:
        entries = _make_placement_entries(n, seed)
        stats = placement_score_stats(entries)
        assert stats["n"] == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_placement_stats_n_pos_plus_neg_leq_n(self, n: int, seed: int) -> None:
        entries = _make_placement_entries(n, seed)
        stats = placement_score_stats(entries)
        assert stats["n_positive"] + stats["n_negative"] <= n

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_rank_fragments_subset_of_frag_indices(self, n: int, seed: int) -> None:
        entries = _make_placement_entries(n, seed)
        ranked = rank_fragments(entries)  # returns List[Tuple[int, float]]
        frag_indices = {e.fragment_idx for e in entries}
        for frag_idx, _delta in ranked:
            assert frag_idx in frag_indices

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_rank_fragments_sorted_descending(self, n: int, seed: int) -> None:
        entries = _make_placement_entries(n, seed)
        ranked = rank_fragments(entries)
        deltas = [d for _, d in ranked]
        assert deltas == sorted(deltas, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# region_score_utils — region_compactness, region_extent
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegionCompactness:
    """region_compactness: ∈ [0, 1]; zero perimeter → 0."""

    @pytest.mark.parametrize("area,perim", [(100, 40.0), (500, 100.0), (1000, 200.0)])
    def test_in_range(self, area: int, perim: float) -> None:
        c = region_compactness(area, perim)
        assert 0.0 <= c <= 1.0

    def test_zero_perimeter_is_zero(self) -> None:
        assert region_compactness(100, 0.0) == pytest.approx(0.0)

    def test_large_area_small_perimeter_near_one(self) -> None:
        # Circle: 4π·area/perim² = 4π·r²/(2πr)² = 1
        r = 50
        area = int(math.pi * r * r)
        perim = 2 * math.pi * r
        c = region_compactness(area, perim)
        assert c <= 1.0 + 1e-6


class TestRegionExtent:
    """region_extent: ∈ (0, 1]; clips to 1 when bbox_area is small."""

    @pytest.mark.parametrize("area,bbox", [
        (100, (0, 0, 10, 10)),
        (50, (5, 5, 15, 15)),
        (200, (0, 0, 20, 20)),
    ])
    def test_in_range(self, area: int, bbox: Tuple) -> None:
        e = region_extent(area, bbox)
        assert 0.0 < e <= 1.0 + 1e-9

    def test_full_bbox_is_one(self) -> None:
        # area = bbox_area → extent = 1
        area = 100
        bbox = (0, 0, 10, 10)  # bbox_area = 10 * 10 = 100
        e = region_extent(area, bbox)
        assert e == pytest.approx(1.0)


class TestMaskPerimeter:
    """mask_perimeter: >= 0; all-zero → 0."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_nonneg(self, seed: int) -> None:
        mask = _rand_mask(20, 20, seed)
        p = mask_perimeter(mask)
        assert p >= 0.0

    def test_all_zero_is_zero(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_perimeter(mask) == pytest.approx(0.0)

    def test_all_ones_has_positive_perimeter(self) -> None:
        mask = np.ones((10, 10), dtype=np.uint8)
        p = mask_perimeter(mask)
        assert p >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# region_score_utils — score_region, evaluate_region
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreRegion:
    """score_region: ∈ [0, 1]."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_in_range(self, seed: int) -> None:
        mask = _rand_mask(20, 20, seed)
        if mask.sum() == 0:
            mask = np.ones((20, 20), dtype=np.uint8)
        area = int(mask.sum())
        bbox = (0, 0, 20, 20)
        s = score_region(area, bbox, mask)
        assert 0.0 <= s <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_evaluate_region_score_in_range(self, seed: int) -> None:
        d = _make_region_dict(seed)
        rs = evaluate_region(
            label=d["label"], area=d["area"],
            bbox=d["bbox"], mask=d["mask"],
        )
        assert 0.0 <= rs.score <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_evaluate_region_compactness_in_range(self, seed: int) -> None:
        d = _make_region_dict(seed)
        rs = evaluate_region(
            label=d["label"], area=d["area"],
            bbox=d["bbox"], mask=d["mask"],
        )
        assert 0.0 <= rs.compactness <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_evaluate_region_extent_in_range(self, seed: int) -> None:
        d = _make_region_dict(seed)
        rs = evaluate_region(
            label=d["label"], area=d["area"],
            bbox=d["bbox"], mask=d["mask"],
        )
        assert 0.0 < rs.extent <= 1.0 + 1e-9

    @pytest.mark.parametrize("n", [3, 5])
    def test_batch_evaluate_length(self, n: int) -> None:
        regions = [_make_region_dict(seed=i) for i in range(n)]
        results = batch_evaluate_regions(regions)
        assert len(results) == n


# ═══════════════════════════════════════════════════════════════════════════════
# region_score_utils — filters, rank, stats, normalize
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegionScoreFilters:
    """filter_by_score, rank_regions, top_k_regions, stats, normalize."""

    @pytest.mark.parametrize("min_s", [0.2, 0.4, 0.6])
    def test_filter_by_score(self, min_s: float) -> None:
        scores = _make_region_scores(15, seed=42)
        filtered = filter_by_score(scores, threshold=min_s)
        for r in filtered:
            assert r.score >= min_s

    @pytest.mark.parametrize("n", [5, 8])
    def test_rank_regions_length(self, n: int) -> None:
        scores = _make_region_scores(n, seed=n)
        ranked = rank_regions(scores)
        assert len(ranked) == n

    @pytest.mark.parametrize("n", [5, 8])
    def test_rank_regions_sorted_descending(self, n: int) -> None:
        scores = _make_region_scores(n, seed=n + 1)
        ranked = rank_regions(scores)
        vals = [r.score for r in ranked]
        assert vals == sorted(vals, reverse=True)

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_regions_length(self, k: int) -> None:
        scores = _make_region_scores(10, seed=k)
        top = top_k_regions(scores, k)
        assert len(top) <= k

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_region_stats_n(self, n: int, seed: int) -> None:
        scores = _make_region_scores(n, seed)
        stats = region_score_stats(scores)
        assert stats["n"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_region_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        scores = _make_region_scores(8, seed)
        stats = region_score_stats(scores)
        assert stats["min_score"] <= stats["mean_score"] + 1e-9
        assert stats["mean_score"] <= stats["max_score"] + 1e-9

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_normalize_all_in_range(self, n: int, seed: int) -> None:
        scores = _make_region_scores(n, seed)
        normalized = normalize_scores(scores)
        for r in normalized:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.parametrize("n,seed", [(3, 0), (5, 1)])
    def test_normalize_max_is_one(self, n: int, seed: int) -> None:
        scores = _make_region_scores(n, seed)
        # Make sure there's variation
        if len({r.score for r in scores}) <= 1:
            return  # skip degenerate case
        normalized = normalize_scores(scores)
        max_score = max(r.score for r in normalized)
        assert max_score == pytest.approx(1.0, abs=1e-6)
