"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.overlap_score_utils
  - puzzle_reconstruction.utils.rank_result_utils
  - puzzle_reconstruction.utils.rotation_score_utils

overlap_score_utils:
    make_overlap_entry:          iou ∈ [0, 1]; penalty ∈ [0, 1]; overlap_area >= 0
    summarise_overlaps:          n_overlaps <= len; total_area >= 0
    filter_significant_overlaps: all iou >= threshold
    top_k_overlaps:              len <= k; sorted descending by iou
    overlap_stats:               n = len; min <= mean <= max iou
    penalty_score:               ∈ [0, 1]
    batch_make_overlap_entries:  len = len(pairs)
    group_by_fragment:           all keys in idx1 values

rank_result_utils:
    make_rank_result_entry:      pair_key canonical; is_top_match = (rank==1)
    entries_from_ranked_pairs:   len = len(pairs)
    summarise_rank_results:      n_entries = len; min <= mean <= max
    filter_high_rank_entries:    all .rank <= threshold
    rerank_entries:              ranks 1..n; sorted descending
    rank_result_stats:           count = n; max >= mean >= min

rotation_score_utils:
    make_entry:                  confidence ∈ [0, 1]
    filter_by_confidence:        all .confidence >= min_confidence
    filter_by_angle_range:       all .angle_deg ∈ [min, max]
    rank_by_confidence:          sorted descending
    aggregate_angles:            result is float
    rotation_score_stats:        n = len; confidence in [0, 1]
    angle_agreement:             ∈ [0, 1]; single entry = 1.0
    batch_make_entries:          len = len(inputs)
    top_k_entries:               len <= k; sorted descending
    group_by_method:             union of all groups = all entries
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.overlap_score_utils import (
    OverlapScoreConfig,
    OverlapScoreEntry,
    OverlapSummary,
    make_overlap_entry,
    summarise_overlaps,
    filter_significant_overlaps,
    top_k_overlaps,
    overlap_stats,
    penalty_score,
    batch_make_overlap_entries,
    group_by_fragment,
)
from puzzle_reconstruction.utils.rank_result_utils import (
    RankResultConfig,
    RankResultEntry,
    RankResultSummary,
    make_rank_result_entry,
    entries_from_ranked_pairs,
    summarise_rank_results,
    filter_high_rank_entries,
    rerank_entries,
    rank_result_stats,
)
from puzzle_reconstruction.utils.rotation_score_utils import (
    RotationScoreConfig,
    RotationScoreEntry,
    make_entry as rotation_make_entry,
    filter_by_confidence,
    filter_by_angle_range,
    rank_by_confidence,
    aggregate_angles,
    rotation_score_stats,
    angle_agreement,
    batch_make_entries as rotation_batch_make,
    top_k_entries as rotation_top_k,
    group_by_method,
)

RNG = np.random.default_rng(3131)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_overlap_entries(n: int, seed: int = 0) -> List[OverlapScoreEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_overlap_entry(
            idx1=i, idx2=i + 1,
            iou=float(rng.uniform(0, 1)),
            overlap_area=float(rng.uniform(0, 500)),
        )
        for i in range(n)
    ]


def _make_rank_entries(n: int, seed: int = 0) -> List[RankResultEntry]:
    rng = np.random.default_rng(seed)
    return [
        make_rank_result_entry(
            frag_i=i, frag_j=i + 1,
            score=float(rng.uniform(0, 1)),
            rank=i + 1,
        )
        for i in range(n)
    ]


def _make_rotation_entries(n: int, seed: int = 0,
                            methods: Optional[List[str]] = None) -> List[RotationScoreEntry]:
    rng = np.random.default_rng(seed)
    methods = methods or ["pca"] * n
    return [
        rotation_make_entry(
            image_id=i,
            angle_deg=float(rng.uniform(-180, 180)),
            confidence=float(rng.uniform(0, 1)),
            method=methods[i % len(methods)],
        )
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# overlap_score_utils — make_overlap_entry, fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlapEntry:
    """make_overlap_entry: iou, penalty, overlap_area invariants."""

    @pytest.mark.parametrize("iou,area", [
        (0.3, 100.0), (0.0, 0.0), (1.0, 500.0), (0.7, 200.0),
    ])
    def test_fields_in_range(self, iou: float, area: float) -> None:
        e = make_overlap_entry(0, 1, iou=iou, overlap_area=area)
        assert 0.0 <= e.iou <= 1.0
        assert e.overlap_area >= 0.0
        assert 0.0 <= e.penalty <= 1.0

    def test_negative_iou_raises(self) -> None:
        with pytest.raises(ValueError):
            OverlapScoreEntry(0, 1, iou=-0.1, overlap_area=100.0)

    def test_negative_area_raises(self) -> None:
        with pytest.raises(ValueError):
            OverlapScoreEntry(0, 1, iou=0.5, overlap_area=-1.0)

    @pytest.mark.parametrize("n", [3, 5])
    def test_batch_make_length(self, n: int) -> None:
        pairs = [(i, i + 1) for i in range(n)]
        ious = [0.1 * i for i in range(n)]
        areas = [float(100 * i) for i in range(n)]
        entries = batch_make_overlap_entries(pairs, ious, areas)
        assert len(entries) == n


# ═══════════════════════════════════════════════════════════════════════════════
# overlap_score_utils — summarise_overlaps, filters, stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlapSummarise:
    """summarise_overlaps and filter invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_n_overlaps_leq_len(self, n: int, seed: int) -> None:
        entries = _make_overlap_entries(n, seed)
        s = summarise_overlaps(entries)
        assert 0 <= s.n_overlaps <= n

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_total_area_nonneg(self, n: int, seed: int) -> None:
        entries = _make_overlap_entries(n, seed)
        s = summarise_overlaps(entries)
        assert s.total_area >= 0.0

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_filter_significant(self, n: int, seed: int) -> None:
        entries = _make_overlap_entries(n, seed)
        significant = filter_significant_overlaps(entries, iou_threshold=0.1)
        for e in significant:
            assert e.iou >= 0.1

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_overlap_entries(10, seed=k)
        top = top_k_overlaps(entries, k)
        assert len(top) <= k

    @pytest.mark.parametrize("k", [3, 5])
    def test_top_k_sorted_descending(self, k: int) -> None:
        entries = _make_overlap_entries(10, seed=k + 10)
        top = top_k_overlaps(entries, k)
        ious = [e.iou for e in top]
        assert ious == sorted(ious, reverse=True)

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_overlap_stats_n(self, n: int, seed: int) -> None:
        entries = _make_overlap_entries(n, seed)
        stats = overlap_stats(entries)
        assert stats["n"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_overlap_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_overlap_entries(8, seed)
        stats = overlap_stats(entries)
        assert stats["min_iou"] <= stats["mean_iou"] + 1e-9
        assert stats["mean_iou"] <= stats["max_iou"] + 1e-9

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_penalty_score_in_range(self, seed: int) -> None:
        entries = _make_overlap_entries(10, seed)
        p = penalty_score(entries)
        assert 0.0 <= p <= 1.0

    def test_penalty_score_empty_is_zero(self) -> None:
        assert penalty_score([]) == 0.0

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_group_by_fragment_keys_in_idx1(self, n: int, seed: int) -> None:
        entries = _make_overlap_entries(n, seed)
        groups = group_by_fragment(entries)
        idx1_values = {e.idx1 for e in entries}
        for k in groups:
            assert k in idx1_values


# ═══════════════════════════════════════════════════════════════════════════════
# rank_result_utils — make_rank_result_entry, is_top_match, pair_key
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankResultEntry:
    """make_rank_result_entry: pair_key and is_top_match invariants."""

    @pytest.mark.parametrize("i,j", [(0, 1), (5, 2), (3, 7)])
    def test_pair_key_canonical(self, i: int, j: int) -> None:
        e = make_rank_result_entry(i, j, score=0.5, rank=1)
        key = e.pair_key
        assert key[0] <= key[1]

    @pytest.mark.parametrize("rank,expected_top", [
        (1, True), (2, False), (5, False),
    ])
    def test_is_top_match(self, rank: int, expected_top: bool) -> None:
        e = make_rank_result_entry(0, 1, score=0.5, rank=rank)
        assert e.is_top_match is expected_top

    @pytest.mark.parametrize("n", [5, 8])
    def test_entries_from_ranked_pairs_length(self, n: int) -> None:
        pairs = [(i, i + 1) for i in range(n)]
        scores = [float(0.1 * i) for i in range(n)]
        entries = entries_from_ranked_pairs(pairs, scores)
        assert len(entries) == n


# ═══════════════════════════════════════════════════════════════════════════════
# rank_result_utils — summarise, filters, rerank, stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankResultSummarise:
    """summarise_rank_results and filter invariants."""

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_n_entries(self, n: int, seed: int) -> None:
        entries = _make_rank_entries(n, seed)
        s = summarise_rank_results(entries)
        assert s.n_entries == n

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_rank_entries(8, seed)
        s = summarise_rank_results(entries)
        assert s.min_score <= s.mean_score + 1e-9
        assert s.mean_score <= s.max_score + 1e-9

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_filter_high_rank(self, threshold: float) -> None:
        entries = _make_rank_entries(10, seed=int(threshold * 10))
        filtered = filter_high_rank_entries(entries, threshold=threshold)
        for e in filtered:
            assert e.score >= threshold

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_rerank_entries_length(self, n: int, seed: int) -> None:
        entries = _make_rank_entries(n, seed)
        reranked = rerank_entries(entries)
        assert len(reranked) == n

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_rerank_ranks_1_to_n(self, n: int, seed: int) -> None:
        entries = _make_rank_entries(n, seed)
        reranked = rerank_entries(entries)
        ranks = sorted(e.rank for e in reranked)
        assert ranks == list(range(1, n + 1))

    @pytest.mark.parametrize("n,seed", [(5, 0), (8, 1)])
    def test_rank_result_stats_count(self, n: int, seed: int) -> None:
        entries = _make_rank_entries(n, seed)
        stats = rank_result_stats(entries)
        assert stats["count"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_rank_stats_min_leq_mean_leq_max(self, seed: int) -> None:
        entries = _make_rank_entries(8, seed)
        stats = rank_result_stats(entries)
        assert stats["min_score"] <= stats["mean_score"] + 1e-9
        assert stats["mean_score"] <= stats["max_score"] + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# rotation_score_utils — make_entry, filter, rank, aggregate, stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotationScoreEntry:
    """make_entry: confidence ∈ [0, 1]; negative image_id raises."""

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_confidence_in_range(self, conf: float) -> None:
        e = rotation_make_entry(0, 45.0, conf, "pca")
        assert 0.0 <= e.confidence <= 1.0

    def test_negative_image_id_raises(self) -> None:
        with pytest.raises(ValueError):
            rotation_make_entry(-1, 45.0, 0.5, "pca")

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            rotation_make_entry(0, 45.0, 1.5, "pca")


class TestRotationFiltersAndRank:
    """filter_by_confidence, filter_by_angle_range, rank_by_confidence."""

    @pytest.mark.parametrize("min_conf", [0.3, 0.5, 0.7])
    def test_filter_by_confidence(self, min_conf: float) -> None:
        entries = _make_rotation_entries(15, seed=42)
        filtered = filter_by_confidence(entries, min_confidence=min_conf)
        for e in filtered:
            assert e.confidence >= min_conf

    @pytest.mark.parametrize("min_a,max_a", [(-45, 45), (0, 90), (-90, 0)])
    def test_filter_by_angle_range(self, min_a: float, max_a: float) -> None:
        entries = _make_rotation_entries(20, seed=99)
        filtered = filter_by_angle_range(entries, min_angle=min_a, max_angle=max_a)
        for e in filtered:
            assert min_a <= e.angle_deg <= max_a

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_rank_by_confidence_sorted(self, seed: int) -> None:
        entries = _make_rotation_entries(10, seed)
        ranked = rank_by_confidence(entries)
        confs = [e.confidence for e in ranked]
        assert confs == sorted(confs, reverse=True)

    @pytest.mark.parametrize("k", [2, 4])
    def test_top_k_length(self, k: int) -> None:
        entries = _make_rotation_entries(10, seed=k)
        top = rotation_top_k(entries, k)
        assert len(top) <= k


class TestRotationAggregateAndStats:
    """aggregate_angles, rotation_score_stats, angle_agreement."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_aggregate_angles_is_float(self, seed: int) -> None:
        entries = _make_rotation_entries(8, seed)
        result = aggregate_angles(entries)
        assert isinstance(result, float)

    def test_aggregate_angles_empty_is_zero(self) -> None:
        assert aggregate_angles([]) == 0.0

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1)])
    def test_rotation_stats_n(self, n: int, seed: int) -> None:
        entries = _make_rotation_entries(n, seed)
        stats = rotation_score_stats(entries)
        assert stats["n"] == n

    @pytest.mark.parametrize("seed", [0, 1])
    def test_rotation_stats_confidence_in_range(self, seed: int) -> None:
        entries = _make_rotation_entries(10, seed)
        stats = rotation_score_stats(entries)
        assert 0.0 <= stats["min_confidence"] <= stats["mean_confidence"]
        assert stats["mean_confidence"] <= stats["max_confidence"] <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_angle_agreement_in_range(self, seed: int) -> None:
        entries = _make_rotation_entries(8, seed)
        score = angle_agreement(entries)
        assert 0.0 <= score <= 1.0

    def test_single_entry_agreement_is_one(self) -> None:
        entries = _make_rotation_entries(1, seed=0)
        assert angle_agreement(entries) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [4, 6])
    def test_batch_make_entries_length(self, n: int) -> None:
        ids = list(range(n))
        angles = [float(10 * i) for i in range(n)]
        confs = [0.5] * n
        methods = ["pca"] * n
        entries = rotation_batch_make(ids, angles, confs, methods)
        assert len(entries) == n

    @pytest.mark.parametrize("n,seed", [(6, 0), (10, 1)])
    def test_group_by_method_union_all_entries(self, n: int, seed: int) -> None:
        methods = ["pca", "moments", "gradient"]
        entries = _make_rotation_entries(n, seed, methods=methods)
        groups = group_by_method(entries)
        all_entries = [e for grp in groups.values() for e in grp]
        assert len(all_entries) == n
