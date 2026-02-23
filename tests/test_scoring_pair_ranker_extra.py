"""Extra tests for puzzle_reconstruction.scoring.pair_ranker."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.scoring.pair_ranker import (
    RankConfig,
    RankedPair,
    RankResult,
    compute_pair_score,
    rank_pairs,
    build_rank_matrix,
    merge_rank_results,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _rp(pair=(0, 1), score=0.8, rank=1):
    return RankedPair(pair=pair, score=score, rank=rank)


def _rr(n=3):
    ranked = [_rp(pair=(i, i + 1), score=round(1.0 - i * 0.1, 1), rank=i + 1)
              for i in range(n)]
    return RankResult(
        ranked=ranked,
        n_pairs=n,
        n_ranked=n,
        top_score=ranked[0].score,
        mean_score=float(np.mean([r.score for r in ranked])),
    )


# ─── TestRankConfigExtra ─────────────────────────────────────────────────────

class TestRankConfigExtra:
    def test_top_k_stored(self):
        cfg = RankConfig(top_k=5)
        assert cfg.top_k == 5

    def test_min_score_stored(self):
        cfg = RankConfig(min_score=0.4)
        assert cfg.min_score == pytest.approx(0.4)

    def test_deduplicate_false(self):
        cfg = RankConfig(deduplicate=False)
        assert cfg.deduplicate is False

    def test_ascending_default_false(self):
        assert RankConfig().ascending is False

    def test_score_field_score(self):
        cfg = RankConfig(score_field="score")
        assert cfg.score_field == "score"

    def test_score_field_rank(self):
        cfg = RankConfig(score_field="rank")
        assert cfg.score_field == "rank"

    def test_large_top_k(self):
        cfg = RankConfig(top_k=1000)
        assert cfg.top_k == 1000


# ─── TestRankedPairExtra ─────────────────────────────────────────────────────

class TestRankedPairExtra:
    def test_pair_stored(self):
        rp = _rp(pair=(2, 5))
        assert rp.pair == (2, 5)

    def test_score_stored(self):
        rp = _rp(score=0.65)
        assert rp.score == pytest.approx(0.65)

    def test_rank_stored(self):
        rp = _rp(rank=3)
        assert rp.rank == 3

    def test_fragment_a(self):
        rp = _rp(pair=(4, 7))
        assert rp.fragment_a == 4

    def test_fragment_b(self):
        rp = _rp(pair=(4, 7))
        assert rp.fragment_b == 7

    def test_n_metrics_with_scores(self):
        rp = RankedPair(pair=(0, 1), score=0.8, rank=1,
                        scores={"color": 0.9, "texture": 0.7, "geometry": 0.5})
        assert rp.n_metrics == 3

    def test_score_one_valid(self):
        rp = RankedPair(pair=(0, 1), score=1.0, rank=1)
        assert rp.score == 1.0

    def test_large_rank(self):
        rp = RankedPair(pair=(0, 1), score=0.5, rank=100)
        assert rp.rank == 100


# ─── TestRankResultExtra ─────────────────────────────────────────────────────

class TestRankResultExtra:
    def test_top_pair_first_ranked(self):
        r = _rr(3)
        assert r.top_pair == r.ranked[0].pair

    def test_compression_ratio_full(self):
        r = _rr(3)
        assert r.compression_ratio == pytest.approx(1.0)

    def test_compression_ratio_half(self):
        r = RankResult(ranked=[_rp()], n_pairs=2, n_ranked=1,
                       top_score=0.8, mean_score=0.8)
        assert r.compression_ratio == pytest.approx(0.5)

    def test_top_score_stored(self):
        r = _rr(4)
        assert r.top_score == r.ranked[0].score

    def test_mean_score_stored(self):
        r = _rr(3)
        assert r.mean_score > 0.0

    def test_n_pairs_stored(self):
        r = _rr(5)
        assert r.n_pairs == 5

    def test_n_ranked_stored(self):
        r = _rr(4)
        assert r.n_ranked == 4

    def test_empty_ranked(self):
        r = RankResult(ranked=[], n_pairs=0, n_ranked=0,
                       top_score=0.0, mean_score=0.0)
        assert r.top_pair is None


# ─── TestComputePairScoreExtra ───────────────────────────────────────────────

class TestComputePairScoreExtra:
    def test_single_metric_passthrough(self):
        s = compute_pair_score({"color": 0.75})
        assert s == pytest.approx(0.75, abs=1e-5)

    def test_all_zero(self):
        s = compute_pair_score({"a": 0.0, "b": 0.0})
        assert s == pytest.approx(0.0)

    def test_all_one(self):
        s = compute_pair_score({"a": 1.0, "b": 1.0})
        assert s == pytest.approx(1.0, abs=1e-5)

    def test_custom_weights_asymmetric(self):
        s = compute_pair_score({"a": 1.0, "b": 0.0},
                                weights={"a": 3.0, "b": 1.0})
        assert s == pytest.approx(0.75, abs=1e-5)

    def test_result_nonneg(self):
        s = compute_pair_score({"x": 0.3, "y": 0.7})
        assert s >= 0.0

    def test_three_metrics_equal_avg(self):
        s = compute_pair_score({"a": 0.3, "b": 0.6, "c": 0.9})
        assert s == pytest.approx(0.6, abs=1e-5)


# ─── TestRankPairsExtra ──────────────────────────────────────────────────────

class TestRankPairsExtra:
    def test_two_pairs_correct_ranks(self):
        r = rank_pairs([(0, 1), (1, 2)], [0.9, 0.5])
        assert r.ranked[0].rank == 1
        assert r.ranked[1].rank == 2

    def test_top_score_stored(self):
        r = rank_pairs([(0, 1), (1, 2)], [0.7, 0.4])
        assert r.top_score == pytest.approx(0.7)

    def test_min_score_filter_all(self):
        cfg = RankConfig(min_score=0.5)
        r = rank_pairs([(0, 1), (1, 2), (2, 3)], [0.8, 0.6, 0.3], cfg)
        assert all(rp.score >= 0.5 for rp in r.ranked)

    def test_top_k_two(self):
        cfg = RankConfig(top_k=2)
        r = rank_pairs([(i, i+1) for i in range(5)],
                       [0.9, 0.8, 0.7, 0.6, 0.5], cfg)
        assert r.n_ranked == 2

    def test_ascending_order(self):
        cfg = RankConfig(ascending=True)
        r = rank_pairs([(0, 1), (1, 2), (2, 3)], [0.4, 0.9, 0.6], cfg)
        scores = [rp.score for rp in r.ranked]
        assert scores == sorted(scores)

    def test_deduplicate_removes_reverse(self):
        cfg = RankConfig(deduplicate=True)
        r = rank_pairs([(0, 1), (1, 0)], [0.8, 0.6], cfg)
        assert r.n_ranked == 1

    def test_n_pairs_counts_input(self):
        r = rank_pairs([(0, 1), (1, 2)], [0.8, 0.6])
        assert r.n_pairs == 2

    def test_all_filtered_gives_zero_ranked(self):
        cfg = RankConfig(min_score=1.0)
        r = rank_pairs([(0, 1)], [0.5], cfg)
        assert r.n_ranked == 0


# ─── TestBuildRankMatrixExtra ────────────────────────────────────────────────

class TestBuildRankMatrixExtra:
    def test_shape_correct(self):
        mat = build_rank_matrix(_rr(3), n_fragments=6)
        assert mat.shape == (6, 6)

    def test_symmetric(self):
        mat = build_rank_matrix(_rr(3), n_fragments=5)
        np.testing.assert_array_equal(mat, mat.T)

    def test_dtype_int(self):
        mat = build_rank_matrix(_rr(2), n_fragments=4)
        assert mat.dtype == int

    def test_ranked_pair_set(self):
        rr = RankResult(
            ranked=[_rp(pair=(0, 2), rank=1)],
            n_pairs=1, n_ranked=1, top_score=0.8, mean_score=0.8,
        )
        mat = build_rank_matrix(rr, n_fragments=4)
        assert mat[0, 2] == 1
        assert mat[2, 0] == 1

    def test_empty_result_all_zeros(self):
        rr = RankResult(ranked=[], n_pairs=0, n_ranked=0,
                        top_score=0.0, mean_score=0.0)
        mat = build_rank_matrix(rr, n_fragments=3)
        assert mat.sum() == 0

    def test_n_fragments_one(self):
        rr = RankResult(ranked=[], n_pairs=0, n_ranked=0,
                        top_score=0.0, mean_score=0.0)
        mat = build_rank_matrix(rr, n_fragments=1)
        assert mat.shape == (1, 1)


# ─── TestMergeRankResultsExtra ───────────────────────────────────────────────

class TestMergeRankResultsExtra:
    def test_three_results_merged(self):
        results = [_rr(2), _rr(3), _rr(1)]
        merged = merge_rank_results(results)
        assert isinstance(merged, RankResult)

    def test_merged_n_ranked_positive(self):
        merged = merge_rank_results([_rr(3), _rr(3)])
        assert merged.n_ranked > 0

    def test_merged_top_score_max(self):
        r1 = RankResult(ranked=[_rp(score=0.9, rank=1)],
                        n_pairs=1, n_ranked=1, top_score=0.9, mean_score=0.9)
        r2 = RankResult(ranked=[_rp(pair=(1, 2), score=0.6, rank=1)],
                        n_pairs=1, n_ranked=1, top_score=0.6, mean_score=0.6)
        merged = merge_rank_results([r1, r2])
        assert merged.top_score >= 0.6

    def test_single_merged_is_same(self):
        r = _rr(4)
        merged = merge_rank_results([r])
        assert merged.n_ranked == r.n_ranked

    def test_merged_ranks_start_at_one(self):
        merged = merge_rank_results([_rr(2)])
        assert merged.ranked[0].rank == 1
