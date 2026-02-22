"""Additional tests for puzzle_reconstruction/scoring/pair_ranker.py."""
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


# ─── TestRankConfigExtra ──────────────────────────────────────────────────────

class TestRankConfigExtra:
    def test_top_k_zero_unlimited(self):
        cfg = RankConfig(top_k=0)
        assert cfg.top_k == 0

    def test_min_score_zero_allowed(self):
        cfg = RankConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_min_score_one_allowed(self):
        cfg = RankConfig(min_score=1.0)
        assert cfg.min_score == 1.0

    def test_score_field_rank_accepted(self):
        cfg = RankConfig(score_field="rank")
        assert cfg.score_field == "rank"

    def test_score_field_invalid_combined_raises(self):
        with pytest.raises(ValueError):
            RankConfig(score_field="combined")

    def test_deduplicate_false(self):
        cfg = RankConfig(deduplicate=False)
        assert cfg.deduplicate is False

    def test_top_k_large(self):
        cfg = RankConfig(top_k=1000)
        assert cfg.top_k == 1000

    def test_ascending_default_false(self):
        assert RankConfig().ascending is False

    def test_score_field_default(self):
        assert RankConfig().score_field == "score"


# ─── TestRankedPairExtra ──────────────────────────────────────────────────────

class TestRankedPairExtra:
    def test_score_zero_allowed(self):
        rp = RankedPair(pair=(0, 1), score=0.0, rank=1)
        assert rp.score == 0.0

    def test_score_one_allowed(self):
        rp = RankedPair(pair=(0, 1), score=1.0, rank=1)
        assert rp.score == 1.0

    def test_rank_one_allowed(self):
        rp = RankedPair(pair=(0, 1), score=0.5, rank=1)
        assert rp.rank == 1

    def test_rank_large_allowed(self):
        rp = RankedPair(pair=(0, 1), score=0.5, rank=999)
        assert rp.rank == 999

    def test_n_metrics_with_three_scores(self):
        rp = RankedPair(
            pair=(0, 1), score=0.5, rank=1,
            scores={"a": 0.3, "b": 0.5, "c": 0.7},
        )
        assert rp.n_metrics == 3

    def test_fragment_a_b_zero_based(self):
        rp = RankedPair(pair=(0, 0), score=0.9, rank=1)
        assert rp.fragment_a == 0
        assert rp.fragment_b == 0

    def test_scores_empty_n_metrics_zero(self):
        rp = RankedPair(pair=(1, 2), score=0.6, rank=2, scores={})
        assert rp.n_metrics == 0


# ─── TestRankResultExtra ──────────────────────────────────────────────────────

class TestRankResultExtra:
    def _single(self):
        rp = RankedPair(pair=(3, 5), score=0.75, rank=1)
        return RankResult(ranked=[rp], n_pairs=1, n_ranked=1,
                          top_score=0.75, mean_score=0.75)

    def test_single_item_top_pair(self):
        r = self._single()
        assert r.top_pair == (3, 5)

    def test_compression_ratio_one(self):
        r = self._single()
        assert abs(r.compression_ratio - 1.0) < 1e-9

    def test_n_ranked_field(self):
        r = self._single()
        assert r.n_ranked == 1

    def test_top_score_field(self):
        r = self._single()
        assert abs(r.top_score - 0.75) < 1e-9

    def test_mean_score_field(self):
        r = self._single()
        assert abs(r.mean_score - 0.75) < 1e-9

    def test_n_pairs_zero_not_raises(self):
        r = RankResult(ranked=[], n_pairs=0, n_ranked=0, top_score=0.0, mean_score=0.0)
        assert r.n_pairs == 0

    def test_compression_ratio_less_than_one(self):
        rp = RankedPair(pair=(0, 1), score=0.9, rank=1)
        r = RankResult(ranked=[rp], n_pairs=10, n_ranked=1,
                       top_score=0.9, mean_score=0.9)
        assert r.compression_ratio < 1.0


# ─── TestComputePairScoreExtra ────────────────────────────────────────────────

class TestComputePairScoreExtra:
    def test_three_equal_weights(self):
        score = compute_pair_score({"a": 0.6, "b": 0.9, "c": 0.3})
        assert abs(score - 0.6) < 1e-5

    def test_zero_score_allowed(self):
        score = compute_pair_score({"a": 0.0})
        assert abs(score - 0.0) < 1e-9

    def test_one_score_allowed(self):
        score = compute_pair_score({"a": 1.0})
        assert abs(score - 1.0) < 1e-9

    def test_all_zeros_gives_zero(self):
        score = compute_pair_score({"x": 0.0, "y": 0.0})
        assert abs(score - 0.0) < 1e-9

    def test_weights_all_zero_returns_zero(self):
        score = compute_pair_score({"a": 0.5, "b": 0.5}, weights={"a": 0.0, "b": 0.0})
        assert score == pytest.approx(0.0)

    def test_custom_weights_unequal(self):
        score = compute_pair_score({"a": 1.0, "b": 0.0}, weights={"a": 3.0, "b": 1.0})
        assert abs(score - 0.75) < 1e-5

    def test_returns_float(self):
        score = compute_pair_score({"a": 0.5})
        assert isinstance(score, float)


# ─── TestRankPairsExtra ───────────────────────────────────────────────────────

class TestRankPairsExtra:
    def test_top_k_zero_returns_all(self):
        pairs = [(i, i + 1) for i in range(6)]
        scores = [float(i) / 5 for i in range(6)]
        result = rank_pairs(pairs, scores, RankConfig(top_k=0))
        assert result.n_ranked == 6

    def test_deduplicate_false_keeps_both(self):
        pairs = [(0, 1), (1, 0)]
        scores = [0.8, 0.6]
        result = rank_pairs(pairs, scores, RankConfig(deduplicate=False))
        assert result.n_ranked == 2

    def test_top_k_larger_than_available(self):
        pairs = [(0, 1), (1, 2)]
        scores = [0.9, 0.7]
        result = rank_pairs(pairs, scores, RankConfig(top_k=100))
        assert result.n_ranked == 2

    def test_min_score_1_filters_all_below(self):
        pairs = [(0, 1), (1, 2)]
        scores = [0.8, 0.6]
        result = rank_pairs(pairs, scores, RankConfig(min_score=1.0))
        assert result.n_ranked == 0

    def test_ranks_contiguous(self):
        pairs = [(0, 1), (0, 2), (0, 3)]
        scores = [0.9, 0.5, 0.7]
        result = rank_pairs(pairs, scores)
        ranks = [rp.rank for rp in result.ranked]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_top_pair_is_highest_score(self):
        pairs = [(0, 1), (2, 3), (4, 5)]
        scores = [0.4, 0.9, 0.6]
        result = rank_pairs(pairs, scores)
        assert result.top_pair == (2, 3)

    def test_mean_score_correct(self):
        pairs = [(0, 1), (0, 2)]
        scores = [0.8, 0.4]
        result = rank_pairs(pairs, scores)
        assert abs(result.mean_score - 0.6) < 1e-6

    def test_empty_pairs_gives_empty_result(self):
        result = rank_pairs([], [])
        assert result.n_ranked == 0
        assert result.top_pair is None

    def test_metric_scores_n_metrics(self):
        pairs = [(0, 1), (0, 2)]
        scores = [0.8, 0.6]
        ms = [{"a": 0.7}, {"a": 0.5}]
        result = rank_pairs(pairs, scores, metric_scores_list=ms)
        assert all(rp.n_metrics == 1 for rp in result.ranked)


# ─── TestBuildRankMatrixExtra ─────────────────────────────────────────────────

class TestBuildRankMatrixExtra:
    def test_diagonal_is_zero(self):
        pairs = [(0, 1), (1, 2)]
        result = rank_pairs(pairs, [0.9, 0.7])
        matrix = build_rank_matrix(result, n_fragments=3)
        import numpy as np
        assert all(matrix[i, i] == 0 for i in range(3))

    def test_n_fragments_1(self):
        result = rank_pairs([], [])
        matrix = build_rank_matrix(result, n_fragments=1)
        import numpy as np
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0

    def test_symmetric_matrix(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        result = rank_pairs(pairs, [0.9, 0.7, 0.5])
        matrix = build_rank_matrix(result, n_fragments=3)
        import numpy as np
        np.testing.assert_array_equal(matrix, matrix.T)

    def test_all_pairs_in_matrix(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        result = rank_pairs(pairs, [0.9, 0.7, 0.5])
        matrix = build_rank_matrix(result, n_fragments=3)
        assert matrix[0, 1] > 0
        assert matrix[0, 2] > 0
        assert matrix[1, 2] > 0

    def test_n_fragments_large(self):
        pairs = [(0, 1)]
        result = rank_pairs(pairs, [0.8])
        matrix = build_rank_matrix(result, n_fragments=10)
        assert matrix.shape == (10, 10)


# ─── TestMergeRankResultsExtra ────────────────────────────────────────────────

class TestMergeRankResultsExtra:
    def test_three_results_merge(self):
        r1 = rank_pairs([(0, 1)], [0.9])
        r2 = rank_pairs([(1, 2)], [0.7])
        r3 = rank_pairs([(0, 2)], [0.5])
        merged = merge_rank_results([r1, r2, r3])
        assert merged.n_ranked >= 1

    def test_merge_preserves_top_score(self):
        r1 = rank_pairs([(0, 1)], [0.9])
        r2 = rank_pairs([(1, 2)], [0.5])
        merged = merge_rank_results([r1, r2])
        assert abs(merged.top_score - 0.9) < 1e-9

    def test_merge_all_distinct_pairs_keeps_all(self):
        r1 = rank_pairs([(0, 1), (0, 2)], [0.9, 0.5])
        r2 = rank_pairs([(1, 2), (0, 3)], [0.7, 0.8])
        merged = merge_rank_results([r1, r2])
        assert merged.n_ranked == 4

    def test_merge_duplicate_keeps_best(self):
        r1 = rank_pairs([(0, 1)], [0.5])
        r2 = rank_pairs([(0, 1)], [0.9])
        r3 = rank_pairs([(0, 1)], [0.3])
        merged = merge_rank_results([r1, r2, r3])
        assert abs(merged.top_score - 0.9) < 1e-9
        assert merged.n_ranked == 1
