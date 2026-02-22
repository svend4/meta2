"""Тесты для puzzle_reconstruction.scoring.pair_ranker."""
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


class TestRankConfig:
    def test_defaults(self):
        cfg = RankConfig()
        assert cfg.top_k == 0
        assert cfg.ascending is False
        assert cfg.deduplicate is True
        assert cfg.min_score == 0.0
        assert cfg.score_field == "score"

    def test_valid_custom(self):
        cfg = RankConfig(top_k=5, ascending=True, min_score=0.3, score_field="rank")
        assert cfg.top_k == 5
        assert cfg.ascending is True
        assert cfg.min_score == 0.3

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            RankConfig(top_k=-1)

    def test_invalid_min_score(self):
        with pytest.raises(ValueError):
            RankConfig(min_score=-0.1)

    def test_invalid_score_field(self):
        with pytest.raises(ValueError):
            RankConfig(score_field="invalid")


class TestRankedPair:
    def test_basic(self):
        rp = RankedPair(pair=(0, 1), score=0.9, rank=1)
        assert rp.fragment_a == 0
        assert rp.fragment_b == 1
        assert rp.rank == 1
        assert rp.n_metrics == 0

    def test_with_scores(self):
        rp = RankedPair(pair=(2, 3), score=0.5, rank=2,
                        scores={"boundary": 0.6, "color": 0.4})
        assert rp.n_metrics == 2

    def test_invalid_score(self):
        with pytest.raises(ValueError):
            RankedPair(pair=(0, 1), score=-0.1, rank=1)

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            RankedPair(pair=(0, 1), score=0.5, rank=0)

    def test_fragment_props(self):
        rp = RankedPair(pair=(5, 7), score=0.7, rank=3)
        assert rp.fragment_a == 5
        assert rp.fragment_b == 7


class TestRankResult:
    def _make(self):
        rp1 = RankedPair(pair=(0, 1), score=0.9, rank=1)
        rp2 = RankedPair(pair=(0, 2), score=0.7, rank=2)
        return RankResult(ranked=[rp1, rp2], n_pairs=5, n_ranked=2,
                          top_score=0.9, mean_score=0.8)

    def test_top_pair(self):
        r = self._make()
        assert r.top_pair == (0, 1)

    def test_compression_ratio(self):
        r = self._make()
        assert abs(r.compression_ratio - 2 / 5) < 1e-9

    def test_top_pair_empty(self):
        r = RankResult(ranked=[], n_pairs=3, n_ranked=0, top_score=0.0, mean_score=0.0)
        assert r.top_pair is None

    def test_compression_zero_n_pairs(self):
        r = RankResult(ranked=[], n_pairs=0, n_ranked=0, top_score=0.0, mean_score=0.0)
        assert r.compression_ratio == 0.0

    def test_invalid_n_pairs(self):
        with pytest.raises(ValueError):
            RankResult(ranked=[], n_pairs=-1, n_ranked=0, top_score=0.0, mean_score=0.0)


class TestComputePairScore:
    def test_equal_weights(self):
        score = compute_pair_score({"a": 0.8, "b": 0.6})
        assert abs(score - 0.7) < 1e-6

    def test_custom_weights(self):
        score = compute_pair_score({"a": 1.0, "b": 0.0},
                                   weights={"a": 2.0, "b": 1.0})
        # (1.0*2 + 0.0*1) / (2+1) = 0.667
        assert abs(score - 2.0 / 3.0) < 1e-5

    def test_single_metric(self):
        score = compute_pair_score({"x": 0.42})
        assert abs(score - 0.42) < 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_pair_score({})

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            compute_pair_score({"a": -0.1})


class TestRankPairs:
    def test_basic_ranking(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.9, 0.5, 0.7]
        result = rank_pairs(pairs, scores)
        assert result.n_pairs == 3
        assert result.ranked[0].score == 0.9
        assert result.ranked[0].rank == 1
        assert result.ranked[-1].rank == len(result.ranked)

    def test_top_k(self):
        pairs = [(i, i + 1) for i in range(5)]
        scores = [float(i) / 4 for i in range(5)]
        result = rank_pairs(pairs, scores, RankConfig(top_k=2))
        assert result.n_ranked == 2

    def test_min_score_filter(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.9, 0.1, 0.7]
        result = rank_pairs(pairs, scores, RankConfig(min_score=0.5))
        assert all(rp.score >= 0.5 for rp in result.ranked)

    def test_deduplicate(self):
        # (1,0) and (0,1) should be deduplicated
        pairs = [(0, 1), (1, 0)]
        scores = [0.8, 0.6]
        result = rank_pairs(pairs, scores, RankConfig(deduplicate=True))
        assert result.n_ranked == 1
        assert result.ranked[0].score == 0.8

    def test_ascending(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        scores = [0.9, 0.5, 0.7]
        result = rank_pairs(pairs, scores, RankConfig(ascending=True))
        assert result.ranked[0].score == 0.5

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            rank_pairs([(0, 1)], [0.5, 0.6])

    def test_empty_below_min_score(self):
        pairs = [(0, 1)]
        scores = [0.1]
        result = rank_pairs(pairs, scores, RankConfig(min_score=0.5))
        assert result.n_ranked == 0
        assert result.top_score == 0.0

    def test_metric_scores_list(self):
        pairs = [(0, 1)]
        scores = [0.8]
        ms = [{"color": 0.7, "boundary": 0.9}]
        result = rank_pairs(pairs, scores, metric_scores_list=ms)
        assert result.ranked[0].n_metrics == 2


class TestBuildRankMatrix:
    def test_basic(self):
        pairs = [(0, 1), (0, 2)]
        scores = [0.9, 0.7]
        result = rank_pairs(pairs, scores)
        matrix = build_rank_matrix(result, n_fragments=3)
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == matrix[1, 0] == 1
        assert matrix[0, 2] == matrix[2, 0] == 2

    def test_invalid_n_fragments(self):
        result = rank_pairs([(0, 1)], [0.5])
        with pytest.raises(ValueError):
            build_rank_matrix(result, n_fragments=0)

    def test_out_of_bounds_pairs_ignored(self):
        pairs = [(0, 1), (5, 6)]
        scores = [0.9, 0.8]
        result = rank_pairs(pairs, scores)
        matrix = build_rank_matrix(result, n_fragments=3)
        assert matrix[0, 1] > 0
        # (5,6) out of bounds — not set
        assert matrix[2, 2] == 0

    def test_symmetric(self):
        pairs = [(0, 2)]
        scores = [0.7]
        result = rank_pairs(pairs, scores)
        matrix = build_rank_matrix(result, n_fragments=4)
        assert matrix[0, 2] == matrix[2, 0]


class TestMergeRankResults:
    def test_basic_merge(self):
        r1 = rank_pairs([(0, 1), (0, 2)], [0.9, 0.5])
        r2 = rank_pairs([(1, 2), (0, 3)], [0.7, 0.8])
        merged = merge_rank_results([r1, r2])
        assert merged.n_ranked >= 1
        assert merged.ranked[0].score == max(rp.score for r in [r1, r2]
                                             for rp in r.ranked)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_rank_results([])

    def test_single_result(self):
        r = rank_pairs([(0, 1)], [0.6])
        merged = merge_rank_results([r])
        assert merged.n_ranked == 1
        assert abs(merged.top_score - 0.6) < 1e-9

    def test_merge_deduplicates(self):
        # Both results contain pair (0,1)
        r1 = rank_pairs([(0, 1)], [0.9])
        r2 = rank_pairs([(0, 1)], [0.6])
        merged = merge_rank_results([r1, r2])
        # After dedup, highest score wins
        assert merged.n_ranked == 1
        assert abs(merged.top_score - 0.9) < 1e-9
