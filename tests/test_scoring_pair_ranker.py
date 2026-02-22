"""Тесты для puzzle_reconstruction.scoring.pair_ranker."""
import pytest
import numpy as np
from puzzle_reconstruction.scoring.pair_ranker import (
    RankConfig,
    RankedPair,
    RankResult,
    compute_pair_score,
    rank_pairs,
    build_rank_matrix,
    merge_rank_results,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_ranked_pair(pair=(0, 1), score=0.8, rank=1) -> RankedPair:
    return RankedPair(pair=pair, score=score, rank=rank)


def _make_rank_result(n=3) -> RankResult:
    ranked = [
        RankedPair(pair=(i, i + 1), score=1.0 - i * 0.1, rank=i + 1)
        for i in range(n)
    ]
    return RankResult(
        ranked=ranked,
        n_pairs=n,
        n_ranked=n,
        top_score=ranked[0].score,
        mean_score=float(np.mean([r.score for r in ranked])),
    )


# ─── TestRankConfig ───────────────────────────────────────────────────────────

class TestRankConfig:
    def test_defaults(self):
        cfg = RankConfig()
        assert cfg.top_k == 0
        assert cfg.ascending is False
        assert cfg.deduplicate is True
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.score_field == "score"

    def test_top_k_zero_ok(self):
        cfg = RankConfig(top_k=0)
        assert cfg.top_k == 0

    def test_top_k_positive_ok(self):
        cfg = RankConfig(top_k=10)
        assert cfg.top_k == 10

    def test_top_k_neg_raises(self):
        with pytest.raises(ValueError):
            RankConfig(top_k=-1)

    def test_min_score_zero_ok(self):
        cfg = RankConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            RankConfig(min_score=-0.1)

    def test_score_field_rank_ok(self):
        cfg = RankConfig(score_field="rank")
        assert cfg.score_field == "rank"

    def test_score_field_score_ok(self):
        cfg = RankConfig(score_field="score")
        assert cfg.score_field == "score"

    def test_score_field_invalid_raises(self):
        with pytest.raises(ValueError):
            RankConfig(score_field="value")

    def test_ascending_true_ok(self):
        cfg = RankConfig(ascending=True)
        assert cfg.ascending is True


# ─── TestRankedPair ───────────────────────────────────────────────────────────

class TestRankedPair:
    def test_basic(self):
        rp = _make_ranked_pair()
        assert rp.pair == (0, 1)

    def test_fragment_a(self):
        rp = _make_ranked_pair(pair=(3, 7))
        assert rp.fragment_a == 3

    def test_fragment_b(self):
        rp = _make_ranked_pair(pair=(3, 7))
        assert rp.fragment_b == 7

    def test_n_metrics_empty(self):
        rp = _make_ranked_pair()
        assert rp.n_metrics == 0

    def test_n_metrics_filled(self):
        rp = RankedPair(pair=(0, 1), score=0.8, rank=1,
                        scores={"color": 0.9, "texture": 0.7})
        assert rp.n_metrics == 2

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            RankedPair(pair=(0, 1), score=-0.1, rank=1)

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError):
            RankedPair(pair=(0, 1), score=0.5, rank=0)

    def test_rank_neg_raises(self):
        with pytest.raises(ValueError):
            RankedPair(pair=(0, 1), score=0.5, rank=-1)

    def test_score_zero_ok(self):
        rp = RankedPair(pair=(0, 1), score=0.0, rank=1)
        assert rp.score == 0.0

    def test_rank_one_ok(self):
        rp = _make_ranked_pair(rank=1)
        assert rp.rank == 1


# ─── TestRankResult ───────────────────────────────────────────────────────────

class TestRankResult:
    def test_top_pair_filled(self):
        r = _make_rank_result(3)
        assert r.top_pair is not None
        assert isinstance(r.top_pair, tuple)

    def test_top_pair_empty(self):
        r = RankResult(ranked=[], n_pairs=0, n_ranked=0,
                       top_score=0.0, mean_score=0.0)
        assert r.top_pair is None

    def test_compression_ratio(self):
        r = RankResult(ranked=[_make_ranked_pair()], n_pairs=4, n_ranked=1,
                       top_score=0.8, mean_score=0.8)
        assert r.compression_ratio == pytest.approx(0.25)

    def test_compression_ratio_zero_pairs(self):
        r = RankResult(ranked=[], n_pairs=0, n_ranked=0,
                       top_score=0.0, mean_score=0.0)
        assert r.compression_ratio == pytest.approx(0.0)

    def test_n_pairs_neg_raises(self):
        with pytest.raises(ValueError):
            RankResult(ranked=[], n_pairs=-1, n_ranked=0,
                       top_score=0.0, mean_score=0.0)

    def test_n_ranked_neg_raises(self):
        with pytest.raises(ValueError):
            RankResult(ranked=[], n_pairs=0, n_ranked=-1,
                       top_score=0.0, mean_score=0.0)

    def test_top_score_neg_raises(self):
        with pytest.raises(ValueError):
            RankResult(ranked=[], n_pairs=0, n_ranked=0,
                       top_score=-0.1, mean_score=0.0)

    def test_mean_score_neg_raises(self):
        with pytest.raises(ValueError):
            RankResult(ranked=[], n_pairs=0, n_ranked=0,
                       top_score=0.0, mean_score=-0.1)


# ─── TestComputePairScore ─────────────────────────────────────────────────────

class TestComputePairScore:
    def test_single_metric(self):
        s = compute_pair_score({"color": 0.8})
        assert s == pytest.approx(0.8, abs=1e-5)

    def test_equal_weights_average(self):
        s = compute_pair_score({"a": 0.4, "b": 0.6})
        assert s == pytest.approx(0.5, abs=1e-4)

    def test_custom_weights(self):
        s = compute_pair_score({"a": 1.0, "b": 0.0},
                                weights={"a": 1.0, "b": 0.0})
        assert s == pytest.approx(1.0 / 1.0, abs=1e-4)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_pair_score({})

    def test_neg_score_raises(self):
        with pytest.raises(ValueError):
            compute_pair_score({"color": -0.1})

    def test_result_nonneg(self):
        s = compute_pair_score({"a": 0.0, "b": 0.0})
        assert s >= 0.0

    def test_weights_none_equal(self):
        s1 = compute_pair_score({"a": 0.5, "b": 0.5}, weights=None)
        s2 = compute_pair_score({"a": 0.5, "b": 0.5},
                                 weights={"a": 1.0, "b": 1.0})
        assert s1 == pytest.approx(s2, abs=1e-5)


# ─── TestRankPairs ────────────────────────────────────────────────────────────

class TestRankPairs:
    def test_returns_rank_result(self):
        r = rank_pairs([(0, 1), (1, 2)], [0.8, 0.6])
        assert isinstance(r, RankResult)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rank_pairs([(0, 1)], [0.8, 0.6])

    def test_sorted_by_score_desc(self):
        r = rank_pairs([(0, 1), (1, 2), (2, 3)], [0.4, 0.9, 0.6])
        scores = [rp.score for rp in r.ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_one_based(self):
        r = rank_pairs([(0, 1), (1, 2)], [0.8, 0.6])
        assert r.ranked[0].rank == 1
        assert r.ranked[1].rank == 2

    def test_min_score_filter(self):
        cfg = RankConfig(min_score=0.7)
        r = rank_pairs([(0, 1), (1, 2)], [0.8, 0.5], cfg)
        assert all(rp.score >= 0.7 for rp in r.ranked)

    def test_top_k_respected(self):
        cfg = RankConfig(top_k=2)
        r = rank_pairs([(0, 1), (1, 2), (2, 3)], [0.9, 0.8, 0.7], cfg)
        assert r.n_ranked == 2

    def test_deduplicate_canonical_pair(self):
        cfg = RankConfig(deduplicate=True)
        # (1,0) and (0,1) should deduplicate to one pair
        r = rank_pairs([(0, 1), (1, 0)], [0.8, 0.6], cfg)
        assert r.n_ranked == 1

    def test_empty_pairs_empty_result(self):
        r = rank_pairs([], [])
        assert r.n_ranked == 0
        assert r.n_pairs == 0

    def test_n_pairs_counts_input(self):
        r = rank_pairs([(0, 1), (1, 2), (2, 3)], [0.9, 0.8, 0.7])
        assert r.n_pairs == 3

    def test_top_score_is_max(self):
        r = rank_pairs([(0, 1), (1, 2)], [0.6, 0.9])
        assert r.top_score == pytest.approx(0.9)

    def test_ascending_order(self):
        cfg = RankConfig(ascending=True)
        r = rank_pairs([(0, 1), (1, 2), (2, 3)], [0.4, 0.9, 0.6], cfg)
        scores = [rp.score for rp in r.ranked]
        assert scores == sorted(scores)

    def test_all_filtered_out_empty_result(self):
        cfg = RankConfig(min_score=1.0)
        r = rank_pairs([(0, 1), (1, 2)], [0.8, 0.6], cfg)
        assert r.n_ranked == 0


# ─── TestBuildRankMatrix ──────────────────────────────────────────────────────

class TestBuildRankMatrix:
    def test_returns_ndarray(self):
        r = _make_rank_result(3)
        mat = build_rank_matrix(r, n_fragments=5)
        assert isinstance(mat, np.ndarray)

    def test_shape(self):
        r = _make_rank_result(2)
        mat = build_rank_matrix(r, n_fragments=4)
        assert mat.shape == (4, 4)

    def test_symmetric(self):
        r = _make_rank_result(3)
        mat = build_rank_matrix(r, n_fragments=5)
        np.testing.assert_array_equal(mat, mat.T)

    def test_n_fragments_zero_raises(self):
        r = _make_rank_result()
        with pytest.raises(ValueError):
            build_rank_matrix(r, n_fragments=0)

    def test_n_fragments_neg_raises(self):
        r = _make_rank_result()
        with pytest.raises(ValueError):
            build_rank_matrix(r, n_fragments=-1)

    def test_dtype_int(self):
        r = _make_rank_result(2)
        mat = build_rank_matrix(r, n_fragments=4)
        assert mat.dtype == int

    def test_unranked_pairs_zero(self):
        r = RankResult(
            ranked=[RankedPair(pair=(0, 1), score=0.9, rank=1)],
            n_pairs=1, n_ranked=1, top_score=0.9, mean_score=0.9,
        )
        mat = build_rank_matrix(r, n_fragments=4)
        # (0,1) and (1,0) should have rank 1; rest 0
        assert mat[0, 1] == 1
        assert mat[1, 0] == 1
        assert mat[0, 2] == 0


# ─── TestMergeRankResults ─────────────────────────────────────────────────────

class TestMergeRankResults:
    def test_returns_rank_result(self):
        r1 = _make_rank_result(2)
        r2 = _make_rank_result(2)
        merged = merge_rank_results([r1, r2])
        assert isinstance(merged, RankResult)

    def test_single_result(self):
        r = _make_rank_result(3)
        merged = merge_rank_results([r])
        assert isinstance(merged, RankResult)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_rank_results([])

    def test_merged_has_pairs(self):
        r1 = _make_rank_result(2)
        r2 = _make_rank_result(2)
        merged = merge_rank_results([r1, r2])
        assert merged.n_ranked > 0

    def test_merge_keeps_best_score(self):
        r1 = RankResult(
            ranked=[RankedPair(pair=(0, 1), score=0.9, rank=1)],
            n_pairs=1, n_ranked=1, top_score=0.9, mean_score=0.9,
        )
        r2 = RankResult(
            ranked=[RankedPair(pair=(1, 2), score=0.7, rank=1)],
            n_pairs=1, n_ranked=1, top_score=0.7, mean_score=0.7,
        )
        merged = merge_rank_results([r1, r2])
        assert merged.top_score >= 0.7
