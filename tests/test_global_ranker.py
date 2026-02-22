"""Тесты для puzzle_reconstruction.scoring.global_ranker."""
import numpy as np
import pytest

from puzzle_reconstruction.scoring.global_ranker import (
    RankedPair,
    RankingConfig,
    normalize_matrix,
    aggregate_score_matrices,
    rank_pairs,
    top_k_candidates,
    global_rank,
    score_vector,
    batch_global_rank,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rand_matrix(n=4, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((n, n)).astype(np.float64)
    np.fill_diagonal(M, 0.0)
    return M


def _matrices(n=4):
    return {
        "boundary": _rand_matrix(n, seed=0),
        "sift": _rand_matrix(n, seed=1),
    }


# ─── TestRankedPair ───────────────────────────────────────────────────────────

class TestRankedPair:
    def test_basic_creation(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.8, rank=0)
        assert rp.score == pytest.approx(0.8)

    def test_pair_property(self):
        rp = RankedPair(idx1=2, idx2=5, score=0.5, rank=3)
        assert rp.pair == (2, 5)

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError):
            RankedPair(idx1=-1, idx2=0, score=0.5, rank=0)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError):
            RankedPair(idx1=0, idx2=-1, score=0.5, rank=0)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            RankedPair(idx1=0, idx2=1, score=-0.1, rank=0)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            RankedPair(idx1=0, idx2=1, score=1.1, rank=0)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError):
            RankedPair(idx1=0, idx2=1, score=0.5, rank=-1)

    def test_component_scores_default_empty(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.5, rank=0)
        assert rp.component_scores == {}

    def test_score_zero_valid(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.0, rank=0)
        assert rp.score == 0.0

    def test_score_one_valid(self):
        rp = RankedPair(idx1=0, idx2=1, score=1.0, rank=0)
        assert rp.score == 1.0


# ─── TestRankingConfig ────────────────────────────────────────────────────────

class TestRankingConfig:
    def test_default_values(self):
        cfg = RankingConfig()
        assert cfg.top_k == 5
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.normalize is True

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            RankingConfig(weights={"a": -1.0})

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError):
            RankingConfig(top_k=0)

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError):
            RankingConfig(min_score=-0.1)

    def test_top_k_one_valid(self):
        cfg = RankingConfig(top_k=1)
        assert cfg.top_k == 1

    def test_zero_weight_valid(self):
        cfg = RankingConfig(weights={"a": 0.0})
        assert cfg.weights["a"] == pytest.approx(0.0)


# ─── TestNormalizeMatrix ──────────────────────────────────────────────────────

class TestNormalizeMatrix:
    def test_output_range(self):
        M = _rand_matrix(5)
        N = normalize_matrix(M)
        mask = ~np.eye(5, dtype=bool)
        vals = N[mask]
        assert vals.min() >= 0.0 - 1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_diagonal_zero(self):
        M = _rand_matrix(4)
        N = normalize_matrix(M)
        assert np.all(np.diag(N) == 0.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.ones((3, 4)))

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.ones(4))

    def test_uniform_matrix_returns_zeros(self):
        M = np.ones((3, 3))
        np.fill_diagonal(M, 0.0)
        N = normalize_matrix(M)
        assert np.allclose(N, 0.0)

    def test_returns_float64(self):
        M = _rand_matrix(3)
        N = normalize_matrix(M)
        assert N.dtype == np.float64


# ─── TestAggregateScoreMatrices ───────────────────────────────────────────────

class TestAggregateScoreMatrices:
    def test_returns_ndarray(self):
        mats = _matrices(4)
        agg = aggregate_score_matrices(mats)
        assert isinstance(agg, np.ndarray)

    def test_shape_preserved(self):
        mats = _matrices(5)
        agg = aggregate_score_matrices(mats)
        assert agg.shape == (5, 5)

    def test_diagonal_zero(self):
        mats = _matrices(4)
        agg = aggregate_score_matrices(mats)
        assert np.allclose(np.diag(agg), 0.0)

    def test_empty_matrices_raises(self):
        with pytest.raises(ValueError):
            aggregate_score_matrices({})

    def test_mismatched_shape_raises(self):
        mats = {
            "a": _rand_matrix(3),
            "b": _rand_matrix(4),
        }
        with pytest.raises(ValueError):
            aggregate_score_matrices(mats)

    def test_symmetric_when_flag_set(self):
        mats = _matrices(4)
        agg = aggregate_score_matrices(mats, symmetric=True)
        assert np.allclose(agg, agg.T)

    def test_single_matrix(self):
        M = _rand_matrix(4)
        agg = aggregate_score_matrices({"a": M})
        assert agg.shape == (4, 4)


# ─── TestRankPairs ────────────────────────────────────────────────────────────

class TestRankPairs:
    def test_returns_list(self):
        M = _rand_matrix(4)
        result = rank_pairs(M)
        assert isinstance(result, list)

    def test_correct_number_of_pairs(self):
        M = _rand_matrix(4)
        result = rank_pairs(M)
        assert len(result) == 6  # C(4,2)

    def test_all_ranked_pairs(self):
        M = _rand_matrix(4)
        result = rank_pairs(M)
        assert all(isinstance(rp, RankedPair) for rp in result)

    def test_sorted_descending(self):
        M = _rand_matrix(5)
        result = rank_pairs(M)
        scores = [rp.score for rp in result]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters(self):
        M = _rand_matrix(4)
        all_pairs = rank_pairs(M, min_score=0.0)
        filtered = rank_pairs(M, min_score=0.9)
        assert len(filtered) <= len(all_pairs)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            rank_pairs(np.ones((3, 4)))

    def test_ranks_sequential(self):
        M = _rand_matrix(4)
        result = rank_pairs(M)
        for i, rp in enumerate(result):
            assert rp.rank == i

    def test_idx1_lt_idx2(self):
        M = _rand_matrix(5)
        result = rank_pairs(M)
        for rp in result:
            assert rp.idx1 < rp.idx2


# ─── TestTopKCandidates ───────────────────────────────────────────────────────

class TestTopKCandidates:
    def _pairs(self, n=4):
        M = _rand_matrix(n)
        return rank_pairs(M)

    def test_returns_dict(self):
        result = top_k_candidates(self._pairs(), n_fragments=4, k=2)
        assert isinstance(result, dict)

    def test_keys_are_fragment_ids(self):
        result = top_k_candidates(self._pairs(), n_fragments=4, k=2)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_at_most_k_per_fragment(self):
        result = top_k_candidates(self._pairs(), n_fragments=4, k=2)
        for candidates in result.values():
            assert len(candidates) <= 2

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_candidates([], n_fragments=4, k=0)

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            top_k_candidates([], n_fragments=0, k=1)

    def test_empty_pairs_empty_lists(self):
        result = top_k_candidates([], n_fragments=3, k=2)
        for v in result.values():
            assert v == []


# ─── TestGlobalRank ───────────────────────────────────────────────────────────

class TestGlobalRank:
    def test_returns_list(self):
        mats = _matrices(4)
        result = global_rank(mats)
        assert isinstance(result, list)

    def test_all_ranked_pairs(self):
        mats = _matrices(4)
        result = global_rank(mats)
        assert all(isinstance(rp, RankedPair) for rp in result)

    def test_sorted_descending(self):
        mats = _matrices(5)
        result = global_rank(mats)
        scores = [rp.score for rp in result]
        assert scores == sorted(scores, reverse=True)

    def test_with_config(self):
        mats = _matrices(4)
        cfg = RankingConfig(top_k=2, min_score=0.0)
        result = global_rank(mats, cfg)
        assert isinstance(result, list)

    def test_default_config(self):
        mats = _matrices(4)
        result = global_rank(mats, None)
        assert isinstance(result, list)


# ─── TestScoreVector ──────────────────────────────────────────────────────────

class TestScoreVector:
    def test_returns_ndarray(self):
        mats = _matrices(4)
        pairs = global_rank(mats)
        sv = score_vector(pairs, n_fragments=4)
        assert isinstance(sv, np.ndarray)

    def test_shape(self):
        mats = _matrices(5)
        pairs = global_rank(mats)
        sv = score_vector(pairs, n_fragments=5)
        assert sv.shape == (5,)

    def test_nonnegative(self):
        mats = _matrices(4)
        pairs = global_rank(mats)
        sv = score_vector(pairs, n_fragments=4)
        assert np.all(sv >= 0.0)

    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError):
            score_vector([], n_fragments=0)

    def test_empty_pairs_zeros(self):
        sv = score_vector([], n_fragments=3)
        assert np.allclose(sv, 0.0)

    def test_dtype_float64(self):
        mats = _matrices(4)
        pairs = global_rank(mats)
        sv = score_vector(pairs, n_fragments=4)
        assert sv.dtype == np.float64


# ─── TestBatchGlobalRank ─────────────────────────────────────────────────────

class TestBatchGlobalRank:
    def test_returns_list(self):
        groups = [_matrices(4), _matrices(3)]
        result = batch_global_rank(groups)
        assert isinstance(result, list)

    def test_correct_length(self):
        groups = [_matrices(4), _matrices(3), _matrices(5)]
        result = batch_global_rank(groups)
        assert len(result) == 3

    def test_each_inner_list(self):
        groups = [_matrices(4)]
        result = batch_global_rank(groups)
        assert isinstance(result[0], list)

    def test_empty_groups(self):
        result = batch_global_rank([])
        assert result == []

    def test_with_config(self):
        cfg = RankingConfig(top_k=3)
        groups = [_matrices(4)]
        result = batch_global_rank(groups, cfg=cfg)
        assert len(result) == 1
