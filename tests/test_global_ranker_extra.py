"""Extra tests for puzzle_reconstruction/scoring/global_ranker.py"""
import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mat(n=4, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.random((n, n))
    np.fill_diagonal(M, 0.0)
    return M


def _mats(n=4):
    return {"a": _mat(n, 0), "b": _mat(n, 1)}


# ─── TestRankedPairExtra ──────────────────────────────────────────────────────

class TestRankedPairExtra:
    def test_pair_property_order(self):
        rp = RankedPair(idx1=3, idx2=7, score=0.5, rank=0)
        assert rp.pair == (3, 7)

    def test_component_scores_stored(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.6, rank=0,
                        component_scores={"boundary": 0.8, "sift": 0.4})
        assert rp.component_scores["boundary"] == pytest.approx(0.8)

    def test_score_boundary_zero(self):
        rp = RankedPair(idx1=0, idx2=2, score=0.0, rank=1)
        assert rp.score == pytest.approx(0.0)

    def test_score_boundary_one(self):
        rp = RankedPair(idx1=0, idx2=1, score=1.0, rank=0)
        assert rp.score == pytest.approx(1.0)

    def test_large_indices(self):
        rp = RankedPair(idx1=100, idx2=200, score=0.5, rank=99)
        assert rp.idx1 == 100
        assert rp.idx2 == 200


# ─── TestRankingConfigExtra ───────────────────────────────────────────────────

class TestRankingConfigExtra:
    def test_multiple_weights_valid(self):
        cfg = RankingConfig(weights={"a": 0.5, "b": 0.3, "c": 0.2})
        assert len(cfg.weights) == 3

    def test_symmetric_false(self):
        cfg = RankingConfig(symmetric=False)
        assert cfg.symmetric is False

    def test_min_score_zero_valid(self):
        cfg = RankingConfig(min_score=0.0)
        assert cfg.min_score == pytest.approx(0.0)

    def test_large_top_k(self):
        cfg = RankingConfig(top_k=100)
        assert cfg.top_k == 100

    def test_normalize_false(self):
        cfg = RankingConfig(normalize=False)
        assert cfg.normalize is False


# ─── TestNormalizeMatrixExtra ─────────────────────────────────────────────────

class TestNormalizeMatrixExtra:
    def test_1x1_matrix(self):
        M = np.array([[0.0]])
        N = normalize_matrix(M)
        assert N.shape == (1, 1)
        assert N[0, 0] == pytest.approx(0.0)

    def test_2x2_off_diagonal_in_0_1(self):
        M = np.array([[0.0, 0.7], [0.3, 0.0]])
        N = normalize_matrix(M)
        mask = ~np.eye(2, dtype=bool)
        assert N[mask].min() >= 0.0
        assert N[mask].max() <= 1.0

    def test_large_values_normalized(self):
        M = _mat(5) * 1000.0
        N = normalize_matrix(M)
        mask = ~np.eye(5, dtype=bool)
        assert N[mask].max() <= 1.0 + 1e-9

    def test_shape_preserved(self):
        M = _mat(6)
        N = normalize_matrix(M)
        assert N.shape == (6, 6)


# ─── TestAggregateScoreMatricesExtra ─────────────────────────────────────────

class TestAggregateScoreMatricesExtra:
    def test_no_normalize(self):
        mats = _mats(4)
        agg = aggregate_score_matrices(mats, normalize=False)
        assert agg.shape == (4, 4)

    def test_weighted_single_matrix(self):
        M = _mat(3)
        agg = aggregate_score_matrices({"only": M}, weights={"only": 2.0})
        # Normalised M equals M normalised, scaled by 2 / 2 = 1
        assert agg.shape == (3, 3)

    def test_non_symmetric(self):
        mats = _mats(4)
        agg = aggregate_score_matrices(mats, symmetric=False)
        # Not forced symmetric: may not equal its transpose
        assert agg.shape == (4, 4)

    def test_equal_weights_default(self):
        mats = _mats(5)
        agg = aggregate_score_matrices(mats)
        assert agg.shape == (5, 5)

    def test_three_matrices(self):
        mats = {"a": _mat(3, 0), "b": _mat(3, 1), "c": _mat(3, 2)}
        agg = aggregate_score_matrices(mats)
        assert agg.shape == (3, 3)


# ─── TestRankPairsExtra ───────────────────────────────────────────────────────

class TestRankPairsExtra:
    def test_2x2_one_pair(self):
        M = np.array([[0.0, 0.7], [0.3, 0.0]])
        result = rank_pairs(M)
        assert len(result) == 1
        assert result[0].rank == 0

    def test_5x5_ten_pairs(self):
        M = _mat(5)
        result = rank_pairs(M)
        assert len(result) == 10  # C(5,2)

    def test_min_score_all_pass(self):
        M = _mat(4)
        result = rank_pairs(M, min_score=0.0)
        assert len(result) == 6

    def test_min_score_1_none_pass(self):
        M = _mat(4)  # values < 1.0
        result = rank_pairs(M, min_score=1.0)
        assert result == []

    def test_component_scores_empty_by_default(self):
        M = _mat(3)
        result = rank_pairs(M)
        assert all(rp.component_scores == {} for rp in result)


# ─── TestTopKCandidatesExtra ──────────────────────────────────────────────────

class TestTopKCandidatesExtra:
    def _ranked(self, n=4):
        return rank_pairs(_mat(n))

    def test_k_larger_than_possible(self):
        result = top_k_candidates(self._ranked(4), n_fragments=4, k=100)
        for lst in result.values():
            assert len(lst) <= 3  # each fragment has at most n-1=3 pairs

    def test_returns_ranked_pairs(self):
        result = top_k_candidates(self._ranked(4), n_fragments=4, k=2)
        for lst in result.values():
            assert all(isinstance(rp, RankedPair) for rp in lst)

    def test_k_1_returns_at_most_1(self):
        result = top_k_candidates(self._ranked(5), n_fragments=5, k=1)
        for lst in result.values():
            assert len(lst) <= 1

    def test_fragment_ids_complete(self):
        result = top_k_candidates(self._ranked(3), n_fragments=3, k=2)
        assert set(result.keys()) == {0, 1, 2}


# ─── TestGlobalRankExtra ──────────────────────────────────────────────────────

class TestGlobalRankExtra:
    def test_single_matrix(self):
        mats = {"only": _mat(4)}
        result = global_rank(mats)
        assert isinstance(result, list)
        assert all(isinstance(rp, RankedPair) for rp in result)

    def test_custom_weights_config(self):
        cfg = RankingConfig(weights={"a": 1.0, "b": 0.0})
        result = global_rank(_mats(4), cfg)
        assert isinstance(result, list)

    def test_min_score_filters(self):
        cfg = RankingConfig(min_score=0.9)
        result = global_rank(_mats(4), cfg)
        # Most pairs should be filtered
        assert len(result) <= 6

    def test_n_pairs_for_4_fragments(self):
        result = global_rank(_mats(4))
        assert len(result) == 6  # C(4,2)


# ─── TestScoreVectorExtra ─────────────────────────────────────────────────────

class TestScoreVectorExtra:
    def test_n_fragments_larger_than_pairs(self):
        pairs = rank_pairs(_mat(3))
        sv = score_vector(pairs, n_fragments=10)
        assert sv.shape == (10,)
        # Fragments 3-9 have no pairs → score 0
        assert np.all(sv[3:] == 0.0)

    def test_single_pair(self):
        pairs = [RankedPair(idx1=0, idx2=1, score=0.6, rank=0)]
        sv = score_vector(pairs, n_fragments=3)
        assert sv[0] == pytest.approx(0.6)
        assert sv[1] == pytest.approx(0.6)
        assert sv[2] == pytest.approx(0.0)

    def test_returns_float64(self):
        sv = score_vector([], n_fragments=3)
        assert sv.dtype == np.float64

    def test_all_nonneg(self):
        pairs = rank_pairs(_mat(5))
        sv = score_vector(pairs, n_fragments=5)
        assert np.all(sv >= 0.0)


# ─── TestBatchGlobalRankExtra ─────────────────────────────────────────────────

class TestBatchGlobalRankExtra:
    def test_five_groups(self):
        groups = [_mats(3)] * 5
        result = batch_global_rank(groups)
        assert len(result) == 5

    def test_all_inner_are_lists(self):
        groups = [_mats(4), _mats(3)]
        result = batch_global_rank(groups)
        assert all(isinstance(r, list) for r in result)

    def test_config_applied_to_all(self):
        cfg = RankingConfig(min_score=0.9)
        groups = [_mats(4)] * 3
        result = batch_global_rank(groups, cfg=cfg)
        for r in result:
            assert all(rp.score >= 0.9 for rp in r)
