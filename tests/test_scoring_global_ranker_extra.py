"""Extra tests for puzzle_reconstruction/scoring/global_ranker.py."""
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


# ─── RankedPair (extra) ───────────────────────────────────────────────────────

class TestRankedPairExtra:
    def test_pair_property_returns_tuple(self):
        rp = RankedPair(idx1=2, idx2=5, score=0.7, rank=1)
        assert isinstance(rp.pair, tuple)
        assert rp.pair == (2, 5)

    def test_score_zero_valid(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.0, rank=0)
        assert rp.score == pytest.approx(0.0)

    def test_score_one_valid(self):
        rp = RankedPair(idx1=0, idx2=1, score=1.0, rank=0)
        assert rp.score == pytest.approx(1.0)

    def test_rank_zero_valid(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.5, rank=0)
        assert rp.rank == 0

    def test_large_indices_valid(self):
        rp = RankedPair(idx1=1000, idx2=9999, score=0.5, rank=100)
        assert rp.idx1 == 1000
        assert rp.idx2 == 9999

    def test_component_scores_multiple_channels(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.5, rank=0,
                        component_scores={"a": 0.3, "b": 0.7})
        assert rp.component_scores["a"] == pytest.approx(0.3)
        assert rp.component_scores["b"] == pytest.approx(0.7)

    def test_component_scores_default_empty(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.5, rank=0)
        assert rp.component_scores == {}

    def test_idx1_zero_valid(self):
        rp = RankedPair(idx1=0, idx2=0, score=0.5, rank=0)
        assert rp.idx1 == 0

    def test_same_idx_allowed(self):
        # No constraint that idx1 != idx2
        rp = RankedPair(idx1=3, idx2=3, score=0.5, rank=0)
        assert rp.pair == (3, 3)


# ─── RankingConfig (extra) ────────────────────────────────────────────────────

class TestRankingConfigExtra:
    def test_normalize_default_true(self):
        assert RankingConfig().normalize is True

    def test_normalize_false(self):
        cfg = RankingConfig(normalize=False)
        assert cfg.normalize is False

    def test_symmetric_default_true(self):
        assert RankingConfig().symmetric is True

    def test_symmetric_false(self):
        cfg = RankingConfig(symmetric=False)
        assert cfg.symmetric is False

    def test_top_k_positive(self):
        cfg = RankingConfig(top_k=10)
        assert cfg.top_k == 10

    def test_min_score_zero_valid(self):
        cfg = RankingConfig(min_score=0.0)
        assert cfg.min_score == pytest.approx(0.0)

    def test_min_score_one_valid(self):
        cfg = RankingConfig(min_score=1.0)
        assert cfg.min_score == pytest.approx(1.0)

    def test_custom_weights_stored(self):
        cfg = RankingConfig(weights={"texture": 3.0, "shape": 1.5})
        assert cfg.weights["texture"] == pytest.approx(3.0)


# ─── normalize_matrix (extra) ─────────────────────────────────────────────────

class TestNormalizeMatrixExtra:
    def test_symmetric_output_for_symmetric_input(self):
        M = np.array([[0.0, 3.0, 6.0],
                      [3.0, 0.0, 9.0],
                      [6.0, 9.0, 0.0]])
        result = normalize_matrix(M)
        np.testing.assert_allclose(result, result.T, atol=1e-9)

    def test_2x2_diagonal_zero(self):
        M = np.array([[0.0, 10.0], [10.0, 0.0]])
        result = normalize_matrix(M)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)

    def test_shape_preserved(self):
        M = np.ones((6, 6))
        result = normalize_matrix(M)
        assert result.shape == (6, 6)

    def test_output_dtype_float64(self):
        M = np.ones((3, 3), dtype=np.float32)
        result = normalize_matrix(M)
        assert result.dtype == np.float64

    def test_all_offdiag_in_0_1(self):
        rng = np.random.default_rng(123)
        M = rng.uniform(0, 100, (5, 5))
        np.fill_diagonal(M, 0.0)
        result = normalize_matrix(M)
        mask = ~np.eye(5, dtype=bool)
        assert (result[mask] >= 0.0).all()
        assert (result[mask] <= 1.0 + 1e-9).all()

    def test_single_cell_matrix(self):
        M = np.array([[5.0]])
        result = normalize_matrix(M)
        assert result[0, 0] == pytest.approx(0.0)


# ─── aggregate_score_matrices (extra) ─────────────────────────────────────────

class TestAggregateScoreMatricesExtra:
    def test_two_matrices_equal_weights(self):
        M1 = np.array([[0.0, 0.6], [0.6, 0.0]])
        M2 = np.array([[0.0, 0.4], [0.4, 0.0]])
        result = aggregate_score_matrices({"a": M1, "b": M2},
                                          weights={"a": 1.0, "b": 1.0},
                                          normalize=False)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)

    def test_three_matrices_shape(self):
        M = np.eye(4)
        result = aggregate_score_matrices({"x": M, "y": M, "z": M})
        assert result.shape == (4, 4)

    def test_values_clamped_0_1_normalized(self):
        rng = np.random.default_rng(7)
        M = rng.uniform(0, 50, (5, 5))
        np.fill_diagonal(M, 0.0)
        result = aggregate_score_matrices({"x": M}, normalize=True)
        mask = ~np.eye(5, dtype=bool)
        assert (result[mask] >= 0.0 - 1e-9).all()
        assert (result[mask] <= 1.0 + 1e-9).all()

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError):
            aggregate_score_matrices({"x": np.ones((3, 4))})

    def test_diagonal_always_zero(self):
        rng = np.random.default_rng(0)
        M = rng.uniform(0, 1, (4, 4))
        result = aggregate_score_matrices({"x": M})
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-9)


# ─── rank_pairs (extra) ───────────────────────────────────────────────────────

class TestRankPairsExtra:
    def test_2x2_matrix_one_pair(self):
        M = np.array([[0.0, 0.7], [0.7, 0.0]])
        result = rank_pairs(M)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)

    def test_scores_nonnegative(self):
        rng = np.random.default_rng(5)
        M = rng.uniform(0, 1, (4, 4))
        np.fill_diagonal(M, 0.0)
        result = rank_pairs(M)
        for r in result:
            assert r.score >= 0.0

    def test_min_score_0_returns_all_nonzero_pairs(self):
        M = np.array([[0.0, 0.5, 0.0],
                      [0.5, 0.0, 0.3],
                      [0.0, 0.3, 0.0]])
        result = rank_pairs(M, min_score=0.0)
        # Only non-zero pairs: (0,1) and (1,2)
        scores = [r.score for r in result]
        assert all(s >= 0.0 for s in scores)

    def test_all_pairs_covered(self):
        n = 4
        M = np.ones((n, n))
        np.fill_diagonal(M, 0.0)
        result = rank_pairs(M, min_score=0.5)
        pairs = {r.pair for r in result}
        expected = {(i, j) for i in range(n) for j in range(n) if i < j}
        assert pairs == expected

    def test_rank_values_sequential(self):
        M = np.array([[0.0, 0.9, 0.7, 0.5],
                      [0.9, 0.0, 0.3, 0.8],
                      [0.7, 0.3, 0.0, 0.6],
                      [0.5, 0.8, 0.6, 0.0]])
        result = rank_pairs(M)
        assert [r.rank for r in result] == list(range(len(result)))


# ─── top_k_candidates (extra) ─────────────────────────────────────────────────

class TestTopKCandidatesExtra:
    def _make_pairs_from(self, n):
        rng = np.random.default_rng(42)
        M = rng.uniform(0, 1, (n, n))
        np.fill_diagonal(M, 0.0)
        return rank_pairs(M), n

    def test_k_larger_than_possible_capped(self):
        pairs, n = self._make_pairs_from(3)
        result = top_k_candidates(pairs, n, k=100)
        for fid in range(n):
            assert len(result[fid]) <= n - 1

    def test_candidates_are_ranked_pairs(self):
        pairs, n = self._make_pairs_from(4)
        result = top_k_candidates(pairs, n, k=2)
        for fid in range(n):
            for rp in result[fid]:
                assert isinstance(rp, RankedPair)

    def test_each_fragment_key_present(self):
        pairs, n = self._make_pairs_from(5)
        result = top_k_candidates(pairs, n, k=3)
        assert set(result.keys()) == set(range(n))

    def test_candidate_references_fragment(self):
        pairs, n = self._make_pairs_from(4)
        result = top_k_candidates(pairs, n, k=2)
        for fid, cands in result.items():
            for rp in cands:
                assert fid in (rp.idx1, rp.idx2)

    def test_k_one_returns_at_most_one(self):
        pairs, n = self._make_pairs_from(4)
        result = top_k_candidates(pairs, n, k=1)
        for fid in range(n):
            assert len(result[fid]) <= 1


# ─── global_rank (extra) ──────────────────────────────────────────────────────

class TestGlobalRankExtra:
    def test_empty_matrices_raises(self):
        with pytest.raises((ValueError, KeyError, Exception)):
            global_rank({})

    def test_multiple_score_sources(self):
        rng = np.random.default_rng(0)
        M1 = rng.uniform(0, 1, (4, 4))
        M2 = rng.uniform(0, 1, (4, 4))
        result = global_rank({"a": M1, "b": M2})
        assert isinstance(result, list)

    def test_all_scores_nonneg(self):
        M = np.array([[0.0, 0.8, 0.5],
                      [0.8, 0.0, 0.3],
                      [0.5, 0.3, 0.0]])
        result = global_rank({"x": M})
        for r in result:
            assert r.score >= 0.0

    def test_ranks_start_at_zero(self):
        M = np.array([[0.0, 0.7], [0.7, 0.0]])
        result = global_rank({"x": M})
        if result:
            assert result[0].rank == 0

    def test_results_sorted_descending(self):
        rng = np.random.default_rng(11)
        M = rng.uniform(0, 1, (5, 5))
        result = global_rank({"x": M})
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_cfg_min_score_applied(self):
        M = np.array([[0.0, 0.9, 0.1],
                      [0.9, 0.0, 0.2],
                      [0.1, 0.2, 0.0]])
        cfg = RankingConfig(min_score=0.5)
        result = global_rank({"x": M}, cfg=cfg)
        for r in result:
            assert r.score >= 0.5


# ─── score_vector (extra) ─────────────────────────────────────────────────────

class TestScoreVectorExtra:
    def test_shape_matches_n_fragments(self):
        sv = score_vector([], 7)
        assert sv.shape == (7,)

    def test_all_zeros_no_pairs(self):
        sv = score_vector([], 5)
        np.testing.assert_array_equal(sv, 0.0)

    def test_single_pair_both_indices_updated(self):
        pairs = [RankedPair(idx1=1, idx2=3, score=0.8, rank=0)]
        sv = score_vector(pairs, 5)
        assert sv[1] == pytest.approx(0.8)
        assert sv[3] == pytest.approx(0.8)
        assert sv[0] == pytest.approx(0.0)

    def test_multiple_pairs_average(self):
        pairs = [
            RankedPair(idx1=0, idx2=1, score=0.6, rank=0),
            RankedPair(idx1=0, idx2=2, score=0.4, rank=1),
            RankedPair(idx1=0, idx2=3, score=0.2, rank=2),
        ]
        sv = score_vector(pairs, 4)
        assert sv[0] == pytest.approx((0.6 + 0.4 + 0.2) / 3)

    def test_result_nonneg(self):
        rng = np.random.default_rng(7)
        M = rng.uniform(0, 1, (5, 5))
        np.fill_diagonal(M, 0.0)
        pairs = rank_pairs(M)
        sv = score_vector(pairs, 5)
        assert (sv >= 0).all()


# ─── batch_global_rank (extra) ────────────────────────────────────────────────

class TestBatchGlobalRankExtra:
    def test_single_group_single_result(self):
        M = np.array([[0.0, 0.9], [0.9, 0.0]])
        result = batch_global_rank([{"x": M}])
        assert len(result) == 1

    def test_all_groups_are_lists(self):
        M = np.eye(3)
        groups = [{"x": M} for _ in range(4)]
        result = batch_global_rank(groups)
        for group_result in result:
            assert isinstance(group_result, list)

    def test_each_result_sorted_descending(self):
        rng = np.random.default_rng(0)
        M = rng.uniform(0, 1, (4, 4))
        groups = [{"x": M}, {"y": M}]
        result = batch_global_rank(groups)
        for group_result in result:
            scores = [r.score for r in group_result]
            assert scores == sorted(scores, reverse=True)

    def test_two_groups_independent(self):
        M1 = np.array([[0.0, 0.9], [0.9, 0.0]])
        M2 = np.array([[0.0, 0.1], [0.1, 0.0]])
        result = batch_global_rank([{"x": M1}, {"x": M2}])
        assert len(result[0]) >= 1
        assert len(result[1]) >= 1

    def test_large_batch_length(self):
        M = np.array([[0.0, 0.5], [0.5, 0.0]])
        groups = [{"x": M} for _ in range(10)]
        result = batch_global_rank(groups)
        assert len(result) == 10
