"""Тесты для puzzle_reconstruction/scoring/global_ranker.py."""
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


# ─── RankedPair ───────────────────────────────────────────────────────────────

class TestRankedPair:
    def test_creation(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.8, rank=0)
        assert rp.idx1 == 0
        assert rp.idx2 == 1
        assert rp.score == pytest.approx(0.8)
        assert rp.rank == 0
        assert rp.component_scores == {}

    def test_negative_idx1_raises(self):
        with pytest.raises(ValueError, match="idx1"):
            RankedPair(idx1=-1, idx2=0, score=0.5, rank=0)

    def test_negative_idx2_raises(self):
        with pytest.raises(ValueError, match="idx2"):
            RankedPair(idx1=0, idx2=-1, score=0.5, rank=0)

    def test_score_above_1_raises(self):
        with pytest.raises(ValueError, match="score"):
            RankedPair(idx1=0, idx2=1, score=1.1, rank=0)

    def test_score_below_0_raises(self):
        with pytest.raises(ValueError, match="score"):
            RankedPair(idx1=0, idx2=1, score=-0.1, rank=0)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError, match="rank"):
            RankedPair(idx1=0, idx2=1, score=0.5, rank=-1)

    def test_pair_property(self):
        rp = RankedPair(idx1=3, idx2=7, score=0.5, rank=2)
        assert rp.pair == (3, 7)

    def test_component_scores_stored(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.5, rank=0,
                        component_scores={"boundary": 0.7})
        assert rp.component_scores["boundary"] == pytest.approx(0.7)

    def test_score_boundary_0(self):
        rp = RankedPair(idx1=0, idx2=1, score=0.0, rank=0)
        assert rp.score == pytest.approx(0.0)

    def test_score_boundary_1(self):
        rp = RankedPair(idx1=0, idx2=1, score=1.0, rank=0)
        assert rp.score == pytest.approx(1.0)


# ─── RankingConfig ────────────────────────────────────────────────────────────

class TestRankingConfig:
    def test_defaults(self):
        cfg = RankingConfig()
        assert "boundary" in cfg.weights
        assert cfg.top_k == 5
        assert cfg.normalize is True
        assert cfg.min_score == 0.0
        assert cfg.symmetric is True

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            RankingConfig(weights={"x": -0.1})

    def test_top_k_zero_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            RankingConfig(top_k=0)

    def test_negative_min_score_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            RankingConfig(min_score=-0.1)

    def test_all_zero_weights_valid(self):
        # Zero weights per source are allowed
        cfg = RankingConfig(weights={"a": 0.0, "b": 0.0})
        assert cfg.weights["a"] == 0.0


# ─── normalize_matrix ─────────────────────────────────────────────────────────

class TestNormalizeMatrix:
    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.ones((2, 3)))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_matrix(np.ones((2, 2, 2)))

    def test_output_diagonal_zero(self):
        M = np.array([[5.0, 3.0, 1.0],
                      [2.0, 4.0, 0.5],
                      [1.5, 2.5, 3.0]])
        result = normalize_matrix(M)
        np.testing.assert_array_equal(np.diag(result), [0.0, 0.0, 0.0])

    def test_off_diagonal_in_0_1(self):
        rng = np.random.default_rng(0)
        M = rng.uniform(0, 10, (5, 5))
        np.fill_diagonal(M, 0.0)
        result = normalize_matrix(M)
        mask = ~np.eye(5, dtype=bool)
        assert np.all(result[mask] >= 0.0)
        assert np.all(result[mask] <= 1.0)

    def test_uniform_matrix_zeros(self):
        M = np.full((3, 3), 5.0)
        result = normalize_matrix(M)
        np.testing.assert_array_equal(result, 0.0)

    def test_returns_float64(self):
        M = np.ones((3, 3))
        result = normalize_matrix(M)
        assert result.dtype == np.float64


# ─── aggregate_score_matrices ─────────────────────────────────────────────────

class TestAggregateScoreMatrices:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_score_matrices({})

    def test_size_mismatch_raises(self):
        m1 = np.ones((3, 3))
        m2 = np.ones((4, 4))
        with pytest.raises(ValueError):
            aggregate_score_matrices({"a": m1, "b": m2})

    def test_returns_nxn_float64(self):
        M = np.eye(4)
        result = aggregate_score_matrices({"x": M})
        assert result.shape == (4, 4)
        assert result.dtype == np.float64

    def test_diagonal_is_zero(self):
        M = np.ones((3, 3))
        result = aggregate_score_matrices({"x": M})
        np.testing.assert_array_equal(np.diag(result), [0.0, 0.0, 0.0])

    def test_equal_weights_by_default(self):
        M1 = np.array([[0.0, 1.0], [1.0, 0.0]])
        M2 = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = aggregate_score_matrices({"a": M1, "b": M2})
        # With equal weights and normalize=True: depends on normalization
        assert result.shape == (2, 2)

    def test_symmetric_output(self):
        rng = np.random.default_rng(42)
        M = rng.uniform(0, 1, (4, 4))
        result = aggregate_score_matrices({"x": M}, symmetric=True)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_single_matrix_normalized_in_01(self):
        rng = np.random.default_rng(7)
        M = rng.uniform(0, 10, (5, 5))
        np.fill_diagonal(M, 0.0)
        result = aggregate_score_matrices({"x": M}, normalize=True)
        mask = ~np.eye(5, dtype=bool)
        assert np.all(result[mask] >= 0.0 - 1e-9)
        assert np.all(result[mask] <= 1.0 + 1e-9)

    def test_custom_weights(self):
        M1 = np.array([[0.0, 0.8], [0.8, 0.0]])
        M2 = np.array([[0.0, 0.2], [0.2, 0.0]])
        # Give all weight to M1
        result = aggregate_score_matrices(
            {"a": M1, "b": M2}, weights={"a": 1.0, "b": 0.0}, normalize=False
        )
        assert result.shape == (2, 2)


# ─── rank_pairs ───────────────────────────────────────────────────────────────

class TestRankPairs:
    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            rank_pairs(np.ones((2, 3)))

    def test_returns_list_of_ranked_pairs(self):
        M = np.array([[0.0, 0.8, 0.3],
                      [0.8, 0.0, 0.5],
                      [0.3, 0.5, 0.0]])
        result = rank_pairs(M)
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, RankedPair)

    def test_sorted_descending(self):
        M = np.array([[0.0, 0.9, 0.1],
                      [0.9, 0.0, 0.5],
                      [0.1, 0.5, 0.0]])
        result = rank_pairs(M)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_sequential(self):
        M = np.array([[0.0, 0.8, 0.3],
                      [0.8, 0.0, 0.5],
                      [0.3, 0.5, 0.0]])
        result = rank_pairs(M)
        assert [r.rank for r in result] == list(range(len(result)))

    def test_upper_triangle_only(self):
        # Should return N*(N-1)/2 pairs for 4x4 matrix
        M = np.ones((4, 4))
        np.fill_diagonal(M, 0.0)
        result = rank_pairs(M)
        assert len(result) == 6  # C(4,2)

    def test_min_score_filter(self):
        M = np.array([[0.0, 0.9, 0.1],
                      [0.9, 0.0, 0.5],
                      [0.1, 0.5, 0.0]])
        result = rank_pairs(M, min_score=0.5)
        for r in result:
            assert r.score >= 0.5

    def test_empty_matrix_returns_empty(self):
        M = np.zeros((3, 3))
        result = rank_pairs(M, min_score=0.1)
        assert result == []

    def test_idx1_lt_idx2(self):
        M = np.array([[0.0, 0.8, 0.3],
                      [0.8, 0.0, 0.5],
                      [0.3, 0.5, 0.0]])
        result = rank_pairs(M)
        for r in result:
            assert r.idx1 < r.idx2


# ─── top_k_candidates ─────────────────────────────────────────────────────────

class TestTopKCandidates:
    def _make_pairs(self):
        M = np.array([[0.0, 0.9, 0.3, 0.5],
                      [0.9, 0.0, 0.7, 0.2],
                      [0.3, 0.7, 0.0, 0.8],
                      [0.5, 0.2, 0.8, 0.0]])
        return rank_pairs(M), 4

    def test_k_zero_raises(self):
        pairs, n = self._make_pairs()
        with pytest.raises(ValueError, match="k"):
            top_k_candidates(pairs, n, 0)

    def test_n_fragments_zero_raises(self):
        pairs, _ = self._make_pairs()
        with pytest.raises(ValueError, match="n_fragments"):
            top_k_candidates(pairs, 0, 1)

    def test_returns_dict_of_lists(self):
        pairs, n = self._make_pairs()
        result = top_k_candidates(pairs, n, k=2)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(v, list)

    def test_keys_are_0_to_n_minus_1(self):
        pairs, n = self._make_pairs()
        result = top_k_candidates(pairs, n, k=2)
        assert set(result.keys()) == set(range(n))

    def test_max_k_candidates_per_fragment(self):
        pairs, n = self._make_pairs()
        result = top_k_candidates(pairs, n, k=2)
        for fid in range(n):
            assert len(result[fid]) <= 2

    def test_empty_pairs_empty_lists(self):
        result = top_k_candidates([], 3, k=2)
        for v in result.values():
            assert v == []


# ─── global_rank ──────────────────────────────────────────────────────────────

class TestGlobalRank:
    def test_returns_list_of_ranked_pairs(self):
        M = np.array([[0.0, 0.8], [0.8, 0.0]])
        result = global_rank({"x": M})
        assert isinstance(result, list)
        for r in result:
            assert isinstance(r, RankedPair)

    def test_sorted_descending(self):
        rng = np.random.default_rng(5)
        M = rng.uniform(0, 1, (5, 5))
        result = global_rank({"x": M})
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_default_config(self):
        M = np.eye(3)
        result = global_rank({"boundary": M, "sift": M, "texture": M})
        assert isinstance(result, list)

    def test_custom_config(self):
        M = np.array([[0.0, 0.9, 0.5],
                      [0.9, 0.0, 0.3],
                      [0.5, 0.3, 0.0]])
        cfg = RankingConfig(min_score=0.5)
        result = global_rank({"x": M}, cfg=cfg)
        for r in result:
            assert r.score >= 0.5


# ─── score_vector ─────────────────────────────────────────────────────────────

class TestScoreVector:
    def test_n_fragments_zero_raises(self):
        with pytest.raises(ValueError, match="n_fragments"):
            score_vector([], 0)

    def test_returns_float64_array(self):
        sv = score_vector([], 3)
        assert sv.dtype == np.float64
        assert sv.shape == (3,)

    def test_empty_pairs_zero_scores(self):
        sv = score_vector([], 4)
        np.testing.assert_array_equal(sv, 0.0)

    def test_scores_averaged(self):
        pairs = [
            RankedPair(idx1=0, idx2=1, score=0.8, rank=0),
            RankedPair(idx1=0, idx2=2, score=0.4, rank=1),
        ]
        sv = score_vector(pairs, 3)
        assert sv[0] == pytest.approx(0.6)  # mean(0.8, 0.4)

    def test_unreferenced_fragment_stays_zero(self):
        pairs = [RankedPair(idx1=0, idx2=1, score=0.5, rank=0)]
        sv = score_vector(pairs, 4)
        assert sv[2] == pytest.approx(0.0)
        assert sv[3] == pytest.approx(0.0)


# ─── batch_global_rank ────────────────────────────────────────────────────────

class TestBatchGlobalRank:
    def test_empty_groups_returns_empty(self):
        result = batch_global_rank([])
        assert result == []

    def test_single_group(self):
        M = np.array([[0.0, 0.8], [0.8, 0.0]])
        result = batch_global_rank([{"x": M}])
        assert len(result) == 1

    def test_multiple_groups(self):
        M = np.array([[0.0, 0.9, 0.5],
                      [0.9, 0.0, 0.3],
                      [0.5, 0.3, 0.0]])
        groups = [{"x": M}, {"x": M}]
        result = batch_global_rank(groups)
        assert len(result) == 2
        for group_result in result:
            assert isinstance(group_result, list)
