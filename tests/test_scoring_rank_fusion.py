"""Tests for puzzle_reconstruction/scoring/rank_fusion.py"""
import pytest
import numpy as np

from puzzle_reconstruction.scoring.rank_fusion import (
    normalize_scores,
    reciprocal_rank_fusion,
    borda_count,
    score_fusion,
    fuse_rankings,
)


# ─── normalize_scores ─────────────────────────────────────────────────────────

class TestNormalizeScores:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalize_scores([])

    def test_single_element(self):
        result = normalize_scores([0.7])
        assert result == [1.0]

    def test_all_same_returns_ones(self):
        result = normalize_scores([3.0, 3.0, 3.0])
        assert result == [1.0, 1.0, 1.0]

    def test_range_01(self):
        result = normalize_scores([1.0, 3.0, 5.0])
        assert min(result) == pytest.approx(0.0)
        assert max(result) == pytest.approx(1.0)

    def test_length_preserved(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_scores(scores)
        assert len(result) == len(scores)

    def test_monotone_preserved(self):
        scores = [1.0, 2.0, 3.0]
        result = normalize_scores(scores)
        assert result[0] <= result[1] <= result[2]

    def test_min_becomes_zero(self):
        result = normalize_scores([2.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0)

    def test_max_becomes_one(self):
        result = normalize_scores([2.0, 5.0, 10.0])
        assert result[-1] == pytest.approx(1.0)


# ─── reciprocal_rank_fusion ───────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[1, 2, 3]], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[1, 2, 3]], k=-5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([])

    def test_single_list(self):
        result = reciprocal_rank_fusion([[3, 1, 2]])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sorted_desc(self):
        result = reciprocal_rank_fusion([[0, 1, 2]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_ranked_id_wins(self):
        """Top-ranked item in all lists should win."""
        result = reciprocal_rank_fusion([[7, 1, 2], [7, 3, 4]])
        top_id, _ = result[0]
        assert top_id == 7

    def test_two_lists_merge(self):
        result = reciprocal_rank_fusion([[0, 1], [1, 0]])
        ids = {r[0] for r in result}
        assert ids == {0, 1}

    def test_scores_are_float(self):
        result = reciprocal_rank_fusion([[1, 2, 3]])
        assert all(isinstance(s, float) for _, s in result)

    def test_ids_are_int(self):
        result = reciprocal_rank_fusion([[10, 20, 30]])
        assert all(isinstance(i, int) for i, _ in result)

    def test_scores_positive(self):
        result = reciprocal_rank_fusion([[1, 2, 3]])
        assert all(s > 0.0 for _, s in result)

    def test_k_affects_score(self):
        r_small = reciprocal_rank_fusion([[1, 2]], k=1)
        r_large = reciprocal_rank_fusion([[1, 2]], k=1000)
        s_small = dict(r_small)
        s_large = dict(r_large)
        # Higher k → lower and more equal scores
        assert s_small[1] > s_large[1]

    def test_single_item_list(self):
        result = reciprocal_rank_fusion([[42]])
        assert len(result) == 1
        assert result[0][0] == 42


# ─── borda_count ──────────────────────────────────────────────────────────────

class TestBordaCount:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            borda_count([])

    def test_single_list_sorted(self):
        result = borda_count([[10, 20, 30]])
        ids = [i for i, _ in result]
        assert ids[0] == 10  # highest score (N-1 points)
        assert ids[-1] == 30  # lowest score (0 points)

    def test_sorted_desc(self):
        result = borda_count([[0, 1, 2], [2, 0, 1]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_two_lists_all_ids_present(self):
        result = borda_count([[0, 1], [1, 0]])
        ids = {i for i, _ in result}
        assert ids == {0, 1}

    def test_scores_nonneg(self):
        result = borda_count([[1, 2, 3]])
        assert all(s >= 0.0 for _, s in result)

    def test_last_item_zero_score(self):
        """Last item in a single list gets 0 Borda points."""
        result = borda_count([[5, 6, 7]])
        id_score = dict(result)
        assert id_score[7] == pytest.approx(0.0)

    def test_single_item_list(self):
        result = borda_count([[99]])
        assert len(result) == 1
        assert result[0][0] == 99

    def test_consensus_winner(self):
        """Item ranked first in all lists should win."""
        result = borda_count([[5, 1, 2], [5, 3, 4], [5, 0, 6]])
        assert result[0][0] == 5


# ─── score_fusion ─────────────────────────────────────────────────────────────

class TestScoreFusion:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            score_fusion([])

    def test_weights_length_mismatch_raises(self):
        sl = [[(0, 0.8), (1, 0.6)]]
        with pytest.raises(ValueError):
            score_fusion(sl, weights=[0.5, 0.5])

    def test_single_list_sorted_desc(self):
        sl = [[(0, 0.3), (1, 0.9), (2, 0.6)]]
        result = score_fusion(sl)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_two_lists_merged(self):
        sl = [[(0, 0.8), (1, 0.6)], [(0, 0.5), (1, 0.9)]]
        result = score_fusion(sl)
        ids = {i for i, _ in result}
        assert ids == {0, 1}

    def test_equal_weights_default(self):
        sl = [[(0, 1.0)], [(0, 0.0)]]
        result = score_fusion(sl)
        assert len(result) == 1

    def test_custom_weights(self):
        sl = [[(0, 1.0), (1, 0.0)], [(0, 0.0), (1, 1.0)]]
        # Weight only second list
        result = score_fusion(sl, weights=[0.0, 1.0])
        id_score = dict(result)
        assert id_score[1] > id_score[0]

    def test_normalize_false(self):
        sl = [[(0, 100.0), (1, 50.0)]]
        result = score_fusion(sl, normalize=False)
        assert len(result) == 2

    def test_empty_sublists_skipped(self):
        sl = [[], [(0, 0.8)]]
        result = score_fusion(sl)
        assert len(result) == 1

    def test_scores_are_float(self):
        sl = [[(1, 0.5), (2, 0.3)]]
        result = score_fusion(sl)
        assert all(isinstance(s, float) for _, s in result)


# ─── fuse_rankings ────────────────────────────────────────────────────────────

class TestFuseRankings:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            fuse_rankings([[1, 2]], method="unknown")

    def test_method_rrf(self):
        result = fuse_rankings([[3, 1, 2]], method="rrf")
        assert isinstance(result, list)
        assert len(result) == 3

    def test_method_borda(self):
        result = fuse_rankings([[3, 1, 2]], method="borda")
        assert isinstance(result, list)
        assert len(result) == 3

    def test_rrf_default_method(self):
        result = fuse_rankings([[1, 2, 3]])
        assert isinstance(result, list)

    def test_rrf_and_borda_same_winner(self):
        """For a single consensus list, both methods rank the first item first."""
        lists = [[10, 20, 30], [10, 20, 30]]
        rrf_result = fuse_rankings(lists, method="rrf")
        borda_result = fuse_rankings(lists, method="borda")
        assert rrf_result[0][0] == borda_result[0][0] == 10

    def test_k_param_passed(self):
        result = fuse_rankings([[1, 2, 3]], method="rrf", k=1)
        assert isinstance(result, list)

    def test_sorted_desc(self):
        result = fuse_rankings([[0, 1, 2, 3]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
