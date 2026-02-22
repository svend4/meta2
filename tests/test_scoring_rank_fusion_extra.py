"""Additional tests for puzzle_reconstruction/scoring/rank_fusion.py."""
import pytest
from puzzle_reconstruction.scoring.rank_fusion import (
    normalize_scores,
    reciprocal_rank_fusion,
    borda_count,
    score_fusion,
    fuse_rankings,
)


# ─── TestNormalizeScoresExtra ─────────────────────────────────────────────────

class TestNormalizeScoresExtra:
    def test_negative_inputs_range_01(self):
        result = normalize_scores([-5.0, 0.0, 5.0])
        assert min(result) == pytest.approx(0.0)
        assert max(result) == pytest.approx(1.0)

    def test_two_elements(self):
        result = normalize_scores([0.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_large_values(self):
        result = normalize_scores([1e6, 1e7, 1e8])
        assert min(result) == pytest.approx(0.0)
        assert max(result) == pytest.approx(1.0)

    def test_returns_list(self):
        result = normalize_scores([1.0, 2.0])
        assert isinstance(result, list)

    def test_all_negative_same_returns_ones(self):
        result = normalize_scores([-3.0, -3.0])
        assert result == [1.0, 1.0]

    def test_order_preserved(self):
        scores = [5.0, 1.0, 3.0]
        result = normalize_scores(scores)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)

    def test_floats_in_result(self):
        result = normalize_scores([1, 2, 3])
        assert all(isinstance(v, float) for v in result)


# ─── TestReciprocalRankFusionExtra ────────────────────────────────────────────

class TestReciprocalRankFusionExtra:
    def test_five_lists_same_winner(self):
        lists = [[9, 1, 2, 3], [9, 4, 5, 6], [9, 7, 8, 0],
                 [9, 2, 3, 4], [9, 5, 6, 7]]
        result = reciprocal_rank_fusion(lists)
        assert result[0][0] == 9

    def test_result_length_union(self):
        result = reciprocal_rank_fusion([[0, 1, 2], [3, 4, 5]])
        assert len(result) == 6

    def test_overlapping_items(self):
        result = reciprocal_rank_fusion([[1, 2], [2, 3]])
        ids = {i for i, _ in result}
        assert ids == {1, 2, 3}

    def test_k60_default_effect(self):
        """Default k=60: scores are positive and < 1."""
        result = reciprocal_rank_fusion([[1, 2, 3]])
        assert all(0.0 < s <= 1.0 for _, s in result)

    def test_duplicate_id_in_same_list(self):
        """If IDs are unique within a list, no duplicate scores."""
        result = reciprocal_rank_fusion([[1, 2, 3, 4]])
        ids = [i for i, _ in result]
        assert len(ids) == len(set(ids))

    def test_single_item_score_positive(self):
        result = reciprocal_rank_fusion([[7]])
        assert result[0][1] > 0.0

    def test_three_lists_all_ids_present(self):
        result = reciprocal_rank_fusion([[0, 1], [2, 3], [4, 5]])
        ids = {i for i, _ in result}
        assert ids == {0, 1, 2, 3, 4, 5}


# ─── TestBordaCountExtra ──────────────────────────────────────────────────────

class TestBordaCountExtra:
    def test_three_lists_winner(self):
        result = borda_count([[3, 1, 2], [3, 2, 1], [3, 0, 4]])
        assert result[0][0] == 3

    def test_consensus_loser(self):
        """Item last in all lists should have lowest score."""
        result = borda_count([[0, 1, 9], [0, 1, 9], [0, 1, 9]])
        id_score = dict(result)
        assert id_score[9] < id_score[0]

    def test_all_items_present(self):
        result = borda_count([[0, 1, 2], [0, 2, 1]])
        ids = {i for i, _ in result}
        assert ids == {0, 1, 2}

    def test_single_list_first_max(self):
        result = borda_count([[10, 20, 30, 40]])
        assert result[0][0] == 10

    def test_scores_integers_or_float(self):
        result = borda_count([[5, 6, 7]])
        assert all(isinstance(s, (int, float)) for _, s in result)

    def test_two_equal_lists(self):
        result = borda_count([[0, 1, 2], [0, 1, 2]])
        assert result[0][0] == 0
        assert result[-1][0] == 2

    def test_five_lists_consensus(self):
        item = 42
        lists = [[item, i] for i in range(5)]
        result = borda_count(lists)
        assert result[0][0] == item


# ─── TestScoreFusionExtra ─────────────────────────────────────────────────────

class TestScoreFusionExtra:
    def test_three_lists_all_ids(self):
        sl = [[(0, 0.9), (1, 0.5)],
              [(2, 0.8), (0, 0.3)],
              [(1, 0.7), (3, 0.4)]]
        result = score_fusion(sl)
        ids = {i for i, _ in result}
        assert ids == {0, 1, 2, 3}

    def test_normalize_true_range(self):
        sl = [[(0, 100.0), (1, 50.0)]]
        result = score_fusion(sl, normalize=True)
        scores = [s for _, s in result]
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) == pytest.approx(0.0)

    def test_partial_overlap(self):
        sl = [[(0, 0.8), (1, 0.5)], [(1, 0.9), (2, 0.4)]]
        result = score_fusion(sl)
        ids = {i for i, _ in result}
        assert ids == {0, 1, 2}

    def test_custom_weights_affect_order(self):
        """Weight only one list: ranking should match that list's ordering."""
        sl = [[(0, 1.0), (1, 0.0)], [(0, 0.0), (1, 1.0)]]
        r_first = score_fusion(sl, weights=[1.0, 0.0])
        r_second = score_fusion(sl, weights=[0.0, 1.0])
        id_first = [i for i, _ in r_first]
        id_second = [i for i, _ in r_second]
        # First list favours 0; second list favours 1
        assert id_first[0] == 0
        assert id_second[0] == 1

    def test_all_empty_sublists_returns_empty(self):
        result = score_fusion([[], []])
        assert result == []

    def test_length_correct(self):
        sl = [[(i, float(i) / 10) for i in range(5)]]
        result = score_fusion(sl)
        assert len(result) == 5

    def test_sorted_descending(self):
        sl = [[(0, 0.1), (1, 0.8), (2, 0.4)]]
        result = score_fusion(sl)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


# ─── TestFuseRankingsExtra ────────────────────────────────────────────────────

class TestFuseRankingsExtra:
    def test_rrf_length_matches_items(self):
        lists = [[0, 1, 2, 3, 4]]
        result = fuse_rankings(lists, method="rrf")
        assert len(result) == 5

    def test_borda_length_matches_items(self):
        lists = [[0, 1, 2, 3, 4]]
        result = fuse_rankings(lists, method="borda")
        assert len(result) == 5

    def test_rrf_returns_sorted(self):
        result = fuse_rankings([[1, 2, 3, 4]], method="rrf")
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_borda_returns_sorted(self):
        result = fuse_rankings([[1, 2, 3, 4]], method="borda")
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            fuse_rankings([])

    def test_k_param_rrf(self):
        for k in [1, 10, 100]:
            result = fuse_rankings([[0, 1, 2]], method="rrf", k=k)
            assert len(result) == 3

    def test_two_methods_same_ids(self):
        lists = [[3, 1, 2], [3, 2, 1]]
        r_rrf = fuse_rankings(lists, method="rrf")
        r_borda = fuse_rankings(lists, method="borda")
        ids_rrf = {i for i, _ in r_rrf}
        ids_borda = {i for i, _ in r_borda}
        assert ids_rrf == ids_borda

    def test_five_lists_rrf(self):
        lists = [[i, i + 1, i + 2] for i in range(5)]
        result = fuse_rankings(lists, method="rrf")
        assert len(result) >= 3
