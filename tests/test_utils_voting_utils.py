"""Тесты для puzzle_reconstruction/utils/voting_utils.py."""
import pytest

from puzzle_reconstruction.utils.voting_utils import (
    VoteConfig,
    cast_pair_votes,
    aggregate_pair_votes,
    cast_position_votes,
    majority_vote,
    weighted_vote,
    rank_fusion,
    batch_vote,
)


# ─── VoteConfig ───────────────────────────────────────────────────────────────

class TestVoteConfig:
    def test_defaults(self):
        cfg = VoteConfig()
        assert cfg.min_votes == 1
        assert cfg.rrf_k == pytest.approx(60.0)
        assert cfg.normalize is True
        assert cfg.weights is None

    def test_min_votes_zero_raises(self):
        with pytest.raises(ValueError, match="min_votes"):
            VoteConfig(min_votes=0)

    def test_min_votes_negative_raises(self):
        with pytest.raises(ValueError, match="min_votes"):
            VoteConfig(min_votes=-1)

    def test_rrf_k_zero_raises(self):
        with pytest.raises(ValueError, match="rrf_k"):
            VoteConfig(rrf_k=0.0)

    def test_rrf_k_negative_raises(self):
        with pytest.raises(ValueError, match="rrf_k"):
            VoteConfig(rrf_k=-1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            VoteConfig(weights=[1.0, -0.1])

    def test_zero_weight_valid(self):
        cfg = VoteConfig(weights=[0.0, 1.0])
        assert cfg.weights[0] == pytest.approx(0.0)

    def test_positive_weights_stored(self):
        cfg = VoteConfig(weights=[2.0, 3.0])
        assert cfg.weights == [2.0, 3.0]

    def test_min_votes_2_valid(self):
        cfg = VoteConfig(min_votes=2)
        assert cfg.min_votes == 2


# ─── cast_pair_votes ──────────────────────────────────────────────────────────

class TestCastPairVotes:
    def test_empty_lists_returns_empty(self):
        result = cast_pair_votes([])
        assert result == {}

    def test_single_pair(self):
        result = cast_pair_votes([[(0, 1)]])
        assert (0, 1) in result
        assert result[(0, 1)] == pytest.approx(1.0)

    def test_canonical_form(self):
        result = cast_pair_votes([[(3, 1)]])
        # Should be stored as (1, 3), not (3, 1)
        assert (1, 3) in result
        assert (3, 1) not in result

    def test_multiple_sources_sum_votes(self):
        result = cast_pair_votes([[(0, 1)], [(0, 1)]])
        assert result[(0, 1)] == pytest.approx(2.0)

    def test_weights_applied(self):
        result = cast_pair_votes([[(0, 1)], [(0, 1)]], weights=[3.0, 1.0])
        assert result[(0, 1)] == pytest.approx(4.0)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_pair_votes([[(0, 1)]], weights=[1.0, 2.0])

    def test_multiple_pairs_accumulated(self):
        result = cast_pair_votes([[(0, 1), (2, 3)], [(0, 1)]])
        assert result[(0, 1)] == pytest.approx(2.0)
        assert result[(2, 3)] == pytest.approx(1.0)

    def test_returns_dict(self):
        result = cast_pair_votes([[(0, 1)]])
        assert isinstance(result, dict)

    def test_none_weights_equal_votes(self):
        result = cast_pair_votes([[(0, 1)], [(0, 1)]], weights=None)
        assert result[(0, 1)] == pytest.approx(2.0)


# ─── aggregate_pair_votes ─────────────────────────────────────────────────────

class TestAggregatePairVotes:
    def test_empty_votes_returns_empty(self):
        result = aggregate_pair_votes({})
        assert result == []

    def test_returns_list(self):
        result = aggregate_pair_votes({(0, 1): 3.0})
        assert isinstance(result, list)

    def test_sorted_descending(self):
        votes = {(0, 1): 5.0, (1, 2): 2.0, (2, 3): 8.0}
        result = aggregate_pair_votes(votes)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_min_votes_filter(self):
        votes = {(0, 1): 5.0, (1, 2): 1.0}
        cfg = VoteConfig(min_votes=3, normalize=False)
        result = aggregate_pair_votes(votes, cfg=cfg)
        pairs = [p for p, _ in result]
        assert (0, 1) in pairs
        assert (1, 2) not in pairs

    def test_normalize_max_is_1(self):
        votes = {(0, 1): 6.0, (1, 2): 3.0}
        cfg = VoteConfig(normalize=True)
        result = aggregate_pair_votes(votes, cfg=cfg)
        max_score = max(s for _, s in result)
        assert max_score == pytest.approx(1.0)

    def test_normalize_false_preserves_values(self):
        votes = {(0, 1): 6.0}
        cfg = VoteConfig(normalize=False)
        result = aggregate_pair_votes(votes, cfg=cfg)
        assert result[0][1] == pytest.approx(6.0)

    def test_single_item(self):
        result = aggregate_pair_votes({(3, 7): 1.0})
        assert len(result) == 1
        assert result[0][0] == (3, 7)

    def test_none_cfg_uses_defaults(self):
        result = aggregate_pair_votes({(0, 1): 2.0}, cfg=None)
        assert isinstance(result, list)


# ─── cast_position_votes ──────────────────────────────────────────────────────

class TestCastPositionVotes:
    def test_empty_list_returns_empty(self):
        result = cast_position_votes([])
        assert result == {}

    def test_returns_dict(self):
        result = cast_position_votes([{0: 0.5}])
        assert isinstance(result, dict)

    def test_single_source(self):
        result = cast_position_votes([{0: 0.8, 1: 0.3}])
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.3)

    def test_multiple_sources_summed(self):
        result = cast_position_votes([{0: 0.5}, {0: 0.3}])
        assert result[0] == pytest.approx(0.8)

    def test_weights_applied(self):
        result = cast_position_votes([{0: 1.0}, {0: 1.0}], weights=[2.0, 0.5])
        assert result[0] == pytest.approx(2.5)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_position_votes([{0: 1.0}], weights=[1.0, 2.0])

    def test_fragments_union(self):
        result = cast_position_votes([{0: 0.5}, {1: 0.7}])
        assert 0 in result
        assert 1 in result


# ─── majority_vote ────────────────────────────────────────────────────────────

class TestMajorityVote:
    def test_empty_returns_none(self):
        assert majority_vote([]) is None

    def test_single_value(self):
        assert majority_vote(["a"]) == "a"

    def test_most_common_wins(self):
        result = majority_vote(["a", "b", "a", "c", "a"])
        assert result == "a"

    def test_integers(self):
        result = majority_vote([1, 2, 2, 3, 2])
        assert result == 2

    def test_uniform_returns_value(self):
        result = majority_vote([5, 5, 5])
        assert result == 5

    def test_two_equal_returns_one_of_them(self):
        result = majority_vote(["x", "y", "x", "y"])
        assert result in ("x", "y")

    def test_none_values(self):
        result = majority_vote([None, None, 1])
        assert result is None


# ─── weighted_vote ────────────────────────────────────────────────────────────

class TestWeightedVote:
    def test_empty_returns_zero(self):
        assert weighted_vote([]) == pytest.approx(0.0)

    def test_equal_weights_is_mean(self):
        result = weighted_vote([1.0, 2.0, 3.0])
        assert result == pytest.approx(2.0)

    def test_custom_weights(self):
        result = weighted_vote([0.0, 1.0], weights=[1.0, 3.0])
        assert result == pytest.approx(0.75)

    def test_single_value(self):
        assert weighted_vote([7.0]) == pytest.approx(7.0)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0, 2.0], weights=[1.0])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0], weights=[-0.1])

    def test_zero_total_weight_returns_zero(self):
        result = weighted_vote([1.0, 2.0], weights=[0.0, 0.0])
        assert result == pytest.approx(0.0)

    def test_none_weights_equal(self):
        result = weighted_vote([2.0, 4.0], weights=None)
        assert result == pytest.approx(3.0)


# ─── rank_fusion ──────────────────────────────────────────────────────────────

class TestRankFusion:
    def test_empty_returns_empty(self):
        result = rank_fusion([])
        assert result == []

    def test_single_list(self):
        result = rank_fusion([["a", "b", "c"]])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sorted_descending(self):
        result = rank_fusion([["a", "b", "c"]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_first_ranked_highest_in_single_list(self):
        result = rank_fusion([["best", "middle", "worst"]])
        # "best" should have highest RRF score
        items = [item for item, _ in result]
        assert items[0] == "best"

    def test_multiple_lists_boosted(self):
        # "a" appears first in both lists → high score
        result = rank_fusion([["a", "b"], ["a", "c"]])
        items = [item for item, _ in result]
        assert items[0] == "a"

    def test_normalize_max_is_1(self):
        cfg = VoteConfig(normalize=True)
        result = rank_fusion([["a", "b", "c"]], cfg=cfg)
        max_s = max(s for _, s in result)
        assert max_s == pytest.approx(1.0)

    def test_normalize_false_preserves_rrf_scores(self):
        cfg = VoteConfig(normalize=False)
        result = rank_fusion([["a"]], cfg=cfg)
        # RRF score = 1/(60+1) for single item at rank 1
        assert result[0][1] == pytest.approx(1.0 / 61.0, rel=1e-4)

    def test_custom_k(self):
        cfg = VoteConfig(rrf_k=10.0, normalize=False)
        result = rank_fusion([["a"]], cfg=cfg)
        assert result[0][1] == pytest.approx(1.0 / 11.0, rel=1e-4)


# ─── batch_vote ───────────────────────────────────────────────────────────────

class TestBatchVote:
    def test_empty_batch_returns_empty(self):
        result = batch_vote([])
        assert result == []

    def test_length_matches_batch(self):
        batch = [
            [[(0, 1), (1, 2)], [(0, 1)]],
            [[(2, 3)]],
        ]
        result = batch_vote(batch)
        assert len(result) == 2

    def test_each_element_is_list(self):
        batch = [[[(0, 1)]]]
        result = batch_vote(batch)
        assert isinstance(result[0], list)

    def test_cfg_min_votes_applied(self):
        # pair (0,1) gets 1 vote, (1,2) gets 2 votes; min_votes=2 → only (1,2)
        batch = [[[(0, 1)], [(1, 2)], [(1, 2)]]]
        cfg = VoteConfig(min_votes=2, normalize=False)
        result = batch_vote(batch, cfg=cfg)
        pairs = [p for p, _ in result[0]]
        assert (1, 2) in pairs
        assert (0, 1) not in pairs

    def test_results_are_sorted_descending(self):
        batch = [[[(0, 1)], [(0, 1)], [(1, 2)]]]
        result = batch_vote(batch)
        scores = [s for _, s in result[0]]
        assert scores == sorted(scores, reverse=True)
