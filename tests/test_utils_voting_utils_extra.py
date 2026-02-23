"""Extra tests for puzzle_reconstruction.utils.voting_utils."""
import pytest

from puzzle_reconstruction.utils.voting_utils import (
    VoteConfig,
    aggregate_pair_votes,
    batch_vote,
    cast_pair_votes,
    cast_position_votes,
    majority_vote,
    rank_fusion,
    weighted_vote,
)


# ─── TestVoteConfigExtra ────────────────────────────────────────────────────

class TestVoteConfigExtra:
    def test_default_min_votes(self):
        assert VoteConfig().min_votes == 1

    def test_default_rrf_k(self):
        assert VoteConfig().rrf_k == pytest.approx(60.0)

    def test_default_normalize(self):
        assert VoteConfig().normalize is True

    def test_default_weights(self):
        assert VoteConfig().weights is None

    def test_min_votes_large(self):
        cfg = VoteConfig(min_votes=100)
        assert cfg.min_votes == 100

    def test_rrf_k_custom(self):
        cfg = VoteConfig(rrf_k=10.0)
        assert cfg.rrf_k == pytest.approx(10.0)

    def test_normalize_false(self):
        cfg = VoteConfig(normalize=False)
        assert cfg.normalize is False

    def test_weights_stored(self):
        cfg = VoteConfig(weights=[1.0, 2.0, 3.0])
        assert cfg.weights == [1.0, 2.0, 3.0]

    def test_min_votes_zero_raises(self):
        with pytest.raises(ValueError):
            VoteConfig(min_votes=0)

    def test_rrf_k_zero_raises(self):
        with pytest.raises(ValueError):
            VoteConfig(rrf_k=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            VoteConfig(weights=[1.0, -0.5])


# ─── TestCastPairVotesExtra ─────────────────────────────────────────────────

class TestCastPairVotesExtra:
    def test_empty(self):
        assert cast_pair_votes([]) == {}

    def test_single_source_single_pair(self):
        result = cast_pair_votes([[(0, 1)]])
        assert (0, 1) in result

    def test_canonical_order(self):
        result = cast_pair_votes([[(5, 2)]])
        assert (2, 5) in result
        assert (5, 2) not in result

    def test_two_sources_sum(self):
        result = cast_pair_votes([[(0, 1)], [(0, 1)]])
        assert result[(0, 1)] == pytest.approx(2.0)

    def test_weights(self):
        result = cast_pair_votes([[(0, 1)], [(0, 1)]], weights=[2.0, 3.0])
        assert result[(0, 1)] == pytest.approx(5.0)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_pair_votes([[(0, 1)]], weights=[1.0, 2.0])

    def test_multiple_pairs(self):
        result = cast_pair_votes([[(0, 1), (2, 3)]])
        assert (0, 1) in result
        assert (2, 3) in result

    def test_returns_dict(self):
        assert isinstance(cast_pair_votes([[(0, 1)]]), dict)


# ─── TestAggregatePairVotesExtra ────────────────────────────────────────────

class TestAggregatePairVotesExtra:
    def test_empty(self):
        assert aggregate_pair_votes({}) == []

    def test_returns_list(self):
        assert isinstance(aggregate_pair_votes({(0, 1): 1.0}), list)

    def test_sorted_descending(self):
        votes = {(0, 1): 2.0, (1, 2): 5.0, (2, 3): 3.0}
        result = aggregate_pair_votes(votes)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_min_votes_filter(self):
        votes = {(0, 1): 5.0, (1, 2): 0.5}
        cfg = VoteConfig(min_votes=3, normalize=False)
        result = aggregate_pair_votes(votes, cfg=cfg)
        pairs = [p for p, _ in result]
        assert (0, 1) in pairs
        assert (1, 2) not in pairs

    def test_normalize_max_one(self):
        votes = {(0, 1): 4.0, (1, 2): 2.0}
        cfg = VoteConfig(normalize=True)
        result = aggregate_pair_votes(votes, cfg=cfg)
        assert max(s for _, s in result) == pytest.approx(1.0)

    def test_normalize_false_preserves(self):
        votes = {(0, 1): 7.0}
        cfg = VoteConfig(normalize=False)
        result = aggregate_pair_votes(votes, cfg=cfg)
        assert result[0][1] == pytest.approx(7.0)

    def test_single_item(self):
        result = aggregate_pair_votes({(0, 1): 3.0})
        assert len(result) == 1


# ─── TestCastPositionVotesExtra ─────────────────────────────────────────────

class TestCastPositionVotesExtra:
    def test_empty(self):
        assert cast_position_votes([]) == {}

    def test_returns_dict(self):
        assert isinstance(cast_position_votes([{0: 1.0}]), dict)

    def test_single_source(self):
        result = cast_position_votes([{0: 0.5, 1: 0.3}])
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.3)

    def test_two_sources_sum(self):
        result = cast_position_votes([{0: 0.5}, {0: 0.3}])
        assert result[0] == pytest.approx(0.8)

    def test_weights(self):
        result = cast_position_votes([{0: 1.0}, {0: 1.0}], weights=[2.0, 0.5])
        assert result[0] == pytest.approx(2.5)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            cast_position_votes([{0: 1.0}], weights=[1.0, 2.0])

    def test_union_of_keys(self):
        result = cast_position_votes([{0: 1.0}, {1: 2.0}])
        assert 0 in result and 1 in result


# ─── TestMajorityVoteExtra ──────────────────────────────────────────────────

class TestMajorityVoteExtra:
    def test_empty_none(self):
        assert majority_vote([]) is None

    def test_single(self):
        assert majority_vote(["x"]) == "x"

    def test_clear_winner(self):
        assert majority_vote(["a", "a", "b"]) == "a"

    def test_integers(self):
        assert majority_vote([3, 3, 1, 2]) == 3

    def test_all_same(self):
        assert majority_vote([7, 7, 7]) == 7

    def test_tie_returns_one(self):
        result = majority_vote(["x", "y"])
        assert result in ("x", "y")

    def test_none_values(self):
        assert majority_vote([None, None, 1]) is None


# ─── TestWeightedVoteExtra ──────────────────────────────────────────────────

class TestWeightedVoteExtra:
    def test_empty_zero(self):
        assert weighted_vote([]) == pytest.approx(0.0)

    def test_equal_weights_mean(self):
        assert weighted_vote([2.0, 4.0]) == pytest.approx(3.0)

    def test_custom_weights(self):
        assert weighted_vote([0.0, 1.0], weights=[1.0, 3.0]) == pytest.approx(0.75)

    def test_single(self):
        assert weighted_vote([5.0]) == pytest.approx(5.0)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0, 2.0], weights=[1.0])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weighted_vote([1.0], weights=[-1.0])

    def test_zero_total_weight(self):
        assert weighted_vote([1.0, 2.0], weights=[0.0, 0.0]) == pytest.approx(0.0)

    def test_none_weights_equal(self):
        assert weighted_vote([3.0, 5.0], weights=None) == pytest.approx(4.0)


# ─── TestRankFusionExtra ────────────────────────────────────────────────────

class TestRankFusionExtra:
    def test_empty(self):
        assert rank_fusion([]) == []

    def test_single_list(self):
        result = rank_fusion([["a", "b"]])
        assert len(result) == 2

    def test_sorted_descending(self):
        result = rank_fusion([["a", "b", "c"]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_first_ranked_highest(self):
        result = rank_fusion([["best", "worst"]])
        assert result[0][0] == "best"

    def test_multiple_lists_boost(self):
        result = rank_fusion([["a", "b"], ["a", "c"]])
        assert result[0][0] == "a"

    def test_normalize_max_one(self):
        cfg = VoteConfig(normalize=True)
        result = rank_fusion([["a", "b"]], cfg=cfg)
        assert max(s for _, s in result) == pytest.approx(1.0)

    def test_custom_k(self):
        cfg = VoteConfig(rrf_k=10.0, normalize=False)
        result = rank_fusion([["a"]], cfg=cfg)
        assert result[0][1] == pytest.approx(1.0 / 11.0, rel=1e-4)


# ─── TestBatchVoteExtra ─────────────────────────────────────────────────────

class TestBatchVoteExtra:
    def test_empty(self):
        assert batch_vote([]) == []

    def test_length(self):
        batch = [[[(0, 1)]], [[(2, 3)]]]
        assert len(batch_vote(batch)) == 2

    def test_each_is_list(self):
        batch = [[[(0, 1)]]]
        result = batch_vote(batch)
        assert isinstance(result[0], list)

    def test_min_votes_applied(self):
        batch = [[[(0, 1)], [(1, 2)], [(1, 2)]]]
        cfg = VoteConfig(min_votes=2, normalize=False)
        result = batch_vote(batch, cfg=cfg)
        pairs = [p for p, _ in result[0]]
        assert (1, 2) in pairs
        assert (0, 1) not in pairs

    def test_sorted_descending(self):
        batch = [[[(0, 1)], [(0, 1)], [(1, 2)]]]
        result = batch_vote(batch)
        scores = [s for _, s in result[0]]
        assert scores == sorted(scores, reverse=True)
