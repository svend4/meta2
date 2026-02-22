"""Тесты для puzzle_reconstruction/scoring/match_scorer.py."""
import pytest
import numpy as np

from puzzle_reconstruction.scoring.match_scorer import (
    ScorerConfig,
    ChannelScore,
    MatchScore,
    score_channel,
    compute_match_score,
    aggregate_match_scores,
    build_score_table,
    filter_confident_pairs,
)


# ─── ScorerConfig ─────────────────────────────────────────────────────────────

class TestScorerConfig:
    def test_defaults(self):
        cfg = ScorerConfig()
        assert "geometry" in cfg.weights
        assert "texture" in cfg.weights
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.max_score == pytest.approx(1.0)
        assert cfg.normalize_input is True

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(weights={"geometry": -0.1})

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError, match="min_score"):
            ScorerConfig(min_score=-0.1)

    def test_max_score_above_1_raises(self):
        with pytest.raises(ValueError, match="max_score"):
            ScorerConfig(max_score=1.1)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=0.8, max_score=0.5)

    def test_total_weight(self):
        cfg = ScorerConfig(weights={"a": 0.5, "b": 0.3})
        assert cfg.total_weight == pytest.approx(0.8)

    def test_total_weight_empty(self):
        cfg = ScorerConfig(weights={})
        assert cfg.total_weight == pytest.approx(0.0)

    def test_normalized_weight(self):
        cfg = ScorerConfig(weights={"a": 0.5, "b": 0.5})
        assert cfg.normalized_weight("a") == pytest.approx(0.5)
        assert cfg.normalized_weight("b") == pytest.approx(0.5)

    def test_normalized_weight_missing_channel(self):
        cfg = ScorerConfig(weights={"a": 1.0})
        assert cfg.normalized_weight("b") == pytest.approx(0.0)

    def test_normalized_weight_zero_total(self):
        cfg = ScorerConfig(weights={})
        assert cfg.normalized_weight("a") == pytest.approx(0.0)

    def test_zero_weight_valid(self):
        cfg = ScorerConfig(weights={"a": 0.0, "b": 1.0})
        assert cfg.weights["a"] == 0.0

    def test_boundary_min_score_0(self):
        cfg = ScorerConfig(min_score=0.0)
        assert cfg.min_score == 0.0

    def test_boundary_max_score_1(self):
        cfg = ScorerConfig(max_score=1.0)
        assert cfg.max_score == 1.0


# ─── ChannelScore ─────────────────────────────────────────────────────────────

class TestChannelScore:
    def test_creation(self):
        cs = ChannelScore(channel="geometry", raw=0.8, norm=0.8, weight=0.35)
        assert cs.channel == "geometry"
        assert cs.raw == pytest.approx(0.8)
        assert cs.norm == pytest.approx(0.8)
        assert cs.weight == pytest.approx(0.35)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError, match="channel"):
            ChannelScore(channel="", raw=0.5, norm=0.5, weight=0.5)

    def test_norm_above_1_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="x", raw=1.2, norm=1.2, weight=0.5)

    def test_norm_below_0_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="x", raw=-0.1, norm=-0.1, weight=0.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            ChannelScore(channel="x", raw=0.5, norm=0.5, weight=-0.1)

    def test_contribution(self):
        cs = ChannelScore(channel="x", raw=0.8, norm=0.8, weight=0.5)
        assert cs.contribution == pytest.approx(0.4)

    def test_contribution_zero_norm(self):
        cs = ChannelScore(channel="x", raw=0.0, norm=0.0, weight=1.0)
        assert cs.contribution == pytest.approx(0.0)

    def test_norm_boundary_0(self):
        cs = ChannelScore(channel="x", raw=0.0, norm=0.0, weight=0.5)
        assert cs.norm == 0.0

    def test_norm_boundary_1(self):
        cs = ChannelScore(channel="x", raw=1.0, norm=1.0, weight=0.5)
        assert cs.norm == 1.0

    def test_zero_weight_valid(self):
        cs = ChannelScore(channel="x", raw=0.5, norm=0.5, weight=0.0)
        assert cs.weight == 0.0
        assert cs.contribution == pytest.approx(0.0)


# ─── MatchScore ───────────────────────────────────────────────────────────────

class TestMatchScore:
    def test_creation(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.75)
        assert ms.id_a == 0
        assert ms.id_b == 1
        assert ms.score == pytest.approx(0.75)
        assert ms.channels == {}
        assert ms.confident is False

    def test_score_above_1_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=1.1)

    def test_score_below_0_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=-0.1)

    def test_pair_key_ordered(self):
        ms = MatchScore(id_a=5, id_b=2, score=0.5)
        assert ms.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        ms = MatchScore(id_a=1, id_b=3, score=0.5)
        assert ms.pair_key == (1, 3)

    def test_dominant_channel_none_when_empty(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.5)
        assert ms.dominant_channel is None

    def test_dominant_channel_highest_contribution(self):
        ch = {
            "geometry": ChannelScore("geometry", 0.9, 0.9, 0.5),
            "texture":  ChannelScore("texture",  0.3, 0.3, 0.1),
        }
        ms = MatchScore(id_a=0, id_b=1, score=0.7, channels=ch)
        assert ms.dominant_channel == "geometry"

    def test_n_channels(self):
        ch = {
            "a": ChannelScore("a", 0.5, 0.5, 0.5),
            "b": ChannelScore("b", 0.5, 0.5, 0.5),
        }
        ms = MatchScore(id_a=0, id_b=1, score=0.5, channels=ch)
        assert ms.n_channels == 2

    def test_n_channels_empty(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.5)
        assert ms.n_channels == 0

    def test_confident_true_at_07(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.7, confident=True)
        assert ms.confident is True

    def test_boundary_score_0(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.0)
        assert ms.score == 0.0

    def test_boundary_score_1(self):
        ms = MatchScore(id_a=0, id_b=1, score=1.0)
        assert ms.score == 1.0


# ─── score_channel ────────────────────────────────────────────────────────────

class TestScoreChannel:
    def test_returns_channel_score(self):
        cs = score_channel("geometry", 0.7, 0.35)
        assert isinstance(cs, ChannelScore)

    def test_channel_stored(self):
        cs = score_channel("texture", 0.5, 0.25)
        assert cs.channel == "texture"

    def test_norm_clipped_above(self):
        cs = score_channel("x", 1.5, 0.5)
        assert cs.norm == pytest.approx(1.0)

    def test_norm_clipped_below(self):
        cs = score_channel("x", -0.5, 0.5)
        assert cs.norm == pytest.approx(0.0)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            score_channel("", 0.5, 0.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            score_channel("x", 0.5, -0.1)

    def test_raw_stored(self):
        cs = score_channel("x", 0.7, 0.3)
        assert cs.raw == pytest.approx(0.7)

    def test_weight_stored(self):
        cs = score_channel("x", 0.5, 0.4)
        assert cs.weight == pytest.approx(0.4)


# ─── compute_match_score ──────────────────────────────────────────────────────

class TestComputeMatchScore:
    def test_returns_match_score(self):
        ms = compute_match_score(0, 1, {"geometry": 0.8, "texture": 0.6})
        assert isinstance(ms, MatchScore)

    def test_ids_stored(self):
        ms = compute_match_score(3, 7, {"geometry": 0.5})
        assert ms.id_a == 3
        assert ms.id_b == 7

    def test_score_in_0_1(self):
        ms = compute_match_score(0, 1, {"geometry": 0.8, "texture": 0.5})
        assert 0.0 <= ms.score <= 1.0

    def test_empty_channel_values_raises(self):
        with pytest.raises(ValueError):
            compute_match_score(0, 1, {})

    def test_confident_true_for_high_score(self):
        ms = compute_match_score(0, 1, {"geometry": 1.0, "texture": 1.0,
                                         "frequency": 1.0, "color": 1.0})
        assert ms.confident is True

    def test_confident_false_for_low_score(self):
        ms = compute_match_score(0, 1, {"geometry": 0.0})
        assert ms.confident is False

    def test_channels_populated(self):
        ms = compute_match_score(0, 1, {"geometry": 0.8, "texture": 0.5})
        assert len(ms.channels) >= 1

    def test_custom_config(self):
        cfg = ScorerConfig(weights={"geometry": 1.0})
        ms = compute_match_score(0, 1, {"geometry": 0.8}, cfg=cfg)
        assert 0.0 <= ms.score <= 1.0

    def test_unknown_channel_gets_zero_weight(self):
        cfg = ScorerConfig(weights={"geometry": 1.0})
        ms = compute_match_score(0, 1, {"unknown_ch": 0.9, "geometry": 0.5}, cfg=cfg)
        assert 0.0 <= ms.score <= 1.0


# ─── aggregate_match_scores ───────────────────────────────────────────────────

class TestAggregateMatchScores:
    def test_empty_returns_none(self):
        result = aggregate_match_scores([])
        assert result is None

    def test_single_score(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.6)
        result = aggregate_match_scores([ms])
        assert isinstance(result, MatchScore)
        assert result.score == pytest.approx(0.6)

    def test_average_of_two(self):
        m1 = MatchScore(id_a=0, id_b=1, score=0.8)
        m2 = MatchScore(id_a=0, id_b=1, score=0.4)
        result = aggregate_match_scores([m1, m2])
        assert result.score == pytest.approx(0.6)

    def test_ids_from_first(self):
        m1 = MatchScore(id_a=2, id_b=5, score=0.5)
        m2 = MatchScore(id_a=2, id_b=5, score=0.7)
        result = aggregate_match_scores([m1, m2])
        assert result.id_a == 2
        assert result.id_b == 5

    def test_result_in_0_1(self):
        scores = [MatchScore(id_a=0, id_b=1, score=s)
                  for s in [0.3, 0.8, 0.6]]
        result = aggregate_match_scores(scores)
        assert 0.0 <= result.score <= 1.0

    def test_confident_high_avg(self):
        scores = [MatchScore(id_a=0, id_b=1, score=0.9),
                  MatchScore(id_a=0, id_b=1, score=0.8)]
        result = aggregate_match_scores(scores)
        assert result.confident is True


# ─── build_score_table ────────────────────────────────────────────────────────

class TestBuildScoreTable:
    def test_returns_dict(self):
        pairs = [(0, 1)]
        cvm = {(0, 1): {"geometry": 0.8}}
        result = build_score_table(pairs, cvm)
        assert isinstance(result, dict)

    def test_key_is_ordered_pair(self):
        pairs = [(1, 0)]
        cvm = {(1, 0): {"geometry": 0.8}}
        result = build_score_table(pairs, cvm)
        assert (0, 1) in result

    def test_missing_pair_gets_zero_score(self):
        pairs = [(0, 1)]
        cvm = {}  # no values for (0,1)
        result = build_score_table(pairs, cvm)
        key = (0, 1)
        assert result[key].score == pytest.approx(0.0)

    def test_multiple_pairs(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        cvm = {
            (0, 1): {"geometry": 0.8},
            (1, 2): {"geometry": 0.5},
            (0, 2): {"geometry": 0.3},
        }
        result = build_score_table(pairs, cvm)
        assert len(result) == 3

    def test_returns_match_score_values(self):
        pairs = [(0, 1)]
        cvm = {(0, 1): {"geometry": 0.9}}
        result = build_score_table(pairs, cvm)
        for v in result.values():
            assert isinstance(v, MatchScore)

    def test_reverse_pair_lookup(self):
        pairs = [(0, 1)]
        cvm = {(1, 0): {"geometry": 0.7}}  # stored as reverse
        result = build_score_table(pairs, cvm)
        assert (0, 1) in result


# ─── filter_confident_pairs ───────────────────────────────────────────────────

class TestFilterConfidentPairs:
    def _make_table(self):
        return {
            (0, 1): MatchScore(id_a=0, id_b=1, score=0.9),
            (1, 2): MatchScore(id_a=1, id_b=2, score=0.5),
            (0, 2): MatchScore(id_a=0, id_b=2, score=0.75),
        }

    def test_returns_list(self):
        result = filter_confident_pairs(self._make_table())
        assert isinstance(result, list)

    def test_threshold_default_07(self):
        result = filter_confident_pairs(self._make_table())
        table = self._make_table()
        for pair in result:
            assert table[pair].score >= 0.7

    def test_sorted_descending(self):
        result = filter_confident_pairs(self._make_table(), threshold=0.0)
        table = self._make_table()
        scores = [table[p].score for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs(self._make_table(), threshold=1.5)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs(self._make_table(), threshold=-0.1)

    def test_empty_table_returns_empty(self):
        result = filter_confident_pairs({})
        assert result == []

    def test_all_below_threshold_returns_empty(self):
        table = {(0, 1): MatchScore(id_a=0, id_b=1, score=0.3)}
        result = filter_confident_pairs(table, threshold=0.7)
        assert result == []

    def test_pairs_are_tuples(self):
        result = filter_confident_pairs(self._make_table(), threshold=0.0)
        for pair in result:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_boundary_threshold_0(self):
        table = {(0, 1): MatchScore(id_a=0, id_b=1, score=0.0)}
        result = filter_confident_pairs(table, threshold=0.0)
        assert len(result) == 1

    def test_boundary_threshold_1(self):
        table = {(0, 1): MatchScore(id_a=0, id_b=1, score=1.0)}
        result = filter_confident_pairs(table, threshold=1.0)
        assert len(result) == 1
