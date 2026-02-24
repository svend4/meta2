"""Extra tests for puzzle_reconstruction/scoring/match_scorer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.scoring.match_scorer import (
    ChannelScore,
    MatchScore,
    ScorerConfig,
    aggregate_match_scores,
    build_score_table,
    compute_match_score,
    filter_confident_pairs,
    score_channel,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cfg(**kw) -> ScorerConfig:
    return ScorerConfig(**kw)


def _channel_vals(g: float = 0.8, t: float = 0.7,
                  f: float = 0.6, c: float = 0.5) -> dict:
    return {"geometry": g, "texture": t, "frequency": f, "color": c}


def _match(id_a: int = 0, id_b: int = 1, score: float = 0.8) -> MatchScore:
    return MatchScore(id_a=id_a, id_b=id_b, score=score)


# ─── ScorerConfig (extra) ─────────────────────────────────────────────────────

class TestScorerConfigExtra:
    def test_default_weights_keys(self):
        cfg = ScorerConfig()
        for k in ("geometry", "texture", "frequency", "color"):
            assert k in cfg.weights

    def test_default_min_score(self):
        assert ScorerConfig().min_score == pytest.approx(0.0)

    def test_default_max_score(self):
        assert ScorerConfig().max_score == pytest.approx(1.0)

    def test_default_normalize_input(self):
        assert ScorerConfig().normalize_input is True

    def test_custom_weights(self):
        cfg = ScorerConfig(weights={"geometry": 1.0, "texture": 0.5})
        assert cfg.weights["geometry"] == pytest.approx(1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(weights={"geometry": -0.1})

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=-0.1)

    def test_max_score_above_1_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(max_score=1.1)

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=0.8, max_score=0.5)

    def test_total_weight(self):
        cfg = ScorerConfig(weights={"a": 0.5, "b": 0.5})
        assert cfg.total_weight == pytest.approx(1.0)

    def test_normalized_weight(self):
        cfg = ScorerConfig(weights={"a": 1.0, "b": 3.0})
        assert cfg.normalized_weight("a") == pytest.approx(0.25)
        assert cfg.normalized_weight("b") == pytest.approx(0.75)

    def test_normalized_weight_unknown_channel(self):
        cfg = ScorerConfig(weights={"a": 1.0})
        assert cfg.normalized_weight("z") == pytest.approx(0.0)

    def test_zero_total_weight_normalized_zero(self):
        cfg = ScorerConfig(weights={})
        assert cfg.normalized_weight("a") == pytest.approx(0.0)


# ─── ChannelScore (extra) ─────────────────────────────────────────────────────

class TestChannelScoreExtra:
    def test_channel_stored(self):
        cs = ChannelScore(channel="geometry", raw=0.7, norm=0.7, weight=1.0)
        assert cs.channel == "geometry"

    def test_raw_stored(self):
        cs = ChannelScore(channel="x", raw=1.5, norm=1.0, weight=0.5)
        assert cs.raw == pytest.approx(1.5)

    def test_norm_stored(self):
        cs = ChannelScore(channel="x", raw=0.6, norm=0.6, weight=1.0)
        assert cs.norm == pytest.approx(0.6)

    def test_weight_stored(self):
        cs = ChannelScore(channel="x", raw=0.5, norm=0.5, weight=2.0)
        assert cs.weight == pytest.approx(2.0)

    def test_contribution(self):
        cs = ChannelScore(channel="x", raw=0.5, norm=0.5, weight=2.0)
        assert cs.contribution == pytest.approx(1.0)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="", raw=0.5, norm=0.5, weight=1.0)

    def test_norm_above_1_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="x", raw=2.0, norm=1.5, weight=1.0)

    def test_norm_below_0_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="x", raw=-0.5, norm=-0.1, weight=1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="x", raw=0.5, norm=0.5, weight=-0.1)

    def test_zero_contribution(self):
        cs = ChannelScore(channel="x", raw=0.0, norm=0.0, weight=5.0)
        assert cs.contribution == pytest.approx(0.0)


# ─── MatchScore (extra) ───────────────────────────────────────────────────────

class TestMatchScoreExtra:
    def test_id_a_stored(self):
        ms = _match(id_a=3)
        assert ms.id_a == 3

    def test_id_b_stored(self):
        ms = _match(id_b=7)
        assert ms.id_b == 7

    def test_score_stored(self):
        ms = _match(score=0.65)
        assert ms.score == pytest.approx(0.65)

    def test_score_above_1_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=1.1)

    def test_score_below_0_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=-0.1)

    def test_confident_default_false(self):
        ms = _match(score=0.5)
        assert ms.confident is False

    def test_confident_true(self):
        ms = _match(score=0.8)
        ms2 = MatchScore(id_a=0, id_b=1, score=0.8, confident=True)
        assert ms2.confident is True

    def test_pair_key_ordered(self):
        ms = MatchScore(id_a=5, id_b=2, score=0.7)
        assert ms.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        ms = MatchScore(id_a=1, id_b=9, score=0.7)
        assert ms.pair_key == (1, 9)

    def test_dominant_channel_none_when_empty(self):
        ms = _match()
        assert ms.dominant_channel is None

    def test_dominant_channel_max_contribution(self):
        cs1 = ChannelScore("geometry", 0.9, 0.9, 2.0)  # contribution=1.8
        cs2 = ChannelScore("texture", 0.5, 0.5, 1.0)   # contribution=0.5
        ms = MatchScore(id_a=0, id_b=1, score=0.8,
                        channels={"geometry": cs1, "texture": cs2})
        assert ms.dominant_channel == "geometry"

    def test_n_channels_empty(self):
        assert _match().n_channels == 0

    def test_n_channels_count(self):
        cs = ChannelScore("x", 0.5, 0.5, 1.0)
        ms = MatchScore(id_a=0, id_b=1, score=0.5, channels={"x": cs})
        assert ms.n_channels == 1


# ─── score_channel (extra) ────────────────────────────────────────────────────

class TestScoreChannelExtra:
    def test_returns_channel_score(self):
        cs = score_channel("geometry", 0.7, 1.0)
        assert isinstance(cs, ChannelScore)

    def test_channel_stored(self):
        cs = score_channel("texture", 0.5, 0.5)
        assert cs.channel == "texture"

    def test_norm_clipped_above(self):
        cs = score_channel("x", 2.0, 1.0)
        assert cs.norm == pytest.approx(1.0)

    def test_norm_clipped_below(self):
        cs = score_channel("x", -1.0, 1.0)
        assert cs.norm == pytest.approx(0.0)

    def test_raw_stored_as_is(self):
        cs = score_channel("x", 1.5, 1.0)
        assert cs.raw == pytest.approx(1.5)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            score_channel("", 0.5, 1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            score_channel("x", 0.5, -1.0)

    def test_normalize_false_still_clips(self):
        cs = score_channel("x", 0.7, 1.0, normalize=False)
        assert 0.0 <= cs.norm <= 1.0

    def test_weight_stored(self):
        cs = score_channel("x", 0.5, 1.5)
        assert cs.weight == pytest.approx(1.5)


# ─── compute_match_score (extra) ──────────────────────────────────────────────

class TestComputeMatchScoreExtra:
    def test_returns_match_score(self):
        ms = compute_match_score(0, 1, _channel_vals())
        assert isinstance(ms, MatchScore)

    def test_id_a_b_stored(self):
        ms = compute_match_score(3, 7, _channel_vals())
        assert ms.id_a == 3
        assert ms.id_b == 7

    def test_score_in_0_1(self):
        ms = compute_match_score(0, 1, _channel_vals())
        assert 0.0 <= ms.score <= 1.0

    def test_empty_channel_values_raises(self):
        with pytest.raises(ValueError):
            compute_match_score(0, 1, {})

    def test_channels_populated(self):
        ms = compute_match_score(0, 1, _channel_vals())
        assert len(ms.channels) == 4

    def test_confident_true_when_high_score(self):
        ms = compute_match_score(0, 1, {"geometry": 1.0}, _cfg(weights={"geometry": 1.0}))
        assert ms.confident is True

    def test_confident_false_when_low_score(self):
        ms = compute_match_score(0, 1, {"geometry": 0.0}, _cfg(weights={"geometry": 1.0}))
        assert ms.confident is False

    def test_custom_cfg(self):
        cfg = ScorerConfig(weights={"geometry": 1.0})
        ms = compute_match_score(0, 1, {"geometry": 0.5}, cfg)
        assert ms.score == pytest.approx(0.5)

    def test_none_cfg_uses_default(self):
        ms = compute_match_score(0, 1, _channel_vals(), None)
        assert isinstance(ms, MatchScore)

    def test_score_clipped_to_min_max(self):
        cfg = ScorerConfig(weights={"g": 1.0}, min_score=0.3, max_score=0.9)
        ms = compute_match_score(0, 1, {"g": 0.0}, cfg)
        assert ms.score >= cfg.min_score


# ─── aggregate_match_scores (extra) ───────────────────────────────────────────

class TestAggregateMatchScoresExtra:
    def test_empty_returns_none(self):
        assert aggregate_match_scores([]) is None

    def test_single_score(self):
        ms = _match(score=0.6)
        result = aggregate_match_scores([ms])
        assert result is not None
        assert result.score == pytest.approx(0.6)

    def test_average_of_two(self):
        result = aggregate_match_scores([_match(score=0.6), _match(score=0.8)])
        assert result is not None
        assert result.score == pytest.approx(0.7)

    def test_returns_match_score(self):
        result = aggregate_match_scores([_match()])
        assert isinstance(result, MatchScore)

    def test_ids_from_first(self):
        result = aggregate_match_scores([_match(id_a=2, id_b=5)])
        assert result.id_a == 2
        assert result.id_b == 5

    def test_confident_true_when_high_avg(self):
        scores = [_match(score=0.8), _match(score=0.9)]
        result = aggregate_match_scores(scores)
        assert result.confident is True

    def test_confident_false_when_low_avg(self):
        scores = [_match(score=0.4), _match(score=0.5)]
        result = aggregate_match_scores(scores)
        assert result.confident is False

    def test_clipped_to_0_1(self):
        result = aggregate_match_scores([_match(score=0.0), _match(score=1.0)])
        assert 0.0 <= result.score <= 1.0


# ─── build_score_table (extra) ────────────────────────────────────────────────

class TestBuildScoreTableExtra:
    def test_empty_pairs_returns_empty(self):
        result = build_score_table([], {})
        assert result == {}

    def test_single_pair(self):
        pairs = [(0, 1)]
        cvm = {(0, 1): _channel_vals()}
        result = build_score_table(pairs, cvm)
        assert len(result) == 1

    def test_key_is_ordered_pair(self):
        pairs = [(5, 2)]
        cvm = {(5, 2): _channel_vals()}
        result = build_score_table(pairs, cvm)
        assert (2, 5) in result

    def test_missing_values_score_zero(self):
        pairs = [(0, 1)]
        result = build_score_table(pairs, {})
        assert result[(0, 1)].score == pytest.approx(0.0)

    def test_reversed_key_lookup(self):
        pairs = [(0, 1)]
        cvm = {(1, 0): _channel_vals()}  # reversed key
        result = build_score_table(pairs, cvm)
        key = (0, 1)
        assert key in result
        assert result[key].score > 0.0

    def test_custom_cfg(self):
        cfg = ScorerConfig(weights={"geometry": 1.0})
        pairs = [(0, 1)]
        cvm = {(0, 1): {"geometry": 0.9}}
        result = build_score_table(pairs, cvm, cfg)
        assert result[(0, 1)].score > 0.0

    def test_multiple_pairs(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        cvm = {(0, 1): _channel_vals(), (1, 2): _channel_vals(), (0, 2): _channel_vals()}
        result = build_score_table(pairs, cvm)
        assert len(result) == 3


# ─── filter_confident_pairs (extra) ───────────────────────────────────────────

class TestFilterConfidentPairsExtra:
    def _table(self, scores: dict) -> dict:
        return {k: _match(score=v) for k, v in scores.items()}

    def test_empty_table_returns_empty(self):
        assert filter_confident_pairs({}) == []

    def test_all_below_threshold(self):
        tbl = self._table({(0, 1): 0.5, (1, 2): 0.4})
        assert filter_confident_pairs(tbl, threshold=0.7) == []

    def test_all_above_threshold(self):
        tbl = self._table({(0, 1): 0.8, (1, 2): 0.9})
        result = filter_confident_pairs(tbl, threshold=0.7)
        assert len(result) == 2

    def test_sorted_by_descending_score(self):
        tbl = self._table({(0, 1): 0.75, (1, 2): 0.95, (0, 2): 0.85})
        result = filter_confident_pairs(tbl, threshold=0.7)
        scores = [tbl[k].score for k in result]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_0_returns_all(self):
        tbl = self._table({(0, 1): 0.0, (1, 2): 0.5})
        result = filter_confident_pairs(tbl, threshold=0.0)
        assert len(result) == 2

    def test_threshold_1_empty(self):
        tbl = self._table({(0, 1): 0.99})
        result = filter_confident_pairs(tbl, threshold=1.0)
        assert result == []

    def test_invalid_threshold_above_1_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs({}, threshold=1.1)

    def test_invalid_threshold_below_0_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs({}, threshold=-0.1)
