"""Extra tests for puzzle_reconstruction/scoring/match_scorer.py."""
from __future__ import annotations

import pytest

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _channel(channel="geo", raw=0.7, norm=0.7, weight=0.35) -> ChannelScore:
    return ChannelScore(channel=channel, raw=raw, norm=norm, weight=weight)


def _match(a=0, b=1, score=0.6, confident=False) -> MatchScore:
    return MatchScore(id_a=a, id_b=b, score=score, confident=confident)


def _default_vals() -> dict:
    return {"geometry": 0.8, "texture": 0.7, "frequency": 0.6, "color": 0.5}


# ─── ScorerConfig ─────────────────────────────────────────────────────────────

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

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(weights={"geometry": -0.1})

    def test_min_score_negative_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=-0.1)

    def test_max_score_gt_one_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(max_score=1.5)

    def test_min_gt_max_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=0.8, max_score=0.5)

    def test_total_weight(self):
        cfg = ScorerConfig(weights={"a": 0.4, "b": 0.6})
        assert cfg.total_weight == pytest.approx(1.0)

    def test_normalized_weight(self):
        cfg = ScorerConfig(weights={"a": 0.5, "b": 0.5})
        assert cfg.normalized_weight("a") == pytest.approx(0.5)

    def test_normalized_weight_missing_channel(self):
        cfg = ScorerConfig()
        assert cfg.normalized_weight("nonexistent") == pytest.approx(0.0)

    def test_normalized_weight_zero_total(self):
        cfg = ScorerConfig(weights={})
        assert cfg.normalized_weight("anything") == pytest.approx(0.0)


# ─── ChannelScore ─────────────────────────────────────────────────────────────

class TestChannelScoreExtra:
    def test_channel_stored(self):
        c = _channel(channel="texture")
        assert c.channel == "texture"

    def test_raw_stored(self):
        c = _channel(raw=0.9)
        assert c.raw == pytest.approx(0.9)

    def test_norm_stored(self):
        c = _channel(norm=0.8)
        assert c.norm == pytest.approx(0.8)

    def test_weight_stored(self):
        c = _channel(weight=0.25)
        assert c.weight == pytest.approx(0.25)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="", raw=0.5, norm=0.5, weight=0.1)

    def test_norm_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="c", raw=0.5, norm=1.5, weight=0.1)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="c", raw=0.5, norm=0.5, weight=-0.1)

    def test_contribution(self):
        c = _channel(norm=0.8, weight=0.5)
        assert c.contribution == pytest.approx(0.4)

    def test_zero_weight_zero_contribution(self):
        c = _channel(norm=0.9, weight=0.0)
        assert c.contribution == pytest.approx(0.0)


# ─── MatchScore ───────────────────────────────────────────────────────────────

class TestMatchScoreExtra:
    def test_ids_stored(self):
        m = _match(a=3, b=7)
        assert m.id_a == 3 and m.id_b == 7

    def test_score_stored(self):
        m = _match(score=0.75)
        assert m.score == pytest.approx(0.75)

    def test_confident_stored(self):
        m = _match(confident=True)
        assert m.confident is True

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=1.5)

    def test_pair_key_ordered(self):
        m = _match(a=5, b=2)
        assert m.pair_key == (2, 5)

    def test_dominant_channel_none_when_empty(self):
        m = _match()
        assert m.dominant_channel is None

    def test_dominant_channel_highest_contribution(self):
        cs = {
            "geo": ChannelScore("geo", 0.9, 0.9, 0.5),
            "tex": ChannelScore("tex", 0.3, 0.3, 0.1),
        }
        m = MatchScore(id_a=0, id_b=1, score=0.7, channels=cs)
        assert m.dominant_channel == "geo"

    def test_n_channels(self):
        cs = {"a": _channel("a"), "b": _channel("b")}
        m = MatchScore(id_a=0, id_b=1, score=0.5, channels=cs)
        assert m.n_channels == 2

    def test_n_channels_empty(self):
        assert _match().n_channels == 0


# ─── score_channel ────────────────────────────────────────────────────────────

class TestScoreChannelExtra:
    def test_returns_channel_score(self):
        cs = score_channel("geo", 0.8, 0.35)
        assert isinstance(cs, ChannelScore)

    def test_channel_stored(self):
        cs = score_channel("tex", 0.7, 0.25)
        assert cs.channel == "tex"

    def test_raw_stored(self):
        cs = score_channel("geo", 1.5, 0.1)
        assert cs.raw == pytest.approx(1.5)

    def test_normalize_clips_norm(self):
        cs = score_channel("geo", 1.5, 0.1, normalize=True)
        assert cs.norm == pytest.approx(1.0)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            score_channel("", 0.5, 0.1)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            score_channel("c", 0.5, -0.1)

    def test_no_normalize_clips_norm(self):
        cs = score_channel("c", 0.6, 0.1, normalize=False)
        assert 0.0 <= cs.norm <= 1.0


# ─── compute_match_score ──────────────────────────────────────────────────────

class TestComputeMatchScoreExtra:
    def test_returns_match_score(self):
        ms = compute_match_score(0, 1, _default_vals())
        assert isinstance(ms, MatchScore)

    def test_ids_stored(self):
        ms = compute_match_score(2, 5, _default_vals())
        assert ms.id_a == 2 and ms.id_b == 5

    def test_score_in_range(self):
        ms = compute_match_score(0, 1, _default_vals())
        assert 0.0 <= ms.score <= 1.0

    def test_empty_channels_raises(self):
        with pytest.raises(ValueError):
            compute_match_score(0, 1, {})

    def test_channels_populated(self):
        ms = compute_match_score(0, 1, {"geometry": 0.8})
        assert "geometry" in ms.channels

    def test_confident_high_score(self):
        ms = compute_match_score(0, 1, {"geometry": 1.0},
                                  cfg=ScorerConfig(weights={"geometry": 1.0}))
        assert ms.confident is True

    def test_none_cfg(self):
        ms = compute_match_score(0, 1, _default_vals(), cfg=None)
        assert isinstance(ms, MatchScore)


# ─── aggregate_match_scores ───────────────────────────────────────────────────

class TestAggregateMatchScoresExtra:
    def test_empty_returns_none(self):
        assert aggregate_match_scores([]) is None

    def test_single_score_preserved(self):
        m = _match(a=0, b=1, score=0.7)
        result = aggregate_match_scores([m])
        assert result is not None
        assert result.score == pytest.approx(0.7)

    def test_average_computed(self):
        m1 = _match(score=0.4)
        m2 = _match(score=0.8)
        result = aggregate_match_scores([m1, m2])
        assert result.score == pytest.approx(0.6, abs=1e-4)

    def test_ids_from_first(self):
        m1 = _match(a=2, b=5, score=0.5)
        m2 = _match(a=2, b=5, score=0.7)
        result = aggregate_match_scores([m1, m2])
        assert result.id_a == 2 and result.id_b == 5

    def test_confident_high_mean(self):
        m1 = _match(score=0.9)
        m2 = _match(score=0.8)
        result = aggregate_match_scores([m1, m2])
        assert result.confident is True

    def test_not_confident_low_mean(self):
        m1 = _match(score=0.3)
        m2 = _match(score=0.4)
        result = aggregate_match_scores([m1, m2])
        assert result.confident is False


# ─── build_score_table ────────────────────────────────────────────────────────

class TestBuildScoreTableExtra:
    def _cv_map(self):
        return {(0, 1): _default_vals(), (1, 2): _default_vals()}

    def test_returns_dict(self):
        result = build_score_table([(0, 1)], self._cv_map())
        assert isinstance(result, dict)

    def test_key_ordered(self):
        result = build_score_table([(1, 0)], {(0, 1): _default_vals()})
        assert (0, 1) in result

    def test_missing_values_score_zero(self):
        result = build_score_table([(0, 1)], {})
        assert result[(0, 1)].score == pytest.approx(0.0)

    def test_multiple_pairs(self):
        result = build_score_table([(0, 1), (1, 2)], self._cv_map())
        assert len(result) == 2

    def test_none_cfg(self):
        result = build_score_table([(0, 1)], self._cv_map(), cfg=None)
        assert (0, 1) in result


# ─── filter_confident_pairs ───────────────────────────────────────────────────

class TestFilterConfidentPairsExtra:
    def _table(self):
        return {
            (0, 1): _match(a=0, b=1, score=0.8),
            (1, 2): _match(a=1, b=2, score=0.4),
            (2, 3): _match(a=2, b=3, score=0.9),
        }

    def test_returns_list(self):
        assert isinstance(filter_confident_pairs(self._table()), list)

    def test_filters_below_threshold(self):
        pairs = filter_confident_pairs(self._table(), threshold=0.7)
        assert (1, 2) not in pairs

    def test_sorted_descending_by_score(self):
        pairs = filter_confident_pairs(self._table(), threshold=0.0)
        scores = [self._table()[p].score for p in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs(self._table(), threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs(self._table(), threshold=-0.1)

    def test_all_pass_with_zero_threshold(self):
        pairs = filter_confident_pairs(self._table(), threshold=0.0)
        assert len(pairs) == 3

    def test_none_pass_high_threshold(self):
        pairs = filter_confident_pairs(self._table(), threshold=0.99)
        assert len(pairs) == 0

    def test_empty_table(self):
        assert filter_confident_pairs({}) == []
