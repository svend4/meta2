"""Тесты для puzzle_reconstruction.scoring.match_scorer."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cfg(**kw) -> ScorerConfig:
    return ScorerConfig(**kw)


def _ch(channel="geo", raw=0.8, norm=0.8, weight=0.5) -> ChannelScore:
    return ChannelScore(channel=channel, raw=raw, norm=norm, weight=weight)


def _ms(id_a=0, id_b=1, score=0.8, confident=True) -> MatchScore:
    return MatchScore(id_a=id_a, id_b=id_b, score=score, confident=confident)


def _default_values() -> dict:
    return {"geometry": 0.9, "texture": 0.7, "frequency": 0.5, "color": 0.6}


# ─── TestScorerConfig ─────────────────────────────────────────────────────────

class TestScorerConfig:
    def test_defaults(self):
        cfg = ScorerConfig()
        assert "geometry" in cfg.weights
        assert cfg.min_score == pytest.approx(0.0)
        assert cfg.max_score == pytest.approx(1.0)
        assert cfg.normalize_input is True

    def test_total_weight(self):
        cfg = ScorerConfig(weights={"a": 0.3, "b": 0.7})
        assert cfg.total_weight == pytest.approx(1.0)

    def test_total_weight_empty(self):
        cfg = ScorerConfig(weights={})
        assert cfg.total_weight == pytest.approx(0.0)

    def test_normalized_weight_known(self):
        cfg = ScorerConfig(weights={"a": 0.4, "b": 0.6})
        assert cfg.normalized_weight("a") == pytest.approx(0.4)

    def test_normalized_weight_unknown(self):
        cfg = ScorerConfig(weights={"a": 1.0})
        assert cfg.normalized_weight("z") == pytest.approx(0.0)

    def test_normalized_weight_zero_total(self):
        cfg = ScorerConfig(weights={})
        assert cfg.normalized_weight("x") == pytest.approx(0.0)

    def test_weight_neg_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(weights={"a": -0.1})

    def test_weight_zero_ok(self):
        cfg = ScorerConfig(weights={"a": 0.0})
        assert cfg.weights["a"] == 0.0

    def test_min_score_neg_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=-0.1)

    def test_max_score_above_one_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(max_score=1.1)

    def test_min_above_max_raises(self):
        with pytest.raises(ValueError):
            ScorerConfig(min_score=0.8, max_score=0.5)

    def test_min_equals_max_ok(self):
        cfg = ScorerConfig(min_score=0.5, max_score=0.5)
        assert cfg.min_score == pytest.approx(0.5)

    def test_custom_weights(self):
        cfg = ScorerConfig(weights={"shape": 2.0, "color": 1.0})
        assert cfg.weights["shape"] == pytest.approx(2.0)


# ─── TestChannelScore ─────────────────────────────────────────────────────────

class TestChannelScore:
    def test_basic(self):
        cs = _ch()
        assert cs.channel == "geo"
        assert cs.raw == pytest.approx(0.8)

    def test_contribution(self):
        cs = ChannelScore(channel="a", raw=1.0, norm=0.8, weight=0.5)
        assert cs.contribution == pytest.approx(0.4)

    def test_contribution_zero_weight(self):
        cs = ChannelScore(channel="a", raw=1.0, norm=1.0, weight=0.0)
        assert cs.contribution == pytest.approx(0.0)

    def test_contribution_zero_norm(self):
        cs = ChannelScore(channel="a", raw=0.0, norm=0.0, weight=1.0)
        assert cs.contribution == pytest.approx(0.0)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="", raw=0.5, norm=0.5, weight=1.0)

    def test_norm_neg_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="a", raw=0.5, norm=-0.1, weight=1.0)

    def test_norm_above_one_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="a", raw=0.5, norm=1.1, weight=1.0)

    def test_weight_neg_raises(self):
        with pytest.raises(ValueError):
            ChannelScore(channel="a", raw=0.5, norm=0.5, weight=-0.1)

    def test_norm_zero_ok(self):
        cs = ChannelScore(channel="a", raw=0.0, norm=0.0, weight=1.0)
        assert cs.norm == 0.0

    def test_norm_one_ok(self):
        cs = ChannelScore(channel="a", raw=1.0, norm=1.0, weight=1.0)
        assert cs.norm == 1.0


# ─── TestMatchScore ───────────────────────────────────────────────────────────

class TestMatchScore:
    def test_basic(self):
        ms = _ms(0, 1, 0.75)
        assert ms.id_a == 0
        assert ms.id_b == 1
        assert ms.score == pytest.approx(0.75)

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            MatchScore(id_a=0, id_b=1, score=1.1)

    def test_score_zero_ok(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.0)
        assert ms.score == 0.0

    def test_score_one_ok(self):
        ms = MatchScore(id_a=0, id_b=1, score=1.0)
        assert ms.score == 1.0

    def test_pair_key_ordered(self):
        ms = MatchScore(id_a=3, id_b=1, score=0.5)
        assert ms.pair_key == (1, 3)

    def test_pair_key_already_ordered(self):
        ms = MatchScore(id_a=1, id_b=3, score=0.5)
        assert ms.pair_key == (1, 3)

    def test_pair_key_same_id(self):
        ms = MatchScore(id_a=2, id_b=2, score=0.5)
        assert ms.pair_key == (2, 2)

    def test_dominant_channel_max_contribution(self):
        ch_a = ChannelScore("a", 1.0, 1.0, 0.3)
        ch_b = ChannelScore("b", 1.0, 1.0, 0.7)
        ms = MatchScore(id_a=0, id_b=1, score=0.8,
                        channels={"a": ch_a, "b": ch_b})
        assert ms.dominant_channel == "b"

    def test_dominant_channel_none_when_empty(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.5)
        assert ms.dominant_channel is None

    def test_n_channels(self):
        ch = {"a": _ch("a"), "b": _ch("b")}
        ms = MatchScore(id_a=0, id_b=1, score=0.5, channels=ch)
        assert ms.n_channels == 2

    def test_n_channels_empty(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.5)
        assert ms.n_channels == 0

    def test_confident_true(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.9, confident=True)
        assert ms.confident is True

    def test_confident_false(self):
        ms = MatchScore(id_a=0, id_b=1, score=0.3, confident=False)
        assert ms.confident is False


# ─── TestScoreChannel ─────────────────────────────────────────────────────────

class TestScoreChannel:
    def test_returns_channel_score(self):
        cs = score_channel("geo", 0.8, 0.5)
        assert isinstance(cs, ChannelScore)

    def test_channel_stored(self):
        cs = score_channel("texture", 0.7, 0.3)
        assert cs.channel == "texture"

    def test_raw_stored(self):
        cs = score_channel("geo", 0.8, 0.5)
        assert cs.raw == pytest.approx(0.8)

    def test_norm_clips_above_one(self):
        cs = score_channel("a", 1.5, 1.0, normalize=True)
        assert cs.norm == pytest.approx(1.0)

    def test_norm_clips_below_zero(self):
        cs = score_channel("a", -0.5, 1.0, normalize=True)
        assert cs.norm == pytest.approx(0.0)

    def test_weight_stored(self):
        cs = score_channel("geo", 0.8, 0.4)
        assert cs.weight == pytest.approx(0.4)

    def test_empty_channel_raises(self):
        with pytest.raises(ValueError):
            score_channel("", 0.5, 1.0)

    def test_neg_weight_raises(self):
        with pytest.raises(ValueError):
            score_channel("a", 0.5, -1.0)

    def test_zero_weight_ok(self):
        cs = score_channel("a", 0.5, 0.0)
        assert cs.weight == 0.0

    def test_no_normalize_pass_through(self):
        cs = score_channel("a", 0.3, 1.0, normalize=False)
        assert cs.norm == pytest.approx(np.clip(0.3, 0.0, 1.0))


# ─── TestComputeMatchScore ────────────────────────────────────────────────────

class TestComputeMatchScore:
    def test_returns_match_score(self):
        ms = compute_match_score(0, 1, _default_values())
        assert isinstance(ms, MatchScore)

    def test_ids_stored(self):
        ms = compute_match_score(3, 7, _default_values())
        assert ms.id_a == 3
        assert ms.id_b == 7

    def test_score_in_range(self):
        ms = compute_match_score(0, 1, _default_values())
        assert 0.0 <= ms.score <= 1.0

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            compute_match_score(0, 1, {})

    def test_channels_populated(self):
        ms = compute_match_score(0, 1, {"geometry": 0.8, "texture": 0.6})
        assert len(ms.channels) > 0

    def test_confident_high_score(self):
        cfg = ScorerConfig(weights={"a": 1.0}, min_score=0.0, max_score=1.0)
        ms = compute_match_score(0, 1, {"a": 1.0}, cfg)
        assert ms.confident is True

    def test_not_confident_low_score(self):
        cfg = ScorerConfig(weights={"a": 1.0})
        ms = compute_match_score(0, 1, {"a": 0.1}, cfg)
        assert ms.confident is False

    def test_clip_to_min_score(self):
        cfg = ScorerConfig(weights={"a": 1.0}, min_score=0.5, max_score=1.0)
        ms = compute_match_score(0, 1, {"a": 0.0}, cfg)
        assert ms.score >= 0.5

    def test_clip_to_max_score(self):
        cfg = ScorerConfig(weights={"a": 1.0}, min_score=0.0, max_score=0.6)
        ms = compute_match_score(0, 1, {"a": 1.0}, cfg)
        assert ms.score <= 0.6

    def test_unknown_channel_weight_zero(self):
        cfg = ScorerConfig(weights={"geo": 1.0})
        # "unknown" не в weights → вес 0 → вклад 0
        ms = compute_match_score(0, 1, {"unknown": 1.0}, cfg)
        assert ms.score == pytest.approx(0.0, abs=0.05)

    def test_single_channel(self):
        cfg = ScorerConfig(weights={"geo": 1.0})
        ms = compute_match_score(0, 1, {"geo": 0.8}, cfg)
        assert ms.score == pytest.approx(0.8, abs=1e-6)

    def test_default_config_used(self):
        ms = compute_match_score(0, 1, _default_values())
        assert isinstance(ms, MatchScore)


# ─── TestAggregateMatchScores ─────────────────────────────────────────────────

class TestAggregateMatchScores:
    def test_empty_returns_none(self):
        assert aggregate_match_scores([]) is None

    def test_single_preserves_score(self):
        ms = _ms(0, 1, 0.8)
        agg = aggregate_match_scores([ms])
        assert agg.score == pytest.approx(0.8)

    def test_mean_score(self):
        scores = [_ms(0, 1, 0.6), _ms(0, 1, 0.8)]
        agg = aggregate_match_scores(scores)
        assert agg.score == pytest.approx(0.7)

    def test_ids_from_first(self):
        agg = aggregate_match_scores([_ms(3, 7, 0.5)])
        assert agg.id_a == 3
        assert agg.id_b == 7

    def test_returns_match_score(self):
        agg = aggregate_match_scores([_ms()])
        assert isinstance(agg, MatchScore)

    def test_confident_high_mean(self):
        agg = aggregate_match_scores([_ms(0, 1, 0.8), _ms(0, 1, 0.9)])
        assert agg.confident is True

    def test_not_confident_low_mean(self):
        agg = aggregate_match_scores([_ms(0, 1, 0.2), _ms(0, 1, 0.3)])
        assert agg.confident is False

    def test_clipped_to_one(self):
        agg = aggregate_match_scores([_ms(0, 1, 1.0), _ms(0, 1, 1.0)])
        assert agg.score <= 1.0

    def test_clipped_to_zero(self):
        agg = aggregate_match_scores([_ms(0, 1, 0.0), _ms(0, 1, 0.0)])
        assert agg.score >= 0.0

    def test_multiple_scores_averaged(self):
        scores = [_ms(0, 1, float(i) / 10) for i in range(1, 6)]
        agg = aggregate_match_scores(scores)
        expected = sum(i / 10 for i in range(1, 6)) / 5
        assert agg.score == pytest.approx(expected, abs=1e-6)


# ─── TestBuildScoreTable ──────────────────────────────────────────────────────

class TestBuildScoreTable:
    def _cvm(self, *pairs, val=0.8) -> dict:
        return {p: {"geometry": val, "texture": val,
                    "frequency": val, "color": val}
                for p in pairs}

    def test_returns_dict(self):
        result = build_score_table([(0, 1)], self._cvm((0, 1)))
        assert isinstance(result, dict)

    def test_key_is_ordered_pair(self):
        result = build_score_table([(3, 1)], self._cvm((3, 1)))
        assert (1, 3) in result

    def test_missing_pair_zero_score(self):
        result = build_score_table([(0, 1)], {})
        assert result[(0, 1)].score == pytest.approx(0.0)

    def test_all_pairs_present(self):
        pairs = [(0, 1), (1, 2), (0, 2)]
        cvm = self._cvm(*pairs)
        result = build_score_table(pairs, cvm)
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            assert key in result

    def test_reverse_key_lookup(self):
        # Передаём (1, 0), но данные под (0, 1)
        cvm = {(0, 1): {"geometry": 0.9, "texture": 0.9,
                        "frequency": 0.9, "color": 0.9}}
        result = build_score_table([(1, 0)], cvm)
        assert result[(0, 1)].score > 0.0

    def test_empty_pairs(self):
        result = build_score_table([], {})
        assert result == {}

    def test_score_in_range(self):
        pairs = [(0, 1)]
        cvm = self._cvm((0, 1))
        result = build_score_table(pairs, cvm)
        for ms in result.values():
            assert 0.0 <= ms.score <= 1.0

    def test_all_match_scores(self):
        pairs = [(0, 1), (0, 2)]
        cvm = self._cvm(*pairs)
        for ms in build_score_table(pairs, cvm).values():
            assert isinstance(ms, MatchScore)


# ─── TestFilterConfidentPairs ─────────────────────────────────────────────────

class TestFilterConfidentPairs:
    def _table(self, pairs: dict) -> dict:
        return {k: _ms(k[0], k[1], v) for k, v in pairs.items()}

    def test_returns_list(self):
        table = self._table({(0, 1): 0.8})
        result = filter_confident_pairs(table)
        assert isinstance(result, list)

    def test_all_above_threshold(self):
        table = self._table({(0, 1): 0.8, (0, 2): 0.9, (1, 2): 0.6})
        result = filter_confident_pairs(table, threshold=0.7)
        assert len(result) == 2

    def test_none_above_threshold(self):
        table = self._table({(0, 1): 0.4, (0, 2): 0.5})
        result = filter_confident_pairs(table, threshold=0.9)
        assert result == []

    def test_all_pass_zero_threshold(self):
        table = self._table({(0, 1): 0.1, (0, 2): 0.9})
        result = filter_confident_pairs(table, threshold=0.0)
        assert len(result) == 2

    def test_sorted_descending(self):
        table = self._table({(0, 1): 0.6, (0, 2): 0.9, (1, 2): 0.75})
        result = filter_confident_pairs(table, threshold=0.0)
        scores = [table[k].score for k in result]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_boundary_included(self):
        table = self._table({(0, 1): 0.7})
        result = filter_confident_pairs(table, threshold=0.7)
        assert len(result) == 1

    def test_neg_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs({}, threshold=-0.1)

    def test_above_one_threshold_raises(self):
        with pytest.raises(ValueError):
            filter_confident_pairs({}, threshold=1.1)

    def test_empty_table(self):
        assert filter_confident_pairs({}) == []

    def test_keys_are_pairs(self):
        table = self._table({(0, 1): 0.8, (0, 2): 0.9})
        result = filter_confident_pairs(table, threshold=0.0)
        for k in result:
            assert isinstance(k, tuple)
            assert len(k) == 2
