"""Тесты для puzzle_reconstruction/scoring/evidence_aggregator.py."""
import pytest

from puzzle_reconstruction.scoring.evidence_aggregator import (
    EvidenceConfig,
    EvidenceScore,
    weight_evidence,
    threshold_evidence,
    compute_confidence,
    aggregate_evidence,
    rank_by_evidence,
    batch_aggregate,
)


# ─── EvidenceConfig ───────────────────────────────────────────────────────────

class TestEvidenceConfig:
    def test_defaults(self):
        cfg = EvidenceConfig()
        assert cfg.weights == {}
        assert cfg.min_threshold == pytest.approx(0.0)
        assert cfg.require_all is False
        assert cfg.confidence_threshold == pytest.approx(0.5)

    def test_min_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="min_threshold"):
            EvidenceConfig(min_threshold=-0.1)

    def test_min_threshold_above_1_raises(self):
        with pytest.raises(ValueError, match="min_threshold"):
            EvidenceConfig(min_threshold=1.1)

    def test_confidence_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            EvidenceConfig(confidence_threshold=1.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            EvidenceConfig(weights={"shape": -0.1})

    def test_zero_weight_valid(self):
        cfg = EvidenceConfig(weights={"shape": 0.0})
        assert cfg.weights["shape"] == pytest.approx(0.0)

    def test_valid_weights_stored(self):
        cfg = EvidenceConfig(weights={"shape": 2.0, "color": 1.5})
        assert cfg.weights["shape"] == pytest.approx(2.0)
        assert cfg.weights["color"] == pytest.approx(1.5)

    def test_boundary_thresholds_valid(self):
        cfg = EvidenceConfig(min_threshold=0.0, confidence_threshold=1.0)
        assert cfg.min_threshold == pytest.approx(0.0)
        assert cfg.confidence_threshold == pytest.approx(1.0)


# ─── EvidenceScore ────────────────────────────────────────────────────────────

class TestEvidenceScore:
    def test_creation(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.7)
        assert es.pair_id == (0, 1)
        assert es.confidence == pytest.approx(0.7)
        assert es.channel_scores == {}
        assert es.weighted_scores == {}
        assert es.n_channels == 0

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            EvidenceScore(pair_id=(0, 1), confidence=-0.1)

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            EvidenceScore(pair_id=(0, 1), confidence=1.1)

    def test_negative_n_channels_raises(self):
        with pytest.raises(ValueError, match="n_channels"):
            EvidenceScore(pair_id=(0, 1), confidence=0.5, n_channels=-1)

    def test_negative_pair_id_a_raises(self):
        with pytest.raises(ValueError, match="pair_id"):
            EvidenceScore(pair_id=(-1, 0), confidence=0.5)

    def test_negative_pair_id_b_raises(self):
        with pytest.raises(ValueError, match="pair_id"):
            EvidenceScore(pair_id=(0, -1), confidence=0.5)

    def test_is_confident_true(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.8)
        assert es.is_confident is True

    def test_is_confident_false(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.3)
        assert es.is_confident is False

    def test_is_confident_at_boundary(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.5)
        assert es.is_confident is True

    def test_dominant_channel_none_when_empty(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.5)
        assert es.dominant_channel is None

    def test_dominant_channel_returns_max(self):
        es = EvidenceScore(
            pair_id=(0, 1), confidence=0.5,
            weighted_scores={"shape": 0.3, "color": 0.8, "texture": 0.5},
        )
        assert es.dominant_channel == "color"

    def test_summary_returns_string(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.6)
        s = es.summary()
        assert isinstance(s, str)
        assert "0.600" in s

    def test_boundary_confidence_zero_valid(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.0)
        assert es.confidence == pytest.approx(0.0)

    def test_boundary_confidence_one_valid(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=1.0)
        assert es.confidence == pytest.approx(1.0)


# ─── weight_evidence ──────────────────────────────────────────────────────────

class TestWeightEvidence:
    def test_applies_weight(self):
        result = weight_evidence({"shape": 0.8}, {"shape": 2.0})
        assert result["shape"] == pytest.approx(1.6)

    def test_missing_weight_defaults_to_1(self):
        result = weight_evidence({"color": 0.5}, {})
        assert result["color"] == pytest.approx(0.5)

    def test_returns_dict(self):
        result = weight_evidence({"x": 0.4}, {"x": 1.0})
        assert isinstance(result, dict)

    def test_empty_scores_returns_empty(self):
        result = weight_evidence({}, {"shape": 2.0})
        assert result == {}

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            weight_evidence({"shape": -0.1}, {})

    def test_score_above_1_raises(self):
        with pytest.raises(ValueError):
            weight_evidence({"shape": 1.1}, {})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            weight_evidence({"shape": 0.5}, {"shape": -1.0})

    def test_zero_weight_zeroes_score(self):
        result = weight_evidence({"shape": 0.9}, {"shape": 0.0})
        assert result["shape"] == pytest.approx(0.0)

    def test_multiple_channels(self):
        scores = {"a": 0.5, "b": 0.8}
        weights = {"a": 2.0, "b": 0.5}
        result = weight_evidence(scores, weights)
        assert result["a"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(0.4)


# ─── threshold_evidence ───────────────────────────────────────────────────────

class TestThresholdEvidence:
    def test_zeros_below_threshold(self):
        scores = {"shape": 0.3, "color": 0.8}
        result = threshold_evidence(scores, min_threshold=0.5)
        assert result["shape"] == pytest.approx(0.0)
        assert result["color"] == pytest.approx(0.8)

    def test_keeps_at_threshold(self):
        result = threshold_evidence({"x": 0.5}, min_threshold=0.5)
        assert result["x"] == pytest.approx(0.5)

    def test_zero_threshold_keeps_all(self):
        scores = {"a": 0.1, "b": 0.9}
        result = threshold_evidence(scores, min_threshold=0.0)
        assert result["a"] == pytest.approx(0.1)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="min_threshold"):
            threshold_evidence({}, min_threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="min_threshold"):
            threshold_evidence({}, min_threshold=-0.1)

    def test_empty_scores_returns_empty(self):
        result = threshold_evidence({}, min_threshold=0.5)
        assert result == {}

    def test_returns_dict(self):
        result = threshold_evidence({"x": 0.5}, min_threshold=0.0)
        assert isinstance(result, dict)


# ─── compute_confidence ───────────────────────────────────────────────────────

class TestComputeConfidence:
    def test_empty_returns_zero(self):
        result = compute_confidence({}, {})
        assert result == pytest.approx(0.0)

    def test_zero_total_weight_returns_zero(self):
        result = compute_confidence({"x": 0.5}, {"x": 0.0})
        assert result == pytest.approx(0.0)

    def test_simple_weighted_average(self):
        # weighted_scores already has scores * weights; weights used for normalization
        # channel "x": weighted_score=0.6, weight=1.0 → confidence = 0.6
        result = compute_confidence({"x": 0.6}, {"x": 1.0})
        assert result == pytest.approx(0.6)

    def test_result_clamped_to_0_1(self):
        result = compute_confidence({"x": 5.0}, {"x": 1.0})
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        result = compute_confidence({"a": 0.5}, {"a": 1.0})
        assert isinstance(result, float)

    def test_missing_weight_defaults_to_1(self):
        # channel not in weights → weight = 1.0
        result = compute_confidence({"a": 0.4}, {})
        assert result == pytest.approx(0.4)


# ─── aggregate_evidence ───────────────────────────────────────────────────────

class TestAggregateEvidence:
    def test_returns_evidence_score(self):
        result = aggregate_evidence({"shape": 0.7, "color": 0.5})
        assert isinstance(result, EvidenceScore)

    def test_confidence_in_0_1(self):
        result = aggregate_evidence({"shape": 0.7, "color": 0.5})
        assert 0.0 <= result.confidence <= 1.0

    def test_pair_id_stored(self):
        result = aggregate_evidence({"shape": 0.5}, pair_id=(3, 7))
        assert result.pair_id == (3, 7)

    def test_n_channels_correct(self):
        result = aggregate_evidence({"a": 0.5, "b": 0.8})
        assert result.n_channels == 2

    def test_channel_scores_present(self):
        result = aggregate_evidence({"shape": 0.6})
        assert "shape" in result.channel_scores

    def test_weighted_scores_present(self):
        result = aggregate_evidence({"shape": 0.6})
        assert "shape" in result.weighted_scores

    def test_require_all_missing_channel_raises(self):
        cfg = EvidenceConfig(
            weights={"shape": 1.0, "color": 1.0},
            require_all=True,
        )
        with pytest.raises(ValueError, match="require_all"):
            aggregate_evidence({"shape": 0.8}, cfg=cfg)

    def test_require_all_zeroed_raises(self):
        cfg = EvidenceConfig(
            weights={"shape": 1.0},
            min_threshold=0.9,
            require_all=True,
        )
        with pytest.raises(ValueError, match="require_all"):
            aggregate_evidence({"shape": 0.5}, cfg=cfg)

    def test_empty_scores_zero_confidence(self):
        result = aggregate_evidence({})
        assert result.confidence == pytest.approx(0.0)

    def test_threshold_zeroes_low_scores(self):
        cfg = EvidenceConfig(min_threshold=0.7)
        result = aggregate_evidence({"shape": 0.3}, cfg=cfg)
        assert result.channel_scores["shape"] == pytest.approx(0.0)

    def test_none_cfg_uses_defaults(self):
        result = aggregate_evidence({"x": 0.5}, cfg=None)
        assert isinstance(result, EvidenceScore)


# ─── rank_by_evidence ─────────────────────────────────────────────────────────

class TestRankByEvidence:
    def test_returns_list(self):
        result = rank_by_evidence([])
        assert isinstance(result, list)

    def test_empty_returns_empty(self):
        assert rank_by_evidence([]) == []

    def test_sorted_descending(self):
        scores = [
            EvidenceScore(pair_id=(0, 1), confidence=0.3),
            EvidenceScore(pair_id=(1, 2), confidence=0.9),
            EvidenceScore(pair_id=(2, 3), confidence=0.6),
        ]
        result = rank_by_evidence(scores)
        conf = [r.confidence for r in result]
        assert conf == sorted(conf, reverse=True)

    def test_does_not_modify_original(self):
        scores = [
            EvidenceScore(pair_id=(0, 1), confidence=0.3),
            EvidenceScore(pair_id=(1, 2), confidence=0.9),
        ]
        original_first = scores[0].confidence
        rank_by_evidence(scores)
        assert scores[0].confidence == pytest.approx(original_first)

    def test_single_item(self):
        scores = [EvidenceScore(pair_id=(0, 1), confidence=0.7)]
        result = rank_by_evidence(scores)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.7)

    def test_length_preserved(self):
        scores = [EvidenceScore(pair_id=(i, i+1), confidence=i*0.1)
                  for i in range(5)]
        result = rank_by_evidence(scores)
        assert len(result) == 5


# ─── batch_aggregate ──────────────────────────────────────────────────────────

class TestBatchAggregate:
    def test_empty_batch_returns_empty(self):
        result = batch_aggregate([])
        assert result == []

    def test_length_matches_batch(self):
        batch = [{"shape": 0.5}, {"color": 0.7}, {"texture": 0.3}]
        result = batch_aggregate(batch)
        assert len(result) == 3

    def test_returns_list_of_evidence_scores(self):
        batch = [{"shape": 0.5}, {"color": 0.7}]
        result = batch_aggregate(batch)
        for r in result:
            assert isinstance(r, EvidenceScore)

    def test_default_pair_ids(self):
        batch = [{"x": 0.5}, {"y": 0.6}]
        result = batch_aggregate(batch)
        assert result[0].pair_id == (0, 1)
        assert result[1].pair_id == (1, 2)

    def test_custom_pair_ids_stored(self):
        batch = [{"x": 0.5}, {"y": 0.6}]
        pair_ids = [(3, 5), (7, 9)]
        result = batch_aggregate(batch, pair_ids=pair_ids)
        assert result[0].pair_id == (3, 5)
        assert result[1].pair_id == (7, 9)

    def test_mismatched_pair_ids_raises(self):
        batch = [{"x": 0.5}, {"y": 0.6}]
        with pytest.raises(ValueError):
            batch_aggregate(batch, pair_ids=[(0, 1)])

    def test_cfg_applied(self):
        cfg = EvidenceConfig(min_threshold=0.9)
        batch = [{"shape": 0.3}]
        result = batch_aggregate(batch, cfg=cfg)
        assert result[0].channel_scores["shape"] == pytest.approx(0.0)
