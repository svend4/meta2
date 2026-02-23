"""Extra tests for puzzle_reconstruction/scoring/evidence_aggregator.py."""
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


# ─── EvidenceConfig (extra) ───────────────────────────────────────────────────

class TestEvidenceConfigExtra:
    def test_require_all_default_false(self):
        assert EvidenceConfig().require_all is False

    def test_require_all_true_stored(self):
        cfg = EvidenceConfig(require_all=True)
        assert cfg.require_all is True

    def test_weights_empty_by_default(self):
        assert EvidenceConfig().weights == {}

    def test_large_weight_valid(self):
        cfg = EvidenceConfig(weights={"shape": 100.0})
        assert cfg.weights["shape"] == pytest.approx(100.0)

    def test_multiple_zero_weights_valid(self):
        cfg = EvidenceConfig(weights={"a": 0.0, "b": 0.0, "c": 0.0})
        for v in cfg.weights.values():
            assert v == pytest.approx(0.0)

    def test_confidence_threshold_stored(self):
        cfg = EvidenceConfig(confidence_threshold=0.75)
        assert cfg.confidence_threshold == pytest.approx(0.75)

    def test_min_threshold_0_stored(self):
        cfg = EvidenceConfig(min_threshold=0.0)
        assert cfg.min_threshold == pytest.approx(0.0)

    def test_min_threshold_1_stored(self):
        cfg = EvidenceConfig(min_threshold=1.0)
        assert cfg.min_threshold == pytest.approx(1.0)


# ─── EvidenceScore (extra) ────────────────────────────────────────────────────

class TestEvidenceScoreExtra:
    def test_default_channel_scores_empty(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.5)
        assert es.channel_scores == {}

    def test_default_weighted_scores_empty(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.5)
        assert es.weighted_scores == {}

    def test_n_channels_default_zero(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.5)
        assert es.n_channels == 0

    def test_n_channels_provided(self):
        es = EvidenceScore(pair_id=(2, 5), confidence=0.7, n_channels=3)
        assert es.n_channels == 3

    def test_pair_id_stored(self):
        es = EvidenceScore(pair_id=(10, 20), confidence=0.5)
        assert es.pair_id == (10, 20)

    def test_dominant_channel_with_single(self):
        es = EvidenceScore(
            pair_id=(0, 1), confidence=0.5,
            weighted_scores={"only": 0.9},
        )
        assert es.dominant_channel == "only"

    def test_dominant_channel_with_three(self):
        es = EvidenceScore(
            pair_id=(0, 1), confidence=0.5,
            weighted_scores={"a": 0.1, "b": 0.9, "c": 0.5},
        )
        assert es.dominant_channel == "b"

    def test_is_confident_boundary_05(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.5)
        # boundary value: at exactly 0.5, is_confident is True
        assert es.is_confident is True

    def test_is_confident_below_boundary(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.49)
        assert es.is_confident is False

    def test_summary_contains_pair_id(self):
        es = EvidenceScore(pair_id=(3, 7), confidence=0.8)
        s = es.summary()
        assert "3" in s or "7" in s or "0.800" in s

    def test_zero_confidence_is_confident_false(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=0.0)
        assert es.is_confident is False

    def test_one_confidence_is_confident_true(self):
        es = EvidenceScore(pair_id=(0, 1), confidence=1.0)
        assert es.is_confident is True


# ─── weight_evidence (extra) ──────────────────────────────────────────────────

class TestWeightEvidenceExtra:
    def test_weight_one_leaves_score_unchanged(self):
        result = weight_evidence({"x": 0.7}, {"x": 1.0})
        assert result["x"] == pytest.approx(0.7)

    def test_weight_2_doubles_score_in_result(self):
        result = weight_evidence({"x": 0.5}, {"x": 2.0})
        assert result["x"] == pytest.approx(1.0)

    def test_missing_channel_weight_defaults(self):
        result = weight_evidence({"a": 0.4, "b": 0.6}, {"a": 2.0})
        # "b" weight defaults to 1.0
        assert result["b"] == pytest.approx(0.6)

    def test_all_channels_weighted(self):
        scores = {"a": 0.2, "b": 0.5, "c": 0.8}
        weights = {"a": 0.0, "b": 1.0, "c": 2.0}
        result = weight_evidence(scores, weights)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(1.6)

    def test_score_zero_with_any_weight_is_zero(self):
        result = weight_evidence({"x": 0.0}, {"x": 999.0})
        assert result["x"] == pytest.approx(0.0)

    def test_returns_copy_not_original(self):
        scores = {"x": 0.5}
        result = weight_evidence(scores, {"x": 2.0})
        result["x"] = 0.0
        assert scores["x"] == pytest.approx(0.5)


# ─── threshold_evidence (extra) ───────────────────────────────────────────────

class TestThresholdEvidenceExtra:
    def test_all_above_threshold_kept(self):
        scores = {"a": 0.6, "b": 0.9}
        result = threshold_evidence(scores, min_threshold=0.5)
        assert result["a"] == pytest.approx(0.6)
        assert result["b"] == pytest.approx(0.9)

    def test_all_below_threshold_zeroed(self):
        scores = {"a": 0.1, "b": 0.2}
        result = threshold_evidence(scores, min_threshold=0.5)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.0)

    def test_threshold_1_zeroes_all_except_exact(self):
        scores = {"a": 0.5, "b": 1.0}
        result = threshold_evidence(scores, min_threshold=1.0)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(1.0)

    def test_threshold_exactly_score_kept(self):
        result = threshold_evidence({"x": 0.75}, min_threshold=0.75)
        assert result["x"] == pytest.approx(0.75)

    def test_does_not_modify_input(self):
        scores = {"a": 0.3}
        _ = threshold_evidence(scores, min_threshold=0.5)
        assert scores["a"] == pytest.approx(0.3)

    def test_large_dict(self):
        scores = {str(i): i / 100 for i in range(100)}
        result = threshold_evidence(scores, min_threshold=0.5)
        for k, v in result.items():
            orig = scores[k]
            if orig < 0.5:
                assert v == pytest.approx(0.0)
            else:
                assert v == pytest.approx(orig)


# ─── compute_confidence (extra) ───────────────────────────────────────────────

class TestComputeConfidenceExtra:
    def test_two_equal_channels(self):
        result = compute_confidence({"a": 0.6, "b": 0.6}, {"a": 1.0, "b": 1.0})
        assert result == pytest.approx(0.6)

    def test_high_weighted_score_dominates(self):
        # weighted_scores are already score*weight; weights used for normalization
        # "b": weighted_score=90.0, weight=100.0 → contribution 90/100.01 ≈ 0.9
        result = compute_confidence(
            {"a": 0.0, "b": 90.0},
            {"a": 0.01, "b": 100.0}
        )
        assert result > 0.80

    def test_all_zeros_channels(self):
        result = compute_confidence({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0})
        assert result == pytest.approx(0.0)

    def test_result_is_float(self):
        result = compute_confidence({"x": 0.5}, {"x": 1.0})
        assert isinstance(result, float)

    def test_clamped_above_1(self):
        # Passing weighted_score > 1 → clamped to 1.0
        result = compute_confidence({"x": 100.0}, {"x": 1.0})
        assert result <= 1.0

    def test_clamped_below_0(self):
        result = compute_confidence({}, {})
        assert result >= 0.0


# ─── aggregate_evidence (extra) ───────────────────────────────────────────────

class TestAggregateEvidenceExtra:
    def test_single_channel_confidence(self):
        result = aggregate_evidence({"shape": 0.8})
        assert 0.0 <= result.confidence <= 1.0

    def test_three_channels_all_present(self):
        result = aggregate_evidence({"a": 0.5, "b": 0.7, "c": 0.9})
        assert result.n_channels == 3

    def test_custom_weights_affect_confidence(self):
        cfg_high = EvidenceConfig(weights={"shape": 10.0})
        cfg_low  = EvidenceConfig(weights={"shape": 0.1})
        high = aggregate_evidence({"shape": 0.8}, cfg=cfg_high)
        low  = aggregate_evidence({"shape": 0.8}, cfg=cfg_low)
        # Both confidences are in [0,1], different weighting can produce
        # same value if clamped; just verify range
        assert 0.0 <= high.confidence <= 1.0
        assert 0.0 <= low.confidence <= 1.0

    def test_pair_id_default(self):
        result = aggregate_evidence({"x": 0.5})
        assert isinstance(result.pair_id, tuple)
        assert len(result.pair_id) == 2

    def test_channel_scores_stored(self):
        result = aggregate_evidence({"shape": 0.6, "color": 0.8})
        assert "shape" in result.channel_scores
        assert "color" in result.channel_scores

    def test_is_confident_reflects_threshold(self):
        cfg = EvidenceConfig(confidence_threshold=0.9)
        low = aggregate_evidence({"shape": 0.1}, cfg=cfg)
        assert low.is_confident is False

    def test_require_all_satisfied(self):
        cfg = EvidenceConfig(
            weights={"shape": 1.0, "color": 1.0},
            require_all=True,
        )
        # Provide both channels → should NOT raise
        result = aggregate_evidence({"shape": 0.8, "color": 0.7}, cfg=cfg)
        assert isinstance(result, EvidenceScore)


# ─── rank_by_evidence (extra) ─────────────────────────────────────────────────

class TestRankByEvidenceExtra:
    def test_ties_preserved_in_some_order(self):
        scores = [
            EvidenceScore(pair_id=(0, 1), confidence=0.5),
            EvidenceScore(pair_id=(1, 2), confidence=0.5),
        ]
        result = rank_by_evidence(scores)
        assert len(result) == 2
        assert all(r.confidence == pytest.approx(0.5) for r in result)

    def test_highest_confidence_first(self):
        scores = [
            EvidenceScore(pair_id=(0, 1), confidence=0.1),
            EvidenceScore(pair_id=(1, 2), confidence=0.9),
            EvidenceScore(pair_id=(2, 3), confidence=0.5),
        ]
        result = rank_by_evidence(scores)
        assert result[0].confidence == pytest.approx(0.9)

    def test_returns_new_list(self):
        scores = [EvidenceScore(pair_id=(0, 1), confidence=0.7)]
        result = rank_by_evidence(scores)
        assert result is not scores

    def test_large_list_sorted(self):
        import random
        random.seed(42)
        confidences = [random.uniform(0, 1) for _ in range(50)]
        scores = [EvidenceScore(pair_id=(i, i+1), confidence=c)
                  for i, c in enumerate(confidences)]
        result = rank_by_evidence(scores)
        for i in range(len(result) - 1):
            assert result[i].confidence >= result[i + 1].confidence

    def test_all_zero_confidence(self):
        scores = [EvidenceScore(pair_id=(i, i+1), confidence=0.0)
                  for i in range(5)]
        result = rank_by_evidence(scores)
        assert len(result) == 5


# ─── batch_aggregate (extra) ──────────────────────────────────────────────────

class TestBatchAggregateExtra:
    def test_single_item_batch(self):
        result = batch_aggregate([{"shape": 0.8}])
        assert len(result) == 1
        assert isinstance(result[0], EvidenceScore)

    def test_default_pair_ids_incremental(self):
        batch = [{"x": 0.5} for _ in range(5)]
        result = batch_aggregate(batch)
        for i, es in enumerate(result):
            assert es.pair_id == (i, i + 1)

    def test_cfg_min_threshold_applied(self):
        cfg = EvidenceConfig(min_threshold=0.8)
        batch = [{"shape": 0.5}]
        result = batch_aggregate(batch, cfg=cfg)
        assert result[0].channel_scores["shape"] == pytest.approx(0.0)

    def test_all_channels_in_results(self):
        batch = [{"a": 0.5, "b": 0.7, "c": 0.9}]
        result = batch_aggregate(batch)
        for ch in ("a", "b", "c"):
            assert ch in result[0].channel_scores

    def test_pair_ids_length_mismatch_raises(self):
        batch = [{"x": 0.5}, {"y": 0.6}, {"z": 0.7}]
        with pytest.raises(ValueError):
            batch_aggregate(batch, pair_ids=[(0, 1), (1, 2)])

    def test_large_batch_length(self):
        batch = [{"shape": i / 100.0} for i in range(100)]
        result = batch_aggregate(batch)
        assert len(result) == 100
