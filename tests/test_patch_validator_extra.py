"""Extra tests for puzzle_reconstruction/matching/patch_validator.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.matching.patch_validator import (
    PatchValidConfig,
    PatchScore,
    PatchValidResult,
    compute_patch_score,
    aggregate_patch_scores,
    validate_patch_pair,
    batch_validate_patches,
    filter_valid_pairs,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _patch(val: int = 128, size: int = 8) -> np.ndarray:
    return np.full((size, size), val, dtype=np.uint8)


def _score(c=0.5, t=0.5, g=0.5, total=0.5, valid=True) -> PatchScore:
    return PatchScore(
        color_score=c, texture_score=t,
        gradient_score=g, total_score=total, valid=valid,
    )


# ─── PatchValidConfig ─────────────────────────────────────────────────────────

class TestPatchValidConfigExtra:
    def test_defaults(self):
        cfg = PatchValidConfig()
        assert cfg.color_weight == pytest.approx(0.4)
        assert cfg.texture_weight == pytest.approx(0.3)
        assert cfg.gradient_weight == pytest.approx(0.3)

    def test_default_min_patch_size(self):
        assert PatchValidConfig().min_patch_size == 4

    def test_default_threshold(self):
        assert PatchValidConfig().threshold == pytest.approx(0.5)

    def test_color_weight_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(color_weight=1.5)

    def test_texture_weight_negative_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(texture_weight=-0.1)

    def test_gradient_weight_gt_one_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(gradient_weight=2.0)

    def test_min_patch_size_zero_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(min_patch_size=0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=-0.1)

    def test_threshold_gt_one_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=1.1)

    def test_weight_sum_property(self):
        cfg = PatchValidConfig(color_weight=0.4, texture_weight=0.3, gradient_weight=0.3)
        assert cfg.weight_sum == pytest.approx(1.0)

    def test_normalized_weights_sum_to_one(self):
        cfg = PatchValidConfig(color_weight=0.5, texture_weight=0.3, gradient_weight=0.2)
        w = cfg.normalized_weights()
        assert sum(w) == pytest.approx(1.0)

    def test_normalized_weights_zero_sum(self):
        cfg = PatchValidConfig(color_weight=0.0, texture_weight=0.0, gradient_weight=0.0)
        w = cfg.normalized_weights()
        assert all(abs(v - 1.0 / 3) < 1e-9 for v in w)

    def test_custom_threshold(self):
        cfg = PatchValidConfig(threshold=0.7)
        assert cfg.threshold == pytest.approx(0.7)


# ─── PatchScore ───────────────────────────────────────────────────────────────

class TestPatchScoreExtra:
    def test_stores_color_score(self):
        s = _score(c=0.8)
        assert s.color_score == pytest.approx(0.8)

    def test_stores_texture_score(self):
        s = _score(t=0.6)
        assert s.texture_score == pytest.approx(0.6)

    def test_stores_gradient_score(self):
        s = _score(g=0.3)
        assert s.gradient_score == pytest.approx(0.3)

    def test_stores_total_score(self):
        s = _score(total=0.7)
        assert s.total_score == pytest.approx(0.7)

    def test_stores_valid(self):
        assert _score(valid=False).valid is False

    def test_color_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=1.2, texture_score=0.5, gradient_score=0.5, total_score=0.5)

    def test_texture_score_negative_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=0.5, texture_score=-0.1, gradient_score=0.5, total_score=0.5)

    def test_total_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=0.5, texture_score=0.5, gradient_score=0.5, total_score=1.5)

    def test_is_strong_true(self):
        assert _score(total=0.9).is_strong is True

    def test_is_strong_false(self):
        assert _score(total=0.7).is_strong is False

    def test_is_strong_boundary(self):
        assert _score(total=0.8).is_strong is False

    def test_dominant_channel_color(self):
        s = PatchScore(color_score=0.9, texture_score=0.5, gradient_score=0.3, total_score=0.6)
        assert s.dominant_channel == "color"

    def test_dominant_channel_texture(self):
        s = PatchScore(color_score=0.3, texture_score=0.9, gradient_score=0.5, total_score=0.6)
        assert s.dominant_channel == "texture"

    def test_dominant_channel_gradient(self):
        s = PatchScore(color_score=0.3, texture_score=0.4, gradient_score=0.8, total_score=0.5)
        assert s.dominant_channel == "gradient"


# ─── PatchValidResult ─────────────────────────────────────────────────────────

class TestPatchValidResultExtra:
    def _make(self, fa=0, fb=1, n=5, passed=True):
        return PatchValidResult(
            fragment_a=fa, fragment_b=fb,
            score=_score(), n_patches=n, passed=passed,
        )

    def test_fragment_a_stored(self):
        r = self._make(fa=3)
        assert r.fragment_a == 3

    def test_fragment_b_stored(self):
        r = self._make(fb=7)
        assert r.fragment_b == 7

    def test_n_patches_stored(self):
        r = self._make(n=10)
        assert r.n_patches == 10

    def test_passed_stored(self):
        assert self._make(passed=False).passed is False

    def test_n_patches_negative_raises(self):
        with pytest.raises(ValueError):
            PatchValidResult(fragment_a=0, fragment_b=1, score=_score(),
                             n_patches=-1, passed=True)

    def test_pair_key_ordered(self):
        r = self._make(fa=5, fb=2)
        assert r.pair_key == (2, 5)

    def test_pair_key_already_ordered(self):
        r = self._make(fa=1, fb=4)
        assert r.pair_key == (1, 4)

    def test_avg_score(self):
        s = PatchScore(color_score=0.6, texture_score=0.6, gradient_score=0.6, total_score=0.6)
        r = PatchValidResult(fragment_a=0, fragment_b=1, score=s, n_patches=1, passed=True)
        assert r.avg_score == pytest.approx(0.6)


# ─── compute_patch_score ──────────────────────────────────────────────────────

class TestComputePatchScoreExtra:
    def test_returns_patch_score(self):
        a = _patch(100)
        b = _patch(100)
        result = compute_patch_score(a, b)
        assert isinstance(result, PatchScore)

    def test_identical_patches_high_color(self):
        a = _patch(100)
        result = compute_patch_score(a, a.copy())
        assert result.color_score > 0.9

    def test_very_different_patches(self):
        a = _patch(0)
        b = _patch(255)
        result = compute_patch_score(a, b)
        assert result.total_score < 0.8

    def test_total_score_in_range(self):
        a = _patch(80)
        b = _patch(120)
        result = compute_patch_score(a, b)
        assert 0.0 <= result.total_score <= 1.0

    def test_too_small_patch_raises(self):
        cfg = PatchValidConfig(min_patch_size=100)
        a = np.zeros((2, 2), dtype=np.uint8)
        b = np.zeros((2, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_patch_score(a, b, cfg)

    def test_custom_cfg_used(self):
        cfg = PatchValidConfig(threshold=0.0)
        a = _patch(0)
        b = _patch(255)
        result = compute_patch_score(a, b, cfg)
        assert result.valid is True  # threshold=0.0 means everything passes

    def test_none_cfg_uses_defaults(self):
        a = _patch(128)
        b = _patch(128)
        result = compute_patch_score(a, b, cfg=None)
        assert isinstance(result, PatchScore)

    def test_valid_flag_reflects_threshold(self):
        cfg = PatchValidConfig(threshold=0.99)
        a = _patch(0)
        b = _patch(128)
        result = compute_patch_score(a, b, cfg)
        assert result.valid is False


# ─── aggregate_patch_scores ───────────────────────────────────────────────────

class TestAggregatePatchScoresExtra:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_patch_scores([])

    def test_single_score_preserved(self):
        s = _score(c=0.7, t=0.6, g=0.5, total=0.6)
        agg = aggregate_patch_scores([s])
        assert agg.color_score == pytest.approx(0.7)
        assert agg.texture_score == pytest.approx(0.6)

    def test_average_computed(self):
        s1 = _score(c=0.4, t=0.4, g=0.4, total=0.4)
        s2 = _score(c=0.8, t=0.8, g=0.8, total=0.8)
        agg = aggregate_patch_scores([s1, s2])
        assert agg.color_score == pytest.approx(0.6, abs=1e-4)

    def test_returns_patch_score(self):
        s = _score()
        assert isinstance(aggregate_patch_scores([s]), PatchScore)

    def test_valid_flag_with_threshold(self):
        cfg = PatchValidConfig(threshold=0.9)
        s = _score(c=0.3, t=0.3, g=0.3, total=0.3)
        agg = aggregate_patch_scores([s], cfg)
        assert agg.valid is False

    def test_none_cfg(self):
        s = _score()
        agg = aggregate_patch_scores([s], cfg=None)
        assert isinstance(agg, PatchScore)

    def test_total_in_range(self):
        scores = [_score(c=float(i) / 10, t=float(i) / 10,
                          g=float(i) / 10, total=float(i) / 10)
                  for i in range(1, 6)]
        agg = aggregate_patch_scores(scores)
        assert 0.0 <= agg.total_score <= 1.0


# ─── validate_patch_pair ──────────────────────────────────────────────────────

class TestValidatePatchPairExtra:
    def _patches(self, n=3, val=128):
        return [_patch(val) for _ in range(n)]

    def test_returns_result(self):
        r = validate_patch_pair(0, 1, self._patches(), self._patches())
        assert isinstance(r, PatchValidResult)

    def test_fragment_ids_stored(self):
        r = validate_patch_pair(2, 5, self._patches(), self._patches())
        assert r.fragment_a == 2
        assert r.fragment_b == 5

    def test_n_patches_matches_min(self):
        r = validate_patch_pair(0, 1, self._patches(5), self._patches(3))
        assert r.n_patches == 3

    def test_empty_patches_not_passed(self):
        r = validate_patch_pair(0, 1, [], [])
        assert r.passed is False
        assert r.n_patches == 0

    def test_identical_patches_passed_with_low_threshold(self):
        cfg = PatchValidConfig(threshold=0.0)
        r = validate_patch_pair(0, 1, self._patches(2), self._patches(2), cfg)
        assert r.passed is True

    def test_none_cfg_uses_defaults(self):
        r = validate_patch_pair(0, 1, self._patches(), self._patches(), cfg=None)
        assert isinstance(r, PatchValidResult)

    def test_zero_score_on_empty(self):
        r = validate_patch_pair(0, 1, [], [])
        assert r.score.total_score == pytest.approx(0.0)

    def test_tiny_patches_skipped(self):
        """Patches too small for min_patch_size should be skipped."""
        cfg = PatchValidConfig(min_patch_size=200)
        small = [np.zeros((2, 2), dtype=np.uint8)]
        r = validate_patch_pair(0, 1, small, small, cfg)
        assert r.n_patches == 0
        assert r.passed is False


# ─── batch_validate_patches ───────────────────────────────────────────────────

class TestBatchValidatePatchesExtra:
    def _map(self, ids, val=128, n=3):
        return {i: [_patch(val) for _ in range(n)] for i in ids}

    def test_returns_dict(self):
        res = batch_validate_patches([(0, 1)], self._map([0, 1]))
        assert isinstance(res, dict)

    def test_key_in_result(self):
        res = batch_validate_patches([(0, 1)], self._map([0, 1]))
        assert (0, 1) in res

    def test_multiple_pairs(self):
        pm = self._map([0, 1, 2])
        res = batch_validate_patches([(0, 1), (1, 2)], pm)
        assert len(res) == 2

    def test_missing_fragment_gives_zero_patches(self):
        pm = self._map([0])
        res = batch_validate_patches([(0, 99)], pm)
        assert res[(0, 99)].n_patches == 0

    def test_empty_pairs_empty_result(self):
        res = batch_validate_patches([], self._map([0, 1]))
        assert len(res) == 0

    def test_none_cfg(self):
        res = batch_validate_patches([(0, 1)], self._map([0, 1]), cfg=None)
        assert (0, 1) in res


# ─── filter_valid_pairs ───────────────────────────────────────────────────────

class TestFilterValidPairsExtra:
    def _result(self, fa, fb, passed):
        return PatchValidResult(
            fragment_a=fa, fragment_b=fb,
            score=_score(), n_patches=1, passed=passed,
        )

    def test_returns_list(self):
        res = {(0, 1): self._result(0, 1, True)}
        assert isinstance(filter_valid_pairs(res), list)

    def test_all_pass(self):
        res = {(0, 1): self._result(0, 1, True),
               (1, 2): self._result(1, 2, True)}
        assert len(filter_valid_pairs(res)) == 2

    def test_none_pass(self):
        res = {(0, 1): self._result(0, 1, False)}
        assert filter_valid_pairs(res) == []

    def test_partial_filter(self):
        res = {(0, 1): self._result(0, 1, True),
               (2, 3): self._result(2, 3, False)}
        assert filter_valid_pairs(res) == [(0, 1)]

    def test_empty_input(self):
        assert filter_valid_pairs({}) == []

    def test_returned_pair_key_matches(self):
        res = {(3, 7): self._result(3, 7, True)}
        pairs = filter_valid_pairs(res)
        assert (3, 7) in pairs
