"""Extra tests for puzzle_reconstruction/matching/patch_validator.py."""
from __future__ import annotations

import numpy as np
import pytest

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

def _patch(h=10, w=10, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _score(c=0.7, t=0.6, g=0.5, total=0.6, valid=True):
    return PatchScore(color_score=c, texture_score=t,
                      gradient_score=g, total_score=total, valid=valid)


# ─── PatchValidConfig ───────────────────────────────────────────────────────

class TestPatchValidConfigExtra:
    def test_defaults(self):
        cfg = PatchValidConfig()
        assert cfg.color_weight == pytest.approx(0.4)
        assert cfg.texture_weight == pytest.approx(0.3)
        assert cfg.gradient_weight == pytest.approx(0.3)
        assert cfg.min_patch_size == 4
        assert cfg.threshold == pytest.approx(0.5)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(color_weight=-0.1)

    def test_weight_above_one_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(texture_weight=1.5)

    def test_zero_min_patch_size_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(min_patch_size=0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=1.5)

    def test_weight_sum(self):
        cfg = PatchValidConfig()
        assert cfg.weight_sum == pytest.approx(1.0)

    def test_normalized_weights(self):
        cfg = PatchValidConfig()
        wc, wt, wg = cfg.normalized_weights()
        assert abs(wc + wt + wg - 1.0) < 1e-6

    def test_zero_weights_fallback(self):
        cfg = PatchValidConfig(color_weight=0.0, texture_weight=0.0,
                               gradient_weight=0.0)
        wc, wt, wg = cfg.normalized_weights()
        assert abs(wc + wt + wg - 1.0) < 1e-6


# ─── PatchScore ─────────────────────────────────────────────────────────────

class TestPatchScoreExtra:
    def test_fields_stored(self):
        s = _score()
        assert s.color_score == pytest.approx(0.7)

    def test_is_strong_true(self):
        s = _score(total=0.85)
        assert s.is_strong is True

    def test_is_strong_false(self):
        s = _score(total=0.5)
        assert s.is_strong is False

    def test_dominant_channel(self):
        s = PatchScore(color_score=0.9, texture_score=0.3,
                       gradient_score=0.5, total_score=0.6)
        assert s.dominant_channel == "color"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=1.5, texture_score=0.5,
                       gradient_score=0.5, total_score=0.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=-0.1, texture_score=0.5,
                       gradient_score=0.5, total_score=0.5)


# ─── PatchValidResult ───────────────────────────────────────────────────────

class TestPatchValidResultExtra:
    def test_pair_key_ordered(self):
        r = PatchValidResult(fragment_a=5, fragment_b=2,
                             score=_score(), n_patches=3, passed=True)
        assert r.pair_key == (2, 5)

    def test_avg_score(self):
        s = PatchScore(color_score=0.6, texture_score=0.3,
                       gradient_score=0.9, total_score=0.6)
        r = PatchValidResult(fragment_a=0, fragment_b=1,
                             score=s, n_patches=1, passed=True)
        assert r.avg_score == pytest.approx(0.6)

    def test_negative_n_patches_raises(self):
        with pytest.raises(ValueError):
            PatchValidResult(fragment_a=0, fragment_b=1,
                             score=_score(), n_patches=-1, passed=True)


# ─── compute_patch_score ────────────────────────────────────────────────────

class TestComputePatchScoreExtra:
    def test_identical_patches(self):
        p = _patch(10, 10, 128)
        s = compute_patch_score(p, p)
        assert 0.0 <= s.total_score <= 1.0
        assert s.color_score == pytest.approx(1.0, abs=0.01)

    def test_different_patches(self):
        a = _patch(10, 10, 0)
        b = _patch(10, 10, 255)
        s = compute_patch_score(a, b)
        assert 0.0 <= s.total_score <= 1.0

    def test_too_small_raises(self):
        a = np.zeros((1, 1), dtype=np.uint8)
        b = np.zeros((1, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_patch_score(a, b)

    def test_valid_flag(self):
        p = _patch(10, 10, 128)
        cfg = PatchValidConfig(threshold=0.0)
        s = compute_patch_score(p, p, cfg)
        assert s.valid is True


# ─── aggregate_patch_scores ─────────────────────────────────────────────────

class TestAggregatePatchScoresExtra:
    def test_single(self):
        s = _score(c=0.8, t=0.6, g=0.4, total=0.6)
        agg = aggregate_patch_scores([s])
        assert agg.color_score == pytest.approx(0.8)

    def test_mean_of_two(self):
        s1 = _score(c=0.4, t=0.4, g=0.4, total=0.4)
        s2 = _score(c=0.8, t=0.8, g=0.8, total=0.8)
        agg = aggregate_patch_scores([s1, s2])
        assert agg.color_score == pytest.approx(0.6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_patch_scores([])


# ─── validate_patch_pair ────────────────────────────────────────────────────

class TestValidatePatchPairExtra:
    def test_basic(self):
        pa = [_patch(10, 10, 128)]
        pb = [_patch(10, 10, 128)]
        r = validate_patch_pair(0, 1, pa, pb)
        assert isinstance(r, PatchValidResult)
        assert r.n_patches == 1

    def test_empty_patches(self):
        r = validate_patch_pair(0, 1, [], [])
        assert r.n_patches == 0
        assert r.passed is False

    def test_mismatched_lengths(self):
        pa = [_patch(), _patch()]
        pb = [_patch()]
        r = validate_patch_pair(0, 1, pa, pb)
        # Should use min(len, len) = 1
        assert r.n_patches == 1


# ─── batch_validate_patches ─────────────────────────────────────────────────

class TestBatchValidateExtra:
    def test_basic(self):
        patch_map = {
            0: [_patch(10, 10, 128)],
            1: [_patch(10, 10, 128)],
        }
        results = batch_validate_patches([(0, 1)], patch_map)
        assert (0, 1) in results

    def test_missing_fragments(self):
        results = batch_validate_patches([(0, 1)], {})
        assert (0, 1) in results
        assert results[(0, 1)].n_patches == 0


# ─── filter_valid_pairs ─────────────────────────────────────────────────────

class TestFilterValidPairsExtra:
    def test_basic(self):
        s_valid = _score(total=0.8, valid=True)
        s_invalid = _score(total=0.2, valid=False)
        r1 = PatchValidResult(0, 1, s_valid, 1, True)
        r2 = PatchValidResult(2, 3, s_invalid, 1, False)
        valid = filter_valid_pairs({(0, 1): r1, (2, 3): r2})
        assert (0, 1) in valid
        assert (2, 3) not in valid

    def test_empty(self):
        assert filter_valid_pairs({}) == []
