"""Тесты для puzzle_reconstruction.matching.patch_validator."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.patch_validator import (
    PatchScore,
    PatchValidConfig,
    PatchValidResult,
    aggregate_patch_scores,
    batch_validate_patches,
    compute_patch_score,
    filter_valid_pairs,
    validate_patch_pair,
)


def _patch(h=8, w=8, channels=None) -> np.ndarray:
    rng = np.random.default_rng(42)
    if channels:
        return rng.integers(0, 256, (h, w, channels), dtype=np.uint8)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ─── TestPatchValidConfig ─────────────────────────────────────────────────────

class TestPatchValidConfig:
    def test_defaults(self):
        cfg = PatchValidConfig()
        assert cfg.color_weight == pytest.approx(0.4)
        assert cfg.texture_weight == pytest.approx(0.3)
        assert cfg.gradient_weight == pytest.approx(0.3)
        assert cfg.min_patch_size == 4
        assert cfg.threshold == pytest.approx(0.5)

    def test_weight_sum(self):
        cfg = PatchValidConfig()
        assert cfg.weight_sum == pytest.approx(1.0)

    def test_normalized_weights_sum_to_one(self):
        cfg = PatchValidConfig(color_weight=0.5, texture_weight=0.3,
                               gradient_weight=0.2)
        wc, wt, wg = cfg.normalized_weights()
        assert wc + wt + wg == pytest.approx(1.0)

    def test_texture_weight_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(texture_weight=-0.1)

    def test_gradient_weight_above_one_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(gradient_weight=1.5)

    def test_min_patch_size_zero_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(min_patch_size=0)

    def test_threshold_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=1.1)


# ─── TestPatchScore ───────────────────────────────────────────────────────────

class TestPatchScore:
    def _make(self, total=0.7, valid=True) -> PatchScore:
        return PatchScore(color_score=0.8, texture_score=0.7,
                          gradient_score=0.6, total_score=total, valid=valid)

    def test_is_strong_true(self):
        ps = self._make(total=0.85)
        assert ps.is_strong is True

    def test_is_strong_false(self):
        ps = self._make(total=0.75)
        assert ps.is_strong is False

    def test_dominant_channel(self):
        ps = PatchScore(color_score=0.9, texture_score=0.5,
                        gradient_score=0.3, total_score=0.7)
        assert ps.dominant_channel == "color"

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=1.1, texture_score=0.5,
                       gradient_score=0.5, total_score=0.7)

    def test_total_score_neg_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=0.5, texture_score=0.5,
                       gradient_score=0.5, total_score=-0.1)


# ─── TestPatchValidResult ─────────────────────────────────────────────────────

class TestPatchValidResult:
    def _make(self, passed=True) -> PatchValidResult:
        score = PatchScore(color_score=0.8, texture_score=0.7,
                           gradient_score=0.6, total_score=0.7, valid=True)
        return PatchValidResult(fragment_a=0, fragment_b=1,
                                score=score, n_patches=3, passed=passed)

    def test_pair_key_ordered(self):
        r = self._make()
        assert r.pair_key == (0, 1)

    def test_pair_key_reverse_ordered(self):
        score = PatchScore(0.5, 0.5, 0.5, 0.5)
        r = PatchValidResult(fragment_a=5, fragment_b=2,
                             score=score, n_patches=1, passed=True)
        assert r.pair_key == (2, 5)

    def test_avg_score(self):
        score = PatchScore(color_score=0.6, texture_score=0.8,
                           gradient_score=0.7, total_score=0.7)
        r = PatchValidResult(fragment_a=0, fragment_b=1,
                             score=score, n_patches=1, passed=True)
        assert r.avg_score == pytest.approx((0.6 + 0.8 + 0.7) / 3.0)

    def test_n_patches_neg_raises(self):
        score = PatchScore(0.5, 0.5, 0.5, 0.5)
        with pytest.raises(ValueError):
            PatchValidResult(fragment_a=0, fragment_b=1,
                             score=score, n_patches=-1, passed=True)


# ─── TestComputePatchScore ────────────────────────────────────────────────────

class TestComputePatchScore:
    def test_returns_patch_score(self):
        pa = _patch()
        pb = _patch()
        result = compute_patch_score(pa, pb)
        assert isinstance(result, PatchScore)

    def test_scores_in_range(self):
        pa = _patch()
        pb = _patch()
        result = compute_patch_score(pa, pb)
        assert 0.0 <= result.total_score <= 1.0
        assert 0.0 <= result.color_score <= 1.0
        assert 0.0 <= result.texture_score <= 1.0
        assert 0.0 <= result.gradient_score <= 1.0

    def test_identical_patches_high_score(self):
        pa = _patch()
        result = compute_patch_score(pa, pa)
        assert result.color_score >= 0.5  # identical → similar

    def test_too_small_patch_raises(self):
        pa = np.zeros((1,), dtype=np.uint8)
        pb = np.zeros((1,), dtype=np.uint8)
        cfg = PatchValidConfig(min_patch_size=4)
        with pytest.raises(ValueError):
            compute_patch_score(pa, pb, cfg)

    def test_bgr_patches(self):
        pa = _patch(channels=3)
        pb = _patch(channels=3)
        result = compute_patch_score(pa, pb)
        assert isinstance(result, PatchScore)


# ─── TestAggregatePatchScores ─────────────────────────────────────────────────

class TestAggregatePatchScores:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_patch_scores([])

    def test_single_score(self):
        ps = PatchScore(0.8, 0.7, 0.6, 0.7)
        result = aggregate_patch_scores([ps])
        assert result.color_score == pytest.approx(0.8)

    def test_mean_aggregation(self):
        ps1 = PatchScore(0.6, 0.6, 0.6, 0.6)
        ps2 = PatchScore(0.8, 0.8, 0.8, 0.8)
        result = aggregate_patch_scores([ps1, ps2])
        assert result.color_score == pytest.approx(0.7)


# ─── TestValidatePatchPair ────────────────────────────────────────────────────

class TestValidatePatchPair:
    def test_returns_patch_valid_result(self):
        patches = [_patch(), _patch()]
        r = validate_patch_pair(0, 1, patches, patches)
        assert isinstance(r, PatchValidResult)

    def test_empty_patches_returns_failed(self):
        r = validate_patch_pair(0, 1, [], [])
        assert r.passed is False
        assert r.n_patches == 0

    def test_fragment_ids_stored(self):
        patches = [_patch()]
        r = validate_patch_pair(3, 7, patches, patches)
        assert r.fragment_a == 3
        assert r.fragment_b == 7

    def test_n_patches_correct(self):
        patches = [_patch(), _patch(), _patch()]
        r = validate_patch_pair(0, 1, patches, patches)
        assert r.n_patches == 3


# ─── TestBatchValidatePatches ─────────────────────────────────────────────────

class TestBatchValidatePatches:
    def test_returns_dict(self):
        patches = [_patch()]
        patch_map = {0: patches, 1: patches}
        result = batch_validate_patches([(0, 1)], patch_map)
        assert isinstance(result, dict)

    def test_keys_are_pairs(self):
        patches = [_patch()]
        patch_map = {0: patches, 1: patches, 2: patches}
        pairs = [(0, 1), (1, 2)]
        result = batch_validate_patches(pairs, patch_map)
        assert set(result.keys()) == {(0, 1), (1, 2)}

    def test_empty_pairs(self):
        result = batch_validate_patches([], {})
        assert result == {}


# ─── TestFilterValidPairs ─────────────────────────────────────────────────────

class TestFilterValidPairs:
    def test_returns_only_passed(self):
        score_pass = PatchScore(0.9, 0.9, 0.9, 0.9)
        score_fail = PatchScore(0.2, 0.2, 0.2, 0.2, valid=False)
        results = {
            (0, 1): PatchValidResult(0, 1, score_pass, 1, True),
            (1, 2): PatchValidResult(1, 2, score_fail, 1, False),
        }
        valid = filter_valid_pairs(results)
        assert (0, 1) in valid
        assert (1, 2) not in valid

    def test_empty_input(self):
        assert filter_valid_pairs({}) == []
