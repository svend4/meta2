"""Тесты для puzzle_reconstruction/matching/patch_validator.py."""
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


def _make_patch(h=8, w=8, val=128, channels=3):
    if channels == 1:
        return np.full((h, w), val, dtype=np.uint8)
    return np.full((h, w, channels), val, dtype=np.uint8)


class TestPatchValidConfig:
    def test_defaults(self):
        c = PatchValidConfig()
        assert c.color_weight == pytest.approx(0.4)
        assert c.threshold == pytest.approx(0.5)

    def test_negative_weights_raise(self):
        with pytest.raises(ValueError):
            PatchValidConfig(texture_weight=-0.1)

    def test_min_patch_size_zero_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(min_patch_size=0)

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=1.5)

    def test_weight_sum_property(self):
        c = PatchValidConfig(color_weight=0.4, texture_weight=0.3, gradient_weight=0.3)
        assert c.weight_sum == pytest.approx(1.0)

    def test_normalized_weights_sum_to_one(self):
        c = PatchValidConfig(color_weight=2.0, texture_weight=1.0, gradient_weight=1.0)
        wc, wt, wg = c.normalized_weights()
        assert wc + wt + wg == pytest.approx(1.0)


class TestPatchScore:
    def test_is_strong(self):
        ps = PatchScore(color_score=0.9, texture_score=0.9, gradient_score=0.9, total_score=0.9)
        assert ps.is_strong

    def test_not_strong_below_threshold(self):
        ps = PatchScore(color_score=0.5, texture_score=0.5, gradient_score=0.5, total_score=0.5)
        assert not ps.is_strong

    def test_dominant_channel(self):
        ps = PatchScore(color_score=0.9, texture_score=0.3, gradient_score=0.5, total_score=0.6)
        assert ps.dominant_channel == "color"

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=1.5, texture_score=0.5, gradient_score=0.5, total_score=0.5)


class TestPatchValidResult:
    def _make_score(self, val=0.5):
        return PatchScore(color_score=val, texture_score=val, gradient_score=val, total_score=val)

    def test_pair_key_ordered(self):
        r = PatchValidResult(fragment_a=5, fragment_b=2, score=self._make_score(),
                              n_patches=1, passed=True)
        assert r.pair_key == (2, 5)

    def test_avg_score(self):
        ps = PatchScore(color_score=0.6, texture_score=0.8, gradient_score=0.4, total_score=0.6)
        r = PatchValidResult(fragment_a=0, fragment_b=1, score=ps, n_patches=1, passed=True)
        assert r.avg_score == pytest.approx((0.6 + 0.8 + 0.4) / 3)

    def test_negative_n_patches_raises(self):
        with pytest.raises(ValueError):
            PatchValidResult(fragment_a=0, fragment_b=1, score=self._make_score(),
                              n_patches=-1, passed=False)


class TestComputePatchScore:
    def test_identical_patches_high_score(self):
        patch = _make_patch(val=100)
        ps = compute_patch_score(patch, patch)
        assert ps.total_score > 0.7

    def test_score_in_range(self):
        pa = _make_patch(val=10)
        pb = _make_patch(val=240)
        ps = compute_patch_score(pa, pb)
        assert 0.0 <= ps.total_score <= 1.0

    def test_small_patch_raises(self):
        cfg = PatchValidConfig(min_patch_size=100)
        patch = _make_patch(h=2, w=2)
        with pytest.raises(ValueError):
            compute_patch_score(patch, patch, cfg=cfg)

    def test_valid_flag_reflects_threshold(self):
        patch = _make_patch(val=128)
        cfg = PatchValidConfig(threshold=0.0)
        ps = compute_patch_score(patch, patch, cfg=cfg)
        assert ps.valid


class TestAggregatePatchScores:
    def test_average_of_two(self):
        p1 = PatchScore(color_score=0.6, texture_score=0.6, gradient_score=0.6, total_score=0.6)
        p2 = PatchScore(color_score=0.8, texture_score=0.8, gradient_score=0.8, total_score=0.8)
        agg = aggregate_patch_scores([p1, p2])
        assert agg.color_score == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_patch_scores([])


class TestValidatePatchPair:
    def test_returns_valid_result_with_patches(self):
        patches = [_make_patch(val=128)]
        r = validate_patch_pair(0, 1, patches, patches)
        assert isinstance(r, PatchValidResult)
        assert r.n_patches == 1

    def test_empty_patches_returns_failed(self):
        r = validate_patch_pair(0, 1, [], [])
        assert not r.passed and r.n_patches == 0


class TestBatchValidatePatches:
    def test_returns_dict_with_all_pairs(self):
        patch_map = {
            0: [_make_patch(val=100)],
            1: [_make_patch(val=100)],
            2: [_make_patch(val=200)],
        }
        results = batch_validate_patches([(0, 1), (0, 2), (1, 2)], patch_map)
        assert len(results) == 3

    def test_missing_patch_returns_failed(self):
        results = batch_validate_patches([(0, 99)], {0: [_make_patch()]})
        assert not results[(0, 99)].passed


class TestFilterValidPairs:
    def test_only_passed_returned(self):
        def _ps(val, passed):
            return PatchValidResult(
                fragment_a=0, fragment_b=1,
                score=PatchScore(color_score=val, texture_score=val,
                                 gradient_score=val, total_score=val),
                n_patches=1, passed=passed)
        results = {(0, 1): _ps(0.8, True), (1, 2): _ps(0.2, False)}
        assert filter_valid_pairs(results) == [(0, 1)]
