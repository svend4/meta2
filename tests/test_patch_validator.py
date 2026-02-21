"""Тесты для puzzle_reconstruction.matching.patch_validator."""
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

def _patch(h: int = 8, w: int = 8, val: int = 128, seed: int = 0,
           noise: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w), val, dtype=np.uint8)
    delta = rng.integers(-noise, noise + 1, (h, w))
    return np.clip(base.astype(int) + delta, 0, 255).astype(np.uint8)


def _identical_patches():
    p = _patch()
    return p, p.copy()


def _dissimilar_patches():
    return np.zeros((8, 8), dtype=np.uint8), np.full((8, 8), 255, dtype=np.uint8)


def _score(c=0.8, t=0.7, g=0.6, total=0.7, valid=True):
    return PatchScore(color_score=c, texture_score=t, gradient_score=g,
                      total_score=total, valid=valid)


# ─── TestPatchValidConfig ─────────────────────────────────────────────────────

class TestPatchValidConfig:
    def test_defaults(self):
        cfg = PatchValidConfig()
        assert cfg.color_weight == pytest.approx(0.4)
        assert cfg.texture_weight == pytest.approx(0.3)
        assert cfg.gradient_weight == pytest.approx(0.3)
        assert cfg.min_patch_size == 4
        assert cfg.threshold == pytest.approx(0.5)

    def test_valid_custom(self):
        cfg = PatchValidConfig(color_weight=0.5, texture_weight=0.3,
                               gradient_weight=0.2, min_patch_size=8,
                               threshold=0.6)
        assert cfg.min_patch_size == 8
        assert cfg.threshold == pytest.approx(0.6)

    def test_color_weight_zero_ok(self):
        cfg = PatchValidConfig(color_weight=0.0)
        assert cfg.color_weight == 0.0

    def test_color_weight_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(color_weight=-0.1)

    def test_color_weight_above_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(color_weight=1.1)

    def test_texture_weight_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(texture_weight=-0.01)

    def test_gradient_weight_above_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(gradient_weight=1.2)

    def test_min_patch_size_one_ok(self):
        cfg = PatchValidConfig(min_patch_size=1)
        assert cfg.min_patch_size == 1

    def test_min_patch_size_zero_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(min_patch_size=0)

    def test_min_patch_size_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(min_patch_size=-1)

    def test_threshold_zero_ok(self):
        cfg = PatchValidConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_one_ok(self):
        cfg = PatchValidConfig(threshold=1.0)
        assert cfg.threshold == 1.0

    def test_threshold_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=-0.1)

    def test_threshold_above_raises(self):
        with pytest.raises(ValueError):
            PatchValidConfig(threshold=1.1)

    def test_weight_sum(self):
        cfg = PatchValidConfig(color_weight=0.5, texture_weight=0.3,
                               gradient_weight=0.2)
        assert cfg.weight_sum == pytest.approx(1.0)

    def test_normalized_weights_sum_to_one(self):
        cfg = PatchValidConfig(color_weight=2.0, texture_weight=1.0,
                               gradient_weight=1.0)
        wc, wt, wg = cfg.normalized_weights()
        assert wc + wt + wg == pytest.approx(1.0)

    def test_normalized_weights_zero_sum_fallback(self):
        cfg = PatchValidConfig(color_weight=0.0, texture_weight=0.0,
                               gradient_weight=0.0)
        wc, wt, wg = cfg.normalized_weights()
        assert wc + wt + wg == pytest.approx(1.0)


# ─── TestPatchScore ───────────────────────────────────────────────────────────

class TestPatchScore:
    def test_basic(self):
        s = _score()
        assert s.color_score == pytest.approx(0.8)
        assert s.valid is True

    def test_is_strong_true(self):
        s = _score(total=0.9)
        assert s.is_strong is True

    def test_is_strong_false(self):
        s = _score(total=0.7)
        assert s.is_strong is False

    def test_is_strong_boundary(self):
        s = _score(total=0.8)
        # 0.8 is NOT > 0.8
        assert s.is_strong is False

    def test_dominant_channel_color(self):
        s = PatchScore(color_score=0.9, texture_score=0.5,
                       gradient_score=0.4, total_score=0.7)
        assert s.dominant_channel == "color"

    def test_dominant_channel_texture(self):
        s = PatchScore(color_score=0.3, texture_score=0.9,
                       gradient_score=0.5, total_score=0.6)
        assert s.dominant_channel == "texture"

    def test_dominant_channel_gradient(self):
        s = PatchScore(color_score=0.2, texture_score=0.3,
                       gradient_score=0.8, total_score=0.5)
        assert s.dominant_channel == "gradient"

    def test_invalid_color_neg_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=-0.1, texture_score=0.5,
                       gradient_score=0.5, total_score=0.5)

    def test_invalid_texture_above_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=0.5, texture_score=1.1,
                       gradient_score=0.5, total_score=0.5)

    def test_invalid_gradient_neg_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=0.5, texture_score=0.5,
                       gradient_score=-0.1, total_score=0.5)

    def test_invalid_total_above_raises(self):
        with pytest.raises(ValueError):
            PatchScore(color_score=0.5, texture_score=0.5,
                       gradient_score=0.5, total_score=1.5)

    def test_all_zero_ok(self):
        s = PatchScore(color_score=0.0, texture_score=0.0,
                       gradient_score=0.0, total_score=0.0)
        assert s.total_score == 0.0

    def test_all_one_ok(self):
        s = PatchScore(color_score=1.0, texture_score=1.0,
                       gradient_score=1.0, total_score=1.0)
        assert s.total_score == 1.0


# ─── TestPatchValidResult ─────────────────────────────────────────────────────

class TestPatchValidResult:
    def _make(self, a=0, b=1, n=5, passed=True):
        return PatchValidResult(fragment_a=a, fragment_b=b,
                                score=_score(), n_patches=n, passed=passed)

    def test_basic(self):
        r = self._make()
        assert r.fragment_a == 0
        assert r.fragment_b == 1
        assert r.n_patches == 5

    def test_pair_key_ordered(self):
        r = self._make(a=3, b=1)
        assert r.pair_key == (1, 3)

    def test_pair_key_already_ordered(self):
        r = self._make(a=0, b=5)
        assert r.pair_key == (0, 5)

    def test_avg_score(self):
        r = self._make()
        expected = (0.8 + 0.7 + 0.6) / 3.0
        assert r.avg_score == pytest.approx(expected)

    def test_passed_stored(self):
        r = self._make(passed=False)
        assert r.passed is False

    def test_n_patches_zero_ok(self):
        r = PatchValidResult(fragment_a=0, fragment_b=1,
                             score=_score(), n_patches=0, passed=False)
        assert r.n_patches == 0

    def test_n_patches_neg_raises(self):
        with pytest.raises(ValueError):
            PatchValidResult(fragment_a=0, fragment_b=1,
                             score=_score(), n_patches=-1, passed=False)


# ─── TestComputePatchScore ────────────────────────────────────────────────────

class TestComputePatchScore:
    def test_returns_patch_score(self):
        a, b = _identical_patches()
        s = compute_patch_score(a, b)
        assert isinstance(s, PatchScore)

    def test_identical_patches_high_score(self):
        a, b = _identical_patches()
        s = compute_patch_score(a, b)
        assert s.total_score > 0.7

    def test_dissimilar_patches_low_score(self):
        a, b = _dissimilar_patches()
        s = compute_patch_score(a, b)
        assert s.total_score < 0.8

    def test_scores_in_range(self):
        a, b = _patch(), _patch(seed=5)
        s = compute_patch_score(a, b)
        for val in (s.color_score, s.texture_score,
                    s.gradient_score, s.total_score):
            assert 0.0 <= val <= 1.0

    def test_threshold_determines_valid(self):
        a, b = _identical_patches()
        cfg_low = PatchValidConfig(threshold=0.0)
        cfg_high = PatchValidConfig(threshold=1.0)
        s_low = compute_patch_score(a, b, cfg_low)
        s_high = compute_patch_score(a, b, cfg_high)
        assert s_low.valid is True
        assert s_high.valid is False

    def test_too_small_patch_raises(self):
        a = np.zeros((1, 1), dtype=np.uint8)
        b = np.zeros((8, 8), dtype=np.uint8)
        cfg = PatchValidConfig(min_patch_size=4)
        with pytest.raises(ValueError):
            compute_patch_score(a, b, cfg)

    def test_3d_patches_ok(self):
        a = np.zeros((8, 8, 3), dtype=np.uint8)
        b = np.zeros((8, 8, 3), dtype=np.uint8)
        s = compute_patch_score(a, b)
        assert isinstance(s, PatchScore)

    def test_weighted_combination(self):
        # Все веса на цвет → total ≈ color_score
        a, b = _identical_patches()
        cfg = PatchValidConfig(color_weight=1.0, texture_weight=0.0,
                               gradient_weight=0.0)
        s = compute_patch_score(a, b, cfg)
        assert s.total_score == pytest.approx(s.color_score, abs=0.01)

    def test_different_seeds_differ(self):
        a = _patch(seed=0)
        b = _patch(seed=99, val=50)
        s = compute_patch_score(a, b)
        assert isinstance(s, PatchScore)


# ─── TestAggregatePatchScores ─────────────────────────────────────────────────

class TestAggregatePatchScores:
    def _scores(self, n=3):
        return [_score(c=0.8, t=0.7, g=0.6, total=0.7) for _ in range(n)]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_patch_scores([])

    def test_single_score_identity(self):
        s = _score(c=0.8, t=0.7, g=0.6, total=0.7)
        agg = aggregate_patch_scores([s])
        assert agg.color_score == pytest.approx(0.8)
        assert agg.texture_score == pytest.approx(0.7)

    def test_mean_of_two(self):
        s1 = _score(c=0.6, t=0.6, g=0.6, total=0.6)
        s2 = _score(c=1.0, t=1.0, g=1.0, total=1.0)
        agg = aggregate_patch_scores([s1, s2])
        assert agg.color_score == pytest.approx(0.8)

    def test_returns_patch_score(self):
        agg = aggregate_patch_scores(self._scores())
        assert isinstance(agg, PatchScore)

    def test_total_in_range(self):
        scores = [_score(total=v * 0.1) for v in range(1, 6)]
        agg = aggregate_patch_scores(scores)
        assert 0.0 <= agg.total_score <= 1.0

    def test_valid_respects_threshold(self):
        scores = [_score(total=0.4)] * 3
        cfg_lo = PatchValidConfig(threshold=0.3)
        cfg_hi = PatchValidConfig(threshold=0.9)
        agg_lo = aggregate_patch_scores(scores, cfg_lo)
        agg_hi = aggregate_patch_scores(scores, cfg_hi)
        assert agg_lo.valid is True
        assert agg_hi.valid is False

    def test_large_list(self):
        scores = [_score() for _ in range(100)]
        agg = aggregate_patch_scores(scores)
        assert isinstance(agg, PatchScore)


# ─── TestValidatePatchPair ────────────────────────────────────────────────────

class TestValidatePatchPair:
    def _patches(self, n=3, val=128, seed=0):
        return [_patch(seed=seed + i, val=val) for i in range(n)]

    def test_returns_result(self):
        pa = self._patches()
        pb = self._patches(seed=10)
        r = validate_patch_pair(0, 1, pa, pb)
        assert isinstance(r, PatchValidResult)

    def test_fragment_ids_stored(self):
        r = validate_patch_pair(3, 7, self._patches(), self._patches(seed=10))
        assert r.fragment_a == 3
        assert r.fragment_b == 7

    def test_n_patches_min_of_two_lists(self):
        pa = self._patches(n=5)
        pb = self._patches(n=3, seed=10)
        r = validate_patch_pair(0, 1, pa, pb)
        assert r.n_patches <= 3

    def test_empty_patches_returns_not_passed(self):
        r = validate_patch_pair(0, 1, [], [])
        assert r.passed is False
        assert r.n_patches == 0

    def test_one_empty_list(self):
        r = validate_patch_pair(0, 1, self._patches(), [])
        assert r.passed is False

    def test_identical_patches_high_score(self):
        p = _patch()
        pa = [p.copy() for _ in range(3)]
        pb = [p.copy() for _ in range(3)]
        cfg = PatchValidConfig(threshold=0.5)
        r = validate_patch_pair(0, 1, pa, pb, cfg)
        assert r.score.total_score > 0.5

    def test_threshold_zero_always_passes(self):
        cfg = PatchValidConfig(threshold=0.0)
        pa = self._patches()
        pb = self._patches(seed=50, val=50)
        r = validate_patch_pair(0, 1, pa, pb, cfg)
        assert r.passed is True

    def test_too_small_patches_skipped(self):
        tiny = [np.zeros((1, 1), dtype=np.uint8)]
        r = validate_patch_pair(0, 1, tiny, tiny)
        assert r.n_patches == 0


# ─── TestBatchValidatePatches ─────────────────────────────────────────────────

class TestBatchValidatePatches:
    def _map(self):
        return {
            0: [_patch(seed=i) for i in range(3)],
            1: [_patch(seed=i + 10) for i in range(3)],
            2: [_patch(seed=i + 20) for i in range(3)],
        }

    def test_returns_dict(self):
        results = batch_validate_patches([(0, 1), (1, 2)], self._map())
        assert isinstance(results, dict)

    def test_keys_match_pairs(self):
        pairs = [(0, 1), (1, 2)]
        results = batch_validate_patches(pairs, self._map())
        assert set(results.keys()) == {(0, 1), (1, 2)}

    def test_all_values_are_results(self):
        results = batch_validate_patches([(0, 1)], self._map())
        for v in results.values():
            assert isinstance(v, PatchValidResult)

    def test_empty_pairs(self):
        results = batch_validate_patches([], self._map())
        assert results == {}

    def test_missing_fragment_empty_patches(self):
        results = batch_validate_patches([(0, 99)], self._map())
        r = results[(0, 99)]
        assert r.n_patches == 0
        assert r.passed is False

    def test_consistent_fragment_ids(self):
        results = batch_validate_patches([(0, 1), (1, 2)], self._map())
        r01 = results[(0, 1)]
        assert r01.fragment_a == 0
        assert r01.fragment_b == 1


# ─── TestFilterValidPairs ─────────────────────────────────────────────────────

class TestFilterValidPairs:
    def _results(self):
        s = _score(total=0.8, valid=True)
        s_fail = _score(total=0.2, valid=False)
        return {
            (0, 1): PatchValidResult(0, 1, s, 3, passed=True),
            (0, 2): PatchValidResult(0, 2, s_fail, 3, passed=False),
            (1, 2): PatchValidResult(1, 2, s, 2, passed=True),
        }

    def test_returns_list(self):
        valid = filter_valid_pairs(self._results())
        assert isinstance(valid, list)

    def test_only_passed_pairs(self):
        valid = filter_valid_pairs(self._results())
        assert (0, 1) in valid
        assert (1, 2) in valid
        assert (0, 2) not in valid

    def test_count_correct(self):
        valid = filter_valid_pairs(self._results())
        assert len(valid) == 2

    def test_empty_results(self):
        assert filter_valid_pairs({}) == []

    def test_all_failed(self):
        s_fail = _score(total=0.2, valid=False)
        results = {
            (0, 1): PatchValidResult(0, 1, s_fail, 2, passed=False),
        }
        assert filter_valid_pairs(results) == []

    def test_all_passed(self):
        s = _score(total=0.9, valid=True)
        results = {
            (i, i + 1): PatchValidResult(i, i + 1, s, 3, passed=True)
            for i in range(5)
        }
        valid = filter_valid_pairs(results)
        assert len(valid) == 5
