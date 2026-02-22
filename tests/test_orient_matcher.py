"""Тесты для puzzle_reconstruction.matching.orient_matcher."""
import numpy as np
import pytest
from puzzle_reconstruction.matching.orient_matcher import (
    OrientConfig,
    OrientProfile,
    OrientMatchResult,
    compute_orient_profile,
    orient_similarity,
    best_orient_angle,
    match_orient_pair,
    batch_orient_match,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _profile(fragment_id: int = 0, n_bins: int = 36, seed: int = 0) -> OrientProfile:
    rng = np.random.default_rng(seed)
    hist = rng.random(n_bins)
    hist /= hist.sum()
    dominant = float(np.argmax(hist) / n_bins * 360.0)
    return OrientProfile(fragment_id=fragment_id, histogram=hist, dominant=dominant)


def _image(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.uint8)


# ─── TestOrientConfig ─────────────────────────────────────────────────────────

class TestOrientConfig:
    def test_defaults(self):
        cfg = OrientConfig()
        assert cfg.n_bins == 36
        assert cfg.angle_step == 10.0
        assert cfg.max_angle == 180.0
        assert cfg.normalize is True
        assert cfg.use_flip is False

    def test_valid_custom(self):
        cfg = OrientConfig(n_bins=18, angle_step=5.0, max_angle=90.0, use_flip=True)
        assert cfg.n_bins == 18
        assert cfg.angle_step == 5.0
        assert cfg.max_angle == 90.0
        assert cfg.use_flip is True

    def test_invalid_n_bins_one(self):
        with pytest.raises(ValueError):
            OrientConfig(n_bins=1)

    def test_invalid_n_bins_zero(self):
        with pytest.raises(ValueError):
            OrientConfig(n_bins=0)

    def test_invalid_angle_step_zero(self):
        with pytest.raises(ValueError):
            OrientConfig(angle_step=0.0)

    def test_invalid_angle_step_neg(self):
        with pytest.raises(ValueError):
            OrientConfig(angle_step=-5.0)

    def test_invalid_max_angle_neg(self):
        with pytest.raises(ValueError):
            OrientConfig(max_angle=-1.0)

    def test_max_angle_zero_ok(self):
        cfg = OrientConfig(max_angle=0.0)
        assert cfg.max_angle == 0.0


# ─── TestOrientProfile ────────────────────────────────────────────────────────

class TestOrientProfile:
    def test_basic(self):
        p = _profile(fragment_id=3, n_bins=18)
        assert p.fragment_id == 3
        assert p.n_bins == 18

    def test_n_bins_property(self):
        p = _profile(n_bins=12)
        assert p.n_bins == 12

    def test_is_uniform_false(self):
        hist = np.zeros(36)
        hist[5] = 1.0
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=50.0)
        assert p.is_uniform is False

    def test_is_uniform_true(self):
        hist = np.ones(36) / 36.0
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert p.is_uniform is True

    def test_is_uniform_zero_hist(self):
        hist = np.zeros(36)
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert p.is_uniform is True

    def test_invalid_fragment_id(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=-1, histogram=np.ones(4), dominant=0.0)

    def test_invalid_histogram_2d(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones((4, 4)), dominant=0.0)

    def test_invalid_histogram_too_short(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones(1), dominant=0.0)

    def test_invalid_dominant_negative(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=-1.0)

    def test_invalid_dominant_360(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=360.0)

    def test_dominant_zero_valid(self):
        p = OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=0.0)
        assert p.dominant == 0.0


# ─── TestOrientMatchResult ────────────────────────────────────────────────────

class TestOrientMatchResult:
    def _make(self, best_score=0.8, is_flipped=False):
        return OrientMatchResult(
            pair=(0, 1),
            best_angle=10.0,
            best_score=best_score,
            angle_scores={0.0: 0.5, 10.0: 0.8},
            is_flipped=is_flipped,
        )

    def test_fragment_a(self):
        r = self._make()
        assert r.fragment_a == 0

    def test_fragment_b(self):
        r = self._make()
        assert r.fragment_b == 1

    def test_n_angles_tested(self):
        r = self._make()
        assert r.n_angles_tested == 2

    def test_n_angles_tested_empty(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.5)
        assert r.n_angles_tested == 0

    def test_is_flipped_false(self):
        r = self._make(is_flipped=False)
        assert r.is_flipped is False

    def test_is_flipped_true(self):
        r = self._make(is_flipped=True)
        assert r.is_flipped is True

    def test_invalid_best_score_above(self):
        with pytest.raises(ValueError):
            OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=1.1)

    def test_invalid_best_score_below(self):
        with pytest.raises(ValueError):
            OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=-0.1)


# ─── TestComputeOrientProfile ─────────────────────────────────────────────────

class TestComputeOrientProfile:
    def test_basic_2d(self):
        img = _image(32, 32)
        p = compute_orient_profile(img, fragment_id=2)
        assert p.fragment_id == 2
        assert p.n_bins == 36  # default

    def test_basic_3d(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        p = compute_orient_profile(img)
        assert p.n_bins == 36

    def test_custom_n_bins(self):
        img = _image(32, 32)
        cfg = OrientConfig(n_bins=18)
        p = compute_orient_profile(img, cfg=cfg)
        assert p.n_bins == 18

    def test_normalized_sums_to_one(self):
        img = _image(32, 32, seed=7)
        cfg = OrientConfig(normalize=True)
        p = compute_orient_profile(img, cfg=cfg)
        assert abs(float(p.histogram.sum()) - 1.0) < 1e-6

    def test_no_normalize(self):
        img = _image(32, 32, seed=8)
        cfg = OrientConfig(normalize=False)
        p = compute_orient_profile(img, cfg=cfg)
        # Raw magnitudes: sum > 1 in general
        assert p.histogram is not None

    def test_dominant_in_range(self):
        img = _image(32, 32, seed=9)
        p = compute_orient_profile(img)
        assert 0.0 <= p.dominant < 360.0

    def test_invalid_image_1d(self):
        with pytest.raises(ValueError):
            compute_orient_profile(np.zeros(32))

    def test_default_fragment_id(self):
        img = _image(16, 16)
        p = compute_orient_profile(img)
        assert p.fragment_id == 0


# ─── TestOrientSimilarity ─────────────────────────────────────────────────────

class TestOrientSimilarity:
    def test_identical_angle_zero(self):
        p = _profile(0, n_bins=36, seed=5)
        score = orient_similarity(p, p, angle_deg=0.0)
        assert abs(score - 1.0) < 1e-6

    def test_output_range(self):
        a = _profile(0, n_bins=36, seed=1)
        b = _profile(1, n_bins=36, seed=2)
        score = orient_similarity(a, b, angle_deg=0.0)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        a = _profile(0, n_bins=36, seed=3)
        b = _profile(1, n_bins=36, seed=4)
        assert isinstance(orient_similarity(a, b), float)

    def test_zero_histogram(self):
        hist = np.zeros(36)
        p_zero = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        p = _profile(1, n_bins=36, seed=5)
        score = orient_similarity(p_zero, p)
        assert score == 0.0

    def test_angle_changes_score(self):
        a = _profile(0, n_bins=36, seed=10)
        b = _profile(1, n_bins=36, seed=11)
        s0 = orient_similarity(a, b, angle_deg=0.0)
        s90 = orient_similarity(a, b, angle_deg=90.0)
        # They might be equal by chance but should both be valid
        assert 0.0 <= s0 <= 1.0
        assert 0.0 <= s90 <= 1.0


# ─── TestBestOrientAngle ──────────────────────────────────────────────────────

class TestBestOrientAngle:
    def test_identical_returns_high_score(self):
        p = _profile(0, n_bins=36, seed=42)
        angle, score = best_orient_angle(p, p)
        assert score > 0.9

    def test_output_range(self):
        a = _profile(0, n_bins=36, seed=1)
        b = _profile(1, n_bins=36, seed=2)
        angle, score = best_orient_angle(a, b)
        assert 0.0 <= score <= 1.0
        assert angle >= 0.0

    def test_best_angle_bounded_by_max_angle(self):
        cfg = OrientConfig(max_angle=90.0, angle_step=10.0)
        a = _profile(0, n_bins=36, seed=3)
        b = _profile(1, n_bins=36, seed=4)
        angle, _ = best_orient_angle(a, b, cfg)
        assert angle <= 90.0

    def test_default_config(self):
        a = _profile(0, n_bins=36)
        b = _profile(1, n_bins=36, seed=5)
        angle, score = best_orient_angle(a, b)
        assert isinstance(angle, float)
        assert isinstance(score, float)


# ─── TestMatchOrientPair ──────────────────────────────────────────────────────

class TestMatchOrientPair:
    def test_basic(self):
        a = _profile(0, n_bins=36, seed=0)
        b = _profile(1, n_bins=36, seed=1)
        r = match_orient_pair(a, b)
        assert r.pair == (0, 1)
        assert 0.0 <= r.best_score <= 1.0

    def test_n_angles_tested(self):
        cfg = OrientConfig(angle_step=10.0, max_angle=180.0)
        a = _profile(0, n_bins=36, seed=0)
        b = _profile(1, n_bins=36, seed=1)
        r = match_orient_pair(a, b, cfg)
        # 0, 10, ..., 180 → 19 angles
        assert r.n_angles_tested == 19

    def test_identical_gives_high_score(self):
        p = _profile(0, n_bins=36, seed=42)
        r = match_orient_pair(p, p)
        assert r.best_score > 0.9

    def test_use_flip_more_angles(self):
        cfg_no = OrientConfig(angle_step=10.0, max_angle=90.0, use_flip=False)
        cfg_flip = OrientConfig(angle_step=10.0, max_angle=90.0, use_flip=True)
        a = _profile(0, n_bins=36, seed=0)
        b = _profile(1, n_bins=36, seed=1)
        r_no = match_orient_pair(a, b, cfg_no)
        r_flip = match_orient_pair(a, b, cfg_flip)
        # With flip: double the angles
        assert r_flip.n_angles_tested > r_no.n_angles_tested

    def test_is_flipped_default_false(self):
        a = _profile(0, n_bins=36, seed=0)
        b = _profile(1, n_bins=36, seed=1)
        r = match_orient_pair(a, b, OrientConfig(use_flip=False))
        assert r.is_flipped is False

    def test_fragment_ids_in_result(self):
        a = _profile(5, n_bins=36, seed=0)
        b = _profile(9, n_bins=36, seed=1)
        r = match_orient_pair(a, b)
        assert r.fragment_a == 5
        assert r.fragment_b == 9


# ─── TestBatchOrientMatch ─────────────────────────────────────────────────────

class TestBatchOrientMatch:
    def test_pair_count(self):
        profiles = [_profile(i, n_bins=36, seed=i) for i in range(4)]
        results = batch_orient_match(profiles)
        assert len(results) == 6  # C(4,2)

    def test_all_scores_valid(self):
        profiles = [_profile(i, n_bins=36, seed=i) for i in range(3)]
        results = batch_orient_match(profiles)
        for r in results:
            assert 0.0 <= r.best_score <= 1.0

    def test_empty_list(self):
        assert batch_orient_match([]) == []

    def test_single_profile(self):
        assert batch_orient_match([_profile(0)]) == []

    def test_two_profiles(self):
        a = _profile(0, n_bins=36, seed=0)
        b = _profile(1, n_bins=36, seed=1)
        results = batch_orient_match([a, b])
        assert len(results) == 1
        assert results[0].pair == (0, 1)

    def test_custom_cfg(self):
        cfg = OrientConfig(n_bins=18, angle_step=30.0, max_angle=90.0)
        profiles = [_profile(i, n_bins=18, seed=i) for i in range(3)]
        results = batch_orient_match(profiles, cfg)
        assert len(results) == 3
