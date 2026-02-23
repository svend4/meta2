"""Extra tests for puzzle_reconstruction/matching/orient_matcher.py"""
import numpy as np
import pytest

from puzzle_reconstruction.matching.orient_matcher import (
    OrientConfig,
    OrientMatchResult,
    OrientProfile,
    batch_orient_match,
    best_orient_angle,
    compute_orient_profile,
    match_orient_pair,
    orient_similarity,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _profile(fragment_id: int = 0, n_bins: int = 36, seed: int = 0) -> OrientProfile:
    rng = np.random.default_rng(seed)
    hist = rng.random(n_bins).astype(np.float32)
    hist /= hist.sum()
    dominant = float(np.argmax(hist) / n_bins * 360.0)
    return OrientProfile(fragment_id=fragment_id, histogram=hist, dominant=dominant)


def _image(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.uint8)


# ─── TestOrientConfigExtra ────────────────────────────────────────────────────

class TestOrientConfigExtra:
    def test_n_bins_2_valid(self):
        cfg = OrientConfig(n_bins=2)
        assert cfg.n_bins == 2

    def test_n_bins_72(self):
        cfg = OrientConfig(n_bins=72)
        assert cfg.n_bins == 72

    def test_angle_step_large(self):
        cfg = OrientConfig(angle_step=45.0)
        assert cfg.angle_step == pytest.approx(45.0)

    def test_max_angle_360(self):
        cfg = OrientConfig(max_angle=360.0)
        assert cfg.max_angle == pytest.approx(360.0)

    def test_normalize_false(self):
        cfg = OrientConfig(normalize=False)
        assert cfg.normalize is False

    def test_normalize_true(self):
        cfg = OrientConfig(normalize=True)
        assert cfg.normalize is True

    def test_use_flip_true(self):
        cfg = OrientConfig(use_flip=True)
        assert cfg.use_flip is True

    def test_angle_step_small_valid(self):
        cfg = OrientConfig(angle_step=0.1)
        assert cfg.angle_step == pytest.approx(0.1)


# ─── TestOrientProfileExtra ───────────────────────────────────────────────────

class TestOrientProfileExtra:
    def test_fragment_id_large(self):
        p = _profile(fragment_id=9999)
        assert p.fragment_id == 9999

    def test_fragment_id_zero(self):
        p = _profile(fragment_id=0)
        assert p.fragment_id == 0

    def test_n_bins_18(self):
        p = _profile(n_bins=18)
        assert p.n_bins == 18

    def test_n_bins_72(self):
        p = _profile(n_bins=72)
        assert p.n_bins == 72

    def test_histogram_normalized(self):
        p = _profile(n_bins=36, seed=7)
        assert abs(float(p.histogram.sum()) - 1.0) < 1e-5

    def test_dominant_359_valid(self):
        hist = np.ones(36) / 36.0
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=359.9)
        assert p.dominant == pytest.approx(359.9)

    def test_is_uniform_near_uniform(self):
        hist = np.ones(36) / 36.0 + 1e-7
        hist /= hist.sum()
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert isinstance(p.is_uniform, bool)

    def test_histogram_all_zeros_uniform(self):
        hist = np.zeros(36)
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert p.is_uniform is True

    def test_histogram_spike_not_uniform(self):
        hist = np.zeros(36)
        hist[10] = 1.0
        p = OrientProfile(fragment_id=0, histogram=hist, dominant=100.0)
        assert p.is_uniform is False

    def test_various_seeds_no_crash(self):
        for s in range(5):
            p = _profile(fragment_id=s, seed=s)
            assert p.n_bins == 36


# ─── TestOrientMatchResultExtra ───────────────────────────────────────────────

class TestOrientMatchResultExtra:
    def test_best_score_zero_valid(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.0)
        assert r.best_score == pytest.approx(0.0)

    def test_best_score_one_valid(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=1.0)
        assert r.best_score == pytest.approx(1.0)

    def test_pair_stored(self):
        r = OrientMatchResult(pair=(3, 7), best_angle=45.0, best_score=0.5)
        assert r.pair == (3, 7)

    def test_best_angle_stored(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=90.0, best_score=0.5)
        assert r.best_angle == pytest.approx(90.0)

    def test_angle_scores_four(self):
        scores = {0.0: 0.5, 30.0: 0.6, 60.0: 0.7, 90.0: 0.8}
        r = OrientMatchResult(pair=(0, 1), best_angle=90.0, best_score=0.8,
                              angle_scores=scores)
        assert r.n_angles_tested == 4

    def test_fragment_a_b_from_pair(self):
        r = OrientMatchResult(pair=(5, 9), best_angle=0.0, best_score=0.3)
        assert r.fragment_a == 5
        assert r.fragment_b == 9

    def test_is_flipped_stored(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.5,
                              is_flipped=True)
        assert r.is_flipped is True


# ─── TestComputeOrientProfileExtra ────────────────────────────────────────────

class TestComputeOrientProfileExtra:
    def test_small_image(self):
        img = _image(h=8, w=8, seed=1)
        p = compute_orient_profile(img)
        assert p.n_bins == 36

    def test_non_square_image(self):
        img = _image(h=16, w=32, seed=2)
        p = compute_orient_profile(img)
        assert p.n_bins == 36

    def test_fragment_id_preserved(self):
        img = _image(32, 32, seed=3)
        p = compute_orient_profile(img, fragment_id=7)
        assert p.fragment_id == 7

    def test_various_seeds_no_crash(self):
        for s in range(5):
            img = _image(32, 32, seed=s)
            p = compute_orient_profile(img)
            assert p.histogram is not None

    def test_n_bins_custom_12(self):
        img = _image(32, 32, seed=0)
        cfg = OrientConfig(n_bins=12)
        p = compute_orient_profile(img, cfg=cfg)
        assert p.n_bins == 12

    def test_dominant_nonneg(self):
        img = _image(32, 32, seed=5)
        p = compute_orient_profile(img)
        assert p.dominant >= 0.0

    def test_dominant_lt_360(self):
        img = _image(32, 32, seed=6)
        p = compute_orient_profile(img)
        assert p.dominant < 360.0

    def test_histogram_shape_matches_n_bins(self):
        img = _image(32, 32, seed=8)
        cfg = OrientConfig(n_bins=24)
        p = compute_orient_profile(img, cfg=cfg)
        assert p.histogram.shape == (24,)


# ─── TestOrientSimilarityExtra ────────────────────────────────────────────────

class TestOrientSimilarityExtra:
    def test_five_pairs_in_range(self):
        for s in range(5):
            a = _profile(0, seed=s)
            b = _profile(1, seed=s + 10)
            score = orient_similarity(a, b)
            assert 0.0 <= score <= 1.0

    def test_returns_float_type(self):
        a = _profile(0, seed=1)
        b = _profile(1, seed=2)
        assert isinstance(orient_similarity(a, b), float)

    def test_large_angle_deg(self):
        a = _profile(0, seed=3)
        b = _profile(1, seed=4)
        score = orient_similarity(a, b, angle_deg=180.0)
        assert 0.0 <= score <= 1.0

    def test_angle_90_valid(self):
        a = _profile(0, seed=5)
        b = _profile(1, seed=6)
        score = orient_similarity(a, b, angle_deg=90.0)
        assert 0.0 <= score <= 1.0

    def test_same_profile_high_score_at_0(self):
        p = _profile(0, seed=42)
        score = orient_similarity(p, p, angle_deg=0.0)
        assert score > 0.9


# ─── TestBestOrientAngleExtra ─────────────────────────────────────────────────

class TestBestOrientAngleExtra:
    def test_returns_tuple(self):
        a = _profile(0, seed=1)
        b = _profile(1, seed=2)
        result = best_orient_angle(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_in_range(self):
        for s in range(5):
            a = _profile(0, seed=s)
            b = _profile(1, seed=s + 10)
            angle, score = best_orient_angle(a, b)
            assert 0.0 <= score <= 1.0

    def test_angle_nonneg(self):
        a = _profile(0, seed=7)
        b = _profile(1, seed=8)
        angle, _ = best_orient_angle(a, b)
        assert angle >= 0.0

    def test_max_angle_30(self):
        cfg = OrientConfig(max_angle=30.0, angle_step=10.0)
        a = _profile(0, seed=3)
        b = _profile(1, seed=4)
        angle, _ = best_orient_angle(a, b, cfg)
        assert angle <= 30.0

    def test_identical_score_near_1(self):
        p = _profile(0, seed=99)
        angle, score = best_orient_angle(p, p)
        assert score > 0.9


# ─── TestMatchOrientPairExtra ─────────────────────────────────────────────────

class TestMatchOrientPairExtra:
    def test_five_random_pairs(self):
        for s in range(5):
            a = _profile(0, seed=s)
            b = _profile(1, seed=s + 10)
            r = match_orient_pair(a, b)
            assert 0.0 <= r.best_score <= 1.0

    def test_angle_step_30_n_tested(self):
        cfg = OrientConfig(angle_step=30.0, max_angle=180.0)
        a = _profile(0, seed=0)
        b = _profile(1, seed=1)
        r = match_orient_pair(a, b, cfg)
        # 0, 30, 60, 90, 120, 150, 180 → 7
        assert r.n_angles_tested == 7

    def test_best_angle_is_float(self):
        a = _profile(0, seed=2)
        b = _profile(1, seed=3)
        r = match_orient_pair(a, b)
        assert isinstance(r.best_angle, float)

    def test_pair_tuple_correct(self):
        a = _profile(3, seed=0)
        b = _profile(7, seed=1)
        r = match_orient_pair(a, b)
        assert r.pair == (3, 7)

    def test_no_flip_n_tested(self):
        cfg = OrientConfig(angle_step=45.0, max_angle=90.0, use_flip=False)
        a = _profile(0, seed=0)
        b = _profile(1, seed=1)
        r = match_orient_pair(a, b, cfg)
        # 0, 45, 90 → 3 angles
        assert r.n_angles_tested == 3


# ─── TestBatchOrientMatchExtra ────────────────────────────────────────────────

class TestBatchOrientMatchExtra:
    def test_five_images_10_pairs(self):
        profiles = [_profile(i, seed=i) for i in range(5)]
        results = batch_orient_match(profiles)
        assert len(results) == 10  # C(5,2)

    def test_all_results_are_orient_match_result(self):
        profiles = [_profile(i, seed=i) for i in range(3)]
        results = batch_orient_match(profiles)
        for r in results:
            assert isinstance(r, OrientMatchResult)

    def test_pairs_cover_all_combinations(self):
        profiles = [_profile(i, seed=i) for i in range(3)]
        results = batch_orient_match(profiles)
        pairs = {r.pair for r in results}
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_with_flip_config(self):
        cfg = OrientConfig(use_flip=True, angle_step=30.0, max_angle=90.0)
        profiles = [_profile(i, n_bins=36, seed=i) for i in range(3)]
        results = batch_orient_match(profiles, cfg)
        assert len(results) == 3

    def test_scores_in_range(self):
        profiles = [_profile(i, seed=i + 5) for i in range(4)]
        for r in batch_orient_match(profiles):
            assert 0.0 <= r.best_score <= 1.0

    def test_fragment_ids_correct(self):
        profiles = [_profile(i, seed=i) for i in range(3)]
        results = batch_orient_match(profiles)
        for r in results:
            assert r.fragment_a < r.fragment_b
