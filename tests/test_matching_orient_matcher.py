"""Тесты для puzzle_reconstruction/matching/orient_matcher.py."""
import pytest
import numpy as np

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=32, w=32, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_profile(n_bins=8, fid=0, seed=None):
    if seed is not None:
        hist = np.random.default_rng(seed).uniform(0, 1, n_bins)
    else:
        hist = np.ones(n_bins) / n_bins
    hist = hist / hist.sum()
    return OrientProfile(fragment_id=fid, histogram=hist, dominant=0.0)


# ─── OrientConfig ─────────────────────────────────────────────────────────────

class TestOrientConfig:
    def test_defaults(self):
        cfg = OrientConfig()
        assert cfg.n_bins == 36
        assert cfg.angle_step == pytest.approx(10.0)
        assert cfg.max_angle == pytest.approx(180.0)
        assert cfg.normalize is True
        assert cfg.use_flip is False

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            OrientConfig(n_bins=1)

    def test_angle_step_zero_raises(self):
        with pytest.raises(ValueError, match="angle_step"):
            OrientConfig(angle_step=0.0)

    def test_angle_step_negative_raises(self):
        with pytest.raises(ValueError):
            OrientConfig(angle_step=-5.0)

    def test_max_angle_negative_raises(self):
        with pytest.raises(ValueError, match="max_angle"):
            OrientConfig(max_angle=-1.0)

    def test_max_angle_zero_valid(self):
        cfg = OrientConfig(max_angle=0.0)
        assert cfg.max_angle == 0.0

    def test_n_bins_2_valid(self):
        cfg = OrientConfig(n_bins=2)
        assert cfg.n_bins == 2


# ─── OrientProfile ────────────────────────────────────────────────────────────

class TestOrientProfile:
    def test_creation(self):
        hist = np.ones(8) / 8
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=90.0)
        assert op.fragment_id == 0
        assert op.dominant == pytest.approx(90.0)
        assert op.n_bins == 8

    def test_negative_fragment_id_raises(self):
        hist = np.ones(4)
        with pytest.raises(ValueError, match="fragment_id"):
            OrientProfile(fragment_id=-1, histogram=hist, dominant=0.0)

    def test_non_1d_histogram_raises(self):
        hist = np.ones((4, 2))
        with pytest.raises(ValueError, match="одномерным"):
            OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)

    def test_histogram_less_than_2_raises(self):
        hist = np.array([1.0])
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)

    def test_dominant_below_0_raises(self):
        hist = np.ones(4)
        with pytest.raises(ValueError, match="dominant"):
            OrientProfile(fragment_id=0, histogram=hist, dominant=-1.0)

    def test_dominant_360_raises(self):
        hist = np.ones(4)
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=hist, dominant=360.0)

    def test_dominant_0_valid(self):
        hist = np.ones(4)
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.dominant == 0.0

    def test_dominant_359_valid(self):
        hist = np.ones(4)
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=359.9)
        assert op.dominant == pytest.approx(359.9)

    def test_n_bins_property(self):
        hist = np.ones(12)
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.n_bins == 12

    def test_is_uniform_true_for_flat_hist(self):
        hist = np.ones(8) / 8.0
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.is_uniform is True

    def test_is_uniform_false_for_peaked_hist(self):
        hist = np.zeros(8)
        hist[0] = 1.0
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.is_uniform is False

    def test_is_uniform_true_for_zero_hist(self):
        hist = np.zeros(4)
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.is_uniform is True


# ─── OrientMatchResult ────────────────────────────────────────────────────────

class TestOrientMatchResult:
    def test_creation(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=45.0, best_score=0.7)
        assert r.pair == (0, 1)
        assert r.best_angle == pytest.approx(45.0)
        assert r.best_score == pytest.approx(0.7)
        assert r.is_flipped is False

    def test_best_score_above_1_raises(self):
        with pytest.raises(ValueError, match="best_score"):
            OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=1.1)

    def test_best_score_below_0_raises(self):
        with pytest.raises(ValueError):
            OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=-0.1)

    def test_fragment_a_property(self):
        r = OrientMatchResult(pair=(3, 5), best_angle=0.0, best_score=0.5)
        assert r.fragment_a == 3

    def test_fragment_b_property(self):
        r = OrientMatchResult(pair=(3, 5), best_angle=0.0, best_score=0.5)
        assert r.fragment_b == 5

    def test_n_angles_tested(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.5,
                              angle_scores={0.0: 0.5, 10.0: 0.6})
        assert r.n_angles_tested == 2

    def test_boundary_score_0(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.0)
        assert r.best_score == 0.0

    def test_boundary_score_1(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=1.0)
        assert r.best_score == 1.0


# ─── compute_orient_profile ───────────────────────────────────────────────────

class TestComputeOrientProfile:
    def test_returns_orient_profile(self):
        gray = make_noisy()
        result = compute_orient_profile(gray)
        assert isinstance(result, OrientProfile)

    def test_fragment_id_stored(self):
        gray = make_noisy()
        result = compute_orient_profile(gray, fragment_id=5)
        assert result.fragment_id == 5

    def test_histogram_shape(self):
        gray = make_noisy()
        cfg = OrientConfig(n_bins=18)
        result = compute_orient_profile(gray, cfg=cfg)
        assert result.histogram.shape == (18,)

    def test_normalized_hist_sums_to_1(self):
        gray = make_noisy()
        cfg = OrientConfig(normalize=True)
        result = compute_orient_profile(gray, cfg=cfg)
        if result.histogram.sum() > 1e-12:
            assert abs(result.histogram.sum() - 1.0) < 1e-6

    def test_dominant_in_range(self):
        gray = make_noisy()
        result = compute_orient_profile(gray)
        assert 0.0 <= result.dominant < 360.0

    def test_3d_image_accepted(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        result = compute_orient_profile(img)
        assert isinstance(result, OrientProfile)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_orient_profile(np.ones((4, 4, 4, 4), dtype=np.uint8))

    def test_uniform_image(self):
        gray = make_gray()
        result = compute_orient_profile(gray)
        assert isinstance(result, OrientProfile)


# ─── orient_similarity ────────────────────────────────────────────────────────

class TestOrientSimilarity:
    def test_same_profile_angle0_returns_1(self):
        p = make_profile(n_bins=8, fid=0, seed=42)
        sim = orient_similarity(p, p, angle_deg=0.0)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_returns_in_0_1(self):
        pa = make_profile(n_bins=8, fid=0, seed=1)
        pb = make_profile(n_bins=8, fid=1, seed=2)
        sim = orient_similarity(pa, pb)
        assert 0.0 <= sim <= 1.0

    def test_empty_histogram_returns_0(self):
        hist = np.zeros(8)
        pa = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        pb = make_profile(n_bins=8, fid=1, seed=0)
        sim = orient_similarity(pa, pb)
        assert sim == pytest.approx(0.0)


# ─── best_orient_angle ────────────────────────────────────────────────────────

class TestBestOrientAngle:
    def test_returns_tuple(self):
        pa = make_profile(n_bins=8, fid=0, seed=0)
        pb = make_profile(n_bins=8, fid=1, seed=1)
        result = best_orient_angle(pa, pb)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_in_0_1(self):
        pa = make_profile(n_bins=8, fid=0, seed=3)
        pb = make_profile(n_bins=8, fid=1, seed=4)
        _, score = best_orient_angle(pa, pb)
        assert 0.0 <= score <= 1.0

    def test_same_profile_score_1(self):
        p = make_profile(n_bins=8, fid=0, seed=0)
        _, score = best_orient_angle(p, p)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_angle_within_max_angle(self):
        pa = make_profile(n_bins=8, fid=0, seed=5)
        pb = make_profile(n_bins=8, fid=1, seed=6)
        cfg = OrientConfig(max_angle=90.0, angle_step=10.0)
        angle, _ = best_orient_angle(pa, pb, cfg=cfg)
        assert 0.0 <= angle <= 90.0


# ─── match_orient_pair ────────────────────────────────────────────────────────

class TestMatchOrientPair:
    def test_returns_orient_match_result(self):
        pa = make_profile(n_bins=8, fid=0)
        pb = make_profile(n_bins=8, fid=1)
        result = match_orient_pair(pa, pb)
        assert isinstance(result, OrientMatchResult)

    def test_pair_stored(self):
        pa = make_profile(n_bins=8, fid=3)
        pb = make_profile(n_bins=8, fid=7)
        result = match_orient_pair(pa, pb)
        assert result.pair == (3, 7)

    def test_best_score_in_0_1(self):
        pa = make_profile(n_bins=8, fid=0, seed=0)
        pb = make_profile(n_bins=8, fid=1, seed=1)
        result = match_orient_pair(pa, pb)
        assert 0.0 <= result.best_score <= 1.0

    def test_angle_scores_nonempty(self):
        pa = make_profile(n_bins=8, fid=0)
        pb = make_profile(n_bins=8, fid=1)
        cfg = OrientConfig(angle_step=10.0, max_angle=90.0)
        result = match_orient_pair(pa, pb, cfg=cfg)
        assert len(result.angle_scores) > 0

    def test_is_flipped_false_by_default(self):
        pa = make_profile(n_bins=8, fid=0)
        pb = make_profile(n_bins=8, fid=1)
        result = match_orient_pair(pa, pb)
        assert result.is_flipped is False

    def test_use_flip_extends_scores(self):
        pa = make_profile(n_bins=8, fid=0, seed=10)
        pb = make_profile(n_bins=8, fid=1, seed=11)
        cfg_no_flip = OrientConfig(angle_step=10.0, max_angle=90.0, use_flip=False)
        cfg_flip = OrientConfig(angle_step=10.0, max_angle=90.0, use_flip=True)
        r_no = match_orient_pair(pa, pb, cfg=cfg_no_flip)
        r_fl = match_orient_pair(pa, pb, cfg=cfg_flip)
        assert r_fl.n_angles_tested > r_no.n_angles_tested


# ─── batch_orient_match ───────────────────────────────────────────────────────

class TestBatchOrientMatch:
    def test_empty_profiles_returns_empty(self):
        result = batch_orient_match([])
        assert result == []

    def test_single_profile_returns_empty(self):
        p = make_profile(n_bins=8, fid=0)
        result = batch_orient_match([p])
        assert result == []

    def test_two_profiles_one_pair(self):
        pa = make_profile(n_bins=8, fid=0)
        pb = make_profile(n_bins=8, fid=1)
        result = batch_orient_match([pa, pb])
        assert len(result) == 1

    def test_three_profiles_three_pairs(self):
        profiles = [make_profile(n_bins=8, fid=i, seed=i) for i in range(3)]
        result = batch_orient_match(profiles)
        assert len(result) == 3  # C(3,2)

    def test_four_profiles_six_pairs(self):
        profiles = [make_profile(n_bins=8, fid=i, seed=i) for i in range(4)]
        result = batch_orient_match(profiles)
        assert len(result) == 6  # C(4,2)

    def test_returns_list_of_orient_match_results(self):
        profiles = [make_profile(n_bins=8, fid=i, seed=i) for i in range(3)]
        result = batch_orient_match(profiles)
        for r in result:
            assert isinstance(r, OrientMatchResult)
