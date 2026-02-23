"""Extra tests for puzzle_reconstruction.matching.orient_matcher."""
from __future__ import annotations

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _noisy(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _profile(n_bins=8, fid=0, seed=None):
    if seed is not None:
        hist = np.random.default_rng(seed).uniform(0.1, 1.0, n_bins)
    else:
        hist = np.ones(n_bins)
    hist = hist / hist.sum()
    return OrientProfile(fragment_id=fid, histogram=hist, dominant=0.0)


# ─── TestOrientConfigExtra ───────────────────────────────────────────────────

class TestOrientConfigExtra:
    def test_default_n_bins(self):
        assert OrientConfig().n_bins == 36

    def test_default_angle_step(self):
        assert OrientConfig().angle_step == pytest.approx(10.0)

    def test_default_max_angle(self):
        assert OrientConfig().max_angle == pytest.approx(180.0)

    def test_default_normalize_true(self):
        assert OrientConfig().normalize is True

    def test_default_use_flip_false(self):
        assert OrientConfig().use_flip is False

    def test_n_bins_2_valid(self):
        assert OrientConfig(n_bins=2).n_bins == 2

    def test_n_bins_100(self):
        assert OrientConfig(n_bins=100).n_bins == 100

    def test_n_bins_1_raises(self):
        with pytest.raises(ValueError):
            OrientConfig(n_bins=1)

    def test_angle_step_positive_ok(self):
        assert OrientConfig(angle_step=5.0).angle_step == pytest.approx(5.0)

    def test_angle_step_zero_raises(self):
        with pytest.raises(ValueError):
            OrientConfig(angle_step=0.0)

    def test_max_angle_0_valid(self):
        assert OrientConfig(max_angle=0.0).max_angle == pytest.approx(0.0)

    def test_max_angle_negative_raises(self):
        with pytest.raises(ValueError):
            OrientConfig(max_angle=-10.0)

    def test_use_flip_true_stored(self):
        assert OrientConfig(use_flip=True).use_flip is True

    def test_normalize_false_stored(self):
        assert OrientConfig(normalize=False).normalize is False


# ─── TestOrientProfileExtra ──────────────────────────────────────────────────

class TestOrientProfileExtra:
    def test_fragment_id_stored(self):
        assert _profile(fid=5).fragment_id == 5

    def test_n_bins_property(self):
        assert _profile(n_bins=12).n_bins == 12

    def test_dominant_stored(self):
        hist = np.ones(8)
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=45.0)
        assert op.dominant == pytest.approx(45.0)

    def test_uniform_flat_hist(self):
        hist = np.ones(8) / 8
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.is_uniform is True

    def test_peaked_hist_not_uniform(self):
        hist = np.zeros(8)
        hist[3] = 1.0
        op = OrientProfile(fragment_id=0, histogram=hist, dominant=0.0)
        assert op.is_uniform is False

    def test_negative_fragment_id_raises(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=-1, histogram=np.ones(4), dominant=0.0)

    def test_2d_hist_raises(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones((4, 2)), dominant=0.0)

    def test_dominant_0_valid(self):
        op = OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=0.0)
        assert op.dominant == pytest.approx(0.0)

    def test_dominant_359_valid(self):
        op = OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=359.0)
        assert op.dominant == pytest.approx(359.0)

    def test_dominant_360_raises(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=360.0)

    def test_dominant_negative_raises(self):
        with pytest.raises(ValueError):
            OrientProfile(fragment_id=0, histogram=np.ones(4), dominant=-1.0)


# ─── TestOrientMatchResultExtra ──────────────────────────────────────────────

class TestOrientMatchResultExtra:
    def test_pair_stored(self):
        r = OrientMatchResult(pair=(2, 5), best_angle=0.0, best_score=0.5)
        assert r.pair == (2, 5)

    def test_fragment_a(self):
        r = OrientMatchResult(pair=(3, 7), best_angle=0.0, best_score=0.5)
        assert r.fragment_a == 3

    def test_fragment_b(self):
        r = OrientMatchResult(pair=(3, 7), best_angle=0.0, best_score=0.5)
        assert r.fragment_b == 7

    def test_best_score_0_valid(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.0)
        assert r.best_score == pytest.approx(0.0)

    def test_best_score_1_valid(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=1.0)
        assert r.best_score == pytest.approx(1.0)

    def test_best_score_above_1_raises(self):
        with pytest.raises(ValueError):
            OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=1.1)

    def test_best_score_below_0_raises(self):
        with pytest.raises(ValueError):
            OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=-0.01)

    def test_is_flipped_default_false(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.5)
        assert r.is_flipped is False

    def test_n_angles_tested_from_dict(self):
        r = OrientMatchResult(pair=(0, 1), best_angle=0.0, best_score=0.5,
                              angle_scores={0.0: 0.5, 10.0: 0.4, 20.0: 0.3})
        assert r.n_angles_tested == 3


# ─── TestComputeOrientProfileExtra ───────────────────────────────────────────

class TestComputeOrientProfileExtra:
    def test_returns_orient_profile(self):
        assert isinstance(compute_orient_profile(_noisy()), OrientProfile)

    def test_fragment_id_default_zero(self):
        assert compute_orient_profile(_gray()).fragment_id == 0

    def test_fragment_id_custom(self):
        assert compute_orient_profile(_noisy(), fragment_id=9).fragment_id == 9

    def test_histogram_n_bins(self):
        cfg = OrientConfig(n_bins=12)
        result = compute_orient_profile(_noisy(), cfg=cfg)
        assert result.histogram.shape == (12,)

    def test_normalized_sums_1(self):
        cfg = OrientConfig(normalize=True)
        result = compute_orient_profile(_noisy(), cfg=cfg)
        if result.histogram.sum() > 1e-12:
            assert abs(result.histogram.sum() - 1.0) < 1e-5

    def test_dominant_in_0_360(self):
        result = compute_orient_profile(_noisy())
        assert 0.0 <= result.dominant < 360.0

    def test_3d_image_ok(self):
        bgr = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        assert isinstance(compute_orient_profile(bgr), OrientProfile)

    def test_uniform_gray_no_crash(self):
        assert isinstance(compute_orient_profile(_gray(fill=200)), OrientProfile)


# ─── TestOrientSimilarityExtra ───────────────────────────────────────────────

class TestOrientSimilarityExtra:
    def test_same_at_0_is_1(self):
        p = _profile(n_bins=8, seed=42)
        assert orient_similarity(p, p, angle_deg=0.0) == pytest.approx(1.0, abs=1e-6)

    def test_result_in_0_1(self):
        pa = _profile(n_bins=8, seed=1)
        pb = _profile(n_bins=8, seed=2)
        sim = orient_similarity(pa, pb)
        assert 0.0 <= sim <= 1.0

    def test_empty_hist_zero(self):
        empty = OrientProfile(fragment_id=0, histogram=np.zeros(8), dominant=0.0)
        pb = _profile(n_bins=8, seed=5)
        assert orient_similarity(empty, pb) == pytest.approx(0.0)


# ─── TestBestOrientAngleExtra ─────────────────────────────────────────────────

class TestBestOrientAngleExtra:
    def test_returns_tuple_len_2(self):
        pa = _profile(fid=0, seed=0)
        pb = _profile(fid=1, seed=1)
        result = best_orient_angle(pa, pb)
        assert isinstance(result, tuple) and len(result) == 2

    def test_score_in_0_1(self):
        pa = _profile(fid=0, seed=3)
        pb = _profile(fid=1, seed=4)
        _, score = best_orient_angle(pa, pb)
        assert 0.0 <= score <= 1.0

    def test_same_profile_score_1(self):
        p = _profile(seed=0)
        _, score = best_orient_angle(p, p)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_angle_respects_max_angle(self):
        pa = _profile(fid=0, seed=5)
        pb = _profile(fid=1, seed=6)
        cfg = OrientConfig(max_angle=90.0, angle_step=10.0)
        angle, _ = best_orient_angle(pa, pb, cfg=cfg)
        assert 0.0 <= angle <= 90.0


# ─── TestMatchOrientPairExtra ─────────────────────────────────────────────────

class TestMatchOrientPairExtra:
    def test_returns_orient_match_result(self):
        pa = _profile(fid=0)
        pb = _profile(fid=1)
        assert isinstance(match_orient_pair(pa, pb), OrientMatchResult)

    def test_pair_ids(self):
        pa = _profile(fid=3)
        pb = _profile(fid=7)
        result = match_orient_pair(pa, pb)
        assert result.pair == (3, 7)

    def test_best_score_in_0_1(self):
        pa = _profile(fid=0, seed=0)
        pb = _profile(fid=1, seed=1)
        result = match_orient_pair(pa, pb)
        assert 0.0 <= result.best_score <= 1.0

    def test_angle_scores_nonempty(self):
        pa = _profile(fid=0)
        pb = _profile(fid=1)
        cfg = OrientConfig(angle_step=10.0, max_angle=90.0)
        result = match_orient_pair(pa, pb, cfg=cfg)
        assert len(result.angle_scores) > 0

    def test_flip_extends_n_angles(self):
        pa = _profile(fid=0, seed=10)
        pb = _profile(fid=1, seed=11)
        cfg_no = OrientConfig(angle_step=10.0, max_angle=90.0, use_flip=False)
        cfg_fl = OrientConfig(angle_step=10.0, max_angle=90.0, use_flip=True)
        r_no = match_orient_pair(pa, pb, cfg=cfg_no)
        r_fl = match_orient_pair(pa, pb, cfg=cfg_fl)
        assert r_fl.n_angles_tested > r_no.n_angles_tested


# ─── TestBatchOrientMatchExtra ────────────────────────────────────────────────

class TestBatchOrientMatchExtra:
    def test_empty_returns_empty(self):
        assert batch_orient_match([]) == []

    def test_single_returns_empty(self):
        assert batch_orient_match([_profile(fid=0)]) == []

    def test_two_profiles_one_result(self):
        results = batch_orient_match([_profile(fid=0), _profile(fid=1)])
        assert len(results) == 1

    def test_three_profiles_three_results(self):
        profiles = [_profile(fid=i, seed=i) for i in range(3)]
        assert len(batch_orient_match(profiles)) == 3

    def test_four_profiles_six_results(self):
        profiles = [_profile(fid=i, seed=i) for i in range(4)]
        assert len(batch_orient_match(profiles)) == 6

    def test_all_are_orient_match_results(self):
        profiles = [_profile(fid=i, seed=i) for i in range(3)]
        for r in batch_orient_match(profiles):
            assert isinstance(r, OrientMatchResult)

    def test_all_scores_in_range(self):
        profiles = [_profile(fid=i, seed=i) for i in range(4)]
        for r in batch_orient_match(profiles):
            assert 0.0 <= r.best_score <= 1.0
