"""Extra tests for puzzle_reconstruction/algorithms/edge_profile.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.edge_profile import (
    EdgeProfile,
    ProfileMatchResult,
    extract_intensity_profile,
    extract_gradient_profile,
    extract_texture_profile,
    normalize_profile,
    profile_correlation,
    profile_dtw,
    match_edge_profiles,
    batch_profile_match,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=11):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 100
    img[:, :, 2] = 50
    return img


def _make_profile(n=64, val=0.0, method="intensity", side=1):
    sig = np.full(n, val, dtype=np.float32)
    return EdgeProfile(signal=sig, side=side, n_samples=n, method=method)


def _from_array(arr, side=1, method="intensity"):
    sig = np.asarray(arr, dtype=np.float32)
    return EdgeProfile(signal=sig, side=side, n_samples=len(sig), method=method)


# ─── EdgeProfile (extra) ─────────────────────────────────────────────────────

class TestEdgeProfileExtra:
    def test_signal_dtype_float32(self):
        p = _make_profile(n=32)
        assert p.signal.dtype == np.float32

    def test_signal_shape(self):
        p = _make_profile(n=16)
        assert p.signal.shape == (16,)

    def test_n_samples_zero_ok(self):
        p = EdgeProfile(signal=np.array([], dtype=np.float32),
                        side=0, n_samples=0, method="intensity")
        assert p.n_samples == 0

    def test_params_dict_default(self):
        p = _make_profile()
        assert isinstance(p.params, dict)

    def test_side_stored(self):
        for s in range(4):
            p = _make_profile(side=s)
            assert p.side == s

    def test_method_stored(self):
        for m in ("intensity", "gradient", "texture"):
            p = _make_profile(method=m)
            assert p.method == m

    def test_repr_has_edge_profile(self):
        p = _make_profile()
        assert "EdgeProfile" in repr(p)


# ─── ProfileMatchResult (extra) ──────────────────────────────────────────────

class TestProfileMatchResultExtra:
    def test_score_stored(self):
        r = ProfileMatchResult(score=0.4, correlation=0.5, dtw_score=0.3)
        assert r.score == pytest.approx(0.4)

    def test_correlation_stored(self):
        r = ProfileMatchResult(score=0.7, correlation=0.8, dtw_score=0.6)
        assert r.correlation == pytest.approx(0.8)

    def test_dtw_score_stored(self):
        r = ProfileMatchResult(score=0.7, correlation=0.8, dtw_score=0.6)
        assert r.dtw_score == pytest.approx(0.6)

    def test_method_is_profile(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        assert r.method == "profile"

    def test_params_dict(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        assert isinstance(r.params, dict)

    def test_repr_contains_score(self):
        r = ProfileMatchResult(score=0.42, correlation=0.5, dtw_score=0.5)
        assert "0.4" in repr(r)

    def test_all_scores_in_range(self):
        r = ProfileMatchResult(score=0.9, correlation=0.8, dtw_score=0.7)
        assert 0.0 <= r.score <= 1.0
        assert 0.0 <= r.correlation <= 1.0
        assert 0.0 <= r.dtw_score <= 1.0


# ─── extract_intensity_profile (extra) ───────────────────────────────────────

class TestExtractIntensityProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(extract_intensity_profile(_noisy(), side=0), EdgeProfile)

    def test_method_intensity(self):
        assert extract_intensity_profile(_noisy(), side=0).method == "intensity"

    def test_dtype_float32(self):
        assert extract_intensity_profile(_noisy(), side=1).signal.dtype == np.float32

    def test_n_samples_default_64(self):
        p = extract_intensity_profile(_noisy(), side=0)
        assert p.n_samples == 64
        assert len(p.signal) == 64

    def test_custom_n_samples(self):
        p = extract_intensity_profile(_noisy(), side=1, n_samples=16)
        assert p.n_samples == 16

    def test_side_stored(self):
        for s in range(4):
            p = extract_intensity_profile(_noisy(), side=s)
            assert p.side == s

    def test_signal_range_0_255(self):
        p = extract_intensity_profile(_noisy(), side=2)
        assert p.signal.min() >= 0.0
        assert p.signal.max() <= 255.0

    def test_constant_uniform_signal(self):
        p = extract_intensity_profile(_gray(val=100), side=0)
        assert np.allclose(p.signal, p.signal[0], atol=1.0)

    def test_bgr_input_ok(self):
        p = extract_intensity_profile(_bgr(), side=1)
        assert len(p.signal) == 64

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        p = extract_intensity_profile(_noisy(64, 64), side=side)
        assert isinstance(p, EdgeProfile)


# ─── extract_gradient_profile (extra) ────────────────────────────────────────

class TestExtractGradientProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(extract_gradient_profile(_noisy(), side=0), EdgeProfile)

    def test_method_gradient(self):
        assert extract_gradient_profile(_noisy(), side=1).method == "gradient"

    def test_dtype_float32(self):
        assert extract_gradient_profile(_noisy(), side=1).signal.dtype == np.float32

    def test_custom_n_samples(self):
        p = extract_gradient_profile(_noisy(), side=2, n_samples=32)
        assert len(p.signal) == 32

    def test_nonneg_signal(self):
        p = extract_gradient_profile(_noisy(), side=3)
        assert np.all(p.signal >= 0.0)

    def test_constant_near_zero(self):
        p = extract_gradient_profile(_gray(val=128), side=1)
        assert p.signal.max() < 10.0

    def test_bgr_input_ok(self):
        p = extract_gradient_profile(_bgr(), side=0)
        assert len(p.signal) == 64

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        p = extract_gradient_profile(_noisy(), side=side)
        assert isinstance(p, EdgeProfile)


# ─── extract_texture_profile (extra) ─────────────────────────────────────────

class TestExtractTextureProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(extract_texture_profile(_noisy(), side=0), EdgeProfile)

    def test_method_texture(self):
        assert extract_texture_profile(_noisy(), side=1).method == "texture"

    def test_dtype_float32(self):
        assert extract_texture_profile(_noisy(), side=0).signal.dtype == np.float32

    def test_custom_n_samples(self):
        p = extract_texture_profile(_noisy(), side=1, n_samples=32)
        assert len(p.signal) == 32

    def test_nonneg_signal(self):
        p = extract_texture_profile(_noisy(), side=2)
        assert np.all(p.signal >= 0.0)

    def test_constant_near_zero(self):
        p = extract_texture_profile(_gray(val=128), side=0)
        assert p.signal.max() < 5.0

    def test_window_param_stored(self):
        p = extract_texture_profile(_noisy(), side=1, window=8)
        assert p.params.get("window") == 8

    def test_bgr_input_ok(self):
        p = extract_texture_profile(_bgr(), side=1)
        assert len(p.signal) == 64


# ─── normalize_profile (extra) ───────────────────────────────────────────────

class TestNormalizeProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(normalize_profile(_make_profile(val=50.0)), EdgeProfile)

    def test_normalized_in_method(self):
        n = normalize_profile(_make_profile())
        assert "normalized" in n.method

    def test_constant_profile_all_zero(self):
        p = _make_profile(n=16, val=200.0)
        n = normalize_profile(p)
        assert np.allclose(n.signal, 0.0, atol=1e-5)

    def test_zero_mean(self):
        p = _from_array(np.arange(8, dtype=np.float32))
        n = normalize_profile(p)
        assert n.signal.mean() == pytest.approx(0.0, abs=1e-4)

    def test_unit_std(self):
        p = _from_array(np.arange(8, dtype=np.float32))
        n = normalize_profile(p)
        assert n.signal.std() == pytest.approx(1.0, abs=1e-3)

    def test_n_samples_preserved(self):
        p = _make_profile(n=48)
        n = normalize_profile(p)
        assert n.n_samples == 48

    def test_side_preserved(self):
        p = _make_profile(side=3)
        n = normalize_profile(p)
        assert n.side == 3

    def test_dtype_float32(self):
        p = _make_profile(n=32)
        n = normalize_profile(p)
        assert n.signal.dtype == np.float32

    def test_normalized_flag_in_params(self):
        p = _make_profile()
        n = normalize_profile(p)
        assert n.params.get("normalized") is True


# ─── profile_correlation (extra) ─────────────────────────────────────────────

class TestProfileCorrelationExtra:
    def test_same_profile_is_one(self):
        p = _from_array(np.arange(1, 9, dtype=np.float32))
        assert profile_correlation(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_in_range(self):
        p1 = _from_array(np.random.rand(32).astype(np.float32))
        p2 = _from_array(np.random.rand(32).astype(np.float32))
        c = profile_correlation(p1, p2)
        assert 0.0 <= c <= 1.0

    def test_returns_float(self):
        p = _make_profile(n=16, val=50.0)
        assert isinstance(profile_correlation(p, p), float)

    def test_constant_same_is_one(self):
        p1 = _make_profile(n=16, val=100.0)
        p2 = _make_profile(n=16, val=100.0)
        assert profile_correlation(p1, p2) == pytest.approx(1.0, abs=1e-4)

    def test_constant_different_is_half(self):
        p1 = _make_profile(n=16, val=50.0)
        p2 = _make_profile(n=16, val=150.0)
        assert profile_correlation(p1, p2) == pytest.approx(0.5, abs=1e-4)

    def test_empty_profile_half(self):
        p = EdgeProfile(signal=np.array([], dtype=np.float32),
                        side=1, n_samples=0, method="intensity")
        assert profile_correlation(p, p) == pytest.approx(0.5, abs=1e-4)

    def test_noisy_same_is_one(self):
        sig = np.random.rand(32).astype(np.float32) * 100
        p = EdgeProfile(signal=sig, side=1, n_samples=32, method="intensity")
        assert profile_correlation(p, p) == pytest.approx(1.0, abs=1e-4)


# ─── profile_dtw (extra) ─────────────────────────────────────────────────────

class TestProfileDtwExtra:
    def test_same_is_one(self):
        p = _from_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert profile_dtw(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_in_range(self):
        p1 = _from_array(np.random.rand(16).astype(np.float32))
        p2 = _from_array(np.random.rand(16).astype(np.float32))
        d = profile_dtw(p1, p2)
        assert 0.0 <= d <= 1.0

    def test_returns_float(self):
        p1 = _from_array(np.arange(8, dtype=np.float32))
        p2 = _from_array(np.arange(8, 0, -1, dtype=np.float32))
        assert isinstance(profile_dtw(p1, p2), float)

    def test_empty_profile_zero(self):
        p = EdgeProfile(signal=np.array([], dtype=np.float32),
                        side=1, n_samples=0, method="x")
        assert profile_dtw(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_identical_noisy_is_one(self):
        sig = np.random.rand(24).astype(np.float32)
        p = EdgeProfile(signal=sig, side=0, n_samples=24, method="intensity")
        assert profile_dtw(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_window_param_ok(self):
        p1 = _from_array(np.random.rand(16).astype(np.float32))
        p2 = _from_array(np.random.rand(16).astype(np.float32))
        d = profile_dtw(p1, p2, window=4)
        assert 0.0 <= d <= 1.0


# ─── match_edge_profiles (extra) ─────────────────────────────────────────────

class TestMatchEdgeProfilesExtra:
    def test_returns_profile_match_result(self):
        assert isinstance(match_edge_profiles(_noisy(), _noisy(seed=5)),
                           ProfileMatchResult)

    def test_score_in_range(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=7))
        assert 0.0 <= r.score <= 1.0

    def test_correlation_in_range(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=3))
        assert 0.0 <= r.correlation <= 1.0

    def test_dtw_in_range(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=3))
        assert 0.0 <= r.dtw_score <= 1.0

    def test_method_is_profile(self):
        assert match_edge_profiles(_noisy(), _noisy()).method == "profile"

    def test_side_params_stored(self):
        r = match_edge_profiles(_noisy(), _noisy(), side1=2, side2=3)
        assert r.params.get("side1") == 2
        assert r.params.get("side2") == 3

    def test_n_samples_stored(self):
        r = match_edge_profiles(_noisy(), _noisy(), n_samples=48)
        assert r.params.get("n_samples") == 48

    def test_weights_stored(self):
        r = match_edge_profiles(_noisy(), _noisy(),
                                w_intensity=0.4, w_gradient=0.4, w_texture=0.2)
        assert r.params.get("w_intensity") == pytest.approx(0.4)

    def test_identical_image_high_score(self):
        img = _noisy()
        r = match_edge_profiles(img, img, side1=0, side2=0)
        assert r.score > 0.5

    def test_gray_input_ok(self):
        r = match_edge_profiles(_gray(), _gray())
        assert isinstance(r, ProfileMatchResult)

    def test_bgr_input_ok(self):
        r = match_edge_profiles(_bgr(), _bgr())
        assert 0.0 <= r.score <= 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = match_edge_profiles(_noisy(), _noisy(seed=9), side1=side, side2=side)
        assert 0.0 <= r.score <= 1.0


# ─── batch_profile_match (extra) ─────────────────────────────────────────────

class TestBatchProfileMatchExtra:
    def test_returns_list(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        results = batch_profile_match(imgs, [(0, 1, 1, 3)])
        assert isinstance(results, list)

    def test_length_matches_pairs(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        pairs = [(0, 1, 0, 2), (1, 2, 1, 3), (2, 3, 2, 0)]
        assert len(batch_profile_match(imgs, pairs)) == 3

    def test_empty_pairs_empty_list(self):
        assert batch_profile_match([_noisy()], []) == []

    def test_each_is_result(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        for r in batch_profile_match(imgs, [(0, 1, 0, 2)]):
            assert isinstance(r, ProfileMatchResult)

    def test_scores_in_range(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (1, 2, 0, 2)]
        for r in batch_profile_match(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_n_samples_forwarded(self):
        imgs = [_noisy(seed=i) for i in range(2)]
        results = batch_profile_match(imgs, [(0, 1, 0, 2)], n_samples=32)
        assert results[0].params.get("n_samples") == 32

    def test_identical_pair_high_score(self):
        img = _noisy()
        results = batch_profile_match([img, img], [(0, 1, 1, 1)])
        assert results[0].score > 0.5
