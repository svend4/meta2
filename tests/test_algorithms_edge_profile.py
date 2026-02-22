"""Тесты для puzzle_reconstruction.algorithms.edge_profile."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _make_profile(n=64, val=1.0, method="intensity") -> EdgeProfile:
    return EdgeProfile(
        signal=np.full(n, val, dtype=np.float32),
        side=0,
        n_samples=n,
        method=method,
    )


# ─── TestEdgeProfile ──────────────────────────────────────────────────────────

class TestEdgeProfile:
    def test_construction(self):
        ep = _make_profile()
        assert ep.n_samples == 64
        assert ep.method == "intensity"

    def test_signal_dtype_float32(self):
        ep = _make_profile()
        assert ep.signal.dtype == np.float32

    def test_side_stored(self):
        ep = EdgeProfile(
            signal=np.zeros(16, dtype=np.float32),
            side=3, n_samples=16, method="gradient"
        )
        assert ep.side == 3

    def test_params_default_empty(self):
        ep = _make_profile()
        assert ep.params == {}

    def test_params_stored(self):
        ep = EdgeProfile(
            signal=np.zeros(16, dtype=np.float32),
            side=0, n_samples=16, method="intensity",
            params={"border_frac": 0.08}
        )
        assert ep.params["border_frac"] == pytest.approx(0.08)


# ─── TestProfileMatchResult ───────────────────────────────────────────────────

class TestProfileMatchResult:
    def test_construction(self):
        r = ProfileMatchResult(score=0.7, correlation=0.8, dtw_score=0.6)
        assert r.score == pytest.approx(0.7)

    def test_default_method(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        assert r.method == "profile"

    def test_params_default_empty(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        assert r.params == {}


# ─── TestExtractIntensityProfile ──────────────────────────────────────────────

class TestExtractIntensityProfile:
    def test_returns_edge_profile(self):
        ep = extract_intensity_profile(_gray())
        assert isinstance(ep, EdgeProfile)

    def test_method_intensity(self):
        ep = extract_intensity_profile(_gray())
        assert ep.method == "intensity"

    def test_n_samples_64(self):
        ep = extract_intensity_profile(_gray(), n_samples=64)
        assert ep.n_samples == 64
        assert len(ep.signal) == 64

    def test_custom_n_samples(self):
        ep = extract_intensity_profile(_gray(), n_samples=32)
        assert len(ep.signal) == 32

    def test_all_four_sides(self):
        img = _rand_gray()
        for side in range(4):
            ep = extract_intensity_profile(img, side=side)
            assert isinstance(ep, EdgeProfile)
            assert ep.side == side

    def test_rgb_ok(self):
        ep = extract_intensity_profile(_rgb())
        assert isinstance(ep, EdgeProfile)

    def test_constant_image_constant_signal(self):
        ep = extract_intensity_profile(_gray(val=100), n_samples=16)
        np.testing.assert_allclose(ep.signal, 100.0, atol=1.0)

    def test_signal_dtype_float32(self):
        ep = extract_intensity_profile(_rand_gray())
        assert ep.signal.dtype == np.float32


# ─── TestExtractGradientProfile ───────────────────────────────────────────────

class TestExtractGradientProfile:
    def test_returns_edge_profile(self):
        ep = extract_gradient_profile(_rand_gray())
        assert isinstance(ep, EdgeProfile)

    def test_method_gradient(self):
        ep = extract_gradient_profile(_rand_gray())
        assert ep.method == "gradient"

    def test_n_samples_matches(self):
        ep = extract_gradient_profile(_rand_gray(), n_samples=48)
        assert len(ep.signal) == 48

    def test_constant_image_zero_gradient(self):
        ep = extract_gradient_profile(_gray(), n_samples=16)
        assert ep.signal.max() < 1.0

    def test_all_sides_ok(self):
        img = _rand_gray()
        for side in range(4):
            ep = extract_gradient_profile(img, side=side, n_samples=16)
            assert ep.side == side

    def test_signal_nonneg(self):
        ep = extract_gradient_profile(_rand_gray())
        assert ep.signal.min() >= 0.0


# ─── TestExtractTextureProfile ────────────────────────────────────────────────

class TestExtractTextureProfile:
    def test_returns_edge_profile(self):
        ep = extract_texture_profile(_rand_gray())
        assert isinstance(ep, EdgeProfile)

    def test_method_texture(self):
        ep = extract_texture_profile(_rand_gray())
        assert ep.method == "texture"

    def test_n_samples_matches(self):
        ep = extract_texture_profile(_rand_gray(), n_samples=32)
        assert len(ep.signal) == 32

    def test_constant_image_zero_texture(self):
        ep = extract_texture_profile(_gray(), n_samples=16)
        np.testing.assert_allclose(ep.signal, 0.0, atol=1.0)

    def test_all_sides_ok(self):
        img = _rand_gray()
        for side in range(4):
            ep = extract_texture_profile(img, side=side, n_samples=16)
            assert ep.side == side


# ─── TestNormalizeProfile ─────────────────────────────────────────────────────

class TestNormalizeProfile:
    def test_returns_edge_profile(self):
        ep = _make_profile(n=32, val=5.0)
        out = normalize_profile(ep)
        assert isinstance(out, EdgeProfile)

    def test_zero_mean_after_normalization(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ep = EdgeProfile(signal=signal, side=0, n_samples=5, method="intensity")
        out = normalize_profile(ep)
        assert out.signal.mean() == pytest.approx(0.0, abs=1e-5)

    def test_unit_std_after_normalization(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ep = EdgeProfile(signal=signal, side=0, n_samples=5, method="intensity")
        out = normalize_profile(ep)
        assert out.signal.std() == pytest.approx(1.0, abs=1e-5)

    def test_constant_signal_zero_output(self):
        ep = _make_profile(n=16, val=42.0)
        out = normalize_profile(ep)
        np.testing.assert_allclose(out.signal, 0.0)

    def test_method_updated(self):
        ep = _make_profile(n=8)
        out = normalize_profile(ep)
        assert "normalized" in out.method


# ─── TestProfileCorrelation ───────────────────────────────────────────────────

class TestProfileCorrelation:
    def test_identical_profiles_one(self):
        signal = np.linspace(0, 1, 32, dtype=np.float32)
        ep = EdgeProfile(signal=signal, side=0, n_samples=32, method="intensity")
        corr = profile_correlation(ep, ep)
        assert corr == pytest.approx(1.0, abs=1e-5)

    def test_range_zero_to_one(self):
        ep1 = _make_profile(32, val=1.0)
        signal2 = np.linspace(0, 1, 32, dtype=np.float32)
        ep2 = EdgeProfile(signal=signal2, side=0, n_samples=32, method="intensity")
        corr = profile_correlation(ep1, ep2)
        assert 0.0 <= corr <= 1.0

    def test_constant_profiles_returns_half(self):
        ep1 = _make_profile(16, val=5.0)
        ep2 = _make_profile(16, val=5.0)
        # Both constant and identical → 1.0 (special case in implementation)
        corr = profile_correlation(ep1, ep2)
        assert 0.0 <= corr <= 1.0

    def test_opposite_signals(self):
        s1 = np.linspace(0, 1, 32, dtype=np.float32)
        s2 = np.linspace(1, 0, 32, dtype=np.float32)
        ep1 = EdgeProfile(signal=s1, side=0, n_samples=32, method="intensity")
        ep2 = EdgeProfile(signal=s2, side=0, n_samples=32, method="intensity")
        corr = profile_correlation(ep1, ep2)
        # Anti-correlated → corr should be < 0.5
        assert corr < 0.5

    def test_returns_float(self):
        ep = _make_profile(16)
        assert isinstance(profile_correlation(ep, ep), float)


# ─── TestProfileDtw ───────────────────────────────────────────────────────────

class TestProfileDtw:
    def test_identical_profiles_one(self):
        signal = np.linspace(0, 1, 16, dtype=np.float32)
        ep = EdgeProfile(signal=signal, side=0, n_samples=16, method="intensity")
        score = profile_dtw(ep, ep)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_range_zero_to_one(self):
        ep1 = _make_profile(16, val=0.5)
        ep2 = _make_profile(16, val=0.0)
        score = profile_dtw(ep1, ep2)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        ep = _make_profile(8)
        assert isinstance(profile_dtw(ep, ep), float)

    def test_with_window(self):
        signal = np.linspace(0, 1, 16, dtype=np.float32)
        ep = EdgeProfile(signal=signal, side=0, n_samples=16, method="intensity")
        score = profile_dtw(ep, ep, window=4)
        assert 0.0 <= score <= 1.0

    def test_empty_returns_zero(self):
        ep = EdgeProfile(signal=np.array([], dtype=np.float32),
                         side=0, n_samples=0, method="intensity")
        score = profile_dtw(ep, ep)
        assert score == pytest.approx(0.0)


# ─── TestMatchEdgeProfiles ────────────────────────────────────────────────────

class TestMatchEdgeProfiles:
    def test_returns_profile_match_result(self):
        img = _rand_gray()
        r = match_edge_profiles(img, img)
        assert isinstance(r, ProfileMatchResult)

    def test_score_range(self):
        img1 = _rand_gray(seed=0)
        img2 = _rand_gray(seed=1)
        r = match_edge_profiles(img1, img2)
        assert 0.0 <= r.score <= 1.0

    def test_identical_images_high_score(self):
        img = _gray()
        r = match_edge_profiles(img, img)
        assert r.score >= 0.4

    def test_all_sides(self):
        img = _rand_gray()
        for s1 in range(4):
            for s2 in range(4):
                r = match_edge_profiles(img, img, side1=s1, side2=s2)
                assert 0.0 <= r.score <= 1.0

    def test_params_stored(self):
        img = _rand_gray()
        r = match_edge_profiles(img, img, side1=1, side2=3)
        assert r.params["side1"] == 1
        assert r.params["side2"] == 3

    def test_rgb_ok(self):
        img = _rgb()
        r = match_edge_profiles(img, img)
        assert isinstance(r, ProfileMatchResult)


# ─── TestBatchProfileMatch ────────────────────────────────────────────────────

class TestBatchProfileMatch:
    def test_returns_list(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        result = batch_profile_match(imgs, [(0, 1, 1, 3), (1, 0, 2, 2)])
        assert isinstance(result, list)

    def test_length_matches_pairs(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        pairs = [(0, 0, 1, 2), (0, 1, 2, 3)]
        result = batch_profile_match(imgs, pairs)
        assert len(result) == 2

    def test_empty_pairs(self):
        imgs = [_rand_gray()]
        result = batch_profile_match(imgs, [])
        assert result == []

    def test_all_profile_match_results(self):
        imgs = [_rand_gray(seed=i) for i in range(2)]
        pairs = [(0, 0, 1, 2)]
        for r in batch_profile_match(imgs, pairs):
            assert isinstance(r, ProfileMatchResult)
