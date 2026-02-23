"""Extra tests for puzzle_reconstruction.algorithms.edge_profile."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _rand(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rgb(h=64, w=64, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _prof(n=64, val=1.0, method="intensity"):
    return EdgeProfile(
        signal=np.full(n, val, dtype=np.float32),
        side=0, n_samples=n, method=method,
    )


def _linprof(n=32):
    return EdgeProfile(
        signal=np.linspace(0, 1, n, dtype=np.float32),
        side=0, n_samples=n, method="intensity",
    )


# ─── TestEdgeProfileExtra ────────────────────────────────────────────────────

class TestEdgeProfileExtra:
    def test_n_samples_stored(self):
        ep = _prof(n=32)
        assert ep.n_samples == 32

    def test_method_stored(self):
        ep = _prof(method="gradient")
        assert ep.method == "gradient"

    def test_signal_dtype(self):
        ep = _prof()
        assert ep.signal.dtype == np.float32

    def test_side_stored(self):
        ep = EdgeProfile(signal=np.zeros(16, dtype=np.float32),
                         side=2, n_samples=16, method="texture")
        assert ep.side == 2

    def test_params_default_empty(self):
        assert _prof().params == {}

    def test_params_custom(self):
        ep = EdgeProfile(signal=np.zeros(8, dtype=np.float32),
                         side=0, n_samples=8, method="intensity",
                         params={"window": 5})
        assert ep.params["window"] == 5

    def test_signal_length_matches_n_samples(self):
        ep = _prof(n=48)
        assert len(ep.signal) == ep.n_samples


# ─── TestProfileMatchResultExtra ─────────────────────────────────────────────

class TestProfileMatchResultExtra:
    def test_score_stored(self):
        r = ProfileMatchResult(score=0.65, correlation=0.7, dtw_score=0.6)
        assert r.score == pytest.approx(0.65)

    def test_correlation_stored(self):
        r = ProfileMatchResult(score=0.5, correlation=0.8, dtw_score=0.4)
        assert r.correlation == pytest.approx(0.8)

    def test_dtw_score_stored(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.9)
        assert r.dtw_score == pytest.approx(0.9)

    def test_default_method(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        assert r.method == "profile"

    def test_params_default_empty(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        assert r.params == {}


# ─── TestExtractIntensityProfileExtra ────────────────────────────────────────

class TestExtractIntensityProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(extract_intensity_profile(_gray(), side=0), EdgeProfile)

    def test_method_intensity(self):
        assert extract_intensity_profile(_gray(), side=0).method == "intensity"

    def test_custom_n_samples(self):
        ep = extract_intensity_profile(_gray(), side=0, n_samples=16)
        assert len(ep.signal) == 16

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        ep = extract_intensity_profile(_rand(), side=side)
        assert ep.side == side

    def test_rgb_ok(self):
        assert isinstance(extract_intensity_profile(_rgb(), side=0), EdgeProfile)

    def test_constant_image_constant_signal(self):
        ep = extract_intensity_profile(_gray(val=100), side=0, n_samples=16)
        np.testing.assert_allclose(ep.signal, 100.0, atol=1.0)

    def test_signal_dtype_float32(self):
        assert extract_intensity_profile(_rand(), side=0).signal.dtype == np.float32

    def test_default_n_samples(self):
        ep = extract_intensity_profile(_gray(), side=0)
        assert ep.n_samples > 0


# ─── TestExtractGradientProfileExtra ────────────────────────────────────────

class TestExtractGradientProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(extract_gradient_profile(_rand(), side=0), EdgeProfile)

    def test_method_gradient(self):
        assert extract_gradient_profile(_rand(), side=0).method == "gradient"

    def test_n_samples_matches(self):
        ep = extract_gradient_profile(_rand(), side=0, n_samples=24)
        assert len(ep.signal) == 24

    def test_constant_zero_gradient(self):
        ep = extract_gradient_profile(_gray(), side=0, n_samples=16)
        assert ep.signal.max() < 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        assert extract_gradient_profile(_rand(), side=side).side == side

    def test_signal_nonneg(self):
        assert extract_gradient_profile(_rand(), side=0).signal.min() >= 0.0

    def test_dtype_float32(self):
        assert extract_gradient_profile(_rand(), side=0).signal.dtype == np.float32


# ─── TestExtractTextureProfileExtra ──────────────────────────────────────────

class TestExtractTextureProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(extract_texture_profile(_rand(), side=0), EdgeProfile)

    def test_method_texture(self):
        assert extract_texture_profile(_rand(), side=0).method == "texture"

    def test_n_samples_matches(self):
        ep = extract_texture_profile(_rand(), side=0, n_samples=20)
        assert len(ep.signal) == 20

    def test_constant_zero_texture(self):
        ep = extract_texture_profile(_gray(), side=0, n_samples=16)
        np.testing.assert_allclose(ep.signal, 0.0, atol=1.0)

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        assert extract_texture_profile(_rand(), side=side).side == side

    def test_dtype_float32(self):
        assert extract_texture_profile(_rand(), side=0).signal.dtype == np.float32


# ─── TestNormalizeProfileExtra ───────────────────────────────────────────────

class TestNormalizeProfileExtra:
    def test_returns_edge_profile(self):
        assert isinstance(normalize_profile(_prof()), EdgeProfile)

    def test_zero_mean(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ep = EdgeProfile(signal=sig, side=0, n_samples=5, method="intensity")
        out = normalize_profile(ep)
        assert out.signal.mean() == pytest.approx(0.0, abs=1e-5)

    def test_unit_std(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ep = EdgeProfile(signal=sig, side=0, n_samples=5, method="intensity")
        out = normalize_profile(ep)
        assert out.signal.std() == pytest.approx(1.0, abs=1e-5)

    def test_constant_zero_output(self):
        ep = _prof(n=16, val=42.0)
        out = normalize_profile(ep)
        np.testing.assert_allclose(out.signal, 0.0)

    def test_method_contains_normalized(self):
        assert "normalized" in normalize_profile(_prof()).method

    def test_preserves_n_samples(self):
        ep = _prof(n=24)
        assert normalize_profile(ep).n_samples == 24


# ─── TestProfileCorrelationExtra ─────────────────────────────────────────────

class TestProfileCorrelationExtra:
    def test_identical_one(self):
        ep = _linprof(32)
        assert profile_correlation(ep, ep) == pytest.approx(1.0, abs=1e-5)

    def test_in_range(self):
        ep1 = _prof(32, val=1.0)
        ep2 = _linprof(32)
        assert 0.0 <= profile_correlation(ep1, ep2) <= 1.0

    def test_returns_float(self):
        assert isinstance(profile_correlation(_prof(16), _prof(16)), float)

    def test_opposite_signals_low(self):
        s1 = np.linspace(0, 1, 32, dtype=np.float32)
        s2 = np.linspace(1, 0, 32, dtype=np.float32)
        ep1 = EdgeProfile(signal=s1, side=0, n_samples=32, method="intensity")
        ep2 = EdgeProfile(signal=s2, side=0, n_samples=32, method="intensity")
        assert profile_correlation(ep1, ep2) < 0.5

    def test_constant_profiles_in_range(self):
        ep1 = _prof(16, val=5.0)
        ep2 = _prof(16, val=5.0)
        assert 0.0 <= profile_correlation(ep1, ep2) <= 1.0


# ─── TestProfileDtwExtra ─────────────────────────────────────────────────────

class TestProfileDtwExtra:
    def test_identical_one(self):
        ep = _linprof(16)
        assert profile_dtw(ep, ep) == pytest.approx(1.0, abs=1e-5)

    def test_in_range(self):
        ep1 = _prof(16, val=0.5)
        ep2 = _prof(16, val=0.0)
        assert 0.0 <= profile_dtw(ep1, ep2) <= 1.0

    def test_returns_float(self):
        assert isinstance(profile_dtw(_prof(8), _prof(8)), float)

    def test_with_window(self):
        ep = _linprof(16)
        assert 0.0 <= profile_dtw(ep, ep, window=4) <= 1.0

    def test_empty_zero(self):
        ep = EdgeProfile(signal=np.array([], dtype=np.float32),
                         side=0, n_samples=0, method="intensity")
        assert profile_dtw(ep, ep) == pytest.approx(0.0)


# ─── TestMatchEdgeProfilesExtra ──────────────────────────────────────────────

class TestMatchEdgeProfilesExtra:
    def test_returns_result(self):
        assert isinstance(match_edge_profiles(_rand(), _rand()), ProfileMatchResult)

    def test_score_in_range(self):
        r = match_edge_profiles(_rand(seed=0), _rand(seed=1))
        assert 0.0 <= r.score <= 1.0

    def test_identical_high_score(self):
        img = _gray()
        assert match_edge_profiles(img, img).score >= 0.4

    @pytest.mark.parametrize("s1,s2", [(0, 2), (1, 3), (2, 0), (3, 1)])
    def test_side_pairs(self, s1, s2):
        img = _rand()
        r = match_edge_profiles(img, img, side1=s1, side2=s2)
        assert 0.0 <= r.score <= 1.0

    def test_params_contain_sides(self):
        r = match_edge_profiles(_rand(), _rand(), side1=1, side2=3)
        assert r.params["side1"] == 1
        assert r.params["side2"] == 3

    def test_rgb_ok(self):
        assert isinstance(match_edge_profiles(_rgb(), _rgb()), ProfileMatchResult)

    def test_correlation_in_range(self):
        r = match_edge_profiles(_rand(seed=0), _rand(seed=1))
        assert 0.0 <= r.correlation <= 1.0

    def test_dtw_score_in_range(self):
        r = match_edge_profiles(_rand(seed=0), _rand(seed=1))
        assert 0.0 <= r.dtw_score <= 1.0


# ─── TestBatchProfileMatchExtra ──────────────────────────────────────────────

class TestBatchProfileMatchExtra:
    def test_returns_list(self):
        imgs = [_rand(seed=i) for i in range(3)]
        result = batch_profile_match(imgs, [(0, 0, 1, 2)])
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_rand(seed=i) for i in range(3)]
        pairs = [(0, 0, 1, 2), (0, 1, 2, 3)]
        assert len(batch_profile_match(imgs, pairs)) == 2

    def test_empty_pairs(self):
        assert batch_profile_match([_rand()], []) == []

    def test_all_results(self):
        imgs = [_rand(seed=i) for i in range(2)]
        for r in batch_profile_match(imgs, [(0, 0, 1, 2)]):
            assert isinstance(r, ProfileMatchResult)

    def test_scores_in_range(self):
        imgs = [_rand(seed=i) for i in range(3)]
        for r in batch_profile_match(imgs, [(0, 0, 1, 2), (0, 1, 2, 0)]):
            assert 0.0 <= r.score <= 1.0

    def test_three_pairs(self):
        imgs = [_rand(seed=i) for i in range(3)]
        pairs = [(0, 0, 1, 2), (0, 1, 2, 0), (1, 1, 2, 3)]
        assert len(batch_profile_match(imgs, pairs)) == 3
