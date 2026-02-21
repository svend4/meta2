"""Тесты для puzzle_reconstruction/algorithms/edge_profile.py."""
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


# ─── EdgeProfile ──────────────────────────────────────────────────────────────

class TestEdgeProfile:
    def test_fields(self):
        sig = np.zeros(64, dtype=np.float32)
        p = EdgeProfile(signal=sig, side=1, n_samples=64, method="intensity")
        assert p.side == 1
        assert p.n_samples == 64
        assert p.method == "intensity"
        assert isinstance(p.params, dict)

    def test_repr(self):
        p = _make_profile(val=100.0)
        s = repr(p)
        assert "EdgeProfile" in s
        assert "intensity" in s

    def test_signal_shape(self):
        p = _make_profile(n=32)
        assert p.signal.shape == (32,)

    def test_signal_dtype(self):
        p = _make_profile()
        assert p.signal.dtype == np.float32


# ─── ProfileMatchResult ───────────────────────────────────────────────────────

class TestProfileMatchResult:
    def test_fields(self):
        r = ProfileMatchResult(score=0.7, correlation=0.8, dtw_score=0.6)
        assert r.score == pytest.approx(0.7)
        assert r.correlation == pytest.approx(0.8)
        assert r.dtw_score == pytest.approx(0.6)
        assert r.method == "profile"
        assert isinstance(r.params, dict)

    def test_repr(self):
        r = ProfileMatchResult(score=0.5, correlation=0.5, dtw_score=0.5)
        s = repr(r)
        assert "ProfileMatchResult" in s
        assert "0.5" in s

    def test_score_in_range(self):
        r = ProfileMatchResult(score=0.9, correlation=0.95, dtw_score=0.85)
        assert 0.0 <= r.score <= 1.0
        assert 0.0 <= r.correlation <= 1.0
        assert 0.0 <= r.dtw_score <= 1.0


# ─── extract_intensity_profile ───────────────────────────────────────────────

class TestExtractIntensityProfile:
    def test_returns_edge_profile(self):
        assert isinstance(extract_intensity_profile(_noisy(), side=1), EdgeProfile)

    def test_method_name(self):
        p = extract_intensity_profile(_noisy(), side=0)
        assert p.method == "intensity"

    def test_n_samples_default(self):
        p = extract_intensity_profile(_noisy(), side=1)
        assert p.n_samples == 64
        assert len(p.signal) == 64

    def test_n_samples_custom(self):
        p = extract_intensity_profile(_noisy(), side=1, n_samples=32)
        assert p.n_samples == 32
        assert len(p.signal) == 32

    def test_signal_dtype(self):
        assert extract_intensity_profile(_noisy(), side=1).signal.dtype == np.float32

    def test_side_stored(self):
        for s in range(4):
            p = extract_intensity_profile(_noisy(), side=s)
            assert p.side == s

    def test_bgr_input(self):
        p = extract_intensity_profile(_bgr(), side=1)
        assert len(p.signal) == 64

    def test_constant_image_uniform_signal(self):
        p = extract_intensity_profile(_gray(val=100), side=0)
        # All pixels equal → uniform profile
        assert np.allclose(p.signal, p.signal[0], atol=1.0)

    def test_signal_in_valid_range(self):
        p = extract_intensity_profile(_noisy(), side=1)
        assert p.signal.min() >= 0.0
        assert p.signal.max() <= 255.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        p = extract_intensity_profile(_noisy(64, 64), side=side)
        assert isinstance(p, EdgeProfile)
        assert len(p.signal) == 64


# ─── extract_gradient_profile ─────────────────────────────────────────────────

class TestExtractGradientProfile:
    def test_returns_edge_profile(self):
        assert isinstance(extract_gradient_profile(_noisy(), side=1), EdgeProfile)

    def test_method_name(self):
        assert extract_gradient_profile(_noisy(), side=1).method == "gradient"

    def test_n_samples(self):
        p = extract_gradient_profile(_noisy(), side=2, n_samples=48)
        assert len(p.signal) == 48

    def test_dtype(self):
        assert extract_gradient_profile(_noisy(), side=1).signal.dtype == np.float32

    def test_constant_near_zero(self):
        p = extract_gradient_profile(_gray(), side=1)
        # Constant image → zero gradient everywhere
        assert np.all(p.signal >= 0.0)
        assert p.signal.max() < 10.0

    def test_nonneg_signal(self):
        p = extract_gradient_profile(_noisy(), side=1)
        assert np.all(p.signal >= 0.0)

    def test_bgr_input(self):
        p = extract_gradient_profile(_bgr(), side=0)
        assert len(p.signal) == 64

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        p = extract_gradient_profile(_noisy(), side=side)
        assert isinstance(p, EdgeProfile)
        assert len(p.signal) == 64


# ─── extract_texture_profile ──────────────────────────────────────────────────

class TestExtractTextureProfile:
    def test_returns_edge_profile(self):
        assert isinstance(extract_texture_profile(_noisy(), side=1), EdgeProfile)

    def test_method_name(self):
        assert extract_texture_profile(_noisy(), side=1).method == "texture"

    def test_n_samples(self):
        p = extract_texture_profile(_noisy(), side=3, n_samples=32)
        assert len(p.signal) == 32

    def test_dtype(self):
        assert extract_texture_profile(_noisy(), side=1).signal.dtype == np.float32

    def test_constant_near_zero(self):
        p = extract_texture_profile(_gray(), side=0)
        # Constant image → zero local std
        assert p.signal.max() < 5.0

    def test_nonneg_signal(self):
        p = extract_texture_profile(_noisy(), side=1)
        assert np.all(p.signal >= 0.0)

    def test_window_param_stored(self):
        p = extract_texture_profile(_noisy(), side=1, window=16)
        assert p.params.get("window") == 16

    def test_bgr_input(self):
        p = extract_texture_profile(_bgr(), side=1)
        assert len(p.signal) == 64

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        p = extract_texture_profile(_noisy(), side=side)
        assert isinstance(p, EdgeProfile)


# ─── normalize_profile ────────────────────────────────────────────────────────

class TestNormalizeProfile:
    def test_returns_edge_profile(self):
        p = _make_profile(val=100.0)
        n = normalize_profile(p)
        assert isinstance(n, EdgeProfile)

    def test_method_contains_normalized(self):
        p = _make_profile()
        n = normalize_profile(p)
        assert "normalized" in n.method

    def test_constant_profile_zero(self):
        p = _make_profile(n=16, val=128.0)
        n = normalize_profile(p)
        assert np.allclose(n.signal, 0.0, atol=1e-6)

    def test_zero_mean_after_normalize(self):
        sig = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        p = EdgeProfile(signal=sig, side=1, n_samples=8, method="intensity")
        n = normalize_profile(p)
        assert n.signal.mean() == pytest.approx(0.0, abs=1e-5)

    def test_unit_std_after_normalize(self):
        sig = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        p = EdgeProfile(signal=sig, side=1, n_samples=8, method="intensity")
        n = normalize_profile(p)
        assert n.signal.std() == pytest.approx(1.0, abs=1e-4)

    def test_n_samples_preserved(self):
        p = _make_profile(n=32)
        n = normalize_profile(p)
        assert n.n_samples == 32

    def test_side_preserved(self):
        p = _make_profile(side=2)
        n = normalize_profile(p)
        assert n.side == 2

    def test_dtype_float32(self):
        p = _make_profile()
        n = normalize_profile(p)
        assert n.signal.dtype == np.float32

    def test_params_normalized_flag(self):
        p = _make_profile()
        n = normalize_profile(p)
        assert n.params.get("normalized") is True


# ─── profile_correlation ──────────────────────────────────────────────────────

class TestProfileCorrelation:
    def _make_from_array(self, arr, side=1):
        sig = arr.astype(np.float32)
        return EdgeProfile(signal=sig, side=side, n_samples=len(sig), method="intensity")

    def test_same_profile_is_one(self):
        p = self._make_from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32))
        assert profile_correlation(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_range(self):
        p1 = self._make_from_array(np.random.rand(32).astype(np.float32) * 255)
        p2 = self._make_from_array(np.random.rand(32).astype(np.float32) * 255)
        c = profile_correlation(p1, p2)
        assert 0.0 <= c <= 1.0

    def test_constant_profiles_same(self):
        p1 = _make_profile(n=16, val=100.0)
        p2 = _make_profile(n=16, val=100.0)
        assert profile_correlation(p1, p2) == pytest.approx(1.0, abs=1e-4)

    def test_constant_profiles_different(self):
        p1 = _make_profile(n=16, val=100.0)
        p2 = _make_profile(n=16, val=200.0)
        # Both constant → uncorrelated, return 0.5
        assert profile_correlation(p1, p2) == pytest.approx(0.5, abs=1e-4)

    def test_returns_float(self):
        p1 = self._make_from_array(np.arange(16, dtype=np.float32))
        p2 = self._make_from_array(np.arange(16, 0, -1, dtype=np.float32))
        assert isinstance(profile_correlation(p1, p2), float)

    def test_empty_profile(self):
        p = EdgeProfile(signal=np.array([], dtype=np.float32), side=1, n_samples=0, method="intensity")
        c = profile_correlation(p, p)
        assert c == pytest.approx(0.5, abs=1e-4)

    def test_identical_noisy_profiles(self):
        sig = np.random.rand(64).astype(np.float32) * 200
        p = EdgeProfile(signal=sig, side=1, n_samples=64, method="intensity")
        assert profile_correlation(p, p) == pytest.approx(1.0, abs=1e-4)


# ─── profile_dtw ──────────────────────────────────────────────────────────────

class TestProfileDtw:
    def _make_from_array(self, arr, side=1):
        sig = arr.astype(np.float32)
        return EdgeProfile(signal=sig, side=side, n_samples=len(sig), method="intensity")

    def test_same_profile_is_one(self):
        p = self._make_from_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert profile_dtw(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_range(self):
        p1 = self._make_from_array(np.random.rand(16).astype(np.float32))
        p2 = self._make_from_array(np.random.rand(16).astype(np.float32))
        d = profile_dtw(p1, p2)
        assert 0.0 <= d <= 1.0

    def test_returns_float(self):
        p1 = self._make_from_array(np.arange(8, dtype=np.float32))
        p2 = self._make_from_array(np.arange(8, 0, -1, dtype=np.float32))
        assert isinstance(profile_dtw(p1, p2), float)

    def test_empty_profile_zero(self):
        p = EdgeProfile(signal=np.array([], dtype=np.float32), side=1, n_samples=0, method="x")
        assert profile_dtw(p, p) == pytest.approx(0.0, abs=1e-4)

    def test_identical_is_one(self):
        sig = np.random.rand(32).astype(np.float32)
        p = EdgeProfile(signal=sig, side=1, n_samples=32, method="intensity")
        assert profile_dtw(p, p) == pytest.approx(1.0, abs=1e-4)

    def test_window_param(self):
        p1 = self._make_from_array(np.random.rand(16).astype(np.float32))
        p2 = self._make_from_array(np.random.rand(16).astype(np.float32))
        d = profile_dtw(p1, p2, window=4)
        assert 0.0 <= d <= 1.0


# ─── match_edge_profiles ──────────────────────────────────────────────────────

class TestMatchEdgeProfiles:
    def test_returns_result(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=99))
        assert isinstance(r, ProfileMatchResult)

    def test_score_in_range(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.score <= 1.0

    def test_correlation_in_range(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.correlation <= 1.0

    def test_dtw_score_in_range(self):
        r = match_edge_profiles(_noisy(), _noisy(seed=5))
        assert 0.0 <= r.dtw_score <= 1.0

    def test_method_name(self):
        r = match_edge_profiles(_noisy(), _noisy())
        assert r.method == "profile"

    def test_side_params_stored(self):
        r = match_edge_profiles(_noisy(), _noisy(), side1=0, side2=2)
        assert r.params.get("side1") == 0
        assert r.params.get("side2") == 2

    def test_n_samples_param(self):
        r = match_edge_profiles(_noisy(), _noisy(), n_samples=32)
        assert r.params.get("n_samples") == 32

    def test_weights_stored(self):
        r = match_edge_profiles(_noisy(), _noisy(),
                                 w_intensity=0.5, w_gradient=0.3, w_texture=0.2)
        assert r.params.get("w_intensity") == pytest.approx(0.5)
        assert r.params.get("w_gradient") == pytest.approx(0.3)
        assert r.params.get("w_texture") == pytest.approx(0.2)

    def test_identical_high_score(self):
        img = _noisy()
        r = match_edge_profiles(img, img, side1=1, side2=1)
        assert r.score > 0.5

    def test_gray_input(self):
        r = match_edge_profiles(_gray(), _gray())
        assert isinstance(r, ProfileMatchResult)

    def test_bgr_input(self):
        r = match_edge_profiles(_bgr(), _bgr())
        assert 0.0 <= r.score <= 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = match_edge_profiles(_noisy(), _noisy(seed=7), side1=side, side2=side)
        assert 0.0 <= r.score <= 1.0


# ─── batch_profile_match ──────────────────────────────────────────────────────

class TestBatchProfileMatch:
    def test_returns_list(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        pairs = [(0, 1, 1, 3), (0, 0, 2, 2)]
        results = batch_profile_match(imgs, pairs)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_result(self):
        imgs = [_noisy(), _noisy(seed=1)]
        for r in batch_profile_match(imgs, [(0, 1, 1, 3)]):
            assert isinstance(r, ProfileMatchResult)

    def test_empty_pairs(self):
        imgs = [_noisy()]
        assert batch_profile_match(imgs, []) == []

    def test_scores_in_range(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        pairs = [(0, 1, 1, 3), (1, 0, 2, 2), (2, 1, 3, 3)]
        for r in batch_profile_match(imgs, pairs):
            assert 0.0 <= r.score <= 1.0

    def test_n_samples_forwarded(self):
        imgs = [_noisy(), _noisy(seed=1)]
        results = batch_profile_match(imgs, [(0, 1, 1, 3)], n_samples=32)
        assert results[0].params.get("n_samples") == 32

    def test_identical_pair_high_score(self):
        img = _noisy()
        results = batch_profile_match([img, img], [(0, 1, 1, 1)])
        assert results[0].score > 0.5
