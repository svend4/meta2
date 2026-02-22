"""Тесты для puzzle_reconstruction/matching/texture_match.py."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.texture_match import (
    TextureMatchResult,
    compute_lbp_histogram,
    lbp_distance,
    compute_gabor_features,
    gabor_distance,
    gradient_orientation_hist,
    texture_match_pair,
    texture_compatibility_matrix,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def make_bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── TextureMatchResult ───────────────────────────────────────────────────────

class TestTextureMatchResult:
    def test_creation(self):
        r = TextureMatchResult(score=0.7, lbp_score=0.8,
                               gabor_score=0.6, gradient_score=0.7)
        assert r.score == pytest.approx(0.7)
        assert r.lbp_score == pytest.approx(0.8)
        assert r.gabor_score == pytest.approx(0.6)
        assert r.gradient_score == pytest.approx(0.7)
        assert r.method == "texture"
        assert r.params == {}

    def test_method_default(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5)
        assert r.method == "texture"

    def test_params_stored(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5,
                               params={"side1": 1})
        assert r.params["side1"] == 1

    def test_no_validation_on_creation(self):
        # No validation in __post_init__, so any float is accepted
        r = TextureMatchResult(score=2.0, lbp_score=-1.0,
                               gabor_score=0.5, gradient_score=0.5)
        assert r.score == pytest.approx(2.0)


# ─── compute_lbp_histogram ────────────────────────────────────────────────────

class TestComputeLbpHistogram:
    def test_returns_float32(self):
        gray = make_noisy()
        hist = compute_lbp_histogram(gray)
        assert hist.dtype == np.float32

    def test_shape_default_bins(self):
        gray = make_noisy()
        hist = compute_lbp_histogram(gray)
        assert hist.shape == (64,)

    def test_shape_custom_bins(self):
        gray = make_noisy()
        hist = compute_lbp_histogram(gray, bins=32)
        assert hist.shape == (32,)

    def test_sum_equals_1_for_noisy_image(self):
        gray = make_noisy()
        hist = compute_lbp_histogram(gray)
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_uniform_image_sums_to_1_or_0(self):
        gray = make_gray(fill=128)
        hist = compute_lbp_histogram(gray)
        s = hist.sum()
        assert s == pytest.approx(0.0, abs=1e-5) or s == pytest.approx(1.0, abs=1e-5)

    def test_all_nonneg(self):
        gray = make_noisy()
        hist = compute_lbp_histogram(gray)
        assert np.all(hist >= 0.0)

    def test_custom_radius_and_points(self):
        gray = make_noisy()
        hist = compute_lbp_histogram(gray, radius=2, n_points=12, bins=64)
        assert hist.shape == (64,)

    def test_accepts_float_image(self):
        gray = make_noisy().astype(np.float32)
        hist = compute_lbp_histogram(gray)
        assert hist.shape == (64,)


# ─── lbp_distance ─────────────────────────────────────────────────────────────

class TestLbpDistance:
    def _make_hist(self, seed=0, bins=64):
        rng = np.random.default_rng(seed)
        h = rng.uniform(0, 1, bins).astype(np.float32)
        h /= h.sum()
        return h

    def test_same_hist_chi2_score_is_1(self):
        h = self._make_hist()
        score = lbp_distance(h, h, metric="chi2")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_same_hist_bhatt_score_is_1(self):
        h = self._make_hist()
        score = lbp_distance(h, h, metric="bhatt")
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_same_hist_corr_score_is_1(self):
        h = self._make_hist()
        score = lbp_distance(h, h, metric="corr")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_chi2_in_0_1(self):
        h1 = self._make_hist(seed=0)
        h2 = self._make_hist(seed=1)
        score = lbp_distance(h1, h2, metric="chi2")
        assert 0.0 <= score <= 1.0

    def test_bhatt_in_0_1(self):
        h1 = self._make_hist(seed=0)
        h2 = self._make_hist(seed=1)
        score = lbp_distance(h1, h2, metric="bhatt")
        assert 0.0 <= score <= 1.0

    def test_corr_in_0_1(self):
        h1 = self._make_hist(seed=0)
        h2 = self._make_hist(seed=1)
        score = lbp_distance(h1, h2, metric="corr")
        assert 0.0 <= score <= 1.0

    def test_unknown_metric_raises(self):
        h = self._make_hist()
        with pytest.raises(ValueError, match="Unknown"):
            lbp_distance(h, h, metric="euclidean")

    def test_returns_float(self):
        h1 = self._make_hist(seed=0)
        h2 = self._make_hist(seed=1)
        score = lbp_distance(h1, h2)
        assert isinstance(score, float)


# ─── compute_gabor_features ───────────────────────────────────────────────────

class TestComputeGaborFeatures:
    def test_returns_float32(self):
        gray = make_noisy()
        feats = compute_gabor_features(gray)
        assert feats.dtype == np.float32

    def test_default_length(self):
        # 2 frequencies × 4 orientations × 2 stats = 16
        gray = make_noisy()
        feats = compute_gabor_features(gray)
        assert feats.shape == (16,)

    def test_custom_frequencies_length(self):
        gray = make_noisy()
        feats = compute_gabor_features(gray, frequencies=(0.1, 0.2, 0.3))
        # 3 × 4 × 2 = 24
        assert feats.shape == (24,)

    def test_custom_orientations_length(self):
        gray = make_noisy()
        feats = compute_gabor_features(gray, frequencies=(0.1,), n_orientations=6)
        # 1 × 6 × 2 = 12
        assert feats.shape == (12,)

    def test_uniform_image_accepted(self):
        gray = make_gray(fill=128)
        feats = compute_gabor_features(gray)
        assert feats.shape == (16,)

    def test_single_frequency_single_orientation(self):
        gray = make_noisy()
        feats = compute_gabor_features(gray, frequencies=(0.2,), n_orientations=1)
        assert feats.shape == (2,)


# ─── gabor_distance ───────────────────────────────────────────────────────────

class TestGaborDistance:
    def test_same_features_returns_1(self):
        gray = make_noisy()
        feats = compute_gabor_features(gray)
        score = gabor_distance(feats, feats)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_result_in_0_1(self):
        f1 = compute_gabor_features(make_noisy(seed=0))
        f2 = compute_gabor_features(make_noisy(seed=1))
        score = gabor_distance(f1, f2)
        assert 0.0 <= score <= 1.0

    def test_empty_returns_0(self):
        score = gabor_distance(np.array([], dtype=np.float32),
                               np.array([], dtype=np.float32))
        assert score == pytest.approx(0.0)

    def test_returns_float(self):
        f1 = compute_gabor_features(make_noisy(seed=0))
        f2 = compute_gabor_features(make_noisy(seed=1))
        score = gabor_distance(f1, f2)
        assert isinstance(score, float)


# ─── gradient_orientation_hist ────────────────────────────────────────────────

class TestGradientOrientationHist:
    def test_returns_float32(self):
        gray = make_noisy()
        hist = gradient_orientation_hist(gray)
        assert hist.dtype == np.float32

    def test_default_shape(self):
        gray = make_noisy()
        hist = gradient_orientation_hist(gray)
        assert hist.shape == (8,)

    def test_custom_bins(self):
        gray = make_noisy()
        hist = gradient_orientation_hist(gray, bins=16)
        assert hist.shape == (16,)

    def test_all_nonneg(self):
        gray = make_noisy()
        hist = gradient_orientation_hist(gray)
        assert np.all(hist >= 0.0)

    def test_sum_1_or_0(self):
        gray = make_noisy()
        hist = gradient_orientation_hist(gray)
        s = hist.sum()
        assert s == pytest.approx(0.0, abs=1e-5) or s == pytest.approx(1.0, abs=1e-5)

    def test_uniform_image(self):
        gray = make_gray(fill=100)
        hist = gradient_orientation_hist(gray)
        assert hist.shape == (8,)


# ─── texture_match_pair ───────────────────────────────────────────────────────

class TestTextureMatchPair:
    def test_returns_texture_match_result(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2)
        assert isinstance(r, TextureMatchResult)

    def test_method_is_texture(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2)
        assert r.method == "texture"

    def test_score_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2)
        assert 0.0 <= r.score <= 1.0

    def test_lbp_score_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2)
        assert 0.0 <= r.lbp_score <= 1.0

    def test_gabor_score_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2)
        assert 0.0 <= r.gabor_score <= 1.0

    def test_gradient_score_in_0_1(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2)
        assert 0.0 <= r.gradient_score <= 1.0

    def test_params_contain_sides(self):
        img1 = make_noisy()
        img2 = make_noisy(seed=1)
        r = texture_match_pair(img1, img2, side1=2, side2=0)
        assert r.params["side1"] == 2
        assert r.params["side2"] == 0

    def test_accepts_bgr_images(self):
        img1 = make_bgr()
        img2 = make_bgr(fill=100)
        r = texture_match_pair(img1, img2)
        assert isinstance(r, TextureMatchResult)

    def test_same_image_high_score(self):
        img = make_noisy()
        r = texture_match_pair(img, img, side1=1, side2=1)
        assert r.score > 0.5

    def test_all_sides_valid(self):
        img1 = make_noisy(h=64, w=64)
        img2 = make_noisy(h=64, w=64, seed=1)
        for s in range(4):
            r = texture_match_pair(img1, img2, side1=s, side2=(s+2) % 4)
            assert 0.0 <= r.score <= 1.0


# ─── texture_compatibility_matrix ─────────────────────────────────────────────

class TestTextureCompatibilityMatrix:
    def test_single_image_returns_1x1(self):
        result = texture_compatibility_matrix([make_noisy()])
        assert result.shape == (1, 1)

    def test_diagonal_is_1(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = texture_compatibility_matrix(images)
        np.testing.assert_allclose(np.diag(result), 1.0)

    def test_symmetric(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = texture_compatibility_matrix(images)
        np.testing.assert_allclose(result, result.T, atol=1e-6)

    def test_returns_float32(self):
        images = [make_noisy(seed=i) for i in range(2)]
        result = texture_compatibility_matrix(images)
        assert result.dtype == np.float32

    def test_shape_nxn(self):
        n = 4
        images = [make_noisy(seed=i) for i in range(n)]
        result = texture_compatibility_matrix(images)
        assert result.shape == (n, n)

    def test_values_in_0_1(self):
        images = [make_noisy(seed=i) for i in range(3)]
        result = texture_compatibility_matrix(images)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_two_images(self):
        img1 = make_noisy(seed=0)
        img2 = make_noisy(seed=1)
        result = texture_compatibility_matrix([img1, img2])
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)
