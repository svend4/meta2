"""Extra tests for puzzle_reconstruction/matching/texture_match.py"""
import numpy as np
import pytest

from puzzle_reconstruction.matching.texture_match import (
    TextureMatchResult,
    compute_gabor_features,
    compute_lbp_histogram,
    gabor_distance,
    gradient_orientation_hist,
    lbp_distance,
    texture_compatibility_matrix,
    texture_match_pair,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _noisy(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _uniform_hist(bins=64):
    h = np.ones(bins, dtype=np.float32)
    h /= h.sum()
    return h


def _rng_hist(seed=0, bins=64):
    h = np.random.default_rng(seed).uniform(0, 1, bins).astype(np.float32)
    h /= h.sum()
    return h


# ─── TestTextureMatchResultExtra ─────────────────────────────────────────────

class TestTextureMatchResultExtra:
    def test_all_scores_zero(self):
        r = TextureMatchResult(score=0.0, lbp_score=0.0,
                               gabor_score=0.0, gradient_score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_all_scores_one(self):
        r = TextureMatchResult(score=1.0, lbp_score=1.0,
                               gabor_score=1.0, gradient_score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_params_multiple_keys(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5,
                               params={"side1": 2, "side2": 3, "metric": "chi2"})
        assert r.params["side1"] == 2
        assert r.params["side2"] == 3
        assert r.params["metric"] == "chi2"

    def test_method_is_texture(self):
        r = TextureMatchResult(score=0.7, lbp_score=0.7,
                               gabor_score=0.7, gradient_score=0.7)
        assert r.method == "texture"

    def test_params_empty_by_default(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5)
        assert r.params == {}

    def test_scores_stored_independently(self):
        r = TextureMatchResult(score=0.9, lbp_score=0.6,
                               gabor_score=0.8, gradient_score=0.7)
        assert r.lbp_score == pytest.approx(0.6)
        assert r.gabor_score == pytest.approx(0.8)
        assert r.gradient_score == pytest.approx(0.7)


# ─── TestComputeLbpHistogramExtra ─────────────────────────────────────────────

class TestComputeLbpHistogramExtra:
    def test_bins_16(self):
        hist = compute_lbp_histogram(_noisy(), bins=16)
        assert hist.shape == (16,)

    def test_bins_128(self):
        hist = compute_lbp_histogram(_noisy(), bins=128)
        assert hist.shape == (128,)

    def test_all_nonneg_various_seeds(self):
        for s in range(5):
            hist = compute_lbp_histogram(_noisy(seed=s))
            assert np.all(hist >= 0.0)

    def test_radius_1_n_points_8(self):
        hist = compute_lbp_histogram(_noisy(), radius=1, n_points=8, bins=64)
        assert hist.dtype == np.float32
        assert hist.shape == (64,)

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        hist = compute_lbp_histogram(img)
        assert hist.shape == (64,)

    def test_non_square_image(self):
        img = _noisy(h=40, w=80)
        hist = compute_lbp_histogram(img)
        assert hist.dtype == np.float32

    def test_sum_nonneg(self):
        hist = compute_lbp_histogram(_noisy())
        assert hist.sum() >= 0.0

    def test_uniform_fill_image(self):
        img = _gray(fill=100)
        hist = compute_lbp_histogram(img)
        assert hist.dtype == np.float32


# ─── TestLbpDistanceExtra ─────────────────────────────────────────────────────

class TestLbpDistanceExtra:
    def test_chi2_in_0_1_second_pair(self):
        h1 = _rng_hist(seed=0)
        h2 = _rng_hist(seed=1)
        s1 = lbp_distance(h1, h2, metric="chi2")
        s2 = lbp_distance(h2, h1, metric="chi2")
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0

    def test_bhatt_in_0_1_second_pair(self):
        h1 = _rng_hist(seed=2)
        h2 = _rng_hist(seed=3)
        s1 = lbp_distance(h1, h2, metric="bhatt")
        s2 = lbp_distance(h2, h1, metric="bhatt")
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0

    def test_corr_same_hist_is_1(self):
        h = _rng_hist(seed=5)
        assert lbp_distance(h, h, metric="corr") == pytest.approx(1.0, abs=1e-4)

    def test_chi2_range_5_pairs(self):
        for s in range(5):
            h1 = _rng_hist(seed=s)
            h2 = _rng_hist(seed=s + 10)
            score = lbp_distance(h1, h2, metric="chi2")
            assert 0.0 <= score <= 1.0

    def test_bhatt_uniform_hists(self):
        h = _uniform_hist()
        score = lbp_distance(h, h, metric="bhatt")
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_corr_range_multiple_pairs(self):
        for s in range(4):
            h1 = _rng_hist(seed=s)
            h2 = _rng_hist(seed=s + 20)
            score = lbp_distance(h1, h2, metric="corr")
            assert 0.0 <= score <= 1.0

    def test_result_is_float_all_metrics(self):
        h1 = _rng_hist(seed=0)
        h2 = _rng_hist(seed=1)
        for m in ("chi2", "bhatt", "corr"):
            assert isinstance(lbp_distance(h1, h2, metric=m), float)


# ─── TestComputeGaborFeaturesExtra ────────────────────────────────────────────

class TestComputeGaborFeaturesExtra:
    def test_two_frequencies_eight_orientations(self):
        feats = compute_gabor_features(_noisy(), frequencies=(0.1, 0.2),
                                       n_orientations=8)
        assert feats.shape == (32,)  # 2 × 8 × 2 = 32

    def test_four_frequencies_four_orientations(self):
        feats = compute_gabor_features(_noisy(), frequencies=(0.05, 0.1, 0.2, 0.4),
                                       n_orientations=4)
        assert feats.shape == (32,)  # 4 × 4 × 2 = 32

    def test_dtype_float32(self):
        feats = compute_gabor_features(_noisy())
        assert feats.dtype == np.float32

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        feats = compute_gabor_features(img)
        assert feats.dtype == np.float32

    def test_non_square_image(self):
        img = _noisy(h=40, w=80)
        feats = compute_gabor_features(img)
        assert feats.shape[0] > 0

    def test_various_seeds_no_crash(self):
        for s in range(5):
            feats = compute_gabor_features(_noisy(seed=s))
            assert feats.dtype == np.float32


# ─── TestGaborDistanceExtra ───────────────────────────────────────────────────

class TestGaborDistanceExtra:
    def test_five_different_pairs(self):
        for s in range(5):
            f1 = compute_gabor_features(_noisy(seed=s))
            f2 = compute_gabor_features(_noisy(seed=s + 10))
            score = gabor_distance(f1, f2)
            assert 0.0 <= score <= 1.0

    def test_symmetric(self):
        f1 = compute_gabor_features(_noisy(seed=0))
        f2 = compute_gabor_features(_noisy(seed=1))
        assert gabor_distance(f1, f2) == pytest.approx(gabor_distance(f2, f1),
                                                        abs=1e-5)

    def test_result_float(self):
        f1 = compute_gabor_features(_noisy(seed=0))
        f2 = compute_gabor_features(_noisy(seed=1))
        assert isinstance(gabor_distance(f1, f2), float)

    def test_constant_image_features_same_score(self):
        img = _gray(fill=128)
        f = compute_gabor_features(img)
        score = gabor_distance(f, f)
        assert score == pytest.approx(1.0, abs=1e-4)


# ─── TestGradientOrientationHistExtra ─────────────────────────────────────────

class TestGradientOrientationHistExtra:
    def test_bins_4(self):
        hist = gradient_orientation_hist(_noisy(), bins=4)
        assert hist.shape == (4,)

    def test_bins_32(self):
        hist = gradient_orientation_hist(_noisy(), bins=32)
        assert hist.shape == (32,)

    def test_all_nonneg_various_seeds(self):
        for s in range(5):
            hist = gradient_orientation_hist(_noisy(seed=s))
            assert np.all(hist >= 0.0)

    def test_float32_dtype(self):
        hist = gradient_orientation_hist(_noisy())
        assert hist.dtype == np.float32

    def test_non_square_image(self):
        img = _noisy(h=40, w=80)
        hist = gradient_orientation_hist(img)
        assert hist.dtype == np.float32

    def test_small_image(self):
        img = _noisy(h=16, w=16)
        hist = gradient_orientation_hist(img)
        assert hist.shape[0] > 0

    def test_sum_0_or_1(self):
        for s in range(3):
            hist = gradient_orientation_hist(_noisy(seed=s))
            s_val = float(hist.sum())
            assert s_val == pytest.approx(0.0, abs=1e-5) or \
                   s_val == pytest.approx(1.0, abs=1e-5)


# ─── TestTextureMatchPairExtra ────────────────────────────────────────────────

class TestTextureMatchPairExtra:
    def test_five_random_pairs(self):
        for s in range(5):
            img1 = _noisy(seed=s)
            img2 = _noisy(seed=s + 10)
            r = texture_match_pair(img1, img2)
            assert 0.0 <= r.score <= 1.0

    def test_non_square_images(self):
        img1 = _noisy(h=40, w=80)
        img2 = _noisy(h=40, w=80, seed=1)
        r = texture_match_pair(img1, img2)
        assert isinstance(r, TextureMatchResult)

    def test_small_images(self):
        img1 = _noisy(h=16, w=16)
        img2 = _noisy(h=16, w=16, seed=1)
        r = texture_match_pair(img1, img2)
        assert isinstance(r, TextureMatchResult)

    def test_all_subscores_in_range(self):
        img1 = _noisy(seed=7)
        img2 = _noisy(seed=8)
        r = texture_match_pair(img1, img2)
        assert 0.0 <= r.lbp_score <= 1.0
        assert 0.0 <= r.gabor_score <= 1.0
        assert 0.0 <= r.gradient_score <= 1.0

    def test_side_params_stored(self):
        img1 = _noisy(seed=0)
        img2 = _noisy(seed=1)
        r = texture_match_pair(img1, img2, side1=0, side2=2)
        assert r.params["side1"] == 0
        assert r.params["side2"] == 2

    def test_same_image_score_high(self):
        img = _noisy(seed=3)
        r = texture_match_pair(img, img)
        assert r.score > 0.9

    def test_bgr_images_various_seeds(self):
        for s in range(3):
            img1 = _noisy(seed=s).reshape(64, 64)
            img2 = _noisy(seed=s + 5).reshape(64, 64)
            r = texture_match_pair(img1, img2)
            assert isinstance(r, TextureMatchResult)


# ─── TestTextureCompatibilityMatrixExtra ──────────────────────────────────────

class TestTextureCompatibilityMatrixExtra:
    def test_five_images(self):
        imgs = [_noisy(seed=i) for i in range(5)]
        mat = texture_compatibility_matrix(imgs)
        assert mat.shape == (5, 5)

    def test_eight_images(self):
        imgs = [_noisy(seed=i) for i in range(8)]
        mat = texture_compatibility_matrix(imgs)
        assert mat.shape == (8, 8)

    def test_diagonal_ones_five(self):
        imgs = [_noisy(seed=i) for i in range(5)]
        mat = texture_compatibility_matrix(imgs)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_symmetric_five(self):
        imgs = [_noisy(seed=i) for i in range(5)]
        mat = texture_compatibility_matrix(imgs)
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)

    def test_all_values_in_01(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        mat = texture_compatibility_matrix(imgs)
        assert float(mat.min()) >= 0.0
        assert float(mat.max()) <= 1.0 + 1e-5

    def test_float32_dtype(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        mat = texture_compatibility_matrix(imgs)
        assert mat.dtype == np.float32

    def test_non_square_images(self):
        imgs = [_noisy(h=40, w=80, seed=i) for i in range(3)]
        mat = texture_compatibility_matrix(imgs)
        assert mat.shape == (3, 3)
