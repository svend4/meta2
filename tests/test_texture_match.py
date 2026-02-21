"""Тесты для puzzle_reconstruction/matching/texture_match.py."""
import numpy as np
import pytest

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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 100
    img[:, :, 2] = 50
    return img


def _stripes(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    img[::4, :] = 255
    return img


# ─── TextureMatchResult ───────────────────────────────────────────────────────

class TestTextureMatchResult:
    def test_fields(self):
        r = TextureMatchResult(score=0.7, lbp_score=0.8,
                                gabor_score=0.6, gradient_score=0.7)
        assert r.score == pytest.approx(0.7)
        assert r.lbp_score == pytest.approx(0.8)
        assert r.gabor_score == pytest.approx(0.6)
        assert r.gradient_score == pytest.approx(0.7)
        assert r.method == "texture"
        assert isinstance(r.params, dict)

    def test_repr(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                                gabor_score=0.5, gradient_score=0.5)
        s = repr(r)
        assert "TextureMatchResult" in s
        assert "0.5" in s

    def test_score_in_range(self):
        r = TextureMatchResult(score=0.9, lbp_score=1.0,
                                gabor_score=0.8, gradient_score=0.9)
        assert 0.0 <= r.score <= 1.0

    def test_sub_scores_in_range(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.3,
                                gabor_score=0.8, gradient_score=0.4)
        assert 0.0 <= r.lbp_score <= 1.0
        assert 0.0 <= r.gabor_score <= 1.0
        assert 0.0 <= r.gradient_score <= 1.0

    def test_custom_params(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                                gabor_score=0.5, gradient_score=0.5,
                                params={"side1": 1, "side2": 3})
        assert r.params["side1"] == 1


# ─── compute_lbp_histogram ────────────────────────────────────────────────────

class TestComputeLbpHistogram:
    def test_returns_array(self):
        h = compute_lbp_histogram(_noisy())
        assert isinstance(h, np.ndarray)

    def test_default_bins(self):
        h = compute_lbp_histogram(_noisy())
        assert len(h) == 64

    def test_custom_bins(self):
        h = compute_lbp_histogram(_noisy(), bins=32)
        assert len(h) == 32

    def test_dtype_float32(self):
        assert compute_lbp_histogram(_noisy()).dtype == np.float32

    def test_sum_one(self):
        h = compute_lbp_histogram(_noisy())
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_nonneg(self):
        assert np.all(compute_lbp_histogram(_noisy()) >= 0)

    def test_bgr_input(self):
        h = compute_lbp_histogram(_bgr())
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_constant_image(self):
        h = compute_lbp_histogram(_gray(val=128))
        assert h.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(h >= 0)

    def test_radius_param(self):
        h1 = compute_lbp_histogram(_stripes(), radius=1)
        h2 = compute_lbp_histogram(_stripes(), radius=2)
        # Different radii → different distributions
        assert not np.allclose(h1, h2)

    def test_n_points_param(self):
        h1 = compute_lbp_histogram(_noisy(), n_points=8)
        h2 = compute_lbp_histogram(_noisy(), n_points=4)
        # Both should be valid histograms
        assert h1.sum() == pytest.approx(1.0, abs=1e-4)
        assert h2.sum() == pytest.approx(1.0, abs=1e-4)

    def test_same_image_identical_histograms(self):
        img = _noisy()
        h1 = compute_lbp_histogram(img)
        h2 = compute_lbp_histogram(img)
        np.testing.assert_allclose(h1, h2)


# ─── lbp_distance ────────────────────────────────────────────────────────────

class TestLbpDistance:
    def test_same_histogram_is_one(self):
        h = compute_lbp_histogram(_noisy())
        assert lbp_distance(h, h, metric="chi2") == pytest.approx(1.0, abs=1e-4)

    def test_same_histogram_bhatt(self):
        h = compute_lbp_histogram(_noisy())
        assert lbp_distance(h, h, metric="bhatt") == pytest.approx(1.0, abs=1e-4)

    def test_same_histogram_corr(self):
        h = compute_lbp_histogram(_noisy())
        assert lbp_distance(h, h, metric="corr") == pytest.approx(1.0, abs=1e-4)

    def test_range_chi2(self):
        h1 = compute_lbp_histogram(_noisy(seed=1))
        h2 = compute_lbp_histogram(_noisy(seed=2))
        d = lbp_distance(h1, h2, metric="chi2")
        assert 0.0 <= d <= 1.0

    def test_range_bhatt(self):
        h1 = compute_lbp_histogram(_noisy(seed=1))
        h2 = compute_lbp_histogram(_noisy(seed=2))
        d = lbp_distance(h1, h2, metric="bhatt")
        assert 0.0 <= d <= 1.0

    def test_range_corr(self):
        h1 = compute_lbp_histogram(_noisy(seed=1))
        h2 = compute_lbp_histogram(_noisy(seed=2))
        d = lbp_distance(h1, h2, metric="corr")
        assert 0.0 <= d <= 1.0

    def test_unknown_metric_raises(self):
        h = compute_lbp_histogram(_noisy())
        with pytest.raises(ValueError):
            lbp_distance(h, h, metric="unknown_xyz")

    def test_returns_float(self):
        h1 = compute_lbp_histogram(_noisy(seed=10))
        h2 = compute_lbp_histogram(_noisy(seed=20))
        assert isinstance(lbp_distance(h1, h2), float)

    @pytest.mark.parametrize("metric", ["chi2", "bhatt", "corr"])
    def test_all_metrics_return_float(self, metric):
        h1 = compute_lbp_histogram(_noisy(seed=10))
        h2 = compute_lbp_histogram(_noisy(seed=20))
        d = lbp_distance(h1, h2, metric=metric)
        assert isinstance(d, float)
        assert 0.0 <= d <= 1.0


# ─── compute_gabor_features ──────────────────────────────────────────────────

class TestComputeGaborFeatures:
    def test_returns_array(self):
        f = compute_gabor_features(_noisy())
        assert isinstance(f, np.ndarray)

    def test_length_default(self):
        # 2 frequencies × 4 orientations × 2 stats = 16
        f = compute_gabor_features(_noisy(), frequencies=(0.1, 0.3), n_orientations=4)
        assert len(f) == 2 * 2 * 4

    def test_length_custom(self):
        f = compute_gabor_features(_noisy(), frequencies=(0.2,), n_orientations=2)
        assert len(f) == 2 * 1 * 2

    def test_dtype_float32(self):
        assert compute_gabor_features(_noisy()).dtype == np.float32

    def test_bgr_input(self):
        f = compute_gabor_features(_bgr())
        assert len(f) == 16

    def test_same_image_same_features(self):
        img = _noisy()
        f1 = compute_gabor_features(img)
        f2 = compute_gabor_features(img)
        np.testing.assert_allclose(f1, f2)

    def test_different_images_different_features(self):
        f1 = compute_gabor_features(_noisy(seed=1))
        f2 = compute_gabor_features(_noisy(seed=99))
        assert not np.allclose(f1, f2)

    def test_constant_image(self):
        f = compute_gabor_features(_gray(val=128))
        assert isinstance(f, np.ndarray)
        assert len(f) == 16


# ─── gabor_distance ───────────────────────────────────────────────────────────

class TestGaborDistance:
    def test_same_features_is_one(self):
        f = compute_gabor_features(_noisy())
        assert gabor_distance(f, f) == pytest.approx(1.0, abs=1e-4)

    def test_empty_features_is_zero(self):
        assert gabor_distance(np.array([]), np.array([])) == pytest.approx(0.0)

    def test_range(self):
        f1 = compute_gabor_features(_noisy(seed=1))
        f2 = compute_gabor_features(_noisy(seed=99))
        d = gabor_distance(f1, f2)
        assert 0.0 <= d <= 1.0

    def test_returns_float(self):
        f1 = compute_gabor_features(_noisy(seed=1))
        f2 = compute_gabor_features(_noisy(seed=2))
        assert isinstance(gabor_distance(f1, f2), float)

    def test_identical_greater_than_different(self):
        img = _noisy()
        f_same = compute_gabor_features(img)
        f_diff = compute_gabor_features(_gray(val=50))
        assert gabor_distance(f_same, f_same) >= gabor_distance(f_same, f_diff)


# ─── gradient_orientation_hist ────────────────────────────────────────────────

class TestGradientOrientationHist:
    def test_returns_array(self):
        h = gradient_orientation_hist(_noisy())
        assert isinstance(h, np.ndarray)

    def test_default_bins(self):
        h = gradient_orientation_hist(_noisy())
        assert len(h) == 8

    def test_custom_bins(self):
        h = gradient_orientation_hist(_noisy(), bins=16)
        assert len(h) == 16

    def test_dtype_float32(self):
        assert gradient_orientation_hist(_noisy()).dtype == np.float32

    def test_sum_one_noisy(self):
        h = gradient_orientation_hist(_noisy())
        assert h.sum() == pytest.approx(1.0, abs=1e-4)

    def test_nonneg(self):
        assert np.all(gradient_orientation_hist(_noisy()) >= 0)

    def test_constant_image_zero(self):
        h = gradient_orientation_hist(_gray())
        # Constant image → zero gradients → zero histogram
        assert h.sum() == pytest.approx(0.0, abs=1e-4)

    def test_bgr_input(self):
        h = gradient_orientation_hist(_bgr())
        assert len(h) == 8

    def test_vertical_stripes_orientation(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, ::4] = 255
        h = gradient_orientation_hist(img, bins=8)
        assert h.sum() == pytest.approx(1.0, abs=1e-4)


# ─── texture_match_pair ───────────────────────────────────────────────────────

class TestTextureMatchPair:
    def test_returns_result(self):
        r = texture_match_pair(_noisy(), _noisy(seed=99))
        assert isinstance(r, TextureMatchResult)

    def test_score_in_range(self):
        r = texture_match_pair(_noisy(), _noisy(seed=99))
        assert 0.0 <= r.score <= 1.0

    def test_method_name(self):
        r = texture_match_pair(_noisy(), _noisy())
        assert r.method == "texture"

    def test_side_params_stored(self):
        r = texture_match_pair(_noisy(), _noisy(), side1=0, side2=2)
        assert r.params.get("side1") == 0
        assert r.params.get("side2") == 2

    def test_border_frac_stored(self):
        r = texture_match_pair(_noisy(), _noisy(), border_frac=0.15)
        assert r.params.get("border_frac") == pytest.approx(0.15)

    def test_weights_stored(self):
        r = texture_match_pair(_noisy(), _noisy(),
                                w_lbp=0.5, w_gabor=0.3, w_gradient=0.2)
        assert r.params.get("w_lbp") == pytest.approx(0.5)
        assert r.params.get("w_gabor") == pytest.approx(0.3)
        assert r.params.get("w_gradient") == pytest.approx(0.2)

    def test_identical_image_high_score(self):
        img = _noisy()
        r = texture_match_pair(img, img, side1=1, side2=1)
        assert r.score > 0.5

    def test_gray_input(self):
        r = texture_match_pair(_gray(), _gray())
        assert isinstance(r, TextureMatchResult)

    def test_bgr_input(self):
        r = texture_match_pair(_bgr(), _bgr())
        assert isinstance(r, TextureMatchResult)
        assert 0.0 <= r.score <= 1.0

    def test_sub_scores_in_range(self):
        r = texture_match_pair(_noisy(), _noisy(seed=2))
        assert 0.0 <= r.lbp_score <= 1.0
        assert 0.0 <= r.gabor_score <= 1.0
        assert 0.0 <= r.gradient_score <= 1.0

    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides(self, side):
        r = texture_match_pair(_noisy(), _noisy(seed=7), side1=side, side2=side)
        assert isinstance(r, TextureMatchResult)
        assert 0.0 <= r.score <= 1.0


# ─── texture_compatibility_matrix ────────────────────────────────────────────

class TestTextureCompatibilityMatrix:
    def test_empty_returns_0x0(self):
        m = texture_compatibility_matrix([])
        assert m.shape == (0, 0)

    def test_single_returns_1x1(self):
        m = texture_compatibility_matrix([_noisy()])
        assert m.shape == (1, 1)
        assert m[0, 0] == pytest.approx(1.0)

    def test_nxn_shape(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        m = texture_compatibility_matrix(imgs)
        assert m.shape == (3, 3)

    def test_dtype_float32(self):
        m = texture_compatibility_matrix([_noisy(), _noisy(seed=1)])
        assert m.dtype == np.float32

    def test_diagonal_is_one(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        m = texture_compatibility_matrix(imgs)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-4)

    def test_symmetric(self):
        imgs = [_noisy(seed=i) for i in range(3)]
        m = texture_compatibility_matrix(imgs)
        np.testing.assert_allclose(m, m.T, atol=1e-6)

    def test_values_in_range(self):
        imgs = [_noisy(seed=i) for i in range(4)]
        m = texture_compatibility_matrix(imgs)
        assert np.all(m >= 0.0)
        assert np.all(m <= 1.0)

    def test_two_images(self):
        m = texture_compatibility_matrix([_noisy(), _noisy(seed=99)])
        assert m.shape == (2, 2)
        assert m[0, 0] == pytest.approx(1.0)
        assert m[1, 1] == pytest.approx(1.0)
        assert m[0, 1] == pytest.approx(m[1, 0], abs=1e-6)
