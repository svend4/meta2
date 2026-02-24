"""Extra tests for puzzle_reconstruction/matching/texture_match.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=32) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _hist(bins=64) -> np.ndarray:
    h = np.ones(bins, dtype=np.float32)
    return h / h.sum()


# ─── TextureMatchResult ───────────────────────────────────────────────────────

class TestTextureMatchResultExtra:
    def test_stores_score(self):
        r = TextureMatchResult(score=0.7, lbp_score=0.6,
                               gabor_score=0.8, gradient_score=0.7)
        assert r.score == pytest.approx(0.7)

    def test_stores_lbp_score(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.4,
                               gabor_score=0.6, gradient_score=0.5)
        assert r.lbp_score == pytest.approx(0.4)

    def test_stores_gabor_score(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.9, gradient_score=0.5)
        assert r.gabor_score == pytest.approx(0.9)

    def test_stores_gradient_score(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.3)
        assert r.gradient_score == pytest.approx(0.3)

    def test_default_method(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5)
        assert r.method == "texture"

    def test_repr_contains_score(self):
        r = TextureMatchResult(score=0.75, lbp_score=0.6,
                               gabor_score=0.8, gradient_score=0.85)
        assert "0.750" in repr(r)

    def test_default_params_empty(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5)
        assert isinstance(r.params, dict)

    def test_custom_params_stored(self):
        r = TextureMatchResult(score=0.5, lbp_score=0.5,
                               gabor_score=0.5, gradient_score=0.5,
                               params={"side1": 0})
        assert r.params["side1"] == 0


# ─── compute_lbp_histogram ────────────────────────────────────────────────────

class TestComputeLbpHistogramExtra:
    def test_returns_array(self):
        h = compute_lbp_histogram(_gray())
        assert isinstance(h, np.ndarray)

    def test_length_equals_bins(self):
        h = compute_lbp_histogram(_gray(), bins=32)
        assert len(h) == 32

    def test_sums_to_one_or_zero(self):
        h = compute_lbp_histogram(_ramp())
        assert float(h.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_dtype_float32(self):
        h = compute_lbp_histogram(_gray())
        assert h.dtype == np.float32

    def test_uniform_image_nonzero(self):
        h = compute_lbp_histogram(_gray(val=100))
        assert h.sum() > 0.0

    def test_custom_radius(self):
        h = compute_lbp_histogram(_gray(h=20, w=20), radius=2)
        assert len(h) == 64


# ─── lbp_distance ─────────────────────────────────────────────────────────────

class TestLbpDistanceExtra:
    def test_identical_histograms_chi2(self):
        h = _hist()
        result = lbp_distance(h, h, metric="chi2")
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_identical_histograms_bhatt(self):
        h = _hist()
        result = lbp_distance(h, h, metric="bhatt")
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_identical_histograms_corr(self):
        h = _hist()
        result = lbp_distance(h, h, metric="corr")
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_result_in_range_chi2(self):
        h1 = _hist()
        h2 = compute_lbp_histogram(_ramp())
        result = lbp_distance(h1, h2, metric="chi2")
        assert 0.0 <= result <= 1.0

    def test_result_in_range_bhatt(self):
        h1 = _hist()
        h2 = compute_lbp_histogram(_ramp())
        result = lbp_distance(h1, h2, metric="bhatt")
        assert 0.0 <= result <= 1.0

    def test_result_in_range_corr(self):
        h1 = _hist()
        h2 = compute_lbp_histogram(_ramp())
        result = lbp_distance(h1, h2, metric="corr")
        assert 0.0 <= result <= 1.0

    def test_unknown_metric_raises(self):
        h = _hist()
        with pytest.raises(ValueError):
            lbp_distance(h, h, metric="unknown")


# ─── compute_gabor_features ───────────────────────────────────────────────────

class TestComputeGaborFeaturesExtra:
    def test_returns_array(self):
        f = compute_gabor_features(_gray())
        assert isinstance(f, np.ndarray)

    def test_dtype_float32(self):
        f = compute_gabor_features(_gray())
        assert f.dtype == np.float32

    def test_length_is_2_x_freqs_x_orientations(self):
        f = compute_gabor_features(_gray(), frequencies=(0.1, 0.3), n_orientations=4)
        assert len(f) == 2 * 2 * 4

    def test_single_freq(self):
        f = compute_gabor_features(_gray(), frequencies=(0.2,), n_orientations=2)
        assert len(f) == 2 * 1 * 2

    def test_bgr_input_ok(self):
        gray = _ramp()
        f = compute_gabor_features(gray)
        assert isinstance(f, np.ndarray)


# ─── gabor_distance ───────────────────────────────────────────────────────────

class TestGaborDistanceExtra:
    def test_identical_features_near_one(self):
        f = compute_gabor_features(_gray())
        result = gabor_distance(f, f)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_result_in_range(self):
        f1 = compute_gabor_features(_gray(val=50))
        f2 = compute_gabor_features(_ramp())
        result = gabor_distance(f1, f2)
        assert 0.0 <= result <= 1.0

    def test_empty_arrays_returns_zero(self):
        assert gabor_distance(np.array([]), np.array([])) == pytest.approx(0.0)

    def test_returns_float(self):
        f = compute_gabor_features(_gray())
        assert isinstance(gabor_distance(f, f), float)


# ─── gradient_orientation_hist ────────────────────────────────────────────────

class TestGradientOrientationHistExtra:
    def test_returns_array(self):
        h = gradient_orientation_hist(_gray())
        assert isinstance(h, np.ndarray)

    def test_length_equals_bins(self):
        h = gradient_orientation_hist(_gray(), bins=8)
        assert len(h) == 8

    def test_dtype_float32(self):
        h = gradient_orientation_hist(_gray())
        assert h.dtype == np.float32

    def test_sums_to_one_or_zero(self):
        h = gradient_orientation_hist(_ramp())
        s = float(h.sum())
        assert s == pytest.approx(1.0, abs=1e-5) or s == pytest.approx(0.0)

    def test_ramp_nonzero(self):
        h = gradient_orientation_hist(_ramp())
        assert h.sum() > 0.0

    def test_custom_bins(self):
        h = gradient_orientation_hist(_ramp(), bins=16)
        assert len(h) == 16


# ─── texture_match_pair ───────────────────────────────────────────────────────

class TestTextureMatchPairExtra:
    def test_returns_texture_match_result(self):
        r = texture_match_pair(_gray(), _gray())
        assert isinstance(r, TextureMatchResult)

    def test_score_in_range(self):
        r = texture_match_pair(_gray(), _ramp())
        assert 0.0 <= r.score <= 1.0

    def test_identical_images_high_score(self):
        img = _gray(val=128)
        r = texture_match_pair(img, img.copy(), side1=1, side2=3)
        assert r.score > 0.0

    def test_method_is_texture(self):
        r = texture_match_pair(_gray(), _gray())
        assert r.method == "texture"

    def test_params_stored_side1(self):
        r = texture_match_pair(_gray(), _gray(), side1=2, side2=0)
        assert r.params["side1"] == 2
        assert r.params["side2"] == 0

    def test_all_sides(self):
        for s in range(4):
            r = texture_match_pair(_gray(32, 32), _gray(32, 32), side1=s, side2=(s + 2) % 4)
            assert 0.0 <= r.score <= 1.0

    def test_bgr_images(self):
        r = texture_match_pair(_bgr(), _bgr())
        assert 0.0 <= r.score <= 1.0

    def test_custom_weights_sum_to_one(self):
        r = texture_match_pair(_gray(), _gray(), w_lbp=0.5, w_gabor=0.3, w_gradient=0.2)
        assert isinstance(r, TextureMatchResult)

    def test_lbp_score_in_range(self):
        r = texture_match_pair(_gray(), _ramp())
        assert 0.0 <= r.lbp_score <= 1.0

    def test_gabor_score_in_range(self):
        r = texture_match_pair(_gray(), _ramp())
        assert 0.0 <= r.gabor_score <= 1.0

    def test_gradient_score_in_range(self):
        r = texture_match_pair(_gray(), _ramp())
        assert 0.0 <= r.gradient_score <= 1.0


# ─── texture_compatibility_matrix ────────────────────────────────────────────

class TestTextureCompatibilityMatrixExtra:
    def test_returns_ndarray(self):
        imgs = [_gray(), _gray()]
        mat = texture_compatibility_matrix(imgs)
        assert isinstance(mat, np.ndarray)

    def test_shape_n_by_n(self):
        imgs = [_gray(), _ramp(), _gray(val=50)]
        mat = texture_compatibility_matrix(imgs)
        assert mat.shape == (3, 3)

    def test_diagonal_one(self):
        imgs = [_gray(), _ramp()]
        mat = texture_compatibility_matrix(imgs)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)

    def test_values_in_range(self):
        imgs = [_gray(), _ramp()]
        mat = texture_compatibility_matrix(imgs)
        assert np.all((mat >= 0.0) & (mat <= 1.0))

    def test_symmetric(self):
        imgs = [_gray(), _ramp()]
        mat = texture_compatibility_matrix(imgs)
        assert mat[0, 1] == pytest.approx(mat[1, 0])

    def test_single_image(self):
        mat = texture_compatibility_matrix([_gray()])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_dtype_float32(self):
        imgs = [_gray(), _ramp()]
        mat = texture_compatibility_matrix(imgs)
        assert mat.dtype == np.float32
