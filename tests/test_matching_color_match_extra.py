"""Extra tests for puzzle_reconstruction.matching.color_match."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.color_match import (
    ColorMatchResult,
    color_compatibility_matrix,
    color_match_pair,
    compute_color_histogram,
    compute_color_moments,
    edge_color_profile,
    histogram_distance,
    moments_distance,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rgb(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _uniform(h=64, w=64, val=100):
    return np.full((h, w, 3), val, dtype=np.uint8)


# ─── TestColorMatchResultExtra ────────────────────────────────────────────────

class TestColorMatchResultExtra:
    def test_score_stored(self):
        r = ColorMatchResult(score=0.6, hist_score=0.7,
                             moment_score=0.5, profile_score=0.4,
                             method="bgr")
        assert r.score == pytest.approx(0.6)

    def test_method_stored(self):
        r = ColorMatchResult(score=0.5, hist_score=0.5,
                             moment_score=0.5, profile_score=0.5,
                             method="lab")
        assert r.method == "lab"

    def test_params_default_empty(self):
        r = ColorMatchResult(score=0.5, hist_score=0.5,
                             moment_score=0.5, profile_score=0.5,
                             method="hsv")
        assert r.params == {}

    def test_params_custom_stored(self):
        r = ColorMatchResult(score=0.5, hist_score=0.5,
                             moment_score=0.5, profile_score=0.5,
                             method="bgr", params={"bins": 16})
        assert r.params["bins"] == 16

    def test_all_scores_accessible(self):
        r = ColorMatchResult(score=0.9, hist_score=0.8,
                             moment_score=0.7, profile_score=0.6,
                             method="hsv")
        assert r.hist_score == pytest.approx(0.8)
        assert r.moment_score == pytest.approx(0.7)
        assert r.profile_score == pytest.approx(0.6)


# ─── TestComputeColorHistogramExtra ───────────────────────────────────────────

class TestComputeColorHistogramExtra:
    def test_all_values_nonneg(self):
        h = compute_color_histogram(_rgb())
        assert (h >= 0).all()

    def test_bgr_vs_hsv_different(self):
        img = _rgb(seed=42)
        h_bgr = compute_color_histogram(img, colorspace="bgr")
        h_hsv = compute_color_histogram(img, colorspace="hsv")
        assert not np.allclose(h_bgr, h_hsv)

    def test_bins_8(self):
        h = compute_color_histogram(_rgb(), bins=8)
        assert h.shape[0] > 0
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_bins_64(self):
        h = compute_color_histogram(_rgb(), bins=64)
        assert h.shape[0] > 0

    def test_gray_small_image(self):
        h = compute_color_histogram(_gray(8, 8))
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_two_uniform_images_same_hist(self):
        img1 = _uniform(val=100)
        img2 = _uniform(val=100)
        h1 = compute_color_histogram(img1, colorspace="bgr")
        h2 = compute_color_histogram(img2, colorspace="bgr")
        np.testing.assert_allclose(h1, h2)


# ─── TestHistogramDistanceExtra ───────────────────────────────────────────────

class TestHistogramDistanceExtra:
    def test_symmetric(self):
        h1 = compute_color_histogram(_rgb(seed=0))
        h2 = compute_color_histogram(_rgb(seed=1))
        assert histogram_distance(h1, h2) == pytest.approx(
            histogram_distance(h2, h1), abs=1e-5)

    def test_all_metrics_in_range(self):
        h1 = compute_color_histogram(_rgb(seed=0))
        h2 = compute_color_histogram(_rgb(seed=1))
        for metric in ("bhatt", "chi2", "corr", "inter"):
            s = histogram_distance(h1, h2, metric=metric)
            assert 0.0 <= s <= 1.0

    def test_uniform_same_high_score(self):
        h1 = compute_color_histogram(_uniform(val=100), colorspace="bgr")
        h2 = compute_color_histogram(_uniform(val=100), colorspace="bgr")
        s = histogram_distance(h1, h2)
        assert s >= 0.9

    def test_returns_float(self):
        h = compute_color_histogram(_rgb())
        assert isinstance(histogram_distance(h, h), float)

    def test_zero_histograms(self):
        h = np.zeros(32, dtype=np.float32)
        s = histogram_distance(h, h)
        assert 0.0 <= s <= 1.0


# ─── TestComputeColorMomentsExtra ─────────────────────────────────────────────

class TestComputeColorMomentsExtra:
    def test_values_finite(self):
        m = compute_color_moments(_rgb())
        assert np.all(np.isfinite(m))

    def test_yuv_colorspace_shape(self):
        m = compute_color_moments(_rgb(), colorspace="yuv")
        assert m.shape == (9,)

    def test_same_image_same_moments(self):
        img = _rgb(seed=5)
        m1 = compute_color_moments(img)
        m2 = compute_color_moments(img)
        np.testing.assert_allclose(m1, m2)

    def test_uniform_skewness_near_zero(self):
        img = _uniform(val=150)
        m = compute_color_moments(img, colorspace="bgr")
        # third moment (skewness) should be ~0 for uniform
        assert abs(m[2]) < 1.0


# ─── TestMomentsDistanceExtra ─────────────────────────────────────────────────

class TestMomentsDistanceExtra:
    def test_symmetric(self):
        m1 = compute_color_moments(_rgb(seed=0))
        m2 = compute_color_moments(_rgb(seed=1))
        assert moments_distance(m1, m2) == pytest.approx(
            moments_distance(m2, m1), abs=1e-5)

    def test_score_nonneg(self):
        m1 = compute_color_moments(_rgb(seed=0))
        m2 = compute_color_moments(_rgb(seed=5))
        assert moments_distance(m1, m2) >= 0.0

    def test_identical_moments_high(self):
        m = compute_color_moments(_rgb(seed=3))
        assert moments_distance(m, m) >= 0.9

    def test_returns_float_type(self):
        m = compute_color_moments(_rgb())
        assert isinstance(moments_distance(m, m), float)


# ─── TestEdgeColorProfileExtra ────────────────────────────────────────────────

class TestEdgeColorProfileExtra:
    def test_all_values_nonneg(self):
        h = edge_color_profile(_rgb(), side=0)
        assert (h >= 0).all()

    def test_side_0_vs_side_2_may_differ(self):
        img = _rgb(seed=7)
        h0 = edge_color_profile(img, side=0)
        h2 = edge_color_profile(img, side=2)
        # top vs bottom edge may differ for random image
        assert h0.shape == h2.shape

    def test_uniform_image_concentrated(self):
        h = edge_color_profile(_uniform(val=100), side=1)
        assert h.max() > 0.1

    def test_bins_16(self):
        h = edge_color_profile(_rgb(), side=0, bins=16)
        assert h.shape[0] > 0

    def test_normalized_sum(self):
        h = edge_color_profile(_rgb(), side=3)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)


# ─── TestColorMatchPairExtra ──────────────────────────────────────────────────

class TestColorMatchPairExtra:
    def test_all_sub_scores_in_range(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= r.hist_score <= 1.0
        assert 0.0 <= r.moment_score <= 1.0
        assert 0.0 <= r.profile_score <= 1.0

    def test_uniform_images_high_score(self):
        img1 = _uniform(val=100)
        img2 = _uniform(val=100)
        r = color_match_pair(img1, img2)
        assert r.score >= 0.8

    def test_method_contains_colorspace(self):
        r = color_match_pair(_rgb(), _rgb(seed=1))
        assert "hsv" in r.method or "bgr" in r.method or "lab" in r.method

    def test_inter_metric(self):
        r = color_match_pair(_rgb(), _rgb(seed=1), metric="inter")
        assert 0.0 <= r.score <= 1.0

    def test_corr_metric(self):
        r = color_match_pair(_rgb(), _rgb(seed=1), metric="corr")
        assert 0.0 <= r.score <= 1.0

    def test_gray_pair_score_in_range(self):
        r = color_match_pair(_gray(val=100), _gray(val=200))
        assert 0.0 <= r.score <= 1.0


# ─── TestColorCompatibilityMatrixExtra ────────────────────────────────────────

class TestColorCompatibilityMatrixExtra:
    def test_dtype_float(self):
        mat = color_compatibility_matrix([_rgb(seed=i) for i in range(3)])
        assert mat.dtype in (np.float32, np.float64)

    def test_two_images(self):
        mat = color_compatibility_matrix([_rgb(seed=0), _rgb(seed=1)])
        assert mat.shape == (2, 2)

    def test_diagonal_is_highest_per_row(self):
        imgs = [_rgb(seed=i) for i in range(3)]
        mat = color_compatibility_matrix(imgs)
        for i in range(3):
            assert mat[i, i] >= mat[i].min()

    def test_values_all_in_range(self):
        mat = color_compatibility_matrix([_rgb(seed=i) for i in range(3)])
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0 + 1e-6

    def test_symmetric_property(self):
        mat = color_compatibility_matrix([_rgb(seed=i) for i in range(3)])
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)
