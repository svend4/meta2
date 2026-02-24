"""Extra tests for puzzle_reconstruction/matching/color_match.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.color_match import (
    ColorMatchResult,
    compute_color_histogram,
    histogram_distance,
    compute_color_moments,
    moments_distance,
    edge_color_profile,
    color_match_pair,
    color_compatibility_matrix,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def _red():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 2] = 200
    return img


def _blue():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    return img


# ─── ColorMatchResult (extra) ────────────────────────────────────────────────

class TestColorMatchResultExtra:
    def test_params_default_empty(self):
        res = ColorMatchResult(score=0.5, hist_score=0.5, moment_score=0.5,
                               profile_score=0.5, method="test")
        assert res.params == {}

    def test_custom_params_stored(self):
        res = ColorMatchResult(score=0.5, hist_score=0.5, moment_score=0.5,
                               profile_score=0.5, method="test", params={"bins": 32})
        assert res.params["bins"] == 32

    def test_method_stored(self):
        res = ColorMatchResult(score=0.5, hist_score=0.5, moment_score=0.5,
                               profile_score=0.5, method="hsv_corr")
        assert res.method == "hsv_corr"

    def test_score_boundary_zero(self):
        res = ColorMatchResult(score=0.0, hist_score=0.0, moment_score=0.0,
                               profile_score=0.0, method="t")
        assert res.score == pytest.approx(0.0)

    def test_score_boundary_one(self):
        res = ColorMatchResult(score=1.0, hist_score=1.0, moment_score=1.0,
                               profile_score=1.0, method="t")
        assert res.score == pytest.approx(1.0)


# ─── compute_color_histogram (extra) ─────────────────────────────────────────

class TestComputeColorHistogramExtra:
    def test_gray_length_equals_bins(self):
        hist = compute_color_histogram(_gray(), bins=32)
        assert len(hist) == 32

    def test_bgr_length_is_3x_bins(self):
        hist = compute_color_histogram(_bgr(), bins=16)
        assert len(hist) == 48

    def test_dtype_float32(self):
        hist = compute_color_histogram(_bgr(), bins=8)
        assert hist.dtype == np.float32

    def test_nonnegative(self):
        hist = compute_color_histogram(_bgr(), bins=16)
        assert (hist >= 0).all()

    def test_hsv_colorspace(self):
        hist = compute_color_histogram(_bgr(), colorspace="hsv", bins=8)
        assert hist.dtype == np.float32

    def test_lab_colorspace(self):
        hist = compute_color_histogram(_bgr(), colorspace="lab", bins=8)
        assert len(hist) > 0

    def test_bgr_colorspace(self):
        hist = compute_color_histogram(_bgr(), colorspace="bgr", bins=8)
        assert len(hist) > 0

    def test_mask_accepted(self):
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        hist = compute_color_histogram(_bgr(), mask=mask, bins=8)
        assert hist.dtype == np.float32

    def test_uniform_gray_single_peak(self):
        img = _gray(val=100)
        hist = compute_color_histogram(img, bins=256)
        assert hist[100] > 0
        assert hist[:100].sum() == pytest.approx(0.0, abs=1e-6)


# ─── histogram_distance (extra) ──────────────────────────────────────────────

class TestHistogramDistanceExtra:
    def test_same_hist_returns_one_bhatt(self):
        hist = compute_color_histogram(_bgr(), bins=16)
        score = histogram_distance(hist, hist, metric="bhatt")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_same_hist_returns_one_corr(self):
        hist = compute_color_histogram(_bgr(), bins=16)
        score = histogram_distance(hist, hist, metric="corr")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_result_in_0_1(self):
        h1 = compute_color_histogram(_red(), colorspace="bgr", bins=16)
        h2 = compute_color_histogram(_blue(), colorspace="bgr", bins=16)
        for metric in ("chi2", "bhatt", "corr", "inter"):
            s = histogram_distance(h1, h2, metric=metric)
            assert 0.0 <= s <= 1.0, f"{metric}: {s}"

    def test_returns_float(self):
        hist = compute_color_histogram(_bgr(), bins=8)
        assert isinstance(histogram_distance(hist, hist), float)

    def test_red_vs_blue_low_score(self):
        h1 = compute_color_histogram(_red(), colorspace="bgr", bins=16)
        h2 = compute_color_histogram(_blue(), colorspace="bgr", bins=16)
        score = histogram_distance(h1, h2, metric="bhatt")
        assert score < 0.9


# ─── compute_color_moments (extra) ───────────────────────────────────────────

class TestComputeColorMomentsExtra:
    def test_gray_returns_3(self):
        m = compute_color_moments(_gray())
        assert len(m) == 3

    def test_bgr_returns_9(self):
        m = compute_color_moments(_bgr())
        assert len(m) == 9

    def test_dtype_float32(self):
        m = compute_color_moments(_bgr())
        assert m.dtype == np.float32

    def test_std_nonneg(self):
        m = compute_color_moments(_bgr())
        for i in range(1, 9, 3):
            assert m[i] >= 0.0

    def test_constant_image_zero_std(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        m = compute_color_moments(img, colorspace="bgr")
        for i in range(1, 9, 3):
            assert m[i] == pytest.approx(0.0, abs=0.01)

    def test_hsv_colorspace(self):
        m = compute_color_moments(_bgr(), colorspace="hsv")
        assert len(m) == 9

    def test_lab_colorspace(self):
        m = compute_color_moments(_bgr(), colorspace="lab")
        assert len(m) == 9


# ─── moments_distance (extra) ────────────────────────────────────────────────

class TestMomentsDistanceExtra:
    def test_same_returns_one(self):
        m = compute_color_moments(_bgr())
        score = moments_distance(m, m)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_result_in_0_1(self):
        m1 = compute_color_moments(_bgr(seed=0))
        m2 = compute_color_moments(_bgr(seed=1))
        score = moments_distance(m1, m2)
        assert 0.0 <= score <= 1.0

    def test_symmetric(self):
        m1 = compute_color_moments(_bgr(seed=0))
        m2 = compute_color_moments(_red())
        assert moments_distance(m1, m2) == pytest.approx(
            moments_distance(m2, m1), abs=1e-6
        )

    def test_returns_float(self):
        m = compute_color_moments(_bgr())
        assert isinstance(moments_distance(m, m), float)

    def test_empty_returns_zero(self):
        score = moments_distance(np.array([]), np.array([]))
        assert score == pytest.approx(0.0)


# ─── edge_color_profile (extra) ──────────────────────────────────────────────

class TestEdgeColorProfileExtra:
    def test_returns_ndarray(self):
        profile = edge_color_profile(_bgr(), side=0, bins=8)
        assert isinstance(profile, np.ndarray)

    def test_dtype_float32(self):
        profile = edge_color_profile(_bgr(), side=0, bins=8)
        assert profile.dtype == np.float32

    def test_bgr_length_is_3x_bins(self):
        profile = edge_color_profile(_bgr(), side=0, bins=16)
        assert len(profile) == 3 * 16

    def test_gray_length_is_bins(self):
        profile = edge_color_profile(_gray(), side=0, bins=16)
        assert len(profile) == 16

    def test_nonneg(self):
        profile = edge_color_profile(_bgr(), side=1, bins=8)
        assert (profile >= 0).all()

    def test_all_sides_accepted(self):
        img = _bgr()
        for side in range(4):
            profile = edge_color_profile(img, side=side, bins=8)
            assert profile.dtype == np.float32


# ─── color_match_pair (extra) ────────────────────────────────────────────────

class TestColorMatchPairExtra:
    def test_returns_color_match_result(self):
        res = color_match_pair(_bgr(), _bgr())
        assert isinstance(res, ColorMatchResult)

    def test_identical_high_score(self):
        img = _bgr(seed=0)
        res = color_match_pair(img, img)
        assert res.score > 0.7

    def test_score_in_range(self):
        res = color_match_pair(_red(), _blue())
        assert 0.0 <= res.score <= 1.0

    def test_method_contains_colorspace(self):
        res = color_match_pair(_bgr(), _bgr(), colorspace="lab")
        assert "lab" in res.method

    def test_sub_scores_in_range(self):
        res = color_match_pair(_bgr(seed=0), _bgr(seed=1))
        for attr in ("hist_score", "moment_score", "profile_score"):
            val = getattr(res, attr)
            assert 0.0 <= val <= 1.0

    def test_side_params_stored(self):
        res = color_match_pair(_bgr(), _bgr(), side1=1, side2=3)
        assert res.params["side1"] == 1
        assert res.params["side2"] == 3

    def test_gray_input_accepted(self):
        res = color_match_pair(_gray(), _gray())
        assert isinstance(res, ColorMatchResult)


# ─── color_compatibility_matrix (extra) ──────────────────────────────────────

class TestColorCompatibilityMatrixExtra:
    def test_shape_nxn(self):
        imgs = [_bgr(seed=i) for i in range(3)]
        mat = color_compatibility_matrix(imgs)
        assert mat.shape == (3, 3)

    def test_dtype_float32(self):
        imgs = [_bgr(seed=i) for i in range(2)]
        mat = color_compatibility_matrix(imgs)
        assert mat.dtype == np.float32

    def test_diagonal_one(self):
        imgs = [_bgr(seed=i) for i in range(3)]
        mat = color_compatibility_matrix(imgs)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=0.01)

    def test_symmetric(self):
        imgs = [_bgr(seed=i) for i in range(3)]
        mat = color_compatibility_matrix(imgs)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_values_in_range(self):
        imgs = [_bgr(seed=i) for i in range(4)]
        mat = color_compatibility_matrix(imgs)
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0 + 1e-5

    def test_empty_gives_0x0(self):
        mat = color_compatibility_matrix([])
        assert mat.shape == (0, 0)

    def test_single_gives_1x1(self):
        mat = color_compatibility_matrix([_bgr()])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(1.0)
