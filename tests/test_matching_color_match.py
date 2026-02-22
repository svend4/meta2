"""Тесты для puzzle_reconstruction.matching.color_match."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rgb(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _uniform(h=64, w=64, c=3, val=100) -> np.ndarray:
    return np.full((h, w, c), val, dtype=np.uint8)


# ─── TestColorMatchResult ─────────────────────────────────────────────────────

class TestColorMatchResult:
    def _make(self, score=0.8) -> ColorMatchResult:
        return ColorMatchResult(score=score, hist_score=0.9,
                                moment_score=0.8, profile_score=0.7,
                                method="hsv")

    def test_basic_fields(self):
        r = self._make()
        assert r.score == pytest.approx(0.8)
        assert r.method == "hsv"

    def test_hist_score_stored(self):
        r = self._make()
        assert r.hist_score == pytest.approx(0.9)

    def test_moment_score_stored(self):
        r = self._make()
        assert r.moment_score == pytest.approx(0.8)

    def test_profile_score_stored(self):
        r = self._make()
        assert r.profile_score == pytest.approx(0.7)

    def test_params_default_empty(self):
        r = self._make()
        assert r.params == {}

    def test_params_custom(self):
        r = ColorMatchResult(score=0.5, hist_score=0.5,
                             moment_score=0.5, profile_score=0.5,
                             method="bgr", params={"bins": 32})
        assert r.params["bins"] == 32


# ─── TestComputeColorHistogram ────────────────────────────────────────────────

class TestComputeColorHistogram:
    def test_returns_ndarray(self):
        h = compute_color_histogram(_rgb())
        assert isinstance(h, np.ndarray)

    def test_float32_dtype(self):
        h = compute_color_histogram(_rgb())
        assert h.dtype == np.float32

    def test_normalized_sum_one(self):
        h = compute_color_histogram(_rgb())
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_gray_1d_histogram(self):
        h = compute_color_histogram(_gray())
        # Grayscale: single channel histogram
        assert h.ndim == 1

    def test_bgr_3channel_histogram(self):
        h = compute_color_histogram(_rgb(), colorspace="bgr")
        assert h.ndim == 1

    def test_custom_bins(self):
        h = compute_color_histogram(_rgb(), bins=16)
        # 3 channels * 16 bins or 16 bins for gray
        assert h.shape[0] > 0

    def test_hsv_colorspace(self):
        h = compute_color_histogram(_rgb(), colorspace="hsv")
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_lab_colorspace(self):
        h = compute_color_histogram(_rgb(), colorspace="lab")
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_yuv_colorspace(self):
        h = compute_color_histogram(_rgb(), colorspace="yuv")
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_uniform_image_concentrated(self):
        # Uniform color → histogram concentrated in one bin
        img = _uniform()
        h = compute_color_histogram(img)
        # At least one bin should dominate
        assert h.max() > 0.1


# ─── TestHistogramDistance ────────────────────────────────────────────────────

class TestHistogramDistance:
    def _hist(self, n=32) -> np.ndarray:
        h = np.zeros(n, dtype=np.float32)
        h[n // 2] = 1.0
        return h

    def test_returns_float(self):
        h = self._hist()
        assert isinstance(histogram_distance(h, h), float)

    def test_identical_histograms_high_score(self):
        h = self._hist()
        s = histogram_distance(h, h)
        assert s >= 0.9

    def test_score_in_range(self):
        h1 = self._hist()
        h2 = np.zeros(32, dtype=np.float32)
        h2[0] = 1.0
        s = histogram_distance(h1, h2)
        assert 0.0 <= s <= 1.0

    def test_bhatt_metric(self):
        h = self._hist()
        s = histogram_distance(h, h, metric="bhatt")
        assert s >= 0.9

    def test_chi2_metric(self):
        h = self._hist()
        s = histogram_distance(h, h, metric="chi2")
        assert s >= 0.9

    def test_corr_metric(self):
        h = self._hist()
        s = histogram_distance(h, h, metric="corr")
        assert s >= 0.9

    def test_inter_metric(self):
        h = self._hist()
        s = histogram_distance(h, h, metric="inter")
        assert s >= 0.9

    def test_invalid_metric_raises(self):
        h = self._hist()
        with pytest.raises(ValueError):
            histogram_distance(h, h, metric="l2")

    def test_different_histograms_lower_score(self):
        h1 = self._hist()
        h2 = np.zeros(32, dtype=np.float32)
        h2[0] = 1.0
        s_same = histogram_distance(h1, h1)
        s_diff = histogram_distance(h1, h2)
        assert s_same >= s_diff


# ─── TestComputeColorMoments ──────────────────────────────────────────────────

class TestComputeColorMoments:
    def test_returns_ndarray(self):
        m = compute_color_moments(_rgb())
        assert isinstance(m, np.ndarray)

    def test_float32_dtype(self):
        m = compute_color_moments(_rgb())
        assert m.dtype == np.float32

    def test_3channel_length_9(self):
        m = compute_color_moments(_rgb())
        assert m.shape == (9,)

    def test_gray_length_3(self):
        m = compute_color_moments(_gray())
        assert m.shape == (3,)

    def test_uniform_std_zero(self):
        img = _uniform(val=100)
        m = compute_color_moments(img, colorspace="bgr")
        # Std should be ~0 for uniform image
        assert abs(m[1]) < 1.0

    def test_hsv_colorspace(self):
        m = compute_color_moments(_rgb(), colorspace="hsv")
        assert m.shape == (9,)

    def test_lab_colorspace(self):
        m = compute_color_moments(_rgb(), colorspace="lab")
        assert m.shape == (9,)

    def test_bgr_colorspace(self):
        m = compute_color_moments(_rgb(), colorspace="bgr")
        assert m.shape == (9,)


# ─── TestMomentsDistance ──────────────────────────────────────────────────────

class TestMomentsDistance:
    def test_returns_float(self):
        m = compute_color_moments(_rgb())
        assert isinstance(moments_distance(m, m), float)

    def test_identical_moments_high_score(self):
        m = compute_color_moments(_rgb())
        s = moments_distance(m, m)
        assert s >= 0.9

    def test_score_in_range(self):
        m1 = compute_color_moments(_rgb(seed=0))
        m2 = compute_color_moments(_rgb(seed=1))
        s = moments_distance(m1, m2)
        assert 0.0 <= s <= 1.0

    def test_empty_array_returns_zero(self):
        s = moments_distance(np.array([]), np.array([]))
        assert s == pytest.approx(0.0)

    def test_similar_images_higher_score(self):
        img1 = _uniform(val=100)
        img2 = _uniform(val=102)  # very similar
        img3 = _uniform(val=200)  # very different
        m1 = compute_color_moments(img1, colorspace="bgr")
        m2 = compute_color_moments(img2, colorspace="bgr")
        m3 = compute_color_moments(img3, colorspace="bgr")
        s_close = moments_distance(m1, m2)
        s_far = moments_distance(m1, m3)
        assert s_close >= s_far


# ─── TestEdgeColorProfile ─────────────────────────────────────────────────────

class TestEdgeColorProfile:
    def test_returns_ndarray(self):
        h = edge_color_profile(_rgb(), side=0)
        assert isinstance(h, np.ndarray)

    def test_float32_dtype(self):
        h = edge_color_profile(_rgb(), side=0)
        assert h.dtype == np.float32

    def test_all_four_sides(self):
        img = _rgb()
        for side in range(4):
            h = edge_color_profile(img, side=side)
            assert h.ndim == 1

    def test_normalized(self):
        h = edge_color_profile(_rgb(), side=1)
        assert h.sum() == pytest.approx(1.0, abs=1e-5)

    def test_custom_bins(self):
        h = edge_color_profile(_rgb(), side=2, bins=8)
        assert h.shape[0] > 0

    def test_gray_image_ok(self):
        h = edge_color_profile(_gray(), side=3)
        assert isinstance(h, np.ndarray)


# ─── TestColorMatchPair ───────────────────────────────────────────────────────

class TestColorMatchPair:
    def test_returns_color_match_result(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1))
        assert isinstance(r, ColorMatchResult)

    def test_score_in_range(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= r.score <= 1.0

    def test_hist_score_in_range(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= r.hist_score <= 1.0

    def test_moment_score_in_range(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= r.moment_score <= 1.0

    def test_profile_score_in_range(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1))
        assert 0.0 <= r.profile_score <= 1.0

    def test_identical_images_high_score(self):
        img = _rgb(seed=0)
        r = color_match_pair(img, img)
        assert r.score >= 0.8

    def test_gray_images_ok(self):
        r = color_match_pair(_gray(), _gray())
        assert isinstance(r, ColorMatchResult)

    def test_bhatt_metric(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1), metric="bhatt")
        assert 0.0 <= r.score <= 1.0

    def test_chi2_metric(self):
        r = color_match_pair(_rgb(seed=0), _rgb(seed=1), metric="chi2")
        assert 0.0 <= r.score <= 1.0

    def test_different_sides(self):
        img1 = _rgb(seed=0)
        img2 = _rgb(seed=1)
        r = color_match_pair(img1, img2, side1=0, side2=2)
        assert isinstance(r, ColorMatchResult)


# ─── TestColorCompatibilityMatrix ────────────────────────────────────────────

class TestColorCompatibilityMatrix:
    def _images(self, n=4) -> list:
        return [_rgb(seed=i) for i in range(n)]

    def test_returns_ndarray(self):
        mat = color_compatibility_matrix(self._images(4))
        assert isinstance(mat, np.ndarray)

    def test_shape_n_by_n(self):
        n = 4
        mat = color_compatibility_matrix(self._images(n))
        assert mat.shape == (n, n)

    def test_diagonal_high(self):
        imgs = self._images(4)
        mat = color_compatibility_matrix(imgs)
        for i in range(4):
            assert mat[i, i] >= 0.8

    def test_values_in_range(self):
        mat = color_compatibility_matrix(self._images(3))
        assert mat.min() >= 0.0
        assert mat.max() <= 1.0

    def test_symmetric(self):
        imgs = self._images(4)
        mat = color_compatibility_matrix(imgs)
        np.testing.assert_allclose(mat, mat.T, atol=1e-5)

    def test_single_image(self):
        mat = color_compatibility_matrix([_rgb()])
        assert mat.shape == (1, 1)
        assert mat[0, 0] >= 0.8
