"""
Тесты для puzzle_reconstruction/matching/color_match.py

Покрывает:
    ColorMatchResult        — поля, repr, score ∈ [0,1]
    compute_color_histogram — 1D gray, 3-канальный BGR; нормировка (сумма=1);
                              colorspaces (hsv/lab/yuv/bgr); mask; ValueError
    histogram_distance      — ∈ [0,1]; одинаковые→1; chi2/bhatt/corr/inter;
                              ValueError на неизвестную метрику
    compute_color_moments   — gray→3 элемента, BGR→9; dtype float32;
                              μ в правильном диапазоне; ValueError на cs
    moments_distance        — ∈ [0,1]; одинаковые→1; пустые→0; симметричность
    edge_color_profile      — нормированный вектор, сумма≈1, 4 стороны
    color_match_pair        — ColorMatchResult, score ∈ [0,1], method,
                              одинаковые изображения → высокий score
    color_compatibility_matrix — NxN float32, симметричная, диагональ=1,
                                 2 изображения, пустой список → 0x0
"""
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


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_img():
    rng = np.random.default_rng(0)
    return (rng.random((60, 60)) * 255).astype(np.uint8)


@pytest.fixture
def bgr_img():
    rng = np.random.default_rng(1)
    return (rng.random((60, 60, 3)) * 255).astype(np.uint8)


@pytest.fixture
def red_bgr():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :, 2] = 220   # R-канал (BGR)
    return img


@pytest.fixture
def blue_bgr():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :, 0] = 220   # B-канал (BGR)
    return img


@pytest.fixture
def sample_result():
    return ColorMatchResult(
        score=0.75,
        hist_score=0.80,
        moment_score=0.70,
        profile_score=0.72,
        method="hsv_bhatt",
    )


# ─── ColorMatchResult ─────────────────────────────────────────────────────────

class TestColorMatchResult:
    def test_fields(self, sample_result):
        assert sample_result.score         == pytest.approx(0.75)
        assert sample_result.hist_score    == pytest.approx(0.80)
        assert sample_result.moment_score  == pytest.approx(0.70)
        assert sample_result.profile_score == pytest.approx(0.72)
        assert sample_result.method        == "hsv_bhatt"

    def test_repr_contains_score(self, sample_result):
        r = repr(sample_result)
        assert "ColorMatchResult" in r
        assert "score=" in r
        assert "hist=" in r
        assert "moment=" in r
        assert "profile=" in r

    def test_default_params(self, sample_result):
        assert sample_result.params == {}

    def test_score_range(self, sample_result):
        assert 0.0 <= sample_result.score <= 1.0

    def test_custom_params(self):
        res = ColorMatchResult(
            score=0.5, hist_score=0.5, moment_score=0.5,
            profile_score=0.5, method="x", params={"bins": 32},
        )
        assert res.params["bins"] == 32


# ─── compute_color_histogram ──────────────────────────────────────────────────

class TestComputeColorHistogram:
    def test_gray_returns_1d(self, gray_img):
        hist = compute_color_histogram(gray_img, bins=16)
        assert hist.ndim == 1
        assert len(hist) == 16

    def test_bgr_returns_3x_bins(self, bgr_img):
        hist = compute_color_histogram(bgr_img, bins=16)
        assert len(hist) == 3 * 16

    def test_gray_sums_to_one(self, gray_img):
        hist = compute_color_histogram(gray_img, bins=32)
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_bgr_sums_to_one(self, bgr_img):
        # Объединённая гистограмма нормируется по суммарному числу пикселей
        hist = compute_color_histogram(bgr_img, bins=32)
        assert hist.sum() == pytest.approx(1.0, abs=1e-4)

    def test_dtype_float32(self, gray_img):
        hist = compute_color_histogram(gray_img, bins=8)
        assert hist.dtype == np.float32

    @pytest.mark.parametrize("cs", ["hsv", "lab", "yuv", "bgr"])
    def test_colorspaces_accepted(self, bgr_img, cs):
        hist = compute_color_histogram(bgr_img, colorspace=cs)
        assert isinstance(hist, np.ndarray)
        assert hist.dtype == np.float32

    def test_unknown_colorspace_raises(self, bgr_img):
        with pytest.raises(ValueError, match="Неизвестное"):
            compute_color_histogram(bgr_img, colorspace="xyz_unknown")

    def test_mask_reduces_pixels(self):
        # Use structured image: inner region all-white, outer all-black
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        img[10:50, 10:50] = 255
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        hist_full = compute_color_histogram(img)
        hist_masked = compute_color_histogram(img, mask=mask)
        # Full image has black and white; masked has only white → different
        assert not np.allclose(hist_full, hist_masked, atol=0.01)

    def test_bins_parameter(self, bgr_img):
        h8  = compute_color_histogram(bgr_img, bins=8)
        h64 = compute_color_histogram(bgr_img, bins=64)
        assert len(h8) == 3 * 8
        assert len(h64) == 3 * 64

    def test_nonnegative_values(self, bgr_img):
        hist = compute_color_histogram(bgr_img)
        assert (hist >= 0).all()


# ─── histogram_distance ───────────────────────────────────────────────────────

class TestHistogramDistance:
    def test_same_hist_returns_one(self, bgr_img):
        hist = compute_color_histogram(bgr_img, bins=16)
        score = histogram_distance(hist, hist, metric="bhatt")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_range_zero_to_one(self, bgr_img, gray_img):
        h1 = compute_color_histogram(bgr_img, bins=16)
        h2 = compute_color_histogram(gray_img, bins=16)
        # Разная длина → используем одинаковые
        h1_g = compute_color_histogram(bgr_img, bins=16, colorspace="bgr")
        h2_g = compute_color_histogram(bgr_img, bins=16, colorspace="bgr")
        for metric in ("chi2", "bhatt", "corr", "inter"):
            score = histogram_distance(h1_g, h2_g, metric=metric)
            assert 0.0 <= score <= 1.0, f"metric={metric}, score={score}"

    @pytest.mark.parametrize("metric", ["chi2", "bhatt", "corr", "inter"])
    def test_all_metrics_accepted(self, bgr_img, metric):
        hist = compute_color_histogram(bgr_img, bins=16)
        score = histogram_distance(hist, hist, metric=metric)
        assert isinstance(score, float)

    def test_identical_red_blue_differ(self, red_bgr, blue_bgr):
        h_red  = compute_color_histogram(red_bgr,  colorspace="bgr", bins=16)
        h_blue = compute_color_histogram(blue_bgr, colorspace="bgr", bins=16)
        score = histogram_distance(h_red, h_blue, metric="bhatt")
        # Красное и синее изображения должны быть несхожи
        assert score < 0.9

    def test_unknown_metric_raises(self, bgr_img):
        hist = compute_color_histogram(bgr_img, bins=8)
        with pytest.raises(ValueError, match="Неизвестная"):
            histogram_distance(hist, hist, metric="unknown_metric")

    def test_returns_float(self, bgr_img):
        hist = compute_color_histogram(bgr_img, bins=8)
        score = histogram_distance(hist, hist)
        assert isinstance(score, float)

    def test_corr_same_hist_equals_one(self, bgr_img):
        hist = compute_color_histogram(bgr_img, bins=16)
        score = histogram_distance(hist, hist, metric="corr")
        assert score == pytest.approx(1.0, abs=0.01)


# ─── compute_color_moments ────────────────────────────────────────────────────

class TestComputeColorMoments:
    def test_gray_returns_3_elements(self, gray_img):
        m = compute_color_moments(gray_img)
        assert len(m) == 3

    def test_bgr_returns_9_elements(self, bgr_img):
        m = compute_color_moments(bgr_img)
        assert len(m) == 9

    def test_dtype_float32(self, bgr_img):
        m = compute_color_moments(bgr_img)
        assert m.dtype == np.float32

    def test_mean_in_valid_range(self, bgr_img):
        m = compute_color_moments(bgr_img)
        # μ каналов ∈ [0, 255]
        for i in range(0, 9, 3):
            assert 0.0 <= m[i] <= 255.0

    def test_std_nonnegative(self, bgr_img):
        m = compute_color_moments(bgr_img)
        for i in range(1, 9, 3):
            assert m[i] >= 0.0

    def test_constant_image_zero_std(self):
        img = np.full((40, 40, 3), 100, dtype=np.uint8)
        m = compute_color_moments(img, colorspace="bgr")
        # σ = 0 для константного изображения
        for i in range(1, 9, 3):
            assert m[i] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("cs", ["bgr", "hsv", "lab", "yuv"])
    def test_colorspaces_accepted(self, bgr_img, cs):
        m = compute_color_moments(bgr_img, colorspace=cs)
        assert len(m) == 9

    def test_unknown_colorspace_raises(self, bgr_img):
        with pytest.raises(ValueError):
            compute_color_moments(bgr_img, colorspace="xyz_cs")


# ─── moments_distance ─────────────────────────────────────────────────────────

class TestMomentsDistance:
    def test_same_moments_returns_one(self, bgr_img):
        m = compute_color_moments(bgr_img)
        score = moments_distance(m, m)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_range_zero_to_one(self, bgr_img, red_bgr):
        m1 = compute_color_moments(bgr_img)
        m2 = compute_color_moments(red_bgr)
        score = moments_distance(m1, m2)
        assert 0.0 <= score <= 1.0

    def test_empty_vectors_return_zero(self):
        score = moments_distance(np.array([]), np.array([]))
        assert score == pytest.approx(0.0)

    def test_symmetric(self, bgr_img, red_bgr):
        m1 = compute_color_moments(bgr_img)
        m2 = compute_color_moments(red_bgr)
        assert moments_distance(m1, m2) == pytest.approx(
            moments_distance(m2, m1), abs=1e-6
        )

    def test_returns_float(self, bgr_img):
        m = compute_color_moments(bgr_img)
        assert isinstance(moments_distance(m, m), float)

    def test_different_images_lower_than_identical(self, bgr_img, red_bgr):
        m_self = compute_color_moments(bgr_img)
        m_red  = compute_color_moments(red_bgr)
        same = moments_distance(m_self, m_self)
        diff = moments_distance(m_self, m_red)
        assert same >= diff


# ─── edge_color_profile ───────────────────────────────────────────────────────

class TestEdgeColorProfile:
    @pytest.mark.parametrize("side", [0, 1, 2, 3])
    def test_all_sides_return_array(self, bgr_img, side):
        profile = edge_color_profile(bgr_img, side=side, bins=16)
        assert isinstance(profile, np.ndarray)
        assert profile.dtype == np.float32

    def test_length_is_3x_bins(self, bgr_img):
        profile = edge_color_profile(bgr_img, side=0, bins=16)
        assert len(profile) == 3 * 16

    def test_gray_returns_bins_length(self, gray_img):
        profile = edge_color_profile(gray_img, side=0, bins=16)
        assert len(profile) == 16

    def test_nonnegative(self, bgr_img):
        profile = edge_color_profile(bgr_img, side=0)
        assert (profile >= 0).all()

    def test_opposite_sides_differ_for_gradient(self):
        """Градиент сверху вниз: профили top и bottom должны различаться."""
        img = np.zeros((60, 60), dtype=np.uint8)
        img[:20, :] = 230
        img[40:, :] = 10
        top    = edge_color_profile(img, side=0, bins=16)
        bottom = edge_color_profile(img, side=2, bins=16)
        assert not np.allclose(top, bottom, atol=0.05)

    def test_border_frac_parameter(self, bgr_img):
        p_thin  = edge_color_profile(bgr_img, side=0, border_frac=0.05)
        p_thick = edge_color_profile(bgr_img, side=0, border_frac=0.30)
        # Обе валидны; могут отличаться
        assert isinstance(p_thin, np.ndarray)
        assert isinstance(p_thick, np.ndarray)


# ─── color_match_pair ─────────────────────────────────────────────────────────

class TestColorMatchPair:
    def test_returns_color_match_result(self, bgr_img):
        res = color_match_pair(bgr_img, bgr_img)
        assert isinstance(res, ColorMatchResult)

    def test_score_in_range(self, bgr_img, red_bgr):
        res = color_match_pair(bgr_img, red_bgr)
        assert 0.0 <= res.score <= 1.0

    def test_identical_images_high_score(self, bgr_img):
        res = color_match_pair(bgr_img, bgr_img)
        assert res.score > 0.7

    def test_method_field_format(self, bgr_img):
        res = color_match_pair(bgr_img, bgr_img,
                                colorspace="hsv", metric="bhatt")
        assert "hsv" in res.method
        assert "bhatt" in res.method

    def test_hist_moment_profile_scores_in_range(self, bgr_img, red_bgr):
        res = color_match_pair(bgr_img, red_bgr)
        for attr in ("hist_score", "moment_score", "profile_score"):
            val = getattr(res, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val}"

    def test_gray_input_accepted(self, gray_img):
        res = color_match_pair(gray_img, gray_img)
        assert isinstance(res, ColorMatchResult)

    def test_side_params_stored(self, bgr_img):
        res = color_match_pair(bgr_img, bgr_img, side1=0, side2=2)
        assert res.params["side1"] == 0
        assert res.params["side2"] == 2

    def test_distinct_colors_lower_score(self, red_bgr, blue_bgr):
        res_same = color_match_pair(red_bgr, red_bgr)
        res_diff = color_match_pair(red_bgr, blue_bgr)
        assert res_same.score >= res_diff.score


# ─── color_compatibility_matrix ───────────────────────────────────────────────

class TestColorCompatibilityMatrix:
    def test_shape_nxn(self, bgr_img, red_bgr):
        mat = color_compatibility_matrix([bgr_img, red_bgr])
        assert mat.shape == (2, 2)

    def test_dtype_float32(self, bgr_img, red_bgr):
        mat = color_compatibility_matrix([bgr_img, red_bgr])
        assert mat.dtype == np.float32

    def test_diagonal_is_one(self, bgr_img, red_bgr, blue_bgr):
        imgs = [bgr_img, red_bgr, blue_bgr]
        mat = color_compatibility_matrix(imgs)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=0.01)

    def test_symmetric(self, bgr_img, red_bgr):
        mat = color_compatibility_matrix([bgr_img, red_bgr])
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_values_in_range(self, bgr_img, red_bgr, blue_bgr):
        mat = color_compatibility_matrix([bgr_img, red_bgr, blue_bgr])
        assert (mat >= 0.0).all()
        assert (mat <= 1.0).all()

    def test_empty_list_gives_empty_matrix(self):
        mat = color_compatibility_matrix([])
        assert mat.shape == (0, 0)

    def test_single_image_gives_1x1(self, bgr_img):
        mat = color_compatibility_matrix([bgr_img])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == pytest.approx(1.0)

    def test_four_images(self, bgr_img, red_bgr, blue_bgr, gray_img):
        imgs = [bgr_img, red_bgr, blue_bgr, bgr_img]
        mat = color_compatibility_matrix(imgs)
        assert mat.shape == (4, 4)
        assert mat.dtype == np.float32
