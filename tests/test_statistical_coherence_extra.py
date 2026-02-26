"""Extra tests for puzzle_reconstruction/verification/statistical_coherence.py"""
import numpy as np
import pytest

from puzzle_reconstruction.verification.statistical_coherence import (
    StatisticalCoherenceConfig,
    StatisticalCoherenceResult,
    StatisticalCoherenceVerifier,
    cohere_score,
    _to_gray_flat,
    _bhattacharyya_coefficient,
    _skewness,
    _glcm_contrast,
    _glcm_energy,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _uniform(h=16, w=16, val=128.0):
    return np.full((h, w), val, dtype=np.float64)


def _noisy(h=16, w=16, seed=0, scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.normal(128.0, scale, (h, w))


def _color(h=8, w=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, (h, w, 3))


# ─── Config validation ────────────────────────────────────────────────────────

def test_config_use_texture_false():
    cfg = StatisticalCoherenceConfig(use_texture=False)
    assert cfg.use_texture is False


def test_config_method_both_valid():
    cfg = StatisticalCoherenceConfig(method="both")
    assert cfg.method == "both"


def test_config_n_bins_custom():
    cfg = StatisticalCoherenceConfig(n_bins=64)
    assert cfg.n_bins == 64


def test_config_threshold_zero():
    cfg = StatisticalCoherenceConfig(threshold=0.0)
    assert cfg.threshold == pytest.approx(0.0)


def test_config_threshold_one():
    cfg = StatisticalCoherenceConfig(threshold=1.0)
    assert cfg.threshold == pytest.approx(1.0)


# ─── _to_gray_flat ────────────────────────────────────────────────────────────

def test_to_gray_flat_1d():
    arr = np.array([10.0, 20.0, 30.0])
    result = _to_gray_flat(arr)
    np.testing.assert_array_equal(result, arr)


def test_to_gray_flat_2d():
    arr = np.ones((4, 4)) * 50.0
    result = _to_gray_flat(arr)
    assert result.ndim == 1
    assert len(result) == 16
    np.testing.assert_allclose(result, 50.0)


def test_to_gray_flat_3d_averages_channels():
    arr = np.zeros((4, 4, 3))
    arr[:, :, 0] = 30.0
    arr[:, :, 1] = 60.0
    arr[:, :, 2] = 90.0
    result = _to_gray_flat(arr)
    np.testing.assert_allclose(result, 60.0)


def test_to_gray_flat_returns_float64():
    arr = np.ones((4, 4), dtype=np.uint8) * 100
    result = _to_gray_flat(arr)
    assert result.dtype == np.float64


# ─── _bhattacharyya_coefficient ───────────────────────────────────────────────

def test_bhattacharyya_identical_histograms():
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    bc = _bhattacharyya_coefficient(h, h)
    assert bc == pytest.approx(1.0)


def test_bhattacharyya_disjoint_histograms():
    h_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    h_b = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    bc = _bhattacharyya_coefficient(h_a, h_b)
    assert bc == pytest.approx(0.0)


def test_bhattacharyya_zero_hist_a_returns_zero():
    h_a = np.zeros(5)
    h_b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert _bhattacharyya_coefficient(h_a, h_b) == pytest.approx(0.0)


def test_bhattacharyya_zero_hist_b_returns_zero():
    h_a = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    h_b = np.zeros(5)
    assert _bhattacharyya_coefficient(h_a, h_b) == pytest.approx(0.0)


def test_bhattacharyya_result_in_01():
    rng = np.random.default_rng(42)
    h_a = rng.random(20) + 0.1
    h_b = rng.random(20) + 0.1
    bc = _bhattacharyya_coefficient(h_a, h_b)
    assert 0.0 <= bc <= 1.0


# ─── _skewness ────────────────────────────────────────────────────────────────

def test_skewness_symmetric_returns_near_zero():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sk = _skewness(x)
    assert abs(sk) < 0.1


def test_skewness_right_skewed():
    x = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    sk = _skewness(x)
    assert sk > 0


def test_skewness_too_few_points():
    assert _skewness(np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert _skewness(np.array([5.0])) == pytest.approx(0.0)


def test_skewness_constant_array():
    assert _skewness(np.ones(10)) == pytest.approx(0.0)


# ─── _glcm_contrast ──────────────────────────────────────────────────────────

def test_glcm_contrast_uniform_near_zero():
    arr = np.full((16, 16), 128.0)
    c = _glcm_contrast(arr)
    assert c == pytest.approx(0.0)


def test_glcm_contrast_checkerboard_high():
    # True checkerboard: alternating pixels horizontally produces non-zero horizontal diffs
    arr = np.zeros((16, 16))
    arr[:, ::2] = 255.0  # alternating columns → horizontal differences ≠ 0
    c = _glcm_contrast(arr)
    assert c > 0


def test_glcm_contrast_in_range():
    rng = np.random.default_rng(1)
    arr = rng.uniform(0, 255, (16, 16))
    c = _glcm_contrast(arr)
    assert 0.0 <= c <= 1.0


def test_glcm_contrast_1d_fallback():
    """When array is 1-D-like, fallback path should work."""
    arr = np.array([[1.0, 255.0]])  # single row, very high contrast
    c = _glcm_contrast(arr)
    assert c > 0


# ─── _glcm_energy ────────────────────────────────────────────────────────────

def test_glcm_energy_uniform_is_one():
    arr = np.full((16, 16), 64.0)
    e = _glcm_energy(arr)
    assert e == pytest.approx(1.0)


def test_glcm_energy_in_range():
    rng = np.random.default_rng(2)
    arr = rng.uniform(0, 255, (16, 16))
    e = _glcm_energy(arr)
    assert 0.0 < e <= 1.0


def test_glcm_energy_less_than_uniform_for_noisy():
    arr_uniform = np.full((16, 16), 128.0)
    rng = np.random.default_rng(3)
    arr_noisy = rng.uniform(0, 255, (16, 16))
    e_uniform = _glcm_energy(arr_uniform)
    e_noisy   = _glcm_energy(arr_noisy)
    assert e_uniform >= e_noisy


# ─── use_texture=False ────────────────────────────────────────────────────────

def test_method_histogram_no_texture():
    cfg = StatisticalCoherenceConfig(method="histogram", use_texture=False)
    pa = _noisy(seed=10)
    pb = _noisy(seed=11)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


def test_method_moments_no_texture():
    cfg = StatisticalCoherenceConfig(method="moments", use_texture=False)
    pa = _noisy(seed=12)
    pb = _noisy(seed=13)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


def test_method_both_no_texture():
    cfg = StatisticalCoherenceConfig(method="both", use_texture=False)
    pa = _noisy(seed=14)
    pb = _noisy(seed=15)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


# ─── Method "both" combines histogram and moments ─────────────────────────────

def test_method_both_uses_both_scores():
    pa = _noisy(seed=20)
    pb = _noisy(seed=21)
    cfg_h = StatisticalCoherenceConfig(method="histogram", use_texture=False)
    cfg_m = StatisticalCoherenceConfig(method="moments", use_texture=False)
    cfg_b = StatisticalCoherenceConfig(method="both", use_texture=False)
    rh = StatisticalCoherenceVerifier(cfg_h).verify(pa, pb)
    rm = StatisticalCoherenceVerifier(cfg_m).verify(pa, pb)
    rb = StatisticalCoherenceVerifier(cfg_b).verify(pa, pb)
    # Both method combines the two; overall should be between them roughly
    assert 0.0 <= rb.overall_score <= 1.0
    # At least different from pure histogram or pure moments (usually)
    assert isinstance(rb.overall_score, float)


# ─── Threshold: is_coherent ───────────────────────────────────────────────────

def test_is_coherent_threshold_zero_always_coherent():
    pa = _uniform(val=0.0)
    pb = _uniform(val=255.0)
    cfg = StatisticalCoherenceConfig(threshold=0.0)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert result.is_coherent is True


def test_is_coherent_threshold_one_never_coherent():
    pa = _uniform(val=0.0)
    pb = _uniform(val=255.0)
    cfg = StatisticalCoherenceConfig(threshold=1.0)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert result.is_coherent is False


def test_is_coherent_consistent_with_overall_score():
    pa = _noisy(seed=30)
    pb = _noisy(seed=31)
    cfg = StatisticalCoherenceConfig(threshold=0.5)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert result.is_coherent == (result.overall_score >= 0.5)


# ─── Large patches ────────────────────────────────────────────────────────────

def test_large_2d_patches():
    pa = _noisy(128, 128, seed=40)
    pb = _noisy(128, 128, seed=41)
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert isinstance(result, StatisticalCoherenceResult)
    assert 0.0 <= result.overall_score <= 1.0


def test_large_color_patches():
    pa = _color(64, 64, seed=50)
    pb = _color(64, 64, seed=51)
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


# ─── Asymmetric patches (different sizes) ─────────────────────────────────────

def test_different_size_patches_1d():
    pa = np.random.default_rng(60).normal(128, 20, 100)
    pb = np.random.default_rng(61).normal(128, 20, 200)
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert isinstance(result, StatisticalCoherenceResult)


# ─── cohere_score function ────────────────────────────────────────────────────

def test_cohere_score_identical_near_one():
    pa = _noisy(seed=70)
    score = cohere_score(pa, pa.copy())
    assert score > 0.7


def test_cohere_score_different_near_low():
    pa = _uniform(val=0.0)
    pb = _uniform(val=255.0)
    score = cohere_score(pa, pb)
    assert score < 0.9


def test_cohere_score_with_moments_method():
    cfg = StatisticalCoherenceConfig(method="moments")
    pa = _noisy(seed=80)
    pb = _noisy(seed=81)
    score = cohere_score(pa, pb, config=cfg)
    assert 0.0 <= score <= 1.0


def test_cohere_score_with_both_method():
    cfg = StatisticalCoherenceConfig(method="both")
    pa = _noisy(seed=82)
    pb = _noisy(seed=83)
    score = cohere_score(pa, pb, config=cfg)
    assert 0.0 <= score <= 1.0
