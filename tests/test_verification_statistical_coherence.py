"""Tests for puzzle_reconstruction.verification.statistical_coherence"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/user/meta2')

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


def make_patch(h=20, w=20, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_color_patch(h=20, w=20, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ─── StatisticalCoherenceConfig ───────────────────────────────────────────────

def test_config_defaults():
    cfg = StatisticalCoherenceConfig()
    assert cfg.n_bins == 32
    assert cfg.method == "histogram"
    assert cfg.threshold == 0.3
    assert cfg.use_texture is True


def test_config_custom():
    cfg = StatisticalCoherenceConfig(n_bins=64, method="moments", threshold=0.5, use_texture=False)
    assert cfg.n_bins == 64
    assert cfg.method == "moments"
    assert cfg.threshold == 0.5


# ─── StatisticalCoherenceResult ───────────────────────────────────────────────

def test_result_is_coherent_true():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.9,
        moment_similarity=0.85,
        texture_similarity=0.8,
        overall_score=0.85,
        is_coherent=True,
    )
    assert r.is_coherent


def test_result_is_coherent_false():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.1,
        moment_similarity=0.1,
        texture_similarity=0.1,
        overall_score=0.1,
        is_coherent=False,
    )
    assert not r.is_coherent


# ─── _to_gray_flat ────────────────────────────────────────────────────────────

def test_to_gray_flat_1d():
    arr = np.array([1.0, 2.0, 3.0])
    result = _to_gray_flat(arr)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, arr)


def test_to_gray_flat_2d():
    arr = np.ones((5, 5)) * 100.0
    result = _to_gray_flat(arr)
    assert result.ndim == 1
    assert len(result) == 25


def test_to_gray_flat_3d():
    arr = np.full((4, 4, 3), 200.0)
    result = _to_gray_flat(arr)
    assert result.ndim == 1
    assert len(result) == 16


def test_to_gray_flat_invalid():
    arr = np.ones((2, 2, 2, 2))
    with pytest.raises(ValueError):
        _to_gray_flat(arr)


# ─── _bhattacharyya_coefficient ───────────────────────────────────────────────

def test_bhattacharyya_identical():
    hist = np.array([10.0, 20.0, 30.0, 40.0])
    bc = _bhattacharyya_coefficient(hist, hist)
    assert bc == pytest.approx(1.0)


def test_bhattacharyya_no_overlap():
    hist_a = np.array([100.0, 0.0, 0.0, 0.0])
    hist_b = np.array([0.0, 0.0, 0.0, 100.0])
    bc = _bhattacharyya_coefficient(hist_a, hist_b)
    assert bc == pytest.approx(0.0)


def test_bhattacharyya_empty():
    bc = _bhattacharyya_coefficient(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert bc == 0.0


def test_bhattacharyya_range():
    hist_a = np.random.rand(32)
    hist_b = np.random.rand(32)
    bc = _bhattacharyya_coefficient(hist_a, hist_b)
    assert 0.0 <= bc <= 1.0


# ─── _skewness ────────────────────────────────────────────────────────────────

def test_skewness_symmetric():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sk = _skewness(x)
    assert abs(sk) < 0.1


def test_skewness_too_short():
    assert _skewness(np.array([1.0, 2.0])) == 0.0


def test_skewness_constant():
    x = np.full(10, 5.0)
    assert _skewness(x) == 0.0


# ─── _glcm_contrast ───────────────────────────────────────────────────────────

def test_glcm_contrast_uniform():
    arr = np.full((10, 10), 128.0)
    c = _glcm_contrast(arr)
    assert c == pytest.approx(0.0)


def test_glcm_contrast_range():
    arr = np.random.rand(10, 10) * 255
    c = _glcm_contrast(arr)
    assert 0.0 <= c <= 1.0


def test_glcm_contrast_1d_fallback():
    arr = np.array([0.0, 255.0, 0.0, 255.0])
    c = _glcm_contrast(arr.reshape(1, -1))
    assert 0.0 <= c <= 1.0


# ─── _glcm_energy ─────────────────────────────────────────────────────────────

def test_glcm_energy_uniform():
    arr = np.full((10, 10), 100.0)
    e = _glcm_energy(arr)
    assert e == pytest.approx(1.0)


def test_glcm_energy_range():
    arr = np.random.rand(10, 10) * 255
    e = _glcm_energy(arr)
    assert 0.0 < e <= 1.0


# ─── StatisticalCoherenceVerifier.verify ──────────────────────────────────────

def test_verify_identical_patches():
    v = StatisticalCoherenceVerifier()
    patch = make_patch(20, 20, fill=128)
    result = v.verify(patch, patch)
    assert isinstance(result, StatisticalCoherenceResult)
    assert result.histogram_similarity == pytest.approx(1.0)


def test_verify_different_patches():
    v = StatisticalCoherenceVerifier()
    p1 = make_patch(20, 20, fill=0)
    p2 = make_patch(20, 20, fill=255)
    result = v.verify(p1, p2)
    # Different patches should have lower score
    assert 0.0 <= result.overall_score <= 1.0


def test_verify_method_histogram():
    cfg = StatisticalCoherenceConfig(method="histogram", use_texture=False)
    v = StatisticalCoherenceVerifier(cfg)
    p1 = make_patch(10, 10)
    result = v.verify(p1, p1)
    assert result.histogram_similarity == pytest.approx(1.0)


def test_verify_method_moments():
    cfg = StatisticalCoherenceConfig(method="moments", use_texture=False)
    v = StatisticalCoherenceVerifier(cfg)
    p1 = make_patch(10, 10)
    result = v.verify(p1, p1)
    assert result.moment_similarity == pytest.approx(1.0, abs=0.05)


def test_verify_method_both():
    cfg = StatisticalCoherenceConfig(method="both")
    v = StatisticalCoherenceVerifier(cfg)
    p1 = make_patch(10, 10, fill=100)
    p2 = make_patch(10, 10, fill=100)
    result = v.verify(p1, p2)
    assert 0.0 <= result.overall_score <= 1.0


def test_verify_color_patches():
    v = StatisticalCoherenceVerifier()
    p1 = make_color_patch(20, 20, fill=100)
    p2 = make_color_patch(20, 20, fill=100)
    result = v.verify(p1, p2)
    assert result.histogram_similarity == pytest.approx(1.0)


def test_verify_is_coherent_threshold():
    cfg = StatisticalCoherenceConfig(threshold=0.1)
    v = StatisticalCoherenceVerifier(cfg)
    p = make_patch(10, 10)
    result = v.verify(p, p)
    assert result.is_coherent is True


def test_verify_1d_arrays():
    v = StatisticalCoherenceVerifier()
    p1 = np.array([10, 20, 30, 40, 50], dtype=np.float64)
    p2 = np.array([10, 20, 30, 40, 50], dtype=np.float64)
    result = v.verify(p1, p2)
    assert result.histogram_similarity == pytest.approx(1.0)


# ─── cohere_score ─────────────────────────────────────────────────────────────

def test_cohere_score_identical():
    patch = make_patch(20, 20, fill=128)
    score = cohere_score(patch, patch)
    assert 0.0 <= score <= 1.0
    # Identical → high score
    assert score > 0.5


def test_cohere_score_range():
    p1 = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    p2 = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    score = cohere_score(p1, p2)
    assert 0.0 <= score <= 1.0


def test_cohere_score_custom_config():
    cfg = StatisticalCoherenceConfig(method="moments", threshold=0.2)
    patch = make_patch(10, 10)
    score = cohere_score(patch, patch, config=cfg)
    assert score >= 0.0
