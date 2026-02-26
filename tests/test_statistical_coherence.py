"""
Tests for puzzle_reconstruction.verification.statistical_coherence.
"""
import numpy as np
import pytest

from puzzle_reconstruction.verification.statistical_coherence import (
    StatisticalCoherenceConfig,
    StatisticalCoherenceResult,
    StatisticalCoherenceVerifier,
    cohere_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_patch(h: int, w: int, value: float) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float64)


def _noisy_patch(h: int, w: int, seed: int = 0, scale: float = 50.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(128.0, scale, (h, w))


def _color_patch(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, (h, w, 3))


# ---------------------------------------------------------------------------
# 1. StatisticalCoherenceConfig defaults
# ---------------------------------------------------------------------------

def test_config_n_bins_default():
    cfg = StatisticalCoherenceConfig()
    assert cfg.n_bins == 32


def test_config_method_default():
    cfg = StatisticalCoherenceConfig()
    assert cfg.method == "histogram"


def test_config_threshold_default():
    cfg = StatisticalCoherenceConfig()
    assert cfg.threshold == pytest.approx(0.3)


def test_config_use_texture_default():
    cfg = StatisticalCoherenceConfig()
    assert cfg.use_texture is True


# ---------------------------------------------------------------------------
# 2. StatisticalCoherenceResult fields
# ---------------------------------------------------------------------------

def test_result_has_histogram_similarity():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.8, moment_similarity=0.7,
        texture_similarity=0.6, overall_score=0.7, is_coherent=True)
    assert r.histogram_similarity == pytest.approx(0.8)


def test_result_has_moment_similarity():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.8, moment_similarity=0.7,
        texture_similarity=0.6, overall_score=0.7, is_coherent=True)
    assert r.moment_similarity == pytest.approx(0.7)


def test_result_has_texture_similarity():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.8, moment_similarity=0.7,
        texture_similarity=0.6, overall_score=0.7, is_coherent=True)
    assert r.texture_similarity == pytest.approx(0.6)


def test_result_has_overall_score():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.8, moment_similarity=0.7,
        texture_similarity=0.6, overall_score=0.7, is_coherent=True)
    assert r.overall_score == pytest.approx(0.7)


def test_result_has_is_coherent():
    r = StatisticalCoherenceResult(
        histogram_similarity=0.8, moment_similarity=0.7,
        texture_similarity=0.6, overall_score=0.7, is_coherent=True)
    assert r.is_coherent is True


# ---------------------------------------------------------------------------
# 3. verify with identical patches → high score
# ---------------------------------------------------------------------------

def test_identical_patches_high_score():
    patch = _noisy_patch(16, 16, seed=10)
    result = StatisticalCoherenceVerifier().verify(patch, patch.copy())
    assert result.overall_score > 0.7


def test_identical_patches_is_coherent():
    patch = _noisy_patch(16, 16, seed=20)
    result = StatisticalCoherenceVerifier().verify(patch, patch.copy())
    assert result.is_coherent is True


# ---------------------------------------------------------------------------
# 4. verify with very different patches → low score
# ---------------------------------------------------------------------------

def test_very_different_patches_low_score():
    pa = _uniform_patch(16, 16, 0.0)    # pure black
    pb = _uniform_patch(16, 16, 255.0)  # pure white
    cfg = StatisticalCoherenceConfig(threshold=0.3)
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    # Different brightness → low histogram/moment similarity
    assert result.overall_score < 0.9  # clearly distinguishable from identical


# ---------------------------------------------------------------------------
# 5. overall_score in [0, 1]
# ---------------------------------------------------------------------------

def test_overall_score_in_range():
    np.random.seed(55)
    pa = np.random.uniform(0, 255, (8, 8))
    pb = np.random.uniform(0, 255, (8, 8))
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 6. histogram_similarity in [0, 1]
# ---------------------------------------------------------------------------

def test_histogram_similarity_in_range():
    np.random.seed(56)
    pa = np.random.uniform(0, 255, (8, 8))
    pb = np.random.uniform(0, 255, (8, 8))
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert 0.0 <= result.histogram_similarity <= 1.0


# ---------------------------------------------------------------------------
# 7. moment_similarity in [0, 1]
# ---------------------------------------------------------------------------

def test_moment_similarity_in_range():
    np.random.seed(57)
    pa = np.random.uniform(0, 255, (8, 8))
    pb = np.random.uniform(0, 255, (8, 8))
    cfg = StatisticalCoherenceConfig(method="moments")
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.moment_similarity <= 1.0


# ---------------------------------------------------------------------------
# 8. texture_similarity in [0, 1]
# ---------------------------------------------------------------------------

def test_texture_similarity_in_range():
    np.random.seed(58)
    pa = np.random.uniform(0, 255, (8, 8))
    pb = np.random.uniform(0, 255, (8, 8))
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert 0.0 <= result.texture_similarity <= 1.0


# ---------------------------------------------------------------------------
# 9. is_coherent reflects threshold
# ---------------------------------------------------------------------------

def test_is_coherent_true_with_low_threshold():
    pa = _noisy_patch(8, 8, seed=1)
    pb = _noisy_patch(8, 8, seed=2)
    cfg = StatisticalCoherenceConfig(threshold=0.01)  # very easy to pass
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert result.is_coherent is True


def test_is_coherent_false_with_high_threshold():
    pa = _uniform_patch(8, 8, 0.0)
    pb = _uniform_patch(8, 8, 255.0)
    cfg = StatisticalCoherenceConfig(threshold=0.999)  # nearly impossible
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert result.is_coherent is False


# ---------------------------------------------------------------------------
# 10. method="histogram" works
# ---------------------------------------------------------------------------

def test_method_histogram():
    pa = _noisy_patch(8, 8, seed=3)
    pb = _noisy_patch(8, 8, seed=4)
    cfg = StatisticalCoherenceConfig(method="histogram")
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 11. method="moments" works
# ---------------------------------------------------------------------------

def test_method_moments():
    pa = _noisy_patch(8, 8, seed=5)
    pb = _noisy_patch(8, 8, seed=6)
    cfg = StatisticalCoherenceConfig(method="moments")
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 12. method="both" works
# ---------------------------------------------------------------------------

def test_method_both():
    pa = _noisy_patch(8, 8, seed=7)
    pb = _noisy_patch(8, 8, seed=8)
    cfg = StatisticalCoherenceConfig(method="both")
    result = StatisticalCoherenceVerifier(cfg).verify(pa, pb)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 13. 1-D input arrays work
# ---------------------------------------------------------------------------

def test_1d_input_arrays():
    np.random.seed(30)
    pa = np.random.uniform(0, 255, 64)
    pb = np.random.uniform(0, 255, 64)
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert isinstance(result, StatisticalCoherenceResult)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 14. 2-D patches work (grayscale)
# ---------------------------------------------------------------------------

def test_2d_patches_grayscale():
    pa = _noisy_patch(16, 16, seed=11)
    pb = _noisy_patch(16, 16, seed=12)
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert isinstance(result, StatisticalCoherenceResult)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 15. 3-D patches work (color)
# ---------------------------------------------------------------------------

def test_3d_patches_color():
    pa = _color_patch(8, 8, seed=13)
    pb = _color_patch(8, 8, seed=14)
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert isinstance(result, StatisticalCoherenceResult)
    assert 0.0 <= result.overall_score <= 1.0


# ---------------------------------------------------------------------------
# 16. Uniform patch vs noisy patch → low histogram similarity
# ---------------------------------------------------------------------------

def test_uniform_vs_noisy_low_similarity():
    uniform = _uniform_patch(16, 16, 128.0)
    noisy   = _noisy_patch(16, 16, seed=15, scale=60.0)
    result = StatisticalCoherenceVerifier().verify(uniform, noisy)
    # Uniform patch has a degenerate histogram → low Bhattacharyya
    assert result.histogram_similarity < 0.9


# ---------------------------------------------------------------------------
# 17. Same noise level → higher texture similarity
# ---------------------------------------------------------------------------

def test_same_noise_level_higher_texture_similarity():
    pa = _noisy_patch(16, 16, seed=16, scale=30.0)
    pb = _noisy_patch(16, 16, seed=17, scale=30.0)
    pc = _uniform_patch(16, 16, 128.0)  # very different texture

    verifier = StatisticalCoherenceVerifier()
    r_same    = verifier.verify(pa, pb)
    r_diff    = verifier.verify(pa, pc)
    assert r_same.texture_similarity >= r_diff.texture_similarity


# ---------------------------------------------------------------------------
# 18. cohere_score function returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_cohere_score_returns_float():
    np.random.seed(40)
    pa = np.random.uniform(0, 255, (8, 8))
    pb = np.random.uniform(0, 255, (8, 8))
    s = cohere_score(pa, pb)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_cohere_score_with_config():
    pa = _noisy_patch(8, 8, seed=41)
    pb = _noisy_patch(8, 8, seed=42)
    cfg = StatisticalCoherenceConfig(method="both")
    s = cohere_score(pa, pb, config=cfg)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 19. Result is deterministic
# ---------------------------------------------------------------------------

def test_result_is_deterministic():
    np.random.seed(50)
    pa = np.random.uniform(0, 255, (12, 12))
    pb = np.random.uniform(0, 255, (12, 12))
    verifier = StatisticalCoherenceVerifier()
    r1 = verifier.verify(pa, pb)
    r2 = verifier.verify(pa, pb)
    assert r1.overall_score == pytest.approx(r2.overall_score)
    assert r1.histogram_similarity == pytest.approx(r2.histogram_similarity)
    assert r1.is_coherent == r2.is_coherent


# ---------------------------------------------------------------------------
# 20. Small patches (4x4) work
# ---------------------------------------------------------------------------

def test_small_patches_4x4():
    np.random.seed(60)
    pa = np.random.uniform(0, 255, (4, 4))
    pb = np.random.uniform(0, 255, (4, 4))
    result = StatisticalCoherenceVerifier().verify(pa, pb)
    assert isinstance(result, StatisticalCoherenceResult)
    assert 0.0 <= result.overall_score <= 1.0
