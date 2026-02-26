"""
Tests for puzzle_reconstruction.verification.color_continuity_verifier.
"""
import numpy as np
import pytest

from puzzle_reconstruction.verification.color_continuity_verifier import (
    ColorContinuityConfig,
    ColorContinuityResult,
    ColorContinuityVerifier,
    verify_color_continuity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pixels(n: int, value, seed: int = 0) -> np.ndarray:
    """Return (n, 3) array filled with *value* (scalar or 3-tuple)."""
    arr = np.full((n, 3), value, dtype=np.float64)
    return arr


def _random_pixels(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, (n, 3))


# ---------------------------------------------------------------------------
# 1. ColorContinuityConfig defaults
# ---------------------------------------------------------------------------

def test_config_seam_width_default():
    cfg = ColorContinuityConfig()
    assert cfg.seam_width == 3


def test_config_method_default():
    cfg = ColorContinuityConfig()
    assert cfg.method == "lab"


def test_config_threshold_default():
    cfg = ColorContinuityConfig()
    assert cfg.threshold == 30.0


def test_config_weight_spatial_default():
    cfg = ColorContinuityConfig()
    assert cfg.weight_spatial is True


# ---------------------------------------------------------------------------
# 2. ColorContinuityResult fields
# ---------------------------------------------------------------------------

def test_result_has_mean_delta():
    r = ColorContinuityResult(mean_delta=5.0, max_delta=10.0, score=0.9,
                              is_valid=True, n_samples=10)
    assert r.mean_delta == 5.0


def test_result_has_max_delta():
    r = ColorContinuityResult(mean_delta=5.0, max_delta=10.0, score=0.9,
                              is_valid=True, n_samples=10)
    assert r.max_delta == 10.0


def test_result_has_score():
    r = ColorContinuityResult(mean_delta=5.0, max_delta=10.0, score=0.9,
                              is_valid=True, n_samples=10)
    assert r.score == 0.9


def test_result_has_is_valid():
    r = ColorContinuityResult(mean_delta=5.0, max_delta=10.0, score=0.9,
                              is_valid=True, n_samples=10)
    assert r.is_valid is True


def test_result_has_n_samples():
    r = ColorContinuityResult(mean_delta=5.0, max_delta=10.0, score=0.9,
                              is_valid=True, n_samples=10)
    assert r.n_samples == 10


# ---------------------------------------------------------------------------
# 3. verify_seam with identical pixels → score ≈ 1.0
# ---------------------------------------------------------------------------

def test_identical_pixels_high_score():
    pixels = _make_pixels(20, 128.0)
    verifier = ColorContinuityVerifier()
    result = verifier.verify_seam(pixels, pixels.copy())
    assert result.score > 0.95


def test_identical_pixels_mean_delta_zero():
    pixels = _make_pixels(10, 50.0)
    verifier = ColorContinuityVerifier()
    result = verifier.verify_seam(pixels, pixels.copy())
    assert result.mean_delta == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. verify_seam with very different colors → low score
# ---------------------------------------------------------------------------

def test_very_different_colors_low_score():
    pa = _make_pixels(20, 0.0)    # black
    pb = _make_pixels(20, 255.0)  # white
    verifier = ColorContinuityVerifier()
    result = verifier.verify_seam(pa, pb)
    assert result.score < 0.3


# ---------------------------------------------------------------------------
# 5. score is in [0, 1]
# ---------------------------------------------------------------------------

def test_score_in_range_random():
    np.random.seed(7)
    pa = np.random.uniform(0, 255, (50, 3))
    pb = np.random.uniform(0, 255, (50, 3))
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# 6. n_samples matches input size
# ---------------------------------------------------------------------------

def test_n_samples_matches_input():
    pa = _make_pixels(15, 100.0)
    pb = _make_pixels(15, 110.0)
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 15


def test_n_samples_uses_min_length():
    pa = _make_pixels(20, 100.0)
    pb = _make_pixels(10, 100.0)
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 10


# ---------------------------------------------------------------------------
# 7. is_valid reflects threshold
# ---------------------------------------------------------------------------

def test_is_valid_true_when_below_threshold():
    pa = _make_pixels(10, 100.0)
    pb = _make_pixels(10, 101.0)  # tiny difference
    cfg = ColorContinuityConfig(threshold=30.0, method="rgb")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert result.is_valid is True


def test_is_valid_false_when_above_threshold():
    pa = _make_pixels(10, 0.0)
    pb = _make_pixels(10, 255.0)
    cfg = ColorContinuityConfig(threshold=30.0, method="rgb")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert result.is_valid is False


# ---------------------------------------------------------------------------
# 8–10. Color spaces work
# ---------------------------------------------------------------------------

def test_rgb_color_space():
    pa = _random_pixels(20, seed=1)
    pb = _random_pixels(20, seed=2)
    cfg = ColorContinuityConfig(method="rgb")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert 0.0 <= result.score <= 1.0
    assert result.n_samples == 20


def test_lab_color_space():
    pa = _random_pixels(20, seed=3)
    pb = _random_pixels(20, seed=4)
    cfg = ColorContinuityConfig(method="lab")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert 0.0 <= result.score <= 1.0


def test_hsv_color_space():
    pa = _random_pixels(20, seed=5)
    pb = _random_pixels(20, seed=6)
    cfg = ColorContinuityConfig(method="hsv")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# 11. mean_delta ≤ max_delta always
# ---------------------------------------------------------------------------

def test_mean_delta_le_max_delta():
    np.random.seed(99)
    pa = np.random.uniform(0, 255, (30, 3))
    pb = np.random.uniform(0, 255, (30, 3))
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.mean_delta <= result.max_delta + 1e-9


# ---------------------------------------------------------------------------
# 12. Module-level verify_color_continuity works
# ---------------------------------------------------------------------------

def test_module_level_function_returns_result():
    pa = _make_pixels(10, 128.0)
    pb = _make_pixels(10, 128.0)
    result = verify_color_continuity(pa, pb)
    assert isinstance(result, ColorContinuityResult)


def test_module_level_function_uses_config():
    pa = _make_pixels(10, 0.0)
    pb = _make_pixels(10, 255.0)
    cfg = ColorContinuityConfig(threshold=500.0, method="rgb")
    result = verify_color_continuity(pa, pb, config=cfg)
    # With a very high threshold, even black vs white should be "valid"
    assert result.is_valid is True


# ---------------------------------------------------------------------------
# 13. Single pixel pair
# ---------------------------------------------------------------------------

def test_single_pixel_pair():
    pa = np.array([[200.0, 100.0, 50.0]])
    pb = np.array([[200.0, 100.0, 50.0]])
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 1
    assert result.mean_delta == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 14. 100 pixel pairs
# ---------------------------------------------------------------------------

def test_hundred_pixel_pairs():
    np.random.seed(11)
    pa = np.random.uniform(0, 255, (100, 3))
    pb = np.random.uniform(0, 255, (100, 3))
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 100
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# 15. All-black vs all-white → low score
# ---------------------------------------------------------------------------

def test_black_vs_white_low_score():
    pa = _make_pixels(20, 0.0)
    pb = _make_pixels(20, 255.0)
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.score < 0.5


# ---------------------------------------------------------------------------
# 16. Similar grays → high score
# ---------------------------------------------------------------------------

def test_similar_grays_high_score():
    pa = _make_pixels(20, 128.0)
    pb = _make_pixels(20, 130.0)
    cfg = ColorContinuityConfig(method="rgb")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert result.score > 0.8


# ---------------------------------------------------------------------------
# 17. score_from_delta at delta=0 → 1.0
# ---------------------------------------------------------------------------

def test_score_from_delta_zero():
    s = ColorContinuityVerifier.score_from_delta(0.0, 30.0)
    assert s == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 18. score_from_delta at large delta → near 0
# ---------------------------------------------------------------------------

def test_score_from_delta_large():
    s = ColorContinuityVerifier.score_from_delta(1000.0, 30.0)
    assert s < 0.01


# ---------------------------------------------------------------------------
# 19. Config threshold affects is_valid
# ---------------------------------------------------------------------------

def test_threshold_affects_is_valid_strict():
    pa = _make_pixels(10, 0.0)
    pb = _make_pixels(10, 20.0)  # small-ish RGB difference
    cfg_strict = ColorContinuityConfig(threshold=1.0, method="rgb")
    cfg_loose  = ColorContinuityConfig(threshold=1000.0, method="rgb")
    r_strict = ColorContinuityVerifier(cfg_strict).verify_seam(pa, pb)
    r_loose  = ColorContinuityVerifier(cfg_loose).verify_seam(pa, pb)
    assert r_strict.is_valid is False
    assert r_loose.is_valid is True


# ---------------------------------------------------------------------------
# 20. Result is deterministic
# ---------------------------------------------------------------------------

def test_result_is_deterministic():
    np.random.seed(42)
    pa = np.random.uniform(0, 255, (25, 3))
    pb = np.random.uniform(0, 255, (25, 3))
    r1 = ColorContinuityVerifier().verify_seam(pa, pb)
    r2 = ColorContinuityVerifier().verify_seam(pa, pb)
    assert r1.mean_delta == pytest.approx(r2.mean_delta)
    assert r1.score == pytest.approx(r2.score)
    assert r1.is_valid == r2.is_valid
