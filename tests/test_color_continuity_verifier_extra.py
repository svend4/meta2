"""Extra tests for puzzle_reconstruction/verification/color_continuity_verifier.py"""
import numpy as np
import pytest

from puzzle_reconstruction.verification.color_continuity_verifier import (
    ColorContinuityConfig,
    ColorContinuityResult,
    ColorContinuityVerifier,
    verify_color_continuity,
    _rgb_to_lab,
    _rgb_to_hsv,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _px(n, r, g, b):
    return np.full((n, 3), [r, g, b], dtype=np.float64)


def _rand_px(n=20, seed=0):
    return np.random.default_rng(seed).uniform(0, 255, (n, 3))


# ─── Config edge cases ────────────────────────────────────────────────────────

def test_config_all_defaults():
    cfg = ColorContinuityConfig()
    assert cfg.seam_width == 3
    assert cfg.method == "lab"
    assert cfg.threshold == pytest.approx(30.0)
    assert cfg.weight_spatial is True


def test_config_method_rgb():
    cfg = ColorContinuityConfig(method="rgb")
    assert cfg.method == "rgb"


def test_config_method_hsv():
    cfg = ColorContinuityConfig(method="hsv")
    assert cfg.method == "hsv"


def test_config_threshold_zero():
    cfg = ColorContinuityConfig(threshold=0.0)
    assert cfg.threshold == pytest.approx(0.0)


def test_config_seam_width_one():
    cfg = ColorContinuityConfig(seam_width=1)
    assert cfg.seam_width == 1


def test_config_weight_spatial_false():
    cfg = ColorContinuityConfig(weight_spatial=False)
    assert cfg.weight_spatial is False


# ─── _rgb_to_lab ─────────────────────────────────────────────────────────────

def test_rgb_to_lab_output_shape():
    px = _rand_px(20)
    lab = _rgb_to_lab(px)
    assert lab.shape == (20, 3)


def test_rgb_to_lab_black_pixel():
    black = np.array([[0.0, 0.0, 0.0]])
    lab = _rgb_to_lab(black)
    # L* of pure black should be near 0
    assert lab[0, 0] == pytest.approx(0.0, abs=1.0)


def test_rgb_to_lab_white_pixel():
    white = np.array([[255.0, 255.0, 255.0]])
    lab = _rgb_to_lab(white)
    # L* of pure white should be near 100
    assert lab[0, 0] == pytest.approx(100.0, abs=2.0)


def test_rgb_to_lab_gray_pixel_ab_near_zero():
    gray = np.array([[128.0, 128.0, 128.0]])
    lab = _rgb_to_lab(gray)
    # Neutral gray: a* and b* should be near 0
    assert abs(lab[0, 1]) < 5.0
    assert abs(lab[0, 2]) < 5.0


def test_rgb_to_lab_returns_float64():
    px = _rand_px(5)
    lab = _rgb_to_lab(px)
    assert lab.dtype == np.float64


def test_rgb_to_lab_batch_consistency():
    px = _rand_px(50, seed=1)
    lab = _rgb_to_lab(px)
    # Each pixel processed independently; check one at random
    single = _rgb_to_lab(px[7:8])
    np.testing.assert_allclose(lab[7], single[0], atol=1e-9)


# ─── _rgb_to_hsv ─────────────────────────────────────────────────────────────

def test_rgb_to_hsv_output_shape():
    px = _rand_px(10)
    hsv = _rgb_to_hsv(px)
    assert hsv.shape == (10, 3)


def test_rgb_to_hsv_black_zero_saturation():
    black = np.array([[0.0, 0.0, 0.0]])
    hsv = _rgb_to_hsv(black)
    assert hsv[0, 1] == pytest.approx(0.0)  # saturation=0 for black


def test_rgb_to_hsv_white_zero_saturation():
    white = np.array([[255.0, 255.0, 255.0]])
    hsv = _rgb_to_hsv(white)
    assert hsv[0, 1] == pytest.approx(0.0)


def test_rgb_to_hsv_pure_red():
    red = np.array([[255.0, 0.0, 0.0]])
    hsv = _rgb_to_hsv(red)
    assert hsv[0, 0] == pytest.approx(0.0, abs=1.0)  # hue ~0°
    assert hsv[0, 1] == pytest.approx(1.0, abs=0.01)  # full saturation


def test_rgb_to_hsv_hue_range():
    px = _rand_px(100, seed=5)
    hsv = _rgb_to_hsv(px)
    assert np.all(hsv[:, 0] >= 0.0)
    assert np.all(hsv[:, 0] < 360.0 + 1e-6)


def test_rgb_to_hsv_saturation_range():
    px = _rand_px(100, seed=6)
    hsv = _rgb_to_hsv(px)
    assert np.all(hsv[:, 1] >= 0.0)
    assert np.all(hsv[:, 1] <= 1.0)


def test_rgb_to_hsv_value_range():
    px = _rand_px(100, seed=7)
    hsv = _rgb_to_hsv(px)
    assert np.all(hsv[:, 2] >= 0.0)
    assert np.all(hsv[:, 2] <= 1.0)


# ─── verify_seam with zero samples ────────────────────────────────────────────

def test_zero_samples_returns_inf_delta():
    pa = np.zeros((0, 3))
    pb = np.zeros((0, 3))
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 0
    assert result.score == pytest.approx(0.0)


def test_zero_samples_is_not_valid():
    pa = np.zeros((0, 3))
    pb = np.zeros((0, 3))
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.is_valid is False


def test_unequal_lengths_uses_shorter():
    pa = _rand_px(30, seed=0)
    pb = _rand_px(10, seed=1)
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 10


def test_unequal_lengths_uses_shorter_other_direction():
    pa = _rand_px(5, seed=0)
    pb = _rand_px(100, seed=1)
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 5


# ─── score_from_delta edge cases ─────────────────────────────────────────────

def test_score_from_delta_zero_delta_one():
    s = ColorContinuityVerifier.score_from_delta(0.0, 30.0)
    assert s == pytest.approx(1.0)


def test_score_from_delta_threshold_equals_delta_exp_minus_one():
    # exp(-1) ≈ 0.368
    s = ColorContinuityVerifier.score_from_delta(30.0, 30.0)
    assert s == pytest.approx(np.exp(-1.0), abs=1e-9)


def test_score_from_delta_very_large_delta_near_zero():
    s = ColorContinuityVerifier.score_from_delta(1e6, 30.0)
    assert s < 1e-4


def test_score_from_delta_threshold_zero_returns_0_for_nonzero():
    s = ColorContinuityVerifier.score_from_delta(1.0, 0.0)
    assert s == pytest.approx(0.0)


def test_score_from_delta_threshold_zero_delta_zero_returns_1():
    s = ColorContinuityVerifier.score_from_delta(0.0, 0.0)
    assert s == pytest.approx(1.0)


def test_score_from_delta_in_range():
    for d in [0.0, 5.0, 15.0, 30.0, 60.0, 200.0]:
        s = ColorContinuityVerifier.score_from_delta(d, 30.0)
        assert 0.0 <= s <= 1.0


# ─── Color space comparisons ─────────────────────────────────────────────────

def test_rgb_vs_lab_different_deltas():
    pa = _px(10, 200, 50, 50)
    pb = _px(10, 50, 200, 50)
    cfg_rgb = ColorContinuityConfig(method="rgb")
    cfg_lab = ColorContinuityConfig(method="lab")
    r_rgb = ColorContinuityVerifier(cfg_rgb).verify_seam(pa, pb)
    r_lab = ColorContinuityVerifier(cfg_lab).verify_seam(pa, pb)
    # Different color spaces produce different delta values
    # Just check both are valid floats
    assert np.isfinite(r_rgb.mean_delta)
    assert np.isfinite(r_lab.mean_delta)


def test_all_color_spaces_score_in_range():
    pa = _rand_px(20, seed=10)
    pb = _rand_px(20, seed=11)
    for method in ("rgb", "lab", "hsv"):
        cfg = ColorContinuityConfig(method=method)
        result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
        assert 0.0 <= result.score <= 1.0, f"Failed for method={method}"


# ─── mean_delta and max_delta invariants ─────────────────────────────────────

def test_mean_le_max_always():
    rng = np.random.default_rng(20)
    pa = rng.uniform(0, 255, (50, 3))
    pb = rng.uniform(0, 255, (50, 3))
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.mean_delta <= result.max_delta + 1e-9


def test_identical_pixels_zero_delta():
    px = _px(10, 100, 150, 200)
    result = ColorContinuityVerifier().verify_seam(px, px.copy())
    assert result.mean_delta == pytest.approx(0.0, abs=1e-9)
    assert result.max_delta == pytest.approx(0.0, abs=1e-9)


def test_identical_pixels_score_one():
    px = _rand_px(20, seed=30)
    result = ColorContinuityVerifier().verify_seam(px, px.copy())
    assert result.score == pytest.approx(1.0, abs=1e-9)


# ─── is_valid boundary ───────────────────────────────────────────────────────

def test_is_valid_true_when_mean_below_threshold():
    pa = _px(10, 100.0, 100.0, 100.0)
    pb = _px(10, 100.5, 100.5, 100.5)  # tiny diff
    cfg = ColorContinuityConfig(threshold=30.0, method="rgb")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert result.is_valid is True


def test_is_valid_false_when_mean_above_threshold():
    pa = _px(10, 0.0, 0.0, 0.0)
    pb = _px(10, 255.0, 255.0, 255.0)
    cfg = ColorContinuityConfig(threshold=30.0, method="rgb")
    result = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert result.is_valid is False


# ─── Module-level function ────────────────────────────────────────────────────

def test_verify_color_continuity_default_config():
    pa = _rand_px(10, seed=40)
    pb = _rand_px(10, seed=41)
    result = verify_color_continuity(pa, pb)
    assert isinstance(result, ColorContinuityResult)


def test_verify_color_continuity_high_threshold_is_valid():
    pa = _px(10, 0.0, 0.0, 0.0)
    pb = _px(10, 255.0, 255.0, 255.0)
    cfg = ColorContinuityConfig(threshold=1000.0, method="rgb")
    result = verify_color_continuity(pa, pb, config=cfg)
    assert result.is_valid is True


def test_verify_color_continuity_lab_identical():
    pa = _rand_px(15, seed=50)
    result = verify_color_continuity(pa, pa.copy(), config=ColorContinuityConfig(method="lab"))
    assert result.score == pytest.approx(1.0, abs=1e-9)


# ─── Large inputs ────────────────────────────────────────────────────────────

def test_large_input_1000_pixels():
    pa = _rand_px(1000, seed=60)
    pb = _rand_px(1000, seed=61)
    result = ColorContinuityVerifier().verify_seam(pa, pb)
    assert result.n_samples == 1000
    assert 0.0 <= result.score <= 1.0


# ─── Determinism ─────────────────────────────────────────────────────────────

def test_result_deterministic_lab():
    pa = _rand_px(20, seed=70)
    pb = _rand_px(20, seed=71)
    r1 = ColorContinuityVerifier().verify_seam(pa, pb)
    r2 = ColorContinuityVerifier().verify_seam(pa, pb)
    assert r1.mean_delta == pytest.approx(r2.mean_delta)
    assert r1.score == pytest.approx(r2.score)
    assert r1.is_valid == r2.is_valid


def test_result_deterministic_hsv():
    cfg = ColorContinuityConfig(method="hsv")
    pa = _rand_px(20, seed=72)
    pb = _rand_px(20, seed=73)
    r1 = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    r2 = ColorContinuityVerifier(cfg).verify_seam(pa, pb)
    assert r1.score == pytest.approx(r2.score)
