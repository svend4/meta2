"""Tests for puzzle_reconstruction/verification/color_continuity_verifier.py."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.verification.color_continuity_verifier import (
    ColorContinuityConfig,
    ColorContinuityResult,
    _rgb_to_lab,
    _rgb_to_hsv,
    ColorContinuityVerifier,
    verify_color_continuity,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _uniform_pixels(color, n=10):
    """Create (n, 3) uint8 array with given RGB color."""
    return np.full((n, 3), color, dtype=np.uint8)


def _random_pixels(n=10, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n, 3), dtype=np.uint8)


# ─── ColorContinuityConfig ────────────────────────────────────────────────────

class TestColorContinuityConfig:
    def test_default_values(self):
        cfg = ColorContinuityConfig()
        assert cfg.seam_width == 3
        assert cfg.method == "lab"
        assert cfg.threshold == 30.0
        assert cfg.weight_spatial is True

    def test_custom_values(self):
        cfg = ColorContinuityConfig(seam_width=5, method="rgb", threshold=50.0, weight_spatial=False)
        assert cfg.seam_width == 5
        assert cfg.method == "rgb"
        assert cfg.threshold == 50.0
        assert cfg.weight_spatial is False


# ─── ColorContinuityResult ────────────────────────────────────────────────────

class TestColorContinuityResult:
    def test_fields(self):
        result = ColorContinuityResult(
            mean_delta=5.0, max_delta=10.0, score=0.85, is_valid=True, n_samples=20
        )
        assert result.mean_delta == 5.0
        assert result.max_delta == 10.0
        assert abs(result.score - 0.85) < 1e-9
        assert result.is_valid is True
        assert result.n_samples == 20


# ─── _rgb_to_lab ──────────────────────────────────────────────────────────────

class TestRgbToLab:
    def test_output_shape(self):
        pixels = _random_pixels(20)
        lab = _rgb_to_lab(pixels)
        assert lab.shape == (20, 3)

    def test_output_dtype_float64(self):
        pixels = _random_pixels(5)
        lab = _rgb_to_lab(pixels)
        assert lab.dtype == np.float64

    def test_black_pixel(self):
        pixels = np.zeros((1, 3), dtype=np.uint8)
        lab = _rgb_to_lab(pixels)
        # Black should have L near 0
        assert abs(lab[0, 0]) < 1.0

    def test_white_pixel(self):
        pixels = np.full((1, 3), 255, dtype=np.uint8)
        lab = _rgb_to_lab(pixels)
        # White should have L near 100
        assert lab[0, 0] > 95.0

    def test_l_channel_in_range(self):
        pixels = _random_pixels(100, seed=42)
        lab = _rgb_to_lab(pixels)
        assert (lab[:, 0] >= 0).all()
        assert (lab[:, 0] <= 100.1).all()

    def test_pure_red(self):
        pixels = np.array([[255, 0, 0]], dtype=np.uint8)
        lab = _rgb_to_lab(pixels)
        # a* should be positive for red
        assert lab[0, 1] > 0

    def test_pure_blue(self):
        pixels = np.array([[0, 0, 255]], dtype=np.uint8)
        lab = _rgb_to_lab(pixels)
        # b* should be negative for blue
        assert lab[0, 2] < 0

    def test_identical_pixels_same_lab(self):
        color = np.array([[100, 150, 200]], dtype=np.uint8)
        lab1 = _rgb_to_lab(color)
        lab2 = _rgb_to_lab(color)
        assert np.allclose(lab1, lab2)


# ─── _rgb_to_hsv ──────────────────────────────────────────────────────────────

class TestRgbToHsv:
    def test_output_shape(self):
        pixels = _random_pixels(15)
        hsv = _rgb_to_hsv(pixels)
        assert hsv.shape == (15, 3)

    def test_output_dtype_float64(self):
        pixels = _random_pixels(5)
        hsv = _rgb_to_hsv(pixels)
        assert hsv.dtype == np.float64

    def test_black_pixel_saturation_zero(self):
        pixels = np.zeros((1, 3), dtype=np.uint8)
        hsv = _rgb_to_hsv(pixels)
        assert abs(hsv[0, 1]) < 1e-9  # Saturation = 0
        assert abs(hsv[0, 2]) < 1e-9  # Value = 0

    def test_white_pixel_saturation_zero_value_one(self):
        pixels = np.full((1, 3), 255, dtype=np.uint8)
        hsv = _rgb_to_hsv(pixels)
        assert abs(hsv[0, 1]) < 1e-9  # Saturation = 0
        assert abs(hsv[0, 2] - 1.0) < 1e-9  # Value = 1

    def test_value_in_range(self):
        pixels = _random_pixels(50, seed=7)
        hsv = _rgb_to_hsv(pixels)
        assert (hsv[:, 2] >= 0.0).all()
        assert (hsv[:, 2] <= 1.0 + 1e-9).all()

    def test_saturation_in_range(self):
        pixels = _random_pixels(50, seed=8)
        hsv = _rgb_to_hsv(pixels)
        assert (hsv[:, 1] >= 0.0).all()
        assert (hsv[:, 1] <= 1.0 + 1e-9).all()

    def test_hue_in_range(self):
        pixels = _random_pixels(50, seed=9)
        hsv = _rgb_to_hsv(pixels)
        assert (hsv[:, 0] >= 0.0).all()
        assert (hsv[:, 0] < 360.0 + 1e-9).all()


# ─── ColorContinuityVerifier.verify_seam ─────────────────────────────────────

class TestVerifySeam:
    def test_identical_pixels_low_delta(self):
        verifier = ColorContinuityVerifier()
        pixels = _uniform_pixels([100, 150, 200], n=10)
        result = verifier.verify_seam(pixels, pixels.copy())
        assert result.mean_delta < 1.0
        assert result.is_valid is True

    def test_different_pixels_high_delta(self):
        verifier = ColorContinuityVerifier(ColorContinuityConfig(threshold=30.0))
        pa = _uniform_pixels([0, 0, 0], n=10)
        pb = _uniform_pixels([255, 255, 255], n=10)
        result = verifier.verify_seam(pa, pb)
        assert result.mean_delta > 0.0

    def test_empty_pixels_returns_invalid(self):
        verifier = ColorContinuityVerifier()
        empty = np.zeros((0, 3), dtype=np.uint8)
        result = verifier.verify_seam(empty, empty)
        assert result.is_valid is False
        assert result.n_samples == 0
        assert math.isinf(result.mean_delta)

    def test_n_samples_is_min_length(self):
        verifier = ColorContinuityVerifier()
        pa = _uniform_pixels([100, 100, 100], n=10)
        pb = _uniform_pixels([100, 100, 100], n=7)
        result = verifier.verify_seam(pa, pb)
        assert result.n_samples == 7

    def test_score_in_0_1_range(self):
        verifier = ColorContinuityVerifier()
        pa = _random_pixels(20, seed=0)
        pb = _random_pixels(20, seed=1)
        result = verifier.verify_seam(pa, pb)
        assert 0.0 <= result.score <= 1.0

    def test_max_delta_ge_mean_delta(self):
        verifier = ColorContinuityVerifier()
        pa = _random_pixels(20, seed=2)
        pb = _random_pixels(20, seed=3)
        result = verifier.verify_seam(pa, pb)
        assert result.max_delta >= result.mean_delta

    def test_is_valid_when_mean_delta_below_threshold(self):
        cfg = ColorContinuityConfig(threshold=200.0, method="rgb")
        verifier = ColorContinuityVerifier(cfg)
        pa = _uniform_pixels([100, 100, 100], n=5)
        pb = _uniform_pixels([105, 105, 105], n=5)
        result = verifier.verify_seam(pa, pb)
        assert result.is_valid is True

    def test_rgb_method(self):
        cfg = ColorContinuityConfig(method="rgb")
        verifier = ColorContinuityVerifier(cfg)
        pa = _uniform_pixels([100, 100, 100], n=5)
        pb = _uniform_pixels([200, 200, 200], n=5)
        result = verifier.verify_seam(pa, pb)
        assert result.n_samples == 5

    def test_hsv_method(self):
        cfg = ColorContinuityConfig(method="hsv")
        verifier = ColorContinuityVerifier(cfg)
        pa = _random_pixels(10, seed=4)
        pb = _random_pixels(10, seed=5)
        result = verifier.verify_seam(pa, pb)
        assert isinstance(result, ColorContinuityResult)


# ─── ColorContinuityVerifier._convert_color_space ────────────────────────────

class TestConvertColorSpace:
    def test_lab_conversion(self):
        verifier = ColorContinuityVerifier()
        pixels = _random_pixels(10).astype(np.float64)
        out = verifier._convert_color_space(pixels, "lab")
        assert out.shape == (10, 3)

    def test_hsv_conversion(self):
        verifier = ColorContinuityVerifier()
        pixels = _random_pixels(10).astype(np.float64)
        out = verifier._convert_color_space(pixels, "hsv")
        assert out.shape == (10, 3)

    def test_rgb_conversion_unchanged(self):
        verifier = ColorContinuityVerifier()
        pixels = _random_pixels(10).astype(np.float64)
        out = verifier._convert_color_space(pixels, "rgb")
        assert np.allclose(out, pixels)

    def test_case_insensitive(self):
        verifier = ColorContinuityVerifier()
        pixels = _random_pixels(5).astype(np.float64)
        out1 = verifier._convert_color_space(pixels, "LAB")
        out2 = verifier._convert_color_space(pixels, "lab")
        assert np.allclose(out1, out2)


# ─── ColorContinuityVerifier._color_delta ────────────────────────────────────

class TestColorDelta:
    def test_identical_arrays_zero_delta(self):
        verifier = ColorContinuityVerifier()
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert abs(verifier._color_delta(a, a)) < 1e-9

    def test_empty_array_returns_inf(self):
        verifier = ColorContinuityVerifier()
        empty = np.zeros((0, 3))
        a = np.zeros((5, 3))
        assert math.isinf(verifier._color_delta(empty, a))

    def test_delta_is_nonnegative(self):
        verifier = ColorContinuityVerifier()
        a = _random_pixels(10, seed=0).astype(np.float64)
        b = _random_pixels(10, seed=1).astype(np.float64)
        assert verifier._color_delta(a, b) >= 0.0


# ─── score_from_delta ─────────────────────────────────────────────────────────

class TestScoreFromDelta:
    def test_zero_delta_gives_score_one(self):
        score = ColorContinuityVerifier.score_from_delta(0.0, threshold=30.0)
        assert abs(score - 1.0) < 1e-9

    def test_delta_equals_threshold_gives_score_1_over_e(self):
        score = ColorContinuityVerifier.score_from_delta(30.0, threshold=30.0)
        assert abs(score - math.exp(-1.0)) < 1e-6

    def test_high_delta_gives_low_score(self):
        score = ColorContinuityVerifier.score_from_delta(1000.0, threshold=30.0)
        assert score < 0.01

    def test_score_in_0_1_range(self):
        for delta in [0.0, 5.0, 30.0, 100.0, 1000.0]:
            score = ColorContinuityVerifier.score_from_delta(delta, threshold=30.0)
            assert 0.0 <= score <= 1.0

    def test_zero_threshold(self):
        score = ColorContinuityVerifier.score_from_delta(10.0, threshold=0.0)
        assert score == 0.0

    def test_zero_threshold_zero_delta(self):
        score = ColorContinuityVerifier.score_from_delta(0.0, threshold=0.0)
        assert score == 1.0


# ─── verify_color_continuity (module-level) ───────────────────────────────────

class TestVerifyColorContinuity:
    def test_returns_result(self):
        pa = _uniform_pixels([100, 100, 100], n=10)
        pb = _uniform_pixels([105, 105, 105], n=10)
        result = verify_color_continuity(pa, pb)
        assert isinstance(result, ColorContinuityResult)

    def test_uses_default_config_when_none(self):
        pa = _uniform_pixels([200, 200, 200], n=5)
        pb = _uniform_pixels([200, 200, 200], n=5)
        result = verify_color_continuity(pa, pb)
        assert result.is_valid is True

    def test_custom_config_applied(self):
        cfg = ColorContinuityConfig(threshold=1.0, method="rgb")
        pa = _uniform_pixels([0, 0, 0], n=5)
        pb = _uniform_pixels([255, 0, 0], n=5)
        result = verify_color_continuity(pa, pb, config=cfg)
        assert result.is_valid is False
