"""Extra tests for puzzle_reconstruction/verification/color_continuity_verifier.py"""
from __future__ import annotations

import math
import pytest
import numpy as np

from puzzle_reconstruction.verification.color_continuity_verifier import (
    ColorContinuityConfig,
    ColorContinuityResult,
    ColorContinuityVerifier,
    _rgb_to_lab,
    _rgb_to_hsv,
    verify_color_continuity,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _uniform(color, n=10):
    return np.full((n, 3), color, dtype=np.uint8)


def _rng_pixels(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 3)).astype(np.uint8)


# ─── ColorContinuityConfig – edge cases ──────────────────────────────────────

class TestColorContinuityConfigEdgeCases:

    def test_zero_seam_width_allowed(self):
        cfg = ColorContinuityConfig(seam_width=0)
        assert cfg.seam_width == 0

    def test_zero_threshold_allowed(self):
        cfg = ColorContinuityConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_large_threshold_allowed(self):
        cfg = ColorContinuityConfig(threshold=1e6)
        assert cfg.threshold == 1e6

    def test_negative_threshold_stored(self):
        cfg = ColorContinuityConfig(threshold=-5.0)
        assert cfg.threshold == -5.0

    def test_unknown_method_stored(self):
        # The config is just a dataclass; it stores whatever is given
        cfg = ColorContinuityConfig(method="xyz_unknown")
        assert cfg.method == "xyz_unknown"

    def test_weight_spatial_false_default_override(self):
        cfg = ColorContinuityConfig(weight_spatial=False)
        assert cfg.weight_spatial is False


# ─── _rgb_to_lab – additional channels and boundary values ───────────────────

class TestRgbToLabExtra:

    def test_mid_gray_is_neutral(self):
        # 128,128,128 → Lab should have a≈0, b≈0
        px = np.array([[128, 128, 128]], dtype=np.uint8)
        lab = _rgb_to_lab(px)
        assert abs(lab[0, 1]) < 5.0
        assert abs(lab[0, 2]) < 5.0

    def test_pure_green_positive_negative(self):
        px = np.array([[0, 255, 0]], dtype=np.uint8)
        lab = _rgb_to_lab(px)
        # green has negative a*
        assert lab[0, 1] < 0

    def test_pure_yellow(self):
        px = np.array([[255, 255, 0]], dtype=np.uint8)
        lab = _rgb_to_lab(px)
        # yellow has positive b*
        assert lab[0, 2] > 0

    def test_batch_consistency(self):
        # Process pixels one-by-one vs. batch – should match
        rng = np.random.default_rng(7)
        px = rng.integers(0, 256, (5, 3)).astype(np.uint8)
        batch = _rgb_to_lab(px)
        for i in range(5):
            single = _rgb_to_lab(px[i:i+1])
            np.testing.assert_allclose(single[0], batch[i], atol=1e-10)

    def test_float_input_same_as_uint8(self):
        px_u8 = np.array([[100, 150, 200]], dtype=np.uint8)
        px_f  = px_u8.astype(np.float64)
        lab_u8 = _rgb_to_lab(px_u8)
        lab_f  = _rgb_to_lab(px_f)
        np.testing.assert_allclose(lab_u8, lab_f, atol=1e-10)

    def test_single_pixel_output_shape(self):
        px = np.array([[255, 0, 0]], dtype=np.uint8)
        assert _rgb_to_lab(px).shape == (1, 3)

    def test_large_batch_no_crash(self):
        px = _rng_pixels(1000, seed=99)
        lab = _rgb_to_lab(px)
        assert lab.shape == (1000, 3)
        assert np.all(np.isfinite(lab))


# ─── _rgb_to_hsv – additional edge cases ─────────────────────────────────────

class TestRgbToHsvExtra:

    def test_pure_red_hue_near_zero(self):
        px = np.array([[255, 0, 0]], dtype=np.uint8)
        hsv = _rgb_to_hsv(px)
        # Red hue is 0°
        assert abs(hsv[0, 0]) < 1e-9 or abs(hsv[0, 0] - 360.0) < 1e-9

    def test_pure_green_hue_near_120(self):
        px = np.array([[0, 255, 0]], dtype=np.uint8)
        hsv = _rgb_to_hsv(px)
        assert abs(hsv[0, 0] - 120.0) < 1e-9

    def test_pure_blue_hue_near_240(self):
        px = np.array([[0, 0, 255]], dtype=np.uint8)
        hsv = _rgb_to_hsv(px)
        assert abs(hsv[0, 0] - 240.0) < 1e-9

    def test_gray_saturation_zero(self):
        for g in [0, 64, 128, 192, 255]:
            px = np.array([[g, g, g]], dtype=np.uint8)
            hsv = _rgb_to_hsv(px)
            assert abs(hsv[0, 1]) < 1e-9

    def test_fully_saturated_value_equals_max_channel(self):
        px = np.array([[200, 50, 50]], dtype=np.uint8)
        hsv = _rgb_to_hsv(px)
        assert abs(hsv[0, 2] - 200 / 255.0) < 1e-6

    def test_batch_and_single_match(self):
        rng = np.random.default_rng(3)
        px = rng.integers(0, 256, (8, 3)).astype(np.uint8)
        batch = _rgb_to_hsv(px)
        for i in range(8):
            single = _rgb_to_hsv(px[i:i+1])
            np.testing.assert_allclose(single[0], batch[i], atol=1e-10)

    def test_large_batch_finite(self):
        px = _rng_pixels(500, seed=11)
        hsv = _rgb_to_hsv(px)
        assert np.all(np.isfinite(hsv))


# ─── verify_seam – mismatched lengths ────────────────────────────────────────

class TestVerifySeamEdgeCases:

    def test_one_pixel_each(self):
        v = ColorContinuityVerifier()
        pa = np.array([[100, 100, 100]], dtype=np.uint8)
        pb = np.array([[100, 100, 100]], dtype=np.uint8)
        r = v.verify_seam(pa, pb)
        assert r.n_samples == 1
        assert r.mean_delta < 1.0

    def test_one_empty_one_nonempty(self):
        v = ColorContinuityVerifier()
        pa = np.zeros((0, 3), dtype=np.uint8)
        pb = _uniform([50, 50, 50], n=5)
        r = v.verify_seam(pa, pb)
        assert r.n_samples == 0
        assert math.isinf(r.mean_delta)
        assert r.is_valid is False

    def test_a_longer_than_b_uses_b_length(self):
        v = ColorContinuityVerifier()
        pa = _uniform([100, 100, 100], n=20)
        pb = _uniform([100, 100, 100], n=5)
        r = v.verify_seam(pa, pb)
        assert r.n_samples == 5

    def test_b_longer_than_a_uses_a_length(self):
        v = ColorContinuityVerifier()
        pa = _uniform([100, 100, 100], n=3)
        pb = _uniform([100, 100, 100], n=15)
        r = v.verify_seam(pa, pb)
        assert r.n_samples == 3

    def test_very_large_input_no_crash(self):
        v = ColorContinuityVerifier()
        pa = _rng_pixels(2000, seed=5)
        pb = _rng_pixels(2000, seed=6)
        r = v.verify_seam(pa, pb)
        assert r.n_samples == 2000
        assert np.isfinite(r.mean_delta)

    def test_score_zero_when_delta_is_inf(self):
        v = ColorContinuityVerifier()
        empty = np.zeros((0, 3), dtype=np.uint8)
        r = v.verify_seam(empty, empty)
        assert r.score == 0.0

    def test_identical_result_is_valid_with_default_threshold(self):
        v = ColorContinuityVerifier()
        pa = _uniform([200, 100, 50], n=8)
        r = v.verify_seam(pa, pa.copy())
        assert r.is_valid is True

    def test_black_vs_white_rgb_method_high_delta(self):
        cfg = ColorContinuityConfig(method="rgb", threshold=300.0)
        v = ColorContinuityVerifier(cfg)
        pa = _uniform([0, 0, 0], n=10)
        pb = _uniform([255, 255, 255], n=10)
        r = v.verify_seam(pa, pb)
        # Euclidean distance in RGB for black vs white ≈ sqrt(3)*255 ≈ 441
        assert r.mean_delta > 400.0

    def test_hsv_method_identical_score_near_one(self):
        cfg = ColorContinuityConfig(method="hsv", threshold=10.0)
        v = ColorContinuityVerifier(cfg)
        pa = _rng_pixels(15, seed=20)
        r = v.verify_seam(pa, pa.copy())
        assert r.score > 0.99

    def test_lab_method_similar_colors_valid(self):
        cfg = ColorContinuityConfig(method="lab", threshold=30.0)
        v = ColorContinuityVerifier(cfg)
        pa = _uniform([128, 128, 128], n=10)
        pb = _uniform([130, 130, 130], n=10)
        r = v.verify_seam(pa, pb)
        assert r.is_valid is True

    def test_unknown_method_falls_through_to_rgb(self):
        # _convert_color_space returns raw pixels for unknown method
        cfg = ColorContinuityConfig(method="unknown_space", threshold=500.0)
        v = ColorContinuityVerifier(cfg)
        pa = _uniform([100, 100, 100], n=5)
        pb = _uniform([100, 100, 100], n=5)
        r = v.verify_seam(pa, pb)
        assert r.mean_delta < 1.0

    def test_float64_input_accepted(self):
        v = ColorContinuityVerifier()
        pa = _uniform([50, 100, 150], n=8).astype(np.float64)
        pb = _uniform([50, 100, 150], n=8).astype(np.float64)
        r = v.verify_seam(pa, pb)
        assert r.mean_delta < 1.0


# ─── _color_delta – additional edge cases ────────────────────────────────────

class TestColorDeltaExtra:

    def test_b_empty_returns_inf(self):
        v = ColorContinuityVerifier()
        a = np.ones((5, 3))
        b = np.zeros((0, 3))
        assert math.isinf(v._color_delta(a, b))

    def test_mismatched_uses_min_length(self):
        v = ColorContinuityVerifier()
        a = np.zeros((10, 3))
        b = np.zeros((4, 3))
        delta = v._color_delta(a, b)
        assert delta == pytest.approx(0.0)

    def test_known_distance(self):
        v = ColorContinuityVerifier()
        # Two 1-pixel arrays with known Euclidean distance
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        assert v._color_delta(a, b) == pytest.approx(1.0)

    def test_symmetric(self):
        v = ColorContinuityVerifier()
        a = _rng_pixels(10, seed=1).astype(np.float64)
        b = _rng_pixels(10, seed=2).astype(np.float64)
        assert abs(v._color_delta(a, b) - v._color_delta(b, a)) < 1e-10


# ─── score_from_delta – additional boundary values ───────────────────────────

class TestScoreFromDeltaExtra:

    def test_monotone_decreasing(self):
        threshold = 30.0
        scores = [
            ColorContinuityVerifier.score_from_delta(d, threshold)
            for d in [0.0, 5.0, 15.0, 30.0, 60.0, 150.0]
        ]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_large_threshold_gives_high_score_for_small_delta(self):
        score = ColorContinuityVerifier.score_from_delta(1.0, threshold=1000.0)
        assert score > 0.99

    def test_score_clipped_to_zero_not_negative(self):
        # exp never goes negative but let's confirm clipping is respected
        score = ColorContinuityVerifier.score_from_delta(1e9, threshold=0.001)
        assert score >= 0.0

    def test_negative_threshold_returns_zero_for_positive_delta(self):
        score = ColorContinuityVerifier.score_from_delta(10.0, threshold=-5.0)
        assert score == 0.0

    def test_threshold_equals_double_delta(self):
        # delta=T/2 → score = exp(-0.5)
        delta = 15.0
        threshold = 30.0
        score = ColorContinuityVerifier.score_from_delta(delta, threshold)
        assert score == pytest.approx(math.exp(-0.5), rel=1e-6)


# ─── verify_color_continuity – module-level convenience ──────────────────────

class TestVerifyColorContinuityExtra:

    def test_passes_custom_config(self):
        cfg = ColorContinuityConfig(method="hsv", threshold=50.0)
        pa = _rng_pixels(10, seed=3)
        pb = _rng_pixels(10, seed=4)
        result = verify_color_continuity(pa, pb, config=cfg)
        assert isinstance(result, ColorContinuityResult)

    def test_high_threshold_makes_valid(self):
        cfg = ColorContinuityConfig(threshold=1e6, method="rgb")
        pa = _uniform([0, 0, 0], n=5)
        pb = _uniform([255, 255, 255], n=5)
        result = verify_color_continuity(pa, pb, config=cfg)
        assert result.is_valid is True

    def test_low_threshold_makes_invalid(self):
        cfg = ColorContinuityConfig(threshold=0.001, method="rgb")
        pa = _uniform([100, 100, 100], n=5)
        pb = _uniform([101, 100, 100], n=5)
        result = verify_color_continuity(pa, pb, config=cfg)
        assert result.is_valid is False

    def test_returns_correct_type(self):
        pa = _rng_pixels(5, seed=7)
        pb = _rng_pixels(5, seed=8)
        result = verify_color_continuity(pa, pb)
        assert type(result).__name__ == "ColorContinuityResult"

    def test_n_samples_equals_min_of_inputs(self):
        pa = _uniform([100, 100, 100], n=12)
        pb = _uniform([100, 100, 100], n=7)
        result = verify_color_continuity(pa, pb)
        assert result.n_samples == 7
