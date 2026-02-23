"""Extra tests for puzzle_reconstruction.preprocessing.noise_analyzer."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.noise_analyzer import (
    NoiseAnalysisResult,
    estimate_noise_sigma,
    estimate_snr,
    detect_jpeg_artifacts,
    estimate_grain,
    analyze_noise,
    batch_analyze_noise,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=7):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _high_noise(h=64, w=64, sigma=40.0, seed=3):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 128.0)
    n = rng.normal(0, sigma, (h, w))
    return np.clip(base + n, 0, 255).astype(np.uint8)


# ─── TestNoiseAnalysisResultExtra ────────────────────────────────────────────

class TestNoiseAnalysisResultExtra:
    def test_all_fields_stored(self):
        r = NoiseAnalysisResult(
            noise_level=5.0, snr_db=20.0,
            jpeg_artifacts=0.1, grain_level=0.2,
            quality="noisy",
        )
        assert r.noise_level == pytest.approx(5.0)
        assert r.snr_db == pytest.approx(20.0)
        assert r.jpeg_artifacts == pytest.approx(0.1)
        assert r.grain_level == pytest.approx(0.2)
        assert r.quality == "noisy"

    def test_params_default_dict(self):
        r = NoiseAnalysisResult(
            noise_level=0.0, snr_db=0.0,
            jpeg_artifacts=0.0, grain_level=0.0,
            quality="clean",
        )
        assert isinstance(r.params, dict)

    def test_params_custom(self):
        r = NoiseAnalysisResult(
            noise_level=1.0, snr_db=10.0,
            jpeg_artifacts=0.01, grain_level=0.02,
            quality="clean", params={"method": "median"},
        )
        assert r.params["method"] == "median"

    def test_repr_contains_quality(self):
        r = NoiseAnalysisResult(
            noise_level=2.0, snr_db=30.0,
            jpeg_artifacts=0.02, grain_level=0.05,
            quality="clean",
        )
        assert "clean" in repr(r)

    def test_quality_noisy(self):
        r = NoiseAnalysisResult(
            noise_level=8.0, snr_db=15.0,
            jpeg_artifacts=0.05, grain_level=0.3,
            quality="noisy",
        )
        assert r.quality == "noisy"

    def test_quality_very_noisy(self):
        r = NoiseAnalysisResult(
            noise_level=20.0, snr_db=5.0,
            jpeg_artifacts=0.2, grain_level=0.5,
            quality="very_noisy",
        )
        assert r.quality == "very_noisy"


# ─── TestEstimateNoiseSigmaExtra ────────────────────────────────────────────

class TestEstimateNoiseSigmaExtra:
    def test_returns_float(self):
        assert isinstance(estimate_noise_sigma(_noisy(seed=10)), float)

    def test_nonnegative(self):
        assert estimate_noise_sigma(_noisy(seed=11)) >= 0.0

    def test_constant_low(self):
        assert estimate_noise_sigma(_gray(val=100)) < 5.0

    def test_high_noise_larger_than_constant(self):
        r_const = estimate_noise_sigma(_gray())
        r_high = estimate_noise_sigma(_high_noise(sigma=50.0))
        assert r_high > r_const

    def test_bgr_input(self):
        r = estimate_noise_sigma(_bgr())
        assert isinstance(r, float)
        assert r >= 0.0

    def test_tiny_image(self):
        r = estimate_noise_sigma(_noisy(h=8, w=8, seed=12))
        assert isinstance(r, float)

    def test_noisy_higher_than_constant(self):
        r_noisy = estimate_noise_sigma(_noisy(seed=20))
        r_const = estimate_noise_sigma(_gray(val=200))
        assert r_noisy > r_const


# ─── TestEstimateSnrExtra ────────────────────────────────────────────────────

class TestEstimateSnrExtra:
    def test_returns_float(self):
        assert isinstance(estimate_snr(_noisy(seed=30)), float)

    def test_constant_high_or_inf(self):
        r = estimate_snr(_gray(val=128))
        assert r > 30.0 or math.isinf(r)

    def test_noisy_finite(self):
        r = estimate_snr(_noisy(seed=31))
        assert math.isfinite(r)

    def test_bgr_input(self):
        assert isinstance(estimate_snr(_bgr()), float)

    def test_zero_image_no_crash(self):
        r = estimate_snr(np.zeros((32, 32), dtype=np.uint8))
        assert isinstance(r, float)

    def test_high_noise_lower_snr(self):
        r_clean = estimate_snr(_gray(val=128))
        r_noisy = estimate_snr(_high_noise(sigma=50.0))
        if math.isfinite(r_clean) and math.isfinite(r_noisy):
            assert r_clean >= r_noisy

    def test_gray_input(self):
        assert isinstance(estimate_snr(_gray()), float)


# ─── TestDetectJpegArtifactsExtra ────────────────────────────────────────────

class TestDetectJpegArtifactsExtra:
    def test_returns_float(self):
        assert isinstance(detect_jpeg_artifacts(_noisy(seed=40)), float)

    def test_range(self):
        r = detect_jpeg_artifacts(_noisy(seed=41))
        assert 0.0 <= r <= 1.0

    def test_constant_near_zero(self):
        assert detect_jpeg_artifacts(_gray(val=100)) < 0.1

    def test_bgr_in_range(self):
        r = detect_jpeg_artifacts(_bgr())
        assert 0.0 <= r <= 1.0

    def test_block_size_8(self):
        r = detect_jpeg_artifacts(_noisy(seed=42), block_size=8)
        assert 0.0 <= r <= 1.0

    def test_block_size_16(self):
        r = detect_jpeg_artifacts(_noisy(seed=43), block_size=16)
        assert 0.0 <= r <= 1.0

    def test_tiny_image(self):
        r = detect_jpeg_artifacts(_noisy(h=8, w=8, seed=44))
        assert isinstance(r, float)

    def test_gray_nonneg(self):
        assert detect_jpeg_artifacts(_gray()) >= 0.0


# ─── TestEstimateGrainExtra ──────────────────────────────────────────────────

class TestEstimateGrainExtra:
    def test_returns_float(self):
        assert isinstance(estimate_grain(_noisy(seed=50)), float)

    def test_range(self):
        r = estimate_grain(_noisy(seed=51))
        assert 0.0 <= r <= 1.0

    def test_constant_zero(self):
        r = estimate_grain(_gray(val=128))
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_noisy_positive(self):
        assert estimate_grain(_noisy(seed=52)) > 0.0

    def test_high_noise_larger(self):
        r_low = estimate_grain(_gray())
        r_high = estimate_grain(_high_noise(sigma=50.0))
        assert r_high > r_low

    def test_bgr_in_range(self):
        r = estimate_grain(_bgr())
        assert 0.0 <= r <= 1.0

    def test_block_size_8(self):
        r = estimate_grain(_noisy(seed=53), block_size=8)
        assert 0.0 <= r <= 1.0

    def test_tiny_no_crash(self):
        r = estimate_grain(_noisy(h=4, w=4, seed=54))
        assert isinstance(r, float)


# ─── TestAnalyzeNoiseExtra ───────────────────────────────────────────────────

class TestAnalyzeNoiseExtra:
    def test_returns_result(self):
        assert isinstance(analyze_noise(_noisy(seed=60)), NoiseAnalysisResult)

    def test_quality_valid(self):
        r = analyze_noise(_noisy(seed=61))
        assert r.quality in {"clean", "noisy", "very_noisy"}

    def test_constant_clean(self):
        r = analyze_noise(_gray(val=128), noise_thresh1=5.0, noise_thresh2=15.0)
        assert r.quality == "clean"

    def test_high_noise_not_clean(self):
        r = analyze_noise(_high_noise(sigma=60.0),
                          noise_thresh1=3.0, noise_thresh2=10.0)
        assert r.quality in {"noisy", "very_noisy"}

    def test_params_forwarded(self):
        r = analyze_noise(_noisy(seed=62), noise_thresh1=2.0, noise_thresh2=8.0,
                          jpeg_block=8, grain_block=16)
        assert r.params.get("noise_thresh1") == pytest.approx(2.0)

    def test_noise_level_matches_sigma(self):
        img = _noisy(seed=63)
        r = analyze_noise(img)
        expected = estimate_noise_sigma(img)
        assert r.noise_level == pytest.approx(expected)

    def test_jpeg_in_range(self):
        r = analyze_noise(_noisy(seed=64))
        assert 0.0 <= r.jpeg_artifacts <= 1.0

    def test_grain_in_range(self):
        r = analyze_noise(_noisy(seed=65))
        assert 0.0 <= r.grain_level <= 1.0

    def test_snr_finite_or_inf(self):
        r = analyze_noise(_noisy(seed=66))
        assert math.isfinite(r.snr_db) or math.isinf(r.snr_db)

    def test_bgr_input(self):
        assert isinstance(analyze_noise(_bgr()), NoiseAnalysisResult)

    def test_gray_input(self):
        assert isinstance(analyze_noise(_gray()), NoiseAnalysisResult)


# ─── TestBatchAnalyzeNoiseExtra ──────────────────────────────────────────────

class TestBatchAnalyzeNoiseExtra:
    def test_returns_list(self):
        results = batch_analyze_noise([_noisy(seed=70), _gray()])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_result(self):
        for r in batch_analyze_noise([_noisy(seed=71), _bgr()]):
            assert isinstance(r, NoiseAnalysisResult)

    def test_empty_list(self):
        assert batch_analyze_noise([]) == []

    def test_kwargs_forwarded(self):
        results = batch_analyze_noise([_gray()],
                                      noise_thresh1=0.0, noise_thresh2=100.0)
        assert results[0].quality in {"clean", "noisy", "very_noisy"}

    def test_quality_all_valid(self):
        imgs = [_gray(), _noisy(seed=72), _high_noise()]
        for r in batch_analyze_noise(imgs):
            assert r.quality in {"clean", "noisy", "very_noisy"}

    def test_noise_levels_nonneg(self):
        for r in batch_analyze_noise([_noisy(seed=73), _gray()]):
            assert r.noise_level >= 0.0

    def test_single_bgr(self):
        results = batch_analyze_noise([_bgr()])
        assert isinstance(results[0], NoiseAnalysisResult)

    def test_three_images(self):
        imgs = [_gray(), _noisy(seed=74), _bgr()]
        results = batch_analyze_noise(imgs)
        assert len(results) == 3
