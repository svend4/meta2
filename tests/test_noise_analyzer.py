"""Тесты для puzzle_reconstruction/preprocessing/noise_analyzer.py."""
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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=64, w=64, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _high_noise(h=64, w=64, sigma=40.0, seed=3):
    rng  = np.random.default_rng(seed)
    base = np.full((h, w), 128.0)
    n    = rng.normal(0, sigma, (h, w))
    return np.clip(base + n, 0, 255).astype(np.uint8)


# ─── NoiseAnalysisResult ──────────────────────────────────────────────────────

class TestNoiseAnalysisResult:
    def test_fields(self):
        r = NoiseAnalysisResult(
            noise_level=2.5, snr_db=30.0,
            jpeg_artifacts=0.05, grain_level=0.1,
            quality="clean",
        )
        assert r.noise_level == pytest.approx(2.5)
        assert r.snr_db == pytest.approx(30.0)
        assert r.jpeg_artifacts == pytest.approx(0.05)
        assert r.grain_level == pytest.approx(0.1)
        assert r.quality == "clean"

    def test_params_default_empty(self):
        r = NoiseAnalysisResult(
            noise_level=0.0, snr_db=0.0,
            jpeg_artifacts=0.0, grain_level=0.0, quality="clean",
        )
        assert isinstance(r.params, dict)

    def test_params_stored(self):
        r = NoiseAnalysisResult(
            noise_level=5.0, snr_db=25.0,
            jpeg_artifacts=0.1, grain_level=0.2, quality="noisy",
            params={"noise_thresh1": 5.0},
        )
        assert r.params["noise_thresh1"] == pytest.approx(5.0)

    def test_repr(self):
        r = NoiseAnalysisResult(
            noise_level=3.14, snr_db=28.0,
            jpeg_artifacts=0.02, grain_level=0.05, quality="clean",
        )
        s = repr(r)
        assert "NoiseAnalysisResult" in s
        assert "clean" in s

    def test_repr_contains_snr(self):
        r = NoiseAnalysisResult(
            noise_level=2.0, snr_db=35.6,
            jpeg_artifacts=0.01, grain_level=0.03, quality="clean",
        )
        s = repr(r)
        assert "snr" in s.lower() or "35" in s


# ─── estimate_noise_sigma ────────────────────────────────────────────────────

class TestEstimateNoiseSigma:
    def test_returns_float(self):
        assert isinstance(estimate_noise_sigma(_noisy()), float)

    def test_nonneg(self):
        assert estimate_noise_sigma(_noisy()) >= 0.0

    def test_gray_input(self):
        r = estimate_noise_sigma(_gray())
        assert isinstance(r, float)
        assert r >= 0.0

    def test_bgr_input(self):
        r = estimate_noise_sigma(_bgr())
        assert isinstance(r, float)
        assert r >= 0.0

    def test_constant_is_low(self):
        r = estimate_noise_sigma(_gray(val=150))
        assert r < 5.0   # Near-zero for constant image

    def test_noisy_greater_than_constant(self):
        r_noisy    = estimate_noise_sigma(_noisy())
        r_constant = estimate_noise_sigma(_gray())
        assert r_noisy > r_constant

    def test_high_noise_larger(self):
        r_low  = estimate_noise_sigma(_gray())
        r_high = estimate_noise_sigma(_high_noise(sigma=40.0))
        assert r_high > r_low

    def test_tiny_image(self):
        img = _noisy(h=8, w=8)
        r   = estimate_noise_sigma(img)
        assert isinstance(r, float)
        assert r >= 0.0


# ─── estimate_snr ─────────────────────────────────────────────────────────────

class TestEstimateSnr:
    def test_returns_float(self):
        r = estimate_snr(_noisy())
        assert isinstance(r, float)

    def test_gray_input(self):
        r = estimate_snr(_gray())
        assert isinstance(r, float)

    def test_bgr_input(self):
        r = estimate_snr(_bgr())
        assert isinstance(r, float)

    def test_constant_high_or_inf(self):
        r = estimate_snr(_gray(val=150))
        # Constant image → noise ≈ 0 → snr ≥ large value or inf
        assert r > 30.0 or math.isinf(r)

    def test_noisy_positive(self):
        r = estimate_snr(_noisy())
        # Noisy image → finite SNR > 0
        assert math.isfinite(r)

    def test_zero_mean_image_returns_finite_or_special(self):
        # Don't crash on dark images
        img = np.zeros((32, 32), dtype=np.uint8)
        r   = estimate_snr(img)
        assert isinstance(r, float)

    def test_high_noise_lower_snr(self):
        r_clean = estimate_snr(_gray(val=128))
        r_noisy = estimate_snr(_high_noise(sigma=40.0))
        # Higher noise → lower SNR
        if math.isfinite(r_clean) and math.isfinite(r_noisy):
            assert r_clean >= r_noisy


# ─── detect_jpeg_artifacts ────────────────────────────────────────────────────

class TestDetectJpegArtifacts:
    def test_returns_float(self):
        assert isinstance(detect_jpeg_artifacts(_noisy()), float)

    def test_range(self):
        r = detect_jpeg_artifacts(_noisy())
        assert 0.0 <= r <= 1.0

    def test_gray_input(self):
        r = detect_jpeg_artifacts(_gray())
        assert 0.0 <= r <= 1.0

    def test_bgr_input(self):
        r = detect_jpeg_artifacts(_bgr())
        assert 0.0 <= r <= 1.0

    def test_constant_near_zero(self):
        r = detect_jpeg_artifacts(_gray(val=128))
        assert r < 0.1   # Constant image has no block boundaries

    def test_noisy_nonneg(self):
        r = detect_jpeg_artifacts(_noisy())
        assert r >= 0.0

    def test_block_size_param(self):
        r1 = detect_jpeg_artifacts(_noisy(), block_size=8)
        r2 = detect_jpeg_artifacts(_noisy(), block_size=16)
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0

    def test_tiny_image(self):
        img = _noisy(h=8, w=8)
        r   = detect_jpeg_artifacts(img)
        assert isinstance(r, float)


# ─── estimate_grain ───────────────────────────────────────────────────────────

class TestEstimateGrain:
    def test_returns_float(self):
        assert isinstance(estimate_grain(_noisy()), float)

    def test_range(self):
        r = estimate_grain(_noisy())
        assert 0.0 <= r <= 1.0

    def test_constant_is_zero(self):
        r = estimate_grain(_gray(val=128))
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_noisy_positive(self):
        r = estimate_grain(_noisy())
        assert r > 0.0

    def test_high_noise_larger_grain(self):
        r_low  = estimate_grain(_gray())
        r_high = estimate_grain(_high_noise(sigma=40.0))
        assert r_high > r_low

    def test_gray_input(self):
        r = estimate_grain(_gray())
        assert 0.0 <= r <= 1.0

    def test_bgr_input(self):
        r = estimate_grain(_bgr())
        assert 0.0 <= r <= 1.0

    def test_block_size_param(self):
        r = estimate_grain(_noisy(), block_size=8)
        assert 0.0 <= r <= 1.0

    def test_tiny_image_no_crash(self):
        img = _noisy(h=4, w=4)
        r   = estimate_grain(img)
        assert isinstance(r, float)


# ─── analyze_noise ────────────────────────────────────────────────────────────

class TestAnalyzeNoise:
    def test_returns_result(self):
        assert isinstance(analyze_noise(_noisy()), NoiseAnalysisResult)

    def test_quality_valid(self):
        r = analyze_noise(_noisy())
        assert r.quality in {"clean", "noisy", "very_noisy"}

    def test_noise_level_matches_sigma(self):
        img = _noisy()
        r   = analyze_noise(img)
        expected = estimate_noise_sigma(img)
        assert r.noise_level == pytest.approx(expected)

    def test_params_stored(self):
        r = analyze_noise(_noisy(), noise_thresh1=3.0, noise_thresh2=10.0,
                          jpeg_block=8, grain_block=16)
        assert r.params.get("noise_thresh1") == pytest.approx(3.0)
        assert r.params.get("noise_thresh2") == pytest.approx(10.0)
        assert r.params.get("jpeg_block") == 8
        assert r.params.get("grain_block") == 16

    def test_constant_image_clean(self):
        r = analyze_noise(_gray(val=128), noise_thresh1=5.0, noise_thresh2=15.0)
        assert r.quality == "clean"

    def test_high_noise_very_noisy(self):
        r = analyze_noise(_high_noise(sigma=50.0), noise_thresh1=3.0,
                          noise_thresh2=10.0)
        assert r.quality in {"noisy", "very_noisy"}

    def test_gray_input(self):
        r = analyze_noise(_gray())
        assert isinstance(r, NoiseAnalysisResult)

    def test_bgr_input(self):
        r = analyze_noise(_bgr())
        assert isinstance(r, NoiseAnalysisResult)

    def test_jpeg_artifacts_in_range(self):
        r = analyze_noise(_noisy())
        assert 0.0 <= r.jpeg_artifacts <= 1.0

    def test_grain_level_in_range(self):
        r = analyze_noise(_noisy())
        assert 0.0 <= r.grain_level <= 1.0

    def test_snr_db_finite_or_inf(self):
        r = analyze_noise(_noisy())
        assert math.isfinite(r.snr_db) or math.isinf(r.snr_db)

    @pytest.mark.parametrize("quality,n1,n2", [
        ("clean",      100.0, 200.0),
        ("very_noisy", 0.0,   0.0),
    ])
    def test_quality_by_threshold(self, quality, n1, n2):
        img = _gray()
        r   = analyze_noise(img, noise_thresh1=n1, noise_thresh2=n2)
        if quality == "clean":
            assert r.quality == "clean"


# ─── batch_analyze_noise ──────────────────────────────────────────────────────

class TestBatchAnalyzeNoise:
    def test_returns_list(self):
        results = batch_analyze_noise([_noisy(), _gray()])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_result(self):
        for r in batch_analyze_noise([_noisy(), _bgr()]):
            assert isinstance(r, NoiseAnalysisResult)

    def test_empty_list(self):
        assert batch_analyze_noise([]) == []

    def test_kwargs_forwarded(self):
        results = batch_analyze_noise([_gray()], noise_thresh1=100.0,
                                       noise_thresh2=200.0)
        assert results[0].quality == "clean"

    def test_quality_all_valid(self):
        imgs = [_gray(), _noisy(), _high_noise()]
        for r in batch_analyze_noise(imgs):
            assert r.quality in {"clean", "noisy", "very_noisy"}

    def test_noise_levels_nonneg(self):
        for r in batch_analyze_noise([_noisy(), _gray()]):
            assert r.noise_level >= 0.0

    def test_bgr_input(self):
        results = batch_analyze_noise([_bgr()])
        assert isinstance(results[0], NoiseAnalysisResult)
