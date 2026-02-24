"""Extra tests for puzzle_reconstruction/preprocessing/noise_analyzer.py."""
from __future__ import annotations

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=50, w=50, sigma=30):
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (h, w), dtype=np.uint8)
    return img


# ─── NoiseAnalysisResult ──────────────────────────────────────────────────────

class TestNoiseAnalysisResultExtra:
    def test_fields(self):
        r = NoiseAnalysisResult(noise_level=5.0, snr_db=20.0,
                                jpeg_artifacts=0.1, grain_level=0.2,
                                quality="clean")
        assert r.quality == "clean"

    def test_repr(self):
        r = NoiseAnalysisResult(noise_level=5.0, snr_db=20.0,
                                jpeg_artifacts=0.1, grain_level=0.2,
                                quality="noisy")
        s = repr(r)
        assert "noisy" in s

    def test_default_params(self):
        r = NoiseAnalysisResult(noise_level=0, snr_db=0,
                                jpeg_artifacts=0, grain_level=0,
                                quality="clean")
        assert r.params == {}


# ─── estimate_noise_sigma ─────────────────────────────────────────────────────

class TestEstimateNoiseSigmaExtra:
    def test_nonnegative(self):
        assert estimate_noise_sigma(_gray()) >= 0

    def test_uniform_low(self):
        # Uniform image should have very low noise
        sigma = estimate_noise_sigma(_gray())
        assert sigma < 1.0

    def test_noisy_higher(self):
        sigma = estimate_noise_sigma(_noisy())
        assert sigma > 0

    def test_bgr_input(self):
        sigma = estimate_noise_sigma(_bgr())
        assert isinstance(sigma, float)


# ─── estimate_snr ─────────────────────────────────────────────────────────────

class TestEstimateSnrExtra:
    def test_uniform_is_inf(self):
        snr = estimate_snr(_gray())
        assert snr == float("inf")

    def test_noisy_finite(self):
        snr = estimate_snr(_noisy())
        assert np.isfinite(snr)

    def test_bgr_input(self):
        snr = estimate_snr(_bgr())
        assert isinstance(snr, float)


# ─── detect_jpeg_artifacts ────────────────────────────────────────────────────

class TestDetectJpegArtifactsExtra:
    def test_range(self):
        val = detect_jpeg_artifacts(_gray())
        assert 0.0 <= val <= 1.0

    def test_uniform_low(self):
        val = detect_jpeg_artifacts(_gray())
        assert val < 0.1

    def test_bgr_input(self):
        val = detect_jpeg_artifacts(_bgr())
        assert isinstance(val, float)


# ─── estimate_grain ───────────────────────────────────────────────────────────

class TestEstimateGrainExtra:
    def test_range(self):
        val = estimate_grain(_gray())
        assert 0.0 <= val <= 1.0

    def test_uniform_low(self):
        val = estimate_grain(_gray())
        assert val < 0.05

    def test_noisy_higher(self):
        val = estimate_grain(_noisy())
        assert val > 0

    def test_bgr_input(self):
        val = estimate_grain(_bgr())
        assert isinstance(val, float)


# ─── analyze_noise ────────────────────────────────────────────────────────────

class TestAnalyzeNoiseExtra:
    def test_returns_result(self):
        r = analyze_noise(_gray())
        assert isinstance(r, NoiseAnalysisResult)

    def test_clean_quality(self):
        r = analyze_noise(_gray())
        assert r.quality == "clean"

    def test_noisy_quality(self):
        r = analyze_noise(_noisy())
        # Noisy image should not be "clean"
        assert r.quality in ("noisy", "very_noisy")

    def test_params_populated(self):
        r = analyze_noise(_gray())
        assert "noise_thresh1" in r.params

    def test_custom_thresholds(self):
        r = analyze_noise(_gray(), noise_thresh1=0.0001, noise_thresh2=0.001)
        assert isinstance(r, NoiseAnalysisResult)


# ─── batch_analyze_noise ──────────────────────────────────────────────────────

class TestBatchAnalyzeNoiseExtra:
    def test_empty(self):
        assert batch_analyze_noise([]) == []

    def test_length(self):
        results = batch_analyze_noise([_gray(), _gray()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_analyze_noise([_gray()])
        assert isinstance(results[0], NoiseAnalysisResult)
