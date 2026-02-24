"""Extra tests for puzzle_reconstruction/matching/spectral_matcher.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.spectral_matcher import (
    SpectralMatchResult,
    magnitude_spectrum,
    log_magnitude,
    spectrum_correlation,
    phase_correlation,
    match_spectra,
    batch_spectral_match,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=32, w=32):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _rand_gray(h=32, w=32):
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


# ─── SpectralMatchResult ────────────────────────────────────────────────────

class TestSpectralMatchResultExtra:
    def test_fields_stored(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.8,
                                phase_shift=(1.0, 2.0))
        assert r.idx1 == 0 and r.idx2 == 1
        assert r.score == pytest.approx(0.8)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=1.5)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=-0.1)

    def test_default_phase_shift(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5)
        assert r.phase_shift == (0.0, 0.0)

    def test_params_default_empty(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5)
        assert r.params == {}


# ─── magnitude_spectrum ─────────────────────────────────────────────────────

class TestMagnitudeSpectrumExtra:
    def test_gray_input(self):
        img = _gray(32, 32)
        s = magnitude_spectrum(img)
        assert s.shape == (32, 32)
        assert s.dtype == np.float64

    def test_bgr_input(self):
        img = _bgr(32, 32)
        s = magnitude_spectrum(img)
        assert s.shape == (32, 32)

    def test_non_negative(self):
        img = _rand_gray()
        s = magnitude_spectrum(img)
        assert np.all(s >= 0)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            magnitude_spectrum(np.zeros(10))


# ─── log_magnitude ──────────────────────────────────────────────────────────

class TestLogMagnitudeExtra:
    def test_range(self):
        s = magnitude_spectrum(_rand_gray())
        lm = log_magnitude(s)
        assert lm.min() >= 0.0 - 1e-10
        assert lm.max() <= 1.0 + 1e-10

    def test_constant_input(self):
        s = np.full((10, 10), 5.0)
        lm = log_magnitude(s)
        assert np.allclose(lm, 0.0)

    def test_shape_preserved(self):
        s = magnitude_spectrum(_gray())
        lm = log_magnitude(s)
        assert lm.shape == s.shape


# ─── spectrum_correlation ───────────────────────────────────────────────────

class TestSpectrumCorrelationExtra:
    def test_identical(self):
        s = magnitude_spectrum(_rand_gray(32, 32))
        c = spectrum_correlation(s, s)
        assert c == pytest.approx(1.0, abs=0.01)

    def test_range(self):
        s1 = magnitude_spectrum(_rand_gray(32, 32))
        s2 = magnitude_spectrum(_rand_gray(32, 32))
        c = spectrum_correlation(s1, s2)
        assert -1.0 <= c <= 1.0

    def test_different_sizes(self):
        s1 = magnitude_spectrum(_gray(32, 32))
        s2 = magnitude_spectrum(_gray(16, 16))
        c = spectrum_correlation(s1, s2)
        assert -1.0 <= c <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            spectrum_correlation(np.zeros((0, 0)), np.zeros((10, 10)))


# ─── phase_correlation ──────────────────────────────────────────────────────

class TestPhaseCorrelationExtra:
    def test_identical(self):
        img = _rand_gray(32, 32)
        score, dy, dx = phase_correlation(img, img)
        assert 0.0 <= score <= 1.0

    def test_returns_three(self):
        img = _gray(32, 32)
        result = phase_correlation(img, img)
        assert len(result) == 3

    def test_bgr_input(self):
        img = _bgr(32, 32)
        score, dy, dx = phase_correlation(img, img)
        assert 0.0 <= score <= 1.0


# ─── match_spectra ──────────────────────────────────────────────────────────

class TestMatchSpectraExtra:
    def test_returns_result(self):
        img = _rand_gray(32, 32)
        r = match_spectra(img, img, idx1=0, idx2=1)
        assert isinstance(r, SpectralMatchResult)
        assert 0.0 <= r.score <= 1.0

    def test_indices_stored(self):
        img = _gray()
        r = match_spectra(img, img, idx1=5, idx2=10)
        assert r.idx1 == 5 and r.idx2 == 10

    def test_params_populated(self):
        img = _gray()
        r = match_spectra(img, img)
        assert "corr_score" in r.params
        assert "phase_score" in r.params

    def test_negative_w_corr_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            match_spectra(img, img, w_corr=-1.0)

    def test_negative_w_phase_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            match_spectra(img, img, w_phase=-1.0)

    def test_both_zero_weights_raises(self):
        img = _gray()
        with pytest.raises(ValueError):
            match_spectra(img, img, w_corr=0.0, w_phase=0.0)


# ─── batch_spectral_match ───────────────────────────────────────────────────

class TestBatchSpectralMatchExtra:
    def test_empty(self):
        img = _gray()
        assert batch_spectral_match(img, []) == []

    def test_length(self):
        img = _gray()
        candidates = [_gray(), _gray(), _gray()]
        results = batch_spectral_match(img, candidates)
        assert len(results) == 3

    def test_query_idx(self):
        img = _gray()
        results = batch_spectral_match(img, [_gray()], query_idx=5)
        assert results[0].idx1 == 5
        assert results[0].idx2 == 0
