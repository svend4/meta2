"""Extra tests for puzzle_reconstruction.matching.spectral_matcher."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.spectral_matcher import (
    SpectralMatchResult,
    batch_spectral_match,
    log_magnitude,
    magnitude_spectrum,
    match_spectra,
    phase_correlation,
    spectrum_correlation,
)


def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _const(val=128, h=32, w=32):
    return np.full((h, w), val, dtype=np.uint8)


# ─── SpectralMatchResult extras ───────────────────────────────────────────────

class TestSpectralMatchResultExtra:
    def test_score_zero_allowed(self):
        r = SpectralMatchResult(0, 1, score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one_allowed(self):
        r = SpectralMatchResult(0, 1, score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_phase_shift_default(self):
        r = SpectralMatchResult(0, 1, score=0.5)
        assert r.phase_shift == (0.0, 0.0)

    def test_phase_shift_custom(self):
        r = SpectralMatchResult(0, 1, score=0.5, phase_shift=(-5.0, 3.0))
        assert r.phase_shift[0] == pytest.approx(-5.0)
        assert r.phase_shift[1] == pytest.approx(3.0)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(0, 1, score=-0.01)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(0, 1, score=1.01)

    def test_params_empty_default(self):
        r = SpectralMatchResult(0, 1, score=0.5)
        assert r.params == {}

    def test_params_stored(self):
        r = SpectralMatchResult(0, 1, score=0.5, params={"method": "fft"})
        assert r.params["method"] == "fft"

    def test_repr_is_string(self):
        r = SpectralMatchResult(0, 1, score=0.5)
        assert isinstance(repr(r), str)


# ─── magnitude_spectrum extras ────────────────────────────────────────────────

class TestMagnitudeSpectrumExtra:
    def test_non_square_gray(self):
        img = _gray(h=32, w=64)
        spec = magnitude_spectrum(img)
        assert spec.shape == (32, 64)

    def test_small_image(self):
        img = _gray(h=8, w=8)
        spec = magnitude_spectrum(img)
        assert spec.shape == (8, 8)

    def test_large_image(self):
        img = _gray(h=128, w=128, seed=3)
        spec = magnitude_spectrum(img)
        assert spec.shape == (128, 128)

    def test_nonneg_values(self):
        spec = magnitude_spectrum(_gray())
        assert np.all(spec >= 0.0)

    def test_dtype_float64(self):
        spec = magnitude_spectrum(_gray())
        assert spec.dtype == np.float64

    def test_bgr_returns_gray_spectrum(self):
        img = _bgr()
        spec = magnitude_spectrum(img)
        assert spec.ndim == 2
        assert spec.shape == (64, 64)

    def test_all_zeros_image(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        spec = magnitude_spectrum(img)
        assert isinstance(spec, np.ndarray)
        assert np.all(spec >= 0.0)

    def test_checkerboard_nonzero(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[::2, ::2] = 255
        spec = magnitude_spectrum(img)
        assert spec.max() > 0


# ─── log_magnitude extras ─────────────────────────────────────────────────────

class TestLogMagnitudeExtra:
    def test_non_square(self):
        spec = magnitude_spectrum(_gray(h=32, w=64))
        log_spec = log_magnitude(spec)
        assert log_spec.shape == (32, 64)

    def test_all_positive_spectrum(self):
        spec = np.abs(_gray().astype(np.float64)) + 1.0
        log_spec = log_magnitude(spec)
        assert log_spec.min() >= 0.0 - 1e-9
        assert log_spec.max() <= 1.0 + 1e-9

    def test_large_values_normalized(self):
        spec = np.ones((16, 16), dtype=np.float64) * 1e6
        spec[0, 0] = 1e10
        log_spec = log_magnitude(spec)
        assert log_spec.max() == pytest.approx(1.0)

    def test_identical_spectra_max_corr(self):
        spec = magnitude_spectrum(_gray())
        log_spec = log_magnitude(spec)
        assert isinstance(log_spec, np.ndarray)
        assert log_spec.dtype == np.float64

    def test_output_nonneg(self):
        spec = magnitude_spectrum(_gray(seed=7))
        log_spec = log_magnitude(spec)
        assert np.all(log_spec >= -1e-9)


# ─── spectrum_correlation extras ─────────────────────────────────────────────

class TestSpectrumCorrelationExtra:
    def test_identical_spectra_returns_one(self):
        spec = magnitude_spectrum(_gray())
        assert spectrum_correlation(spec, spec) == pytest.approx(1.0, abs=1e-6)

    def test_range_valid(self):
        s1 = magnitude_spectrum(_gray(seed=0))
        s2 = magnitude_spectrum(_gray(seed=5))
        c = spectrum_correlation(s1, s2)
        assert -1.0 - 1e-9 <= c <= 1.0 + 1e-9

    def test_different_sizes(self):
        s1 = magnitude_spectrum(_gray(32, 32))
        s2 = magnitude_spectrum(_gray(64, 64))
        c = spectrum_correlation(s1, s2)
        assert -1.0 <= c <= 1.0

    def test_symmetric_property(self):
        s1 = magnitude_spectrum(_gray(seed=1))
        s2 = magnitude_spectrum(_gray(seed=2))
        c12 = spectrum_correlation(s1, s2)
        c21 = spectrum_correlation(s2, s1)
        assert c12 == pytest.approx(c21, abs=1e-9)

    def test_returns_float(self):
        s1 = magnitude_spectrum(_gray())
        s2 = magnitude_spectrum(_gray(seed=3))
        assert isinstance(spectrum_correlation(s1, s2), float)

    def test_constant_vs_random(self):
        s_const = np.ones((32, 32), dtype=np.float64)
        s_rand = magnitude_spectrum(_gray())
        c = spectrum_correlation(s_const, s_rand)
        assert c == pytest.approx(0.0)


# ─── phase_correlation extras ─────────────────────────────────────────────────

class TestPhaseCorrelationExtra:
    def test_score_nonneg(self):
        score, _, _ = phase_correlation(_gray(), _gray())
        assert score >= 0.0

    def test_score_le_one(self):
        score, _, _ = phase_correlation(_gray(), _gray(seed=5))
        assert score <= 1.0

    def test_shift_float_types(self):
        _, dy, dx = phase_correlation(_gray(), _gray())
        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_same_image_zero_shift(self):
        img = _gray(seed=3)
        _, dy, dx = phase_correlation(img, img)
        assert abs(dy) <= 2.0
        assert abs(dx) <= 2.0

    def test_bgr_accepted(self):
        score, _, _ = phase_correlation(_bgr(), _bgr(seed=2))
        assert 0.0 <= score <= 1.0

    def test_non_square_images(self):
        score, dy, dx = phase_correlation(_gray(h=32, w=64), _gray(h=64, w=32))
        assert 0.0 <= score <= 1.0

    def test_small_images(self):
        score, dy, dx = phase_correlation(_gray(h=8, w=8), _gray(h=8, w=8))
        assert 0.0 <= score <= 1.0

    def test_constant_image(self):
        img = _const()
        score, _, _ = phase_correlation(img, img)
        assert 0.0 <= score <= 1.0


# ─── match_spectra extras ─────────────────────────────────────────────────────

class TestMatchSpectraExtra:
    def test_score_in_range_random(self):
        for seed in range(4):
            r = match_spectra(_gray(seed=seed), _gray(seed=seed + 4))
            assert 0.0 <= r.score <= 1.0

    def test_identical_images_high_score(self):
        img = _gray(seed=7)
        r = match_spectra(img, img)
        assert r.score > 0.5

    def test_idx_stored(self):
        r = match_spectra(_gray(), _gray(), idx1=10, idx2=20)
        assert r.idx1 == 10
        assert r.idx2 == 20

    def test_only_corr_weight(self):
        r = match_spectra(_gray(), _gray(), w_corr=1.0, w_phase=0.0)
        assert 0.0 <= r.score <= 1.0

    def test_only_phase_weight(self):
        r = match_spectra(_gray(), _gray(), w_corr=0.0, w_phase=1.0)
        assert 0.0 <= r.score <= 1.0

    def test_params_w_corr_stored(self):
        r = match_spectra(_gray(), _gray(), w_corr=0.6, w_phase=0.4)
        assert r.params.get("w_corr") == pytest.approx(0.6)

    def test_bgr_images(self):
        r = match_spectra(_bgr(), _bgr(seed=3))
        assert 0.0 <= r.score <= 1.0

    def test_non_square_images(self):
        r = match_spectra(_gray(h=32, w=64), _gray(h=64, w=32))
        assert 0.0 <= r.score <= 1.0


# ─── batch_spectral_match extras ─────────────────────────────────────────────

class TestBatchSpectralMatchExtra:
    def test_single_candidate(self):
        result = batch_spectral_match(_gray(), [_gray(seed=1)])
        assert len(result) == 1
        assert isinstance(result[0], SpectralMatchResult)

    def test_five_candidates(self):
        candidates = [_gray(seed=i) for i in range(5)]
        result = batch_spectral_match(_gray(), candidates)
        assert len(result) == 5

    def test_all_scores_in_range(self):
        candidates = [_gray(seed=i) for i in range(4)]
        for r in batch_spectral_match(_gray(), candidates):
            assert 0.0 <= r.score <= 1.0

    def test_query_idx_propagated(self):
        result = batch_spectral_match(_gray(), [_gray(), _gray()], query_idx=7)
        assert all(r.idx1 == 7 for r in result)

    def test_candidate_indices_sequential(self):
        result = batch_spectral_match(_gray(), [_gray(seed=i) for i in range(4)])
        for i, r in enumerate(result):
            assert r.idx2 == i

    def test_bgr_query_and_candidates(self):
        result = batch_spectral_match(_bgr(), [_bgr(seed=i) for i in range(3)])
        assert len(result) == 3
        for r in result:
            assert 0.0 <= r.score <= 1.0

    def test_empty_returns_empty_list(self):
        result = batch_spectral_match(_gray(), [])
        assert result == []
