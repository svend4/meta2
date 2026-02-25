"""Тесты для puzzle_reconstruction.matching.spectral_matcher."""
import pytest
import numpy as np
from puzzle_reconstruction.matching.spectral_matcher import (
    SpectralMatchResult,
    batch_spectral_match,
    log_magnitude,
    magnitude_spectrum,
    match_spectra,
    phase_correlation,
    spectrum_correlation,
)


def _gray(h=32, w=32) -> np.ndarray:
    """Тестовое серое изображение uint8."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def _bgr(h=32, w=32) -> np.ndarray:
    """Тестовое BGR-изображение uint8."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ─── TestSpectralMatchResult ──────────────────────────────────────────────────

class TestSpectralMatchResult:
    def test_basic_construction(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.8)
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.score == pytest.approx(0.8)

    def test_score_zero_ok(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.0)
        assert r.score == pytest.approx(0.0)

    def test_score_one_ok(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=1.0)
        assert r.score == pytest.approx(1.0)

    def test_score_neg_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=1.01)

    def test_phase_shift_default(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5)
        assert r.phase_shift == (0.0, 0.0)

    def test_custom_phase_shift(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5, phase_shift=(2.0, -3.0))
        assert r.phase_shift == (2.0, -3.0)


# ─── TestMagnitudeSpectrum ────────────────────────────────────────────────────

class TestMagnitudeSpectrum:
    def test_returns_2d_array(self):
        img = _gray()
        spec = magnitude_spectrum(img)
        assert spec.ndim == 2

    def test_shape_matches_input(self):
        img = _gray(32, 48)
        spec = magnitude_spectrum(img)
        assert spec.shape == (32, 48)

    def test_bgr_input(self):
        img = _bgr()
        spec = magnitude_spectrum(img)
        assert spec.ndim == 2

    def test_non_negative(self):
        img = _gray()
        spec = magnitude_spectrum(img)
        assert (spec >= 0).all()

    def test_invalid_ndim_raises(self):
        arr = np.zeros((5,))
        with pytest.raises((ValueError, Exception)):
            magnitude_spectrum(arr)


# ─── TestLogMagnitude ─────────────────────────────────────────────────────────

class TestLogMagnitude:
    def test_returns_normalized(self):
        spec = magnitude_spectrum(_gray())
        log = log_magnitude(spec)
        assert log.min() >= 0.0
        assert log.max() <= 1.0

    def test_uniform_spectrum_returns_zeros(self):
        spec = np.zeros((16, 16))
        log = log_magnitude(spec)
        assert (log == 0).all()

    def test_shape_preserved(self):
        spec = np.ones((8, 10)) * 5.0
        log = log_magnitude(spec)
        assert log.shape == (8, 10)


# ─── TestSpectrumCorrelation ──────────────────────────────────────────────────

class TestSpectrumCorrelation:
    def test_identical_spectra_high_correlation(self):
        img = _gray()
        s = magnitude_spectrum(img)
        corr = spectrum_correlation(s, s)
        assert corr > 0.9

    def test_returns_float(self):
        s1 = magnitude_spectrum(_gray(32, 32))
        s2 = magnitude_spectrum(_gray(32, 32))
        assert isinstance(spectrum_correlation(s1, s2), float)

    def test_correlation_in_range(self):
        s1 = magnitude_spectrum(_gray())
        s2 = magnitude_spectrum(_bgr())
        corr = spectrum_correlation(s1, s2)
        assert -1.0 <= corr <= 1.0

    def test_different_sizes_handled(self):
        s1 = magnitude_spectrum(_gray(32, 32))
        s2 = magnitude_spectrum(_gray(16, 16))
        corr = spectrum_correlation(s1, s2)
        assert isinstance(corr, float)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            spectrum_correlation(np.array([]).reshape(0, 4),
                                  np.array([]).reshape(0, 4))


# ─── TestPhaseCorrelation ─────────────────────────────────────────────────────

class TestPhaseCorrelation:
    def test_returns_tuple_3(self):
        img1 = _gray()
        img2 = _gray()
        result = phase_correlation(img1, img2)
        assert len(result) == 3

    def test_score_in_range(self):
        img1 = _gray()
        img2 = _gray()
        score, dy, dx = phase_correlation(img1, img2)
        assert 0.0 <= score <= 1.0

    def test_same_image_high_score(self):
        img = _gray()
        score, dy, dx = phase_correlation(img, img)
        # Identical images should have high peak
        assert score > 0.0

    def test_bgr_input(self):
        img = _bgr()
        score, dy, dx = phase_correlation(img, img)
        assert isinstance(score, float)


# ─── TestMatchSpectra ─────────────────────────────────────────────────────────

class TestMatchSpectra:
    def test_returns_spectral_match_result(self):
        img1 = _gray()
        img2 = _gray()
        r = match_spectra(img1, img2)
        assert isinstance(r, SpectralMatchResult)

    def test_score_in_range(self):
        img1 = _gray()
        img2 = _gray()
        r = match_spectra(img1, img2)
        assert 0.0 <= r.score <= 1.0

    def test_idx_stored(self):
        img1 = _gray()
        img2 = _gray()
        r = match_spectra(img1, img2, idx1=3, idx2=7)
        assert r.idx1 == 3
        assert r.idx2 == 7

    def test_negative_w_corr_raises(self):
        img1 = _gray()
        img2 = _gray()
        with pytest.raises(ValueError):
            match_spectra(img1, img2, w_corr=-0.1)

    def test_negative_w_phase_raises(self):
        img1 = _gray()
        img2 = _gray()
        with pytest.raises(ValueError):
            match_spectra(img1, img2, w_phase=-0.1)

    def test_both_weights_zero_raises(self):
        img1 = _gray()
        img2 = _gray()
        with pytest.raises(ValueError):
            match_spectra(img1, img2, w_corr=0.0, w_phase=0.0)

    def test_same_image_score_ok(self):
        img = _gray()
        r = match_spectra(img, img)
        assert r.score >= 0.0


# ─── TestBatchSpectralMatch ───────────────────────────────────────────────────

class TestBatchSpectralMatch:
    def test_returns_list(self):
        query = _gray()
        candidates = [_gray(), _gray(), _gray()]
        results = batch_spectral_match(query, candidates)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_empty_candidates(self):
        query = _gray()
        results = batch_spectral_match(query, [])
        assert results == []

    def test_query_idx_stored(self):
        query = _gray()
        candidates = [_gray()]
        results = batch_spectral_match(query, candidates, query_idx=5)
        assert results[0].idx1 == 5

    def test_candidate_idx_assigned(self):
        query = _gray()
        candidates = [_gray(), _gray()]
        results = batch_spectral_match(query, candidates)
        assert results[0].idx2 == 0
        assert results[1].idx2 == 1
