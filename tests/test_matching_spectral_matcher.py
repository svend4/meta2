"""Тесты для puzzle_reconstruction/matching/spectral_matcher.py."""
import pytest
import numpy as np

from puzzle_reconstruction.matching.spectral_matcher import (
    SpectralMatchResult,
    magnitude_spectrum,
    log_magnitude,
    spectrum_correlation,
    phase_correlation,
    match_spectra,
    batch_spectral_match,
)


def _make_img(h=32, w=32, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestSpectralMatchResult:
    def test_basic_creation(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.7)
        assert r.score == pytest.approx(0.7)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=1.5)

    def test_default_phase_shift(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5)
        assert r.phase_shift == (0.0, 0.0)


class TestMagnitudeSpectrum:
    def test_returns_2d_array(self):
        img = _make_img()
        spec = magnitude_spectrum(img)
        assert spec.ndim == 2

    def test_same_shape_as_input(self):
        img = _make_img(40, 50)
        spec = magnitude_spectrum(img)
        assert spec.shape == (40, 50)

    def test_nonnegative(self):
        img = _make_img()
        spec = magnitude_spectrum(img)
        assert np.all(spec >= 0.0)

    def test_works_with_grayscale(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        spec = magnitude_spectrum(img)
        assert spec.shape == (32, 32)


class TestLogMagnitude:
    def test_returns_normalized(self):
        img = _make_img()
        spec = magnitude_spectrum(img)
        log_spec = log_magnitude(spec)
        assert log_spec.min() >= 0.0
        assert log_spec.max() <= 1.0 + 1e-9

    def test_zero_spectrum_returns_zeros(self):
        spec = np.zeros((32, 32))
        log_spec = log_magnitude(spec)
        assert np.all(log_spec == 0.0)


class TestSpectrumCorrelation:
    def test_identical_spectra_high_corr(self):
        img = _make_img()
        spec = magnitude_spectrum(img)
        corr = spectrum_correlation(spec, spec)
        assert corr > 0.9

    def test_corr_in_range(self):
        s1 = magnitude_spectrum(_make_img(val=50))
        s2 = magnitude_spectrum(_make_img(val=200))
        corr = spectrum_correlation(s1, s2)
        assert -1.0 <= corr <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            spectrum_correlation(np.array([]).reshape(0, 0), np.ones((4, 4)))


class TestPhaseCorrelation:
    def test_returns_three_values(self):
        img = _make_img()
        result = phase_correlation(img, img)
        assert len(result) == 3

    def test_score_in_range(self):
        img = _make_img()
        score, dy, dx = phase_correlation(img, img)
        assert 0.0 <= score <= 1.0


class TestMatchSpectra:
    def test_returns_spectral_match_result(self):
        img = _make_img()
        r = match_spectra(img, img)
        assert isinstance(r, SpectralMatchResult)

    def test_identical_images_high_score(self):
        np.random.seed(42)
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        r = match_spectra(img, img)
        assert r.score >= 0.5

    def test_negative_weight_raises(self):
        img = _make_img()
        with pytest.raises(ValueError):
            match_spectra(img, img, w_corr=-0.1)

    def test_zero_weights_raises(self):
        img = _make_img()
        with pytest.raises(ValueError):
            match_spectra(img, img, w_corr=0.0, w_phase=0.0)

    def test_score_in_range(self):
        img1 = _make_img(val=50)
        img2 = _make_img(val=200)
        r = match_spectra(img1, img2)
        assert 0.0 <= r.score <= 1.0


class TestBatchSpectralMatch:
    def test_batch_returns_correct_count(self):
        query = _make_img()
        candidates = [_make_img(val=v) for v in [50, 100, 150]]
        results = batch_spectral_match(query, candidates)
        assert len(results) == 3

    def test_batch_empty_candidates(self):
        query = _make_img()
        results = batch_spectral_match(query, [])
        assert results == []
