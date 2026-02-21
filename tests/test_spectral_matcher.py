"""Tests for puzzle_reconstruction.matching.spectral_matcher."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h: int = 64, w: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── SpectralMatchResult ─────────────────────────────────────────────────────

class TestSpectralMatchResult:
    def test_fields_stored(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.7)
        assert r.idx1 == 0
        assert r.idx2 == 1
        assert r.score == pytest.approx(0.7)

    def test_default_phase_shift_zero(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5)
        assert r.phase_shift == (0.0, 0.0)

    def test_default_params_empty(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5)
        assert r.params == {}

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=-0.1)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError):
            SpectralMatchResult(idx1=0, idx2=1, score=1.001)

    def test_boundary_scores_allowed(self):
        SpectralMatchResult(idx1=0, idx2=1, score=0.0)
        SpectralMatchResult(idx1=0, idx2=1, score=1.0)

    def test_phase_shift_stored(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5, phase_shift=(3.0, -2.0))
        assert r.phase_shift == (3.0, -2.0)

    def test_params_stored(self):
        r = SpectralMatchResult(idx1=0, idx2=1, score=0.5,
                                params={"w_corr": 0.5})
        assert r.params["w_corr"] == pytest.approx(0.5)


# ─── magnitude_spectrum ──────────────────────────────────────────────────────

class TestMagnitudeSpectrum:
    def test_shape_preserved_gray(self):
        img = _gray(48, 64)
        spec = magnitude_spectrum(img)
        assert spec.shape == (48, 64)

    def test_shape_preserved_bgr(self):
        img = _bgr(48, 64)
        spec = magnitude_spectrum(img)
        assert spec.shape == (48, 64)

    def test_dtype_float64(self):
        spec = magnitude_spectrum(_gray())
        assert spec.dtype == np.float64

    def test_non_negative(self):
        spec = magnitude_spectrum(_gray())
        assert np.all(spec >= 0)

    def test_not_all_zero_for_nonzero_image(self):
        img = np.ones((32, 32), dtype=np.uint8) * 128
        spec = magnitude_spectrum(img)
        assert spec.max() > 0

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            magnitude_spectrum(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_uniform_image_peak_at_center(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        spec = magnitude_spectrum(img)
        cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
        # DC component (center) should be largest
        assert spec[cy, cx] == spec.max()


# ─── log_magnitude ───────────────────────────────────────────────────────────

class TestLogMagnitude:
    def test_shape_preserved(self):
        spec = magnitude_spectrum(_gray(48, 64))
        log_spec = log_magnitude(spec)
        assert log_spec.shape == (48, 64)

    def test_dtype_float64(self):
        spec = magnitude_spectrum(_gray())
        log_spec = log_magnitude(spec)
        assert log_spec.dtype == np.float64

    def test_values_in_unit_interval(self):
        spec = magnitude_spectrum(_gray())
        log_spec = log_magnitude(spec)
        assert log_spec.min() >= 0.0 - 1e-9
        assert log_spec.max() <= 1.0 + 1e-9

    def test_zero_spectrum_all_zero(self):
        spec = np.zeros((16, 16), dtype=np.float64)
        log_spec = log_magnitude(spec)
        assert np.all(log_spec == 0.0)

    def test_max_is_one_for_nonzero_input(self):
        spec = magnitude_spectrum(_gray())
        if spec.max() > 0:
            log_spec = log_magnitude(spec)
            assert log_spec.max() == pytest.approx(1.0)


# ─── spectrum_correlation ────────────────────────────────────────────────────

class TestSpectrumCorrelation:
    def test_identical_returns_one(self):
        s = magnitude_spectrum(_gray())
        corr = spectrum_correlation(s, s)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_range_minus_one_to_one(self):
        s1 = magnitude_spectrum(_gray(seed=0))
        s2 = magnitude_spectrum(_gray(seed=1))
        corr = spectrum_correlation(s1, s2)
        assert -1.0 <= corr <= 1.0

    def test_different_sizes_handled(self):
        s1 = magnitude_spectrum(_gray(32, 32))
        s2 = magnitude_spectrum(_gray(64, 64))
        corr = spectrum_correlation(s1, s2)
        assert -1.0 <= corr <= 1.0

    def test_empty_spectrum_raises(self):
        with pytest.raises(ValueError):
            spectrum_correlation(np.zeros((0, 0)), np.zeros((4, 4)))

    def test_zero_variance_returns_zero(self):
        # Constant spectrum → zero variance → Pearson = 0
        s1 = np.ones((16, 16), dtype=np.float64)
        s2 = magnitude_spectrum(_gray())
        corr = spectrum_correlation(s1, s2)
        assert corr == pytest.approx(0.0)

    def test_symmetric(self):
        s1 = magnitude_spectrum(_gray(seed=0))
        s2 = magnitude_spectrum(_gray(seed=2))
        assert spectrum_correlation(s1, s2) == pytest.approx(
            spectrum_correlation(s2, s1)
        )


# ─── phase_correlation ───────────────────────────────────────────────────────

class TestPhaseCorrelation:
    def test_returns_three_values(self):
        result = phase_correlation(_gray(), _gray())
        assert len(result) == 3

    def test_score_in_unit_interval(self):
        score, dy, dx = phase_correlation(_gray(), _gray())
        assert 0.0 <= score <= 1.0

    def test_same_image_low_shift(self):
        img = _gray()
        score, dy, dx = phase_correlation(img, img)
        # Identical images → shift should be ~0
        assert abs(dy) <= 1.0 and abs(dx) <= 1.0

    def test_bgr_images_accepted(self):
        score, dy, dx = phase_correlation(_bgr(), _bgr())
        assert 0.0 <= score <= 1.0

    def test_different_size_images(self):
        score, dy, dx = phase_correlation(_gray(32, 32), _gray(64, 64))
        assert 0.0 <= score <= 1.0

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            phase_correlation(
                np.zeros((4, 4, 4, 4), dtype=np.uint8), _gray()
            )

    def test_shift_is_numeric(self):
        _, dy, dx = phase_correlation(_gray(), _gray())
        assert isinstance(dy, float)
        assert isinstance(dx, float)


# ─── match_spectra ───────────────────────────────────────────────────────────

class TestMatchSpectra:
    def test_returns_spectral_match_result(self):
        r = match_spectra(_gray(), _gray())
        assert isinstance(r, SpectralMatchResult)

    def test_score_in_unit_interval(self):
        r = match_spectra(_gray(), _gray())
        assert 0.0 <= r.score <= 1.0

    def test_idx_stored(self):
        r = match_spectra(_gray(), _gray(), idx1=3, idx2=7)
        assert r.idx1 == 3
        assert r.idx2 == 7

    def test_phase_shift_tuple(self):
        r = match_spectra(_gray(), _gray())
        assert isinstance(r.phase_shift, tuple)
        assert len(r.phase_shift) == 2

    def test_negative_w_corr_raises(self):
        with pytest.raises(ValueError):
            match_spectra(_gray(), _gray(), w_corr=-0.1)

    def test_negative_w_phase_raises(self):
        with pytest.raises(ValueError):
            match_spectra(_gray(), _gray(), w_phase=-0.1)

    def test_both_weights_zero_raises(self):
        with pytest.raises(ValueError):
            match_spectra(_gray(), _gray(), w_corr=0.0, w_phase=0.0)

    def test_params_stored(self):
        r = match_spectra(_gray(), _gray(), w_corr=0.7, w_phase=0.3)
        assert r.params.get("w_corr") == pytest.approx(0.7)
        assert r.params.get("w_phase") == pytest.approx(0.3)

    def test_only_phase_weight(self):
        r = match_spectra(_gray(), _gray(), w_corr=0.0, w_phase=1.0)
        assert 0.0 <= r.score <= 1.0

    def test_bgr_images(self):
        r = match_spectra(_bgr(), _bgr())
        assert 0.0 <= r.score <= 1.0


# ─── batch_spectral_match ────────────────────────────────────────────────────

class TestBatchSpectralMatch:
    def test_returns_list(self):
        result = batch_spectral_match(_gray(), [_gray(), _gray()])
        assert isinstance(result, list)

    def test_length_matches_candidates(self):
        candidates = [_gray(seed=i) for i in range(4)]
        result = batch_spectral_match(_gray(), candidates)
        assert len(result) == 4

    def test_empty_candidates_returns_empty(self):
        result = batch_spectral_match(_gray(), [])
        assert result == []

    def test_all_spectral_match_results(self):
        result = batch_spectral_match(_gray(), [_gray(), _gray()])
        assert all(isinstance(r, SpectralMatchResult) for r in result)

    def test_query_idx_stored(self):
        result = batch_spectral_match(_gray(), [_gray()], query_idx=5)
        assert result[0].idx1 == 5

    def test_candidate_idx_increments(self):
        result = batch_spectral_match(_gray(), [_gray(), _gray(), _gray()])
        for i, r in enumerate(result):
            assert r.idx2 == i

    def test_scores_in_unit_interval(self):
        result = batch_spectral_match(_gray(), [_gray(), _gray()])
        assert all(0.0 <= r.score <= 1.0 for r in result)
