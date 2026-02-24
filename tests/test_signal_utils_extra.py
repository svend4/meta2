"""Extra tests for puzzle_reconstruction/utils/signal_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.signal_utils import (
    compute_autocorrelation,
    compute_cross_correlation,
    find_peaks,
    find_valleys,
    normalize_signal,
    phase_shift,
    resample_signal,
    segment_signal,
    signal_energy,
    smooth_signal,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sine(n: int = 64, freq: float = 4.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * freq, n)
    return np.sin(t)


def _flat(val: float = 1.0, n: int = 16) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


def _ramp(n: int = 16) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


def _pulse(n: int = 16, pos: int = 8) -> np.ndarray:
    s = np.zeros(n, dtype=np.float64)
    s[pos] = 1.0
    return s


# ─── smooth_signal (extra) ────────────────────────────────────────────────────

class TestSmoothSignalExtra:
    def test_returns_ndarray(self):
        assert isinstance(smooth_signal(_sine()), np.ndarray)

    def test_output_length_preserved_gaussian(self):
        s = _sine(32)
        assert smooth_signal(s, method="gaussian").shape == s.shape

    def test_output_length_preserved_moving_avg(self):
        s = _sine(32)
        assert smooth_signal(s, method="moving_avg").shape == s.shape

    def test_dtype_float64_gaussian(self):
        assert smooth_signal(_sine(), method="gaussian").dtype == np.float64

    def test_dtype_float64_moving_avg(self):
        assert smooth_signal(_ramp(), method="moving_avg").dtype == np.float64

    def test_flat_signal_unchanged_gaussian(self):
        s = _flat(5.0, 32)
        result = smooth_signal(s, method="gaussian", sigma=1.0)
        assert np.allclose(result, 5.0, atol=1e-6)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="fft")

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="gaussian", sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="gaussian", sigma=-1.0)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_ramp(), method="moving_avg", window=0)

    def test_large_sigma_smooths(self):
        noisy = np.random.default_rng(0).standard_normal(64)
        smoothed = smooth_signal(noisy, method="gaussian", sigma=5.0)
        # Smoothed signal should have smaller std
        assert smoothed.std() < noisy.std()

    def test_moving_avg_window_1_identity(self):
        s = _ramp(16)
        result = smooth_signal(s, method="moving_avg", window=1)
        assert np.allclose(result, s)


# ─── normalize_signal (extra) ─────────────────────────────────────────────────

class TestNormalizeSignalExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_signal(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert normalize_signal(_ramp()).dtype == np.float64

    def test_output_in_0_1(self):
        result = normalize_signal(_ramp())
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_range(self):
        result = normalize_signal(_ramp(), out_min=2.0, out_max=5.0)
        assert result.min() == pytest.approx(2.0)
        assert result.max() == pytest.approx(5.0)

    def test_constant_signal_returns_out_min(self):
        result = normalize_signal(_flat(3.0), out_min=0.0, out_max=1.0)
        assert np.all(result == pytest.approx(0.0))

    def test_out_min_ge_out_max_raises(self):
        with pytest.raises(ValueError):
            normalize_signal(_ramp(), out_min=1.0, out_max=0.0)

    def test_out_min_equals_out_max_raises(self):
        with pytest.raises(ValueError):
            normalize_signal(_ramp(), out_min=0.5, out_max=0.5)

    def test_length_preserved(self):
        s = _ramp(20)
        assert normalize_signal(s).shape == s.shape

    def test_min_maps_to_out_min(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(s, out_min=0.0, out_max=1.0)
        assert result[0] == pytest.approx(0.0)

    def test_max_maps_to_out_max(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_signal(s, out_min=0.0, out_max=1.0)
        assert result[-1] == pytest.approx(1.0)


# ─── find_peaks (extra) ───────────────────────────────────────────────────────

class TestFindPeaksExtra:
    def test_returns_ndarray(self):
        assert isinstance(find_peaks(_sine()), np.ndarray)

    def test_dtype_int64(self):
        assert find_peaks(_sine()).dtype == np.int64

    def test_short_signal_empty(self):
        assert len(find_peaks(np.array([1.0, 2.0]))) == 0

    def test_empty_signal(self):
        assert len(find_peaks(np.array([]))) == 0

    def test_flat_signal_no_peaks(self):
        assert len(find_peaks(_flat(5.0))) == 0

    def test_single_peak_detected(self):
        s = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        peaks = find_peaks(s)
        assert 2 in peaks

    def test_multiple_peaks_detected(self):
        s = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        peaks = find_peaks(s)
        assert len(peaks) == 2

    def test_min_height_filters_low_peaks(self):
        s = np.array([0.0, 0.5, 0.0, 2.0, 0.0])
        peaks = find_peaks(s, min_height=1.0)
        assert 1 not in peaks
        assert 3 in peaks

    def test_min_distance_enforced(self):
        s = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        peaks = find_peaks(s, min_distance=3)
        # Peaks at 1, 3, 5; with min_distance=3, only 1 and 5 (or less)
        if len(peaks) > 1:
            assert peaks[-1] - peaks[0] >= 3

    def test_min_distance_zero_raises(self):
        with pytest.raises(ValueError):
            find_peaks(_sine(), min_distance=0)

    def test_min_distance_negative_raises(self):
        with pytest.raises(ValueError):
            find_peaks(_sine(), min_distance=-1)

    def test_peaks_are_valid_indices(self):
        s = _sine(64)
        peaks = find_peaks(s)
        assert all(0 <= i < len(s) for i in peaks)


# ─── find_valleys (extra) ─────────────────────────────────────────────────────

class TestFindValleysExtra:
    def test_returns_ndarray(self):
        assert isinstance(find_valleys(_sine()), np.ndarray)

    def test_dtype_int64(self):
        assert find_valleys(_sine()).dtype == np.int64

    def test_flat_signal_no_valleys(self):
        assert len(find_valleys(_flat(5.0))) == 0

    def test_single_valley_detected(self):
        s = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
        valleys = find_valleys(s)
        assert 2 in valleys

    def test_min_distance_zero_raises(self):
        with pytest.raises(ValueError):
            find_valleys(_sine(), min_distance=0)

    def test_valleys_are_valid_indices(self):
        s = _sine(64)
        valleys = find_valleys(s)
        assert all(0 <= i < len(s) for i in valleys)

    def test_sine_has_valleys(self):
        # Sine wave should have valleys
        assert len(find_valleys(_sine(64, freq=2.0))) > 0


# ─── compute_autocorrelation (extra) ──────────────────────────────────────────

class TestComputeAutocorrelationExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_autocorrelation(_sine()), np.ndarray)

    def test_dtype_float64(self):
        assert compute_autocorrelation(_sine()).dtype == np.float64

    def test_output_length(self):
        s = _sine(32)
        result = compute_autocorrelation(s)
        assert len(result) == 2 * len(s) - 1

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_autocorrelation(np.array([]))

    def test_normalized_peak_is_one(self):
        s = _sine(32)
        ac = compute_autocorrelation(s, normalize=True)
        assert ac[len(ac) // 2] == pytest.approx(1.0, abs=1e-9)

    def test_unnormalized_larger_than_normalized(self):
        s = _sine(32) * 5.0
        ac_norm = compute_autocorrelation(s, normalize=True)
        ac_unnorm = compute_autocorrelation(s, normalize=False)
        assert abs(ac_unnorm).max() >= abs(ac_norm).max()

    def test_flat_signal_autocorr_normalized(self):
        s = _flat(2.0, 16)
        ac = compute_autocorrelation(s, normalize=True)
        # Peak at center should be 1
        assert ac[len(ac) // 2] == pytest.approx(1.0, abs=1e-9)


# ─── compute_cross_correlation (extra) ────────────────────────────────────────

class TestComputeCrossCorrelationExtra:
    def test_returns_ndarray(self):
        s = _sine(32)
        assert isinstance(compute_cross_correlation(s, s), np.ndarray)

    def test_dtype_float64(self):
        s = _sine(32)
        assert compute_cross_correlation(s, s).dtype == np.float64

    def test_output_length(self):
        s1 = _sine(16)
        s2 = _sine(16)
        result = compute_cross_correlation(s1, s2)
        assert len(result) == 2 * 16 - 1

    def test_empty_s1_raises(self):
        with pytest.raises(ValueError):
            compute_cross_correlation(np.array([]), np.array([1.0]))

    def test_empty_s2_raises(self):
        with pytest.raises(ValueError):
            compute_cross_correlation(np.array([1.0]), np.array([]))

    def test_identical_signals_peak_center(self):
        s = _sine(32)
        cc = compute_cross_correlation(s, s, normalize=True)
        center = len(cc) // 2
        assert cc[center] == pytest.approx(1.0, abs=1e-9)

    def test_unnormalized_option(self):
        s = _sine(32) * 2.0
        cc_norm = compute_cross_correlation(s, s, normalize=True)
        cc_unnorm = compute_cross_correlation(s, s, normalize=False)
        assert abs(cc_unnorm).max() > abs(cc_norm).max()

    def test_different_lengths_allowed(self):
        s1 = _sine(16)
        s2 = _sine(8)
        result = compute_cross_correlation(s1, s2)
        assert len(result) == 16 + 8 - 1


# ─── signal_energy (extra) ────────────────────────────────────────────────────

class TestSignalEnergyExtra:
    def test_returns_float(self):
        assert isinstance(signal_energy(_sine()), float)

    def test_zero_signal_energy_zero(self):
        assert signal_energy(np.zeros(16)) == pytest.approx(0.0)

    def test_unit_vector_energy(self):
        s = np.array([1.0, 0.0, 0.0, 0.0])
        assert signal_energy(s) == pytest.approx(1.0)

    def test_all_ones_energy(self):
        s = np.ones(8)
        assert signal_energy(s) == pytest.approx(8.0)

    def test_nonneg_energy(self):
        assert signal_energy(_sine()) >= 0.0

    def test_energy_scales_with_amplitude(self):
        s = _sine(32)
        e1 = signal_energy(s)
        e2 = signal_energy(s * 2.0)
        assert e2 == pytest.approx(4.0 * e1)

    def test_single_value(self):
        assert signal_energy(np.array([3.0])) == pytest.approx(9.0)


# ─── segment_signal (extra) ───────────────────────────────────────────────────

class TestSegmentSignalExtra:
    def test_returns_list(self):
        assert isinstance(segment_signal(_ramp(16), 8.0), list)

    def test_all_above_threshold_one_segment(self):
        segs = segment_signal(_flat(5.0, 16), 3.0, above=True)
        assert len(segs) == 1
        assert segs[0] == (0, 16)

    def test_all_below_threshold_no_segment(self):
        segs = segment_signal(_flat(1.0, 16), 3.0, above=True)
        assert segs == []

    def test_below_mode(self):
        segs = segment_signal(_flat(1.0, 16), 3.0, above=False)
        assert len(segs) == 1

    def test_mixed_signal_segments(self):
        s = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 5.0, 0.0])
        segs = segment_signal(s, 3.0, above=True)
        assert len(segs) == 2

    def test_segment_tuple_format(self):
        s = np.array([5.0, 5.0, 5.0])
        segs = segment_signal(s, 3.0)
        assert segs[0] == (0, 3)

    def test_segment_indices_valid(self):
        s = _sine(32)
        segs = segment_signal(s, 0.0, above=True)
        for start, end in segs:
            assert 0 <= start < end <= len(s)

    def test_empty_signal(self):
        segs = segment_signal(np.array([]), 0.5)
        assert segs == []


# ─── resample_signal (extra) ──────────────────────────────────────────────────

class TestResampleSignalExtra:
    def test_returns_ndarray(self):
        assert isinstance(resample_signal(_ramp(16), 32), np.ndarray)

    def test_dtype_float64(self):
        assert resample_signal(_ramp(16), 32).dtype == np.float64

    def test_output_length(self):
        result = resample_signal(_ramp(16), 8)
        assert len(result) == 8

    def test_upsample_preserves_endpoints(self):
        s = np.array([0.0, 1.0])
        result = resample_signal(s, 100)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_downsample_preserves_endpoints(self):
        s = np.linspace(0.0, 1.0, 100)
        result = resample_signal(s, 10)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_same_size_identity(self):
        s = np.array([1.0, 2.0, 3.0])
        result = resample_signal(s, 3)
        assert np.allclose(result, s)

    def test_n_out_zero_raises(self):
        with pytest.raises(ValueError):
            resample_signal(_ramp(16), 0)

    def test_n_out_negative_raises(self):
        with pytest.raises(ValueError):
            resample_signal(_ramp(16), -1)

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            resample_signal(np.array([]), 5)

    def test_single_value_resampled_constant(self):
        result = resample_signal(np.array([7.0]), 10)
        assert np.all(result == pytest.approx(7.0))


# ─── phase_shift (extra) ──────────────────────────────────────────────────────

class TestPhaseShiftExtra:
    def test_returns_tuple(self):
        s = _sine(32)
        result = phase_shift(s, s)
        assert isinstance(result, tuple) and len(result) == 2

    def test_identical_signals_zero_shift(self):
        s = _sine(32)
        shift, _ = phase_shift(s, s)
        assert shift == 0

    def test_peak_value_in_0_1_for_normalized(self):
        s = _sine(32)
        _, peak = phase_shift(s, s)
        assert 0.0 <= peak <= 1.0 + 1e-9

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            phase_shift(_sine(16), _sine(32))

    def test_empty_signals_raises(self):
        with pytest.raises(ValueError):
            phase_shift(np.array([]), np.array([]))

    def test_shift_is_integer(self):
        s = _sine(32)
        shift, _ = phase_shift(s, s)
        assert isinstance(shift, int)

    def test_known_shift_detected(self):
        n = 64
        s = _sine(n, freq=2.0)
        # Shift s by 5 positions
        k = 5
        s2 = np.roll(s, k)
        shift, _ = phase_shift(s, s2)
        # Shift should be close to -k or +k
        assert abs(shift) == k or abs(abs(shift) - k) <= 2
