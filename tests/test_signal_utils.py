"""Tests for puzzle_reconstruction.utils.signal_utils."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sine(n: int = 64, freq: float = 4.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * freq, n)
    return np.sin(t)


def _impulse(n: int = 16, pos: int = 8) -> np.ndarray:
    s = np.zeros(n)
    s[pos] = 1.0
    return s


def _ramp(n: int = 16) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


# ─── smooth_signal ────────────────────────────────────────────────────────────

class TestSmoothSignal:
    def test_returns_float64(self):
        assert smooth_signal(_sine()).dtype == np.float64

    def test_same_length(self):
        s = _sine(32)
        assert len(smooth_signal(s)) == 32

    def test_gaussian_default(self):
        s = _sine(64)
        result = smooth_signal(s)
        assert result.dtype == np.float64

    def test_moving_avg_method(self):
        s = _sine(64)
        result = smooth_signal(s, method="moving_avg", window=5)
        assert len(result) == 64

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="gaussian", sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="gaussian", sigma=-1.0)

    def test_window_less_than_1_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="moving_avg", window=0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_sine(), method="median")

    def test_gaussian_reduces_amplitude(self):
        s = _sine(64, freq=8.0)
        smoothed = smooth_signal(s, method="gaussian", sigma=3.0)
        assert np.max(np.abs(smoothed)) <= np.max(np.abs(s)) + 1e-10

    def test_constant_signal_unchanged(self):
        s = np.ones(32, dtype=np.float64)
        result = smooth_signal(s, method="gaussian", sigma=1.0)
        np.testing.assert_array_almost_equal(result, s, decimal=10)


# ─── normalize_signal ─────────────────────────────────────────────────────────

class TestNormalizeSignal:
    def test_returns_float64(self):
        assert normalize_signal(_ramp()).dtype == np.float64

    def test_same_length(self):
        s = _ramp(20)
        assert len(normalize_signal(s)) == 20

    def test_out_min_geq_out_max_raises(self):
        with pytest.raises(ValueError):
            normalize_signal(_ramp(), out_min=1.0, out_max=0.0)
        with pytest.raises(ValueError):
            normalize_signal(_ramp(), out_min=0.5, out_max=0.5)

    def test_range_0_to_1(self):
        result = normalize_signal(_sine(), out_min=0.0, out_max=1.0)
        assert result.min() == pytest.approx(0.0, abs=1e-10)
        assert result.max() == pytest.approx(1.0, abs=1e-10)

    def test_custom_range(self):
        result = normalize_signal(_ramp(), out_min=-1.0, out_max=1.0)
        assert result.min() == pytest.approx(-1.0, abs=1e-10)
        assert result.max() == pytest.approx(1.0, abs=1e-10)

    def test_constant_signal_fills_out_min(self):
        s = np.ones(10)
        result = normalize_signal(s, out_min=0.0, out_max=1.0)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_monotonic_input_monotonic_output(self):
        s = _ramp(20)
        result = normalize_signal(s)
        assert np.all(np.diff(result) >= 0)


# ─── find_peaks ───────────────────────────────────────────────────────────────

class TestFindPeaks:
    def test_returns_int64(self):
        assert find_peaks(_sine()).dtype == np.int64

    def test_short_signal_empty(self):
        assert len(find_peaks(np.array([1.0, 2.0]))) == 0

    def test_min_distance_less_than_1_raises(self):
        with pytest.raises(ValueError):
            find_peaks(_sine(), min_distance=0)

    def test_finds_known_peaks(self):
        # Two clear peaks at positions 2 and 8
        s = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0])
        peaks = find_peaks(s)
        assert 2 in peaks
        assert 8 in peaks

    def test_min_height_filters(self):
        s = np.array([0.0, 0.3, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0])
        peaks = find_peaks(s, min_height=0.5)
        assert 4 in peaks
        assert 1 not in peaks
        assert 7 not in peaks

    def test_min_distance_enforced(self):
        s = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        peaks = find_peaks(s, min_distance=3)
        # With distance=3, consecutive peaks at 1,3,5 cannot all be kept
        for i in range(len(peaks) - 1):
            assert peaks[i + 1] - peaks[i] >= 3

    def test_flat_signal_no_peaks(self):
        s = np.ones(20)
        assert len(find_peaks(s)) == 0

    def test_returns_array(self):
        result = find_peaks(_sine(64))
        assert isinstance(result, np.ndarray)


# ─── find_valleys ─────────────────────────────────────────────────────────────

class TestFindValleys:
    def test_returns_int64(self):
        assert find_valleys(_sine()).dtype == np.int64

    def test_finds_known_valleys(self):
        s = np.array([1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0])
        valleys = find_valleys(s)
        assert 2 in valleys
        assert 6 in valleys

    def test_min_distance_raises(self):
        with pytest.raises(ValueError):
            find_valleys(_sine(), min_distance=0)

    def test_no_valleys_on_monotone(self):
        s = _ramp(20)
        assert len(find_valleys(s)) == 0

    def test_sine_valleys_and_peaks_complementary(self):
        s = _sine(64, freq=3.0)
        peaks = find_peaks(s)
        valleys = find_valleys(s)
        # Sine has equal number of peaks and valleys
        assert len(peaks) > 0
        assert len(valleys) > 0


# ─── compute_autocorrelation ─────────────────────────────────────────────────

class TestComputeAutocorrelation:
    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_autocorrelation(np.array([]))

    def test_returns_float64(self):
        assert compute_autocorrelation(_sine()).dtype == np.float64

    def test_length_2n_minus_1(self):
        s = _sine(16)
        ac = compute_autocorrelation(s)
        assert len(ac) == 2 * 16 - 1

    def test_normalized_peak_is_one(self):
        s = _sine(32)
        ac = compute_autocorrelation(s, normalize=True)
        assert ac[len(ac) // 2] == pytest.approx(1.0, abs=1e-6)

    def test_unnormalized_peak_equals_energy(self):
        s = np.array([1.0, 2.0, 3.0])
        ac = compute_autocorrelation(s, normalize=False)
        energy = float(np.dot(s, s))
        assert ac[len(ac) // 2] == pytest.approx(energy, rel=1e-6)

    def test_symmetric(self):
        s = _sine(32)
        ac = compute_autocorrelation(s, normalize=True)
        n = len(ac)
        np.testing.assert_array_almost_equal(ac[:n // 2], ac[n // 2 + 1:][::-1], decimal=10)


# ─── compute_cross_correlation ────────────────────────────────────────────────

class TestComputeCrossCorrelation:
    def test_empty_signal1_raises(self):
        with pytest.raises(ValueError):
            compute_cross_correlation(np.array([]), np.array([1.0]))

    def test_empty_signal2_raises(self):
        with pytest.raises(ValueError):
            compute_cross_correlation(np.array([1.0]), np.array([]))

    def test_returns_float64(self):
        assert compute_cross_correlation(_sine(16), _sine(16)).dtype == np.float64

    def test_length_n_plus_m_minus_1(self):
        s1 = np.ones(10)
        s2 = np.ones(6)
        cc = compute_cross_correlation(s1, s2)
        assert len(cc) == 10 + 6 - 1

    def test_same_signal_peak_at_center(self):
        s = _sine(32)
        cc = compute_cross_correlation(s, s, normalize=True)
        assert cc[len(cc) // 2] == pytest.approx(1.0, abs=1e-6)

    def test_normalized_peak_leq_one(self):
        s1 = _sine(32, freq=2.0)
        s2 = _sine(32, freq=4.0)
        cc = compute_cross_correlation(s1, s2, normalize=True)
        assert cc.max() <= 1.0 + 1e-10

    def test_unnormalized_larger_than_normalized(self):
        s = _sine(32) * 10.0
        cc_norm = compute_cross_correlation(s, s, normalize=True)
        cc_raw = compute_cross_correlation(s, s, normalize=False)
        assert cc_raw.max() >= cc_norm.max() - 1e-10


# ─── signal_energy ────────────────────────────────────────────────────────────

class TestSignalEnergy:
    def test_returns_float(self):
        assert isinstance(signal_energy(_sine()), float)

    def test_non_negative(self):
        assert signal_energy(_sine()) >= 0.0

    def test_zero_signal_returns_zero(self):
        assert signal_energy(np.zeros(20)) == pytest.approx(0.0)

    def test_known_value(self):
        # [3, 4] → 9 + 16 = 25
        assert signal_energy(np.array([3.0, 4.0])) == pytest.approx(25.0)

    def test_unit_impulse_returns_one(self):
        s = _impulse(16, pos=8)
        assert signal_energy(s) == pytest.approx(1.0)

    def test_scale_invariant_squared(self):
        s = _sine(32)
        e1 = signal_energy(s)
        e4 = signal_energy(2.0 * s)
        assert e4 == pytest.approx(4.0 * e1, rel=1e-9)


# ─── segment_signal ───────────────────────────────────────────────────────────

class TestSegmentSignal:
    def test_returns_list(self):
        s = np.array([0.0, 1.0, 1.0, 0.0])
        result = segment_signal(s, threshold=0.5)
        assert isinstance(result, list)

    def test_known_segments_above(self):
        s = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        segs = segment_signal(s, threshold=0.5, above=True)
        assert (2, 4) in segs
        assert (6, 7) in segs

    def test_known_segments_below(self):
        s = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        segs = segment_signal(s, threshold=0.5, above=False)
        assert (1, 3) in segs
        assert (5, 6) in segs

    def test_all_above_single_segment(self):
        s = np.ones(10)
        segs = segment_signal(s, threshold=0.5)
        assert segs == [(0, 10)]

    def test_none_above_empty(self):
        s = np.zeros(10)
        segs = segment_signal(s, threshold=0.5, above=True)
        assert segs == []

    def test_ends_in_segment(self):
        s = np.array([0.0, 0.0, 1.0, 1.0])
        segs = segment_signal(s, threshold=0.5)
        assert (2, 4) in segs

    def test_segments_non_overlapping(self):
        s = _sine(64)
        segs = segment_signal(s, threshold=0.0)
        for i in range(len(segs) - 1):
            assert segs[i][1] <= segs[i + 1][0]


# ─── resample_signal ──────────────────────────────────────────────────────────

class TestResampleSignal:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resample_signal(np.array([]), n_out=10)

    def test_n_out_less_than_1_raises(self):
        with pytest.raises(ValueError):
            resample_signal(_ramp(), n_out=0)

    def test_length_n_out(self):
        result = resample_signal(_ramp(10), n_out=20)
        assert len(result) == 20

    def test_dtype_float64(self):
        assert resample_signal(_ramp(), n_out=8).dtype == np.float64

    def test_same_length_unchanged(self):
        s = _ramp(16)
        result = resample_signal(s, n_out=16)
        np.testing.assert_array_almost_equal(result, s, decimal=10)

    def test_single_point_fills(self):
        result = resample_signal(np.array([5.0]), n_out=8)
        np.testing.assert_array_almost_equal(result, np.full(8, 5.0))

    def test_endpoints_preserved(self):
        s = _ramp(10)
        result = resample_signal(s, n_out=20)
        assert result[0] == pytest.approx(s[0], abs=1e-6)
        assert result[-1] == pytest.approx(s[-1], abs=1e-6)

    def test_downsample(self):
        result = resample_signal(_ramp(100), n_out=10)
        assert len(result) == 10


# ─── phase_shift ──────────────────────────────────────────────────────────────

class TestPhaseShift:
    def test_returns_tuple(self):
        s = _sine(32)
        result = phase_shift(s, s)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shift_is_int(self):
        s = _sine(32)
        shift, _ = phase_shift(s, s)
        assert isinstance(shift, int)

    def test_peak_value_is_float(self):
        s = _sine(32)
        _, peak = phase_shift(s, s)
        assert isinstance(peak, float)

    def test_same_signal_zero_shift(self):
        s = _sine(64, freq=4.0)
        shift, _ = phase_shift(s, s)
        assert shift == 0

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            phase_shift(_sine(10), _sine(12))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            phase_shift(np.array([]), np.array([]))

    def test_known_shift(self):
        n = 64
        s = _sine(n, freq=4.0)
        k = 5
        s_shifted = np.roll(s, k)
        shift, _ = phase_shift(s, s_shifted)
        # The peak of cross-correlation should be near k
        assert abs(shift - k) <= 2

    def test_peak_value_leq_one(self):
        s = _sine(32)
        _, peak = phase_shift(s, s)
        assert peak <= 1.0 + 1e-10
