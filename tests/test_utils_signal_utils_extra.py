"""Extra tests for puzzle_reconstruction/utils/signal_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.signal_utils import (
    smooth_signal,
    normalize_signal,
    find_peaks,
    find_valleys,
    compute_autocorrelation,
    compute_cross_correlation,
    signal_energy,
    segment_signal,
    resample_signal,
    phase_shift,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sig(n=20, val=0.5) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


def _ramp(n=20) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _impulse(n=16, pos=8) -> np.ndarray:
    s = np.zeros(n)
    s[pos] = 1.0
    return s


# ─── smooth_signal ────────────────────────────────────────────────────────────

class TestSmoothSignalExtra:
    def test_returns_ndarray(self):
        assert isinstance(smooth_signal(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert smooth_signal(_ramp()).dtype == np.float64

    def test_same_length(self):
        s = _ramp(15)
        assert len(smooth_signal(s)) == 15

    def test_gaussian_method(self):
        out = smooth_signal(_ramp(), method="gaussian")
        assert len(out) == 20

    def test_moving_average_method(self):
        out = smooth_signal(_ramp(), method="moving_avg")
        assert len(out) == 20

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_ramp(), method="wavelet")

    def test_sigma_zero_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_ramp(), method="gaussian", sigma=0.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_ramp(), method="gaussian", sigma=-1.0)

    def test_window_lt_1_raises(self):
        with pytest.raises(ValueError):
            smooth_signal(_ramp(), method="moving_average", window=0)

    def test_constant_unchanged(self):
        s = _sig(10, 3.0)
        out = smooth_signal(s, method="gaussian")
        np.testing.assert_allclose(out, 3.0, atol=1e-10)


# ─── normalize_signal ─────────────────────────────────────────────────────────

class TestNormalizeSignalExtra:
    def test_returns_ndarray(self):
        assert isinstance(normalize_signal(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert normalize_signal(_ramp()).dtype == np.float64

    def test_min_is_out_min(self):
        out = normalize_signal(_ramp())
        assert out.min() == pytest.approx(0.0)

    def test_max_is_out_max(self):
        out = normalize_signal(_ramp())
        assert out.max() == pytest.approx(1.0)

    def test_custom_range(self):
        out = normalize_signal(_ramp(), out_min=2.0, out_max=5.0)
        assert out.min() == pytest.approx(2.0)
        assert out.max() == pytest.approx(5.0)

    def test_out_min_ge_out_max_raises(self):
        with pytest.raises(ValueError):
            normalize_signal(_ramp(), out_min=1.0, out_max=0.0)

    def test_constant_signal_returns_out_min(self):
        out = normalize_signal(_sig(5, 7.0))
        assert np.allclose(out, 0.0)

    def test_length_preserved(self):
        assert len(normalize_signal(_ramp(12))) == 12


# ─── find_peaks ───────────────────────────────────────────────────────────────

class TestFindPeaksExtra:
    def test_returns_ndarray(self):
        s = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        assert isinstance(find_peaks(s), np.ndarray)

    def test_dtype_int64(self):
        s = np.array([0.0, 1.0, 0.0])
        assert find_peaks(s).dtype == np.int64

    def test_finds_peak(self):
        s = np.array([0.0, 1.0, 0.0])
        peaks = find_peaks(s)
        assert 1 in peaks

    def test_no_peaks_in_flat(self):
        s = _sig(10, 1.0)
        assert len(find_peaks(s)) == 0

    def test_min_height_filters(self):
        s = np.array([0.0, 0.5, 0.0, 1.0, 0.0])
        # peak at 1 (val=0.5) filtered, peak at 3 (val=1.0) kept
        peaks = find_peaks(s, min_height=0.8)
        assert 1 not in peaks
        assert 3 in peaks

    def test_min_distance_lt_1_raises(self):
        with pytest.raises(ValueError):
            find_peaks(_ramp(), min_distance=0)

    def test_short_signal_empty(self):
        s = np.array([1.0, 0.0])
        assert len(find_peaks(s)) == 0

    def test_min_distance_enforced(self):
        s = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        peaks = find_peaks(s, min_distance=4)
        assert len(peaks) <= 1


# ─── find_valleys ─────────────────────────────────────────────────────────────

class TestFindValleysExtra:
    def test_returns_ndarray(self):
        s = np.array([1.0, 0.0, 1.0])
        assert isinstance(find_valleys(s), np.ndarray)

    def test_finds_valley(self):
        s = np.array([1.0, 0.0, 1.0])
        valleys = find_valleys(s)
        assert 1 in valleys

    def test_flat_signal_no_valleys(self):
        s = _sig(10, 1.0)
        assert len(find_valleys(s)) == 0

    def test_min_distance_lt_1_raises(self):
        with pytest.raises(ValueError):
            find_valleys(_ramp(), min_distance=0)

    def test_max_depth_filters(self):
        s = np.array([1.0, 0.5, 1.0, 0.0, 1.0])
        # max_depth=0.3 keeps deep valleys (val <= 0.3), filters shallow (val > 0.3)
        valleys = find_valleys(s, max_depth=0.3)
        assert 1 not in valleys  # val=0.5 > 0.3 → filtered
        assert 3 in valleys      # val=0.0 <= 0.3 → kept


# ─── compute_autocorrelation ──────────────────────────────────────────────────

class TestComputeAutocorrelationExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_autocorrelation(_ramp()), np.ndarray)

    def test_dtype_float64(self):
        assert compute_autocorrelation(_ramp()).dtype == np.float64

    def test_length_2n_minus_1(self):
        n = 10
        out = compute_autocorrelation(_ramp(n))
        assert len(out) == 2 * n - 1

    def test_normalized_center_one(self):
        out = compute_autocorrelation(_ramp(10), normalize=True)
        mid = len(out) // 2
        assert out[mid] == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_autocorrelation(np.array([]))

    def test_symmetric(self):
        out = compute_autocorrelation(_ramp(8), normalize=True)
        np.testing.assert_allclose(out, out[::-1], atol=1e-10)


# ─── compute_cross_correlation ────────────────────────────────────────────────

class TestComputeCrossCorrelationExtra:
    def test_returns_ndarray(self):
        a = _ramp(8)
        assert isinstance(compute_cross_correlation(a, a), np.ndarray)

    def test_length_n_plus_m_minus_1(self):
        a = _ramp(8)
        b = _ramp(5)
        out = compute_cross_correlation(a, b)
        assert len(out) == 8 + 5 - 1

    def test_empty_signal1_raises(self):
        with pytest.raises(ValueError):
            compute_cross_correlation(np.array([]), _ramp(5))

    def test_empty_signal2_raises(self):
        with pytest.raises(ValueError):
            compute_cross_correlation(_ramp(5), np.array([]))

    def test_dtype_float64(self):
        a = _ramp(6)
        assert compute_cross_correlation(a, a).dtype == np.float64


# ─── signal_energy ────────────────────────────────────────────────────────────

class TestSignalEnergyExtra:
    def test_returns_float(self):
        assert isinstance(signal_energy(_ramp()), float)

    def test_nonneg(self):
        assert signal_energy(_ramp()) >= 0.0

    def test_zero_signal_zero_energy(self):
        assert signal_energy(np.zeros(10)) == pytest.approx(0.0)

    def test_unit_impulse_energy_one(self):
        s = np.zeros(10)
        s[5] = 1.0
        assert signal_energy(s) == pytest.approx(1.0)

    def test_constant_signal(self):
        s = _sig(5, 2.0)
        assert signal_energy(s) == pytest.approx(4.0 * 5)


# ─── segment_signal ───────────────────────────────────────────────────────────

class TestSegmentSignalExtra:
    def test_returns_list(self):
        s = np.array([0.0, 1.0, 1.0, 0.0])
        assert isinstance(segment_signal(s, 0.5), list)

    def test_no_segment_all_below(self):
        s = np.zeros(5)
        assert segment_signal(s, 0.5) == []

    def test_one_segment(self):
        s = np.array([0.0, 1.0, 1.0, 0.0])
        segs = segment_signal(s, 0.5)
        assert len(segs) == 1

    def test_two_segments(self):
        s = np.array([1.0, 0.0, 1.0])
        segs = segment_signal(s, 0.5)
        assert len(segs) == 2

    def test_below_mode(self):
        s = np.array([0.0, 1.0, 0.0])
        segs = segment_signal(s, 0.5, above=False)
        assert len(segs) == 2  # index 0 and 2

    def test_elements_are_tuples(self):
        s = np.array([1.0, 1.0, 0.0])
        for seg in segment_signal(s, 0.5):
            assert isinstance(seg, tuple) and len(seg) == 2


# ─── resample_signal ──────────────────────────────────────────────────────────

class TestResampleSignalExtra:
    def test_returns_ndarray(self):
        assert isinstance(resample_signal(_ramp(), 5), np.ndarray)

    def test_dtype_float64(self):
        assert resample_signal(_ramp(), 5).dtype == np.float64

    def test_length_n_out(self):
        assert len(resample_signal(_ramp(10), 7)) == 7

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            resample_signal(np.array([]), 5)

    def test_n_out_lt_1_raises(self):
        with pytest.raises(ValueError):
            resample_signal(_ramp(), 0)

    def test_endpoints_preserved(self):
        s = np.array([0.0, 1.0])
        out = resample_signal(s, 5)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)

    def test_upsample_length(self):
        assert len(resample_signal(_ramp(5), 10)) == 10


# ─── phase_shift ──────────────────────────────────────────────────────────────

class TestPhaseShiftExtra:
    def test_returns_tuple(self):
        s = _ramp(8)
        result = phase_shift(s, s)
        assert isinstance(result, tuple) and len(result) == 2

    def test_identical_shift_zero(self):
        s = _ramp(10)
        shift, _ = phase_shift(s, s)
        assert shift == 0

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError):
            phase_shift(_ramp(5), _ramp(8))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            phase_shift(np.array([]), np.array([]))

    def test_peak_value_nonneg(self):
        s = _ramp(10)
        _, peak = phase_shift(s, s)
        assert peak >= 0.0

    def test_rolled_signal_shift(self):
        s = np.zeros(16)
        s[4] = 1.0
        t = np.roll(s, 3)
        shift, _ = phase_shift(s, t)
        assert abs(shift) == 3
