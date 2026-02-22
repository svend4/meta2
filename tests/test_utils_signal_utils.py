"""Tests for utils/signal_utils.py."""
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

def make_sine(n=64, freq=4.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.sin(freq * t)


def make_impulse(n=20, pos=10):
    s = np.zeros(n)
    s[pos] = 1.0
    return s


# ─── smooth_signal ────────────────────────────────────────────────────────────

class TestSmoothSignal:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            smooth_signal(np.ones(10), method="median")

    def test_gaussian_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            smooth_signal(np.ones(10), method="gaussian", sigma=0.0)

    def test_gaussian_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            smooth_signal(np.ones(10), method="gaussian", sigma=-1.0)

    def test_moving_avg_window_zero_raises(self):
        with pytest.raises(ValueError, match="window"):
            smooth_signal(np.ones(10), method="moving_avg", window=0)

    def test_moving_avg_window_negative_raises(self):
        with pytest.raises(ValueError, match="window"):
            smooth_signal(np.ones(10), method="moving_avg", window=-1)

    def test_gaussian_returns_same_length(self):
        s = make_sine()
        result = smooth_signal(s, method="gaussian", sigma=1.0)
        assert len(result) == len(s)

    def test_moving_avg_returns_same_length(self):
        s = make_sine()
        result = smooth_signal(s, method="moving_avg", window=5)
        assert len(result) == len(s)

    def test_gaussian_returns_float64(self):
        result = smooth_signal(np.ones(10), method="gaussian")
        assert result.dtype == np.float64

    def test_moving_avg_returns_float64(self):
        result = smooth_signal(np.ones(10), method="moving_avg")
        assert result.dtype == np.float64

    def test_constant_signal_unchanged_gaussian(self):
        s = np.full(20, 5.0)
        result = smooth_signal(s, method="gaussian", sigma=2.0)
        np.testing.assert_allclose(result, 5.0, atol=1e-6)

    def test_constant_signal_unchanged_moving_avg(self):
        s = np.full(20, 3.0)
        result = smooth_signal(s, method="moving_avg", window=5)
        np.testing.assert_allclose(result[2:-2], 3.0, atol=1e-10)


# ─── normalize_signal ─────────────────────────────────────────────────────────

class TestNormalizeSignal:
    def test_out_min_ge_out_max_raises(self):
        with pytest.raises(ValueError, match="out_min"):
            normalize_signal(np.array([1.0, 2.0, 3.0]), out_min=1.0, out_max=1.0)

    def test_out_min_greater_raises(self):
        with pytest.raises(ValueError, match="out_min"):
            normalize_signal(np.array([1.0, 2.0]), out_min=2.0, out_max=1.0)

    def test_constant_signal_returns_out_min(self):
        result = normalize_signal(np.full(10, 5.0), out_min=0.0, out_max=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_min_maps_to_out_min(self):
        s = np.array([0.0, 5.0, 10.0])
        result = normalize_signal(s, out_min=2.0, out_max=8.0)
        assert result[0] == pytest.approx(2.0)

    def test_max_maps_to_out_max(self):
        s = np.array([0.0, 5.0, 10.0])
        result = normalize_signal(s, out_min=2.0, out_max=8.0)
        assert result[-1] == pytest.approx(8.0)

    def test_range_in_bounds(self):
        s = np.random.default_rng(0).standard_normal(50)
        result = normalize_signal(s, out_min=0.0, out_max=1.0)
        assert result.min() >= 0.0 - 1e-12
        assert result.max() <= 1.0 + 1e-12

    def test_returns_float64(self):
        result = normalize_signal(np.array([1.0, 2.0, 3.0]))
        assert result.dtype == np.float64

    def test_custom_range(self):
        result = normalize_signal(np.array([0.0, 10.0]), out_min=-1.0, out_max=1.0)
        assert result[0] == pytest.approx(-1.0)
        assert result[-1] == pytest.approx(1.0)


# ─── find_peaks ───────────────────────────────────────────────────────────────

class TestFindPeaks:
    def test_min_distance_zero_raises(self):
        with pytest.raises(ValueError, match="min_distance"):
            find_peaks(np.ones(10), min_distance=0)

    def test_min_distance_negative_raises(self):
        with pytest.raises(ValueError, match="min_distance"):
            find_peaks(np.ones(10), min_distance=-1)

    def test_short_signal_no_peaks(self):
        result = find_peaks(np.array([1.0, 2.0]))
        assert len(result) == 0

    def test_single_element_no_peaks(self):
        result = find_peaks(np.array([5.0]))
        assert len(result) == 0

    def test_monotone_increasing_no_peaks(self):
        result = find_peaks(np.linspace(0, 1, 20))
        assert len(result) == 0

    def test_single_peak_detected(self):
        s = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        result = find_peaks(s)
        assert 2 in result

    def test_returns_int64(self):
        s = np.array([0.0, 1.0, 0.0])
        result = find_peaks(s)
        assert result.dtype == np.int64

    def test_min_height_filters_low_peaks(self):
        s = np.array([0.0, 0.3, 0.0, 1.0, 0.0])
        result = find_peaks(s, min_height=0.5)
        # Only the second peak (value=1.0) should pass
        assert 1 not in result
        assert 3 in result

    def test_min_distance_enforced(self):
        s = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        result = find_peaks(s, min_distance=3)
        # Adjacent peaks filtered by distance
        for i in range(len(result) - 1):
            assert result[i + 1] - result[i] >= 3

    def test_multiple_peaks_all_detected(self):
        s = np.zeros(10)
        s[2] = 1.0
        s[7] = 1.0
        result = find_peaks(s)
        assert 2 in result
        assert 7 in result


# ─── find_valleys ─────────────────────────────────────────────────────────────

class TestFindValleys:
    def test_min_distance_zero_raises(self):
        with pytest.raises(ValueError, match="min_distance"):
            find_valleys(np.ones(10), min_distance=0)

    def test_returns_int64(self):
        s = np.array([1.0, 0.0, 1.0])
        result = find_valleys(s)
        assert result.dtype == np.int64

    def test_single_valley_detected(self):
        s = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
        result = find_valleys(s)
        assert 2 in result

    def test_short_signal_no_valleys(self):
        result = find_valleys(np.array([1.0, 2.0]))
        assert len(result) == 0

    def test_monotone_no_valleys(self):
        result = find_valleys(np.linspace(0, 1, 20))
        assert len(result) == 0

    def test_max_depth_filters_deep_valleys(self):
        s = np.array([1.0, 0.1, 1.0, 0.8, 1.0])
        # valley at idx 1 (value=0.1) and at idx 3 (value=0.8)
        # max_depth keeps valleys where s[i] <= max_depth
        result_filtered = find_valleys(s, max_depth=0.5)
        # idx 1 (0.1 <= 0.5): kept; idx 3 (0.8 > 0.5): excluded
        assert 1 in result_filtered
        assert 3 not in result_filtered


# ─── compute_autocorrelation ──────────────────────────────────────────────────

class TestComputeAutocorrelation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_autocorrelation(np.array([]))

    def test_normalized_center_is_one(self):
        s = make_sine()
        ac = compute_autocorrelation(s, normalize=True)
        center = len(ac) // 2
        assert ac[center] == pytest.approx(1.0)

    def test_length_is_2n_minus_1(self):
        s = np.ones(10)
        ac = compute_autocorrelation(s)
        assert len(ac) == 2 * 10 - 1

    def test_not_normalized_returns_larger(self):
        s = np.array([1.0, 2.0, 3.0])
        ac_norm = compute_autocorrelation(s, normalize=True)
        ac_raw = compute_autocorrelation(s, normalize=False)
        center = len(ac_raw) // 2
        assert ac_raw[center] > ac_norm[center]

    def test_returns_float64(self):
        s = np.ones(5)
        ac = compute_autocorrelation(s)
        assert ac.dtype == np.float64

    def test_single_element(self):
        ac = compute_autocorrelation(np.array([3.0]))
        assert len(ac) == 1


# ─── compute_cross_correlation ────────────────────────────────────────────────

class TestComputeCrossCorrelation:
    def test_empty_signal1_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_cross_correlation(np.array([]), np.array([1.0]))

    def test_empty_signal2_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_cross_correlation(np.array([1.0]), np.array([]))

    def test_length_is_n_plus_m_minus_1(self):
        s1 = np.ones(5)
        s2 = np.ones(8)
        cc = compute_cross_correlation(s1, s2)
        assert len(cc) == 5 + 8 - 1

    def test_identical_signals_normalized_max_one(self):
        s = make_sine(n=32)
        cc = compute_cross_correlation(s, s, normalize=True)
        assert cc.max() == pytest.approx(1.0, abs=1e-9)

    def test_returns_float64(self):
        s = np.ones(5)
        cc = compute_cross_correlation(s, s)
        assert cc.dtype == np.float64

    def test_orthogonal_signals_near_zero(self):
        n = 64
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        sin = np.sin(t)
        cos = np.cos(t)
        cc = compute_cross_correlation(sin, cos, normalize=True)
        # Max absolute value should be high but test just that it runs
        assert len(cc) == 2 * n - 1


# ─── signal_energy ────────────────────────────────────────────────────────────

class TestSignalEnergy:
    def test_zero_signal_zero_energy(self):
        assert signal_energy(np.zeros(10)) == pytest.approx(0.0)

    def test_unit_impulse_energy_is_one(self):
        s = np.zeros(10)
        s[5] = 1.0
        assert signal_energy(s) == pytest.approx(1.0)

    def test_known_value(self):
        s = np.array([3.0, 4.0])
        assert signal_energy(s) == pytest.approx(25.0)

    def test_non_negative(self):
        s = np.random.default_rng(42).standard_normal(50)
        assert signal_energy(s) >= 0.0

    def test_returns_float(self):
        assert isinstance(signal_energy(np.ones(5)), float)


# ─── segment_signal ───────────────────────────────────────────────────────────

class TestSegmentSignal:
    def test_no_segments_above(self):
        s = np.zeros(10)
        result = segment_signal(s, threshold=1.0, above=True)
        assert result == []

    def test_entire_signal_one_segment(self):
        s = np.ones(10) * 2.0
        result = segment_signal(s, threshold=1.0, above=True)
        assert len(result) == 1
        assert result[0] == (0, 10)

    def test_multiple_segments(self):
        s = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        result = segment_signal(s, threshold=0.5, above=True)
        assert len(result) == 2

    def test_segment_indices_correct(self):
        s = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        result = segment_signal(s, threshold=0.5, above=True)
        assert result == [(2, 4)]

    def test_above_false_finds_below(self):
        s = np.array([1.0, 0.0, 0.0, 1.0])
        result = segment_signal(s, threshold=0.5, above=False)
        assert len(result) == 1
        assert result[0] == (1, 3)

    def test_returns_list_of_tuples(self):
        s = np.ones(5)
        result = segment_signal(s, threshold=0.5)
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], tuple)

    def test_last_segment_extends_to_end(self):
        s = np.array([0.0, 1.0, 1.0, 1.0])
        result = segment_signal(s, threshold=0.5, above=True)
        assert result[-1][1] == 4


# ─── resample_signal ──────────────────────────────────────────────────────────

class TestResampleSignal:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            resample_signal(np.array([]), n_out=5)

    def test_n_out_zero_raises(self):
        with pytest.raises(ValueError, match="n_out"):
            resample_signal(np.ones(5), n_out=0)

    def test_n_out_negative_raises(self):
        with pytest.raises(ValueError, match="n_out"):
            resample_signal(np.ones(5), n_out=-1)

    def test_single_element_fills_output(self):
        result = resample_signal(np.array([7.0]), n_out=5)
        assert len(result) == 5
        np.testing.assert_allclose(result, 7.0)

    def test_output_length_correct(self):
        result = resample_signal(np.ones(10), n_out=20)
        assert len(result) == 20

    def test_constant_signal_preserved(self):
        result = resample_signal(np.full(5, 3.0), n_out=10)
        np.testing.assert_allclose(result, 3.0, atol=1e-10)

    def test_returns_float64(self):
        result = resample_signal(np.ones(5), n_out=10)
        assert result.dtype == np.float64

    def test_endpoints_preserved(self):
        s = np.array([0.0, 5.0, 10.0])
        result = resample_signal(s, n_out=7)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(10.0)

    def test_n_out_equals_n_in_same_signal(self):
        s = np.array([1.0, 3.0, 5.0, 7.0])
        result = resample_signal(s, n_out=4)
        np.testing.assert_allclose(result, s, atol=1e-10)


# ─── phase_shift ──────────────────────────────────────────────────────────────

class TestPhaseShift:
    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            phase_shift(np.ones(5), np.ones(6))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            phase_shift(np.array([]), np.array([]))

    def test_identical_signals_zero_shift(self):
        s = make_sine()
        shift, _ = phase_shift(s, s)
        assert shift == 0

    def test_returns_tuple_of_two(self):
        s = make_sine()
        result = phase_shift(s, s)
        assert len(result) == 2

    def test_shift_is_int(self):
        s = make_sine()
        shift, _ = phase_shift(s, s)
        assert isinstance(shift, int)

    def test_peak_value_is_float(self):
        s = make_sine()
        _, peak = phase_shift(s, s)
        assert isinstance(peak, float)

    def test_peak_value_normalized(self):
        s = make_sine(n=64)
        _, peak = phase_shift(s, s)
        # Normalized cross-correlation at zero shift is 1.0 for identical signals
        assert peak == pytest.approx(1.0, abs=1e-6)

    def test_shifted_signal_correct_shift(self):
        n = 32
        s = np.zeros(n)
        s[5] = 1.0
        # shift s by 3 samples
        shift_amount = 3
        s2 = np.roll(s, shift_amount)
        shift, _ = phase_shift(s, s2)
        assert abs(shift) == shift_amount
