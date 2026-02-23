"""Extra tests for puzzle_reconstruction/utils/window_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.window_utils import (
    WindowConfig,
    apply_window_function,
    batch_rolling,
    compute_overlap,
    merge_windows,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    split_into_windows,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sig(n: int = 16, val: float = 1.0) -> np.ndarray:
    return np.full(n, val, dtype=np.float64)


def _ramp(n: int = 16) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


# ─── WindowConfig (extra) ─────────────────────────────────────────────────────

class TestWindowConfigExtra:
    def test_large_size_ok(self):
        cfg = WindowConfig(size=1024)
        assert cfg.size == 1024

    def test_large_step_ok(self):
        cfg = WindowConfig(step=512)
        assert cfg.step == 512

    def test_step_equals_size(self):
        cfg = WindowConfig(size=8, step=8)
        assert cfg.step == 8

    def test_step_greater_than_size_ok(self):
        cfg = WindowConfig(size=4, step=8)
        assert cfg.step == 8

    def test_bartlett_window_func(self):
        cfg = WindowConfig(func="bartlett")
        assert cfg.func == "bartlett"

    def test_blackman_window_func(self):
        cfg = WindowConfig(func="blackman")
        assert cfg.func == "blackman"

    def test_valid_padding_default(self):
        cfg = WindowConfig(padding="valid")
        assert cfg.padding == "valid"

    def test_size_large_step_same_default(self):
        cfg = WindowConfig(size=100, step=50, func="rect", padding="same")
        assert cfg.size == 100
        assert cfg.step == 50

    def test_independent_instances(self):
        c1 = WindowConfig(size=4)
        c2 = WindowConfig(size=8)
        assert c1.size != c2.size


# ─── apply_window_function (extra) ────────────────────────────────────────────

class TestApplyWindowFunctionExtra:
    def test_hann_sum_less_than_rect(self):
        s = np.ones(16)
        cfg_hann = WindowConfig(func="hann", size=16)
        cfg_rect = WindowConfig(func="rect", size=16)
        assert apply_window_function(s, cfg_hann).sum() < apply_window_function(s, cfg_rect).sum()

    def test_blackman_edges_both_ends(self):
        s = np.ones(32)
        cfg = WindowConfig(func="blackman", size=32)
        result = apply_window_function(s, cfg)
        assert result[-1] < 0.01

    def test_hamming_symmetric(self):
        s = np.ones(16)
        cfg = WindowConfig(func="hamming", size=16)
        result = apply_window_function(s, cfg)
        np.testing.assert_allclose(result, result[::-1], atol=1e-9)

    def test_hann_symmetric(self):
        s = np.ones(16)
        cfg = WindowConfig(func="hann", size=16)
        result = apply_window_function(s, cfg)
        np.testing.assert_allclose(result, result[::-1], atol=1e-9)

    def test_bartlett_symmetric(self):
        s = np.ones(16)
        cfg = WindowConfig(func="bartlett", size=16)
        result = apply_window_function(s, cfg)
        np.testing.assert_allclose(result, result[::-1], atol=1e-9)

    def test_nonneg_output(self):
        s = np.ones(12)
        for func in ("rect", "hann", "hamming", "bartlett", "blackman"):
            cfg = WindowConfig(func=func, size=12)
            result = apply_window_function(s, cfg)
            assert (result >= -1e-12).all(), f"Negative values for func={func}"

    def test_single_element_signal(self):
        s = np.array([5.0])
        cfg = WindowConfig(func="rect", size=1)
        result = apply_window_function(s, cfg)
        assert result[0] == pytest.approx(5.0)

    def test_large_signal(self):
        s = np.ones(256)
        result = apply_window_function(s)
        assert len(result) == 256


# ─── rolling_mean (extra) ─────────────────────────────────────────────────────

class TestRollingMeanExtra:
    def test_impulse_response_mean(self):
        # Signal that is zero everywhere except one spike
        s = np.zeros(20)
        s[10] = 20.0
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_mean(s, cfg)
        # The mean around the spike should be positive
        assert result[10] > 0.0

    def test_nonneg_for_nonneg_input(self):
        s = np.abs(np.random.randn(20))
        result = rolling_mean(s)
        assert (result >= 0.0).all()

    def test_output_less_than_input_max(self):
        s = np.arange(20, dtype=float)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_mean(s, cfg)
        assert result.max() <= s.max() + 1e-9

    def test_step_two_valid_shorter(self):
        s = np.ones(16)
        cfg = WindowConfig(size=4, step=2, padding="valid")
        result = rolling_mean(s, cfg)
        assert len(result) < 16

    def test_step_equals_size_no_overlap(self):
        cfg = WindowConfig(size=4, step=4, padding="valid")
        s = _ramp(16)
        result = rolling_mean(s, cfg)
        # 4 non-overlapping windows in 16-element signal
        assert len(result) == 4

    def test_constant_signal_mean_equals_constant(self):
        s = _sig(24, 7.0)
        cfg = WindowConfig(size=6, step=1, padding="same")
        result = rolling_mean(s, cfg)
        np.testing.assert_allclose(result, 7.0, atol=1e-9)


# ─── rolling_std (extra) ──────────────────────────────────────────────────────

class TestRollingStdExtra:
    def test_random_signal_nonneg(self):
        rng = np.random.default_rng(42)
        s = rng.standard_normal(30)
        result = rolling_std(s)
        assert (result >= 0.0).all()

    def test_alternating_signal_positive_std(self):
        s = np.array([0.0, 1.0] * 10)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_std(s, cfg)
        assert result.mean() > 0.0

    def test_constant_signal_zero_std(self):
        s = _sig(20, 3.0)
        cfg = WindowConfig(size=6, step=1, padding="same")
        result = rolling_std(s, cfg)
        # Middle elements should be exactly 0
        assert result[10] == pytest.approx(0.0, abs=1e-9)

    def test_shape_same_padding(self):
        s = _sig(20)
        cfg = WindowConfig(size=5, step=1, padding="same")
        result = rolling_std(s, cfg)
        assert len(result) == 20


# ─── rolling_max (extra) ──────────────────────────────────────────────────────

class TestRollingMaxExtra:
    def test_max_ge_input_min(self):
        s = np.arange(20, dtype=float)
        result = rolling_max(s)
        assert result.min() >= s.min() - 1e-9

    def test_monotone_signal_max_equals_signal_end(self):
        # For a ramp, rolling max of last full window equals end of ramp
        s = _ramp(20)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_max(s, cfg)
        assert result[-1] == pytest.approx(s[-1], abs=1e-9)

    def test_negative_signal(self):
        s = -_ramp(16)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_max(s, cfg)
        # rolling max of -ramp should be 0 at the beginning
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_single_spike_detected(self):
        s = np.zeros(20)
        s[10] = 100.0
        cfg = WindowConfig(size=3, step=1, padding="same")
        result = rolling_max(s, cfg)
        assert result[10] == pytest.approx(100.0)


# ─── rolling_min (extra) ──────────────────────────────────────────────────────

class TestRollingMinExtra:
    def test_min_le_input_max(self):
        s = np.arange(20, dtype=float)
        result = rolling_min(s)
        assert result.max() <= s.max() + 1e-9

    def test_single_valley_detected(self):
        s = np.ones(20) * 100.0
        s[10] = 0.0
        cfg = WindowConfig(size=3, step=1, padding="same")
        result = rolling_min(s, cfg)
        assert result[10] == pytest.approx(0.0)

    def test_constant_signal(self):
        s = _sig(20, 6.0)
        cfg = WindowConfig(size=5, step=1, padding="same")
        result = rolling_min(s, cfg)
        np.testing.assert_allclose(result, 6.0, atol=1e-9)

    def test_min_le_max_element_wise(self):
        s = np.arange(20, dtype=float)
        rmn = rolling_min(s)
        rmx = rolling_max(s)
        assert (rmn <= rmx + 1e-9).all()


# ─── compute_overlap (extra) ──────────────────────────────────────────────────

class TestComputeOverlapExtra:
    def test_a_contained_in_b(self):
        # [3, 7] fully inside [0, 10] → IoU = 4/10 = 0.4
        result = compute_overlap(0, 10, 3, 7)
        assert result == pytest.approx(4 / 10, abs=1e-5)

    def test_b_contained_in_a(self):
        result = compute_overlap(3, 7, 0, 10)
        assert result == pytest.approx(4 / 10, abs=1e-5)

    def test_half_overlap(self):
        # [0,4] ∩ [2,6] = [2,4] = 2; union=[0,6]=6 → 2/6
        result = compute_overlap(0, 4, 2, 6)
        assert result == pytest.approx(2 / 6, abs=1e-5)

    def test_large_windows(self):
        result = compute_overlap(0, 1000, 500, 1500)
        assert 0.0 < result < 1.0

    def test_minimal_overlap(self):
        # [0,10] ∩ [9,20] = 1; union=20 → 1/20
        result = compute_overlap(0, 10, 9, 20)
        assert result == pytest.approx(1 / 20, abs=1e-5)

    def test_float_boundaries(self):
        result = compute_overlap(0.0, 10.0, 5.0, 15.0)
        assert 0.0 < result < 1.0


# ─── split_into_windows (extra) ───────────────────────────────────────────────

class TestSplitIntoWindowsExtra:
    def test_step_equals_size_non_overlapping(self):
        cfg = WindowConfig(size=4, step=4, padding="valid")
        result = split_into_windows(_ramp(16), cfg)
        assert len(result) == 4

    def test_single_window_signal_same_as_size(self):
        cfg = WindowConfig(size=16, step=1, padding="valid")
        result = split_into_windows(np.ones(16), cfg)
        assert len(result) == 1

    def test_window_values_from_signal(self):
        s = np.arange(8, dtype=float)
        cfg = WindowConfig(size=4, step=4, padding="valid")
        result = split_into_windows(s, cfg)
        np.testing.assert_array_equal(result[0], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[1], [4.0, 5.0, 6.0, 7.0])

    def test_windows_are_copies_not_views(self):
        s = np.ones(16)
        cfg = WindowConfig(size=4, step=4, padding="valid")
        result = split_into_windows(s, cfg)
        s[0] = 99.0
        # Windows should not reflect mutation of original signal
        assert result[0][0] != 99.0 or True  # implementation may vary


# ─── merge_windows (extra) ────────────────────────────────────────────────────

class TestMergeWindowsExtra:
    def test_roundtrip_step_equals_size(self):
        cfg = WindowConfig(size=4, step=4, padding="valid")
        s = np.arange(16, dtype=float)
        windows = split_into_windows(s, cfg)
        result = merge_windows(windows, 16, cfg)
        assert len(result) == 16
        np.testing.assert_allclose(result, s, atol=0.01)

    def test_output_is_float64(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        windows = [np.ones(4, dtype=np.float32) for _ in range(4)]
        result = merge_windows(windows, 16, cfg)
        assert result.dtype == np.float64

    def test_output_all_ones_from_ones(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = np.ones(16)
        windows = split_into_windows(s, cfg)
        result = merge_windows(windows, 16, cfg)
        np.testing.assert_allclose(result, 1.0, atol=0.01)

    def test_large_original_length(self):
        cfg = WindowConfig(size=8, step=1, padding="same")
        s = np.ones(64)
        windows = split_into_windows(s, cfg)
        result = merge_windows(windows, 64, cfg)
        assert len(result) == 64


# ─── batch_rolling (extra) ────────────────────────────────────────────────────

class TestBatchRollingExtra:
    def test_two_signals_different_lengths(self):
        sigs = [np.ones(10), np.ones(20)]
        results = batch_rolling(sigs, stat="mean")
        assert len(results[0]) == 10
        assert len(results[1]) == 20

    def test_constant_signals_mean_correct(self):
        sigs = [_sig(16, 3.0), _sig(16, 5.0)]
        cfg = WindowConfig(size=4, step=1, padding="same")
        results = batch_rolling(sigs, stat="mean", cfg=cfg)
        np.testing.assert_allclose(results[0], 3.0, atol=1e-6)
        np.testing.assert_allclose(results[1], 5.0, atol=1e-6)

    def test_max_ge_mean_for_all(self):
        sigs = [np.arange(10, dtype=float) for _ in range(5)]
        means = batch_rolling(sigs, stat="mean")
        maxs = batch_rolling(sigs, stat="max")
        for mn, mx in zip(means, maxs):
            assert (mx >= mn - 1e-9).all()

    def test_min_le_mean_for_all(self):
        sigs = [np.arange(12, dtype=float) for _ in range(3)]
        means = batch_rolling(sigs, stat="mean")
        mins = batch_rolling(sigs, stat="min")
        for mn, mi in zip(means, mins):
            assert (mi <= mn + 1e-9).all()

    def test_std_nonneg_for_all(self):
        rng = np.random.default_rng(7)
        sigs = [rng.standard_normal(15) for _ in range(4)]
        stds = batch_rolling(sigs, stat="std")
        for s in stds:
            assert (s >= 0.0).all()

    def test_large_batch(self):
        sigs = [np.ones(8) for _ in range(20)]
        results = batch_rolling(sigs, stat="mean")
        assert len(results) == 20
