"""Расширенные тесты для puzzle_reconstruction/utils/window_utils.py."""
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


# ─── TestWindowConfig ─────────────────────────────────────────────────────────

class TestWindowConfig:
    def test_defaults(self):
        c = WindowConfig()
        assert c.size == 8
        assert c.step == 1
        assert c.func == "rect"
        assert c.padding == "same"

    def test_size_zero_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(size=0)

    def test_size_negative_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(size=-3)

    def test_step_zero_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(step=0)

    def test_step_negative_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(step=-1)

    def test_invalid_func_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(func="boxcar")

    def test_invalid_padding_raises(self):
        with pytest.raises(ValueError):
            WindowConfig(padding="circular")

    def test_all_valid_funcs(self):
        for f in ("rect", "hann", "hamming", "bartlett", "blackman"):
            c = WindowConfig(func=f)
            assert c.func == f

    def test_both_paddings(self):
        for p in ("same", "valid"):
            c = WindowConfig(padding=p)
            assert c.padding == p

    def test_size_1_ok(self):
        c = WindowConfig(size=1)
        assert c.size == 1

    def test_step_1_ok(self):
        c = WindowConfig(step=1)
        assert c.step == 1

    def test_custom_values(self):
        c = WindowConfig(size=4, step=2, func="hann", padding="valid")
        assert c.size == 4
        assert c.step == 2
        assert c.func == "hann"
        assert c.padding == "valid"


# ─── TestApplyWindowFunction ──────────────────────────────────────────────────

class TestApplyWindowFunction:
    def test_returns_ndarray(self):
        assert isinstance(apply_window_function(np.ones(8)), np.ndarray)

    def test_same_length(self):
        s = np.ones(16)
        assert len(apply_window_function(s)) == 16

    def test_float64_output(self):
        s = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        result = apply_window_function(s)
        assert result.dtype == np.float64

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            apply_window_function(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            apply_window_function(np.array([]))

    def test_rect_is_identity(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cfg = WindowConfig(func="rect")
        result = apply_window_function(s, cfg)
        assert np.allclose(result, s)

    def test_hann_reduces_edges(self):
        s = np.ones(16)
        cfg = WindowConfig(func="hann", size=16)
        result = apply_window_function(s, cfg)
        # Hann window: first element near 0
        assert result[0] < 0.2

    def test_hamming_center_approx_one(self):
        s = np.ones(16)
        cfg = WindowConfig(func="hamming", size=16)
        result = apply_window_function(s, cfg)
        mid = len(result) // 2
        assert result[mid] > 0.8

    def test_bartlett_triangular(self):
        s = np.ones(16)
        cfg = WindowConfig(func="bartlett", size=16)
        result = apply_window_function(s, cfg)
        assert result[0] < result[len(result) // 2]

    def test_blackman_edges_near_zero(self):
        s = np.ones(32)
        cfg = WindowConfig(func="blackman", size=32)
        result = apply_window_function(s, cfg)
        assert result[0] < 0.01

    def test_different_length_interpolates(self):
        s = np.ones(20)
        cfg = WindowConfig(func="hann", size=8)  # size != len(signal)
        result = apply_window_function(s, cfg)
        assert len(result) == 20

    def test_float32_accepted(self):
        s = np.ones(8, dtype=np.float32)
        result = apply_window_function(s)
        assert result.dtype == np.float64


# ─── TestRollingMean ──────────────────────────────────────────────────────────

class TestRollingMean:
    def test_returns_ndarray(self):
        assert isinstance(rolling_mean(np.ones(10)), np.ndarray)

    def test_float64(self):
        assert rolling_mean(np.ones(10)).dtype == np.float64

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            rolling_mean(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rolling_mean(np.array([]))

    def test_same_padding_length(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = np.ones(16)
        result = rolling_mean(s, cfg)
        assert len(result) == 16

    def test_constant_signal(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = _sig(20, 5.0)
        result = rolling_mean(s, cfg)
        assert np.allclose(result, 5.0, atol=1e-6)

    def test_ramp_mean(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = _ramp(20)
        result = rolling_mean(s, cfg)
        assert result.dtype == np.float64

    def test_size_1_returns_same(self):
        cfg = WindowConfig(size=1, step=1, padding="same")
        s = np.array([1.0, 2.0, 3.0, 4.0])
        result = rolling_mean(s, cfg)
        assert np.allclose(result, s)

    def test_step_larger_than_size(self):
        cfg = WindowConfig(size=3, step=5, padding="valid")
        s = np.arange(20, dtype=float)
        result = rolling_mean(s, cfg)
        assert result.dtype == np.float64


# ─── TestRollingStd ───────────────────────────────────────────────────────────

class TestRollingStd:
    def test_returns_ndarray(self):
        assert isinstance(rolling_std(np.ones(10)), np.ndarray)

    def test_float64(self):
        assert rolling_std(np.ones(10)).dtype == np.float64

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            rolling_std(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rolling_std(np.array([]))

    def test_constant_near_zero(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = _sig(20, 3.0)
        result = rolling_std(s, cfg)
        # Constant signal → std near 0 (except edge effects)
        assert result.mean() < 0.5

    def test_same_padding_length(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = _sig(16)
        result = rolling_std(s, cfg)
        assert len(result) == 16

    def test_nonneg(self):
        s = np.random.randn(20)
        result = rolling_std(s)
        assert np.all(result >= 0.0)


# ─── TestRollingMax ───────────────────────────────────────────────────────────

class TestRollingMax:
    def test_returns_ndarray(self):
        assert isinstance(rolling_max(np.ones(10)), np.ndarray)

    def test_float64(self):
        assert rolling_max(np.ones(10)).dtype == np.float64

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            rolling_max(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rolling_max(np.array([]))

    def test_constant_signal(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = _sig(20, 7.0)
        result = rolling_max(s, cfg)
        assert np.allclose(result, 7.0, atol=1e-6)

    def test_max_ge_mean(self):
        s = np.arange(20, dtype=float)
        rm = rolling_mean(s)
        rmx = rolling_max(s)
        assert np.all(rmx >= rm - 1e-9)

    def test_same_padding_length(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        assert len(rolling_max(_sig(16), cfg)) == 16


# ─── TestRollingMin ───────────────────────────────────────────────────────────

class TestRollingMin:
    def test_returns_ndarray(self):
        assert isinstance(rolling_min(np.ones(10)), np.ndarray)

    def test_float64(self):
        assert rolling_min(np.ones(10)).dtype == np.float64

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            rolling_min(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rolling_min(np.array([]))

    def test_constant_signal(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        s = _sig(20, 2.0)
        result = rolling_min(s, cfg)
        assert np.allclose(result, 2.0, atol=1e-6)

    def test_min_le_mean(self):
        s = np.arange(20, dtype=float)
        rm = rolling_mean(s)
        rmn = rolling_min(s)
        assert np.all(rmn <= rm + 1e-9)

    def test_same_padding_length(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        assert len(rolling_min(_sig(16), cfg)) == 16


# ─── TestComputeOverlap ───────────────────────────────────────────────────────

class TestComputeOverlap:
    def test_returns_float(self):
        assert isinstance(compute_overlap(0, 5, 3, 8), float)

    def test_in_0_1(self):
        assert 0.0 <= compute_overlap(0, 5, 3, 8) <= 1.0

    def test_identical_windows(self):
        assert compute_overlap(0, 10, 0, 10) == pytest.approx(1.0)

    def test_non_overlapping(self):
        assert compute_overlap(0, 5, 6, 10) == pytest.approx(0.0)

    def test_partial_overlap(self):
        val = compute_overlap(0, 6, 3, 9)
        assert 0.0 < val < 1.0

    def test_a_start_ge_end_raises(self):
        with pytest.raises(ValueError):
            compute_overlap(5, 5, 0, 10)

    def test_a_start_gt_end_raises(self):
        with pytest.raises(ValueError):
            compute_overlap(7, 3, 0, 10)

    def test_b_start_ge_end_raises(self):
        with pytest.raises(ValueError):
            compute_overlap(0, 10, 5, 5)

    def test_contained_window(self):
        # [2, 8] fully inside [0, 10]
        val = compute_overlap(0, 10, 2, 8)
        assert val > 0.5

    def test_adjacent_non_overlapping(self):
        # [0, 5) and [5, 10) share no overlap
        assert compute_overlap(0, 5, 5, 10) == pytest.approx(0.0)

    def test_symmetry(self):
        a = compute_overlap(0, 5, 3, 8)
        b = compute_overlap(3, 8, 0, 5)
        assert a == pytest.approx(b)


# ─── TestSplitIntoWindows ─────────────────────────────────────────────────────

class TestSplitIntoWindows:
    def test_returns_list(self):
        result = split_into_windows(np.ones(16))
        assert isinstance(result, list)

    def test_nonempty_for_nonempty_signal(self):
        result = split_into_windows(np.ones(16))
        assert len(result) > 0

    def test_each_window_ndarray(self):
        for w in split_into_windows(np.ones(16)):
            assert isinstance(w, np.ndarray)

    def test_each_window_size(self):
        cfg = WindowConfig(size=4, step=2, padding="same")
        for w in split_into_windows(np.ones(16), cfg):
            assert len(w) == 4

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            split_into_windows(np.ones((4, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            split_into_windows(np.array([]))

    def test_valid_fewer_than_same(self):
        s = np.ones(20)
        cfg_same  = WindowConfig(size=4, step=2, padding="same")
        cfg_valid = WindowConfig(size=4, step=2, padding="valid")
        assert len(split_into_windows(s, cfg_same)) >= len(split_into_windows(s, cfg_valid))

    def test_float64_windows(self):
        for w in split_into_windows(_sig(16)):
            assert w.dtype == np.float64

    def test_step_1_many_windows(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = split_into_windows(np.ones(16), cfg)
        assert len(result) >= 16


# ─── TestMergeWindows ─────────────────────────────────────────────────────────

class TestMergeWindows:
    def _cfg(self):
        return WindowConfig(size=4, step=1, padding="same")

    def _make_windows(self, n: int = 4, size: int = 4):
        return [np.ones(size) for _ in range(n)]

    def test_returns_ndarray(self):
        cfg = self._cfg()
        result = merge_windows(self._make_windows(), 16, cfg)
        assert isinstance(result, np.ndarray)

    def test_output_length(self):
        cfg = self._cfg()
        result = merge_windows(self._make_windows(4, 4), 16, cfg)
        assert len(result) == 16

    def test_float64(self):
        cfg = self._cfg()
        result = merge_windows(self._make_windows(), 16, cfg)
        assert result.dtype == np.float64

    def test_empty_windows_raises(self):
        with pytest.raises(ValueError):
            merge_windows([], 10)

    def test_original_length_zero_raises(self):
        with pytest.raises(ValueError):
            merge_windows(self._make_windows(), 0, self._cfg())

    def test_original_length_negative_raises(self):
        with pytest.raises(ValueError):
            merge_windows(self._make_windows(), -5, self._cfg())

    def test_constant_windows_recover_constant(self):
        cfg = WindowConfig(size=4, step=1, padding="same")
        signal = np.ones(16)
        windows = split_into_windows(signal, cfg)
        result = merge_windows(windows, 16, cfg)
        assert np.allclose(result, 1.0, atol=0.01)


# ─── TestBatchRolling ─────────────────────────────────────────────────────────

class TestBatchRolling:
    def test_returns_list(self):
        assert isinstance(batch_rolling([np.ones(10)]), list)

    def test_length_matches(self):
        sigs = [np.ones(10), np.ones(12), np.ones(8)]
        assert len(batch_rolling(sigs)) == 3

    def test_each_ndarray(self):
        for r in batch_rolling([np.ones(10), np.ones(8)]):
            assert isinstance(r, np.ndarray)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_rolling([])

    def test_invalid_stat_raises(self):
        with pytest.raises(ValueError):
            batch_rolling([np.ones(10)], stat="median")

    def test_stat_mean(self):
        result = batch_rolling([np.ones(10)], stat="mean")
        assert len(result) == 1

    def test_stat_std(self):
        result = batch_rolling([np.ones(10)], stat="std")
        assert result[0].dtype == np.float64

    def test_stat_max(self):
        result = batch_rolling([np.arange(10, dtype=float)], stat="max")
        assert isinstance(result[0], np.ndarray)

    def test_stat_min(self):
        result = batch_rolling([np.arange(10, dtype=float)], stat="min")
        assert isinstance(result[0], np.ndarray)

    def test_cfg_passed_through(self):
        cfg = WindowConfig(size=3, step=1, padding="same")
        s = _sig(20, 5.0)
        result = batch_rolling([s], stat="mean", cfg=cfg)
        assert np.allclose(result[0], 5.0, atol=1e-6)

    def test_all_float64(self):
        sigs = [np.ones(10), np.arange(8, dtype=np.int32)]
        for r in batch_rolling(sigs, stat="mean"):
            assert r.dtype == np.float64
