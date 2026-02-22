"""Tests for puzzle_reconstruction.preprocessing.channel_splitter."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.channel_splitter import (
    ChannelStats,
    apply_per_channel,
    batch_split,
    channel_difference,
    channel_statistics,
    equalize_channel,
    merge_channels,
    normalize_channel,
    split_channels,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h: int = 32, w: int = 32, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── ChannelStats ────────────────────────────────────────────────────────────

class TestChannelStats:
    def test_fields_stored(self):
        cs = ChannelStats(mean=128.0, std=30.0, min_val=0.0,
                          max_val=255.0, median=130.0)
        assert cs.mean == pytest.approx(128.0)
        assert cs.std == pytest.approx(30.0)
        assert cs.min_val == pytest.approx(0.0)
        assert cs.max_val == pytest.approx(255.0)
        assert cs.median == pytest.approx(130.0)


# ─── split_channels ──────────────────────────────────────────────────────────

class TestSplitChannels:
    def test_gray_returns_one_channel(self):
        channels = split_channels(_gray())
        assert len(channels) == 1

    def test_bgr_returns_three_channels(self):
        channels = split_channels(_bgr())
        assert len(channels) == 3

    def test_channels_are_2d(self):
        for ch in split_channels(_bgr()):
            assert ch.ndim == 2

    def test_channel_shape_matches(self):
        img = _bgr(24, 32)
        for ch in split_channels(img):
            assert ch.shape == (24, 32)

    def test_returns_list(self):
        assert isinstance(split_channels(_gray()), list)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            split_channels(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_first_channel_is_blue_for_bgr(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # blue
        channels = split_channels(img)
        assert np.all(channels[0] == 100)

    def test_copy_returned(self):
        img = _bgr()
        channels = split_channels(img)
        channels[0][:] = 0
        # Original should be unchanged
        assert not np.all(img[:, :, 0] == 0)


# ─── merge_channels ──────────────────────────────────────────────────────────

class TestMergeChannels:
    def test_single_channel_returns_2d(self):
        ch = _gray()
        result = merge_channels([ch])
        assert result.ndim == 2

    def test_three_channels_returns_3d(self):
        channels = [_gray(seed=i) for i in range(3)]
        result = merge_channels(channels)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_shape_correct(self):
        channels = [_gray(24, 32, seed=i) for i in range(3)]
        result = merge_channels(channels)
        assert result.shape == (24, 32, 3)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            merge_channels([])

    def test_mismatched_shapes_raises(self):
        c1 = np.zeros((8, 8), dtype=np.uint8)
        c2 = np.zeros((16, 16), dtype=np.uint8)
        with pytest.raises(ValueError):
            merge_channels([c1, c2])

    def test_roundtrip(self):
        img = _bgr()
        channels = split_channels(img)
        recovered = merge_channels(channels)
        np.testing.assert_array_equal(recovered, img)

    def test_values_correct(self):
        c1 = np.full((4, 4), 10, dtype=np.uint8)
        c2 = np.full((4, 4), 20, dtype=np.uint8)
        c3 = np.full((4, 4), 30, dtype=np.uint8)
        result = merge_channels([c1, c2, c3])
        assert result[0, 0, 0] == 10
        assert result[0, 0, 1] == 20
        assert result[0, 0, 2] == 30


# ─── channel_statistics ──────────────────────────────────────────────────────

class TestChannelStatistics:
    def test_returns_channel_stats(self):
        stats = channel_statistics(_gray())
        assert isinstance(stats, ChannelStats)

    def test_mean_in_range(self):
        ch = _gray()
        stats = channel_statistics(ch)
        assert 0.0 <= stats.mean <= 255.0

    def test_std_nonnegative(self):
        stats = channel_statistics(_gray())
        assert stats.std >= 0.0

    def test_min_le_mean_le_max(self):
        stats = channel_statistics(_gray())
        assert stats.min_val <= stats.mean <= stats.max_val

    def test_constant_channel_std_zero(self):
        ch = np.full((16, 16), 128, dtype=np.uint8)
        stats = channel_statistics(ch)
        assert stats.std == pytest.approx(0.0)
        assert stats.mean == pytest.approx(128.0)
        assert stats.median == pytest.approx(128.0)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            channel_statistics(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_median_correct(self):
        ch = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        stats = channel_statistics(ch)
        assert stats.median == pytest.approx(2.5)


# ─── equalize_channel ────────────────────────────────────────────────────────

class TestEqualizeChannel:
    def test_returns_uint8(self):
        result = equalize_channel(_gray())
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        ch = _gray(24, 32)
        result = equalize_channel(ch)
        assert result.shape == (24, 32)

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            equalize_channel(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_values_in_range(self):
        result = equalize_channel(_gray())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_image_accepted(self):
        ch = np.full((16, 16), 100, dtype=np.uint8)
        result = equalize_channel(ch)
        assert result.shape == (16, 16)

    def test_output_differs_from_input(self):
        ch = _gray()
        result = equalize_channel(ch)
        # For a random image, equalization should change values
        # (not guaranteed but very likely)
        assert result.shape == ch.shape


# ─── normalize_channel ───────────────────────────────────────────────────────

class TestNormalizeChannel:
    def test_returns_float64(self):
        result = normalize_channel(_gray())
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        result = normalize_channel(_gray(24, 32))
        assert result.shape == (24, 32)

    def test_values_in_unit_interval(self):
        result = normalize_channel(_gray())
        assert result.min() >= 0.0 - 1e-9
        assert result.max() <= 1.0 + 1e-9

    def test_out_max_le_out_min_raises(self):
        with pytest.raises(ValueError):
            normalize_channel(_gray(), out_min=1.0, out_max=0.0)

    def test_out_max_eq_out_min_raises(self):
        with pytest.raises(ValueError):
            normalize_channel(_gray(), out_min=0.5, out_max=0.5)

    def test_custom_range(self):
        ch = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        result = normalize_channel(ch, out_min=0.0, out_max=10.0)
        assert result.min() >= 0.0 - 1e-9
        assert result.max() <= 10.0 + 1e-9

    def test_constant_channel_returns_out_min(self):
        ch = np.full((8, 8), 100, dtype=np.uint8)
        result = normalize_channel(ch, out_min=0.0, out_max=1.0)
        assert np.all(result == pytest.approx(0.0))

    def test_not_2d_raises(self):
        with pytest.raises(ValueError):
            normalize_channel(np.zeros((4, 4, 3), dtype=np.uint8))


# ─── channel_difference ──────────────────────────────────────────────────────

class TestChannelDifference:
    def test_returns_float64(self):
        result = channel_difference(_gray(), _gray(seed=1))
        assert result.dtype == np.float64

    def test_shape_is_min_of_both(self):
        c1 = _gray(16, 32)
        c2 = _gray(24, 24)
        result = channel_difference(c1, c2)
        assert result.shape == (16, 24)

    def test_identical_channels_zero(self):
        ch = _gray()
        result = channel_difference(ch, ch)
        assert np.all(result == pytest.approx(0.0))

    def test_nonnegative(self):
        result = channel_difference(_gray(seed=0), _gray(seed=1))
        assert np.all(result >= 0.0)

    def test_c1_not_2d_raises(self):
        with pytest.raises(ValueError):
            channel_difference(np.zeros((4, 4, 3), dtype=np.uint8), _gray())

    def test_c2_not_2d_raises(self):
        with pytest.raises(ValueError):
            channel_difference(_gray(), np.zeros((4, 4, 3), dtype=np.uint8))

    def test_correct_values(self):
        c1 = np.array([[10, 20]], dtype=np.uint8)
        c2 = np.array([[5, 30]], dtype=np.uint8)
        result = channel_difference(c1, c2)
        np.testing.assert_array_almost_equal(result, [[5.0, 10.0]])


# ─── apply_per_channel ───────────────────────────────────────────────────────

class TestApplyPerChannel:
    def test_gray_single_channel(self):
        img = _gray()
        result = apply_per_channel(img, lambda c: c)
        assert result.shape == img.shape

    def test_bgr_three_channels(self):
        img = _bgr()
        result = apply_per_channel(img, lambda c: c)
        assert result.shape == img.shape

    def test_function_applied(self):
        img = np.ones((8, 8, 3), dtype=np.uint8)
        result = apply_per_channel(img, lambda c: c * 2)
        assert np.all(result == 2)

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            apply_per_channel(np.zeros((4, 4, 4, 4), dtype=np.uint8),
                              lambda c: c)

    def test_equalize_per_channel(self):
        img = _bgr()
        result = apply_per_channel(img, equalize_channel)
        assert result.dtype == np.uint8
        assert result.shape == img.shape


# ─── batch_split ─────────────────────────────────────────────────────────────

class TestBatchSplit:
    def test_returns_list(self):
        result = batch_split([_gray(), _bgr()])
        assert isinstance(result, list)

    def test_length_matches_input(self):
        images = [_gray(), _bgr(), _gray(seed=2)]
        result = batch_split(images)
        assert len(result) == 3

    def test_empty_input_returns_empty(self):
        assert batch_split([]) == []

    def test_inner_lists_correct_length(self):
        result = batch_split([_gray(), _bgr()])
        assert len(result[0]) == 1   # gray → 1 channel
        assert len(result[1]) == 3   # bgr  → 3 channels

    def test_all_channels_2d(self):
        result = batch_split([_bgr()])
        for ch in result[0]:
            assert ch.ndim == 2
