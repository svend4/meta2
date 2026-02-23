"""Extra tests for puzzle_reconstruction.preprocessing.channel_splitter."""
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


# ─── TestChannelStatsExtra ──────────────────────────────────────────────────

class TestChannelStatsExtra:
    def test_mean_stored(self):
        cs = ChannelStats(mean=100.0, std=20.0, min_val=0.0,
                          max_val=255.0, median=110.0)
        assert cs.mean == pytest.approx(100.0)

    def test_std_stored(self):
        cs = ChannelStats(mean=100.0, std=25.5, min_val=0.0,
                          max_val=255.0, median=100.0)
        assert cs.std == pytest.approx(25.5)

    def test_median_stored(self):
        cs = ChannelStats(mean=100.0, std=20.0, min_val=0.0,
                          max_val=255.0, median=99.0)
        assert cs.median == pytest.approx(99.0)

    def test_min_max_stored(self):
        cs = ChannelStats(mean=128.0, std=30.0, min_val=10.0,
                          max_val=245.0, median=130.0)
        assert cs.min_val == pytest.approx(10.0)
        assert cs.max_val == pytest.approx(245.0)

    def test_zero_std(self):
        cs = ChannelStats(mean=128.0, std=0.0, min_val=128.0,
                          max_val=128.0, median=128.0)
        assert cs.std == pytest.approx(0.0)


# ─── TestSplitChannelsExtra ─────────────────────────────────────────────────

class TestSplitChannelsExtra:
    def test_channel_dtype(self):
        for ch in split_channels(_bgr()):
            assert ch.dtype == np.uint8

    def test_four_channel_image(self):
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        channels = split_channels(img)
        assert len(channels) == 4

    def test_gray_shape(self):
        ch = split_channels(_gray())[0]
        assert ch.shape == (32, 32)

    def test_values_match_original(self):
        img = _bgr(8, 8)
        channels = split_channels(img)
        np.testing.assert_array_equal(channels[0], img[:, :, 0])
        np.testing.assert_array_equal(channels[1], img[:, :, 1])
        np.testing.assert_array_equal(channels[2], img[:, :, 2])

    def test_small_image(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        channels = split_channels(img)
        assert len(channels) == 3

    def test_rectangular_image(self):
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        channels = split_channels(img)
        for ch in channels:
            assert ch.shape == (8, 16)


# ─── TestMergeChannelsExtra ─────────────────────────────────────────────────

class TestMergeChannelsExtra:
    def test_two_channels(self):
        channels = [_gray(seed=i) for i in range(2)]
        result = merge_channels(channels)
        assert result.ndim == 3
        assert result.shape[2] == 2

    def test_four_channels(self):
        channels = [_gray(seed=i) for i in range(4)]
        result = merge_channels(channels)
        assert result.shape[2] == 4

    def test_dtype_preserved(self):
        channels = [_gray(seed=i) for i in range(3)]
        result = merge_channels(channels)
        assert result.dtype == np.uint8

    def test_single_channel_shape(self):
        result = merge_channels([_gray(16, 24)])
        assert result.shape == (16, 24)

    def test_values_match(self):
        c1 = np.full((4, 4), 50, dtype=np.uint8)
        c2 = np.full((4, 4), 150, dtype=np.uint8)
        result = merge_channels([c1, c2])
        assert result[0, 0, 0] == 50
        assert result[0, 0, 1] == 150


# ─── TestChannelStatisticsExtra ─────────────────────────────────────────────

class TestChannelStatisticsExtra:
    def test_binary_channel(self):
        ch = np.array([[0, 0], [255, 255]], dtype=np.uint8)
        stats = channel_statistics(ch)
        assert stats.mean == pytest.approx(127.5)
        assert stats.min_val == pytest.approx(0.0)
        assert stats.max_val == pytest.approx(255.0)

    def test_all_zeros(self):
        ch = np.zeros((16, 16), dtype=np.uint8)
        stats = channel_statistics(ch)
        assert stats.mean == pytest.approx(0.0)
        assert stats.std == pytest.approx(0.0)
        assert stats.median == pytest.approx(0.0)

    def test_all_255(self):
        ch = np.full((16, 16), 255, dtype=np.uint8)
        stats = channel_statistics(ch)
        assert stats.mean == pytest.approx(255.0)
        assert stats.std == pytest.approx(0.0)

    def test_large_channel(self):
        ch = _gray(256, 256)
        stats = channel_statistics(ch)
        assert 0.0 <= stats.mean <= 255.0
        assert stats.std >= 0.0

    def test_min_le_median_le_max(self):
        stats = channel_statistics(_gray())
        assert stats.min_val <= stats.median <= stats.max_val

    def test_single_pixel(self):
        ch = np.array([[42]], dtype=np.uint8)
        stats = channel_statistics(ch)
        assert stats.mean == pytest.approx(42.0)
        assert stats.median == pytest.approx(42.0)
        assert stats.std == pytest.approx(0.0)


# ─── TestEqualizeChannelExtra ───────────────────────────────────────────────

class TestEqualizeChannelExtra:
    def test_all_zeros(self):
        ch = np.zeros((16, 16), dtype=np.uint8)
        result = equalize_channel(ch)
        assert result.dtype == np.uint8

    def test_all_255(self):
        ch = np.full((16, 16), 255, dtype=np.uint8)
        result = equalize_channel(ch)
        assert result.dtype == np.uint8

    def test_small_image(self):
        ch = np.array([[0, 128], [64, 255]], dtype=np.uint8)
        result = equalize_channel(ch)
        assert result.shape == (2, 2)

    def test_large_image(self):
        ch = _gray(128, 128)
        result = equalize_channel(ch)
        assert result.shape == (128, 128)

    def test_values_span_range(self):
        ch = _gray(64, 64)
        result = equalize_channel(ch)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_returns_ndarray(self):
        result = equalize_channel(_gray())
        assert isinstance(result, np.ndarray)


# ─── TestNormalizeChannelExtra ──────────────────────────────────────────────

class TestNormalizeChannelExtra:
    def test_all_zeros(self):
        ch = np.zeros((16, 16), dtype=np.uint8)
        result = normalize_channel(ch)
        assert result.dtype == np.float64
        assert np.all(result == pytest.approx(0.0))

    def test_all_255(self):
        ch = np.full((16, 16), 255, dtype=np.uint8)
        result = normalize_channel(ch)
        assert np.all(result == pytest.approx(0.0))

    def test_binary_channel(self):
        ch = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = normalize_channel(ch)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_custom_range_2_to_5(self):
        ch = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize_channel(ch, out_min=2.0, out_max=5.0)
        assert float(result.min()) >= 2.0 - 1e-9
        assert float(result.max()) <= 5.0 + 1e-9

    def test_shape_preserved(self):
        result = normalize_channel(_gray(12, 20))
        assert result.shape == (12, 20)

    def test_monotone_preserved(self):
        ch = np.array([[0, 50, 100, 150, 200, 250]], dtype=np.uint8)
        result = normalize_channel(ch)
        vals = result.flatten()
        assert list(vals) == sorted(vals)


# ─── TestChannelDifferenceExtra ─────────────────────────────────────────────

class TestChannelDifferenceExtra:
    def test_symmetric(self):
        c1 = _gray(16, 16, seed=0)
        c2 = _gray(16, 16, seed=1)
        d1 = channel_difference(c1, c2)
        d2 = channel_difference(c2, c1)
        np.testing.assert_array_almost_equal(d1, d2)

    def test_max_difference(self):
        c1 = np.zeros((4, 4), dtype=np.uint8)
        c2 = np.full((4, 4), 255, dtype=np.uint8)
        result = channel_difference(c1, c2)
        assert np.all(result == pytest.approx(255.0))

    def test_dtype_float64(self):
        result = channel_difference(_gray(seed=0), _gray(seed=1))
        assert result.dtype == np.float64

    def test_single_pixel(self):
        c1 = np.array([[10]], dtype=np.uint8)
        c2 = np.array([[20]], dtype=np.uint8)
        result = channel_difference(c1, c2)
        assert float(result[0, 0]) == pytest.approx(10.0)

    def test_large_images(self):
        c1 = _gray(128, 128, seed=0)
        c2 = _gray(128, 128, seed=1)
        result = channel_difference(c1, c2)
        assert result.shape == (128, 128)


# ─── TestApplyPerChannelExtra ───────────────────────────────────────────────

class TestApplyPerChannelExtra:
    def test_identity_gray(self):
        img = _gray()
        result = apply_per_channel(img, lambda c: c)
        np.testing.assert_array_equal(result, img)

    def test_identity_bgr(self):
        img = _bgr()
        result = apply_per_channel(img, lambda c: c)
        np.testing.assert_array_equal(result, img)

    def test_double_values(self):
        img = np.full((8, 8, 3), 10, dtype=np.uint8)
        result = apply_per_channel(img, lambda c: (c * 2).astype(np.uint8))
        assert np.all(result == 20)

    def test_normalize_per_channel(self):
        img = _bgr(16, 16)
        result = apply_per_channel(img, normalize_channel)
        assert result.dtype == np.float64

    def test_returns_ndarray(self):
        result = apply_per_channel(_bgr(), lambda c: c)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved_gray(self):
        img = _gray(20, 30)
        result = apply_per_channel(img, lambda c: c)
        assert result.shape == (20, 30)

    def test_shape_preserved_bgr(self):
        img = _bgr(20, 30)
        result = apply_per_channel(img, lambda c: c)
        assert result.shape == (20, 30, 3)


# ─── TestBatchSplitExtra ────────────────────────────────────────────────────

class TestBatchSplitExtra:
    def test_single_gray(self):
        result = batch_split([_gray()])
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_single_bgr(self):
        result = batch_split([_bgr()])
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_mixed_types(self):
        images = [_gray(), _bgr(), _gray(seed=2)]
        result = batch_split(images)
        assert len(result[0]) == 1
        assert len(result[1]) == 3
        assert len(result[2]) == 1

    def test_all_channels_correct_shape(self):
        result = batch_split([_bgr(16, 24)])
        for ch in result[0]:
            assert ch.shape == (16, 24)

    def test_multiple_bgr(self):
        images = [_bgr(seed=i) for i in range(5)]
        result = batch_split(images)
        assert len(result) == 5
        for r in result:
            assert len(r) == 3

    def test_channels_are_ndarray(self):
        result = batch_split([_bgr()])
        for ch in result[0]:
            assert isinstance(ch, np.ndarray)

    def test_channel_dtype(self):
        result = batch_split([_bgr()])
        for ch in result[0]:
            assert ch.dtype == np.uint8
