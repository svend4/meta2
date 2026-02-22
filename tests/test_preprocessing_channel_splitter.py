"""Tests for puzzle_reconstruction/preprocessing/channel_splitter.py"""
import pytest
import numpy as np

from puzzle_reconstruction.preprocessing.channel_splitter import (
    ChannelStats,
    split_channels,
    merge_channels,
    channel_statistics,
    equalize_channel,
    normalize_channel,
    channel_difference,
    apply_per_channel,
    batch_split,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=10, w=10, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=10, w=10):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 50
    img[:, :, 1] = 100
    img[:, :, 2] = 200
    return img


# ─── TestChannelStats ─────────────────────────────────────────────────────────

class TestChannelStats:
    def test_construction(self):
        stats = ChannelStats(mean=100.0, std=20.0, min_val=0.0, max_val=255.0, median=100.0)
        assert stats.mean == 100.0
        assert stats.std == 20.0
        assert stats.min_val == 0.0
        assert stats.max_val == 255.0
        assert stats.median == 100.0


# ─── TestSplitChannels ────────────────────────────────────────────────────────

class TestSplitChannels:
    def test_gray_returns_single_channel(self):
        channels = split_channels(_gray())
        assert len(channels) == 1

    def test_gray_channel_is_2d(self):
        channels = split_channels(_gray())
        assert channels[0].ndim == 2

    def test_bgr_returns_three_channels(self):
        channels = split_channels(_bgr())
        assert len(channels) == 3

    def test_each_channel_is_2d(self):
        for c in split_channels(_bgr()):
            assert c.ndim == 2

    def test_channel_values_correct(self):
        img = _bgr()
        channels = split_channels(img)
        assert (channels[0] == 50).all()
        assert (channels[1] == 100).all()
        assert (channels[2] == 200).all()

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            split_channels(np.zeros((5, 5, 3, 1), dtype=np.uint8))

    def test_returns_copy(self):
        img = _gray()
        channels = split_channels(img)
        channels[0][:] = 0
        assert (img != 0).any()


# ─── TestMergeChannels ────────────────────────────────────────────────────────

class TestMergeChannels:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_channels([])

    def test_single_channel_returns_2d(self):
        result = merge_channels([_gray()])
        assert result.ndim == 2

    def test_three_channels_returns_3d(self):
        c = _gray()
        result = merge_channels([c, c, c])
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_shape_mismatch_raises(self):
        c1 = _gray(h=5, w=5)
        c2 = _gray(h=6, w=5)
        with pytest.raises(ValueError):
            merge_channels([c1, c2])

    def test_roundtrip(self):
        img = _bgr()
        channels = split_channels(img)
        rebuilt = merge_channels(channels)
        np.testing.assert_array_equal(rebuilt, img)

    def test_values_preserved(self):
        c0 = np.full((4, 4), 10, dtype=np.uint8)
        c1 = np.full((4, 4), 20, dtype=np.uint8)
        result = merge_channels([c0, c1])
        assert (result[:, :, 0] == 10).all()
        assert (result[:, :, 1] == 20).all()


# ─── TestChannelStatistics ────────────────────────────────────────────────────

class TestChannelStatistics:
    def test_returns_channel_stats(self):
        stats = channel_statistics(_gray())
        assert isinstance(stats, ChannelStats)

    def test_mean_correct(self):
        c = np.array([[0, 100], [200, 100]], dtype=np.uint8)
        stats = channel_statistics(c)
        assert stats.mean == pytest.approx(100.0)

    def test_constant_channel_zero_std(self):
        stats = channel_statistics(_gray(val=50))
        assert stats.std == pytest.approx(0.0)

    def test_min_max_correct(self):
        c = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        stats = channel_statistics(c)
        assert stats.min_val == 0.0
        assert stats.max_val == 255.0

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            channel_statistics(_bgr())


# ─── TestEqualizeChannel ──────────────────────────────────────────────────────

class TestEqualizeChannel:
    def test_returns_ndarray(self):
        result = equalize_channel(_gray())
        assert isinstance(result, np.ndarray)

    def test_output_shape_preserved(self):
        c = _gray(h=8, w=12)
        result = equalize_channel(c)
        assert result.shape == (8, 12)

    def test_output_dtype_uint8(self):
        result = equalize_channel(_gray())
        assert result.dtype == np.uint8

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            equalize_channel(_bgr())

    def test_constant_channel(self):
        # equalizing a constant channel should still return uint8 array
        result = equalize_channel(_gray(val=100))
        assert result.dtype == np.uint8
        assert result.shape == (10, 10)


# ─── TestNormalizeChannel ─────────────────────────────────────────────────────

class TestNormalizeChannel:
    def test_returns_float64(self):
        result = normalize_channel(_gray())
        assert result.dtype == np.float64

    def test_output_shape_preserved(self):
        result = normalize_channel(_gray(h=5, w=7))
        assert result.shape == (5, 7)

    def test_range_0_1(self):
        c = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize_channel(c)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_custom_range(self):
        c = np.array([[0, 255]], dtype=np.uint8)
        result = normalize_channel(c, out_min=-1.0, out_max=1.0)
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_out_max_le_out_min_raises(self):
        with pytest.raises(ValueError):
            normalize_channel(_gray(), out_min=1.0, out_max=0.0)

    def test_out_max_equal_out_min_raises(self):
        with pytest.raises(ValueError):
            normalize_channel(_gray(), out_min=0.5, out_max=0.5)

    def test_constant_channel_returns_out_min(self):
        c = _gray(val=128)
        result = normalize_channel(c, out_min=0.0, out_max=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            normalize_channel(_bgr())


# ─── TestChannelDifference ────────────────────────────────────────────────────

class TestChannelDifference:
    def test_returns_float64(self):
        result = channel_difference(_gray(val=100), _gray(val=50))
        assert result.dtype == np.float64

    def test_same_channel_zero_diff(self):
        c = _gray(val=100)
        result = channel_difference(c, c)
        assert (result == 0.0).all()

    def test_diff_value_correct(self):
        c1 = np.array([[100, 200]], dtype=np.uint8)
        c2 = np.array([[50, 150]], dtype=np.uint8)
        result = channel_difference(c1, c2)
        np.testing.assert_allclose(result, [[50.0, 50.0]])

    def test_clips_to_min_size(self):
        c1 = _gray(h=5, w=8)
        c2 = _gray(h=3, w=6)
        result = channel_difference(c1, c2)
        assert result.shape == (3, 6)

    def test_c1_3d_raises(self):
        with pytest.raises(ValueError):
            channel_difference(_bgr(), _gray())

    def test_c2_3d_raises(self):
        with pytest.raises(ValueError):
            channel_difference(_gray(), _bgr())

    def test_non_negative_result(self):
        c1 = np.array([[50, 200]], dtype=np.uint8)
        c2 = np.array([[200, 50]], dtype=np.uint8)
        result = channel_difference(c1, c2)
        assert (result >= 0).all()


# ─── TestApplyPerChannel ──────────────────────────────────────────────────────

class TestApplyPerChannel:
    def test_gray_returns_2d(self):
        result = apply_per_channel(_gray(), lambda c: c)
        assert result.ndim == 2

    def test_bgr_returns_3d(self):
        result = apply_per_channel(_bgr(), lambda c: c)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_identity_preserves_values(self):
        img = _bgr()
        result = apply_per_channel(img, lambda c: c)
        np.testing.assert_array_equal(result, img)

    def test_func_applied_to_each_channel(self):
        img = _bgr()
        result = apply_per_channel(img, lambda c: np.zeros_like(c))
        assert (result == 0).all()

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            apply_per_channel(np.zeros((5, 5, 3, 1), dtype=np.uint8), lambda c: c)


# ─── TestBatchSplit ───────────────────────────────────────────────────────────

class TestBatchSplit:
    def test_returns_list_of_lists(self):
        result = batch_split([_gray(), _bgr()])
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_length_matches(self):
        images = [_gray(), _bgr(), _gray()]
        result = batch_split(images)
        assert len(result) == 3

    def test_gray_gives_one_channel(self):
        result = batch_split([_gray()])
        assert len(result[0]) == 1

    def test_bgr_gives_three_channels(self):
        result = batch_split([_bgr()])
        assert len(result[0]) == 3

    def test_all_channels_2d(self):
        for channels in batch_split([_gray(), _bgr()]):
            for c in channels:
                assert c.ndim == 2
