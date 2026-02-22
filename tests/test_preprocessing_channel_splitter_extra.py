"""Extra tests for puzzle_reconstruction.preprocessing.channel_splitter."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=16, w=16, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=16, w=16):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 30
    img[:, :, 1] = 120
    img[:, :, 2] = 210
    return img


def _random_gray(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


# ─── ChannelStats extras ──────────────────────────────────────────────────────

class TestChannelStatsExtra:
    def test_repr_is_string(self):
        cs = ChannelStats(mean=50.0, std=10.0, min_val=0.0, max_val=100.0, median=50.0)
        assert isinstance(repr(cs), str)

    def test_median_stored(self):
        cs = ChannelStats(mean=100.0, std=5.0, min_val=90.0, max_val=110.0, median=99.5)
        assert cs.median == pytest.approx(99.5)

    def test_std_zero_constant(self):
        c = _gray(val=128)
        stats = channel_statistics(c)
        assert stats.std == pytest.approx(0.0)

    def test_mean_stored(self):
        cs = ChannelStats(mean=77.7, std=1.0, min_val=0.0, max_val=255.0, median=77.7)
        assert cs.mean == pytest.approx(77.7)

    def test_min_val_zero(self):
        cs = ChannelStats(mean=50.0, std=20.0, min_val=0.0, max_val=100.0, median=50.0)
        assert cs.min_val == pytest.approx(0.0)

    def test_max_val_stored(self):
        cs = ChannelStats(mean=50.0, std=20.0, min_val=0.0, max_val=200.0, median=50.0)
        assert cs.max_val == pytest.approx(200.0)


# ─── split_channels extras ────────────────────────────────────────────────────

class TestSplitChannelsExtra:
    def test_single_pixel_gray(self):
        img = np.array([[200]], dtype=np.uint8)
        channels = split_channels(img)
        assert len(channels) == 1
        assert channels[0][0, 0] == 200

    def test_large_image(self):
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        channels = split_channels(img)
        assert len(channels) == 1
        assert channels[0].shape == (256, 256)

    def test_bgr_channel_dtypes_uint8(self):
        for c in split_channels(_bgr()):
            assert c.dtype == np.uint8

    def test_gray_channel_dtype_uint8(self):
        channels = split_channels(_gray())
        assert channels[0].dtype == np.uint8

    def test_non_square_gray(self):
        img = _gray(h=8, w=24)
        channels = split_channels(img)
        assert channels[0].shape == (8, 24)

    def test_non_square_bgr(self):
        img = np.zeros((8, 24, 3), dtype=np.uint8)
        channels = split_channels(img)
        assert len(channels) == 3
        for c in channels:
            assert c.shape == (8, 24)

    def test_modifying_channel_doesnt_affect_original(self):
        img = _bgr()
        original_copy = img.copy()
        channels = split_channels(img)
        channels[0][:] = 0
        np.testing.assert_array_equal(img, original_copy)


# ─── merge_channels extras ────────────────────────────────────────────────────

class TestMergeChannelsExtra:
    def test_two_channels_shape(self):
        c = _gray(h=8, w=8)
        result = merge_channels([c, c])
        assert result.shape == (8, 8, 2)

    def test_four_channels_shape(self):
        c = _gray(h=4, w=4)
        result = merge_channels([c, c, c, c])
        assert result.ndim == 3
        assert result.shape[2] == 4

    def test_dtype_preserved(self):
        c = _gray()
        result = merge_channels([c, c, c])
        assert result.dtype == np.uint8

    def test_roundtrip_bgr(self):
        img = _bgr()
        channels = split_channels(img)
        rebuilt = merge_channels(channels)
        np.testing.assert_array_equal(rebuilt, img)

    def test_single_channel_shape(self):
        c = _gray(h=5, w=7)
        result = merge_channels([c])
        assert result.ndim == 2
        assert result.shape == (5, 7)

    def test_all_zeros_channels(self):
        c = np.zeros((4, 4), dtype=np.uint8)
        result = merge_channels([c, c, c])
        assert (result == 0).all()


# ─── channel_statistics extras ────────────────────────────────────────────────

class TestChannelStatisticsExtra:
    def test_returns_channel_stats(self):
        assert isinstance(channel_statistics(_gray()), ChannelStats)

    def test_mean_all_zeros(self):
        c = np.zeros((8, 8), dtype=np.uint8)
        stats = channel_statistics(c)
        assert stats.mean == pytest.approx(0.0)

    def test_mean_all_255(self):
        c = np.full((8, 8), 255, dtype=np.uint8)
        stats = channel_statistics(c)
        assert stats.mean == pytest.approx(255.0)

    def test_min_max_known(self):
        c = np.array([[0, 100, 200]], dtype=np.uint8)
        stats = channel_statistics(c)
        assert stats.min_val == pytest.approx(0.0)
        assert stats.max_val == pytest.approx(200.0)

    def test_median_of_sorted(self):
        c = np.array([[10, 20, 30]], dtype=np.uint8)
        stats = channel_statistics(c)
        assert stats.median == pytest.approx(20.0)

    def test_non_square_channel(self):
        c = _gray(h=8, w=24)
        stats = channel_statistics(c)
        assert isinstance(stats, ChannelStats)


# ─── equalize_channel extras ──────────────────────────────────────────────────

class TestEqualizeChannelExtra:
    def test_non_square_shape_preserved(self):
        c = _gray(h=8, w=24)
        result = equalize_channel(c)
        assert result.shape == (8, 24)

    def test_all_zeros_channel(self):
        c = np.zeros((8, 8), dtype=np.uint8)
        result = equalize_channel(c)
        assert result.dtype == np.uint8
        assert result.shape == (8, 8)

    def test_all_255_channel(self):
        c = np.full((8, 8), 255, dtype=np.uint8)
        result = equalize_channel(c)
        assert result.dtype == np.uint8

    def test_random_image_valid(self):
        c = _random_gray(h=32, w=32)
        result = equalize_channel(c)
        assert result.dtype == np.uint8
        assert result.shape == c.shape

    def test_values_in_0_255(self):
        c = _random_gray(h=16, w=16)
        result = equalize_channel(c)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255


# ─── normalize_channel extras ─────────────────────────────────────────────────

class TestNormalizeChannelExtra:
    def test_float32_input_accepted(self):
        c = _gray().astype(np.float32)
        result = normalize_channel(c)
        assert result.dtype == np.float64

    def test_out_min_neg1_out_max_1(self):
        c = np.array([[0, 255]], dtype=np.uint8)
        result = normalize_channel(c, out_min=-1.0, out_max=1.0)
        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_channel_all_out_min(self):
        c = np.full((6, 6), 77, dtype=np.uint8)
        result = normalize_channel(c, out_min=0.0, out_max=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_non_square_shape(self):
        c = _gray(h=6, w=18)
        result = normalize_channel(c)
        assert result.shape == (6, 18)

    def test_out_min_0_out_max_255(self):
        c = np.array([[50, 150]], dtype=np.uint8)
        result = normalize_channel(c, out_min=0.0, out_max=255.0)
        assert result.min() >= 0.0
        assert result.max() <= 255.0


# ─── channel_difference extras ────────────────────────────────────────────────

class TestChannelDifferenceExtra:
    def test_nonneg_asymmetric(self):
        c1 = np.array([[10, 200]], dtype=np.uint8)
        c2 = np.array([[200, 10]], dtype=np.uint8)
        result = channel_difference(c1, c2)
        assert (result >= 0).all()

    def test_large_images(self):
        c1 = _random_gray(h=64, w=64, seed=0)
        c2 = _random_gray(h=64, w=64, seed=1)
        result = channel_difference(c1, c2)
        assert result.shape == (64, 64)
        assert (result >= 0).all()

    def test_result_dtype_float64(self):
        result = channel_difference(_gray(val=100), _gray(val=50))
        assert result.dtype == np.float64

    def test_clip_to_smaller_both_dims(self):
        c1 = _gray(h=10, w=12)
        c2 = _gray(h=8, w=9)
        result = channel_difference(c1, c2)
        assert result.shape == (8, 9)

    def test_zero_difference_same_channel(self):
        c = _random_gray()
        result = channel_difference(c, c)
        assert (result == 0.0).all()


# ─── apply_per_channel extras ─────────────────────────────────────────────────

class TestApplyPerChannelExtra:
    def test_scale_by_zero(self):
        img = _bgr()
        result = apply_per_channel(img, lambda c: np.zeros_like(c))
        assert (result == 0).all()

    def test_invert_gray(self):
        img = _gray(val=100)
        result = apply_per_channel(img, lambda c: (255 - c.astype(np.int32)).clip(0, 255).astype(np.uint8))
        assert result.ndim == 2

    def test_identity_bgr_shape(self):
        img = _bgr(h=8, w=8)
        result = apply_per_channel(img, lambda c: c)
        assert result.shape == img.shape

    def test_non_square_bgr(self):
        img = np.zeros((8, 24, 3), dtype=np.uint8)
        result = apply_per_channel(img, lambda c: c)
        assert result.shape == (8, 24, 3)

    def test_func_receives_2d_channels(self):
        received_shapes = []
        img = _bgr()
        apply_per_channel(img, lambda c: (received_shapes.append(c.ndim), c)[1])
        assert all(s == 2 for s in received_shapes)


# ─── batch_split extras ───────────────────────────────────────────────────────

class TestBatchSplitExtra:
    def test_empty_list_returns_empty(self):
        assert batch_split([]) == []

    def test_single_gray(self):
        result = batch_split([_gray()])
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_single_bgr(self):
        result = batch_split([_bgr()])
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_five_images(self):
        images = [_gray() if i % 2 == 0 else _bgr() for i in range(5)]
        result = batch_split(images)
        assert len(result) == 5

    def test_channel_counts_correct(self):
        images = [_gray(), _bgr(), _gray()]
        result = batch_split(images)
        assert len(result[0]) == 1
        assert len(result[1]) == 3
        assert len(result[2]) == 1

    def test_all_channels_2d(self):
        for channels in batch_split([_gray(), _bgr()]):
            for c in channels:
                assert c.ndim == 2
