"""Тесты для puzzle_reconstruction/utils/tile_utils.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.tile_utils import (
    TileConfig,
    Tile,
    compute_tile_grid,
    tile_image,
    reassemble_tiles,
    tile_overlap_ratio,
    filter_tiles_by_content,
    batch_tile_images,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=128, w=128, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_bgr(h=128, w=128):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 100
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    return img


def make_tile(data=None, row=0, col=0, y=0, x=0, sh=64, sw=64):
    if data is None:
        data = np.full((32, 32), 128, dtype=np.uint8)
    return Tile(data=data, row=row, col=col, y=y, x=x, source_h=sh, source_w=sw)


# ─── TileConfig ───────────────────────────────────────────────────────────────

class TestTileConfig:
    def test_defaults(self):
        cfg = TileConfig()
        assert cfg.tile_h == 64
        assert cfg.tile_w == 64
        assert cfg.stride_h == 0
        assert cfg.stride_w == 0
        assert cfg.pad_value == 0

    def test_tile_h_zero_raises(self):
        with pytest.raises(ValueError, match="tile_h"):
            TileConfig(tile_h=0)

    def test_tile_h_negative_raises(self):
        with pytest.raises(ValueError, match="tile_h"):
            TileConfig(tile_h=-1)

    def test_tile_w_zero_raises(self):
        with pytest.raises(ValueError, match="tile_w"):
            TileConfig(tile_w=0)

    def test_stride_h_negative_raises(self):
        with pytest.raises(ValueError, match="stride_h"):
            TileConfig(stride_h=-1)

    def test_stride_w_negative_raises(self):
        with pytest.raises(ValueError, match="stride_w"):
            TileConfig(stride_w=-1)

    def test_pad_value_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pad_value"):
            TileConfig(pad_value=256)

    def test_pad_value_negative_raises(self):
        with pytest.raises(ValueError, match="pad_value"):
            TileConfig(pad_value=-1)

    def test_effective_stride_h_default(self):
        cfg = TileConfig(tile_h=32, stride_h=0)
        assert cfg.effective_stride_h == 32

    def test_effective_stride_w_default(self):
        cfg = TileConfig(tile_w=32, stride_w=0)
        assert cfg.effective_stride_w == 32

    def test_effective_stride_h_custom(self):
        cfg = TileConfig(tile_h=32, stride_h=16)
        assert cfg.effective_stride_h == 16

    def test_effective_stride_w_custom(self):
        cfg = TileConfig(tile_w=32, stride_w=8)
        assert cfg.effective_stride_w == 8


# ─── Tile ─────────────────────────────────────────────────────────────────────

class TestTile:
    def test_creation(self):
        data = np.zeros((32, 64), dtype=np.uint8)
        t = Tile(data=data, row=1, col=2, y=32, x=64, source_h=256, source_w=256)
        assert t.row == 1
        assert t.col == 2
        assert t.y == 32
        assert t.x == 64

    def test_h_property(self):
        data = np.zeros((48, 64), dtype=np.uint8)
        t = make_tile(data=data)
        assert t.h == 48

    def test_w_property(self):
        data = np.zeros((48, 64), dtype=np.uint8)
        t = make_tile(data=data)
        assert t.w == 64

    def test_color_tile(self):
        data = np.zeros((32, 32, 3), dtype=np.uint8)
        t = Tile(data=data, row=0, col=0, y=0, x=0, source_h=64, source_w=64)
        assert t.h == 32
        assert t.w == 32


# ─── compute_tile_grid ────────────────────────────────────────────────────────

class TestComputeTileGrid:
    def test_single_tile(self):
        cfg = TileConfig(tile_h=128, tile_w=128)
        result = compute_tile_grid(128, 128, cfg)
        assert len(result) == 1
        assert result[0] == (0, 0, 0, 0)

    def test_2x2_grid(self):
        cfg = TileConfig(tile_h=64, tile_w=64)
        result = compute_tile_grid(128, 128, cfg)
        assert len(result) == 4

    def test_3x3_grid(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = compute_tile_grid(96, 96, cfg)
        assert len(result) == 9

    def test_returns_list_of_tuples(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = compute_tile_grid(64, 64, cfg)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 4

    def test_row_col_ordering(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = compute_tile_grid(64, 64, cfg)
        # Should have rows 0,0 and 1,1
        rows = [r[2] for r in result]
        cols = [r[3] for r in result]
        assert 0 in rows and 1 in rows
        assert 0 in cols and 1 in cols

    def test_img_h_zero_raises(self):
        cfg = TileConfig()
        with pytest.raises(ValueError, match="img_h"):
            compute_tile_grid(0, 64, cfg)

    def test_img_w_zero_raises(self):
        cfg = TileConfig()
        with pytest.raises(ValueError, match="img_w"):
            compute_tile_grid(64, 0, cfg)

    def test_overlapping_stride(self):
        cfg = TileConfig(tile_h=32, tile_w=32, stride_h=16, stride_w=16)
        result = compute_tile_grid(64, 64, cfg)
        # With 50% overlap and 64/32 image: more tiles than non-overlapping
        assert len(result) > 4


# ─── tile_image ───────────────────────────────────────────────────────────────

class TestTileImage:
    def test_returns_list(self):
        img = make_gray(128, 128)
        result = tile_image(img)
        assert isinstance(result, list)

    def test_all_tiles_correct_size(self):
        img = make_gray(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        for t in tiles:
            assert t.data.shape == (64, 64)

    def test_2x2_produces_4_tiles(self):
        img = make_gray(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        assert len(tiles) == 4

    def test_color_image_preserved(self):
        img = make_bgr(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        for t in tiles:
            assert t.data.ndim == 3
            assert t.data.shape[2] == 3

    def test_source_dimensions_stored(self):
        img = make_gray(96, 64)
        tiles = tile_image(img)
        for t in tiles:
            assert t.source_h == 96
            assert t.source_w == 64

    def test_tile_positions_correct(self):
        img = make_gray(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        ys = {t.y for t in tiles}
        xs = {t.x for t in tiles}
        assert 0 in ys and 64 in ys
        assert 0 in xs and 64 in xs

    def test_padding_on_remainder(self):
        img = make_gray(96, 96)  # not divisible by 64
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        # All tiles should be 64x64 (padded)
        for t in tiles:
            assert t.data.shape == (64, 64)

    def test_1d_image_raises(self):
        with pytest.raises(ValueError):
            tile_image(np.zeros(128, dtype=np.uint8))

    def test_none_cfg_uses_defaults(self):
        img = make_gray(64, 64)
        result = tile_image(img, cfg=None)
        assert isinstance(result, list)

    def test_pad_value_applied(self):
        img = np.zeros((96, 96), dtype=np.uint8)  # remainder needs padding
        cfg = TileConfig(tile_h=64, tile_w=64, pad_value=255)
        tiles = tile_image(img, cfg=cfg)
        # Find a padded tile
        padded_tiles = [t for t in tiles if t.y + 64 > 96 or t.x + 64 > 96]
        if padded_tiles:
            t = padded_tiles[0]
            # Padded region should be 255
            assert t.data.max() == 255


# ─── reassemble_tiles ─────────────────────────────────────────────────────────

class TestReassembleTiles:
    def test_roundtrip_gray(self):
        img = make_gray(128, 128, fill=100)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (128, 128))
        assert result.shape == (128, 128)
        np.testing.assert_array_equal(result, img)

    def test_roundtrip_color(self):
        img = make_bgr(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (128, 128))
        assert result.shape == (128, 128, 3)

    def test_dtype_uint8(self):
        img = make_gray(64, 64)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (64, 64))
        assert result.dtype == np.uint8

    def test_empty_tiles_raises(self):
        with pytest.raises(ValueError):
            reassemble_tiles([], (64, 64))

    def test_negative_out_shape_raises(self):
        img = make_gray(64, 64)
        tiles = tile_image(img)
        with pytest.raises(ValueError):
            reassemble_tiles(tiles, (0, 64))

    def test_correct_output_shape(self):
        img = make_gray(128, 64)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (128, 64))
        assert result.shape == (128, 64)


# ─── tile_overlap_ratio ───────────────────────────────────────────────────────

class TestTileOverlapRatio:
    def test_identical_tiles_full_overlap(self):
        t = make_tile(y=0, x=0)
        ratio = tile_overlap_ratio(t, t)
        assert ratio == pytest.approx(1.0)

    def test_non_overlapping_tiles(self):
        t1 = make_tile(y=0, x=0)   # 32x32 at (0,0)
        t2 = make_tile(y=0, x=64)  # 32x32 at (0,64)
        ratio = tile_overlap_ratio(t1, t2)
        assert ratio == pytest.approx(0.0)

    def test_partial_overlap(self):
        d = np.zeros((32, 32), dtype=np.uint8)
        t1 = Tile(d, 0, 0, 0, 0, 64, 64)   # 32x32 at (0,0)
        t2 = Tile(d, 0, 0, 0, 16, 64, 64)  # 32x32 at (0,16), overlap 16px wide
        ratio = tile_overlap_ratio(t1, t2)
        assert 0.0 < ratio < 1.0

    def test_ratio_in_0_1(self):
        t1 = make_tile(y=0, x=0)
        t2 = make_tile(y=16, x=16)
        ratio = tile_overlap_ratio(t1, t2)
        assert 0.0 <= ratio <= 1.0

    def test_commutative(self):
        d = np.zeros((32, 32), dtype=np.uint8)
        t1 = Tile(d, 0, 0, 0, 0, 64, 64)
        t2 = Tile(d, 0, 0, 0, 16, 64, 64)
        assert tile_overlap_ratio(t1, t2) == pytest.approx(tile_overlap_ratio(t2, t1))


# ─── filter_tiles_by_content ──────────────────────────────────────────────────

class TestFilterTilesByContent:
    def test_returns_list(self):
        tiles = tile_image(make_gray(64, 64))
        result = filter_tiles_by_content(tiles)
        assert isinstance(result, list)

    def test_zero_min_keeps_all(self):
        tiles = tile_image(make_gray(64, 64, fill=128))
        result = filter_tiles_by_content(tiles, min_foreground=0.0)
        assert len(result) == len(tiles)

    def test_one_min_keeps_none_unless_all_nonzero(self):
        img = np.zeros((64, 64), dtype=np.uint8)  # all black
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = filter_tiles_by_content(tiles, min_foreground=0.01)
        assert result == []

    def test_negative_min_raises(self):
        tiles = tile_image(make_gray(64, 64))
        with pytest.raises(ValueError, match="min_foreground"):
            filter_tiles_by_content(tiles, min_foreground=-0.1)

    def test_above_1_min_raises(self):
        tiles = tile_image(make_gray(64, 64))
        with pytest.raises(ValueError, match="min_foreground"):
            filter_tiles_by_content(tiles, min_foreground=1.1)

    def test_nonzero_foreground_keeps_tiles(self):
        img = make_gray(64, 64, fill=100)  # all nonzero → foreground ratio = 1.0
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = filter_tiles_by_content(tiles, min_foreground=0.9)
        assert len(result) == len(tiles)

    def test_empty_input_returns_empty(self):
        result = filter_tiles_by_content([], min_foreground=0.1)
        assert result == []


# ─── batch_tile_images ────────────────────────────────────────────────────────

class TestBatchTileImages:
    def test_empty_returns_empty(self):
        result = batch_tile_images([])
        assert result == []

    def test_length_matches(self):
        images = [make_gray(64, 64), make_gray(64, 64), make_gray(64, 64)]
        result = batch_tile_images(images)
        assert len(result) == 3

    def test_each_element_is_list_of_tiles(self):
        images = [make_gray(64, 64)]
        result = batch_tile_images(images)
        assert isinstance(result[0], list)
        for t in result[0]:
            assert isinstance(t, Tile)

    def test_cfg_applied(self):
        images = [make_gray(128, 128)]
        cfg = TileConfig(tile_h=64, tile_w=64)
        result = batch_tile_images(images, cfg=cfg)
        assert len(result[0]) == 4

    def test_color_images(self):
        images = [make_bgr(64, 64)]
        result = batch_tile_images(images)
        assert result[0][0].data.ndim == 3
