"""Extra tests for puzzle_reconstruction/utils/tile_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

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

def _gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 50
    img[:, :, 1] = 100
    img[:, :, 2] = 150
    return img


def _tile(data=None, row=0, col=0, y=0, x=0, sh=64, sw=64):
    if data is None:
        data = np.full((32, 32), 200, dtype=np.uint8)
    return Tile(data=data, row=row, col=col, y=y, x=x, source_h=sh, source_w=sw)


# ─── TileConfig (extra) ───────────────────────────────────────────────────────

class TestTileConfigExtra:
    def test_large_tile_sizes(self):
        cfg = TileConfig(tile_h=512, tile_w=512)
        assert cfg.tile_h == 512
        assert cfg.tile_w == 512

    def test_tile_h_ne_tile_w_ok(self):
        cfg = TileConfig(tile_h=32, tile_w=64)
        assert cfg.tile_h == 32
        assert cfg.tile_w == 64

    def test_stride_h_equals_tile_h(self):
        cfg = TileConfig(tile_h=32, stride_h=32)
        assert cfg.effective_stride_h == 32

    def test_stride_w_equals_tile_w(self):
        cfg = TileConfig(tile_w=32, stride_w=32)
        assert cfg.effective_stride_w == 32

    def test_pad_value_zero_boundary(self):
        cfg = TileConfig(pad_value=0)
        assert cfg.pad_value == 0

    def test_pad_value_255_boundary(self):
        cfg = TileConfig(pad_value=255)
        assert cfg.pad_value == 255

    def test_effective_stride_h_zero_input_equals_tile_h(self):
        cfg = TileConfig(tile_h=16, stride_h=0)
        assert cfg.effective_stride_h == 16

    def test_overlapping_stride_smaller_than_tile(self):
        cfg = TileConfig(tile_h=32, stride_h=16)
        assert cfg.effective_stride_h < cfg.tile_h

    def test_independent_configs(self):
        c1 = TileConfig(tile_h=32)
        c2 = TileConfig(tile_h=64)
        assert c1.tile_h != c2.tile_h


# ─── Tile (extra) ─────────────────────────────────────────────────────────────

class TestTileExtra:
    def test_row_col_both_positive(self):
        t = _tile(row=5, col=7)
        assert t.row == 5
        assert t.col == 7

    def test_y_x_large_values(self):
        t = _tile(y=1024, x=2048)
        assert t.y == 1024
        assert t.x == 2048

    def test_source_dimensions_stored(self):
        t = _tile(sh=256, sw=128)
        assert t.source_h == 256
        assert t.source_w == 128

    def test_h_w_properties_from_data(self):
        data = np.zeros((20, 40), dtype=np.uint8)
        t = _tile(data=data)
        assert t.h == 20
        assert t.w == 40

    def test_3channel_h_w(self):
        data = np.zeros((10, 15, 3), dtype=np.uint8)
        t = Tile(data=data, row=0, col=0, y=0, x=0, source_h=20, source_w=30)
        assert t.h == 10
        assert t.w == 15

    def test_data_stored_correctly(self):
        data = np.full((8, 8), 77, dtype=np.uint8)
        t = _tile(data=data)
        assert (t.data == 77).all()

    def test_row_zero_col_zero(self):
        t = _tile(row=0, col=0)
        assert t.row == 0
        assert t.col == 0


# ─── compute_tile_grid (extra) ────────────────────────────────────────────────

class TestComputeTileGridExtra:
    def test_4x2_grid(self):
        cfg = TileConfig(tile_h=32, tile_w=64)
        result = compute_tile_grid(128, 128, cfg)
        assert len(result) == 8  # 128/32=4 rows, 128/64=2 cols → 8 tiles

    def test_positions_non_negative(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        for item in compute_tile_grid(64, 64, cfg):
            assert item[0] >= 0  # y
            assert item[1] >= 0  # x

    def test_overlapping_more_tiles_than_nonoverlapping(self):
        cfg_nooverlap = TileConfig(tile_h=32, tile_w=32, stride_h=32, stride_w=32)
        cfg_overlap = TileConfig(tile_h=32, tile_w=32, stride_h=16, stride_w=16)
        n_no = len(compute_tile_grid(64, 64, cfg_nooverlap))
        n_ov = len(compute_tile_grid(64, 64, cfg_overlap))
        assert n_ov >= n_no

    def test_row_0_included(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = compute_tile_grid(64, 64, cfg)
        rows = [r[2] for r in result]
        assert 0 in rows

    def test_col_0_included(self):
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = compute_tile_grid(64, 64, cfg)
        cols = [r[3] for r in result]
        assert 0 in cols

    def test_large_image_many_tiles(self):
        cfg = TileConfig(tile_h=16, tile_w=16)
        result = compute_tile_grid(256, 256, cfg)
        assert len(result) == 256


# ─── tile_image (extra) ───────────────────────────────────────────────────────

class TestTileImageExtra:
    def test_row_col_assigned(self):
        img = _gray(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        rows = {t.row for t in tiles}
        cols = {t.col for t in tiles}
        assert len(rows) == 2
        assert len(cols) == 2

    def test_data_dtype_preserved(self):
        img = _gray(64, 64)
        tiles = tile_image(img, cfg=TileConfig(tile_h=64, tile_w=64))
        assert tiles[0].data.dtype == np.uint8

    def test_bgr_all_3_channels(self):
        img = _bgr(64, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg=cfg)
        assert all(t.data.shape[2] == 3 for t in tiles)

    def test_tile_count_x_times_y(self):
        img = _gray(128, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg=cfg)
        # 4 rows × 2 cols = 8 tiles
        assert len(tiles) == 8

    def test_y_x_positions_cover_image(self):
        img = _gray(128, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        ys = sorted({t.y for t in tiles})
        xs = sorted({t.x for t in tiles})
        assert ys == [0, 64]
        assert xs == [0, 64]

    def test_identical_tiles_same_content_from_uniform_image(self):
        img = _gray(128, 128, fill=200)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        for t in tiles:
            assert (t.data == 200).all()

    def test_non_square_image(self):
        img = _gray(64, 128)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        assert len(tiles) == 2


# ─── reassemble_tiles (extra) ─────────────────────────────────────────────────

class TestReassembleTilesExtra:
    def test_roundtrip_non_square(self):
        img = _gray(64, 128, fill=77)
        cfg = TileConfig(tile_h=64, tile_w=64)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (64, 128))
        assert result.shape == (64, 128)
        np.testing.assert_array_equal(result, img)

    def test_bgr_roundtrip(self):
        img = _bgr(64, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (64, 64))
        assert result.shape == (64, 64, 3)
        np.testing.assert_array_equal(result, img)

    def test_output_values_match_original(self):
        img = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        cfg = TileConfig(tile_h=32, tile_w=32)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (64, 64))
        np.testing.assert_array_equal(result, img)

    def test_negative_height_raises(self):
        t = _tile()
        with pytest.raises(ValueError):
            reassemble_tiles([t], (-1, 64))

    def test_shape_4x4_tiles_correct(self):
        img = _gray(64, 64)
        cfg = TileConfig(tile_h=16, tile_w=16)
        tiles = tile_image(img, cfg=cfg)
        result = reassemble_tiles(tiles, (64, 64))
        assert result.shape == (64, 64)


# ─── tile_overlap_ratio (extra) ───────────────────────────────────────────────

class TestTileOverlapRatioExtra:
    def test_complete_containment(self):
        outer = Tile(np.zeros((32, 32), dtype=np.uint8), 0, 0, 0, 0, 64, 64)
        inner = Tile(np.zeros((8, 8), dtype=np.uint8), 0, 0, 8, 8, 64, 64)
        ratio = tile_overlap_ratio(outer, inner)
        assert ratio > 0.0

    def test_far_apart_zero_overlap(self):
        t1 = Tile(np.zeros((16, 16), dtype=np.uint8), 0, 0, 0, 0, 128, 128)
        t2 = Tile(np.zeros((16, 16), dtype=np.uint8), 0, 0, 100, 100, 128, 128)
        assert tile_overlap_ratio(t1, t2) == pytest.approx(0.0)

    def test_symmetric(self):
        t1 = Tile(np.zeros((32, 32), dtype=np.uint8), 0, 0, 0, 0, 64, 64)
        t2 = Tile(np.zeros((32, 32), dtype=np.uint8), 0, 0, 16, 16, 64, 64)
        assert tile_overlap_ratio(t1, t2) == pytest.approx(tile_overlap_ratio(t2, t1))

    def test_output_in_0_1(self):
        t1 = Tile(np.zeros((20, 20), dtype=np.uint8), 0, 0, 0, 0, 64, 64)
        t2 = Tile(np.zeros((20, 20), dtype=np.uint8), 0, 0, 5, 5, 64, 64)
        ratio = tile_overlap_ratio(t1, t2)
        assert 0.0 <= ratio <= 1.0

    def test_adjacent_no_overlap(self):
        t1 = Tile(np.zeros((32, 32), dtype=np.uint8), 0, 0, 0, 0, 128, 128)
        t2 = Tile(np.zeros((32, 32), dtype=np.uint8), 0, 0, 0, 32, 128, 128)
        assert tile_overlap_ratio(t1, t2) == pytest.approx(0.0)


# ─── filter_tiles_by_content (extra) ─────────────────────────────────────────

class TestFilterTilesByContentExtra:
    def test_half_content_tiles_filtered(self):
        # Create tiles: half zero, half nonzero
        d_full = np.full((32, 32), 200, dtype=np.uint8)
        d_empty = np.zeros((32, 32), dtype=np.uint8)
        t_full = _tile(data=d_full)
        t_empty = _tile(data=d_empty)
        result = filter_tiles_by_content([t_full, t_empty], min_foreground=0.5)
        assert len(result) == 1
        assert (result[0].data == 200).all()

    def test_threshold_zero_keeps_all(self):
        tiles = [_tile() for _ in range(5)]
        result = filter_tiles_by_content(tiles, min_foreground=0.0)
        assert len(result) == 5

    def test_threshold_one_keeps_fully_filled(self):
        d = np.full((8, 8), 200, dtype=np.uint8)
        t = _tile(data=d)
        result = filter_tiles_by_content([t], min_foreground=1.0)
        assert len(result) == 1

    def test_color_tiles_filtered(self):
        d_full = np.full((16, 16, 3), 100, dtype=np.uint8)
        d_empty = np.zeros((16, 16, 3), dtype=np.uint8)
        t_full = Tile(d_full, 0, 0, 0, 0, 32, 32)
        t_empty = Tile(d_empty, 0, 0, 0, 16, 32, 32)
        result = filter_tiles_by_content([t_full, t_empty], min_foreground=0.1)
        assert len(result) == 1

    def test_returns_tile_objects(self):
        tiles = [_tile(data=np.full((8, 8), 50, dtype=np.uint8))]
        result = filter_tiles_by_content(tiles, min_foreground=0.0)
        assert isinstance(result[0], Tile)


# ─── batch_tile_images (extra) ────────────────────────────────────────────────

class TestBatchTileImagesExtra:
    def test_single_image_in_list(self):
        result = batch_tile_images([_gray(64, 64)])
        assert len(result) == 1

    def test_multiple_images_same_size(self):
        images = [_gray(64, 64), _gray(64, 64), _gray(64, 64)]
        result = batch_tile_images(images)
        assert len(result) == 3

    def test_all_tiles_are_tiles(self):
        images = [_gray(64, 64)]
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = batch_tile_images(images, cfg=cfg)
        for t in result[0]:
            assert isinstance(t, Tile)

    def test_different_size_images(self):
        images = [_gray(64, 64), _gray(128, 128)]
        cfg = TileConfig(tile_h=64, tile_w=64)
        result = batch_tile_images(images, cfg=cfg)
        assert len(result[0]) == 1
        assert len(result[1]) == 4

    def test_bgr_images_in_batch(self):
        images = [_bgr(64, 64)]
        cfg = TileConfig(tile_h=32, tile_w=32)
        result = batch_tile_images(images, cfg=cfg)
        assert result[0][0].data.ndim == 3

    def test_batch_consistent_tile_size(self):
        images = [_gray(128, 128), _gray(128, 128)]
        cfg = TileConfig(tile_h=64, tile_w=64)
        result = batch_tile_images(images, cfg=cfg)
        for tiles in result:
            for t in tiles:
                assert t.data.shape == (64, 64)

    def test_large_batch(self):
        images = [_gray(64, 64) for _ in range(10)]
        result = batch_tile_images(images)
        assert len(result) == 10
