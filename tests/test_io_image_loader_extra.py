"""Extra tests for puzzle_reconstruction.io.image_loader."""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from puzzle_reconstruction.io.image_loader import (
    LoadConfig,
    LoadedImage,
    batch_load,
    list_image_files,
    load_from_array,
    resize_image,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _rgb(h=32, w=32):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _save_png(arr, path):
    cv2.imwrite(path, arr)


# ─── TestLoadConfigExtra ──────────────────────────────────────────────────────

class TestLoadConfigExtra:
    def test_normalize_default_false(self):
        assert LoadConfig().normalize is False

    def test_target_size_tuple_stored(self):
        cfg = LoadConfig(target_size=(100, 200))
        assert cfg.target_size == (100, 200)

    def test_target_size_non_square_ok(self):
        cfg = LoadConfig(target_size=(128, 64))
        assert cfg.target_size[0] == 128

    def test_all_valid_color_modes(self):
        for mode in ("bgr", "gray", "rgb"):
            cfg = LoadConfig(color_mode=mode)
            assert cfg.color_mode == mode

    def test_normalize_false_stores(self):
        cfg = LoadConfig(normalize=False)
        assert cfg.normalize is False

    def test_target_size_1x1_ok(self):
        cfg = LoadConfig(target_size=(1, 1))
        assert cfg.target_size == (1, 1)

    def test_color_mode_case_sensitive_upper_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(color_mode="BGR")


# ─── TestLoadedImageExtra ─────────────────────────────────────────────────────

class TestLoadedImageExtra:
    def _make(self, arr=None, image_id=0, path="test.png"):
        if arr is None:
            arr = _gray()
        return LoadedImage(data=arr, path=path,
                           image_id=image_id,
                           original_size=(arr.shape[1], arr.shape[0]))

    def test_size_width_then_height(self):
        li = self._make(_gray(48, 64))
        w, h = li.size
        assert w == 64
        assert h == 48

    def test_len_rgb(self):
        li = self._make(_rgb(16, 16))
        # len = h * w (pixels, not values)
        assert len(li) == 16 * 16 * 3

    def test_path_empty_string_ok(self):
        li = self._make(path="")
        assert li.path == ""

    def test_image_id_zero_ok(self):
        li = self._make(image_id=0)
        assert li.image_id == 0

    def test_image_id_large_ok(self):
        li = self._make(image_id=9999)
        assert li.image_id == 9999

    def test_shape_3d_rgb(self):
        li = self._make(_rgb(20, 30))
        assert li.shape == (20, 30, 3)

    def test_original_size_reflects_arr(self):
        arr = _gray(50, 60)
        li = LoadedImage(data=arr, path="", image_id=0,
                         original_size=(60, 50))
        assert li.original_size == (60, 50)


# ─── TestLoadFromArrayExtra ───────────────────────────────────────────────────

class TestLoadFromArrayExtra:
    def test_2d_gray_preserved(self):
        arr = _gray(24, 32)
        li = load_from_array(arr)
        assert li.data.shape[:2] == (24, 32)

    def test_gray_mode_from_rgb(self):
        cfg = LoadConfig(color_mode="gray")
        li = load_from_array(_rgb(), cfg=cfg)
        assert li.data.ndim == 2

    def test_normalize_range_0_to_1(self):
        cfg = LoadConfig(normalize=True)
        arr = np.full((10, 10), 255, dtype=np.uint8)
        li = load_from_array(arr, cfg=cfg)
        assert li.data.max() <= 1.0 + 1e-6

    def test_no_normalize_preserves_values(self):
        arr = _gray(val=200)
        li = load_from_array(arr)
        assert li.data.max() == 200

    def test_original_size_hw(self):
        arr = _gray(48, 64)
        li = load_from_array(arr)
        assert li.original_size == (64, 48)  # (w, h)

    def test_target_size_changes_shape(self):
        cfg = LoadConfig(target_size=(16, 8))
        li = load_from_array(_gray(32, 32), cfg=cfg)
        assert li.data.shape[:2] == (8, 16)

    def test_rgb_stays_rgb(self):
        cfg = LoadConfig(color_mode="rgb")
        li = load_from_array(_rgb(), cfg=cfg)
        assert li.data.ndim == 3
        assert li.data.shape[2] == 3

    def test_large_image_id_ok(self):
        li = load_from_array(_gray(), image_id=500)
        assert li.image_id == 500


# ─── TestListImageFilesExtra ──────────────────────────────────────────────────

class TestListImageFilesExtra:
    def test_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            _save_png(_gray(), os.path.join(d, "img.png"))
            files = list_image_files(d)
            assert len(files) == 1

    def test_multiple_extensions(self):
        with tempfile.TemporaryDirectory() as d:
            _save_png(_gray(), os.path.join(d, "a.png"))
            _save_png(_gray(), os.path.join(d, "b.jpg"))
            files = list_image_files(d, extensions=(".png", ".jpg"))
            assert len(files) == 2

    def test_full_paths_returned(self):
        with tempfile.TemporaryDirectory() as d:
            _save_png(_gray(), os.path.join(d, "img.png"))
            files = list_image_files(d)
            assert all(os.path.isabs(f) for f in files)

    def test_custom_extension_only(self):
        with tempfile.TemporaryDirectory() as d:
            _save_png(_gray(), os.path.join(d, "img.png"))
            _save_png(_gray(), os.path.join(d, "img2.jpg"))
            files = list_image_files(d, extensions=(".png",))
            assert all(f.endswith(".png") for f in files)
            assert len(files) == 1

    def test_empty_dir_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            assert list_image_files(d) == []

    def test_files_exist(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                _save_png(_gray(), os.path.join(d, f"img_{i}.png"))
            for f in list_image_files(d):
                assert os.path.isfile(f)


# ─── TestBatchLoadExtra ───────────────────────────────────────────────────────

class TestBatchLoadExtra:
    def _make_files(self, tmpdir, n=3):
        paths = []
        for i in range(n):
            p = os.path.join(tmpdir, f"img_{i}.png")
            _save_png(_gray(), p)
            paths.append(p)
        return paths

    def test_ids_start_at_zero(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 4)
            ids = [li.image_id for li in batch_load(paths)]
            assert ids[0] == 0

    def test_config_normalize_applied(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 2)
            cfg = LoadConfig(normalize=True)
            for li in batch_load(paths, cfg=cfg):
                assert li.data.max() <= 1.0 + 1e-6

    def test_single_file_list(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 1)
            result = batch_load(paths)
            assert len(result) == 1

    def test_paths_stored(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 2)
            for li, p in zip(batch_load(paths), paths):
                assert li.path == p


# ─── TestResizeImageExtra ─────────────────────────────────────────────────────

class TestResizeImageExtra:
    def test_gray_resized(self):
        result = resize_image(_gray(32, 32), (16, 16))
        assert result.shape == (16, 16)

    def test_downscale_rgb(self):
        result = resize_image(_rgb(64, 64), (32, 32))
        assert result.shape == (32, 32, 3)

    def test_non_square_target(self):
        result = resize_image(_gray(32, 32), (10, 20))
        assert result.shape[:2] == (20, 10)

    def test_dtype_float32_preserved(self):
        arr = _gray().astype(np.float32) / 255.0
        result = resize_image(arr, (16, 16))
        assert result.dtype == np.float32

    def test_values_in_valid_range_uint8(self):
        result = resize_image(_gray(val=200), (16, 16))
        assert result.max() <= 255
        assert result.min() >= 0

    def test_zero_size_raises(self):
        with pytest.raises((ValueError, cv2.error)):
            resize_image(_gray(), (0, 16))
