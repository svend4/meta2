"""Тесты для puzzle_reconstruction.io.image_loader."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from puzzle_reconstruction.io.image_loader import (
    LoadConfig,
    LoadedImage,
    load_image,
    load_from_array,
    list_image_files,
    batch_load,
    load_from_directory,
    resize_image,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_png(tmp_path):
    """Сохранить одно серое PNG во временную директорию."""
    img = np.full((32, 32), 128, dtype=np.uint8)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def tmp_dir_with_images(tmp_path):
    """Директория с 3 PNG-файлами."""
    for i in range(3):
        img = np.full((16, 16), i * 80, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i:02d}.png"), img)
    return str(tmp_path)


# ─── TestLoadConfig ───────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_default_values(self):
        cfg = LoadConfig()
        assert cfg.color_mode == "bgr"
        assert cfg.target_size is None
        assert cfg.normalize is False

    def test_valid_color_modes(self):
        for mode in ("gray", "bgr", "rgb"):
            cfg = LoadConfig(color_mode=mode)
            assert cfg.color_mode == mode

    def test_invalid_color_mode_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(color_mode="hsv")

    def test_target_size_w_zero_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(0, 32))

    def test_target_size_h_zero_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(32, 0))

    def test_target_size_valid(self):
        cfg = LoadConfig(target_size=(64, 48))
        assert cfg.target_size == (64, 48)

    def test_normalize_flag(self):
        cfg = LoadConfig(normalize=True)
        assert cfg.normalize is True


# ─── TestLoadedImage ──────────────────────────────────────────────────────────

class TestLoadedImage:
    def _make(self, data=None):
        if data is None:
            data = np.zeros((32, 32, 3), dtype=np.uint8)
        return LoadedImage(data=data, path="test.png", image_id=0,
                           original_size=(32, 32))

    def test_basic_creation(self):
        li = self._make()
        assert li.image_id == 0

    def test_negative_image_id_raises(self):
        with pytest.raises(ValueError):
            LoadedImage(data=np.zeros((8, 8, 3), dtype=np.uint8),
                        path="", image_id=-1, original_size=(8, 8))

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            LoadedImage(data=np.zeros((8, 8, 3, 2), dtype=np.uint8),
                        path="", image_id=0, original_size=(8, 8))

    def test_shape_property_3d(self):
        li = self._make(np.zeros((32, 48, 3), dtype=np.uint8))
        assert li.shape == (32, 48, 3)

    def test_shape_property_2d(self):
        li = self._make(np.zeros((32, 48), dtype=np.uint8))
        assert li.shape == (32, 48)

    def test_size_property(self):
        li = self._make(np.zeros((32, 48, 3), dtype=np.uint8))
        assert li.size == (48, 32)  # (width, height)

    def test_len_property(self):
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        li = self._make(data)
        assert len(li) == 4 * 4 * 3


# ─── TestLoadImage ────────────────────────────────────────────────────────────

class TestLoadImage:
    def test_returns_loaded_image(self, tmp_png):
        li = load_image(tmp_png)
        assert isinstance(li, LoadedImage)

    def test_path_stored(self, tmp_png):
        li = load_image(tmp_png)
        assert li.path == tmp_png

    def test_image_id_stored(self, tmp_png):
        li = load_image(tmp_png, image_id=5)
        assert li.image_id == 5

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/img.png")

    def test_negative_image_id_raises(self, tmp_png):
        with pytest.raises(ValueError):
            load_image(tmp_png, image_id=-1)

    def test_gray_mode(self, tmp_png):
        cfg = LoadConfig(color_mode="gray")
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.ndim == 2

    def test_rgb_mode(self, tmp_png):
        cfg = LoadConfig(color_mode="rgb")
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.ndim == 3

    def test_target_size_applied(self, tmp_png):
        cfg = LoadConfig(target_size=(16, 16))
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.shape[:2] == (16, 16)

    def test_normalize_float32(self, tmp_png):
        cfg = LoadConfig(normalize=True)
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.dtype == np.float32
        assert li.data.max() <= 1.0 + 1e-6


# ─── TestLoadFromArray ────────────────────────────────────────────────────────

class TestLoadFromArray:
    def _arr(self, h=32, w=32):
        return np.full((h, w, 3), 100, dtype=np.uint8)

    def test_returns_loaded_image(self):
        li = load_from_array(self._arr())
        assert isinstance(li, LoadedImage)

    def test_path_empty_string(self):
        li = load_from_array(self._arr())
        assert li.path == ""

    def test_original_size_correct(self):
        li = load_from_array(np.zeros((48, 64, 3), dtype=np.uint8))
        assert li.original_size == (64, 48)  # (w, h)

    def test_ndim_4_raises(self):
        with pytest.raises(ValueError):
            load_from_array(np.zeros((4, 4, 3, 2), dtype=np.uint8))

    def test_negative_image_id_raises(self):
        with pytest.raises(ValueError):
            load_from_array(self._arr(), image_id=-1)

    def test_gray_mode(self):
        cfg = LoadConfig(color_mode="gray")
        li = load_from_array(self._arr(), cfg=cfg)
        assert li.data.ndim == 2

    def test_target_size_applied(self):
        cfg = LoadConfig(target_size=(16, 16))
        li = load_from_array(self._arr(32, 32), cfg=cfg)
        assert li.data.shape[:2] == (16, 16)

    def test_normalize(self):
        cfg = LoadConfig(normalize=True)
        li = load_from_array(self._arr(), cfg=cfg)
        assert li.data.dtype == np.float32


# ─── TestListImageFiles ───────────────────────────────────────────────────────

class TestListImageFiles:
    def test_returns_sorted_list(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images)
        assert files == sorted(files)

    def test_correct_count(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images)
        assert len(files) == 3

    def test_all_png(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images, extensions=(".png",))
        assert all(f.endswith(".png") for f in files)

    def test_nonexistent_dir_raises(self):
        with pytest.raises(NotADirectoryError):
            list_image_files("/nonexistent/directory")

    def test_wrong_extension_empty(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images, extensions=(".xyz",))
        assert files == []

    def test_empty_directory(self, tmp_path):
        files = list_image_files(str(tmp_path))
        assert files == []


# ─── TestBatchLoad ────────────────────────────────────────────────────────────

class TestBatchLoad:
    def test_returns_list(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        result = batch_load(paths)
        assert isinstance(result, list)

    def test_correct_length(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        result = batch_load(paths)
        assert len(result) == 3

    def test_empty_paths(self):
        assert batch_load([]) == []

    def test_image_ids_sequential(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        result = batch_load(paths)
        ids = [li.image_id for li in result]
        assert ids == list(range(len(paths)))

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            batch_load(["/nonexistent/img.png"])

    def test_each_loaded_image(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        result = batch_load(paths)
        assert all(isinstance(li, LoadedImage) for li in result)


# ─── TestLoadFromDirectory ────────────────────────────────────────────────────

class TestLoadFromDirectory:
    def test_returns_list(self, tmp_dir_with_images):
        result = load_from_directory(tmp_dir_with_images)
        assert isinstance(result, list)

    def test_correct_count(self, tmp_dir_with_images):
        result = load_from_directory(tmp_dir_with_images)
        assert len(result) == 3

    def test_nonexistent_raises(self):
        with pytest.raises(NotADirectoryError):
            load_from_directory("/nonexistent/dir")

    def test_config_applied(self, tmp_dir_with_images):
        cfg = LoadConfig(color_mode="gray")
        result = load_from_directory(tmp_dir_with_images, cfg=cfg)
        assert all(li.data.ndim == 2 for li in result)


# ─── TestResizeImage ──────────────────────────────────────────────────────────

class TestResizeImage:
    def test_output_shape(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = resize_image(img, (32, 16))
        assert out.shape == (16, 32, 3)

    def test_gray_image(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        out = resize_image(img, (32, 32))
        assert out.shape == (32, 32)

    def test_width_zero_raises(self):
        with pytest.raises(ValueError):
            resize_image(np.zeros((32, 32), dtype=np.uint8), (0, 32))

    def test_height_zero_raises(self):
        with pytest.raises(ValueError):
            resize_image(np.zeros((32, 32), dtype=np.uint8), (32, 0))

    def test_same_size(self):
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        out = resize_image(img, (16, 16))
        assert out.shape == img.shape

    def test_upscale(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        out = resize_image(img, (32, 32))
        assert out.shape == (32, 32)
