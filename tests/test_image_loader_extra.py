"""Extra tests for puzzle_reconstruction.io.image_loader."""
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


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_png(tmp_path):
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def tmp_jpg(tmp_path):
    img = np.full((48, 48, 3), 200, dtype=np.uint8)
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def tmp_dir_with_images(tmp_path):
    for i in range(4):
        img = np.full((16, 16, 3), i * 60, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i:02d}.png"), img)
    return str(tmp_path)


@pytest.fixture
def tmp_dir_mixed(tmp_path):
    """3 PNGs + 1 JPEG + 1 TXT."""
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"a_{i}.png"),
                    np.full((8, 8, 3), i * 80, dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "b.jpg"),
                np.full((8, 8, 3), 100, dtype=np.uint8))
    (tmp_path / "readme.txt").write_text("ignore me")
    return str(tmp_path)


# ─── LoadConfig extras ────────────────────────────────────────────────────────

class TestLoadConfigExtra:
    def test_repr_is_string(self):
        assert isinstance(repr(LoadConfig()), str)

    def test_target_size_none_default(self):
        assert LoadConfig().target_size is None

    def test_normalize_false_default(self):
        assert LoadConfig().normalize is False

    def test_normalize_true(self):
        assert LoadConfig(normalize=True).normalize is True

    def test_gray_mode_stored(self):
        assert LoadConfig(color_mode="gray").color_mode == "gray"

    def test_rgb_mode_stored(self):
        assert LoadConfig(color_mode="rgb").color_mode == "rgb"

    def test_target_size_large(self):
        cfg = LoadConfig(target_size=(1024, 768))
        assert cfg.target_size == (1024, 768)

    def test_target_size_w_negative_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(-1, 32))

    def test_target_size_h_negative_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(32, -1))


# ─── LoadedImage extras ───────────────────────────────────────────────────────

class TestLoadedImageExtra:
    def _li(self, h=32, w=32, c=3, image_id=0, path="test.png"):
        data = np.zeros((h, w, c) if c > 1 else (h, w), dtype=np.uint8)
        return LoadedImage(data=data, path=path, image_id=image_id,
                           original_size=(w, h))

    def test_repr_is_string(self):
        assert isinstance(repr(self._li()), str)

    def test_shape_2d(self):
        li = self._li(h=20, w=30, c=1)
        assert li.shape == (20, 30)

    def test_shape_3d(self):
        li = self._li(h=20, w=30, c=3)
        assert li.shape == (20, 30, 3)

    def test_size_width_height_order(self):
        li = self._li(h=20, w=40)
        w, h = li.size
        assert w == 40
        assert h == 20

    def test_len_2d(self):
        li = self._li(h=4, w=6, c=1)
        assert len(li) == 4 * 6

    def test_len_3d(self):
        li = self._li(h=4, w=6, c=3)
        assert len(li) == 4 * 6 * 3

    def test_path_stored(self):
        li = self._li(path="/some/path/img.png")
        assert li.path == "/some/path/img.png"

    def test_image_id_large(self):
        li = self._li(image_id=9999)
        assert li.image_id == 9999

    def test_original_size_stored(self):
        li = LoadedImage(data=np.zeros((32, 48, 3), dtype=np.uint8),
                         path="", image_id=0, original_size=(200, 150))
        assert li.original_size == (200, 150)


# ─── load_image extras ────────────────────────────────────────────────────────

class TestLoadImageExtra:
    def test_default_bgr_3_channels(self, tmp_png):
        li = load_image(tmp_png)
        assert li.data.ndim == 3

    def test_original_size_stored(self, tmp_png):
        li = load_image(tmp_png)
        w, h = li.original_size
        assert w > 0 and h > 0

    def test_target_size_changes_data_shape(self, tmp_png):
        cfg = LoadConfig(target_size=(8, 8))
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.shape[:2] == (8, 8)

    def test_image_id_zero_default(self, tmp_png):
        li = load_image(tmp_png)
        assert li.image_id == 0

    def test_large_image_id(self, tmp_png):
        li = load_image(tmp_png, image_id=999)
        assert li.image_id == 999

    def test_normalize_values_in_01(self, tmp_png):
        cfg = LoadConfig(normalize=True)
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.min() >= 0.0
        assert li.data.max() <= 1.0 + 1e-6

    def test_gray_output_2d(self, tmp_png):
        cfg = LoadConfig(color_mode="gray")
        li = load_image(tmp_png, cfg=cfg)
        assert li.data.ndim == 2

    def test_jpeg_loads_ok(self, tmp_jpg):
        li = load_image(tmp_jpg)
        assert isinstance(li, LoadedImage)
        assert li.data.ndim == 3


# ─── load_from_array extras ───────────────────────────────────────────────────

class TestLoadFromArrayExtra:
    def _arr(self, h=32, w=32, c=3, val=100):
        return np.full((h, w, c), val, dtype=np.uint8)

    def test_shape_preserved(self):
        arr = self._arr(40, 60)
        li = load_from_array(arr)
        assert li.data.shape[:2] == (40, 60)

    def test_image_id_default_zero(self):
        li = load_from_array(self._arr())
        assert li.image_id == 0

    def test_custom_image_id(self):
        li = load_from_array(self._arr(), image_id=7)
        assert li.image_id == 7

    def test_normalize_float32(self):
        cfg = LoadConfig(normalize=True)
        li = load_from_array(self._arr(val=255), cfg=cfg)
        assert li.data.dtype == np.float32
        assert li.data.max() <= 1.0 + 1e-6

    def test_gray_array_input_accepted(self):
        gray = np.full((32, 32), 128, dtype=np.uint8)
        li = load_from_array(gray)
        # 2D gray input may be promoted to 3D BGR; either is acceptable
        assert li.data.ndim in (2, 3)

    def test_target_size_resizes(self):
        cfg = LoadConfig(target_size=(8, 8))
        li = load_from_array(self._arr(32, 32), cfg=cfg)
        assert li.data.shape[:2] == (8, 8)

    def test_path_always_empty(self):
        li = load_from_array(self._arr())
        assert li.path == ""

    def test_returns_loaded_image(self):
        assert isinstance(load_from_array(self._arr()), LoadedImage)


# ─── list_image_files extras ──────────────────────────────────────────────────

class TestListImageFilesExtra:
    def test_absolute_paths(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images)
        for f in files:
            assert os.path.isabs(f)

    def test_four_images(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images)
        assert len(files) == 4

    def test_png_only_filter(self, tmp_dir_mixed):
        files = list_image_files(tmp_dir_mixed, extensions=(".png",))
        assert all(f.endswith(".png") for f in files)
        assert len(files) == 3

    def test_jpg_only_filter(self, tmp_dir_mixed):
        files = list_image_files(tmp_dir_mixed, extensions=(".jpg",))
        assert len(files) == 1

    def test_multiple_extensions(self, tmp_dir_mixed):
        files = list_image_files(tmp_dir_mixed, extensions=(".png", ".jpg"))
        assert len(files) == 4

    def test_no_txt_files(self, tmp_dir_mixed):
        files = list_image_files(tmp_dir_mixed)
        assert not any(f.endswith(".txt") for f in files)

    def test_sorted_order(self, tmp_dir_with_images):
        files = list_image_files(tmp_dir_with_images)
        assert files == sorted(files)


# ─── batch_load extras ────────────────────────────────────────────────────────

class TestBatchLoadExtra:
    def test_single_path(self, tmp_png):
        result = batch_load([tmp_png])
        assert len(result) == 1
        assert result[0].image_id == 0

    def test_config_applied_to_all(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        cfg = LoadConfig(color_mode="gray")
        result = batch_load(paths, cfg=cfg)
        assert all(li.data.ndim == 2 for li in result)

    def test_ids_start_at_zero(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        result = batch_load(paths)
        assert result[0].image_id == 0

    def test_paths_stored(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        result = batch_load(paths)
        for li, p in zip(result, paths):
            assert li.path == p

    def test_normalize_applied(self, tmp_dir_with_images):
        paths = list_image_files(tmp_dir_with_images)
        cfg = LoadConfig(normalize=True)
        result = batch_load(paths, cfg=cfg)
        assert all(li.data.dtype == np.float32 for li in result)


# ─── load_from_directory extras ───────────────────────────────────────────────

class TestLoadFromDirectoryExtra:
    def test_four_images(self, tmp_dir_with_images):
        result = load_from_directory(tmp_dir_with_images)
        assert len(result) == 4

    def test_all_loaded_images(self, tmp_dir_with_images):
        for li in load_from_directory(tmp_dir_with_images):
            assert isinstance(li, LoadedImage)

    def test_config_gray_applied(self, tmp_dir_with_images):
        cfg = LoadConfig(color_mode="gray")
        result = load_from_directory(tmp_dir_with_images, cfg=cfg)
        assert all(li.data.ndim == 2 for li in result)

    def test_mixed_dir_loads_images(self, tmp_dir_mixed):
        result = load_from_directory(tmp_dir_mixed)
        # Only image files should be loaded (not .txt)
        assert all(isinstance(li, LoadedImage) for li in result)

    def test_ids_sequential(self, tmp_dir_with_images):
        result = load_from_directory(tmp_dir_with_images)
        ids = [li.image_id for li in result]
        assert ids == list(range(4))


# ─── resize_image extras ──────────────────────────────────────────────────────

class TestResizeImageExtra:
    def test_bgr_to_small(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        out = resize_image(img, (32, 32))
        assert out.shape == (32, 32, 3)

    def test_gray_to_large(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        out = resize_image(img, (64, 64))
        assert out.shape == (64, 64)

    def test_non_square_output(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = resize_image(img, (100, 50))
        assert out.shape == (50, 100, 3)

    def test_dtype_preserved_uint8(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        out = resize_image(img, (16, 16))
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        out = resize_image(img, (16, 16))
        assert out.min() >= 0
        assert out.max() <= 255

    def test_width_negative_raises(self):
        with pytest.raises(ValueError):
            resize_image(np.zeros((32, 32), dtype=np.uint8), (-1, 32))
