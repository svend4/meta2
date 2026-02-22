"""Тесты для puzzle_reconstruction.io.image_loader."""
import os
import tempfile
import pytest
import numpy as np
import cv2
from puzzle_reconstruction.io.image_loader import (
    LoadConfig,
    LoadedImage,
    load_from_array,
    list_image_files,
    batch_load,
    resize_image,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray_arr(h: int = 32, w: int = 32, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rgb_arr(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _save_png(arr: np.ndarray, path: str) -> None:
    cv2.imwrite(path, arr)


# ─── TestLoadConfig ───────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_defaults(self):
        cfg = LoadConfig()
        assert cfg.color_mode == "bgr"
        assert cfg.target_size is None
        assert cfg.normalize is False

    def test_color_mode_gray_ok(self):
        cfg = LoadConfig(color_mode="gray")
        assert cfg.color_mode == "gray"

    def test_color_mode_rgb_ok(self):
        cfg = LoadConfig(color_mode="rgb")
        assert cfg.color_mode == "rgb"

    def test_color_mode_bgr_ok(self):
        cfg = LoadConfig(color_mode="bgr")
        assert cfg.color_mode == "bgr"

    def test_color_mode_invalid_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(color_mode="hsv")

    def test_color_mode_empty_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(color_mode="")

    def test_target_size_none_ok(self):
        cfg = LoadConfig(target_size=None)
        assert cfg.target_size is None

    def test_target_size_valid_ok(self):
        cfg = LoadConfig(target_size=(64, 64))
        assert cfg.target_size == (64, 64)

    def test_target_size_zero_w_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(0, 64))

    def test_target_size_zero_h_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(64, 0))

    def test_target_size_neg_raises(self):
        with pytest.raises(ValueError):
            LoadConfig(target_size=(-1, 64))

    def test_normalize_true_ok(self):
        cfg = LoadConfig(normalize=True)
        assert cfg.normalize is True


# ─── TestLoadedImage ──────────────────────────────────────────────────────────

class TestLoadedImage:
    def _make(self, arr=None, image_id=0) -> LoadedImage:
        if arr is None:
            arr = _gray_arr()
        return LoadedImage(data=arr, path="test.png", image_id=image_id,
                           original_size=(arr.shape[1], arr.shape[0]))

    def test_basic(self):
        li = self._make()
        assert li.image_id == 0

    def test_shape(self):
        li = self._make(_gray_arr(48, 64))
        assert li.shape == (48, 64)

    def test_size(self):
        li = self._make(_gray_arr(48, 64))
        assert li.size == (64, 48)

    def test_len(self):
        li = self._make(_gray_arr(32, 32))
        assert len(li) == 32 * 32

    def test_image_id_neg_raises(self):
        with pytest.raises(ValueError):
            LoadedImage(data=_gray_arr(), path="", image_id=-1,
                        original_size=(32, 32))

    def test_1d_data_raises(self):
        with pytest.raises(ValueError):
            LoadedImage(data=np.zeros(32), path="", image_id=0,
                        original_size=(32, 1))

    def test_4d_data_raises(self):
        with pytest.raises(ValueError):
            LoadedImage(data=np.zeros((2, 2, 3, 2)), path="", image_id=0,
                        original_size=(2, 2))

    def test_3d_data_ok(self):
        li = self._make(_rgb_arr())
        assert li.shape == (32, 32, 3)

    def test_path_stored(self):
        li = self._make()
        assert li.path == "test.png"

    def test_original_size_stored(self):
        arr = _gray_arr(48, 64)
        li = LoadedImage(data=arr, path="", image_id=0,
                         original_size=(64, 48))
        assert li.original_size == (64, 48)


# ─── TestLoadFromArray ────────────────────────────────────────────────────────

class TestLoadFromArray:
    def test_returns_loaded_image(self):
        li = load_from_array(_gray_arr())
        assert isinstance(li, LoadedImage)

    def test_image_id_stored(self):
        li = load_from_array(_gray_arr(), image_id=5)
        assert li.image_id == 5

    def test_neg_image_id_raises(self):
        with pytest.raises(ValueError):
            load_from_array(_gray_arr(), image_id=-1)

    def test_path_empty(self):
        li = load_from_array(_gray_arr())
        assert li.path == ""

    def test_gray_color_mode_ok(self):
        cfg = LoadConfig(color_mode="gray")
        li = load_from_array(_rgb_arr(), cfg=cfg)
        assert isinstance(li, LoadedImage)

    def test_rgb_color_mode_ok(self):
        cfg = LoadConfig(color_mode="rgb")
        li = load_from_array(_rgb_arr(), cfg=cfg)
        assert li.data.ndim == 3

    def test_normalize_float32(self):
        cfg = LoadConfig(normalize=True)
        li = load_from_array(_gray_arr(), cfg=cfg)
        assert li.data.dtype == np.float32

    def test_normalize_values_in_range(self):
        cfg = LoadConfig(normalize=True)
        li = load_from_array(_gray_arr(val=200), cfg=cfg)
        assert li.data.max() <= 1.0 + 1e-6

    def test_no_normalize_uint8(self):
        li = load_from_array(_gray_arr())
        assert li.data.dtype == np.uint8

    def test_target_size_applied(self):
        cfg = LoadConfig(target_size=(16, 24))
        li = load_from_array(_gray_arr(32, 32), cfg=cfg)
        assert li.data.shape[:2] == (24, 16)

    def test_1d_array_raises(self):
        with pytest.raises(ValueError):
            load_from_array(np.zeros(32))

    def test_4d_array_raises(self):
        with pytest.raises(ValueError):
            load_from_array(np.zeros((4, 4, 3, 2)))

    def test_original_size_stored(self):
        arr = _gray_arr(48, 64)
        li = load_from_array(arr)
        assert li.original_size == (64, 48)


# ─── TestLoadFromFile (integration) ───────────────────────────────────────────

class TestLoadFromFile:
    def test_file_not_found_raises(self):
        from puzzle_reconstruction.io.image_loader import load_image
        with pytest.raises(FileNotFoundError):
            load_image("/tmp/nonexistent_12345.png")

    def test_load_saved_png(self):
        from puzzle_reconstruction.io.image_loader import load_image
        arr = _gray_arr()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_png(arr, path)
            li = load_image(path)
            assert isinstance(li, LoadedImage)
        finally:
            os.unlink(path)

    def test_image_id_stored_file(self):
        from puzzle_reconstruction.io.image_loader import load_image
        arr = _gray_arr()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            _save_png(arr, path)
            li = load_image(path, image_id=7)
            assert li.image_id == 7
        finally:
            os.unlink(path)


# ─── TestListImageFiles ───────────────────────────────────────────────────────

class TestListImageFiles:
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            assert list_image_files(d) == []

    def test_finds_png_files(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                _save_png(_gray_arr(), os.path.join(d, f"img_{i}.png"))
            files = list_image_files(d, extensions=(".png",))
            assert len(files) == 3

    def test_sorted_output(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                _save_png(_gray_arr(), os.path.join(d, f"img_{i}.png"))
            files = list_image_files(d)
            assert files == sorted(files)

    def test_non_image_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            _save_png(_gray_arr(), os.path.join(d, "img.png"))
            with open(os.path.join(d, "doc.txt"), "w") as f:
                f.write("text")
            files = list_image_files(d)
            assert all(f.endswith(".png") for f in files)

    def test_not_directory_raises(self):
        with pytest.raises(NotADirectoryError):
            list_image_files("/tmp/nonexistent_dir_12345")

    def test_recursive_false_no_subdir(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub")
            os.makedirs(sub)
            _save_png(_gray_arr(), os.path.join(sub, "img.png"))
            files = list_image_files(d, recursive=False)
            assert files == []

    def test_recursive_true_finds_subdir(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub")
            os.makedirs(sub)
            _save_png(_gray_arr(), os.path.join(sub, "img.png"))
            files = list_image_files(d, recursive=True)
            assert len(files) == 1


# ─── TestBatchLoad ────────────────────────────────────────────────────────────

class TestBatchLoad:
    def _make_files(self, tmpdir: str, n: int = 3) -> list:
        paths = []
        for i in range(n):
            p = os.path.join(tmpdir, f"img_{i}.png")
            _save_png(_gray_arr(), p)
            paths.append(p)
        return paths

    def test_returns_list(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 3)
            result = batch_load(paths)
            assert isinstance(result, list)

    def test_length_matches(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 4)
            assert len(batch_load(paths)) == 4

    def test_empty_list(self):
        assert batch_load([]) == []

    def test_all_loaded_images(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 2)
            for li in batch_load(paths):
                assert isinstance(li, LoadedImage)

    def test_image_ids_sequential(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._make_files(d, 3)
            for i, li in enumerate(batch_load(paths)):
                assert li.image_id == i

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            batch_load(["/tmp/nonexistent_file_999.png"])


# ─── TestResizeImage ──────────────────────────────────────────────────────────

class TestResizeImage:
    def test_returns_ndarray(self):
        result = resize_image(_gray_arr(), (16, 16))
        assert isinstance(result, np.ndarray)

    def test_target_size_applied(self):
        result = resize_image(_gray_arr(32, 32), (16, 24))
        assert result.shape[:2] == (24, 16)

    def test_rgb_resized(self):
        result = resize_image(_rgb_arr(32, 32), (16, 16))
        assert result.shape == (16, 16, 3)

    def test_dtype_preserved(self):
        arr = _gray_arr()
        result = resize_image(arr, (16, 16))
        assert result.dtype == arr.dtype

    def test_same_size_ok(self):
        arr = _gray_arr(32, 32)
        result = resize_image(arr, (32, 32))
        assert result.shape == (32, 32)

    def test_upscale_ok(self):
        result = resize_image(_gray_arr(16, 16), (64, 64))
        assert result.shape == (64, 64)
