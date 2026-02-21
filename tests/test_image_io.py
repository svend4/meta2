"""Тесты для puzzle_reconstruction/utils/image_io.py."""
import os
import numpy as np
import cv2
import pytest

from puzzle_reconstruction.utils.image_io import (
    ImageRecord,
    load_image,
    save_image,
    load_directory,
    filter_by_extension,
    parse_fragment_id,
    resize_to_max,
    batch_resize,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _save_png(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# ─── ImageRecord ──────────────────────────────────────────────────────────────

class TestImageRecord:
    def test_fields(self):
        img = _gray()
        r   = ImageRecord(path="/tmp/test.png", image=img)
        assert r.path == "/tmp/test.png"
        assert r.image is img

    def test_meta_default_empty(self):
        r = ImageRecord(path="x.png", image=_gray())
        assert isinstance(r.meta, dict)
        assert len(r.meta) == 0

    def test_meta_stored(self):
        r = ImageRecord(path="x.png", image=_gray(), meta={"fragment_id": 7})
        assert r.meta["fragment_id"] == 7

    def test_shape_property_gray(self):
        r = ImageRecord(path="x.png", image=_gray(32, 48))
        assert r.shape == (32, 48)

    def test_shape_property_bgr(self):
        r = ImageRecord(path="x.png", image=_bgr(40, 60))
        assert r.shape == (40, 60, 3)

    def test_repr(self):
        r = ImageRecord(path="/tmp/fragment_001.png", image=_gray(32, 32))
        s = repr(r)
        assert "ImageRecord" in s
        assert "fragment_001.png" in s


# ─── load_image ───────────────────────────────────────────────────────────────

class TestLoadImage:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/fragment.png")

    def test_returns_ndarray(self, tmp_path):
        p = str(tmp_path / "test.png")
        _save_png(p, _gray())
        img = load_image(p)
        assert isinstance(img, np.ndarray)

    def test_default_bgr_channels(self, tmp_path):
        p = str(tmp_path / "test.png")
        _save_png(p, _bgr())
        img = load_image(p)
        assert img.ndim == 3   # BGR

    def test_gray_flag(self, tmp_path):
        p = str(tmp_path / "gray.png")
        _save_png(p, _gray())
        img = load_image(p, flags=cv2.IMREAD_GRAYSCALE)
        assert img.ndim == 2

    def test_correct_shape(self, tmp_path):
        p = str(tmp_path / "shape.png")
        _save_png(p, _gray(40, 60))
        img = load_image(p, flags=cv2.IMREAD_GRAYSCALE)
        assert img.shape == (40, 60)

    def test_dtype_uint8(self, tmp_path):
        p = str(tmp_path / "dtype.png")
        _save_png(p, _gray())
        img = load_image(p)
        assert img.dtype == np.uint8


# ─── save_image ───────────────────────────────────────────────────────────────

class TestSaveImage:
    def test_returns_bool(self, tmp_path):
        p = str(tmp_path / "out.png")
        r = save_image(p, _gray())
        assert isinstance(r, bool)

    def test_returns_true_on_success(self, tmp_path):
        p = str(tmp_path / "out.png")
        assert save_image(p, _gray()) is True

    def test_file_exists_after_save(self, tmp_path):
        p = str(tmp_path / "out.png")
        save_image(p, _gray())
        assert os.path.isfile(p)

    def test_creates_nested_dirs(self, tmp_path):
        p = str(tmp_path / "sub" / "deep" / "out.png")
        save_image(p, _gray(), mkdir=True)
        assert os.path.isfile(p)

    def test_saved_loadable(self, tmp_path):
        img = _gray(32, 32)
        p   = str(tmp_path / "roundtrip.png")
        save_image(p, img)
        loaded = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.shape == (32, 32)

    def test_bgr_image(self, tmp_path):
        p = str(tmp_path / "bgr.png")
        assert save_image(p, _bgr()) is True


# ─── filter_by_extension ──────────────────────────────────────────────────────

class TestFilterByExtension:
    def test_empty_input(self):
        assert filter_by_extension([]) == []

    def test_keeps_png(self):
        paths = ["/a/b.png", "/c/d.jpg", "/e/f.txt"]
        result = filter_by_extension(paths, (".png",))
        assert result == ["/a/b.png"]

    def test_keeps_multiple(self):
        paths  = ["/a.png", "/b.jpg", "/c.bmp", "/d.py"]
        result = filter_by_extension(paths, (".png", ".jpg"))
        assert set(result) == {"/a.png", "/b.jpg"}

    def test_case_insensitive(self):
        paths  = ["/a.PNG", "/b.JPG", "/c.bmp"]
        result = filter_by_extension(paths, (".png", ".jpg"))
        assert "/a.PNG" in result
        assert "/b.JPG" in result

    def test_all_filtered_out(self):
        paths  = ["/a.docx", "/b.py"]
        result = filter_by_extension(paths, (".png",))
        assert result == []

    def test_returns_list(self):
        assert isinstance(filter_by_extension(["/x.png"], (".png",)), list)


# ─── parse_fragment_id ────────────────────────────────────────────────────────

class TestParseFragmentId:
    def test_simple_number(self):
        assert parse_fragment_id("fragment_042.png") == 42

    def test_img_prefix(self):
        assert parse_fragment_id("img003.jpg") == 3

    def test_no_digits_returns_none(self):
        assert parse_fragment_id("scan.png") is None

    def test_full_path(self):
        assert parse_fragment_id("/home/user/fragment_007.png") == 7

    def test_last_number_used(self):
        assert parse_fragment_id("group2_fragment_013.png") == 13

    def test_single_digit(self):
        assert parse_fragment_id("f5.png") == 5

    def test_zero_padded(self):
        assert parse_fragment_id("piece_0001.bmp") == 1

    def test_returns_int_or_none(self):
        r = parse_fragment_id("img001.png")
        assert isinstance(r, int)

    def test_extension_ignored(self):
        assert parse_fragment_id("x123") == 123


# ─── load_directory ───────────────────────────────────────────────────────────

class TestLoadDirectory:
    def test_not_a_directory_raises(self, tmp_path):
        p = str(tmp_path / "notadir.png")
        _save_png(p, _gray())
        with pytest.raises(NotADirectoryError):
            load_directory(p)

    def test_returns_list(self, tmp_path):
        d = str(tmp_path)
        _save_png(str(tmp_path / "frag_001.png"), _gray())
        result = load_directory(d)
        assert isinstance(result, list)

    def test_loads_images(self, tmp_path):
        _save_png(str(tmp_path / "a.png"), _gray(16, 16))
        _save_png(str(tmp_path / "b.png"), _gray(16, 16))
        result = load_directory(str(tmp_path))
        assert len(result) == 2

    def test_each_is_image_record(self, tmp_path):
        _save_png(str(tmp_path / "x.png"), _gray())
        for r in load_directory(str(tmp_path)):
            assert isinstance(r, ImageRecord)

    def test_sorted_by_name(self, tmp_path):
        for name in ["c.png", "a.png", "b.png"]:
            _save_png(str(tmp_path / name), _gray())
        result = load_directory(str(tmp_path))
        names  = [os.path.basename(r.path) for r in result]
        assert names == sorted(names)

    def test_fragment_id_in_meta(self, tmp_path):
        _save_png(str(tmp_path / "fragment_007.png"), _gray())
        result = load_directory(str(tmp_path))
        assert result[0].meta.get("fragment_id") == 7

    def test_non_image_skipped(self, tmp_path):
        # Create a non-image file
        with open(str(tmp_path / "readme.txt"), "w") as f:
            f.write("hello")
        _save_png(str(tmp_path / "img.png"), _gray())
        result = load_directory(str(tmp_path))
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = load_directory(str(tmp_path))
        assert result == []


# ─── resize_to_max ────────────────────────────────────────────────────────────

class TestResizeToMax:
    def test_returns_ndarray(self):
        assert isinstance(resize_to_max(_bgr(32, 32), max_side=64), np.ndarray)

    def test_smaller_image_not_enlarged(self):
        img    = _gray(32, 32)
        result = resize_to_max(img, max_side=64)
        assert result.shape == (32, 32)

    def test_larger_image_shrunk(self):
        img    = _gray(200, 200)
        result = resize_to_max(img, max_side=100)
        assert max(result.shape[:2]) <= 100

    def test_exact_max_side(self):
        img    = _gray(200, 100)
        result = resize_to_max(img, max_side=100)
        assert max(result.shape[:2]) == 100

    def test_aspect_ratio_preserved(self):
        img = _gray(200, 400)
        result = resize_to_max(img, max_side=200)
        h, w   = result.shape[:2]
        assert abs(w / h - 2.0) < 0.1

    def test_bgr_input(self):
        img    = _bgr(256, 256)
        result = resize_to_max(img, max_side=128)
        assert result.ndim == 3
        assert max(result.shape[:2]) <= 128

    def test_dtype_preserved(self):
        img    = _gray(200, 200)
        result = resize_to_max(img, max_side=100)
        assert result.dtype == np.uint8

    def test_exact_size_returns_copy(self):
        img    = _gray(64, 64)
        result = resize_to_max(img, max_side=64)
        assert result.shape == (64, 64)
        # Should be a copy, not same object
        assert result is not img

    def test_wide_image(self):
        img    = _gray(50, 500)
        result = resize_to_max(img, max_side=100)
        assert result.shape[1] == 100


# ─── batch_resize ─────────────────────────────────────────────────────────────

class TestBatchResize:
    def test_returns_list(self):
        imgs   = [_gray(200, 200), _bgr(300, 300)]
        result = batch_resize(imgs, max_side=100)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_ndarray(self):
        for r in batch_resize([_gray(200, 200)], max_side=100):
            assert isinstance(r, np.ndarray)

    def test_empty_list(self):
        assert batch_resize([]) == []

    def test_all_within_max_side(self):
        imgs = [_gray(500, 200), _gray(100, 600), _bgr(400, 400)]
        for r in batch_resize(imgs, max_side=256):
            assert max(r.shape[:2]) <= 256

    def test_small_images_not_enlarged(self):
        img    = _gray(32, 32)
        result = batch_resize([img], max_side=512)
        assert result[0].shape == (32, 32)

    def test_dtype_uint8(self):
        for r in batch_resize([_gray(200, 200)], max_side=100):
            assert r.dtype == np.uint8
