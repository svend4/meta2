"""Extra tests for puzzle_reconstruction.utils.image_io."""
import os
import cv2
import numpy as np
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _save_png(path, img):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    cv2.imwrite(path, img)


# ─── TestImageRecordExtra ────────────────────────────────────────────────────

class TestImageRecordExtra:
    def test_path_stored(self):
        r = ImageRecord(path="/tmp/frag.png", image=_gray())
        assert r.path == "/tmp/frag.png"

    def test_image_stored(self):
        img = _gray(32, 48)
        r = ImageRecord(path="x.png", image=img)
        assert r.image is img

    def test_meta_custom_keys(self):
        r = ImageRecord(path="x.png", image=_gray(),
                        meta={"fragment_id": 5, "scale": 0.5})
        assert r.meta["scale"] == pytest.approx(0.5)

    def test_shape_gray(self):
        r = ImageRecord(path="x.png", image=_gray(20, 30))
        assert r.shape == (20, 30)

    def test_shape_bgr(self):
        r = ImageRecord(path="x.png", image=_bgr(16, 24))
        assert r.shape == (16, 24, 3)

    def test_repr_has_path(self):
        r = ImageRecord(path="/tmp/frag_007.png", image=_gray())
        assert "frag_007.png" in repr(r)

    def test_repr_has_class_name(self):
        r = ImageRecord(path="x.png", image=_gray())
        assert "ImageRecord" in repr(r)

    def test_meta_default_empty_dict(self):
        r = ImageRecord(path="x.png", image=_gray())
        assert r.meta == {}

    def test_large_image(self):
        img = _gray(1024, 768)
        r = ImageRecord(path="big.png", image=img)
        assert r.shape == (1024, 768)


# ─── TestLoadImageExtra ─────────────────────────────────────────────────────

class TestLoadImageExtra:
    def test_load_bgr(self, tmp_path):
        p = str(tmp_path / "bgr.png")
        _save_png(p, _bgr())
        img = load_image(p)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_gray_flag(self, tmp_path):
        p = str(tmp_path / "gray.png")
        _save_png(p, _gray())
        img = load_image(p, flags=cv2.IMREAD_GRAYSCALE)
        assert img.ndim == 2

    def test_load_shape_correct(self, tmp_path):
        p = str(tmp_path / "shape.png")
        _save_png(p, _gray(32, 48))
        img = load_image(p, flags=cv2.IMREAD_GRAYSCALE)
        assert img.shape == (32, 48)

    def test_load_dtype_uint8(self, tmp_path):
        p = str(tmp_path / "dtype.png")
        _save_png(p, _bgr())
        img = load_image(p)
        assert img.dtype == np.uint8

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/file/path.png")

    def test_values_roundtrip_gray(self, tmp_path):
        p = str(tmp_path / "rt.png")
        orig = _gray(16, 16, val=77)
        _save_png(p, orig)
        loaded = load_image(p, flags=cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(loaded, orig)

    def test_small_image(self, tmp_path):
        p = str(tmp_path / "small.png")
        _save_png(p, _gray(4, 4))
        img = load_image(p, flags=cv2.IMREAD_GRAYSCALE)
        assert img.shape == (4, 4)


# ─── TestSaveImageExtra ─────────────────────────────────────────────────────

class TestSaveImageExtra:
    def test_returns_true_gray(self, tmp_path):
        p = str(tmp_path / "g.png")
        assert save_image(p, _gray()) is True

    def test_returns_true_bgr(self, tmp_path):
        p = str(tmp_path / "b.png")
        assert save_image(p, _bgr()) is True

    def test_file_exists(self, tmp_path):
        p = str(tmp_path / "exists.png")
        save_image(p, _gray())
        assert os.path.isfile(p)

    def test_nested_mkdir(self, tmp_path):
        p = str(tmp_path / "a" / "b" / "c.png")
        save_image(p, _gray(), mkdir=True)
        assert os.path.isfile(p)

    def test_roundtrip_bgr(self, tmp_path):
        p = str(tmp_path / "rt.png")
        orig = _bgr(16, 16)
        save_image(p, orig)
        loaded = cv2.imread(p)
        assert loaded is not None
        assert loaded.shape == orig.shape

    def test_returns_bool_type(self, tmp_path):
        r = save_image(str(tmp_path / "x.png"), _gray())
        assert isinstance(r, bool)


# ─── TestFilterByExtensionExtra ─────────────────────────────────────────────

class TestFilterByExtensionExtra:
    def test_single_png(self):
        assert filter_by_extension(["/a.png"], (".png",)) == ["/a.png"]

    def test_jpg_bmp(self):
        paths = ["/a.jpg", "/b.bmp", "/c.txt"]
        result = filter_by_extension(paths, (".jpg", ".bmp"))
        assert set(result) == {"/a.jpg", "/b.bmp"}

    def test_case_insensitive_png(self):
        result = filter_by_extension(["/A.PNG"], (".png",))
        assert "/A.PNG" in result

    def test_no_match_empty(self):
        assert filter_by_extension(["/a.docx"], (".png",)) == []

    def test_empty_paths_empty(self):
        assert filter_by_extension([], (".png",)) == []

    def test_multiple_extensions(self):
        paths = ["/a.png", "/b.jpg", "/c.jpeg", "/d.bmp"]
        result = filter_by_extension(paths, (".png", ".jpg", ".jpeg", ".bmp"))
        assert len(result) == 4

    def test_preserves_order(self):
        paths = ["/c.png", "/a.png", "/b.png"]
        result = filter_by_extension(paths, (".png",))
        assert result == paths

    def test_no_extension_not_matched(self):
        result = filter_by_extension(["/noext"], (".png",))
        assert result == []


# ─── TestParseFragmentIdExtra ───────────────────────────────────────────────

class TestParseFragmentIdExtra:
    def test_three_digits(self):
        assert parse_fragment_id("fragment_123.png") == 123

    def test_no_digits_none(self):
        assert parse_fragment_id("no_digits.png") is None

    def test_path_with_slash(self):
        assert parse_fragment_id("/data/pieces/img_099.jpg") == 99

    def test_trailing_number(self):
        assert parse_fragment_id("piece42") == 42

    def test_large_number(self):
        assert parse_fragment_id("frag_99999.png") == 99999

    def test_zero(self):
        assert parse_fragment_id("fragment_000.png") == 0

    def test_returns_int(self):
        result = parse_fragment_id("img001.png")
        assert isinstance(result, int)

    def test_multiple_numbers_last(self):
        assert parse_fragment_id("set2_piece_013.bmp") == 13


# ─── TestLoadDirectoryExtra ─────────────────────────────────────────────────

class TestLoadDirectoryExtra:
    def test_empty_dir(self, tmp_path):
        result = load_directory(str(tmp_path))
        assert result == []

    def test_one_image(self, tmp_path):
        _save_png(str(tmp_path / "frag_001.png"), _gray())
        result = load_directory(str(tmp_path))
        assert len(result) == 1

    def test_returns_image_record_list(self, tmp_path):
        _save_png(str(tmp_path / "frag_001.png"), _gray())
        for r in load_directory(str(tmp_path)):
            assert isinstance(r, ImageRecord)

    def test_sorted_by_name(self, tmp_path):
        for name in ["c.png", "a.png", "b.png"]:
            _save_png(str(tmp_path / name), _gray())
        names = [os.path.basename(r.path) for r in load_directory(str(tmp_path))]
        assert names == sorted(names)

    def test_txt_skipped(self, tmp_path):
        with open(str(tmp_path / "notes.txt"), "w") as f:
            f.write("skip")
        _save_png(str(tmp_path / "img.png"), _gray())
        result = load_directory(str(tmp_path))
        assert len(result) == 1

    def test_meta_fragment_id(self, tmp_path):
        _save_png(str(tmp_path / "fragment_007.png"), _gray())
        result = load_directory(str(tmp_path))
        assert result[0].meta.get("fragment_id") == 7

    def test_not_dir_raises(self, tmp_path):
        p = str(tmp_path / "file.png")
        _save_png(p, _gray())
        with pytest.raises(NotADirectoryError):
            load_directory(p)


# ─── TestResizeToMaxExtra ───────────────────────────────────────────────────

class TestResizeToMaxExtra:
    def test_square_image_shrunk(self):
        img = _gray(200, 200)
        result = resize_to_max(img, max_side=100)
        assert result.shape == (100, 100)

    def test_tall_image(self):
        img = _gray(400, 100)
        result = resize_to_max(img, max_side=200)
        assert result.shape[0] == 200
        assert result.shape[1] == 50

    def test_wide_image(self):
        img = _gray(100, 400)
        result = resize_to_max(img, max_side=200)
        assert result.shape[0] == 50
        assert result.shape[1] == 200

    def test_no_enlarge_small_image(self):
        img = _gray(32, 32)
        result = resize_to_max(img, max_side=100)
        assert result.shape == (32, 32)

    def test_bgr_channels_preserved(self):
        img = _bgr(200, 200)
        result = resize_to_max(img, max_side=100)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_dtype_uint8(self):
        result = resize_to_max(_gray(200, 200), max_side=100)
        assert result.dtype == np.uint8

    def test_returns_ndarray(self):
        assert isinstance(resize_to_max(_gray(200, 200), max_side=100), np.ndarray)

    def test_exact_max_side(self):
        img = _gray(300, 200)
        result = resize_to_max(img, max_side=150)
        assert result.shape[0] == 150


# ─── TestBatchResizeExtra ───────────────────────────────────────────────────

class TestBatchResizeExtra:
    def test_empty_list(self):
        assert batch_resize([]) == []

    def test_single_gray(self):
        result = batch_resize([_gray(200, 200)], max_side=100)
        assert len(result) == 1
        assert result[0].shape == (100, 100)

    def test_three_images(self):
        imgs = [_gray(300, 300), _bgr(200, 400), _gray(100, 50)]
        result = batch_resize(imgs, max_side=150)
        assert len(result) == 3

    def test_all_within_max(self):
        imgs = [_gray(500, 200), _gray(300, 600)]
        for r in batch_resize(imgs, max_side=100):
            assert max(r.shape[:2]) <= 100

    def test_small_not_enlarged(self):
        img = _gray(32, 32)
        result = batch_resize([img], max_side=200)
        assert result[0].shape == (32, 32)

    def test_all_dtype_uint8(self):
        imgs = [_gray(200, 200), _bgr(300, 300)]
        for r in batch_resize(imgs, max_side=100):
            assert r.dtype == np.uint8

    def test_five_images(self):
        imgs = [_gray(100 + i * 50, 100 + i * 50) for i in range(5)]
        result = batch_resize(imgs, max_side=100)
        assert len(result) == 5
