"""Extra tests for puzzle_reconstruction/utils/image_io.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.image_io import (
    ImageRecord,
    filter_by_extension,
    parse_fragment_id,
    resize_to_max,
    batch_resize,
    save_image,
    load_image,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=32, w=32, c=3) -> np.ndarray:
    return np.zeros((h, w, c), dtype=np.uint8)


def _record(path="img.png", h=32, w=32) -> ImageRecord:
    return ImageRecord(path=path, image=_img(h, w))


# ─── ImageRecord ──────────────────────────────────────────────────────────────

class TestImageRecordExtra:
    def test_stores_path(self):
        r = _record(path="/tmp/test.png")
        assert r.path == "/tmp/test.png"

    def test_shape_property(self):
        r = _record(h=64, w=48)
        assert r.shape == (64, 48, 3)

    def test_repr_contains_name(self):
        r = _record(path="fragment_001.png")
        assert "fragment_001.png" in repr(r)

    def test_meta_default_empty(self):
        r = _record()
        assert r.meta == {}

    def test_meta_stored(self):
        r = ImageRecord(path="x.png", image=_img(), meta={"fragment_id": 5})
        assert r.meta["fragment_id"] == 5


# ─── filter_by_extension ──────────────────────────────────────────────────────

class TestFilterByExtensionExtra:
    def test_keeps_png(self):
        paths = ["a.png", "b.jpg", "c.txt"]
        result = filter_by_extension(paths, (".png",))
        assert result == ["a.png"]

    def test_case_insensitive(self):
        paths = ["A.PNG", "b.jpg"]
        result = filter_by_extension(paths, (".png",))
        assert "A.PNG" in result

    def test_empty_input(self):
        assert filter_by_extension([]) == []

    def test_all_supported_default(self):
        paths = ["a.png", "b.bmp", "c.csv"]
        result = filter_by_extension(paths)
        assert "c.csv" not in result and len(result) == 2


# ─── parse_fragment_id ────────────────────────────────────────────────────────

class TestParseFragmentIdExtra:
    def test_fragment_042(self):
        assert parse_fragment_id("fragment_042.png") == 42

    def test_img003(self):
        assert parse_fragment_id("img003.jpg") == 3

    def test_no_digits_returns_none(self):
        assert parse_fragment_id("scan.png") is None

    def test_multiple_numbers_last(self):
        assert parse_fragment_id("img2_piece007.png") == 7

    def test_full_path(self):
        assert parse_fragment_id("/data/fragment_005.png") == 5


# ─── resize_to_max ────────────────────────────────────────────────────────────

class TestResizeToMaxExtra:
    def test_small_image_unchanged_shape(self):
        img = _img(64, 64)
        result = resize_to_max(img, max_side=128)
        assert result.shape[:2] == (64, 64)

    def test_large_image_resized(self):
        img = _img(2048, 1024)
        result = resize_to_max(img, max_side=512)
        assert max(result.shape[:2]) <= 512

    def test_returns_copy_when_small(self):
        img = _img(32, 32)
        result = resize_to_max(img, max_side=100)
        assert result is not img

    def test_aspect_ratio_preserved(self):
        img = _img(800, 400)
        result = resize_to_max(img, max_side=400)
        h, w = result.shape[:2]
        assert abs(w / h - 0.5) < 0.05


# ─── batch_resize ─────────────────────────────────────────────────────────────

class TestBatchResizeExtra:
    def test_length_preserved(self):
        imgs = [_img(64, 64), _img(128, 128), _img(32, 32)]
        result = batch_resize(imgs, max_side=50)
        assert len(result) == 3

    def test_each_resized(self):
        imgs = [_img(200, 200)]
        result = batch_resize(imgs, max_side=100)
        assert max(result[0].shape[:2]) <= 100

    def test_empty_input(self):
        assert batch_resize([]) == []


# ─── save_image / load_image (roundtrip via tmp_path) ────────────────────────

class TestSaveLoadImageExtra:
    def test_save_creates_file(self, tmp_path):
        path = str(tmp_path / "test.png")
        img = _img(16, 16)
        assert save_image(path, img) is True

    def test_load_returns_array(self, tmp_path):
        path = str(tmp_path / "img.png")
        img = _img(16, 16)
        save_image(path, img)
        loaded = load_image(path)
        assert isinstance(loaded, np.ndarray)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(str(tmp_path / "nonexistent.png"))

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "img.png")
        save_image(path, _img())
        import os
        assert os.path.isfile(path)
