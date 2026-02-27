"""Tests for puzzle_reconstruction.utils.image_io
Using only numpy and pytest (no cv2 in test code).
"""
import os
import tempfile
import numpy as np
import pytest
from puzzle_reconstruction.utils.image_io import (
    ImageRecord, filter_by_extension, parse_fragment_id,
    resize_to_max, batch_resize, load_directory,
    _SUPPORTED_EXT,
)

np.random.seed(42)


# ── ImageRecord ───────────────────────────────────────────────────────────────

def test_image_record_shape_property():
    img = np.zeros((100, 80, 3), dtype=np.uint8)
    rec = ImageRecord(path="/tmp/test.png", image=img)
    assert rec.shape == (100, 80, 3)


def test_image_record_shape_grayscale():
    img = np.zeros((50, 60), dtype=np.uint8)
    rec = ImageRecord(path="/tmp/gray.png", image=img)
    assert rec.shape == (50, 60)


def test_image_record_meta_default_empty():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    rec = ImageRecord(path="/tmp/a.png", image=img)
    assert rec.meta == {}


def test_image_record_repr_contains_name():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    rec = ImageRecord(path="/tmp/myphoto.png", image=img)
    r = repr(rec)
    assert "myphoto.png" in r


# ── filter_by_extension ───────────────────────────────────────────────────────

def test_filter_by_extension_keeps_png():
    paths = ["/a/b/img.png", "/a/b/img.jpg", "/a/b/doc.pdf"]
    result = filter_by_extension(paths, extensions=(".png",))
    assert result == ["/a/b/img.png"]


def test_filter_by_extension_case_insensitive():
    paths = ["/a/IMG.PNG", "/a/img.png", "/a/IMG.Png"]
    result = filter_by_extension(paths, extensions=(".png",))
    assert len(result) == 3


def test_filter_by_extension_empty_list():
    assert filter_by_extension([], extensions=(".png",)) == []


def test_filter_by_extension_default_ext():
    paths = ["/a/img.png", "/a/img.jpg", "/a/img.bmp", "/a/doc.txt"]
    result = filter_by_extension(paths)
    assert "/a/doc.txt" not in result
    assert len(result) == 3


def test_filter_by_extension_tiff():
    paths = ["/a/scan.tiff", "/a/scan.tif", "/a/scan.txt"]
    result = filter_by_extension(paths, extensions=(".tiff", ".tif"))
    assert len(result) == 2


# ── parse_fragment_id ─────────────────────────────────────────────────────────

def test_parse_fragment_id_basic():
    assert parse_fragment_id("fragment_042.png") == 42


def test_parse_fragment_id_leading_zeros():
    assert parse_fragment_id("img003.jpg") == 3


def test_parse_fragment_id_no_digits():
    assert parse_fragment_id("scan.png") is None


def test_parse_fragment_id_multiple_groups():
    # Should return last group
    assert parse_fragment_id("part2_fragment_007.jpg") == 7


def test_parse_fragment_id_full_path():
    assert parse_fragment_id("/some/path/fragment_100.png") == 100


def test_parse_fragment_id_only_digits():
    assert parse_fragment_id("123.png") == 123


# ── resize_to_max ─────────────────────────────────────────────────────────────

def test_resize_to_max_already_small():
    img = np.zeros((100, 80, 3), dtype=np.uint8)
    out = resize_to_max(img, max_side=200)
    assert out.shape == (100, 80, 3)


def test_resize_to_max_large_landscape():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    out = resize_to_max(img, max_side=100)
    h, w = out.shape[:2]
    assert max(h, w) <= 100


def test_resize_to_max_large_portrait():
    img = np.zeros((400, 200, 3), dtype=np.uint8)
    out = resize_to_max(img, max_side=100)
    h, w = out.shape[:2]
    assert max(h, w) <= 100


def test_resize_to_max_aspect_ratio():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    out = resize_to_max(img, max_side=100)
    h, w = out.shape[:2]
    # aspect ratio preserved approximately
    assert abs(w / h - 400 / 200) < 0.1


def test_resize_to_max_returns_copy():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out = resize_to_max(img, max_side=200)
    assert out is not img


def test_resize_to_max_grayscale():
    img = np.zeros((200, 400), dtype=np.uint8)
    out = resize_to_max(img, max_side=100)
    assert out.ndim == 2
    assert max(out.shape) <= 100


# ── batch_resize ──────────────────────────────────────────────────────────────

def test_batch_resize_returns_same_count():
    images = [np.zeros((300, 400, 3), dtype=np.uint8) for _ in range(5)]
    out = batch_resize(images, max_side=100)
    assert len(out) == 5


def test_batch_resize_all_small():
    images = [np.zeros((300, 400, 3), dtype=np.uint8) for _ in range(3)]
    out = batch_resize(images, max_side=100)
    assert all(max(img.shape[:2]) <= 100 for img in out)


def test_batch_resize_empty_list():
    out = batch_resize([], max_side=100)
    assert out == []


# ── load_directory errors ─────────────────────────────────────────────────────

def test_load_directory_not_a_directory_raises():
    with pytest.raises(NotADirectoryError):
        load_directory("/nonexistent/path/xyz")


def test_load_directory_file_path_raises():
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        with pytest.raises(NotADirectoryError):
            load_directory(f.name)


def test_load_directory_empty_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_directory(tmpdir)
        assert result == []
