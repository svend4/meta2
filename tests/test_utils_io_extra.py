"""Extra tests for puzzle_reconstruction/utils/io.py."""
from __future__ import annotations

import json
import numpy as np
import pytest

from puzzle_reconstruction.utils.io import (
    FragmentSetInfo,
    load_image_dir,
    fragments_from_images,
    save_fragments_npz,
    load_fragments_npz,
    IMAGE_EXTENSIONS,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=32, w=32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ─── FragmentSetInfo ──────────────────────────────────────────────────────────

class TestFragmentSetInfoExtra:
    def test_stores_n_fragments(self):
        info = FragmentSetInfo(n_fragments=5)
        assert info.n_fragments == 5

    def test_default_source_dir_none(self):
        assert FragmentSetInfo(n_fragments=0).source_dir is None

    def test_summary_contains_n(self):
        info = FragmentSetInfo(n_fragments=3)
        s = info.summary()
        assert "3" in s

    def test_failed_paths_default_empty(self):
        assert FragmentSetInfo(n_fragments=0).failed_paths == []


# ─── IMAGE_EXTENSIONS ─────────────────────────────────────────────────────────

class TestImageExtensionsExtra:
    def test_contains_png(self):
        assert ".png" in IMAGE_EXTENSIONS

    def test_contains_jpg(self):
        assert ".jpg" in IMAGE_EXTENSIONS


# ─── load_image_dir ───────────────────────────────────────────────────────────

class TestLoadImageDirExtra:
    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image_dir(str(tmp_path / "nonexistent"))

    def test_not_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            load_image_dir(str(f))

    def test_returns_tuple(self, tmp_path):
        result = load_image_dir(str(tmp_path))
        assert isinstance(result, tuple) and len(result) == 2

    def test_empty_dir_zero_fragments(self, tmp_path):
        images, info = load_image_dir(str(tmp_path))
        assert info.n_fragments == 0

    def test_info_source_dir_set(self, tmp_path):
        _, info = load_image_dir(str(tmp_path))
        assert info.source_dir is not None


# ─── fragments_from_images ────────────────────────────────────────────────────

class TestFragmentsFromImagesExtra:
    def test_returns_list(self):
        imgs = [_img()]
        result = fragments_from_images(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_img(), _img(), _img()]
        result = fragments_from_images(imgs)
        assert len(result) == 3

    def test_fragment_ids_start_at_start_id(self):
        imgs = [_img(), _img()]
        result = fragments_from_images(imgs, start_id=5)
        assert result[0].fragment_id == 5
        assert result[1].fragment_id == 6

    def test_empty_input(self):
        assert fragments_from_images([]) == []


# ─── save_fragments_npz / load_fragments_npz ─────────────────────────────────

class TestFragmentsNpzExtra:
    def test_save_creates_file(self, tmp_path):
        imgs = [_img(16, 16)]
        frags = fragments_from_images(imgs)
        path = str(tmp_path / "frags.npz")
        save_fragments_npz(frags, path)
        import os
        assert os.path.isfile(path)

    def test_roundtrip_count(self, tmp_path):
        imgs = [_img(16, 16), _img(24, 24)]
        frags = fragments_from_images(imgs)
        path = str(tmp_path / "frags.npz")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path)
        assert len(loaded) == 2

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fragments_npz(str(tmp_path / "missing.npz"))

    def test_fragment_id_preserved(self, tmp_path):
        imgs = [_img()]
        frags = fragments_from_images(imgs, start_id=7)
        path = str(tmp_path / "frags.npz")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path)
        assert loaded[0].fragment_id == 7
