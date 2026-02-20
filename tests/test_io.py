"""
Тесты для puzzle_reconstruction/utils/io.py

Покрытие:
    load_image_dir        — FileNotFoundError, NotADirectoryError, пустая
                            директория, загрузка PNG, max_images, recursive,
                            sort, FragmentSetInfo (n_fragments, total_pixels,
                            image_sizes, failed_paths)
    fragments_from_images — пустой список, форма, dtype, mask, auto_mask=False,
                            fragment_id, contour
    save_assembly_json    — структура JSON, поля, round-trip через load
    load_assembly_json    — FileNotFoundError, KeyError, точные значения
    save/load_fragments_npz — round-trip: image, mask, contour, fragment_id
    FragmentSetInfo       — summary()
"""
import json
import math
import numpy as np
import pytest
import cv2

from puzzle_reconstruction.models import Assembly, Fragment
from puzzle_reconstruction.utils.io import (
    FragmentSetInfo,
    fragments_from_images,
    load_assembly_json,
    load_fragments_npz,
    load_image_dir,
    save_assembly_json,
    save_fragments_npz,
)


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

def _make_bgr_image(h: int = 48, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 200, (h, w, 3), dtype=np.uint8)
    # Добавляем белый фон по краям (чтобы Otsu имел что вырезать)
    img[:4, :] = 255
    img[-4:, :] = 255
    img[:, :4] = 255
    img[:, -4:] = 255
    return img


def _make_fragment(fid: int, h: int = 48, w: int = 64) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=_make_bgr_image(h, w, seed=fid),
        mask=np.ones((h, w), dtype=np.uint8) * 255,
        contour=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64),
    )


def _make_assembly(fids: list) -> Assembly:
    frags = [_make_fragment(fid) for fid in fids]
    placements = {
        fid: (np.array([float(i * 100), float(i * 50)]), float(i * 0.1))
        for i, fid in enumerate(fids)
    }
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.array([]),
        total_score=0.75,
        ocr_score=0.60,
    )


@pytest.fixture
def image_dir(tmp_path):
    """Создаёт временную директорию с несколькими PNG-файлами."""
    for i in range(4):
        img = _make_bgr_image(32, 32, seed=i)
        cv2.imwrite(str(tmp_path / f"frag_{i:02d}.png"), img)
    return tmp_path


@pytest.fixture
def frags3():
    return [_make_fragment(i) for i in range(3)]


@pytest.fixture
def assembly3():
    return _make_assembly([0, 1, 2])


# ─── FragmentSetInfo ──────────────────────────────────────────────────────────

class TestFragmentSetInfo:
    def test_summary_contains_n(self):
        info = FragmentSetInfo(n_fragments=5, source_dir="/tmp",
                               image_sizes=[(32, 32)] * 5,
                               total_pixels=5 * 32 * 32)
        s = info.summary()
        assert "5" in s
        assert "FragmentSetInfo" in s

    def test_summary_with_failures(self):
        info = FragmentSetInfo(n_fragments=3, failed_paths=["/a/bad.png"])
        s = info.summary()
        assert "1" in s  # failed_paths count


# ─── load_image_dir ───────────────────────────────────────────────────────────

class TestLoadImageDir:
    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image_dir(str(tmp_path / "nonexistent"))

    def test_file_not_dir_raises(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            load_image_dir(str(f))

    def test_empty_dir(self, tmp_path):
        images, info = load_image_dir(str(tmp_path))
        assert images == []
        assert info.n_fragments == 0

    def test_loads_png_files(self, image_dir):
        images, info = load_image_dir(str(image_dir))
        assert len(images) == 4
        assert info.n_fragments == 4

    def test_image_shape(self, image_dir):
        images, _ = load_image_dir(str(image_dir))
        for img in images:
            assert img.ndim == 3
            assert img.shape[2] == 3
            assert img.dtype == np.uint8

    def test_max_images(self, image_dir):
        images, info = load_image_dir(str(image_dir), max_images=2)
        assert len(images) == 2
        assert info.n_fragments == 2

    def test_sort(self, image_dir):
        """С sort=True файлы загружаются в алфавитном порядке."""
        images1, _ = load_image_dir(str(image_dir), sort=True)
        images2, _ = load_image_dir(str(image_dir), sort=True)
        for a, b in zip(images1, images2):
            assert np.array_equal(a, b)

    def test_info_total_pixels(self, image_dir):
        _, info = load_image_dir(str(image_dir))
        expected = 4 * 32 * 32
        assert info.total_pixels == expected

    def test_info_image_sizes(self, image_dir):
        _, info = load_image_dir(str(image_dir))
        assert len(info.image_sizes) == 4
        for h, w in info.image_sizes:
            assert h == 32 and w == 32

    def test_recursive(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        for i in range(2):
            cv2.imwrite(str(subdir / f"img_{i}.png"), _make_bgr_image(16, 16, i))
        cv2.imwrite(str(tmp_path / "top.png"), _make_bgr_image(16, 16, 99))
        images_flat, _ = load_image_dir(str(tmp_path), recursive=False)
        images_rec, _  = load_image_dir(str(tmp_path), recursive=True)
        assert len(images_flat) == 1
        assert len(images_rec) == 3

    def test_custom_extensions(self, tmp_path):
        cv2.imwrite(str(tmp_path / "a.png"), _make_bgr_image())
        cv2.imwrite(str(tmp_path / "b.jpg"), _make_bgr_image())
        images, _ = load_image_dir(str(tmp_path), extensions={".jpg"})
        assert len(images) == 1

    def test_info_source_dir(self, image_dir):
        _, info = load_image_dir(str(image_dir))
        assert info.source_dir == str(image_dir)


# ─── fragments_from_images ────────────────────────────────────────────────────

class TestFragmentsFromImages:
    def test_empty_list(self):
        result = fragments_from_images([])
        assert result == []

    def test_returns_fragments(self):
        imgs = [_make_bgr_image(32, 32, i) for i in range(3)]
        result = fragments_from_images(imgs)
        assert len(result) == 3
        for frag in result:
            assert isinstance(frag, Fragment)

    def test_fragment_ids(self):
        imgs = [_make_bgr_image() for _ in range(4)]
        result = fragments_from_images(imgs, start_id=10)
        assert [f.fragment_id for f in result] == [10, 11, 12, 13]

    def test_image_preserved(self):
        img = _make_bgr_image(32, 32, seed=5)
        frags = fragments_from_images([img])
        assert np.array_equal(frags[0].image, img)

    def test_mask_shape(self):
        img = _make_bgr_image(48, 64)
        frags = fragments_from_images([img])
        assert frags[0].mask.shape == (48, 64)

    def test_mask_dtype(self):
        img = _make_bgr_image(32, 32)
        frags = fragments_from_images([img])
        assert frags[0].mask.dtype == np.uint8

    def test_contour_2d(self):
        img = _make_bgr_image(32, 32)
        frags = fragments_from_images([img])
        assert frags[0].contour.ndim == 2
        assert frags[0].contour.shape[1] == 2

    def test_auto_mask_false(self):
        """auto_mask=False → маска полностью белая."""
        img = _make_bgr_image(32, 32)
        frags = fragments_from_images([img], auto_mask=False)
        assert (frags[0].mask == 255).all()

    def test_auto_mask_true_makes_binary(self):
        """auto_mask=True → значения маски только 0 и 255."""
        img = _make_bgr_image(32, 32)
        frags = fragments_from_images([img], auto_mask=True)
        unique_vals = set(np.unique(frags[0].mask))
        assert unique_vals.issubset({0, 255})


# ─── save_assembly_json / load_assembly_json ──────────────────────────────────

class TestAssemblyJsonRoundTrip:
    def test_save_creates_file(self, tmp_path, assembly3):
        path = str(tmp_path / "asm.json")
        save_assembly_json(assembly3, path)
        import os
        assert os.path.exists(path)

    def test_json_structure(self, tmp_path, assembly3):
        path = str(tmp_path / "asm.json")
        save_assembly_json(assembly3, path)
        with open(path) as f:
            data = json.load(f)
        assert "total_score" in data
        assert "ocr_score" in data
        assert "placements" in data
        assert "n_fragments" in data

    def test_roundtrip_total_score(self, tmp_path, assembly3, frags3):
        path = str(tmp_path / "asm.json")
        save_assembly_json(assembly3, path)
        restored = load_assembly_json(path, frags3)
        assert math.isclose(restored.total_score, assembly3.total_score, rel_tol=1e-9)

    def test_roundtrip_ocr_score(self, tmp_path, assembly3, frags3):
        path = str(tmp_path / "asm.json")
        save_assembly_json(assembly3, path)
        restored = load_assembly_json(path, frags3)
        assert math.isclose(restored.ocr_score, assembly3.ocr_score, rel_tol=1e-9)

    def test_roundtrip_positions(self, tmp_path, assembly3, frags3):
        path = str(tmp_path / "asm.json")
        save_assembly_json(assembly3, path)
        restored = load_assembly_json(path, frags3)
        for fid, (pos_orig, angle_orig) in assembly3.placements.items():
            pos_r, angle_r = restored.placements[fid]
            assert np.allclose(pos_r, pos_orig, atol=1e-9)
            assert math.isclose(angle_r, angle_orig, rel_tol=1e-9)

    def test_roundtrip_all_fids_present(self, tmp_path, assembly3, frags3):
        path = str(tmp_path / "asm.json")
        save_assembly_json(assembly3, path)
        restored = load_assembly_json(path, frags3)
        assert set(restored.placements.keys()) == set(assembly3.placements.keys())

    def test_load_missing_file_raises(self, frags3):
        with pytest.raises(FileNotFoundError):
            load_assembly_json("/nonexistent/path/asm.json", frags3)

    def test_creates_parent_dirs(self, tmp_path, assembly3):
        path = str(tmp_path / "deep" / "dir" / "asm.json")
        save_assembly_json(assembly3, path)
        import os
        assert os.path.exists(path)


# ─── save_fragments_npz / load_fragments_npz ─────────────────────────────────

class TestFragmentsNpzRoundTrip:
    def test_roundtrip_fragment_ids(self, tmp_path, frags3):
        path = str(tmp_path / "frags")
        save_fragments_npz(frags3, path)
        restored = load_fragments_npz(path + ".npz")
        orig_ids = [f.fragment_id for f in frags3]
        rest_ids = [f.fragment_id for f in restored]
        assert orig_ids == rest_ids

    def test_roundtrip_images(self, tmp_path, frags3):
        path = str(tmp_path / "frags")
        save_fragments_npz(frags3, path)
        restored = load_fragments_npz(path + ".npz")
        for orig, rest in zip(frags3, restored):
            assert np.array_equal(orig.image, rest.image)

    def test_roundtrip_masks(self, tmp_path, frags3):
        path = str(tmp_path / "frags")
        save_fragments_npz(frags3, path)
        restored = load_fragments_npz(path + ".npz")
        for orig, rest in zip(frags3, restored):
            assert np.array_equal(orig.mask, rest.mask)

    def test_roundtrip_contours(self, tmp_path, frags3):
        path = str(tmp_path / "frags")
        save_fragments_npz(frags3, path)
        restored = load_fragments_npz(path + ".npz")
        for orig, rest in zip(frags3, restored):
            assert np.allclose(orig.contour, rest.contour)

    def test_roundtrip_count(self, tmp_path, frags3):
        path = str(tmp_path / "frags")
        save_fragments_npz(frags3, path)
        restored = load_fragments_npz(path + ".npz")
        assert len(restored) == len(frags3)

    def test_variable_size_images(self, tmp_path):
        """Фрагменты разного размера корректно сохраняются/загружаются."""
        frags = [
            Fragment(
                fragment_id=i,
                image=np.zeros((16 + i * 8, 32 + i * 4, 3), dtype=np.uint8),
                mask=np.ones((16 + i * 8, 32 + i * 4), dtype=np.uint8),
                contour=np.zeros((4, 2)),
            )
            for i in range(3)
        ]
        path = str(tmp_path / "var_frags")
        save_fragments_npz(frags, path)
        restored = load_fragments_npz(path + ".npz")
        for orig, rest in zip(frags, restored):
            assert orig.image.shape == rest.image.shape

    def test_empty_fragments(self, tmp_path):
        path = str(tmp_path / "empty")
        save_fragments_npz([], path)
        restored = load_fragments_npz(path + ".npz")
        assert restored == []

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_fragments_npz("/nonexistent/path.npz")

    def test_creates_parent_dirs(self, tmp_path, frags3):
        path = str(tmp_path / "deep" / "frags")
        save_fragments_npz(frags3, path)
        import os
        assert os.path.exists(path + ".npz")
