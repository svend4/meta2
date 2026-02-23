"""Extra tests for puzzle_reconstruction/utils/io.py"""
import json
import os
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=32, w=32, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 200, (h, w, 3), dtype=np.uint8)
    img[:4, :] = 255
    img[-4:, :] = 255
    img[:, :4] = 255
    img[:, -4:] = 255
    return img


def _frag(fid=0, h=32, w=32) -> Fragment:
    return Fragment(
        fragment_id=fid,
        image=_bgr(h, w, seed=fid),
        mask=np.ones((h, w), dtype=np.uint8) * 255,
        contour=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64),
    )


def _assembly(fids=(0, 1)) -> Assembly:
    frags = [_frag(fid) for fid in fids]
    placements = {
        fid: (np.array([float(i * 80), 0.0]), float(i * 0.2))
        for i, fid in enumerate(fids)
    }
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.zeros((1, 1)),
        total_score=0.65,
        ocr_score=0.50,
    )


# ─── TestFragmentSetInfoExtra ─────────────────────────────────────────────────

class TestFragmentSetInfoExtra:
    def test_total_pixels_stored(self):
        info = FragmentSetInfo(n_fragments=2, total_pixels=2048)
        assert info.total_pixels == 2048

    def test_failed_paths_default_empty(self):
        info = FragmentSetInfo(n_fragments=3)
        assert info.failed_paths == []

    def test_image_sizes_default_empty(self):
        info = FragmentSetInfo(n_fragments=0)
        assert info.image_sizes == []

    def test_summary_contains_class_name(self):
        info = FragmentSetInfo(n_fragments=1)
        s = info.summary()
        assert "FragmentSetInfo" in s

    def test_multiple_failed_paths(self):
        info = FragmentSetInfo(n_fragments=5,
                               failed_paths=["/a.png", "/b.png", "/c.png"])
        s = info.summary()
        assert "3" in s


# ─── TestLoadImageDirExtra ────────────────────────────────────────────────────

class TestLoadImageDirExtra:
    def test_source_dir_in_info(self, tmp_path):
        cv2.imwrite(str(tmp_path / "img.png"), _bgr())
        _, info = load_image_dir(str(tmp_path))
        assert info.source_dir == str(tmp_path)

    def test_max_images_1(self, tmp_path):
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"img_{i}.png"), _bgr(seed=i))
        images, info = load_image_dir(str(tmp_path), max_images=1)
        assert len(images) == 1
        assert info.n_fragments == 1

    def test_image_sizes_populated(self, tmp_path):
        cv2.imwrite(str(tmp_path / "a.png"), _bgr(16, 24))
        _, info = load_image_dir(str(tmp_path))
        assert len(info.image_sizes) == 1
        assert info.image_sizes[0] == (16, 24)

    def test_jpg_extension(self, tmp_path):
        cv2.imwrite(str(tmp_path / "a.jpg"), _bgr())
        cv2.imwrite(str(tmp_path / "b.png"), _bgr(seed=1))
        images, _ = load_image_dir(str(tmp_path), extensions={".jpg"})
        assert len(images) == 1

    def test_images_are_ndarray(self, tmp_path):
        cv2.imwrite(str(tmp_path / "x.png"), _bgr())
        images, _ = load_image_dir(str(tmp_path))
        assert isinstance(images[0], np.ndarray)


# ─── TestFragmentsFromImagesExtra ─────────────────────────────────────────────

class TestFragmentsFromImagesExtra:
    def test_start_id_5(self):
        imgs = [_bgr(seed=i) for i in range(3)]
        result = fragments_from_images(imgs, start_id=5)
        assert [f.fragment_id for f in result] == [5, 6, 7]

    def test_single_image(self):
        result = fragments_from_images([_bgr()])
        assert len(result) == 1
        assert result[0].fragment_id == 0

    def test_contour_nonempty(self):
        result = fragments_from_images([_bgr()])
        assert len(result[0].contour) > 0

    def test_mask_binary(self):
        result = fragments_from_images([_bgr()], auto_mask=True)
        unique = set(np.unique(result[0].mask).tolist())
        assert unique.issubset({0, 255})

    def test_auto_mask_false_all_white(self):
        result = fragments_from_images([_bgr()], auto_mask=False)
        assert (result[0].mask == 255).all()


# ─── TestAssemblyJsonRoundTripExtra ──────────────────────────────────────────

class TestAssemblyJsonRoundTripExtra:
    def test_json_has_n_fragments(self, tmp_path):
        asm = _assembly([0, 1, 2])
        path = str(tmp_path / "asm.json")
        save_assembly_json(asm, path)
        with open(path) as f:
            data = json.load(f)
        assert data["n_fragments"] == 3

    def test_total_score_roundtrip(self, tmp_path):
        asm = _assembly([0, 1])
        frags = [_frag(i) for i in [0, 1]]
        path = str(tmp_path / "asm.json")
        save_assembly_json(asm, path)
        loaded = load_assembly_json(path, frags)
        assert abs(loaded.total_score - asm.total_score) < 1e-9

    def test_placements_keys_preserved(self, tmp_path):
        asm = _assembly([10, 20])
        frags = [_frag(i) for i in [10, 20]]
        path = str(tmp_path / "asm.json")
        save_assembly_json(asm, path)
        loaded = load_assembly_json(path, frags)
        assert set(loaded.placements.keys()) == {10, 20}

    def test_position_roundtrip(self, tmp_path):
        asm = _assembly([0, 1])
        frags = [_frag(i) for i in [0, 1]]
        path = str(tmp_path / "asm.json")
        save_assembly_json(asm, path)
        loaded = load_assembly_json(path, frags)
        for fid, (pos_orig, _) in asm.placements.items():
            pos_r, _ = loaded.placements[fid]
            assert np.allclose(pos_r, pos_orig, atol=1e-9)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_assembly_json("/no/such/file.json", [])


# ─── TestFragmentsNpzRoundTripExtra ──────────────────────────────────────────

class TestFragmentsNpzRoundTripExtra:
    def test_single_fragment_roundtrip(self, tmp_path):
        frags = [_frag(0)]
        path = str(tmp_path / "single")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path + ".npz")
        assert len(loaded) == 1
        assert np.array_equal(loaded[0].image, frags[0].image)

    def test_fragment_id_5(self, tmp_path):
        frags = [_frag(5)]
        path = str(tmp_path / "frag5")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path + ".npz")
        assert loaded[0].fragment_id == 5

    def test_contour_shape_preserved(self, tmp_path):
        frags = [_frag(0)]
        path = str(tmp_path / "ctour")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path + ".npz")
        assert loaded[0].contour.shape == frags[0].contour.shape

    def test_creates_parent_dirs(self, tmp_path):
        frags = [_frag(0)]
        path = str(tmp_path / "sub" / "dir" / "frags")
        save_fragments_npz(frags, path)
        assert os.path.exists(path + ".npz")

    def test_nonexistent_npz_raises(self):
        with pytest.raises(FileNotFoundError):
            load_fragments_npz("/no/such/file.npz")
