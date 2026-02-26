"""Tests for puzzle_reconstruction.utils.io (avoiding cv2 in test logic)"""
import json
import os
import tempfile
import pytest
import numpy as np

from puzzle_reconstruction.utils.io import (
    FragmentSetInfo,
    load_image_dir,
    save_assembly_json,
    load_assembly_json,
    save_fragments_npz,
    load_fragments_npz,
)
from puzzle_reconstruction.models import Fragment, Assembly


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_fragment(fid=0, h=20, w=20):
    np.random.seed(fid + 1)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    return Fragment(fragment_id=fid, image=img, mask=mask, contour=contour)


def _make_assembly(frags):
    placements = {f.fragment_id: (np.array([float(i), float(i)]), float(i) * 0.1)
                  for i, f in enumerate(frags)}
    return Assembly(
        fragments=frags,
        placements=placements,
        compat_matrix=np.eye(len(frags)),
        total_score=0.75,
        ocr_score=0.5,
    )


# ── FragmentSetInfo ──────────────────────────────────────────────────────────

def test_fragment_set_info_defaults():
    info = FragmentSetInfo(n_fragments=5)
    assert info.n_fragments == 5
    assert info.source_dir is None
    assert info.image_sizes == []
    assert info.total_pixels == 0
    assert info.failed_paths == []


def test_fragment_set_info_summary_format():
    info = FragmentSetInfo(n_fragments=3, total_pixels=1000)
    s = info.summary()
    assert "n=3" in s
    assert "failed=0" in s


def test_fragment_set_info_with_failed():
    info = FragmentSetInfo(n_fragments=2, failed_paths=["/bad/path.jpg"])
    s = info.summary()
    assert "failed=1" in s


# ── load_image_dir ───────────────────────────────────────────────────────────

def test_load_image_dir_not_found():
    with pytest.raises(FileNotFoundError):
        load_image_dir("/nonexistent/path/that/does/not/exist")


def test_load_image_dir_not_directory():
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(NotADirectoryError):
            load_image_dir(tmp.name)


def test_load_image_dir_empty_dir_returns_empty():
    with tempfile.TemporaryDirectory() as d:
        images, info = load_image_dir(d)
        assert images == []
        assert info.n_fragments == 0


def test_load_image_dir_returns_tuple():
    with tempfile.TemporaryDirectory() as d:
        result = load_image_dir(d)
        assert isinstance(result, tuple)
        assert len(result) == 2


def test_load_image_dir_info_type():
    with tempfile.TemporaryDirectory() as d:
        _, info = load_image_dir(d)
        assert isinstance(info, FragmentSetInfo)


# ── save_assembly_json / load_assembly_json ──────────────────────────────────

def test_save_and_load_assembly_json():
    frags = [_make_fragment(i) for i in range(3)]
    assembly = _make_assembly(frags)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "assembly.json")
        save_assembly_json(assembly, path)
        assert os.path.exists(path)

        loaded = load_assembly_json(path, frags)
        assert loaded.total_score == pytest.approx(0.75)
        assert loaded.ocr_score == pytest.approx(0.5)


def test_load_assembly_json_placements_correct():
    frags = [_make_fragment(i) for i in range(2)]
    assembly = _make_assembly(frags)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "asm.json")
        save_assembly_json(assembly, path)
        loaded = load_assembly_json(path, frags)
        for fid in assembly.placements:
            pos_orig, angle_orig = assembly.placements[fid]
            pos_load, angle_load = loaded.placements[fid]
            np.testing.assert_allclose(pos_orig, pos_load, atol=1e-9)
            assert angle_load == pytest.approx(angle_orig)


def test_load_assembly_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_assembly_json("/nonexistent/assembly.json", [])


def test_save_assembly_json_creates_parent_dirs():
    frags = [_make_fragment(0)]
    assembly = _make_assembly(frags)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "subdir", "nested", "asm.json")
        save_assembly_json(assembly, path)
        assert os.path.exists(path)


def test_save_assembly_json_valid_json():
    frags = [_make_fragment(0)]
    assembly = _make_assembly(frags)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "asm.json")
        save_assembly_json(assembly, path)
        with open(path) as f:
            data = json.load(f)
        assert "total_score" in data
        assert "placements" in data
        assert "n_fragments" in data


def test_save_assembly_json_scores_are_float():
    frags = [_make_fragment(0)]
    assembly = _make_assembly(frags)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "asm.json")
        save_assembly_json(assembly, path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data["total_score"], float)
        assert isinstance(data["ocr_score"], float)


# ── save_fragments_npz / load_fragments_npz ──────────────────────────────────

def test_save_and_load_fragments_npz():
    frags = [_make_fragment(i) for i in range(3)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frags.npz")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path)
        assert len(loaded) == 3


def test_load_fragments_npz_ids_correct():
    frags = [_make_fragment(i) for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frags.npz")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path)
        ids_loaded = {f.fragment_id for f in loaded}
        ids_orig = {f.fragment_id for f in frags}
        assert ids_loaded == ids_orig


def test_load_fragments_npz_images_preserved():
    frags = [_make_fragment(i) for i in range(2)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frags.npz")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path)
        for orig, ld in zip(frags, loaded):
            np.testing.assert_array_equal(orig.image, ld.image)


def test_load_fragments_npz_masks_preserved():
    frags = [_make_fragment(i) for i in range(2)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frags.npz")
        save_fragments_npz(frags, path)
        loaded = load_fragments_npz(path)
        for orig, ld in zip(frags, loaded):
            np.testing.assert_array_equal(orig.mask, ld.mask)


def test_load_fragments_npz_not_found():
    with pytest.raises(FileNotFoundError):
        load_fragments_npz("/nonexistent/path/frags.npz")


def test_save_fragments_npz_creates_parent_dirs():
    frags = [_make_fragment(0)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "frags.npz")
        save_fragments_npz(frags, path)
        assert os.path.exists(path) or os.path.exists(path + ".npz")


def test_save_and_load_single_fragment():
    frag = _make_fragment(99, h=15, w=15)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "single.npz")
        save_fragments_npz([frag], path)
        loaded = load_fragments_npz(path)
        assert len(loaded) == 1
        assert loaded[0].fragment_id == 99
