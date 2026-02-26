"""Tests for puzzle_reconstruction.utils.patch_extractor"""
import numpy as np
import pytest
from puzzle_reconstruction.utils.patch_extractor import (
    Patch, PatchSet,
    extract_grid_patches, extract_sliding_patches,
    extract_random_patches, extract_border_patches,
    filter_patches, batch_extract_patches,
)

np.random.seed(42)


def _make_img(h=128, w=128, channels=3):
    if channels == 1:
        return np.random.randint(50, 200, (h, w), dtype=np.uint8)
    return np.random.randint(50, 200, (h, w, channels), dtype=np.uint8)


# ── Patch ─────────────────────────────────────────────────────────────────────

def test_patch_repr():
    p = Patch(image=np.zeros((32, 32, 3), dtype=np.uint8),
              x=10, y=20, w=32, h=32, source_id=5)
    r = repr(p)
    assert "32" in r


def test_patch_meta_default_empty():
    p = Patch(image=np.zeros((8, 8), dtype=np.uint8), x=0, y=0, w=8, h=8)
    assert p.meta == {}


# ── PatchSet ──────────────────────────────────────────────────────────────────

def test_patch_set_n_patches():
    patches = [Patch(image=np.zeros((8, 8), dtype=np.uint8), x=i, y=0, w=8, h=8)
               for i in range(5)]
    ps = PatchSet(patches=patches, source_id=0,
                  image_shape=(64, 64), method="test")
    assert ps.n_patches == 5


def test_patch_set_repr():
    ps = PatchSet(patches=[], source_id=0, image_shape=(64, 64), method="grid")
    r = repr(ps)
    assert "n=0" in r


# ── extract_grid_patches ──────────────────────────────────────────────────────

def test_extract_grid_patches_returns_patch_set():
    img = _make_img(128, 128)
    ps = extract_grid_patches(img, patch_size=32)
    assert isinstance(ps, PatchSet)
    assert ps.method == "grid"


def test_extract_grid_patches_count():
    img = _make_img(64, 64)
    ps = extract_grid_patches(img, patch_size=32, stride=32)
    assert ps.n_patches == 4  # 2x2 grid


def test_extract_grid_patches_patch_size():
    img = _make_img(128, 128)
    ps = extract_grid_patches(img, patch_size=32)
    assert all(p.w == 32 and p.h == 32 for p in ps.patches)


def test_extract_grid_patches_patch_image_shape():
    img = _make_img(64, 64)
    ps = extract_grid_patches(img, patch_size=16, stride=16)
    for p in ps.patches:
        assert p.image.shape[:2] == (16, 16)


def test_extract_grid_patches_image_smaller_than_patch():
    img = _make_img(10, 10)
    ps = extract_grid_patches(img, patch_size=32)
    assert ps.n_patches == 0


def test_extract_grid_patches_source_id():
    img = _make_img(64, 64)
    ps = extract_grid_patches(img, patch_size=32, source_id=7)
    assert ps.source_id == 7
    assert all(p.source_id == 7 for p in ps.patches)


def test_extract_grid_patches_with_overlap():
    img = _make_img(64, 64)
    ps_no_overlap = extract_grid_patches(img, patch_size=16, stride=16)
    ps_overlap = extract_grid_patches(img, patch_size=16, stride=8)
    assert ps_overlap.n_patches > ps_no_overlap.n_patches


def test_extract_grid_patches_grayscale():
    img = _make_img(64, 64, channels=1)
    ps = extract_grid_patches(img, patch_size=16, stride=16)
    assert ps.n_patches == 16
    assert all(p.image.ndim == 2 for p in ps.patches)


# ── extract_sliding_patches ───────────────────────────────────────────────────

def test_extract_sliding_patches_method():
    img = _make_img(64, 64)
    ps = extract_sliding_patches(img, patch_size=16, stride=8)
    assert ps.method == "sliding"


def test_extract_sliding_patches_more_than_grid():
    img = _make_img(64, 64)
    ps_grid = extract_grid_patches(img, patch_size=16, stride=16)
    ps_slide = extract_sliding_patches(img, patch_size=16, stride=8)
    assert ps_slide.n_patches > ps_grid.n_patches


# ── extract_random_patches ────────────────────────────────────────────────────

def test_extract_random_patches_count():
    img = _make_img(128, 128)
    ps = extract_random_patches(img, patch_size=32, n_patches=20, seed=42)
    assert ps.n_patches == 20


def test_extract_random_patches_reproducible():
    img = _make_img(128, 128)
    ps1 = extract_random_patches(img, patch_size=16, n_patches=10, seed=0)
    ps2 = extract_random_patches(img, patch_size=16, n_patches=10, seed=0)
    assert ps1.patches[0].x == ps2.patches[0].x
    assert ps1.patches[0].y == ps2.patches[0].y


def test_extract_random_patches_method():
    img = _make_img(64, 64)
    ps = extract_random_patches(img, patch_size=16, n_patches=5, seed=1)
    assert ps.method == "random"


def test_extract_random_patches_small_image():
    img = _make_img(10, 10)
    ps = extract_random_patches(img, patch_size=32, n_patches=10)
    assert ps.n_patches == 0


def test_extract_random_patches_bounds():
    img = _make_img(100, 100)
    ps = extract_random_patches(img, patch_size=20, n_patches=50, seed=5)
    for p in ps.patches:
        assert 0 <= p.x and p.x + p.w <= 100
        assert 0 <= p.y and p.y + p.h <= 100


# ── extract_border_patches ────────────────────────────────────────────────────

def test_extract_border_patches_method():
    img = _make_img(128, 128)
    ps = extract_border_patches(img, patch_size=32, n_per_side=4)
    assert ps.method == "border"


def test_extract_border_patches_max_count():
    img = _make_img(128, 128)
    n_per = 4
    ps = extract_border_patches(img, patch_size=32, n_per_side=n_per)
    assert ps.n_patches <= 4 * n_per


def test_extract_border_patches_has_all_sides():
    img = _make_img(128, 128)
    ps = extract_border_patches(img, patch_size=16, n_per_side=4)
    sides = {p.meta.get("side") for p in ps.patches}
    assert {0, 1, 2, 3}.issubset(sides)


def test_extract_border_patches_patch_size():
    img = _make_img(128, 128)
    ps = extract_border_patches(img, patch_size=24, n_per_side=3)
    for p in ps.patches:
        assert p.w == 24 and p.h == 24


# ── filter_patches ────────────────────────────────────────────────────────────

def test_filter_patches_removes_dark():
    dark = Patch(image=np.zeros((32, 32, 3), dtype=np.uint8), x=0, y=0, w=32, h=32)
    bright_img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    bright = Patch(image=bright_img, x=0, y=0, w=32, h=32)
    result = filter_patches([dark, bright], min_brightness=10.0, min_entropy=0.0)
    # Dark patch (mean=0) should be filtered out, bright patch (mean=128) kept
    result_ids = [id(p) for p in result]
    assert id(dark) not in result_ids
    assert id(bright) in result_ids


def test_filter_patches_removes_bright():
    bright_img = np.ones((32, 32, 3), dtype=np.uint8) * 255
    saturated = Patch(image=bright_img, x=0, y=0, w=32, h=32)
    result = filter_patches([saturated], max_brightness=245.0)
    assert saturated not in result


def test_filter_patches_empty_input():
    assert filter_patches([]) == []


def test_filter_patches_natural_image_keeps_most():
    img = _make_img(128, 128)
    ps = extract_grid_patches(img, patch_size=32, stride=32)
    filtered = filter_patches(ps.patches, min_brightness=10.0,
                               max_brightness=245.0, min_entropy=0.1)
    assert len(filtered) > 0


# ── batch_extract_patches ─────────────────────────────────────────────────────

def test_batch_extract_patches_returns_list():
    images = [_make_img(64, 64) for _ in range(3)]
    result = batch_extract_patches(images, method="grid", patch_size=16)
    assert len(result) == 3
    assert all(isinstance(ps, PatchSet) for ps in result)


def test_batch_extract_patches_source_ids():
    images = [_make_img(64, 64) for _ in range(3)]
    result = batch_extract_patches(images, method="grid", patch_size=16)
    for i, ps in enumerate(result):
        assert ps.source_id == i


def test_batch_extract_patches_invalid_method_raises():
    images = [_make_img(64, 64)]
    with pytest.raises(ValueError):
        batch_extract_patches(images, method="unknown_method")


def test_batch_extract_patches_random():
    images = [_make_img(128, 128) for _ in range(2)]
    result = batch_extract_patches(images, method="random",
                                    patch_size=32, n_patches=10, seed=0)
    assert all(ps.n_patches == 10 for ps in result)


def test_batch_extract_patches_empty_list():
    result = batch_extract_patches([], method="grid", patch_size=32)
    assert result == []
