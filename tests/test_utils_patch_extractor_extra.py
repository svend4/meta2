"""Extra tests for puzzle_reconstruction/utils/patch_extractor.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.patch_extractor import (
    Patch,
    PatchSet,
    extract_grid_patches,
    extract_sliding_patches,
    extract_random_patches,
    extract_border_patches,
    filter_patches,
    batch_extract_patches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h=64, w=64, c=3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(30, 200, (h, w, c), dtype=np.uint8)


def _gray(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(30, 200, (h, w), dtype=np.uint8)


# ─── Patch ────────────────────────────────────────────────────────────────────

class TestPatchExtra:
    def test_repr_contains_xy(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        p = Patch(image=img, x=2, y=3, w=8, h=8, source_id=0)
        assert "2" in repr(p) and "3" in repr(p)

    def test_stores_meta(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        p = Patch(image=img, x=0, y=0, w=8, h=8, meta={"side": 1})
        assert p.meta["side"] == 1

    def test_default_source_id(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        p = Patch(image=img, x=0, y=0, w=4, h=4)
        assert p.source_id == 0


# ─── PatchSet ─────────────────────────────────────────────────────────────────

class TestPatchSetExtra:
    def test_n_patches(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        p = Patch(image=img, x=0, y=0, w=8, h=8)
        ps = PatchSet(patches=[p, p], source_id=0, image_shape=(64, 64), method="grid")
        assert ps.n_patches == 2

    def test_repr_contains_method(self):
        ps = PatchSet(patches=[], source_id=0, image_shape=(64, 64), method="grid")
        assert "grid" in repr(ps)


# ─── extract_grid_patches ─────────────────────────────────────────────────────

class TestExtractGridPatchesExtra:
    def test_returns_patchset(self):
        ps = extract_grid_patches(_img(), patch_size=16)
        assert isinstance(ps, PatchSet)

    def test_method_is_grid(self):
        ps = extract_grid_patches(_img(), patch_size=16)
        assert ps.method == "grid"

    def test_no_patches_when_image_too_small(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        ps = extract_grid_patches(img, patch_size=32)
        assert ps.n_patches == 0

    def test_patch_size_correct(self):
        ps = extract_grid_patches(_img(64, 64), patch_size=16)
        for p in ps.patches:
            assert p.w == 16 and p.h == 16

    def test_source_id_stored(self):
        ps = extract_grid_patches(_img(), patch_size=16, source_id=5)
        assert ps.source_id == 5

    def test_works_with_grayscale(self):
        ps = extract_grid_patches(_gray(64, 64), patch_size=16)
        assert ps.n_patches > 0


# ─── extract_sliding_patches ──────────────────────────────────────────────────

class TestExtractSlidingPatchesExtra:
    def test_method_is_sliding(self):
        ps = extract_sliding_patches(_img(), patch_size=16, stride=8)
        assert ps.method == "sliding"

    def test_more_patches_than_grid(self):
        img = _img(64, 64)
        grid_ps = extract_grid_patches(img, patch_size=16)
        sliding_ps = extract_sliding_patches(img, patch_size=16, stride=8)
        assert sliding_ps.n_patches >= grid_ps.n_patches


# ─── extract_random_patches ───────────────────────────────────────────────────

class TestExtractRandomPatchesExtra:
    def test_returns_patchset(self):
        ps = extract_random_patches(_img(), patch_size=16, n_patches=10, seed=0)
        assert isinstance(ps, PatchSet)

    def test_method_is_random(self):
        ps = extract_random_patches(_img(), patch_size=16, n_patches=5, seed=0)
        assert ps.method == "random"

    def test_respects_n_patches(self):
        ps = extract_random_patches(_img(64, 64), patch_size=16, n_patches=20, seed=1)
        assert ps.n_patches == 20

    def test_image_too_small_returns_empty(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        ps = extract_random_patches(img, patch_size=32)
        assert ps.n_patches == 0

    def test_seed_reproducible(self):
        ps1 = extract_random_patches(_img(), patch_size=16, n_patches=5, seed=99)
        ps2 = extract_random_patches(_img(), patch_size=16, n_patches=5, seed=99)
        assert ps1.patches[0].x == ps2.patches[0].x


# ─── extract_border_patches ───────────────────────────────────────────────────

class TestExtractBorderPatchesExtra:
    def test_returns_patchset(self):
        ps = extract_border_patches(_img(64, 64), patch_size=16, n_per_side=4)
        assert isinstance(ps, PatchSet)

    def test_method_is_border(self):
        ps = extract_border_patches(_img(64, 64), patch_size=16)
        assert ps.method == "border"

    def test_side_meta_present(self):
        ps = extract_border_patches(_img(64, 64), patch_size=16, n_per_side=4)
        sides = {p.meta.get("side") for p in ps.patches}
        assert sides <= {0, 1, 2, 3}


# ─── filter_patches ───────────────────────────────────────────────────────────

class TestFilterPatchesExtra:
    def test_filters_dark_patches(self):
        # All-black patches should fail brightness check
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        ps = extract_grid_patches(img, patch_size=16)
        filtered = filter_patches(ps.patches, min_brightness=10.0)
        assert len(filtered) == 0

    def test_passes_bright_patches(self):
        # Random patches with reasonable brightness should pass
        ps = extract_grid_patches(_img(64, 64), patch_size=16)
        filtered = filter_patches(ps.patches, min_brightness=5.0,
                                   max_brightness=250.0, min_entropy=0.0)
        assert len(filtered) > 0

    def test_empty_input(self):
        assert filter_patches([]) == []


# ─── batch_extract_patches ────────────────────────────────────────────────────

class TestBatchExtractPatchesExtra:
    def test_returns_list(self):
        result = batch_extract_patches([_img(), _img()], method="grid", patch_size=16)
        assert isinstance(result, list) and len(result) == 2

    def test_each_is_patchset(self):
        result = batch_extract_patches([_img()], method="grid", patch_size=16)
        assert isinstance(result[0], PatchSet)

    def test_empty_input(self):
        assert batch_extract_patches([], method="grid") == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_extract_patches([_img()], method="unknown")

    def test_source_id_assigned(self):
        result = batch_extract_patches([_img(), _img()], method="grid", patch_size=16)
        assert result[0].source_id == 0
        assert result[1].source_id == 1
