"""Extra tests for puzzle_reconstruction/utils/patch_extractor.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.utils.patch_extractor import (
    Patch,
    PatchSet,
    batch_extract_patches,
    extract_border_patches,
    extract_grid_patches,
    extract_random_patches,
    extract_sliding_patches,
    filter_patches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _img(h: int = 64, w: int = 64, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    if channels == 1:
        return rng.integers(30, 200, (h, w), dtype=np.uint8)
    return rng.integers(30, 200, (h, w, channels), dtype=np.uint8)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    return _img(h, w, channels=1)


def _patch(x: int = 0, y: int = 0, size: int = 16) -> Patch:
    img = _img(size, size)
    return Patch(image=img, x=x, y=y, w=size, h=size, source_id=0)


# ─── Patch (extra) ────────────────────────────────────────────────────────────

class TestPatchExtra:
    def test_image_stored(self):
        p = _patch()
        assert isinstance(p.image, np.ndarray)

    def test_x_stored(self):
        assert _patch(x=10).x == 10

    def test_y_stored(self):
        assert _patch(y=20).y == 20

    def test_w_stored(self):
        assert _patch(size=16).w == 16

    def test_h_stored(self):
        assert _patch(size=16).h == 16

    def test_source_id_default_zero(self):
        assert _patch().source_id == 0

    def test_meta_default_empty(self):
        assert _patch().meta == {}

    def test_repr_contains_xy(self):
        p = _patch(x=5, y=7)
        assert "5" in repr(p) and "7" in repr(p)

    def test_image_shape_matches_wh(self):
        p = _patch(size=16)
        assert p.image.shape[0] == p.h
        assert p.image.shape[1] == p.w


# ─── PatchSet (extra) ─────────────────────────────────────────────────────────

class TestPatchSetExtra:
    def _make(self, n: int = 4) -> PatchSet:
        return PatchSet(
            patches=[_patch() for _ in range(n)],
            source_id=1,
            image_shape=(64, 64),
            method="grid",
        )

    def test_n_patches_property(self):
        ps = self._make(5)
        assert ps.n_patches == 5

    def test_source_id_stored(self):
        ps = self._make()
        assert ps.source_id == 1

    def test_image_shape_stored(self):
        ps = self._make()
        assert ps.image_shape == (64, 64)

    def test_method_stored(self):
        ps = self._make()
        assert ps.method == "grid"

    def test_repr_contains_n(self):
        ps = self._make(3)
        assert "n=3" in repr(ps)

    def test_empty_patch_set(self):
        ps = PatchSet(patches=[], source_id=0, image_shape=(64, 64), method="grid")
        assert ps.n_patches == 0


# ─── extract_grid_patches (extra) ─────────────────────────────────────────────

class TestExtractGridPatchesExtra:
    def test_returns_patch_set(self):
        result = extract_grid_patches(_img())
        assert isinstance(result, PatchSet)

    def test_method_is_grid(self):
        assert extract_grid_patches(_img()).method == "grid"

    def test_image_shape_stored(self):
        result = extract_grid_patches(_img(64, 64))
        assert result.image_shape == (64, 64)

    def test_patch_size_default_32(self):
        result = extract_grid_patches(_img(64, 64))
        for p in result.patches:
            assert p.w == 32 and p.h == 32

    def test_custom_patch_size(self):
        result = extract_grid_patches(_img(64, 64), patch_size=16)
        for p in result.patches:
            assert p.w == 16 and p.h == 16

    def test_correct_count_no_overlap(self):
        # 64x64 img, 32x32 patches → 4 patches
        result = extract_grid_patches(_img(64, 64), patch_size=32)
        assert result.n_patches == 4

    def test_stride_changes_count(self):
        r1 = extract_grid_patches(_img(64, 64), patch_size=16, stride=16)
        r2 = extract_grid_patches(_img(64, 64), patch_size=16, stride=8)
        assert r2.n_patches >= r1.n_patches

    def test_img_smaller_than_patch_empty(self):
        result = extract_grid_patches(_img(10, 10), patch_size=32)
        assert result.n_patches == 0

    def test_source_id_propagated(self):
        result = extract_grid_patches(_img(), source_id=7)
        assert result.source_id == 7
        for p in result.patches:
            assert p.source_id == 7

    def test_patch_images_correct_size(self):
        result = extract_grid_patches(_img(64, 64), patch_size=16)
        for p in result.patches:
            assert p.image.shape[:2] == (16, 16)

    def test_grayscale_supported(self):
        result = extract_grid_patches(_gray(64, 64), patch_size=32)
        assert result.n_patches > 0


# ─── extract_sliding_patches (extra) ──────────────────────────────────────────

class TestExtractSlidingPatchesExtra:
    def test_returns_patch_set(self):
        assert isinstance(extract_sliding_patches(_img()), PatchSet)

    def test_method_is_sliding(self):
        assert extract_sliding_patches(_img()).method == "sliding"

    def test_more_patches_with_smaller_stride(self):
        r1 = extract_sliding_patches(_img(64, 64), patch_size=16, stride=16)
        r2 = extract_sliding_patches(_img(64, 64), patch_size=16, stride=8)
        assert r2.n_patches >= r1.n_patches

    def test_stride_one_max_patches(self):
        result = extract_sliding_patches(_img(32, 32), patch_size=16, stride=1)
        # (32-16+1)^2 = 17^2 = 289 patches
        assert result.n_patches == (32 - 16 + 1) ** 2

    def test_source_id_stored(self):
        result = extract_sliding_patches(_img(), source_id=5)
        assert result.source_id == 5


# ─── extract_random_patches (extra) ───────────────────────────────────────────

class TestExtractRandomPatchesExtra:
    def test_returns_patch_set(self):
        assert isinstance(extract_random_patches(_img()), PatchSet)

    def test_method_is_random(self):
        assert extract_random_patches(_img()).method == "random"

    def test_n_patches_correct(self):
        result = extract_random_patches(_img(64, 64), n_patches=10, seed=0)
        assert result.n_patches == 10

    def test_small_img_empty(self):
        result = extract_random_patches(_img(8, 8), patch_size=32)
        assert result.n_patches == 0

    def test_reproducible_with_seed(self):
        r1 = extract_random_patches(_img(64, 64), n_patches=5, seed=42)
        r2 = extract_random_patches(_img(64, 64), n_patches=5, seed=42)
        for p1, p2 in zip(r1.patches, r2.patches):
            assert p1.x == p2.x and p1.y == p2.y

    def test_different_seeds_different_positions(self):
        r1 = extract_random_patches(_img(64, 64), n_patches=5, seed=0)
        r2 = extract_random_patches(_img(64, 64), n_patches=5, seed=99)
        positions_1 = [(p.x, p.y) for p in r1.patches]
        positions_2 = [(p.x, p.y) for p in r2.patches]
        assert positions_1 != positions_2

    def test_patches_within_bounds(self):
        img = _img(64, 64)
        result = extract_random_patches(img, patch_size=16, n_patches=20, seed=0)
        for p in result.patches:
            assert p.x + p.w <= 64
            assert p.y + p.h <= 64

    def test_source_id_stored(self):
        result = extract_random_patches(_img(), source_id=3)
        assert result.source_id == 3


# ─── extract_border_patches (extra) ───────────────────────────────────────────

class TestExtractBorderPatchesExtra:
    def test_returns_patch_set(self):
        assert isinstance(extract_border_patches(_img()), PatchSet)

    def test_method_is_border(self):
        assert extract_border_patches(_img()).method == "border"

    def test_max_4_sides(self):
        result = extract_border_patches(_img(64, 64), patch_size=16, n_per_side=4)
        assert result.n_patches <= 4 * 4

    def test_meta_has_side_key(self):
        result = extract_border_patches(_img(64, 64), patch_size=16, n_per_side=2)
        for p in result.patches:
            assert "side" in p.meta

    def test_side_values_in_0_3(self):
        result = extract_border_patches(_img(64, 64), patch_size=16, n_per_side=2)
        for p in result.patches:
            assert p.meta["side"] in (0, 1, 2, 3)

    def test_source_id_stored(self):
        result = extract_border_patches(_img(), source_id=2)
        assert result.source_id == 2

    def test_n_per_side_one(self):
        result = extract_border_patches(_img(64, 64), patch_size=16, n_per_side=1)
        assert result.n_patches >= 1

    def test_img_smaller_than_patch_empty(self):
        result = extract_border_patches(_img(8, 8), patch_size=32, n_per_side=4)
        assert result.n_patches == 0


# ─── filter_patches (extra) ───────────────────────────────────────────────────

class TestFilterPatchesExtra:
    def test_returns_list(self):
        ps = extract_grid_patches(_img(64, 64), patch_size=16)
        assert isinstance(filter_patches(ps.patches), list)

    def test_empty_input(self):
        assert filter_patches([]) == []

    def test_all_pass_with_loose_thresholds(self):
        ps = extract_grid_patches(_img(64, 64), patch_size=16)
        result = filter_patches(ps.patches, min_brightness=0.0,
                                max_brightness=256.0, min_entropy=0.0)
        assert len(result) == ps.n_patches

    def test_all_filtered_by_brightness(self):
        # Very dark image — all patches should be filtered
        dark = np.ones((64, 64, 3), dtype=np.uint8) * 5
        ps = extract_grid_patches(dark, patch_size=16)
        result = filter_patches(ps.patches, min_brightness=50.0)
        assert result == []

    def test_all_filtered_by_entropy(self):
        # Constant image → zero entropy
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        ps = extract_grid_patches(flat, patch_size=16)
        result = filter_patches(ps.patches, min_entropy=2.0)
        assert result == []

    def test_patches_with_valid_brightness_pass(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        ps = extract_grid_patches(img, patch_size=16)
        result = filter_patches(ps.patches, min_brightness=100.0,
                                max_brightness=200.0, min_entropy=0.0)
        assert len(result) == ps.n_patches

    def test_grayscale_patches_filtered(self):
        gray_img = _gray(64, 64)
        ps = extract_grid_patches(gray_img, patch_size=16)
        result = filter_patches(ps.patches, min_brightness=0.0,
                                max_brightness=256.0, min_entropy=0.0)
        assert isinstance(result, list)


# ─── batch_extract_patches (extra) ────────────────────────────────────────────

class TestBatchExtractPatchesExtra:
    def test_returns_list(self):
        result = batch_extract_patches([_img()])
        assert isinstance(result, list)

    def test_empty_input(self):
        assert batch_extract_patches([]) == []

    def test_length_matches_images(self):
        imgs = [_img() for _ in range(3)]
        result = batch_extract_patches(imgs, method="grid")
        assert len(result) == 3

    def test_all_patch_sets(self):
        for ps in batch_extract_patches([_img(), _img()]):
            assert isinstance(ps, PatchSet)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_extract_patches([_img()], method="unknown")

    def test_grid_method(self):
        result = batch_extract_patches([_img(64, 64)], method="grid", patch_size=32)
        assert result[0].method == "grid"

    def test_sliding_method(self):
        result = batch_extract_patches([_img(64, 64)], method="sliding",
                                       patch_size=32, stride=16)
        assert result[0].method == "sliding"

    def test_random_method(self):
        result = batch_extract_patches([_img(64, 64)], method="random",
                                       patch_size=16, n_patches=5)
        assert result[0].method == "random"

    def test_border_method(self):
        result = batch_extract_patches([_img(64, 64)], method="border",
                                       patch_size=16)
        assert result[0].method == "border"

    def test_source_id_increments(self):
        imgs = [_img() for _ in range(3)]
        results = batch_extract_patches(imgs, method="grid")
        for i, ps in enumerate(results):
            assert ps.source_id == i
