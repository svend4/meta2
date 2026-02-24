"""Тесты для puzzle_reconstruction/utils/patch_extractor.py."""
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


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=128, w=128, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _noisy(h=128, w=128, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=128, w=128):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 150
    img[:, :, 2] = 100
    return img


def _small(h=10, w=10):
    return np.full((h, w), 200, dtype=np.uint8)


# ─── Patch ────────────────────────────────────────────────────────────────────

class TestPatch:
    def test_fields(self):
        arr = np.zeros((32, 32), dtype=np.uint8)
        p = Patch(image=arr, x=5, y=10, w=32, h=32, source_id=1)
        assert p.x == 5
        assert p.y == 10
        assert p.w == 32
        assert p.h == 32
        assert p.source_id == 1

    def test_meta_default_empty(self):
        arr = np.zeros((16, 16), dtype=np.uint8)
        p = Patch(image=arr, x=0, y=0, w=16, h=16)
        assert isinstance(p.meta, dict)
        assert len(p.meta) == 0

    def test_meta_stored(self):
        arr = np.zeros((16, 16), dtype=np.uint8)
        p = Patch(image=arr, x=0, y=0, w=16, h=16, meta={"side": 2})
        assert p.meta["side"] == 2

    def test_repr(self):
        arr = np.zeros((32, 32), dtype=np.uint8)
        p = Patch(image=arr, x=4, y=8, w=32, h=32, source_id=0)
        s = repr(p)
        assert "Patch" in s
        assert "4" in s or "8" in s

    def test_source_id_default(self):
        arr = np.zeros((8, 8), dtype=np.uint8)
        p = Patch(image=arr, x=0, y=0, w=8, h=8)
        assert p.source_id == 0

    def test_image_stored(self):
        arr = np.full((16, 16), 99, dtype=np.uint8)
        p = Patch(image=arr, x=0, y=0, w=16, h=16)
        assert p.image.shape == (16, 16)
        assert p.image[0, 0] == 99


# ─── PatchSet ─────────────────────────────────────────────────────────────────

class TestPatchSet:
    def _make(self, n=3):
        patches = [Patch(image=np.zeros((8, 8), dtype=np.uint8),
                         x=0, y=0, w=8, h=8) for _ in range(n)]
        return PatchSet(patches=patches, source_id=0, image_shape=(64, 64), method="grid")

    def test_n_patches(self):
        ps = self._make(5)
        assert ps.n_patches == 5

    def test_n_patches_empty(self):
        ps = PatchSet(patches=[], source_id=0, image_shape=(64, 64), method="grid")
        assert ps.n_patches == 0

    def test_repr(self):
        ps = self._make(2)
        s = repr(ps)
        assert "PatchSet" in s
        assert "grid" in s

    def test_fields(self):
        ps = self._make()
        assert ps.source_id == 0
        assert ps.image_shape == (64, 64)
        assert ps.method == "grid"

    def test_patches_list(self):
        ps = self._make(4)
        assert isinstance(ps.patches, list)
        assert len(ps.patches) == 4


# ─── extract_grid_patches ────────────────────────────────────────────────────

class TestExtractGridPatches:
    def test_returns_patchset(self):
        assert isinstance(extract_grid_patches(_noisy()), PatchSet)

    def test_method(self):
        assert extract_grid_patches(_noisy()).method == "grid"

    def test_n_patches_correct(self):
        # 128×128 with patch_size=32, stride=32 → 4×4=16
        r = extract_grid_patches(_gray(128, 128), patch_size=32)
        assert r.n_patches == 16

    def test_n_patches_non_overlapping(self):
        r = extract_grid_patches(_gray(64, 64), patch_size=16)
        assert r.n_patches == 16   # 4×4

    def test_all_patches_correct_size(self):
        r = extract_grid_patches(_noisy(), patch_size=32)
        for p in r.patches:
            assert p.w == 32
            assert p.h == 32
            assert p.image.shape == (32, 32)

    def test_custom_stride(self):
        r = extract_grid_patches(_gray(64, 64), patch_size=16, stride=8)
        # with stride=8, 8 steps per axis → 7×7 = 49
        assert r.n_patches > 16

    def test_source_id_stored(self):
        r = extract_grid_patches(_noisy(), source_id=5)
        assert r.source_id == 5
        for p in r.patches:
            assert p.source_id == 5

    def test_bgr_input(self):
        r = extract_grid_patches(_bgr(64, 64), patch_size=16)
        assert r.n_patches == 16
        for p in r.patches:
            assert p.image.shape == (16, 16, 3)

    def test_image_smaller_than_patch(self):
        r = extract_grid_patches(_small(10, 10), patch_size=32)
        assert r.n_patches == 0

    def test_image_shape_stored(self):
        r = extract_grid_patches(_gray(96, 80), patch_size=16)
        assert r.image_shape == (96, 80)

    def test_patch_positions_nonneg(self):
        r = extract_grid_patches(_gray(128, 128), patch_size=32)
        for p in r.patches:
            assert p.x >= 0
            assert p.y >= 0


# ─── extract_sliding_patches ──────────────────────────────────────────────────

class TestExtractSlidingPatches:
    def test_returns_patchset(self):
        assert isinstance(extract_sliding_patches(_noisy()), PatchSet)

    def test_method(self):
        assert extract_sliding_patches(_noisy()).method == "sliding"

    def test_more_than_grid(self):
        # stride=16 produces more patches than stride=32
        grid   = extract_grid_patches(_gray(128, 128), patch_size=32, stride=32)
        slid   = extract_sliding_patches(_gray(128, 128), patch_size=32, stride=16)
        assert slid.n_patches >= grid.n_patches

    def test_all_patches_correct_size(self):
        r = extract_sliding_patches(_noisy(), patch_size=32, stride=16)
        for p in r.patches:
            assert p.image.shape[0] == 32
            assert p.image.shape[1] == 32

    def test_source_id(self):
        r = extract_sliding_patches(_noisy(), source_id=3)
        assert r.source_id == 3
        for p in r.patches:
            assert p.source_id == 3

    def test_bgr_input(self):
        r = extract_sliding_patches(_bgr(64, 64), patch_size=16, stride=8)
        for p in r.patches:
            assert p.image.ndim == 3

    def test_image_shape_stored(self):
        r = extract_sliding_patches(_gray(80, 96), patch_size=16)
        assert r.image_shape == (80, 96)


# ─── extract_random_patches ───────────────────────────────────────────────────

class TestExtractRandomPatches:
    def test_returns_patchset(self):
        assert isinstance(extract_random_patches(_noisy()), PatchSet)

    def test_method(self):
        assert extract_random_patches(_noisy()).method == "random"

    def test_n_patches_correct(self):
        r = extract_random_patches(_noisy(), patch_size=32, n_patches=20, seed=42)
        assert r.n_patches == 20

    def test_all_patches_correct_size(self):
        r = extract_random_patches(_noisy(), patch_size=32, n_patches=10, seed=1)
        for p in r.patches:
            assert p.image.shape[0] == 32
            assert p.image.shape[1] == 32

    def test_seed_reproducible(self):
        r1 = extract_random_patches(_noisy(), patch_size=16, n_patches=5, seed=99)
        r2 = extract_random_patches(_noisy(), patch_size=16, n_patches=5, seed=99)
        for p1, p2 in zip(r1.patches, r2.patches):
            assert p1.x == p2.x
            assert p1.y == p2.y

    def test_different_seeds_differ(self):
        r1 = extract_random_patches(_noisy(), patch_size=16, n_patches=10, seed=1)
        r2 = extract_random_patches(_noisy(), patch_size=16, n_patches=10, seed=2)
        xs1 = [p.x for p in r1.patches]
        xs2 = [p.x for p in r2.patches]
        assert xs1 != xs2

    def test_source_id(self):
        r = extract_random_patches(_noisy(), source_id=7)
        assert r.source_id == 7
        for p in r.patches:
            assert p.source_id == 7

    def test_too_small_image_empty(self):
        r = extract_random_patches(_small(10, 10), patch_size=32, n_patches=5)
        assert r.n_patches == 0

    def test_seed_none_no_crash(self):
        r = extract_random_patches(_noisy(), patch_size=16, n_patches=5, seed=None)
        assert isinstance(r, PatchSet)

    def test_bgr_input(self):
        r = extract_random_patches(_bgr(64, 64), patch_size=16, n_patches=5, seed=0)
        for p in r.patches:
            assert p.image.ndim == 3

    def test_patch_in_bounds(self):
        h, w, ps = 128, 128, 32
        r = extract_random_patches(_noisy(h, w), patch_size=ps, n_patches=30, seed=5)
        for p in r.patches:
            assert p.x + p.w <= w
            assert p.y + p.h <= h


# ─── extract_border_patches ───────────────────────────────────────────────────

class TestExtractBorderPatches:
    def test_returns_patchset(self):
        assert isinstance(extract_border_patches(_noisy()), PatchSet)

    def test_method(self):
        assert extract_border_patches(_noisy()).method == "border"

    def test_max_patches_4_sides(self):
        r = extract_border_patches(_gray(128, 128), patch_size=16, n_per_side=4)
        assert r.n_patches <= 16   # 4 × 4

    def test_side_metadata_present(self):
        r = extract_border_patches(_gray(128, 128), patch_size=16, n_per_side=4)
        for p in r.patches:
            assert "side" in p.meta

    def test_side_values_valid(self):
        r = extract_border_patches(_gray(128, 128), patch_size=16, n_per_side=4)
        for p in r.patches:
            assert p.meta["side"] in {0, 1, 2, 3}

    def test_all_patches_correct_size(self):
        r = extract_border_patches(_noisy(128, 128), patch_size=24, n_per_side=5)
        for p in r.patches:
            assert p.image.shape[0] == 24
            assert p.image.shape[1] == 24

    def test_source_id(self):
        r = extract_border_patches(_noisy(), source_id=2)
        assert r.source_id == 2
        for p in r.patches:
            assert p.source_id == 2

    def test_bgr_input(self):
        r = extract_border_patches(_bgr(128, 128), patch_size=16, n_per_side=4)
        for p in r.patches:
            assert p.image.ndim == 3

    def test_all_sides_represented(self):
        r = extract_border_patches(_gray(128, 128), patch_size=16, n_per_side=4)
        sides = {p.meta["side"] for p in r.patches}
        assert len(sides) == 4

    def test_image_shape_stored(self):
        r = extract_border_patches(_gray(80, 96), patch_size=16)
        assert r.image_shape == (80, 96)


# ─── filter_patches ───────────────────────────────────────────────────────────

class TestFilterPatches:
    def _make_patch(self, val=128, size=16, src=0):
        img = np.full((size, size), val, dtype=np.uint8)
        return Patch(image=img, x=0, y=0, w=size, h=size, source_id=src)

    def test_returns_list(self):
        patches = [self._make_patch(128)]
        assert isinstance(filter_patches(patches), list)

    def test_empty_input(self):
        assert filter_patches([]) == []

    def test_mid_brightness_passes(self):
        p = self._make_patch(val=128)
        result = filter_patches([p], min_brightness=10.0, max_brightness=245.0,
                                min_entropy=0.0)
        assert p in result

    def test_very_dark_removed(self):
        dark = self._make_patch(val=3)
        result = filter_patches([dark], min_brightness=10.0)
        assert dark not in result

    def test_very_bright_removed(self):
        bright = self._make_patch(val=253)
        result = filter_patches([bright], max_brightness=245.0)
        assert bright not in result

    def test_noisy_patch_passes(self):
        rng   = np.random.default_rng(42)
        img   = rng.integers(50, 200, (32, 32), dtype=np.uint8)
        patch = Patch(image=img, x=0, y=0, w=32, h=32)
        result = filter_patches([patch])
        assert patch in result

    def test_entropy_filter_removes_constant(self):
        p = self._make_patch(val=128)  # Constant → entropy ≈ 0
        result = filter_patches([p], min_entropy=3.0)
        assert p not in result

    def test_bgr_input(self):
        img   = np.full((16, 16, 3), 128, dtype=np.uint8)
        patch = Patch(image=img, x=0, y=0, w=16, h=16)
        result = filter_patches([patch], min_entropy=0.0)
        # Should not crash on BGR
        assert isinstance(result, list)


# ─── batch_extract_patches ────────────────────────────────────────────────────

class TestBatchExtractPatches:
    def test_returns_list(self):
        imgs    = [_noisy() for _ in range(3)]
        results = batch_extract_patches(imgs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_is_patchset(self):
        for ps in batch_extract_patches([_gray(), _noisy()]):
            assert isinstance(ps, PatchSet)

    def test_empty_list(self):
        assert batch_extract_patches([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_extract_patches([_noisy()], method="spiral_xyz")

    @pytest.mark.parametrize("method", ["grid", "sliding", "random", "border"])
    def test_all_methods(self, method):
        results = batch_extract_patches([_gray(64, 64), _noisy(64, 64)],
                                         method=method, patch_size=16)
        assert len(results) == 2
        for ps in results:
            assert isinstance(ps, PatchSet)
            assert ps.method == method

    def test_source_id_assigned_correctly(self):
        results = batch_extract_patches([_gray(), _noisy(), _bgr()])
        for i, ps in enumerate(results):
            assert ps.source_id == i

    def test_kwargs_forwarded_patch_size(self):
        results = batch_extract_patches([_gray(64, 64)], method="grid", patch_size=16)
        for p in results[0].patches:
            assert p.w == 16
            assert p.h == 16

    def test_random_n_patches_forwarded(self):
        results = batch_extract_patches([_noisy()], method="random",
                                         patch_size=16, n_patches=12, seed=0)
        assert results[0].n_patches == 12

    def test_border_n_per_side_forwarded(self):
        results = batch_extract_patches([_noisy(128, 128)], method="border",
                                         patch_size=16, n_per_side=3)
        assert results[0].n_patches <= 12   # 4 sides × 3

    def test_bgr_input(self):
        results = batch_extract_patches([_bgr(64, 64)], method="grid", patch_size=16)
        for p in results[0].patches:
            assert p.image.ndim == 3
