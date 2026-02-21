"""Тесты для puzzle_reconstruction.preprocessing.patch_sampler."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.patch_sampler import (
    PatchSample,
    SampleConfig,
    SampleResult,
    batch_sample_patches,
    extract_patch_images,
    sample_border_patches,
    sample_grid_patches,
    sample_patches,
    sample_random_patches,
    sample_stride_patches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _rgb(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ─── TestSampleConfig ─────────────────────────────────────────────────────────

class TestSampleConfig:
    def test_defaults(self):
        cfg = SampleConfig()
        assert cfg.patch_size == 32
        assert cfg.n_patches == 16
        assert cfg.mode == "grid"
        assert cfg.stride == 8
        assert cfg.seed == 0

    def test_patch_size_lt_2_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(patch_size=1)

    def test_n_patches_lt_1_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(n_patches=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(mode="spiral")

    def test_stride_lt_1_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(stride=0)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(seed=-1)

    def test_valid_modes(self):
        for mode in ("grid", "random", "border", "stride"):
            cfg = SampleConfig(mode=mode)
            assert cfg.mode == mode


# ─── TestPatchSample ──────────────────────────────────────────────────────────

class TestPatchSample:
    def _make(self, idx=0, x=10, y=5, w=32, h=32):
        return PatchSample(idx=idx, x=x, y=y, w=w, h=h)

    def test_basic_construction(self):
        p = self._make()
        assert p.x == 10
        assert p.y == 5
        assert p.w == 32
        assert p.h == 32

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=-1, x=0, y=0, w=10, h=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=-1, y=0, w=10, h=10)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=-1, w=10, h=10)

    def test_w_lt_1_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=0, w=0, h=10)

    def test_h_lt_1_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=0, w=10, h=0)

    def test_x2_property(self):
        p = self._make(x=10, w=20)
        assert p.x2 == 30

    def test_y2_property(self):
        p = self._make(y=5, h=15)
        assert p.y2 == 20

    def test_area_property(self):
        p = self._make(w=10, h=20)
        assert p.area == 200

    def test_center_property(self):
        p = self._make(x=0, y=0, w=10, h=20)
        cx, cy = p.center
        assert abs(cx - 5.0) < 1e-9
        assert abs(cy - 10.0) < 1e-9


# ─── TestSampleResult ─────────────────────────────────────────────────────────

class TestSampleResult:
    def _make(self, n=3, h=64, w=64):
        samples = [PatchSample(idx=i, x=i*10, y=0, w=10, h=10) for i in range(n)]
        return SampleResult(samples=samples, image_shape=(h, w), n_patches=n)

    def test_basic_construction(self):
        r = self._make()
        assert r.n_patches == 3

    def test_negative_n_patches_raises(self):
        with pytest.raises(ValueError):
            SampleResult(samples=[], image_shape=(64, 64), n_patches=-1)

    def test_coverage_ratio_empty(self):
        r = SampleResult(samples=[], image_shape=(64, 64), n_patches=0)
        assert r.coverage_ratio == 0.0

    def test_coverage_ratio_positive(self):
        r = self._make(n=1, h=10, w=10)
        # one 10x10 patch on 64x64 image: area=100, img_area=4096
        assert r.coverage_ratio > 0.0

    def test_coverage_ratio_full(self):
        samples = [PatchSample(idx=0, x=0, y=0, w=10, h=10)]
        r = SampleResult(samples=samples, image_shape=(10, 10), n_patches=1)
        assert abs(r.coverage_ratio - 1.0) < 1e-9


# ─── TestSampleGridPatches ────────────────────────────────────────────────────

class TestSampleGridPatches:
    def test_basic(self):
        patches = sample_grid_patches(64, 64, 32)
        assert len(patches) > 0
        assert all(isinstance(p, PatchSample) for p in patches)

    def test_image_smaller_than_patch(self):
        patches = sample_grid_patches(10, 10, 32)
        # ps clipped to min(32,10,10)=10
        assert len(patches) >= 1

    def test_max_patches_limit(self):
        patches = sample_grid_patches(128, 128, 16, max_patches=4)
        assert len(patches) <= 4

    def test_no_out_of_bounds(self):
        h, w, ps = 60, 60, 20
        patches = sample_grid_patches(h, w, ps)
        for p in patches:
            assert p.x2 <= w
            assert p.y2 <= h

    def test_invalid_image_size_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(0, 64, 16)

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(64, 64, 1)

    def test_invalid_max_patches_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(64, 64, 16, max_patches=0)

    def test_idx_sequential(self):
        patches = sample_grid_patches(64, 64, 32)
        for i, p in enumerate(patches):
            assert p.idx == i


# ─── TestSampleRandomPatches ──────────────────────────────────────────────────

class TestSampleRandomPatches:
    def test_returns_n_patches(self):
        patches = sample_random_patches(64, 64, 16, n_patches=10)
        assert len(patches) == 10

    def test_reproducible_with_seed(self):
        p1 = sample_random_patches(64, 64, 16, n_patches=5, seed=42)
        p2 = sample_random_patches(64, 64, 16, n_patches=5, seed=42)
        assert all(a.x == b.x and a.y == b.y for a, b in zip(p1, p2))

    def test_different_seeds_differ(self):
        p1 = sample_random_patches(64, 64, 16, n_patches=5, seed=0)
        p2 = sample_random_patches(64, 64, 16, n_patches=5, seed=99)
        # Not all patches should be identical
        assert not all(a.x == b.x and a.y == b.y for a, b in zip(p1, p2))

    def test_no_out_of_bounds(self):
        h, w, ps = 64, 64, 16
        patches = sample_random_patches(h, w, ps, n_patches=20, seed=7)
        for p in patches:
            assert p.x2 <= w
            assert p.y2 <= h

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(0, 64, 16, n_patches=5)

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(64, 64, 1, n_patches=5)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(64, 64, 16, n_patches=5, seed=-1)


# ─── TestSampleBorderPatches ──────────────────────────────────────────────────

class TestSampleBorderPatches:
    def test_returns_patches(self):
        patches = sample_border_patches(64, 64, 16)
        assert len(patches) > 0

    def test_max_patches_limit(self):
        patches = sample_border_patches(64, 64, 16, max_patches=4)
        assert len(patches) <= 4

    def test_no_out_of_bounds(self):
        h, w, ps = 64, 64, 16
        patches = sample_border_patches(h, w, ps)
        for p in patches:
            assert p.x2 <= w
            assert p.y2 <= h

    def test_no_duplicates(self):
        patches = sample_border_patches(64, 64, 16)
        positions = [(p.x, p.y) for p in patches]
        assert len(positions) == len(set(positions))

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            sample_border_patches(0, 64, 16)

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError):
            sample_border_patches(64, 64, 1)


# ─── TestSampleStridePatches ──────────────────────────────────────────────────

class TestSampleStridePatches:
    def test_basic(self):
        patches = sample_stride_patches(64, 64, 16, stride=8)
        assert len(patches) > 0

    def test_max_patches_limit(self):
        patches = sample_stride_patches(64, 64, 16, stride=4, max_patches=5)
        assert len(patches) <= 5

    def test_no_out_of_bounds(self):
        h, w, ps, st = 64, 64, 16, 8
        patches = sample_stride_patches(h, w, ps, stride=st)
        for p in patches:
            assert p.x2 <= w
            assert p.y2 <= h

    def test_stride_affects_count(self):
        p_small = sample_stride_patches(64, 64, 16, stride=4)
        p_large = sample_stride_patches(64, 64, 16, stride=16)
        assert len(p_small) >= len(p_large)

    def test_invalid_stride_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(64, 64, 16, stride=0)

    def test_invalid_image_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(0, 64, 16)


# ─── TestSamplePatches ────────────────────────────────────────────────────────

class TestSamplePatches:
    def test_returns_sample_result(self):
        result = sample_patches(_gray())
        assert isinstance(result, SampleResult)

    def test_grid_mode(self):
        cfg = SampleConfig(mode="grid", patch_size=16, n_patches=4)
        result = sample_patches(_gray(64, 64), cfg)
        assert result.n_patches <= 4

    def test_random_mode(self):
        cfg = SampleConfig(mode="random", patch_size=16, n_patches=5, seed=0)
        result = sample_patches(_gray(64, 64), cfg)
        assert result.n_patches == 5

    def test_border_mode(self):
        cfg = SampleConfig(mode="border", patch_size=16, n_patches=20)
        result = sample_patches(_gray(64, 64), cfg)
        assert result.n_patches > 0

    def test_stride_mode(self):
        cfg = SampleConfig(mode="stride", patch_size=16, stride=8, n_patches=50)
        result = sample_patches(_gray(64, 64), cfg)
        assert result.n_patches > 0

    def test_image_shape_stored(self):
        img = _rgb(40, 60)
        result = sample_patches(img)
        assert result.image_shape[:2] == (40, 60)

    def test_non_2d_3d_raises(self):
        with pytest.raises(ValueError):
            sample_patches(np.zeros((4, 4, 3, 2)))

    def test_default_config(self):
        result = sample_patches(_gray(64, 64))
        assert isinstance(result, SampleResult)

    def test_rgb_image(self):
        result = sample_patches(_rgb(64, 64), SampleConfig(mode="grid", patch_size=16))
        assert result.n_patches > 0


# ─── TestExtractPatchImages ───────────────────────────────────────────────────

class TestExtractPatchImages:
    def test_returns_list_of_arrays(self):
        img = _gray(64, 64)
        result = sample_patches(img, SampleConfig(mode="grid", patch_size=16, n_patches=4))
        patches = extract_patch_images(img, result)
        assert isinstance(patches, list)
        assert all(isinstance(p, np.ndarray) for p in patches)

    def test_count_matches_result(self):
        img = _gray(64, 64)
        result = sample_patches(img, SampleConfig(mode="random", patch_size=16, n_patches=6))
        patches = extract_patch_images(img, result)
        assert len(patches) == result.n_patches

    def test_patch_shape_correct(self):
        img = _gray(64, 64)
        cfg = SampleConfig(mode="grid", patch_size=16, n_patches=4)
        result = sample_patches(img, cfg)
        patches = extract_patch_images(img, result)
        assert all(p.shape[0] <= 16 and p.shape[1] <= 16 for p in patches)

    def test_copy_not_view(self):
        img = _gray(64, 64)
        result = sample_patches(img, SampleConfig(mode="grid", patch_size=16))
        patches = extract_patch_images(img, result)
        # Modifying patch should not affect original
        if patches:
            patches[0][:] = 255
        assert img[0, 0] == 0


# ─── TestBatchSamplePatches ───────────────────────────────────────────────────

class TestBatchSamplePatches:
    def test_returns_list(self):
        imgs = [_gray(), _gray()]
        results = batch_sample_patches(imgs)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_is_sample_result(self):
        imgs = [_gray(), _rgb()]
        results = batch_sample_patches(imgs)
        assert all(isinstance(r, SampleResult) for r in results)

    def test_empty_list(self):
        results = batch_sample_patches([])
        assert results == []

    def test_custom_config(self):
        cfg = SampleConfig(mode="random", patch_size=16, n_patches=3, seed=1)
        imgs = [_gray(32, 32)]
        results = batch_sample_patches(imgs, cfg)
        assert results[0].n_patches == 3
