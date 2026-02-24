"""Extra tests for puzzle_reconstruction/preprocessing/patch_sampler.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.patch_sampler import (
    SampleConfig,
    PatchSample,
    SampleResult,
    sample_grid_patches,
    sample_random_patches,
    sample_border_patches,
    sample_stride_patches,
    sample_patches,
    extract_patch_images,
    batch_sample_patches,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=100, w=100, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=100, w=100, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


# ─── SampleConfig ─────────────────────────────────────────────────────────────

class TestSampleConfigExtra:
    def test_defaults(self):
        c = SampleConfig()
        assert c.patch_size == 32
        assert c.n_patches == 16
        assert c.mode == "grid"

    def test_valid_modes(self):
        for m in ("grid", "random", "border", "stride"):
            SampleConfig(mode=m)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(mode="bad")

    def test_small_patch_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(patch_size=1)

    def test_zero_n_patches_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(n_patches=0)

    def test_zero_stride_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(stride=0)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(seed=-1)


# ─── PatchSample ──────────────────────────────────────────────────────────────

class TestPatchSampleExtra:
    def test_properties(self):
        p = PatchSample(idx=0, x=10, y=20, w=30, h=40)
        assert p.x2 == 40
        assert p.y2 == 60
        assert p.area == 1200
        assert p.center == (25.0, 40.0)

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=-1, x=0, y=0, w=10, h=10)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=-1, y=0, w=10, h=10)

    def test_zero_w_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=0, w=0, h=10)


# ─── SampleResult ─────────────────────────────────────────────────────────────

class TestSampleResultExtra:
    def test_coverage_ratio(self):
        samples = [PatchSample(idx=0, x=0, y=0, w=50, h=50)]
        r = SampleResult(samples=samples, image_shape=(100, 100), n_patches=1)
        assert r.coverage_ratio == pytest.approx(0.25)

    def test_negative_n_patches_raises(self):
        with pytest.raises(ValueError):
            SampleResult(samples=[], image_shape=(10, 10), n_patches=-1)


# ─── sample_grid_patches ──────────────────────────────────────────────────────

class TestSampleGridPatchesExtra:
    def test_basic(self):
        patches = sample_grid_patches(100, 100, 32)
        assert len(patches) > 0
        assert all(isinstance(p, PatchSample) for p in patches)

    def test_max_patches(self):
        patches = sample_grid_patches(100, 100, 10, max_patches=3)
        assert len(patches) == 3

    def test_patch_larger_than_image(self):
        patches = sample_grid_patches(10, 10, 32)
        # patch_size clamped to image size
        assert all(p.w <= 10 for p in patches)

    def test_bad_image_size_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(0, 10, 5)


# ─── sample_random_patches ────────────────────────────────────────────────────

class TestSampleRandomPatchesExtra:
    def test_count(self):
        patches = sample_random_patches(100, 100, 10, n_patches=5)
        assert len(patches) == 5

    def test_deterministic(self):
        p1 = sample_random_patches(100, 100, 10, 5, seed=42)
        p2 = sample_random_patches(100, 100, 10, 5, seed=42)
        assert [(p.x, p.y) for p in p1] == [(p.x, p.y) for p in p2]

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(100, 100, 10, 5, seed=-1)


# ─── sample_border_patches ────────────────────────────────────────────────────

class TestSampleBorderPatchesExtra:
    def test_basic(self):
        patches = sample_border_patches(100, 100, 10)
        assert len(patches) > 0

    def test_on_border(self):
        patches = sample_border_patches(100, 100, 10)
        for p in patches:
            on_border = (p.x == 0 or p.x2 == 100 or
                         p.y == 0 or p.y2 == 100)
            assert on_border


# ─── sample_stride_patches ────────────────────────────────────────────────────

class TestSampleStridePatchesExtra:
    def test_basic(self):
        patches = sample_stride_patches(100, 100, 10, stride=10)
        assert len(patches) > 0

    def test_zero_stride_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(100, 100, 10, stride=0)


# ─── sample_patches ──────────────────────────────────────────────────────────

class TestSamplePatchesExtra:
    def test_grid(self):
        cfg = SampleConfig(mode="grid", patch_size=20, n_patches=10)
        r = sample_patches(_gray(), cfg)
        assert isinstance(r, SampleResult)
        assert r.n_patches > 0

    def test_random(self):
        cfg = SampleConfig(mode="random", patch_size=20, n_patches=5)
        r = sample_patches(_gray(), cfg)
        assert r.n_patches == 5

    def test_border(self):
        cfg = SampleConfig(mode="border", patch_size=20)
        r = sample_patches(_gray(), cfg)
        assert r.n_patches > 0

    def test_stride(self):
        cfg = SampleConfig(mode="stride", patch_size=20, stride=10)
        r = sample_patches(_gray(), cfg)
        assert r.n_patches > 0

    def test_bgr(self):
        r = sample_patches(_bgr())
        assert r.image_shape == (100, 100, 3)


# ─── extract_patch_images ─────────────────────────────────────────────────────

class TestExtractPatchImagesExtra:
    def test_basic(self):
        img = _gray()
        r = sample_patches(img, SampleConfig(patch_size=20, n_patches=4))
        patches = extract_patch_images(img, r)
        assert len(patches) == r.n_patches
        for p in patches:
            assert p.shape[0] > 0 and p.shape[1] > 0


# ─── batch_sample_patches ────────────────────────────────────────────────────

class TestBatchSamplePatchesExtra:
    def test_empty(self):
        assert batch_sample_patches([]) == []

    def test_length(self):
        results = batch_sample_patches([_gray(), _gray()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_sample_patches([_gray()])
        assert isinstance(results[0], SampleResult)
