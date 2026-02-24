"""Extra tests for puzzle_reconstruction/preprocessing/patch_sampler.py."""
from __future__ import annotations

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

def _img(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w), dtype=np.uint8)


def _sample(idx: int = 0, x: int = 0, y: int = 0,
            w: int = 16, h: int = 16) -> PatchSample:
    return PatchSample(idx=idx, x=x, y=y, w=w, h=h)


def _result(n: int = 4, img_h: int = 64, img_w: int = 64) -> SampleResult:
    samples = [_sample(idx=i, x=i * 16, y=0) for i in range(n)]
    return SampleResult(samples=samples, image_shape=(img_h, img_w), n_patches=n)


# ─── SampleConfig (extra) ─────────────────────────────────────────────────────

class TestSampleConfigExtra:
    def test_default_patch_size(self):
        assert SampleConfig().patch_size == 32

    def test_default_n_patches(self):
        assert SampleConfig().n_patches == 16

    def test_default_mode(self):
        assert SampleConfig().mode == "grid"

    def test_default_stride(self):
        assert SampleConfig().stride == 8

    def test_default_seed(self):
        assert SampleConfig().seed == 0

    def test_patch_size_lt_2_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(patch_size=1)

    def test_n_patches_zero_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(n_patches=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(mode="diagonal")

    def test_stride_zero_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(stride=0)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            SampleConfig(seed=-1)

    def test_custom_values(self):
        cfg = SampleConfig(patch_size=8, n_patches=10, mode="random",
                           stride=4, seed=42)
        assert cfg.patch_size == 8
        assert cfg.n_patches == 10

    def test_all_modes_valid(self):
        for mode in ("grid", "random", "border", "stride"):
            cfg = SampleConfig(mode=mode)
            assert cfg.mode == mode


# ─── PatchSample (extra) ──────────────────────────────────────────────────────

class TestPatchSampleExtra:
    def test_idx_stored(self):
        assert _sample(idx=5).idx == 5

    def test_x_stored(self):
        assert _sample(x=10).x == 10

    def test_y_stored(self):
        assert _sample(y=20).y == 20

    def test_w_stored(self):
        assert _sample(w=24).w == 24

    def test_h_stored(self):
        assert _sample(h=32).h == 32

    def test_negative_idx_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=-1, x=0, y=0, w=8, h=8)

    def test_negative_x_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=-1, y=0, w=8, h=8)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=-1, w=8, h=8)

    def test_w_zero_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=0, w=0, h=8)

    def test_h_zero_raises(self):
        with pytest.raises(ValueError):
            PatchSample(idx=0, x=0, y=0, w=8, h=0)

    def test_x2_property(self):
        s = _sample(x=5, w=10)
        assert s.x2 == 15

    def test_y2_property(self):
        s = _sample(y=7, h=12)
        assert s.y2 == 19

    def test_area_property(self):
        s = _sample(w=4, h=5)
        assert s.area == 20

    def test_center_property(self):
        s = _sample(x=0, y=0, w=16, h=16)
        cx, cy = s.center
        assert cx == pytest.approx(8.0)
        assert cy == pytest.approx(8.0)


# ─── SampleResult (extra) ─────────────────────────────────────────────────────

class TestSampleResultExtra:
    def test_samples_stored(self):
        r = _result(3)
        assert len(r.samples) == 3

    def test_n_patches_stored(self):
        r = _result(5)
        assert r.n_patches == 5

    def test_image_shape_stored(self):
        r = _result(img_h=32, img_w=48)
        assert r.image_shape == (32, 48)

    def test_negative_n_patches_raises(self):
        with pytest.raises(ValueError):
            SampleResult(samples=[], image_shape=(64, 64), n_patches=-1)

    def test_coverage_ratio_zero_patches(self):
        r = SampleResult(samples=[], image_shape=(64, 64), n_patches=0)
        assert r.coverage_ratio == pytest.approx(0.0)

    def test_coverage_ratio_single_patch(self):
        s = _sample(x=0, y=0, w=32, h=32)
        r = SampleResult(samples=[s], image_shape=(64, 64), n_patches=1)
        assert r.coverage_ratio == pytest.approx(0.25)  # 32*32 / 64*64

    def test_coverage_ratio_full(self):
        s = _sample(x=0, y=0, w=64, h=64)
        r = SampleResult(samples=[s], image_shape=(64, 64), n_patches=1)
        assert r.coverage_ratio == pytest.approx(1.0)

    def test_coverage_ratio_zero_image(self):
        r = SampleResult(samples=[_sample()], image_shape=(0, 0), n_patches=1)
        assert r.coverage_ratio == pytest.approx(0.0)


# ─── sample_grid_patches (extra) ──────────────────────────────────────────────

class TestSampleGridPatchesExtra:
    def test_returns_list(self):
        assert isinstance(sample_grid_patches(64, 64, 16), list)

    def test_all_patch_samples(self):
        for s in sample_grid_patches(64, 64, 16):
            assert isinstance(s, PatchSample)

    def test_correct_count(self):
        # 64x64, patch=32 → 4 patches (2x2)
        result = sample_grid_patches(64, 64, 32)
        assert len(result) == 4

    def test_max_patches_respected(self):
        result = sample_grid_patches(64, 64, 16, max_patches=3)
        assert len(result) <= 3

    def test_image_too_small_clamped(self):
        # patch_size is clamped to min(patch_size, h, w), so patches still extracted
        result = sample_grid_patches(8, 8, 32)
        assert len(result) >= 1
        # but patch w/h are clamped
        for s in result:
            assert s.w <= 8 and s.h <= 8

    def test_image_h_zero_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(0, 64, 16)

    def test_patch_size_lt_2_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(64, 64, 1)

    def test_max_patches_lt_1_raises(self):
        with pytest.raises(ValueError):
            sample_grid_patches(64, 64, 16, max_patches=0)

    def test_idx_sequential(self):
        result = sample_grid_patches(64, 64, 16)
        for i, s in enumerate(result):
            assert s.idx == i

    def test_patches_within_bounds(self):
        for s in sample_grid_patches(64, 64, 16):
            assert s.x + s.w <= 64
            assert s.y + s.h <= 64


# ─── sample_random_patches (extra) ────────────────────────────────────────────

class TestSampleRandomPatchesExtra:
    def test_returns_list(self):
        assert isinstance(sample_random_patches(64, 64, 16, 10), list)

    def test_n_patches_correct(self):
        result = sample_random_patches(64, 64, 16, 12, seed=0)
        assert len(result) == 12

    def test_image_too_small_clamped(self):
        # patch_size clamped to min(32, 8, 8) = 8, so patches are still returned
        result = sample_random_patches(8, 8, 32, 10)
        assert len(result) == 10
        for s in result:
            assert s.w <= 8 and s.h <= 8

    def test_image_h_zero_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(0, 64, 16, 5)

    def test_patch_size_lt_2_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(64, 64, 1, 5)

    def test_n_patches_lt_1_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(64, 64, 16, 0)

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError):
            sample_random_patches(64, 64, 16, 5, seed=-1)

    def test_reproducible_with_seed(self):
        r1 = sample_random_patches(64, 64, 16, 5, seed=7)
        r2 = sample_random_patches(64, 64, 16, 5, seed=7)
        for s1, s2 in zip(r1, r2):
            assert s1.x == s2.x and s1.y == s2.y

    def test_patches_within_bounds(self):
        for s in sample_random_patches(64, 64, 16, 20, seed=0):
            assert s.x + s.w <= 64
            assert s.y + s.h <= 64


# ─── sample_border_patches (extra) ────────────────────────────────────────────

class TestSampleBorderPatchesExtra:
    def test_returns_list(self):
        assert isinstance(sample_border_patches(64, 64, 16), list)

    def test_all_patch_samples(self):
        for s in sample_border_patches(64, 64, 16):
            assert isinstance(s, PatchSample)

    def test_max_patches_respected(self):
        result = sample_border_patches(64, 64, 16, max_patches=4)
        assert len(result) <= 4

    def test_image_too_small_clamped(self):
        # patch_size clamped to min(32, 4, 4) = 4
        result = sample_border_patches(4, 4, 32)
        for s in result:
            assert s.w <= 4 and s.h <= 4

    def test_image_h_zero_raises(self):
        with pytest.raises(ValueError):
            sample_border_patches(0, 64, 16)

    def test_patch_size_lt_2_raises(self):
        with pytest.raises(ValueError):
            sample_border_patches(64, 64, 1)

    def test_patches_within_bounds(self):
        for s in sample_border_patches(64, 64, 16):
            assert s.x + s.w <= 64
            assert s.y + s.h <= 64

    def test_no_duplicate_positions(self):
        result = sample_border_patches(64, 64, 16)
        positions = [(s.x, s.y) for s in result]
        assert len(positions) == len(set(positions))


# ─── sample_stride_patches (extra) ────────────────────────────────────────────

class TestSampleStridePatchesExtra:
    def test_returns_list(self):
        assert isinstance(sample_stride_patches(64, 64, 16), list)

    def test_max_patches_respected(self):
        result = sample_stride_patches(64, 64, 16, stride=8, max_patches=5)
        assert len(result) <= 5

    def test_stride_1_max_patches(self):
        # (32-16+1)^2 = 289
        result = sample_stride_patches(32, 32, 16, stride=1, max_patches=1000)
        assert len(result) == (32 - 16 + 1) ** 2

    def test_image_h_zero_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(0, 64, 16)

    def test_patch_size_lt_2_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(64, 64, 1)

    def test_stride_zero_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(64, 64, 16, stride=0)

    def test_max_patches_zero_raises(self):
        with pytest.raises(ValueError):
            sample_stride_patches(64, 64, 16, max_patches=0)

    def test_patches_within_bounds(self):
        for s in sample_stride_patches(64, 64, 16, stride=8):
            assert s.x + s.w <= 64
            assert s.y + s.h <= 64

    def test_idx_sequential(self):
        result = sample_stride_patches(64, 64, 16, stride=16)
        for i, s in enumerate(result):
            assert s.idx == i


# ─── sample_patches (extra) ───────────────────────────────────────────────────

class TestSamplePatchesExtra:
    def test_returns_sample_result(self):
        assert isinstance(sample_patches(_img()), SampleResult)

    def test_none_cfg_uses_defaults(self):
        result = sample_patches(_img(), cfg=None)
        assert isinstance(result, SampleResult)

    def test_1d_raises(self):
        with pytest.raises(ValueError):
            sample_patches(np.zeros(64))

    def test_grid_mode(self):
        cfg = SampleConfig(mode="grid", patch_size=16, n_patches=100)
        result = sample_patches(_img(64, 64), cfg)
        assert result.n_patches > 0

    def test_random_mode(self):
        cfg = SampleConfig(mode="random", patch_size=16, n_patches=8)
        result = sample_patches(_img(64, 64), cfg)
        assert result.n_patches == 8

    def test_border_mode(self):
        cfg = SampleConfig(mode="border", patch_size=16, n_patches=10)
        result = sample_patches(_img(64, 64), cfg)
        assert isinstance(result, SampleResult)

    def test_stride_mode(self):
        cfg = SampleConfig(mode="stride", patch_size=16, stride=8, n_patches=50)
        result = sample_patches(_img(64, 64), cfg)
        assert result.n_patches > 0

    def test_image_shape_stored(self):
        img = _img(32, 48)
        result = sample_patches(img)
        assert result.image_shape[:2] == (32, 48)

    def test_n_patches_matches_samples(self):
        result = sample_patches(_img())
        assert result.n_patches == len(result.samples)

    def test_grayscale_supported(self):
        result = sample_patches(_gray(64, 64))
        assert isinstance(result, SampleResult)


# ─── extract_patch_images (extra) ─────────────────────────────────────────────

class TestExtractPatchImagesExtra:
    def test_returns_list(self):
        img = _img(64, 64)
        result = sample_patches(img, SampleConfig(mode="grid", patch_size=16))
        assert isinstance(extract_patch_images(img, result), list)

    def test_length_matches_n_patches(self):
        img = _img(64, 64)
        result = sample_patches(img, SampleConfig(mode="grid", patch_size=16))
        patches = extract_patch_images(img, result)
        assert len(patches) == result.n_patches

    def test_patches_are_ndarrays(self):
        img = _img(64, 64)
        result = sample_patches(img, SampleConfig(mode="grid", patch_size=16))
        for p in extract_patch_images(img, result):
            assert isinstance(p, np.ndarray)

    def test_empty_samples(self):
        img = _img(64, 64)
        empty_result = SampleResult(samples=[], image_shape=(64, 64), n_patches=0)
        assert extract_patch_images(img, empty_result) == []

    def test_patch_shape_matches_sample(self):
        img = _img(64, 64)
        cfg = SampleConfig(mode="grid", patch_size=16)
        result = sample_patches(img, cfg)
        patches = extract_patch_images(img, result)
        for patch, s in zip(patches, result.samples):
            assert patch.shape[0] == s.h
            assert patch.shape[1] == s.w


# ─── batch_sample_patches (extra) ─────────────────────────────────────────────

class TestBatchSamplePatchesExtra:
    def test_returns_list(self):
        assert isinstance(batch_sample_patches([_img()]), list)

    def test_empty_input(self):
        assert batch_sample_patches([]) == []

    def test_length_matches_images(self):
        imgs = [_img() for _ in range(4)]
        result = batch_sample_patches(imgs)
        assert len(result) == 4

    def test_all_sample_results(self):
        for r in batch_sample_patches([_img(), _img()]):
            assert isinstance(r, SampleResult)

    def test_none_cfg_uses_defaults(self):
        result = batch_sample_patches([_img()], cfg=None)
        assert len(result) == 1

    def test_custom_cfg_applied(self):
        cfg = SampleConfig(mode="random", patch_size=8, n_patches=5)
        results = batch_sample_patches([_img(64, 64)], cfg=cfg)
        assert results[0].n_patches == 5
