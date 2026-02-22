"""Tests for puzzle_reconstruction/algorithms/patch_aligner.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.patch_aligner import (
    AlignConfig,
    AlignResult,
    phase_correlate,
    ncc_score,
    align_patches,
    refine_alignment,
    batch_align,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_patch(h=32, w=32, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def make_gradient_patch(h=32, w=32):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def make_bgr_patch(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 130
    img[:, :, 2] = 200
    return img


# ─── AlignConfig ──────────────────────────────────────────────────────────────

class TestAlignConfig:
    def test_defaults(self):
        cfg = AlignConfig()
        assert cfg.method == "combined"
        assert cfg.max_shift == pytest.approx(20.0)
        assert cfg.upsample_factor == 1
        assert cfg.ncc_threshold == pytest.approx(0.5)
        assert cfg.refine_radius == 2

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(method="unknown")

    def test_valid_methods(self):
        for m in ("phase", "ncc", "combined"):
            cfg = AlignConfig(method=m)
            assert cfg.method == m

    def test_max_shift_zero_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(max_shift=0.0)

    def test_max_shift_negative_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(max_shift=-5.0)

    def test_upsample_factor_zero_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(upsample_factor=0)

    def test_upsample_factor_negative_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(upsample_factor=-1)

    def test_ncc_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(ncc_threshold=1.1)

    def test_ncc_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(ncc_threshold=-0.1)

    def test_refine_radius_negative_raises(self):
        with pytest.raises(ValueError):
            AlignConfig(refine_radius=-1)

    def test_refine_radius_zero_valid(self):
        cfg = AlignConfig(refine_radius=0)
        assert cfg.refine_radius == 0


# ─── AlignResult ──────────────────────────────────────────────────────────────

class TestAlignResult:
    def test_basic_creation(self):
        r = AlignResult(shift=(1.0, 2.0), ncc=0.8, psnr=30.0,
                        success=True, method="combined")
        assert r.shift == (1.0, 2.0)
        assert r.ncc == pytest.approx(0.8)
        assert r.success is True

    def test_ncc_above_one_raises(self):
        with pytest.raises(ValueError):
            AlignResult(shift=(0.0, 0.0), ncc=1.1, psnr=30.0,
                        success=True, method="combined")

    def test_ncc_below_minus_one_raises(self):
        with pytest.raises(ValueError):
            AlignResult(shift=(0.0, 0.0), ncc=-1.1, psnr=30.0,
                        success=True, method="combined")

    def test_psnr_negative_raises(self):
        with pytest.raises(ValueError):
            AlignResult(shift=(0.0, 0.0), ncc=0.5, psnr=-1.0,
                        success=True, method="combined")

    def test_shift_magnitude_zero(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=0.5, psnr=0.0,
                        success=True, method="phase")
        assert r.shift_magnitude == pytest.approx(0.0)

    def test_shift_magnitude_pythagorean(self):
        r = AlignResult(shift=(3.0, 4.0), ncc=0.5, psnr=0.0,
                        success=True, method="phase")
        assert r.shift_magnitude == pytest.approx(5.0)

    def test_params_default_empty(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=0.0, psnr=0.0,
                        success=False, method="ncc")
        assert r.params == {}

    def test_ncc_zero_valid(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=0.0, psnr=0.0,
                        success=False, method="ncc")
        assert r.ncc == pytest.approx(0.0)


# ─── phase_correlate ──────────────────────────────────────────────────────────

class TestPhaseCorrelate:
    def test_upsample_factor_zero_raises(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        with pytest.raises(ValueError):
            phase_correlate(a, b, upsample_factor=0)

    def test_shape_mismatch_raises(self):
        a = make_gradient_patch(h=32, w=32)
        b = make_gradient_patch(h=32, w=16)
        with pytest.raises(ValueError):
            phase_correlate(a, b)

    def test_returns_tuple_of_two(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = phase_correlate(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        dy, dx = phase_correlate(a, b)
        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_identical_patches_near_zero(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        dy, dx = phase_correlate(a, b)
        # Should be near zero for identical patches
        assert abs(dy) < 5.0
        assert abs(dx) < 5.0

    def test_bgr_input(self):
        a = make_bgr_patch()
        b = make_bgr_patch()
        dy, dx = phase_correlate(a, b)
        assert isinstance(dy, float)

    def test_upsample_factor_1(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        dy, dx = phase_correlate(a, b, upsample_factor=1)
        assert isinstance(dy, float)

    def test_upsample_factor_2(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        dy, dx = phase_correlate(a, b, upsample_factor=2)
        assert isinstance(dy, float)


# ─── ncc_score ────────────────────────────────────────────────────────────────

class TestNccScore:
    def test_shape_mismatch_raises(self):
        a = make_gradient_patch(h=32, w=32)
        b = make_gradient_patch(h=32, w=16)
        with pytest.raises(ValueError):
            ncc_score(a, b)

    def test_identical_patches_one(self):
        a = make_gradient_patch()
        result = ncc_score(a, a)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_constant_patches_zero(self):
        a = make_patch(value=128)
        b = make_patch(value=200)
        result = ncc_score(a, b)
        assert result == pytest.approx(0.0)

    def test_range_minus1_to_1(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = ncc_score(a, b)
        assert -1.0 <= result <= 1.0

    def test_returns_float(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = ncc_score(a, b)
        assert isinstance(result, float)

    def test_bgr_input(self):
        a = make_bgr_patch()
        b = make_bgr_patch()
        result = ncc_score(a, b)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_inverted_patch(self):
        a = make_gradient_patch().astype(np.float32)
        b = (255.0 - a).astype(np.uint8)
        result = ncc_score(a.astype(np.uint8), b)
        assert result < 0.0


# ─── align_patches ────────────────────────────────────────────────────────────

class TestAlignPatches:
    def test_returns_align_result(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = align_patches(a, b)
        assert isinstance(result, AlignResult)

    def test_identical_patches_success(self):
        a = make_gradient_patch()
        cfg = AlignConfig(ncc_threshold=0.9)
        result = align_patches(a, a, cfg=cfg)
        assert result.success is True

    def test_method_phase(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        cfg = AlignConfig(method="phase")
        result = align_patches(a, b, cfg=cfg)
        assert result.method == "phase"

    def test_method_ncc(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        cfg = AlignConfig(method="ncc")
        result = align_patches(a, b, cfg=cfg)
        assert result.method == "ncc"

    def test_method_combined(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        cfg = AlignConfig(method="combined")
        result = align_patches(a, b, cfg=cfg)
        assert result.method == "combined"

    def test_ncc_in_range(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = align_patches(a, b)
        assert -1.0 <= result.ncc <= 1.0

    def test_psnr_nonneg(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = align_patches(a, b)
        assert result.psnr >= 0.0

    def test_default_config(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = align_patches(a, b)
        assert result.method in ("phase", "ncc", "combined")


# ─── refine_alignment ─────────────────────────────────────────────────────────

class TestRefineAlignment:
    def test_radius_negative_raises(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        with pytest.raises(ValueError):
            refine_alignment(a, b, initial_shift=(0.0, 0.0), radius=-1)

    def test_returns_tuple_of_two(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = refine_alignment(a, b, initial_shift=(0.0, 0.0), radius=1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_floats(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        dy, dx = refine_alignment(a, b, initial_shift=(0.0, 0.0))
        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_radius_zero_returns_initial(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        dy, dx = refine_alignment(a, b, initial_shift=(2.0, 3.0), radius=0)
        assert dy == pytest.approx(2.0)
        assert dx == pytest.approx(3.0)

    def test_identical_patches_shift_near_zero(self):
        a = make_gradient_patch()
        dy, dx = refine_alignment(a, a, initial_shift=(0.0, 0.0), radius=2)
        # With identical patches, refined shift should stay near zero
        assert abs(dy) <= 2.0
        assert abs(dx) <= 2.0


# ─── batch_align ──────────────────────────────────────────────────────────────

class TestBatchAlign:
    def test_empty_list(self):
        result = batch_align([])
        assert result == []

    def test_returns_list(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = batch_align([(a, b), (a, b)])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_all_align_results(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = batch_align([(a, b), (b, a)])
        assert all(isinstance(r, AlignResult) for r in result)

    def test_custom_config(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        cfg = AlignConfig(method="ncc")
        result = batch_align([(a, b)], cfg=cfg)
        assert result[0].method == "ncc"

    def test_single_pair(self):
        a = make_gradient_patch()
        b = make_gradient_patch()
        result = batch_align([(a, b)])
        assert len(result) == 1
