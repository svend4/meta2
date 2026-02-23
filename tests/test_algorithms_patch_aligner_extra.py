"""Extra tests for puzzle_reconstruction/algorithms/patch_aligner.py"""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _flat(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _grad(h=32, w=32):
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(col, (h, 1))


def _bgr(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 130
    img[:, :, 2] = 200
    return img


# ─── TestAlignConfigExtra ─────────────────────────────────────────────────────

class TestAlignConfigExtra:
    def test_ncc_threshold_zero_valid(self):
        cfg = AlignConfig(ncc_threshold=0.0)
        assert cfg.ncc_threshold == pytest.approx(0.0)

    def test_ncc_threshold_one_valid(self):
        cfg = AlignConfig(ncc_threshold=1.0)
        assert cfg.ncc_threshold == pytest.approx(1.0)

    def test_default_max_shift(self):
        cfg = AlignConfig()
        assert cfg.max_shift == pytest.approx(20.0)

    def test_large_upsample_factor_valid(self):
        cfg = AlignConfig(upsample_factor=8)
        assert cfg.upsample_factor == 8

    def test_phase_method(self):
        cfg = AlignConfig(method="phase")
        assert cfg.method == "phase"


# ─── TestAlignResultExtra ─────────────────────────────────────────────────────

class TestAlignResultExtra:
    def test_method_stored(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=0.5, psnr=0.0,
                        success=True, method="phase")
        assert r.method == "phase"

    def test_psnr_zero_valid(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=0.0, psnr=0.0,
                        success=False, method="ncc")
        assert r.psnr == pytest.approx(0.0)

    def test_ncc_minus_one_valid(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=-1.0, psnr=0.0,
                        success=False, method="ncc")
        assert r.ncc == pytest.approx(-1.0)

    def test_ncc_one_valid(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=1.0, psnr=0.0,
                        success=True, method="combined")
        assert r.ncc == pytest.approx(1.0)

    def test_shift_magnitude_positive(self):
        r = AlignResult(shift=(1.0, 1.0), ncc=0.0, psnr=0.0,
                        success=False, method="phase")
        assert r.shift_magnitude == pytest.approx(np.sqrt(2.0), abs=1e-9)

    def test_params_custom(self):
        r = AlignResult(shift=(0.0, 0.0), ncc=0.0, psnr=0.0,
                        success=False, method="ncc",
                        params={"key": 42})
        assert r.params["key"] == 42


# ─── TestPhaseCorrelateExtra ──────────────────────────────────────────────────

class TestPhaseCorrelateExtra:
    def test_non_square_patch(self):
        a = _grad(h=16, w=32)
        b = _grad(h=16, w=32)
        dy, dx = phase_correlate(a, b)
        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_upsample_factor_4(self):
        a = _grad()
        b = _grad()
        dy, dx = phase_correlate(a, b, upsample_factor=4)
        assert isinstance(dy, float)

    def test_constant_patches(self):
        a = _flat(val=100)
        b = _flat(val=200)
        dy, dx = phase_correlate(a, b)
        # constant patches: trivial result but should not raise
        assert isinstance(dy, float)

    def test_bgr_both_patches(self):
        a = _bgr()
        b = _bgr()
        dy, dx = phase_correlate(a, b)
        assert isinstance(dx, float)


# ─── TestNccScoreExtra ────────────────────────────────────────────────────────

class TestNccScoreExtra:
    def test_scaled_patch_same_ncc(self):
        # gradient 0..127, doubling gives 0..254 with no clipping → same NCC
        col = np.linspace(0, 127, 32, dtype=np.uint8)
        a = np.tile(col, (32, 1))
        b = np.clip(a.astype(np.uint16) * 2, 0, 255).astype(np.uint8)
        result = ncc_score(a, b)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_non_square_patches(self):
        a = _grad(h=16, w=32)
        b = _grad(h=16, w=32)
        result = ncc_score(a, b)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_all_same_value_returns_zero(self):
        a = _flat(val=50)
        b = _flat(val=50)
        result = ncc_score(a, b)
        assert result == pytest.approx(0.0)

    def test_bgr_identical(self):
        a = _bgr()
        result = ncc_score(a, a)
        assert result == pytest.approx(1.0, abs=1e-5)


# ─── TestAlignPatchesExtra ────────────────────────────────────────────────────

class TestAlignPatchesExtra:
    def test_method_attribute_matches_config(self):
        a = _grad()
        cfg = AlignConfig(method="phase")
        result = align_patches(a, a, cfg=cfg)
        assert result.method == "phase"

    def test_psnr_nonneg_identical(self):
        a = _grad()
        result = align_patches(a, a)
        assert result.psnr >= 0.0

    def test_ncc_range(self):
        a = _grad()
        b = _flat()
        result = align_patches(a, b)
        assert -1.0 <= result.ncc <= 1.0

    def test_success_flag_low_threshold(self):
        a = _grad()
        cfg = AlignConfig(ncc_threshold=0.0)
        result = align_patches(a, a, cfg=cfg)
        assert result.success is True

    def test_params_contains_max_shift(self):
        a = _grad()
        result = align_patches(a, a)
        assert "max_shift" in result.params

    def test_ncc_method_zero_shift(self):
        a = _grad()
        cfg = AlignConfig(method="ncc")
        result = align_patches(a, a, cfg=cfg)
        assert result.shift == (0.0, 0.0)


# ─── TestRefineAlignmentExtra ─────────────────────────────────────────────────

class TestRefineAlignmentExtra:
    def test_large_radius(self):
        a = _grad()
        b = _grad()
        dy, dx = refine_alignment(a, b, initial_shift=(0.0, 0.0), radius=5)
        assert isinstance(dy, float)
        assert isinstance(dx, float)

    def test_non_square_patches(self):
        a = _grad(h=16, w=32)
        b = _grad(h=16, w=32)
        dy, dx = refine_alignment(a, b, initial_shift=(0.0, 0.0), radius=2)
        assert isinstance(dy, float)

    def test_initial_shift_preserved_radius_zero(self):
        a = _grad()
        b = _flat()
        dy, dx = refine_alignment(a, b, initial_shift=(3.0, -2.0), radius=0)
        assert dy == pytest.approx(3.0)
        assert dx == pytest.approx(-2.0)


# ─── TestBatchAlignExtra ──────────────────────────────────────────────────────

class TestBatchAlignExtra:
    def test_mixed_methods(self):
        a = _grad()
        b = _grad()
        cfg_ncc = AlignConfig(method="ncc")
        results = batch_align([(a, b), (b, a)], cfg=cfg_ncc)
        assert all(r.method == "ncc" for r in results)

    def test_three_pairs(self):
        pairs = [(_grad(), _flat()), (_flat(), _grad()), (_grad(), _grad())]
        results = batch_align(pairs)
        assert len(results) == 3

    def test_all_results_have_ncc_in_range(self):
        a = _grad()
        b = _flat()
        results = batch_align([(a, b), (b, a), (a, a)])
        assert all(-1.0 <= r.ncc <= 1.0 for r in results)
