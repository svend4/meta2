"""Extra tests for puzzle_reconstruction/utils/patch_utils.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.patch_utils import (
    PatchConfig,
    batch_compare,
    compare_patches,
    extract_patch,
    extract_patches,
    normalize_patch,
    patch_mse,
    patch_ncc,
    patch_ssd,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)

def _bgr(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)

def _patch(h=16, w=16, fill=100):
    return np.full((h, w), fill, dtype=np.uint8)


# ─── TestPatchConfigExtra ─────────────────────────────────────────────────────

class TestPatchConfigExtra:
    def test_patch_1x1_valid(self):
        cfg = PatchConfig(patch_h=1, patch_w=1)
        assert cfg.patch_h == 1
        assert cfg.patch_w == 1

    def test_normalize_true_stored(self):
        cfg = PatchConfig(normalize=True)
        assert cfg.normalize is True

    def test_large_patch_valid(self):
        cfg = PatchConfig(patch_h=256, patch_w=256)
        assert cfg.patch_h == 256
        assert cfg.patch_w == 256

    def test_pad_value_0_valid(self):
        cfg = PatchConfig(pad_value=0)
        assert cfg.pad_value == 0

    def test_pad_value_128_valid(self):
        cfg = PatchConfig(pad_value=128)
        assert cfg.pad_value == 128

    def test_zscore_norm_mode(self):
        cfg = PatchConfig(norm_mode="zscore")
        assert cfg.norm_mode == "zscore"

    def test_minmax_norm_mode(self):
        cfg = PatchConfig(norm_mode="minmax")
        assert cfg.norm_mode == "minmax"

    def test_patch_h_100_valid(self):
        cfg = PatchConfig(patch_h=100)
        assert cfg.patch_h == 100


# ─── TestExtractPatchExtra ────────────────────────────────────────────────────

class TestExtractPatchExtra:
    def test_different_patch_sizes(self):
        img = _gray(64, 64)
        for h, w in [(8, 8), (16, 32), (32, 16)]:
            cfg = PatchConfig(patch_h=h, patch_w=w)
            result = extract_patch(img, 32, 32, cfg=cfg)
            assert result.shape == (h, w)

    def test_float32_input(self):
        img = _gray(64, 64).astype(np.float32)
        result = extract_patch(img, 32, 32)
        assert isinstance(result, np.ndarray)

    def test_corner_top_left(self):
        img = _gray(64, 64, fill=200)
        cfg = PatchConfig(patch_h=8, patch_w=8, pad_value=0)
        result = extract_patch(img, 0, 0, cfg=cfg)
        assert result.shape == (8, 8)

    def test_corner_bottom_right(self):
        img = _gray(64, 64, fill=200)
        cfg = PatchConfig(patch_h=8, patch_w=8, pad_value=0)
        result = extract_patch(img, 64, 64, cfg=cfg)
        assert result.shape == (8, 8)

    def test_normalize_zscore(self):
        img = _gray(64, 64, fill=128)
        cfg = PatchConfig(patch_h=8, patch_w=8, normalize=True, norm_mode="zscore")
        result = extract_patch(img, 32, 32, cfg=cfg)
        assert result.dtype == np.float32

    def test_bgr_image_correct_shape(self):
        img = _bgr(64, 64)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        result = extract_patch(img, 32, 32, cfg=cfg)
        assert result.shape == (8, 8, 3)

    def test_large_center_outside_image(self):
        img = _gray(32, 32)
        cfg = PatchConfig(patch_h=8, patch_w=8, pad_value=0)
        result = extract_patch(img, 100, 100, cfg=cfg)
        assert result.shape == (8, 8)


# ─── TestExtractPatchesExtra ─────────────────────────────────────────────────

class TestExtractPatchesExtra:
    def test_many_centers(self):
        img = _gray(64, 64)
        centers = [(i * 5, i * 5) for i in range(10)]
        result = extract_patches(img, centers)
        assert len(result) == 10

    def test_color_image(self):
        img = _bgr(64, 64)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        result = extract_patches(img, [(16, 16), (32, 32)], cfg=cfg)
        assert len(result) == 2
        assert result[0].shape == (8, 8, 3)

    def test_all_patches_same_shape(self):
        img = _gray(64, 64)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        centers = [(i * 10, 10) for i in range(5)]
        result = extract_patches(img, centers, cfg=cfg)
        for p in result:
            assert p.shape == (8, 8)

    def test_single_center(self):
        img = _gray(64, 64)
        result = extract_patches(img, [(32, 32)])
        assert len(result) == 1


# ─── TestNormalizePatchExtra ──────────────────────────────────────────────────

class TestNormalizePatchExtra:
    def test_zscore_std_approx_one(self):
        p = np.arange(16, dtype=np.float64).reshape(4, 4)
        result = normalize_patch(p, mode="zscore")
        # std should be ~1 after zscore
        std = float(result.std())
        assert abs(std - 1.0) < 1e-4

    def test_large_patch_minmax(self):
        p = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
        result = normalize_patch(p, mode="minmax")
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_float32_input_minmax(self):
        p = np.linspace(0.0, 1.0, 16).reshape(4, 4).astype(np.float32)
        result = normalize_patch(p, mode="minmax")
        assert result.dtype == np.float32

    def test_zscore_output_shape(self):
        p = np.arange(24, dtype=np.float32).reshape(4, 6)
        result = normalize_patch(p, mode="zscore")
        assert result.shape == (4, 6)

    def test_minmax_max_is_1(self):
        p = np.array([[0, 50, 100]], dtype=np.uint8)
        result = normalize_patch(p, mode="minmax")
        assert abs(float(result.max()) - 1.0) < 1e-6


# ─── TestPatchSsdExtra ────────────────────────────────────────────────────────

class TestPatchSsdExtra:
    def test_known_2(self):
        a = np.array([[0.0, 1.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0]], dtype=np.float32)
        # SSD = 1 + 1 = 2
        assert patch_ssd(a, b) == pytest.approx(2.0)

    def test_large_patches_nonneg(self):
        a = _patch(h=64, w=64, fill=50)
        b = _patch(h=64, w=64, fill=150)
        assert patch_ssd(a, b) >= 0.0

    def test_opposite_fills(self):
        a = _patch(fill=0)
        b = _patch(fill=255)
        ssd = patch_ssd(a, b)
        assert ssd > 0.0

    def test_single_pixel(self):
        a = np.array([[3.0]], dtype=np.float32)
        b = np.array([[7.0]], dtype=np.float32)
        # SSD = (3-7)^2 = 16
        assert patch_ssd(a, b) == pytest.approx(16.0)


# ─── TestPatchNccExtra ────────────────────────────────────────────────────────

class TestPatchNccExtra:
    def test_random_pair_in_range(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        b = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        result = patch_ncc(a, b)
        assert -1.0 <= result <= 1.0

    def test_orthogonal_approx_zero(self):
        # Two uncorrelated constant-zero arrays give NCC=0
        a = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        b = np.zeros((1, 4), dtype=np.float32)
        result = patch_ncc(a, b)
        assert result == pytest.approx(0.0)

    def test_multiple_seeds_in_range(self):
        for s in range(5):
            rng = np.random.default_rng(s)
            a = rng.integers(0, 256, (8, 8), dtype=np.uint8)
            b = rng.integers(0, 256, (8, 8), dtype=np.uint8)
            result = patch_ncc(a, b)
            assert -1.0 <= result <= 1.0


# ─── TestPatchMseExtra ────────────────────────────────────────────────────────

class TestPatchMseExtra:
    def test_large_patches_nonneg(self):
        a = _patch(h=64, w=64, fill=50)
        b = _patch(h=64, w=64, fill=150)
        assert patch_mse(a, b) >= 0.0

    def test_known_zero(self):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        assert patch_mse(a, a) == pytest.approx(0.0)

    def test_known_value_4(self):
        a = np.array([[0.0]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)
        # MSE = (2-0)^2 / 1 = 4
        assert patch_mse(a, b) == pytest.approx(4.0)

    def test_returns_float(self):
        a = _patch()
        b = _patch(fill=200)
        assert isinstance(patch_mse(a, b), float)


# ─── TestComparePatchesExtra ─────────────────────────────────────────────────

class TestComparePatchesExtra:
    def test_ssd_nonneg(self):
        a = _patch(fill=50)
        b = _patch(fill=200)
        result = compare_patches(a, b, method="ssd")
        assert result >= 0.0

    def test_mse_nonneg(self):
        a = _patch(fill=50)
        b = _patch(fill=200)
        result = compare_patches(a, b, method="mse")
        assert result >= 0.0

    def test_ncc_range(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        b = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        result = compare_patches(a, b, method="ncc")
        assert -1.0 <= result <= 1.0

    def test_all_three_methods(self):
        p = np.arange(16, dtype=np.float32).reshape(4, 4)
        for method in ("ssd", "mse", "ncc"):
            result = compare_patches(p, p, method=method)
            assert isinstance(result, float)


# ─── TestBatchCompareExtra ────────────────────────────────────────────────────

class TestBatchCompareExtra:
    def test_five_pairs(self):
        p = _patch()
        pairs = [(p, p)] * 5
        result = batch_compare(pairs)
        assert len(result) == 5

    def test_ssd_all_zero_for_identical(self):
        p = _patch()
        pairs = [(p, p), (p, p)]
        result = batch_compare(pairs, method="ssd")
        assert all(r == pytest.approx(0.0) for r in result)

    def test_mse_all_zero_for_identical(self):
        p = _patch()
        result = batch_compare([(p, p)], method="mse")
        assert result[0] == pytest.approx(0.0)

    def test_ncc_identical_is_1(self):
        p = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = batch_compare([(p, p)], method="ncc")
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_mixed_ncc_ssd(self):
        p = _patch()
        ncc_result = batch_compare([(p, p)], method="ncc")
        ssd_result = batch_compare([(p, p)], method="ssd")
        assert isinstance(ncc_result[0], float)
        assert isinstance(ssd_result[0], float)
