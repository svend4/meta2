"""Тесты для puzzle_reconstruction/utils/patch_utils.py."""
import pytest
import numpy as np

from puzzle_reconstruction.utils.patch_utils import (
    PatchConfig,
    extract_patch,
    extract_patches,
    normalize_patch,
    compare_patches,
    patch_ssd,
    patch_ncc,
    patch_mse,
    batch_compare,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128):
    return np.full((h, w), fill, dtype=np.uint8)


def make_bgr(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_patch(h=16, w=16, fill=100):
    return np.full((h, w), fill, dtype=np.uint8)


# ─── PatchConfig ──────────────────────────────────────────────────────────────

class TestPatchConfig:
    def test_defaults(self):
        cfg = PatchConfig()
        assert cfg.patch_h == 32
        assert cfg.patch_w == 32
        assert cfg.pad_value == 0
        assert cfg.normalize is False
        assert cfg.norm_mode == "minmax"

    def test_patch_h_zero_raises(self):
        with pytest.raises(ValueError, match="patch_h"):
            PatchConfig(patch_h=0)

    def test_patch_h_negative_raises(self):
        with pytest.raises(ValueError, match="patch_h"):
            PatchConfig(patch_h=-1)

    def test_patch_w_zero_raises(self):
        with pytest.raises(ValueError, match="patch_w"):
            PatchConfig(patch_w=0)

    def test_patch_w_negative_raises(self):
        with pytest.raises(ValueError, match="patch_w"):
            PatchConfig(patch_w=-1)

    def test_pad_value_256_raises(self):
        with pytest.raises(ValueError, match="pad_value"):
            PatchConfig(pad_value=256)

    def test_pad_value_negative_raises(self):
        with pytest.raises(ValueError, match="pad_value"):
            PatchConfig(pad_value=-1)

    def test_invalid_norm_mode_raises(self):
        with pytest.raises(ValueError, match="norm_mode"):
            PatchConfig(norm_mode="l2")

    def test_valid_norm_modes(self):
        for m in ("minmax", "zscore"):
            cfg = PatchConfig(norm_mode=m)
            assert cfg.norm_mode == m

    def test_pad_value_boundary(self):
        cfg = PatchConfig(pad_value=255)
        assert cfg.pad_value == 255

    def test_patch_h_1_valid(self):
        cfg = PatchConfig(patch_h=1)
        assert cfg.patch_h == 1


# ─── extract_patch ────────────────────────────────────────────────────────────

class TestExtractPatch:
    def test_returns_ndarray(self):
        img = make_gray()
        result = extract_patch(img, 32, 32)
        assert isinstance(result, np.ndarray)

    def test_shape_gray(self):
        img = make_gray(64, 64)
        cfg = PatchConfig(patch_h=16, patch_w=16)
        result = extract_patch(img, 32, 32, cfg=cfg)
        assert result.shape == (16, 16)

    def test_shape_color(self):
        img = make_bgr(64, 64)
        cfg = PatchConfig(patch_h=16, patch_w=16)
        result = extract_patch(img, 32, 32, cfg=cfg)
        assert result.shape == (16, 16, 3)

    def test_center_within_image_no_pad(self):
        img = make_gray(64, 64, fill=200)
        cfg = PatchConfig(patch_h=8, patch_w=8, pad_value=0)
        result = extract_patch(img, 32, 32, cfg=cfg)
        np.testing.assert_array_equal(result, 200)

    def test_center_at_corner_uses_pad(self):
        img = make_gray(64, 64, fill=100)
        cfg = PatchConfig(patch_h=16, patch_w=16, pad_value=255)
        result = extract_patch(img, 0, 0, cfg=cfg)
        assert result.shape == (16, 16)
        # Top-left quarter is padding (255)
        assert result[0, 0] == 255

    def test_1d_image_raises(self):
        img = np.zeros(64, dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_patch(img, 0, 0)

    def test_4d_image_raises(self):
        img = np.zeros((4, 4, 3, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            extract_patch(img, 0, 0)

    def test_none_cfg_uses_defaults(self):
        img = make_gray()
        result = extract_patch(img, 32, 32, cfg=None)
        assert result.shape == (32, 32)

    def test_normalize_true_float_output(self):
        img = make_gray(64, 64, fill=128)
        cfg = PatchConfig(patch_h=8, patch_w=8, normalize=True, norm_mode="minmax")
        result = extract_patch(img, 32, 32, cfg=cfg)
        assert result.dtype == np.float32

    def test_center_outside_image(self):
        img = make_gray(32, 32)
        cfg = PatchConfig(patch_h=16, patch_w=16, pad_value=77)
        result = extract_patch(img, -10, -10, cfg=cfg)
        assert result.shape == (16, 16)

    def test_preserves_image_values(self):
        img = np.arange(64, dtype=np.uint8).reshape(8, 8)
        cfg = PatchConfig(patch_h=4, patch_w=4)
        result = extract_patch(img, 4, 4, cfg=cfg)
        assert result.shape == (4, 4)


# ─── extract_patches ──────────────────────────────────────────────────────────

class TestExtractPatches:
    def test_empty_centers_returns_empty(self):
        result = extract_patches(make_gray(), [])
        assert result == []

    def test_length_matches_centers(self):
        img = make_gray()
        centers = [(10, 10), (20, 20), (30, 30)]
        result = extract_patches(img, centers)
        assert len(result) == 3

    def test_all_ndarrays(self):
        img = make_gray()
        result = extract_patches(img, [(10, 10)])
        assert isinstance(result[0], np.ndarray)

    def test_correct_shape(self):
        img = make_gray(64, 64)
        cfg = PatchConfig(patch_h=8, patch_w=8)
        result = extract_patches(img, [(16, 16)], cfg=cfg)
        assert result[0].shape == (8, 8)


# ─── normalize_patch ──────────────────────────────────────────────────────────

class TestNormalizePatch:
    def test_minmax_returns_float32(self):
        p = make_patch()
        result = normalize_patch(p, mode="minmax")
        assert result.dtype == np.float32

    def test_minmax_in_0_1(self):
        p = np.array([[0, 100, 200]], dtype=np.uint8)
        result = normalize_patch(p, mode="minmax")
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0

    def test_minmax_uniform_returns_zeros(self):
        p = make_patch(fill=128)
        result = normalize_patch(p, mode="minmax")
        np.testing.assert_array_equal(result, 0.0)

    def test_zscore_returns_float32(self):
        p = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = normalize_patch(p, mode="zscore")
        assert result.dtype == np.float32

    def test_zscore_mean_approx_zero(self):
        p = np.arange(16, dtype=np.float64).reshape(4, 4)
        result = normalize_patch(p, mode="zscore")
        assert float(result.mean()) == pytest.approx(0.0, abs=1e-5)

    def test_zscore_uniform_returns_zeros(self):
        p = make_patch(fill=50)
        result = normalize_patch(p, mode="zscore")
        np.testing.assert_array_equal(result, 0.0)

    def test_invalid_mode_raises(self):
        p = make_patch()
        with pytest.raises(ValueError, match="mode"):
            normalize_patch(p, mode="l2")

    def test_same_shape(self):
        p = make_patch(h=8, w=12)
        result = normalize_patch(p, mode="minmax")
        assert result.shape == (8, 12)


# ─── patch_ssd ────────────────────────────────────────────────────────────────

class TestPatchSsd:
    def test_identical_returns_zero(self):
        p = make_patch()
        assert patch_ssd(p, p) == pytest.approx(0.0)

    def test_nonneg(self):
        a = make_patch(fill=100)
        b = make_patch(fill=200)
        assert patch_ssd(a, b) >= 0.0

    def test_shape_mismatch_raises(self):
        a = make_patch(h=8, w=8)
        b = make_patch(h=8, w=16)
        with pytest.raises(ValueError):
            patch_ssd(a, b)

    def test_returns_float(self):
        a = make_patch()
        b = make_patch(fill=200)
        assert isinstance(patch_ssd(a, b), float)

    def test_known_value(self):
        a = np.array([[0, 0]], dtype=np.float32)
        b = np.array([[1, 2]], dtype=np.float32)
        # SSD = 1^2 + 2^2 = 5
        assert patch_ssd(a, b) == pytest.approx(5.0)


# ─── patch_ncc ────────────────────────────────────────────────────────────────

class TestPatchNcc:
    def test_identical_returns_1(self):
        p = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert patch_ncc(p, p) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_returns_neg1(self):
        a = np.array([[1.0, -1.0]], dtype=np.float32)
        b = np.array([[-1.0, 1.0]], dtype=np.float32)
        assert patch_ncc(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_uniform_returns_zero(self):
        a = make_patch(fill=100)
        b = make_patch(fill=200)
        assert patch_ncc(a, b) == pytest.approx(0.0)

    def test_in_neg1_1(self):
        a = np.random.default_rng(0).integers(0, 255, (8, 8), dtype=np.uint8)
        b = np.random.default_rng(1).integers(0, 255, (8, 8), dtype=np.uint8)
        result = patch_ncc(a, b)
        assert -1.0 <= result <= 1.0

    def test_shape_mismatch_raises(self):
        a = make_patch(h=8, w=8)
        b = make_patch(h=4, w=8)
        with pytest.raises(ValueError):
            patch_ncc(a, b)

    def test_returns_float(self):
        a = make_patch()
        assert isinstance(patch_ncc(a, a), float)


# ─── patch_mse ────────────────────────────────────────────────────────────────

class TestPatchMse:
    def test_identical_returns_zero(self):
        p = make_patch()
        assert patch_mse(p, p) == pytest.approx(0.0)

    def test_nonneg(self):
        a = make_patch(fill=0)
        b = make_patch(fill=255)
        assert patch_mse(a, b) >= 0.0

    def test_known_value(self):
        a = np.array([[0.0, 2.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0]], dtype=np.float32)
        # MSE = (0 + 4) / 2 = 2
        assert patch_mse(a, b) == pytest.approx(2.0)

    def test_shape_mismatch_raises(self):
        a = make_patch(h=8, w=8)
        b = make_patch(h=16, w=8)
        with pytest.raises(ValueError):
            patch_mse(a, b)

    def test_returns_float(self):
        assert isinstance(patch_mse(make_patch(), make_patch()), float)


# ─── compare_patches ──────────────────────────────────────────────────────────

class TestComparePatches:
    def test_ncc_identical(self):
        p = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert compare_patches(p, p, method="ncc") == pytest.approx(1.0, abs=1e-5)

    def test_ssd_identical(self):
        p = make_patch()
        assert compare_patches(p, p, method="ssd") == pytest.approx(0.0)

    def test_mse_identical(self):
        p = make_patch()
        assert compare_patches(p, p, method="mse") == pytest.approx(0.0)

    def test_unknown_method_raises(self):
        p = make_patch()
        with pytest.raises(ValueError, match="method"):
            compare_patches(p, p, method="ssim")

    def test_shape_mismatch_raises(self):
        a = make_patch(h=8)
        b = make_patch(h=16)
        with pytest.raises(ValueError):
            compare_patches(a, b)

    def test_returns_float(self):
        p = make_patch()
        result = compare_patches(p, p, method="ncc")
        assert isinstance(result, float)


# ─── batch_compare ────────────────────────────────────────────────────────────

class TestBatchCompare:
    def test_empty_returns_empty(self):
        assert batch_compare([]) == []

    def test_length_matches(self):
        p = make_patch()
        pairs = [(p, p), (p, p), (p, p)]
        result = batch_compare(pairs)
        assert len(result) == 3

    def test_all_floats(self):
        p = make_patch()
        result = batch_compare([(p, p)])
        assert isinstance(result[0], float)

    def test_ncc_method(self):
        p = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = batch_compare([(p, p)], method="ncc")
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_ssd_method(self):
        p = make_patch()
        result = batch_compare([(p, p)], method="ssd")
        assert result[0] == pytest.approx(0.0)

    def test_mse_method(self):
        p = make_patch()
        result = batch_compare([(p, p)], method="mse")
        assert result[0] == pytest.approx(0.0)

    def test_invalid_method_raises(self):
        p = make_patch()
        with pytest.raises(ValueError):
            batch_compare([(p, p)], method="pearson")
