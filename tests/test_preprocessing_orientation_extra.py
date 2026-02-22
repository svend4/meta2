"""Additional tests for puzzle_reconstruction/preprocessing/orientation.py — internal helpers."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.orientation import (
    _pca_angle,
    _to_gray,
    estimate_orientation,
    rotate_to_upright,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h: int = 64, w: int = 64, fill: int = 255) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _gray(h: int = 64, w: int = 64, fill: int = 255) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _hline_img(h: int = 64, w: int = 64) -> np.ndarray:
    img = _bgr(h, w)
    mid = h // 2
    img[mid - 1: mid + 1, :] = 0
    return img


# ─── TestToGrayExtra ──────────────────────────────────────────────────────────

class TestToGrayExtra:
    def test_very_small_image(self):
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = _to_gray(img)
        assert result.shape == (1, 1)
        assert result.ndim == 2

    def test_non_square_bgr(self):
        img = _bgr(32, 80)
        result = _to_gray(img)
        assert result.shape == (32, 80)

    def test_white_bgr_produces_white_gray(self):
        img = np.full((8, 8, 3), 255, dtype=np.uint8)
        result = _to_gray(img)
        assert np.all(result == 255)

    def test_black_bgr_produces_black_gray(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        result = _to_gray(img)
        assert np.all(result == 0)

    def test_gray_already_2d_passthrough(self):
        g = np.arange(16, dtype=np.uint8).reshape(4, 4)
        result = _to_gray(g)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, g)

    def test_output_dtype_uint8(self):
        img = _bgr(16, 16)
        result = _to_gray(img)
        assert result.dtype == np.uint8

    def test_large_bgr(self):
        img = _bgr(128, 128)
        result = _to_gray(img)
        assert result.shape == (128, 128)


# ─── TestPcaAngleExtra ────────────────────────────────────────────────────────

class TestPcaAngleExtra:
    def test_output_is_float(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        binary[10:22, 5:27] = 255
        assert isinstance(_pca_angle(binary), float)

    def test_finite_for_block(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[20:44, 20:44] = 255
        assert np.isfinite(_pca_angle(binary))

    def test_all_white_returns_float(self):
        binary = np.full((32, 32), 255, dtype=np.uint8)
        result = _pca_angle(binary)
        assert isinstance(result, float)

    def test_single_row_pixels(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        binary[15, 10:22] = 255
        result = _pca_angle(binary)
        assert isinstance(result, float)

    def test_diagonal_pixels(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        for i in range(20):
            binary[i, i] = 255
        result = _pca_angle(binary)
        assert isinstance(result, float)

    def test_small_input_no_crash(self):
        binary = np.zeros((4, 4), dtype=np.uint8)
        binary[1:3, 1:3] = 255
        result = _pca_angle(binary)
        assert isinstance(result, float)

    def test_large_binary(self):
        binary = np.zeros((128, 128), dtype=np.uint8)
        binary[40:88, 20:108] = 255
        result = _pca_angle(binary)
        assert np.isfinite(result)


# ─── TestEstimateOrientationExtra ────────────────────────────────────────────

class TestEstimateOrientationExtra:
    def test_vertical_line_finite(self):
        img = _bgr(64, 64)
        img[:, 30:34] = 0
        result = estimate_orientation(img)
        assert np.isfinite(result)

    def test_noisy_image_finite(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        result = estimate_orientation(img)
        assert np.isfinite(result)

    def test_solid_black_image(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = estimate_orientation(img)
        assert isinstance(result, float)

    def test_mask_empty_zeros_no_crash(self):
        img = _hline_img()
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = estimate_orientation(img, mask=mask)
        assert isinstance(result, float)

    def test_large_128x128(self):
        img = _bgr(128, 128)
        result = estimate_orientation(img)
        assert isinstance(result, float)

    def test_reproducible_across_calls(self):
        img = _hline_img()
        a = estimate_orientation(img)
        b = estimate_orientation(img)
        assert a == pytest.approx(b)


# ─── TestRotateToUprightExtra ─────────────────────────────────────────────────

class TestRotateToUprightExtra:
    def test_tiny_angle_no_distortion(self):
        img = _bgr(64, 64, fill=200)
        result = rotate_to_upright(img, 0.001)
        assert result.shape == img.shape

    def test_three_quarter_pi(self):
        img = _bgr(64, 64)
        result = rotate_to_upright(img, 3 * math.pi / 4)
        assert result.shape == img.shape

    def test_negative_pi_over_3(self):
        img = _bgr(64, 64)
        result = rotate_to_upright(img, -math.pi / 3)
        assert result.shape == img.shape

    def test_grayscale_border_255(self):
        """Gray image corner pixels after rotation should be 255 (white fill)."""
        img = np.zeros((64, 64), dtype=np.uint8)
        result = rotate_to_upright(img, math.pi / 4)
        assert result[0, 0] == 255

    def test_channel_count_unchanged(self):
        img = _bgr(48, 64)
        result = rotate_to_upright(img, 0.3)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_dtype_uint8_for_any_angle(self):
        img = _bgr(32, 32)
        for a in [0.0, 0.5, math.pi / 2, math.pi, -0.3]:
            assert rotate_to_upright(img, a).dtype == np.uint8

    def test_gray_shape_preserved_various_angles(self):
        img = _gray(48, 48)
        for a in [0.1, math.pi / 4, math.pi / 2]:
            result = rotate_to_upright(img, a)
            assert result.shape == img.shape

    def test_values_bounded_0_255(self):
        img = _bgr(32, 32, fill=128)
        result = rotate_to_upright(img, 0.4)
        assert result.min() >= 0
        assert result.max() <= 255
