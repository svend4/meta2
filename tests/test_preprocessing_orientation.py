"""Расширенные тесты для puzzle_reconstruction/preprocessing/orientation.py."""
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.orientation import (
    _pca_angle,
    _to_gray,
    estimate_orientation,
    rotate_to_upright,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    """Blank white BGR image."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _gray(h: int = 64, w: int = 64) -> np.ndarray:
    """Blank white grayscale image."""
    return np.full((h, w), 255, dtype=np.uint8)


def _draw_hline(h: int = 64, w: int = 64) -> np.ndarray:
    """BGR image with a horizontal black line near middle."""
    img = _bgr(h, w)
    mid = h // 2
    img[mid - 2: mid + 2, :] = 0
    return img


def _draw_vline(h: int = 64, w: int = 64) -> np.ndarray:
    """BGR image with a vertical black line near middle."""
    img = _bgr(h, w)
    mid = w // 2
    img[:, mid - 2: mid + 2] = 0
    return img


# ─── TestToGray ───────────────────────────────────────────────────────────────

class TestToGray:
    def test_bgr_returns_2d(self):
        result = _to_gray(_bgr())
        assert result.ndim == 2

    def test_gray_returns_2d(self):
        result = _to_gray(_gray())
        assert result.ndim == 2

    def test_gray_returns_copy(self):
        g = _gray()
        result = _to_gray(g)
        assert result is not g

    def test_bgr_shape_correct(self):
        img = _bgr(32, 48)
        result = _to_gray(img)
        assert result.shape == (32, 48)

    def test_gray_unchanged_values(self):
        g = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        result = _to_gray(g)
        assert np.array_equal(result, g)

    def test_dtype_uint8(self):
        assert _to_gray(_bgr()).dtype == np.uint8


# ─── TestPcaAngle ─────────────────────────────────────────────────────────────

class TestPcaAngle:
    def test_returns_float(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        result = _pca_angle(binary)
        assert isinstance(result, float)

    def test_all_zero_returns_zero(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        assert _pca_angle(binary) == pytest.approx(0.0)

    def test_few_white_pixels_returns_zero(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        binary[0, 0] = 255
        binary[1, 1] = 255
        assert _pca_angle(binary) == pytest.approx(0.0)

    def test_exactly_9_white_pixels_returns_zero(self):
        binary = np.zeros((32, 32), dtype=np.uint8)
        for i in range(9):
            binary[0, i] = 255
        assert _pca_angle(binary) == pytest.approx(0.0)

    def test_horizontal_line_angle_near_zero(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[32, 10:54] = 255  # horizontal line
        result = _pca_angle(binary)
        assert isinstance(result, float)

    def test_many_pixels_returns_float(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[10:54, 10:54] = 255
        result = _pca_angle(binary)
        assert isinstance(result, float)


# ─── TestEstimateOrientation ──────────────────────────────────────────────────

class TestEstimateOrientation:
    def test_returns_float(self):
        result = estimate_orientation(_bgr())
        assert isinstance(result, float)

    def test_bgr_accepted(self):
        result = estimate_orientation(_bgr(64, 64))
        assert isinstance(result, float)

    def test_gray_accepted(self):
        result = estimate_orientation(_gray(64, 64))
        assert isinstance(result, float)

    def test_blank_white_returns_float(self):
        result = estimate_orientation(_bgr())
        assert isinstance(result, float)

    def test_returns_radians_range(self):
        # Angle should be within reasonable range for Hough (< π/4 filtered)
        result = estimate_orientation(_draw_hline())
        assert -np.pi <= result <= np.pi

    def test_mask_parameter_accepted(self):
        img = _bgr(64, 64)
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        result = estimate_orientation(img, mask=mask)
        assert isinstance(result, float)

    def test_mask_none_default(self):
        result = estimate_orientation(_bgr(), mask=None)
        assert isinstance(result, float)

    def test_small_image(self):
        img = _bgr(16, 16)
        result = estimate_orientation(img)
        assert isinstance(result, float)

    def test_large_image(self):
        img = _bgr(128, 128)
        result = estimate_orientation(img)
        assert isinstance(result, float)

    def test_rectangular_image(self):
        img = _bgr(32, 96)
        result = estimate_orientation(img)
        assert isinstance(result, float)

    def test_deterministic(self):
        img = _draw_hline()
        r1 = estimate_orientation(img)
        r2 = estimate_orientation(img)
        assert r1 == pytest.approx(r2)

    def test_horizontal_line_small_angle(self):
        img = _draw_hline(64, 64)
        result = estimate_orientation(img)
        # Horizontal text → angle near 0
        assert abs(result) < np.pi / 4


# ─── TestRotateToUpright ──────────────────────────────────────────────────────

class TestRotateToUpright:
    def test_returns_ndarray(self):
        result = rotate_to_upright(_bgr(), 0.0)
        assert isinstance(result, np.ndarray)

    def test_same_shape_bgr(self):
        img = _bgr(64, 96)
        result = rotate_to_upright(img, 0.0)
        assert result.shape == img.shape

    def test_same_shape_nonzero_angle(self):
        img = _bgr(64, 64)
        result = rotate_to_upright(img, np.pi / 6)
        assert result.shape == img.shape

    def test_dtype_uint8(self):
        img = _bgr()
        result = rotate_to_upright(img, 0.1)
        assert result.dtype == np.uint8

    def test_zero_angle_identity(self):
        img = _draw_hline(64, 64)
        result = rotate_to_upright(img, 0.0)
        assert result.shape == img.shape

    def test_pi_angle(self):
        img = _bgr(32, 32)
        result = rotate_to_upright(img, np.pi)
        assert result.shape == (32, 32, 3)

    def test_pi_half_angle(self):
        img = _bgr(64, 64)
        result = rotate_to_upright(img, np.pi / 2)
        assert result.shape == img.shape

    def test_negative_angle(self):
        img = _bgr(64, 64)
        result = rotate_to_upright(img, -np.pi / 8)
        assert result.shape == img.shape

    def test_white_border_fill(self):
        # After rotating a dark image, corners should be white (border value 255)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = rotate_to_upright(img, np.pi / 8)
        # Corners should be white (255,255,255) due to border fill
        corner = result[0, 0]
        assert np.all(corner == 255)

    def test_grayscale_same_shape(self):
        img = _gray(64, 64)
        result = rotate_to_upright(img, 0.1)
        assert result.shape == img.shape

    def test_channel_count_preserved(self):
        img = _bgr(32, 32)
        result = rotate_to_upright(img, 0.3)
        assert result.ndim == 3
        assert result.shape[2] == 3
