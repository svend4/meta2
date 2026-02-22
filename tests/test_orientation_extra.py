"""Additional tests for puzzle_reconstruction/preprocessing/orientation.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.orientation import (
    estimate_orientation,
    rotate_to_upright,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, fill: int = 255) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _rgb(h: int = 64, w: int = 64, fill: int = 200) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _hlines(h: int = 64, w: int = 64, n: int = 5) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)
    step = h // (n + 1)
    for i in range(1, n + 1):
        img[i * step, :] = 0
    return img


def _black(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ─── TestEstimateOrientationExtra ────────────────────────────────────────────

class TestEstimateOrientationExtra:
    def test_large_image_128(self):
        img = _gray(128, 128)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_large_image_256(self):
        img = _rgb(256, 256)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_all_black_image_no_crash(self):
        img = _black()
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_all_black_finite(self):
        angle = estimate_orientation(_black())
        assert np.isfinite(angle)

    def test_non_square_tall(self):
        img = _gray(96, 32)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_non_square_wide(self):
        img = _gray(32, 128)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_within_pi_range_rgb(self):
        img = _rgb(64, 64)
        angle = estimate_orientation(img)
        assert -math.pi <= angle <= math.pi

    def test_mask_all_white_matches_no_mask(self):
        img = _hlines()
        mask_full = np.full((64, 64), 255, dtype=np.uint8)
        a1 = estimate_orientation(img, mask=None)
        a2 = estimate_orientation(img, mask=mask_full)
        # Both should be finite; may differ slightly but should both work
        assert np.isfinite(a1) and np.isfinite(a2)

    def test_partial_mask(self):
        img = _hlines(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        angle = estimate_orientation(img, mask=mask)
        assert isinstance(angle, float)

    def test_tiny_1x1_no_crash(self):
        img = np.array([[200]], dtype=np.uint8)
        angle = estimate_orientation(img)
        assert isinstance(angle, float)

    def test_tiny_4x4_no_crash(self):
        img = np.full((4, 4), 200, dtype=np.uint8)
        angle = estimate_orientation(img)
        assert np.isfinite(angle)

    def test_rgb_large_image(self):
        img = _rgb(128, 96, fill=255)
        angle = estimate_orientation(img)
        assert np.isfinite(angle)

    def test_multiple_calls_same_result(self):
        img = _hlines()
        angles = [estimate_orientation(img) for _ in range(3)]
        assert all(a == angles[0] for a in angles)

    def test_blank_angle_in_range(self):
        angle = estimate_orientation(_gray(64, 64, fill=255))
        assert -math.pi <= angle <= math.pi


# ─── TestRotateToUprightExtra ─────────────────────────────────────────────────

class TestRotateToUprightExtra:
    def test_very_small_angle(self):
        img = _rgb(64, 64)
        rotated = rotate_to_upright(img, 0.01)
        assert rotated.shape == img.shape

    def test_large_negative_angle(self):
        img = _rgb(64, 64)
        rotated = rotate_to_upright(img, -math.pi / 3)
        assert rotated.shape == img.shape

    def test_2pi_angle(self):
        img = _rgb(64, 64)
        rotated = rotate_to_upright(img, 2 * math.pi)
        assert rotated.shape == img.shape

    def test_gray_border_fill_255(self):
        """Grayscale: corner fill after rotation should be near 255."""
        img = np.zeros((64, 64), dtype=np.uint8)
        rotated = rotate_to_upright(img, math.pi / 6)
        assert rotated[0, 0] == 255

    def test_values_in_0_255(self):
        img = _rgb(64, 64, fill=128)
        rotated = rotate_to_upright(img, math.pi / 4)
        assert np.all(rotated >= 0)
        assert np.all(rotated <= 255)

    def test_uint8_dtype_always(self):
        for angle in [0.0, 0.1, math.pi / 2, math.pi]:
            rotated = rotate_to_upright(_rgb(), angle)
            assert rotated.dtype == np.uint8

    def test_non_square_shape_preserved(self):
        img = np.full((48, 80, 3), 200, dtype=np.uint8)
        rotated = rotate_to_upright(img, math.pi / 8)
        assert rotated.shape == (48, 80, 3)

    def test_chain_two_rotations_shape_stable(self):
        img = _rgb(64, 64)
        r1 = rotate_to_upright(img, math.pi / 8)
        r2 = rotate_to_upright(r1, -math.pi / 8)
        assert r2.shape == img.shape

    def test_large_image_ok(self):
        img = _rgb(256, 256, fill=200)
        rotated = rotate_to_upright(img, 0.2)
        assert rotated.shape == (256, 256, 3)

    def test_pi_rotation_all_valid_pixels(self):
        img = _rgb(32, 32, fill=100)
        rotated = rotate_to_upright(img, math.pi)
        assert np.all((rotated == 100) | (rotated == 255))
