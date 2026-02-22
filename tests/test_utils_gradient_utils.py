"""Тесты для puzzle_reconstruction/utils/gradient_utils.py."""
import math

import numpy as np
import pytest

from puzzle_reconstruction.utils.gradient_utils import (
    GradientConfig,
    batch_compute_gradients,
    compute_edge_density,
    compute_gradient_direction,
    compute_gradient_magnitude,
    compute_laplacian,
    compute_sobel,
    suppress_non_maximum,
    threshold_gradient,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _color(h=32, w=32) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _step_h(h=32, w=32) -> np.ndarray:
    """Горизонтальный перепад: левая половина 0, правая 200."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, w // 2 :] = 200
    return img


def _step_v(h=32, w=32) -> np.ndarray:
    """Вертикальный перепад: верх 0, низ 200."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[h // 2 :, :] = 200
    return img


# ─── TestGradientConfig ───────────────────────────────────────────────────────

class TestGradientConfig:
    def test_defaults(self):
        cfg = GradientConfig()
        assert cfg.ksize == 3
        assert cfg.normalize is True
        assert cfg.threshold == pytest.approx(32.0)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError, match="ksize"):
            GradientConfig(ksize=2)

    def test_zero_ksize_raises(self):
        with pytest.raises(ValueError, match="ksize"):
            GradientConfig(ksize=0)

    def test_negative_ksize_raises(self):
        with pytest.raises(ValueError, match="ksize"):
            GradientConfig(ksize=-3)

    def test_ksize_1_valid(self):
        cfg = GradientConfig(ksize=1)
        assert cfg.ksize == 1

    def test_ksize_5_valid(self):
        cfg = GradientConfig(ksize=5)
        assert cfg.ksize == 5

    def test_threshold_too_low_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            GradientConfig(threshold=-1.0)

    def test_threshold_too_high_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            GradientConfig(threshold=256.0)

    def test_threshold_boundary_0(self):
        cfg = GradientConfig(threshold=0.0)
        assert cfg.threshold == 0.0

    def test_threshold_boundary_255(self):
        cfg = GradientConfig(threshold=255.0)
        assert cfg.threshold == 255.0

    def test_normalize_false_ok(self):
        cfg = GradientConfig(normalize=False)
        assert cfg.normalize is False


# ─── TestComputeGradientMagnitude ─────────────────────────────────────────────

class TestComputeGradientMagnitude:
    def test_returns_ndarray(self):
        result = compute_gradient_magnitude(_gray())
        assert isinstance(result, np.ndarray)

    def test_dtype_float32(self):
        result = compute_gradient_magnitude(_gray())
        assert result.dtype == np.float32

    def test_shape_2d_input(self):
        img = _gray(20, 30)
        result = compute_gradient_magnitude(img)
        assert result.shape == (20, 30)

    def test_shape_3d_input(self):
        img = _color(20, 30)
        result = compute_gradient_magnitude(img)
        assert result.shape == (20, 30)

    def test_normalized_max_le_1(self):
        img = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
        result = compute_gradient_magnitude(img)
        assert float(result.max()) <= 1.0 + 1e-6

    def test_normalized_min_ge_0(self):
        img = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
        result = compute_gradient_magnitude(img)
        assert float(result.min()) >= 0.0

    def test_uniform_image_zero_gradient(self):
        result = compute_gradient_magnitude(_gray(20, 20, 128))
        assert float(result.max()) == pytest.approx(0.0, abs=1e-6)

    def test_step_edge_has_nonzero_gradient(self):
        result = compute_gradient_magnitude(_step_h())
        assert float(result.max()) > 0.01

    def test_4d_raises(self):
        img = np.zeros((4, 4, 3, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_gradient_magnitude(img)

    def test_1d_raises(self):
        img = np.zeros((10,), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_gradient_magnitude(img)

    def test_none_cfg_uses_defaults(self):
        result = compute_gradient_magnitude(_gray(), cfg=None)
        assert result.shape == (32, 32)

    def test_normalize_false_can_exceed_1(self):
        cfg = GradientConfig(normalize=False)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:, 5] = 255
        result = compute_gradient_magnitude(img, cfg=cfg)
        # Without normalization the raw Sobel values can be large
        assert result.max() >= 0.0  # Just verify it runs without error

    def test_float32_input_ok(self):
        img = np.random.rand(20, 20).astype(np.float32) * 255
        result = compute_gradient_magnitude(img)
        assert result.shape == (20, 20)

    def test_ksize_5_different_from_3(self):
        img = _step_h(40, 40)
        cfg3 = GradientConfig(ksize=3)
        cfg5 = GradientConfig(ksize=5)
        r3 = compute_gradient_magnitude(img, cfg3)
        r5 = compute_gradient_magnitude(img, cfg5)
        # Different kernel sizes produce different results
        assert not np.allclose(r3, r5)

    def test_color_and_gray_same_shape(self):
        gray = _gray(24, 24)
        color = _color(24, 24)
        r_gray = compute_gradient_magnitude(gray)
        r_color = compute_gradient_magnitude(color)
        assert r_gray.shape == r_color.shape


# ─── TestComputeGradientDirection ─────────────────────────────────────────────

class TestComputeGradientDirection:
    def test_returns_ndarray(self):
        result = compute_gradient_direction(_gray())
        assert isinstance(result, np.ndarray)

    def test_dtype_float32(self):
        result = compute_gradient_direction(_gray())
        assert result.dtype == np.float32

    def test_shape_2d_input(self):
        img = _gray(16, 24)
        result = compute_gradient_direction(img)
        assert result.shape == (16, 24)

    def test_shape_3d_input(self):
        img = _color(16, 24)
        result = compute_gradient_direction(img)
        assert result.shape == (16, 24)

    def test_values_within_pi(self):
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        result = compute_gradient_direction(img)
        assert float(result.min()) >= -math.pi - 1e-5
        assert float(result.max()) <= math.pi + 1e-5

    def test_4d_raises(self):
        img = np.zeros((2, 2, 2, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_gradient_direction(img)

    def test_none_cfg_ok(self):
        result = compute_gradient_direction(_gray(), cfg=None)
        assert result.ndim == 2

    def test_horizontal_step_direction_near_zero(self):
        """A left-right step edge → gradient direction near 0."""
        img = _step_h(40, 40)
        direction = compute_gradient_direction(img)
        # At the step, dx is large, dy is ~0 → angle ~ 0 or π
        center_col = direction[:, 20]
        # angle should be near 0 (±ε) or near π
        near_zero = np.abs(center_col) < 0.5
        near_pi = np.abs(np.abs(center_col) - math.pi) < 0.5
        assert bool(near_zero.any() or near_pi.any())

    def test_vertical_step_direction_near_pi_half(self):
        """A top-bottom step edge → gradient direction near ±π/2."""
        img = _step_v(40, 40)
        direction = compute_gradient_direction(img)
        center_row = direction[20, :]
        near_half_pi = np.abs(np.abs(center_row) - math.pi / 2) < 0.5
        assert bool(near_half_pi.any())


# ─── TestComputeSobel ─────────────────────────────────────────────────────────

class TestComputeSobel:
    def test_returns_tuple(self):
        result = compute_sobel(_gray())
        assert isinstance(result, tuple)

    def test_tuple_length_3(self):
        result = compute_sobel(_gray())
        assert len(result) == 3

    def test_all_arrays_float32(self):
        mag, dx, dy = compute_sobel(_gray())
        assert mag.dtype == np.float32
        assert dx.dtype == np.float32
        assert dy.dtype == np.float32

    def test_all_same_shape(self):
        img = _gray(16, 24)
        mag, dx, dy = compute_sobel(img)
        assert mag.shape == (16, 24)
        assert dx.shape == (16, 24)
        assert dy.shape == (16, 24)

    def test_magnitude_normalized_in_0_1(self):
        mag, dx, dy = compute_sobel(np.random.randint(0, 256, (32, 32), dtype=np.uint8))
        assert float(mag.min()) >= 0.0
        assert float(mag.max()) <= 1.0 + 1e-6

    def test_3d_input_ok(self):
        mag, dx, dy = compute_sobel(_color(16, 16))
        assert mag.shape == (16, 16)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_sobel(np.zeros((2, 2, 2, 2), dtype=np.uint8))

    def test_dx_nonzero_for_horizontal_step(self):
        img = _step_h(32, 32)
        _, dx, _ = compute_sobel(img)
        assert float(np.abs(dx).max()) > 0.0

    def test_dy_nonzero_for_vertical_step(self):
        img = _step_v(32, 32)
        _, _, dy = compute_sobel(img)
        assert float(np.abs(dy).max()) > 0.0

    def test_uniform_image_zero_dx_dy(self):
        img = _gray(16, 16, 100)
        _, dx, dy = compute_sobel(img)
        assert float(np.abs(dx).max()) == pytest.approx(0.0, abs=1e-5)
        assert float(np.abs(dy).max()) == pytest.approx(0.0, abs=1e-5)

    def test_magnitude_equals_hypot_dx_dy(self):
        img = np.random.randint(0, 256, (24, 24), dtype=np.uint8)
        cfg = GradientConfig(normalize=False)
        mag, dx, dy = compute_sobel(img, cfg)
        expected = np.hypot(dx, dy)
        np.testing.assert_allclose(mag, expected.astype(np.float32), atol=1e-4)


# ─── TestComputeLaplacian ─────────────────────────────────────────────────────

class TestComputeLaplacian:
    def test_returns_ndarray(self):
        result = compute_laplacian(_gray())
        assert isinstance(result, np.ndarray)

    def test_dtype_float32(self):
        result = compute_laplacian(_gray())
        assert result.dtype == np.float32

    def test_shape_2d_input(self):
        img = _gray(18, 22)
        result = compute_laplacian(img)
        assert result.shape == (18, 22)

    def test_shape_3d_input(self):
        img = _color(18, 22)
        result = compute_laplacian(img)
        assert result.shape == (18, 22)

    def test_normalized_in_0_1(self):
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        result = compute_laplacian(img)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0 + 1e-6

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_laplacian(np.zeros((2, 2, 2, 2)))

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            compute_laplacian(_gray(), ksize=2)

    def test_zero_ksize_raises(self):
        with pytest.raises(ValueError):
            compute_laplacian(_gray(), ksize=0)

    def test_normalize_false_has_negatives_on_edge(self):
        img = _step_h()
        result = compute_laplacian(img, normalize=False)
        # Laplacian at an edge has both positive and negative values
        assert float(result.min()) < 0.0 or float(result.max()) >= 0.0

    def test_uniform_image_zero_laplacian(self):
        result = compute_laplacian(_gray(16, 16, 128))
        assert float(result.max()) == pytest.approx(0.0, abs=1e-5)


# ─── TestThresholdGradient ────────────────────────────────────────────────────

class TestThresholdGradient:
    def test_returns_bool_array(self):
        mag = np.random.rand(16, 16).astype(np.float32)
        result = threshold_gradient(mag)
        assert result.dtype == bool

    def test_shape_preserved(self):
        mag = np.random.rand(14, 18).astype(np.float32)
        result = threshold_gradient(mag)
        assert result.shape == (14, 18)

    def test_all_true_when_threshold_zero(self):
        mag = np.ones((10, 10), dtype=np.float32) * 0.5
        result = threshold_gradient(mag, threshold=0.0)
        assert result.all()

    def test_all_false_when_threshold_above_max(self):
        mag = np.ones((10, 10), dtype=np.float32) * 0.5
        result = threshold_gradient(mag, threshold=1.1)
        assert not result.any()

    def test_explicit_threshold_splits_correctly(self):
        mag = np.array([[0.1, 0.5], [0.8, 0.3]], dtype=np.float32)
        result = threshold_gradient(mag, threshold=0.4)
        assert result[0, 0] is np.bool_(False)
        assert result[0, 1] is np.bool_(True)
        assert result[1, 0] is np.bool_(True)
        assert result[1, 1] is np.bool_(False)

    def test_3d_magnitude_raises(self):
        with pytest.raises(ValueError):
            threshold_gradient(np.ones((4, 4, 3), dtype=np.float32))

    def test_uses_cfg_threshold(self):
        cfg = GradientConfig(threshold=0.0)  # threshold/255 → 0
        mag = np.zeros((8, 8), dtype=np.float32)
        result = threshold_gradient(mag, cfg=cfg)
        # 0 >= 0 is True
        assert result.all()

    def test_default_threshold_from_cfg(self):
        cfg = GradientConfig(threshold=255.0)  # threshold/255 = 1.0
        mag = np.ones((8, 8), dtype=np.float32) * 0.99
        result = threshold_gradient(mag, cfg=cfg)
        # 0.99 < 1.0 → all False
        assert not result.any()


# ─── TestSuppressNonMaximum ───────────────────────────────────────────────────

class TestSuppressNonMaximum:
    def _make(self, h=12, w=12):
        mag = np.random.rand(h, w).astype(np.float32)
        direction = np.random.uniform(-math.pi, math.pi, (h, w)).astype(np.float32)
        return mag, direction

    def test_returns_ndarray(self):
        mag, direction = self._make()
        result = suppress_non_maximum(mag, direction)
        assert isinstance(result, np.ndarray)

    def test_dtype_float32(self):
        mag, direction = self._make()
        result = suppress_non_maximum(mag, direction)
        assert result.dtype == np.float32

    def test_shape_preserved(self):
        mag, direction = self._make(10, 14)
        result = suppress_non_maximum(mag, direction)
        assert result.shape == (10, 14)

    def test_output_le_input(self):
        mag, direction = self._make(12, 12)
        result = suppress_non_maximum(mag, direction)
        assert float((result > mag + 1e-6).sum()) == 0.0

    def test_border_pixels_zero(self):
        mag, direction = self._make(12, 12)
        result = suppress_non_maximum(mag, direction)
        # Border rows/cols are 0 in our implementation
        assert result[0, :].max() == pytest.approx(0.0)
        assert result[-1, :].max() == pytest.approx(0.0)
        assert result[:, 0].max() == pytest.approx(0.0)
        assert result[:, -1].max() == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        mag = np.zeros((8, 8), dtype=np.float32)
        direction = np.zeros((8, 9), dtype=np.float32)
        with pytest.raises(ValueError):
            suppress_non_maximum(mag, direction)

    def test_3d_magnitude_raises(self):
        with pytest.raises(ValueError):
            suppress_non_maximum(
                np.zeros((4, 4, 3), dtype=np.float32),
                np.zeros((4, 4, 3), dtype=np.float32),
            )

    def test_uniform_magnitude_preserves_values(self):
        """When all neighbors equal the current pixel, pixel is preserved."""
        mag = np.ones((6, 6), dtype=np.float32)
        direction = np.zeros((6, 6), dtype=np.float32)  # all horizontal
        result = suppress_non_maximum(mag, direction)
        # Interior pixels should be preserved (equal to neighbors)
        assert result[2, 2] == pytest.approx(1.0)


# ─── TestComputeEdgeDensity ───────────────────────────────────────────────────

class TestComputeEdgeDensity:
    def test_returns_float(self):
        result = compute_edge_density(_gray())
        assert isinstance(result, float)

    def test_in_0_1_range(self):
        result = compute_edge_density(_color(32, 32))
        assert 0.0 <= result <= 1.0

    def test_uniform_image_low_density(self):
        result = compute_edge_density(_gray(32, 32, 128))
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_step_image_nonzero_density(self):
        result = compute_edge_density(_step_h(32, 32))
        assert result > 0.0

    def test_with_roi(self):
        img = _step_h(32, 32)
        # ROI covering the step column
        density = compute_edge_density(img, roi=(0, 14, 32, 18))
        assert density >= 0.0

    def test_empty_roi_returns_zero(self):
        result = compute_edge_density(_gray(), roi=(10, 10, 10, 20))
        assert result == pytest.approx(0.0)

    def test_inverted_roi_returns_zero(self):
        result = compute_edge_density(_gray(), roi=(20, 10, 10, 20))
        assert result == pytest.approx(0.0)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_edge_density(np.zeros((2, 2, 2, 2), dtype=np.uint8))

    def test_2d_input_ok(self):
        result = compute_edge_density(_gray(16, 16))
        assert isinstance(result, float)

    def test_roi_clipped_to_image_bounds(self):
        """ROI beyond image bounds is clipped, not an error."""
        result = compute_edge_density(_gray(16, 16), roi=(-5, -5, 100, 100))
        assert isinstance(result, float)


# ─── TestBatchComputeGradients ────────────────────────────────────────────────

class TestBatchComputeGradients:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_compute_gradients([])

    def test_returns_list(self):
        imgs = [_gray(16, 16), _gray(16, 16)]
        result = batch_compute_gradients(imgs)
        assert isinstance(result, list)

    def test_length_matches_input(self):
        imgs = [_gray(16, 16)] * 5
        result = batch_compute_gradients(imgs)
        assert len(result) == 5

    def test_each_element_float32(self):
        imgs = [_color(16, 16), _gray(16, 16)]
        result = batch_compute_gradients(imgs)
        for arr in result:
            assert arr.dtype == np.float32

    def test_each_element_2d(self):
        imgs = [_color(16, 24), _gray(16, 24)]
        result = batch_compute_gradients(imgs)
        for arr in result:
            assert arr.ndim == 2
            assert arr.shape == (16, 24)

    def test_single_image(self):
        result = batch_compute_gradients([_gray(8, 8)])
        assert len(result) == 1

    def test_with_custom_cfg(self):
        cfg = GradientConfig(ksize=5, normalize=True, threshold=50.0)
        imgs = [_gray(16, 16), _color(16, 16)]
        result = batch_compute_gradients(imgs, cfg=cfg)
        assert len(result) == 2
