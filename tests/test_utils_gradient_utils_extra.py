"""Extra tests for puzzle_reconstruction/utils/gradient_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.gradient_utils import (
    GradientConfig,
    compute_gradient_magnitude,
    compute_gradient_direction,
    compute_sobel,
    compute_laplacian,
    threshold_gradient,
    suppress_non_maximum,
    compute_edge_density,
    batch_compute_gradients,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _ramp(h=32, w=32) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


def _bgr(h=32, w=32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ─── GradientConfig ───────────────────────────────────────────────────────────

class TestGradientConfigExtra:
    def test_default_ksize(self):
        assert GradientConfig().ksize == 3

    def test_default_normalize(self):
        assert GradientConfig().normalize is True

    def test_default_threshold(self):
        assert GradientConfig().threshold == pytest.approx(32.0)

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=4)

    def test_zero_ksize_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(ksize=0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(threshold=-1.0)

    def test_threshold_gt_255_raises(self):
        with pytest.raises(ValueError):
            GradientConfig(threshold=256.0)

    def test_valid_odd_ksize(self):
        cfg = GradientConfig(ksize=5)
        assert cfg.ksize == 5


# ─── compute_gradient_magnitude ───────────────────────────────────────────────

class TestComputeGradientMagnitudeExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_gradient_magnitude(_gray()), np.ndarray)

    def test_shape_hw(self):
        img = _gray(20, 30)
        out = compute_gradient_magnitude(img)
        assert out.shape == (20, 30)

    def test_dtype_float32(self):
        out = compute_gradient_magnitude(_gray())
        assert out.dtype == np.float32

    def test_uniform_image_near_zero(self):
        out = compute_gradient_magnitude(_gray(val=200))
        assert out.max() < 0.01

    def test_ramp_has_nonzero_gradient(self):
        out = compute_gradient_magnitude(_ramp())
        assert out.max() > 0.0

    def test_normalize_true_range(self):
        cfg = GradientConfig(normalize=True)
        out = compute_gradient_magnitude(_ramp(), cfg)
        assert out.max() <= 1.0 + 1e-6

    def test_bgr_image_ok(self):
        out = compute_gradient_magnitude(_bgr())
        assert out.shape == (32, 32)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_magnitude(np.zeros((2, 4, 4, 3), dtype=np.uint8))

    def test_none_cfg(self):
        out = compute_gradient_magnitude(_gray(), cfg=None)
        assert isinstance(out, np.ndarray)


# ─── compute_gradient_direction ───────────────────────────────────────────────

class TestComputeGradientDirectionExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_gradient_direction(_gray()), np.ndarray)

    def test_shape_hw(self):
        img = _gray(16, 24)
        out = compute_gradient_direction(img)
        assert out.shape == (16, 24)

    def test_dtype_float32(self):
        assert compute_gradient_direction(_gray()).dtype == np.float32

    def test_values_in_pi_range(self):
        out = compute_gradient_direction(_ramp())
        assert np.all(out >= -np.pi - 1e-6)
        assert np.all(out <= np.pi + 1e-6)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_gradient_direction(np.zeros((2, 4, 4, 3), dtype=np.uint8))


# ─── compute_sobel ────────────────────────────────────────────────────────────

class TestComputeSobelExtra:
    def test_returns_tuple_of_3(self):
        result = compute_sobel(_gray())
        assert isinstance(result, tuple) and len(result) == 3

    def test_shapes_equal(self):
        mag, dx, dy = compute_sobel(_gray(16, 24))
        assert mag.shape == dx.shape == dy.shape == (16, 24)

    def test_dtypes_float32(self):
        mag, dx, dy = compute_sobel(_gray())
        assert mag.dtype == np.float32
        assert dx.dtype == np.float32
        assert dy.dtype == np.float32

    def test_uniform_mag_near_zero(self):
        mag, _, _ = compute_sobel(_gray(val=128))
        assert mag.max() < 0.01

    def test_ramp_dx_nonzero(self):
        _, dx, _ = compute_sobel(_ramp())
        assert np.abs(dx).max() > 0.0

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_sobel(np.zeros((2, 4, 4, 3), dtype=np.uint8))


# ─── compute_laplacian ────────────────────────────────────────────────────────

class TestComputeLaplacianExtra:
    def test_returns_ndarray(self):
        assert isinstance(compute_laplacian(_gray()), np.ndarray)

    def test_shape_hw(self):
        out = compute_laplacian(_gray(20, 30))
        assert out.shape == (20, 30)

    def test_dtype_float32(self):
        assert compute_laplacian(_gray()).dtype == np.float32

    def test_uniform_image_near_zero(self):
        out = compute_laplacian(_gray(val=100))
        assert out.max() < 0.01

    def test_normalized_range(self):
        out = compute_laplacian(_ramp(), normalize=True)
        assert out.max() <= 1.0 + 1e-6

    def test_even_ksize_raises(self):
        with pytest.raises(ValueError):
            compute_laplacian(_gray(), ksize=4)

    def test_zero_ksize_raises(self):
        with pytest.raises(ValueError):
            compute_laplacian(_gray(), ksize=0)

    def test_bgr_image_ok(self):
        out = compute_laplacian(_bgr())
        assert out.shape == (32, 32)


# ─── threshold_gradient ───────────────────────────────────────────────────────

class TestThresholdGradientExtra:
    def test_returns_bool_array(self):
        mag = np.zeros((8, 8), dtype=np.float32)
        out = threshold_gradient(mag, 0.5)
        assert out.dtype == bool

    def test_shape_preserved(self):
        mag = np.zeros((16, 24), dtype=np.float32)
        out = threshold_gradient(mag, 0.1)
        assert out.shape == (16, 24)

    def test_zero_threshold_all_true(self):
        mag = np.ones((4, 4), dtype=np.float32)
        out = threshold_gradient(mag, 0.0)
        assert out.all()

    def test_high_threshold_all_false(self):
        mag = np.zeros((4, 4), dtype=np.float32)
        out = threshold_gradient(mag, 0.5)
        assert not out.any()

    def test_non_2d_raises(self):
        mag = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            threshold_gradient(mag)

    def test_none_threshold_uses_cfg(self):
        cfg = GradientConfig(threshold=0.0)
        mag = np.ones((4, 4), dtype=np.float32) * 0.5
        out = threshold_gradient(mag, cfg=cfg, threshold=None)
        assert out.all()


# ─── suppress_non_maximum ─────────────────────────────────────────────────────

class TestSuppressNonMaximumExtra:
    def _mag_dir(self, h=10, w=10):
        mag = np.random.rand(h, w).astype(np.float32)
        direction = np.zeros((h, w), dtype=np.float32)
        return mag, direction

    def test_returns_ndarray(self):
        mag, direction = self._mag_dir()
        assert isinstance(suppress_non_maximum(mag, direction), np.ndarray)

    def test_shape_preserved(self):
        mag, direction = self._mag_dir(12, 14)
        out = suppress_non_maximum(mag, direction)
        assert out.shape == (12, 14)

    def test_dtype_float32(self):
        mag, direction = self._mag_dir()
        out = suppress_non_maximum(mag, direction)
        assert out.dtype == np.float32

    def test_output_le_magnitude(self):
        mag, direction = self._mag_dir()
        out = suppress_non_maximum(mag, direction)
        assert np.all(out <= mag + 1e-6)

    def test_magnitude_not_2d_raises(self):
        mag = np.zeros((4, 4, 2), dtype=np.float32)
        direction = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            suppress_non_maximum(mag, direction)

    def test_shape_mismatch_raises(self):
        mag = np.zeros((4, 4), dtype=np.float32)
        direction = np.zeros((4, 6), dtype=np.float32)
        with pytest.raises(ValueError):
            suppress_non_maximum(mag, direction)


# ─── compute_edge_density ─────────────────────────────────────────────────────

class TestComputeEdgeDensityExtra:
    def test_returns_float(self):
        assert isinstance(compute_edge_density(_gray()), float)

    def test_result_in_range(self):
        density = compute_edge_density(_gray())
        assert 0.0 <= density <= 1.0

    def test_uniform_image_low_density(self):
        density = compute_edge_density(_gray(val=128))
        assert density < 0.1

    def test_ramp_higher_density(self):
        d_uniform = compute_edge_density(_gray(val=128))
        d_ramp = compute_edge_density(_ramp())
        assert d_ramp >= d_uniform

    def test_roi_restricts_region(self):
        img = _ramp(32, 32)
        density_full = compute_edge_density(img)
        density_roi = compute_edge_density(img, roi=(0, 0, 16, 16))
        assert isinstance(density_roi, float)

    def test_empty_roi_returns_zero(self):
        density = compute_edge_density(_gray(), roi=(10, 10, 5, 5))
        assert density == pytest.approx(0.0)

    def test_4d_raises(self):
        with pytest.raises(ValueError):
            compute_edge_density(np.zeros((2, 4, 4, 3), dtype=np.uint8))


# ─── batch_compute_gradients ──────────────────────────────────────────────────

class TestBatchComputeGradientsExtra:
    def test_returns_list(self):
        result = batch_compute_gradients([_gray()])
        assert isinstance(result, list)

    def test_length_matches(self):
        result = batch_compute_gradients([_gray(), _ramp()])
        assert len(result) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            batch_compute_gradients([])

    def test_each_element_is_ndarray(self):
        for out in batch_compute_gradients([_gray(), _ramp()]):
            assert isinstance(out, np.ndarray)

    def test_shapes_correct(self):
        imgs = [_gray(16, 24), _gray(8, 8)]
        results = batch_compute_gradients(imgs)
        assert results[0].shape == (16, 24)
        assert results[1].shape == (8, 8)

    def test_none_cfg(self):
        result = batch_compute_gradients([_gray()], cfg=None)
        assert len(result) == 1
