"""Extra tests for puzzle_reconstruction/algorithms/gradient_flow.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.gradient_flow import (
    GradientField,
    GradientStats,
    compute_gradient,
    compute_magnitude,
    compute_orientation,
    compute_divergence,
    compute_curl,
    flow_along_boundary,
    compare_gradient_fields,
    batch_gradient_fields,
    compute_gradient_stats,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _rand(h=32, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _field(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    gx = rng.uniform(-50, 50, (h, w)).astype(np.float32)
    gy = rng.uniform(-50, 50, (h, w)).astype(np.float32)
    return GradientField(gx=gx, gy=gy)


def _contour(n=16, r=10.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([16 + r * np.cos(t), 16 + r * np.sin(t)], axis=1).astype(np.float32)


# ─── GradientField (extra) ────────────────────────────────────────────────────

class TestGradientFieldExtra:
    def test_gx_stored_correctly(self):
        gx = np.ones((8, 8), dtype=np.float32) * 3.0
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        np.testing.assert_array_equal(f.gx, gx)

    def test_gy_stored_correctly(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.ones((8, 8), dtype=np.float32) * 7.0
        f = GradientField(gx=gx, gy=gy)
        np.testing.assert_array_equal(f.gy, gy)

    def test_non_square_shape(self):
        gx = np.zeros((16, 24), dtype=np.float32)
        gy = np.zeros((16, 24), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert f.shape == (16, 24)

    def test_large_field(self):
        gx = np.random.randn(256, 256).astype(np.float32)
        gy = np.random.randn(256, 256).astype(np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert f.shape == (256, 256)

    def test_extra_params_stored(self):
        gx = np.zeros((4, 4), dtype=np.float32)
        gy = np.zeros((4, 4), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy, params={"ksize": 5, "normalize": True})
        assert f.params["ksize"] == 5
        assert f.params["normalize"] is True


# ─── GradientStats (extra) ────────────────────────────────────────────────────

class TestGradientStatsExtra:
    def test_all_fields_stored(self):
        s = GradientStats(
            mean_magnitude=3.0, std_magnitude=1.5,
            mean_orientation=0.3, dominant_angle=0.5,
            edge_density=0.4,
        )
        assert s.mean_magnitude == pytest.approx(3.0)
        assert s.std_magnitude == pytest.approx(1.5)
        assert s.mean_orientation == pytest.approx(0.3)
        assert s.dominant_angle == pytest.approx(0.5)
        assert s.edge_density == pytest.approx(0.4)

    def test_zero_values_ok(self):
        s = GradientStats(
            mean_magnitude=0.0, std_magnitude=0.0,
            mean_orientation=0.0, dominant_angle=0.0,
            edge_density=0.0,
        )
        assert s.mean_magnitude == pytest.approx(0.0)
        assert s.edge_density == pytest.approx(0.0)

    def test_params_with_multiple_keys(self):
        s = GradientStats(
            mean_magnitude=1.0, std_magnitude=0.5,
            mean_orientation=0.0, dominant_angle=0.0,
            edge_density=0.2,
            params={"threshold": 5.0, "n_bins": 36},
        )
        assert s.params["threshold"] == pytest.approx(5.0)
        assert s.params["n_bins"] == 36


# ─── compute_gradient (extra) ─────────────────────────────────────────────────

class TestComputeGradientExtra:
    def test_ksize_3_returns_field(self):
        f = compute_gradient(_rand(), ksize=3)
        assert isinstance(f, GradientField)

    def test_ksize_5_returns_field(self):
        f = compute_gradient(_rand(), ksize=5)
        assert isinstance(f, GradientField)

    def test_ksize_7_returns_field(self):
        f = compute_gradient(_rand(), ksize=7)
        assert isinstance(f, GradientField)

    def test_constant_image_near_zero_gx(self):
        f = compute_gradient(_gray(val=200))
        assert np.abs(f.gx).max() < 1e-3

    def test_constant_image_near_zero_gy(self):
        f = compute_gradient(_gray(val=200))
        assert np.abs(f.gy).max() < 1e-3

    def test_gx_gy_shapes_match(self):
        img = _rand(20, 30)
        f = compute_gradient(img)
        assert f.gx.shape == f.gy.shape

    def test_normalize_max_le_one(self):
        img = _rand(32, 32)
        f = compute_gradient(img, normalize=True)
        mag = np.sqrt(f.gx ** 2 + f.gy ** 2)
        assert mag.max() <= 1.0 + 1e-5

    def test_non_square_image_ok(self):
        img = _rand(20, 40)
        f = compute_gradient(img)
        assert f.shape == (20, 40)


# ─── compute_magnitude (extra) ────────────────────────────────────────────────

class TestComputeMagnitudeExtra:
    def test_known_value(self):
        gx = np.full((4, 4), 3.0, dtype=np.float32)
        gy = np.full((4, 4), 4.0, dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        mag = compute_magnitude(f)
        np.testing.assert_allclose(mag, 5.0, atol=1e-5)

    def test_all_nonneg(self):
        f = _field()
        mag = compute_magnitude(f)
        assert (mag >= 0.0).all()

    def test_finite_values(self):
        f = _field()
        mag = compute_magnitude(f)
        assert np.all(np.isfinite(mag))

    def test_large_field_ok(self):
        f = _field(128, 128)
        mag = compute_magnitude(f)
        assert mag.shape == (128, 128)


# ─── compute_orientation (extra) ──────────────────────────────────────────────

class TestComputeOrientationExtra:
    def test_known_angle_45deg(self):
        gx = np.full((4, 4), 1.0, dtype=np.float32)
        gy = np.full((4, 4), 1.0, dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        o = compute_orientation(f)
        np.testing.assert_allclose(o, np.pi / 4, atol=1e-5)

    def test_finite_values(self):
        f = _field()
        o = compute_orientation(f)
        assert np.all(np.isfinite(o))

    def test_all_in_range(self):
        for seed in range(3):
            f = _field(seed=seed)
            o = compute_orientation(f)
            assert o.min() >= -np.pi - 1e-5
            assert o.max() <= np.pi + 1e-5

    def test_shape_matches_field(self):
        f = _field(20, 30)
        o = compute_orientation(f)
        assert o.shape == (20, 30)


# ─── compute_divergence and curl (extra) ──────────────────────────────────────

class TestComputeDivergenceCurlExtra:
    def test_divergence_finite(self):
        f = _field()
        d = compute_divergence(f)
        assert np.all(np.isfinite(d))

    def test_curl_finite(self):
        f = _field()
        c = compute_curl(f)
        assert np.all(np.isfinite(c))

    def test_divergence_large_field(self):
        f = _field(64, 64)
        d = compute_divergence(f)
        assert d.shape == (64, 64)

    def test_curl_large_field(self):
        f = _field(64, 64)
        c = compute_curl(f)
        assert c.shape == (64, 64)

    def test_uniform_field_near_zero_divergence(self):
        gx = np.full((8, 8), 5.0, dtype=np.float32)
        gy = np.full((8, 8), 3.0, dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        d = compute_divergence(f)
        assert np.abs(d).max() < 1e-2

    def test_uniform_field_near_zero_curl(self):
        gx = np.full((8, 8), 5.0, dtype=np.float32)
        gy = np.full((8, 8), 3.0, dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        c = compute_curl(f)
        assert np.abs(c).max() < 1e-2


# ─── flow_along_boundary (extra) ──────────────────────────────────────────────

class TestFlowAlongBoundaryExtra:
    def test_large_contour(self):
        f = compute_gradient(_rand(64, 64))
        contour = _contour(n=64)
        flow = flow_along_boundary(f, contour)
        assert len(flow) == 64

    def test_single_point(self):
        f = compute_gradient(_rand())
        pt = np.array([[10.0, 10.0]], dtype=np.float32)
        flow = flow_along_boundary(f, pt)
        assert len(flow) == 1

    def test_window_3_ok(self):
        f = compute_gradient(_rand())
        flow = flow_along_boundary(f, _contour(20), window=3)
        assert len(flow) == 20

    def test_output_dtype_float32(self):
        f = compute_gradient(_rand())
        flow = flow_along_boundary(f, _contour(10))
        assert flow.dtype == np.float32

    def test_finite_output(self):
        f = compute_gradient(_rand())
        flow = flow_along_boundary(f, _contour(16))
        assert np.all(np.isfinite(flow))

    def test_opencv_format_reshaped(self):
        f = compute_gradient(_rand())
        contour = _contour(12).reshape(-1, 1, 2)
        flow = flow_along_boundary(f, contour)
        assert len(flow) == 12


# ─── compare_gradient_fields (extra) ──────────────────────────────────────────

class TestCompareGradientFieldsExtra:
    def test_result_float_type(self):
        f1 = _field(seed=0)
        f2 = _field(seed=1)
        result = compare_gradient_fields(f1, f2)
        assert isinstance(result, float)

    def test_symmetric(self):
        f1 = compute_gradient(_rand(seed=1))
        f2 = compute_gradient(_rand(seed=2))
        s1 = compare_gradient_fields(f1, f2)
        s2 = compare_gradient_fields(f2, f1)
        assert s1 == pytest.approx(s2, abs=1e-6)

    def test_different_fields_not_one(self):
        f1 = compute_gradient(_rand(seed=0))
        f2 = compute_gradient(_rand(seed=99))
        result = compare_gradient_fields(f1, f2)
        assert result < 1.0

    def test_empty_mask_zero(self):
        f1 = _field()
        f2 = _field(seed=1)
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = compare_gradient_fields(f1, f2, mask=mask)
        assert result == pytest.approx(0.0)

    def test_finite_result(self):
        f1 = _field()
        f2 = _field(seed=3)
        assert np.isfinite(compare_gradient_fields(f1, f2))


# ─── batch_gradient_fields (extra) ────────────────────────────────────────────

class TestBatchGradientFieldsExtra:
    def test_single_image(self):
        result = batch_gradient_fields([_rand()])
        assert len(result) == 1

    def test_consistent_with_individual(self):
        img = _rand(seed=5)
        batch = batch_gradient_fields([img])
        individual = compute_gradient(img)
        np.testing.assert_allclose(batch[0].gx, individual.gx, atol=1e-6)

    def test_large_batch(self):
        imgs = [_rand(seed=i) for i in range(10)]
        result = batch_gradient_fields(imgs)
        assert len(result) == 10

    def test_all_shapes_match_inputs(self):
        imgs = [_rand(h=16, w=32, seed=i) for i in range(3)]
        for f in batch_gradient_fields(imgs):
            assert f.shape == (16, 32)

    def test_ksize_stored_in_params(self):
        imgs = [_rand()]
        result = batch_gradient_fields(imgs, ksize=7)
        assert result[0].params["ksize"] == 7


# ─── compute_gradient_stats (extra) ───────────────────────────────────────────

class TestComputeGradientStatsExtra:
    def test_constant_image_zero_edge_density(self):
        f = compute_gradient(_gray(val=128))
        s = compute_gradient_stats(f, threshold=1.0)
        assert s.edge_density == pytest.approx(0.0, abs=1e-5)

    def test_random_image_positive_edge_density(self):
        f = compute_gradient(_rand())
        s = compute_gradient_stats(f, threshold=1.0)
        assert s.edge_density > 0.0

    def test_large_threshold_low_edge_density(self):
        f = compute_gradient(_rand())
        s_low = compute_gradient_stats(f, threshold=1.0)
        s_high = compute_gradient_stats(f, threshold=10000.0)
        assert s_low.edge_density >= s_high.edge_density

    def test_mean_magnitude_finite(self):
        f = _field()
        s = compute_gradient_stats(f)
        assert np.isfinite(s.mean_magnitude)

    def test_dominant_angle_finite(self):
        f = _field()
        s = compute_gradient_stats(f)
        assert np.isfinite(s.dominant_angle)

    def test_n_bins_16_ok(self):
        f = _field()
        s = compute_gradient_stats(f, n_orientation_bins=16)
        assert s.params["n_orientation_bins"] == 16

    def test_returns_gradient_stats_type(self):
        f = compute_gradient(_rand())
        s = compute_gradient_stats(f)
        assert isinstance(s, GradientStats)
