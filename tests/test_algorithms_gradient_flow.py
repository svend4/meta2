"""Тесты для puzzle_reconstruction.algorithms.gradient_flow."""
import pytest
import numpy as np
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=32, w=32, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=32, w=32, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=32, w=32, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _field(h=32, w=32, seed=0) -> GradientField:
    rng = np.random.default_rng(seed)
    gx = rng.uniform(-50, 50, (h, w)).astype(np.float32)
    gy = rng.uniform(-50, 50, (h, w)).astype(np.float32)
    return GradientField(gx=gx, gy=gy)


def _circle_contour(n=32, r=10.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([16 + r * np.cos(t), 16 + r * np.sin(t)], axis=1).astype(np.float32)


# ─── TestGradientField ────────────────────────────────────────────────────────

class TestGradientField:
    def test_basic_shape(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert f.shape == (8, 8)

    def test_shape_property(self):
        f = _field(16, 24)
        assert f.shape == (16, 24)

    def test_params_default_empty(self):
        f = GradientField(gx=np.zeros((4, 4), dtype=np.float32),
                          gy=np.zeros((4, 4), dtype=np.float32))
        assert f.params == {}

    def test_shape_mismatch_raises(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 9), dtype=np.float32)
        with pytest.raises(ValueError):
            GradientField(gx=gx, gy=gy)

    def test_params_stored(self):
        f = GradientField(gx=np.zeros((4, 4), dtype=np.float32),
                          gy=np.zeros((4, 4), dtype=np.float32),
                          params={"ksize": 3})
        assert f.params["ksize"] == 3

    def test_3d_gx_shape(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert len(f.shape) == 2


# ─── TestGradientStats ────────────────────────────────────────────────────────

class TestGradientStats:
    def test_basic_fields(self):
        s = GradientStats(
            mean_magnitude=5.0, std_magnitude=1.0,
            mean_orientation=0.5, dominant_angle=0.0,
            edge_density=0.3,
        )
        assert s.mean_magnitude == pytest.approx(5.0)
        assert s.edge_density == pytest.approx(0.3)

    def test_params_default_empty(self):
        s = GradientStats(
            mean_magnitude=0.0, std_magnitude=0.0,
            mean_orientation=0.0, dominant_angle=0.0,
            edge_density=0.0,
        )
        assert s.params == {}

    def test_params_stored(self):
        s = GradientStats(
            mean_magnitude=1.0, std_magnitude=0.5,
            mean_orientation=0.0, dominant_angle=0.0,
            edge_density=0.1,
            params={"threshold": 10.0},
        )
        assert s.params["threshold"] == pytest.approx(10.0)


# ─── TestComputeGradient ──────────────────────────────────────────────────────

class TestComputeGradient:
    def test_returns_gradient_field(self):
        f = compute_gradient(_rand_gray())
        assert isinstance(f, GradientField)

    def test_output_shape_matches_input(self):
        img = _rand_gray(24, 32)
        f = compute_gradient(img)
        assert f.shape == (24, 32)

    def test_ksize_3_default(self):
        f = compute_gradient(_rand_gray())
        assert f.params["ksize"] == 3

    def test_ksize_1_ok(self):
        f = compute_gradient(_rand_gray(), ksize=1)
        assert isinstance(f, GradientField)

    def test_ksize_5_ok(self):
        f = compute_gradient(_rand_gray(), ksize=5)
        assert isinstance(f, GradientField)

    def test_ksize_7_ok(self):
        f = compute_gradient(_rand_gray(), ksize=7)
        assert isinstance(f, GradientField)

    def test_ksize_2_raises(self):
        with pytest.raises(ValueError):
            compute_gradient(_rand_gray(), ksize=2)

    def test_ksize_4_raises(self):
        with pytest.raises(ValueError):
            compute_gradient(_rand_gray(), ksize=4)

    def test_ksize_0_raises(self):
        with pytest.raises(ValueError):
            compute_gradient(_rand_gray(), ksize=0)

    def test_rgb_ok(self):
        f = compute_gradient(_rand_rgb())
        assert isinstance(f, GradientField)

    def test_normalize_true(self):
        f = compute_gradient(_rand_gray(), normalize=True)
        assert f.params["normalize"] is True
        # Normalized: max magnitude ~ 1
        mag = np.sqrt(f.gx ** 2 + f.gy ** 2)
        assert mag.max() <= 1.0 + 1e-5

    def test_normalize_false_default(self):
        f = compute_gradient(_rand_gray())
        assert f.params["normalize"] is False

    def test_constant_image_zero_gradient(self):
        f = compute_gradient(_gray())
        mag = np.sqrt(f.gx ** 2 + f.gy ** 2)
        assert mag.max() < 1e-3


# ─── TestComputeMagnitudeOrientation ─────────────────────────────────────────

class TestComputeMagnitudeOrientation:
    def test_magnitude_shape(self):
        f = _field(16, 24)
        mag = compute_magnitude(f)
        assert mag.shape == (16, 24)

    def test_magnitude_dtype(self):
        f = _field()
        assert compute_magnitude(f).dtype == np.float32

    def test_magnitude_nonneg(self):
        f = _field()
        assert compute_magnitude(f).min() >= 0.0

    def test_zero_field_zero_magnitude(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        assert compute_magnitude(f).max() == pytest.approx(0.0)

    def test_orientation_shape(self):
        f = _field(16, 24)
        o = compute_orientation(f)
        assert o.shape == (16, 24)

    def test_orientation_dtype(self):
        f = _field()
        assert compute_orientation(f).dtype == np.float32

    def test_orientation_range(self):
        f = _field()
        o = compute_orientation(f)
        assert o.min() >= -np.pi - 1e-5
        assert o.max() <= np.pi + 1e-5


# ─── TestComputeDivergenceCurl ────────────────────────────────────────────────

class TestComputeDivergenceCurl:
    def test_divergence_shape(self):
        f = _field(16, 16)
        d = compute_divergence(f)
        assert d.shape == (16, 16)

    def test_divergence_dtype(self):
        f = _field()
        assert compute_divergence(f).dtype == np.float32

    def test_curl_shape(self):
        f = _field(16, 16)
        c = compute_curl(f)
        assert c.shape == (16, 16)

    def test_curl_dtype(self):
        f = _field()
        assert compute_curl(f).dtype == np.float32

    def test_zero_field_zero_divergence(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        d = compute_divergence(f)
        assert abs(d).max() < 1e-3

    def test_zero_field_zero_curl(self):
        gx = np.zeros((8, 8), dtype=np.float32)
        gy = np.zeros((8, 8), dtype=np.float32)
        f = GradientField(gx=gx, gy=gy)
        c = compute_curl(f)
        assert abs(c).max() < 1e-3


# ─── TestFlowAlongBoundary ────────────────────────────────────────────────────

class TestFlowAlongBoundary:
    def test_returns_array(self):
        f = compute_gradient(_rand_gray())
        contour = _circle_contour(20)
        flow = flow_along_boundary(f, contour)
        assert isinstance(flow, np.ndarray)

    def test_output_length(self):
        f = compute_gradient(_rand_gray())
        contour = _circle_contour(20)
        flow = flow_along_boundary(f, contour)
        assert len(flow) == 20

    def test_dtype_float32(self):
        f = compute_gradient(_rand_gray())
        contour = _circle_contour(16)
        assert flow_along_boundary(f, contour).dtype == np.float32

    def test_empty_contour(self):
        f = compute_gradient(_rand_gray())
        flow = flow_along_boundary(f, np.zeros((0, 2), dtype=np.float32))
        assert len(flow) == 0

    def test_window_zero_raises(self):
        f = compute_gradient(_rand_gray())
        with pytest.raises(ValueError):
            flow_along_boundary(f, _circle_contour(10), window=0)

    def test_window_negative_raises(self):
        f = compute_gradient(_rand_gray())
        with pytest.raises(ValueError):
            flow_along_boundary(f, _circle_contour(10), window=-1)

    def test_opencv_contour_format(self):
        # Shape (N, 1, 2)
        f = compute_gradient(_rand_gray())
        contour = _circle_contour(12).reshape(-1, 1, 2)
        flow = flow_along_boundary(f, contour)
        assert len(flow) == 12

    def test_window_two_ok(self):
        f = compute_gradient(_rand_gray())
        flow = flow_along_boundary(f, _circle_contour(16), window=2)
        assert len(flow) == 16


# ─── TestCompareGradientFields ────────────────────────────────────────────────

class TestCompareGradientFields:
    def test_identical_fields_one(self):
        f = compute_gradient(_rand_gray(seed=1))
        s = compare_gradient_fields(f, f)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_result_in_range(self):
        f1 = compute_gradient(_rand_gray(seed=1))
        f2 = compute_gradient(_rand_gray(seed=2))
        s = compare_gradient_fields(f1, f2)
        assert -1.0 <= s <= 1.0

    def test_shape_mismatch_raises(self):
        f1 = _field(16, 16)
        f2 = _field(16, 24)
        with pytest.raises(ValueError):
            compare_gradient_fields(f1, f2)

    def test_with_mask(self):
        f1 = compute_gradient(_rand_gray(seed=3))
        f2 = compute_gradient(_rand_gray(seed=4))
        mask = np.ones((32, 32), dtype=np.uint8)
        s = compare_gradient_fields(f1, f2, mask=mask)
        assert -1.0 <= s <= 1.0

    def test_empty_mask_returns_zero(self):
        f1 = _field()
        f2 = _field()
        mask = np.zeros((32, 32), dtype=np.uint8)
        s = compare_gradient_fields(f1, f2, mask=mask)
        assert s == pytest.approx(0.0)

    def test_returns_float(self):
        f1 = _field()
        f2 = _field(seed=1)
        assert isinstance(compare_gradient_fields(f1, f2), float)


# ─── TestBatchGradientFields ──────────────────────────────────────────────────

class TestBatchGradientFields:
    def test_returns_list(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        result = batch_gradient_fields(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_rand_gray(seed=i) for i in range(5)]
        assert len(batch_gradient_fields(imgs)) == 5

    def test_empty_list(self):
        assert batch_gradient_fields([]) == []

    def test_all_gradient_fields(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        for f in batch_gradient_fields(imgs):
            assert isinstance(f, GradientField)

    def test_ksize_passed(self):
        imgs = [_rand_gray(seed=0)]
        result = batch_gradient_fields(imgs, ksize=5)
        assert result[0].params["ksize"] == 5


# ─── TestComputeGradientStats ─────────────────────────────────────────────────

class TestComputeGradientStats:
    def test_returns_gradient_stats(self):
        f = compute_gradient(_rand_gray())
        s = compute_gradient_stats(f)
        assert isinstance(s, GradientStats)

    def test_mean_magnitude_nonneg(self):
        f = compute_gradient(_rand_gray())
        s = compute_gradient_stats(f)
        assert s.mean_magnitude >= 0.0

    def test_std_magnitude_nonneg(self):
        f = compute_gradient(_rand_gray())
        s = compute_gradient_stats(f)
        assert s.std_magnitude >= 0.0

    def test_edge_density_in_range(self):
        f = compute_gradient(_rand_gray())
        s = compute_gradient_stats(f)
        assert 0.0 <= s.edge_density <= 1.0

    def test_dominant_angle_in_range(self):
        f = compute_gradient(_rand_gray())
        s = compute_gradient_stats(f)
        assert -np.pi <= s.dominant_angle <= np.pi

    def test_threshold_neg_raises(self):
        f = _field()
        with pytest.raises(ValueError):
            compute_gradient_stats(f, threshold=-1.0)

    def test_n_bins_zero_raises(self):
        f = _field()
        with pytest.raises(ValueError):
            compute_gradient_stats(f, n_orientation_bins=0)

    def test_threshold_zero_ok(self):
        f = compute_gradient(_rand_gray())
        s = compute_gradient_stats(f, threshold=0.0)
        # threshold=0 → all pixels count as edges
        assert s.edge_density == pytest.approx(1.0, abs=1e-5)

    def test_params_stored(self):
        f = _field()
        s = compute_gradient_stats(f, threshold=5.0, n_orientation_bins=18)
        assert s.params["threshold"] == pytest.approx(5.0)
        assert s.params["n_orientation_bins"] == 18

    def test_custom_bins(self):
        f = _field()
        s = compute_gradient_stats(f, n_orientation_bins=72)
        assert s.params["n_orientation_bins"] == 72
