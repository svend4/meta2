"""Extra tests for puzzle_reconstruction.algorithms.rotation_estimator
(supplementing test_algorithms_rotation_estimator.py)."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.rotation_estimator import (
    RotationEstimate,
    batch_estimate_rotations,
    estimate_by_gradient,
    estimate_by_moments,
    estimate_by_pca,
    estimate_rotation_pair,
    refine_rotation,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _grad(h=64, w=64):
    return np.tile(np.arange(w, dtype=np.uint8), (h, 1))


def _rand(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _bgr(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _contour(n=60, angle_deg=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.radians(angle_deg)
    x = 20 * np.cos(t)
    y = 8 * np.sin(t)
    cx = x * np.cos(rad) - y * np.sin(rad)
    cy = x * np.sin(rad) + y * np.cos(rad)
    return np.stack([cx + 32, cy + 32], axis=1).astype(np.float64)


# ─── RotationEstimate extras2 ─────────────────────────────────────────────────

class TestRotationEstimateExtra2:
    def test_repr_contains_angle(self):
        r = RotationEstimate(angle_deg=22.5, confidence=0.8, method="pca")
        assert "22" in repr(r) or "22.5" in repr(r) or "pca" in repr(r)

    def test_angle_90_stored(self):
        r = RotationEstimate(angle_deg=90.0, confidence=0.5, method="gradient")
        assert r.angle_deg == pytest.approx(90.0)

    def test_confidence_half(self):
        r = RotationEstimate(angle_deg=0.0, confidence=0.5, method="moments")
        assert r.confidence == pytest.approx(0.5)

    def test_params_multiple_keys(self):
        r = RotationEstimate(angle_deg=0.0, confidence=1.0, method="refine",
                             params={"initial_angle": 5.0, "search_range": 3.0,
                                     "n_steps": 10})
        assert r.params["n_steps"] == 10

    def test_method_test_stored(self):
        r = RotationEstimate(angle_deg=0.0, confidence=0.0, method="test")
        assert r.method == "test"


# ─── estimate_by_pca extras2 ─────────────────────────────────────────────────

class TestEstimateByPcaExtra2:
    def test_contour_at_0_degrees(self):
        r = estimate_by_pca(_contour(100, 0.0))
        assert isinstance(r, RotationEstimate)
        assert r.method == "pca"

    def test_contour_at_45_degrees(self):
        r = estimate_by_pca(_contour(100, 45.0))
        assert -90.0 < r.angle_deg <= 90.0

    def test_contour_at_minus_45(self):
        r = estimate_by_pca(_contour(100, -45.0))
        assert -90.0 < r.angle_deg <= 90.0

    def test_very_large_contour(self):
        r = estimate_by_pca(_contour(500, 30.0))
        assert isinstance(r, RotationEstimate)

    def test_flat_horizontal_contour(self):
        xs = np.linspace(0, 100, 50)
        pts = np.stack([xs, np.zeros(50)], axis=1)
        r = estimate_by_pca(pts)
        assert abs(r.angle_deg) < 5.0

    def test_eigenvalues_key_present(self):
        r = estimate_by_pca(_contour(60, 20.0))
        assert "eigenvalues" in r.params

    def test_confidence_range(self):
        r = estimate_by_pca(_contour(80, 10.0))
        assert 0.0 <= r.confidence <= 1.0


# ─── estimate_by_moments extras2 ─────────────────────────────────────────────

class TestEstimateByMomentsExtra2:
    def test_small_image_16x16(self):
        img = _rand(h=16, w=16)
        r = estimate_by_moments(img)
        assert isinstance(r, RotationEstimate)

    def test_large_image_128x128(self):
        img = _rand(h=128, w=128)
        r = estimate_by_moments(img)
        assert isinstance(r, RotationEstimate)

    def test_rgb_3channel(self):
        img = _bgr(h=64, w=64)
        r = estimate_by_moments(img)
        assert r.method == "moments"

    def test_non_square_48x80(self):
        img = _rand(h=48, w=80)
        r = estimate_by_moments(img)
        assert isinstance(r, RotationEstimate)

    def test_params_has_mu11(self):
        r = estimate_by_moments(_grad())
        assert "mu11" in r.params

    def test_confidence_in_range(self):
        r = estimate_by_moments(_rand())
        assert 0.0 <= r.confidence <= 1.0

    def test_angle_in_range(self):
        r = estimate_by_moments(_rand())
        assert -90.0 < r.angle_deg <= 90.0


# ─── estimate_by_gradient extras2 ────────────────────────────────────────────

class TestEstimateByGradientExtra2:
    def test_n_bins_36(self):
        r = estimate_by_gradient(_rand(), n_bins=36)
        assert r.params["n_bins"] == 36

    def test_n_bins_360(self):
        r = estimate_by_gradient(_rand(), n_bins=360)
        assert isinstance(r, RotationEstimate)

    def test_very_small_image_8x8(self):
        img = _rand(h=8, w=8)
        r = estimate_by_gradient(img)
        assert isinstance(r, RotationEstimate)

    def test_deterministic_multiple_calls(self):
        img = _rand(seed=5)
        angles = [estimate_by_gradient(img).angle_deg for _ in range(3)]
        assert all(abs(a - angles[0]) < 1e-6 for a in angles)

    def test_bgr_accepted(self):
        r = estimate_by_gradient(_bgr())
        assert r.method == "gradient"

    def test_confidence_range(self):
        r = estimate_by_gradient(_rand())
        assert 0.0 <= r.confidence <= 1.0

    def test_angle_in_range(self):
        r = estimate_by_gradient(_rand())
        assert -90.0 < r.angle_deg <= 90.0

    def test_n_bins_180(self):
        r = estimate_by_gradient(_grad(), n_bins=180)
        assert r.params["n_bins"] == 180


# ─── refine_rotation extras2 ─────────────────────────────────────────────────

class TestRefineRotationExtra2:
    def test_search_range_1_n_steps_5(self):
        r = refine_rotation(_rand(), initial_angle=0.0,
                            search_range=1.0, n_steps=5)
        assert isinstance(r, RotationEstimate)

    def test_search_range_45_large(self):
        r = refine_rotation(_rand(), initial_angle=0.0,
                            search_range=45.0, n_steps=10)
        assert -90.0 <= r.angle_deg <= 90.0

    def test_n_steps_50(self):
        r = refine_rotation(_rand(), initial_angle=0.0,
                            search_range=5.0, n_steps=50)
        assert isinstance(r, RotationEstimate)

    def test_angle_within_initial_plus_range(self):
        init = 10.0
        sr = 5.0
        r = refine_rotation(_rand(), initial_angle=init,
                            search_range=sr, n_steps=10)
        assert init - sr - 1e-9 <= r.angle_deg <= init + sr + 1e-9

    def test_params_n_steps_stored(self):
        r = refine_rotation(_rand(), initial_angle=0.0,
                            search_range=3.0, n_steps=15)
        assert r.params["n_steps"] == 15

    def test_rgb_accepted(self):
        r = refine_rotation(_bgr(), initial_angle=5.0)
        assert r.method == "refine"

    def test_confidence_range(self):
        r = refine_rotation(_rand(), initial_angle=0.0)
        assert 0.0 <= r.confidence <= 1.0


# ─── estimate_rotation_pair extras2 ──────────────────────────────────────────

class TestEstimateRotationPairExtra2:
    def test_gradient_method_explicit(self):
        r1, r2 = estimate_rotation_pair(_rand(), _rand(), method="gradient")
        assert r1.method == r2.method == "gradient"

    def test_moments_method_explicit(self):
        r1, r2 = estimate_rotation_pair(_rand(), _rand(), method="moments")
        assert r1.method == r2.method == "moments"

    def test_rgb_pairs(self):
        r1, r2 = estimate_rotation_pair(_bgr(), _bgr())
        assert isinstance(r1, RotationEstimate)
        assert isinstance(r2, RotationEstimate)

    def test_different_images(self):
        r1, r2 = estimate_rotation_pair(_rand(seed=0), _rand(seed=99))
        assert -90.0 < r1.angle_deg <= 90.0
        assert -90.0 < r2.angle_deg <= 90.0

    def test_both_confidence_in_range(self):
        r1, r2 = estimate_rotation_pair(_rand(), _rand())
        assert 0.0 <= r1.confidence <= 1.0
        assert 0.0 <= r2.confidence <= 1.0

    def test_invalid_method_pca_raises(self):
        with pytest.raises(ValueError):
            estimate_rotation_pair(_rand(), _rand(), method="pca")


# ─── batch_estimate_rotations extras2 ────────────────────────────────────────

class TestBatchEstimateRotationsExtra2:
    def test_large_batch_15_images(self):
        imgs = [_rand(seed=i) for i in range(15)]
        result = batch_estimate_rotations(imgs, method="gradient")
        assert len(result) == 15

    def test_confidence_all_in_range(self):
        imgs = [_rand(seed=i) for i in range(6)]
        for r in batch_estimate_rotations(imgs):
            assert 0.0 <= r.confidence <= 1.0

    def test_moments_method_all(self):
        imgs = [_rand(seed=i) for i in range(4)]
        result = batch_estimate_rotations(imgs, method="moments")
        assert all(r.method == "moments" for r in result)

    def test_gradient_method_all(self):
        imgs = [_rand(seed=i) for i in range(4)]
        result = batch_estimate_rotations(imgs, method="gradient")
        assert all(r.method == "gradient" for r in result)

    def test_bgr_images(self):
        imgs = [_bgr(seed=i) for i in range(3)]
        result = batch_estimate_rotations(imgs, method="gradient")
        assert all(isinstance(r, RotationEstimate) for r in result)

    def test_angles_in_range(self):
        imgs = [_rand(seed=i) for i in range(5)]
        for r in batch_estimate_rotations(imgs):
            assert -90.0 < r.angle_deg <= 90.0
