"""Extra tests for puzzle_reconstruction.algorithms.rotation_estimator."""
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

def _gray(h=64, w=64, value=128):
    return np.full((h, w), value, dtype=np.uint8)


def _gradient_gray(h=64, w=64):
    return np.tile(np.arange(w, dtype=np.uint8), (h, 1))


def _bgr(h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(w, dtype=np.uint8)[np.newaxis, :]
    return img


def _horizontal_contour(n=60):
    xs = np.linspace(0, 60, n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    return np.stack([xs, ys], axis=1)


def _tilted_contour(angle_deg=30.0, n=60):
    t = np.linspace(0, 60, n, dtype=np.float64)
    rad = np.radians(angle_deg)
    return np.stack([t * np.cos(rad), t * np.sin(rad)], axis=1)


# ─── RotationEstimate extras ──────────────────────────────────────────────────

class TestRotationEstimateExtra:
    def test_repr_is_string(self):
        r = RotationEstimate(angle_deg=12.0, confidence=0.9, method="pca")
        assert isinstance(repr(r), str)

    def test_negative_angle_stored(self):
        r = RotationEstimate(angle_deg=-45.0, confidence=0.5, method="gradient")
        assert r.angle_deg == pytest.approx(-45.0)

    def test_zero_angle_stored(self):
        r = RotationEstimate(angle_deg=0.0, confidence=1.0, method="moments")
        assert r.angle_deg == pytest.approx(0.0)

    def test_confidence_zero_valid(self):
        r = RotationEstimate(angle_deg=0.0, confidence=0.0, method="pca")
        assert r.confidence == pytest.approx(0.0)

    def test_confidence_one_valid(self):
        r = RotationEstimate(angle_deg=0.0, confidence=1.0, method="pca")
        assert r.confidence == pytest.approx(1.0)

    def test_custom_params_stored(self):
        r = RotationEstimate(angle_deg=5.0, confidence=0.7, method="pca",
                             params={"eigenvalues": [10.0, 1.0]})
        assert r.params["eigenvalues"] == [10.0, 1.0]

    def test_method_values_stored(self):
        for m in ("pca", "moments", "gradient", "refine"):
            r = RotationEstimate(angle_deg=0.0, confidence=0.0, method=m)
            assert r.method == m

    def test_repr_contains_method(self):
        r = RotationEstimate(angle_deg=10.0, confidence=0.5, method="gradient")
        assert "gradient" in repr(r) or "10" in repr(r)


# ─── estimate_by_pca extras ───────────────────────────────────────────────────

class TestEstimateByPcaExtra:
    def test_cv2_format_n_1_2_accepted(self):
        c = _horizontal_contour().reshape(-1, 1, 2)
        r = estimate_by_pca(c)
        assert isinstance(r, RotationEstimate)

    def test_large_contour(self):
        c = _horizontal_contour(n=200)
        r = estimate_by_pca(c)
        assert isinstance(r, RotationEstimate)

    def test_tilted_45_angle_in_range(self):
        c = _tilted_contour(45.0)
        r = estimate_by_pca(c)
        assert -90.0 < r.angle_deg <= 90.0

    def test_tilted_minus30_angle_in_range(self):
        c = _tilted_contour(-30.0)
        r = estimate_by_pca(c)
        assert -90.0 < r.angle_deg <= 90.0

    def test_eigenvalues_are_real_numbers(self):
        c = _tilted_contour(20.0)
        r = estimate_by_pca(c)
        evs = r.params["eigenvalues"]
        assert all(isinstance(ev, (int, float, np.floating)) for ev in evs)

    def test_two_points_minimum_ok(self):
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        r = estimate_by_pca(c)
        assert isinstance(r, RotationEstimate)

    def test_confidence_nonneg(self):
        r = estimate_by_pca(_horizontal_contour())
        assert r.confidence >= 0.0

    def test_horizontal_near_zero_angle(self):
        c = _horizontal_contour(100)
        r = estimate_by_pca(c)
        assert abs(r.angle_deg) < 2.0


# ─── estimate_by_moments extras ───────────────────────────────────────────────

class TestEstimateByMomentsExtra:
    def test_non_square_gradient(self):
        img = np.tile(np.arange(96, dtype=np.uint8), (48, 1))
        r = estimate_by_moments(img)
        assert isinstance(r, RotationEstimate)

    def test_large_image(self):
        img = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
        r = estimate_by_moments(img)
        assert isinstance(r, RotationEstimate)

    def test_params_has_mu20_mu02(self):
        r = estimate_by_moments(_gradient_gray())
        assert "mu20" in r.params
        assert "mu02" in r.params

    def test_angle_range_gradient_image(self):
        r = estimate_by_moments(_gradient_gray())
        assert -90.0 < r.angle_deg <= 90.0

    def test_confidence_range(self):
        r = estimate_by_moments(_gradient_gray())
        assert 0.0 <= r.confidence <= 1.0

    def test_method_moments(self):
        r = estimate_by_moments(_gradient_gray())
        assert r.method == "moments"

    def test_bgr_image_accepted(self):
        r = estimate_by_moments(_bgr())
        assert isinstance(r, RotationEstimate)


# ─── estimate_by_gradient extras ─────────────────────────────────────────────

class TestEstimateByGradientExtra:
    def test_n_bins_2_minimum_ok(self):
        r = estimate_by_gradient(_gradient_gray(), n_bins=2)
        assert isinstance(r, RotationEstimate)

    def test_n_bins_180(self):
        r = estimate_by_gradient(_gradient_gray(), n_bins=180)
        assert isinstance(r, RotationEstimate)
        assert r.params["n_bins"] == 180

    def test_n_bins_360(self):
        r = estimate_by_gradient(_gradient_gray(), n_bins=360)
        assert isinstance(r, RotationEstimate)

    def test_angle_range_nonneg_image(self):
        r = estimate_by_gradient(_gradient_gray())
        assert -90.0 < r.angle_deg <= 90.0

    def test_bgr_accepted(self):
        r = estimate_by_gradient(_bgr())
        assert isinstance(r, RotationEstimate)

    def test_confidence_range(self):
        r = estimate_by_gradient(_gradient_gray())
        assert 0.0 <= r.confidence <= 1.0

    def test_method_gradient(self):
        r = estimate_by_gradient(_gradient_gray())
        assert r.method == "gradient"

    def test_non_square_image(self):
        img = np.tile(np.arange(96, dtype=np.uint8), (32, 1))
        r = estimate_by_gradient(img)
        assert isinstance(r, RotationEstimate)


# ─── refine_rotation extras ───────────────────────────────────────────────────

class TestRefineRotationExtra:
    def test_n_steps_2_minimum_ok(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0, n_steps=2)
        assert isinstance(r, RotationEstimate)

    def test_n_steps_20(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0, n_steps=20)
        assert isinstance(r, RotationEstimate)

    def test_large_search_range(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0,
                            search_range=45.0, n_steps=10)
        assert -90.0 <= r.angle_deg <= 90.0

    def test_initial_angle_stored_in_params(self):
        r = refine_rotation(_gradient_gray(), initial_angle=7.5,
                            search_range=5.0, n_steps=10)
        assert r.params["initial_angle"] == pytest.approx(7.5)

    def test_n_steps_stored(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0,
                            search_range=5.0, n_steps=8)
        assert r.params["n_steps"] == 8

    def test_confidence_range(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0)
        assert 0.0 <= r.confidence <= 1.0

    def test_method_refine(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0)
        assert r.method == "refine"

    def test_negative_initial_angle(self):
        r = refine_rotation(_gradient_gray(), initial_angle=-10.0,
                            search_range=5.0, n_steps=10)
        assert isinstance(r, RotationEstimate)


# ─── estimate_rotation_pair extras ───────────────────────────────────────────

class TestEstimateRotationPairExtra:
    def test_gradient_method_both(self):
        r1, r2 = estimate_rotation_pair(_gradient_gray(), _gradient_gray(),
                                        method="gradient")
        assert r1.method == "gradient"
        assert r2.method == "gradient"

    def test_moments_method_both(self):
        r1, r2 = estimate_rotation_pair(_gradient_gray(), _gradient_gray(),
                                        method="moments")
        assert r1.method == "moments"
        assert r2.method == "moments"

    def test_both_confidence_in_range(self):
        r1, r2 = estimate_rotation_pair(_gradient_gray(), _gradient_gray())
        assert 0.0 <= r1.confidence <= 1.0
        assert 0.0 <= r2.confidence <= 1.0

    def test_both_angles_in_range(self):
        r1, r2 = estimate_rotation_pair(_gradient_gray(), _gradient_gray())
        assert -90.0 < r1.angle_deg <= 90.0
        assert -90.0 < r2.angle_deg <= 90.0

    def test_pca_method_raises(self):
        with pytest.raises(ValueError):
            estimate_rotation_pair(_gradient_gray(), _gradient_gray(), method="pca")

    def test_returns_tuple(self):
        result = estimate_rotation_pair(_gradient_gray(), _gradient_gray())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bgr_images(self):
        r1, r2 = estimate_rotation_pair(_bgr(), _bgr())
        assert isinstance(r1, RotationEstimate)
        assert isinstance(r2, RotationEstimate)


# ─── batch_estimate_rotations extras ─────────────────────────────────────────

class TestBatchEstimateRotationsExtra:
    def test_single_image_gradient(self):
        result = batch_estimate_rotations([_gradient_gray()])
        assert len(result) == 1
        assert isinstance(result[0], RotationEstimate)

    def test_ten_images(self):
        imgs = [_gradient_gray()] * 10
        result = batch_estimate_rotations(imgs, method="gradient")
        assert len(result) == 10

    def test_moments_method_all(self):
        imgs = [_gradient_gray()] * 4
        for r in batch_estimate_rotations(imgs, method="moments"):
            assert r.method == "moments"
            assert 0.0 <= r.confidence <= 1.0

    def test_gradient_method_all(self):
        imgs = [_gradient_gray()] * 4
        for r in batch_estimate_rotations(imgs, method="gradient"):
            assert r.method == "gradient"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_estimate_rotations([_gradient_gray()], method="xyz")

    def test_all_angles_in_range(self):
        imgs = [_gradient_gray()] * 5
        for r in batch_estimate_rotations(imgs):
            assert -90.0 < r.angle_deg <= 90.0

    def test_bgr_images_accepted(self):
        imgs = [_bgr()] * 3
        result = batch_estimate_rotations(imgs, method="gradient")
        assert len(result) == 3
        for r in result:
            assert isinstance(r, RotationEstimate)
