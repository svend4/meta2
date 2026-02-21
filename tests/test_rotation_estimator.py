"""Tests for puzzle_reconstruction.algorithms.rotation_estimator."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gray(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _gradient_gray(h: int = 64, w: int = 64) -> np.ndarray:
    """Image with horizontal gradient (distinct rows)."""
    img = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    return img


def _bgr(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(w, dtype=np.uint8)[np.newaxis, :]
    return img


def _horizontal_line_contour(length: int = 50) -> np.ndarray:
    """Horizontal line contour: (length, 2) with y=0."""
    xs = np.linspace(0, length, length, dtype=np.float64)
    ys = np.zeros(length, dtype=np.float64)
    return np.stack([xs, ys], axis=1)


def _tilted_contour(angle_deg: float = 30.0, length: int = 50) -> np.ndarray:
    """Contour along a direction specified by angle_deg."""
    t = np.linspace(0, length, length, dtype=np.float64)
    rad = np.radians(angle_deg)
    xs = t * np.cos(rad)
    ys = t * np.sin(rad)
    return np.stack([xs, ys], axis=1)


# ─── RotationEstimate ────────────────────────────────────────────────────────

class TestRotationEstimate:
    def test_fields_stored(self):
        r = RotationEstimate(angle_deg=15.0, confidence=0.8, method="pca")
        assert r.angle_deg == pytest.approx(15.0)
        assert r.confidence == pytest.approx(0.8)
        assert r.method == "pca"
        assert r.params == {}

    def test_params_stored(self):
        r = RotationEstimate(0.0, 1.0, "gradient", params={"n_bins": 90})
        assert r.params["n_bins"] == 90

    def test_repr_contains_angle(self):
        r = RotationEstimate(angle_deg=45.0, confidence=0.5, method="moments")
        assert "45" in repr(r) or "45.0" in repr(r)


# ─── estimate_by_pca ─────────────────────────────────────────────────────────

class TestEstimateByPca:
    def test_returns_rotation_estimate(self):
        c = _horizontal_line_contour()
        r = estimate_by_pca(c)
        assert isinstance(r, RotationEstimate)

    def test_method_is_pca(self):
        assert estimate_by_pca(_horizontal_line_contour()).method == "pca"

    def test_angle_in_range(self):
        r = estimate_by_pca(_tilted_contour(30.0))
        assert -90.0 < r.angle_deg <= 90.0

    def test_confidence_in_unit_interval(self):
        r = estimate_by_pca(_tilted_contour(45.0))
        assert 0.0 <= r.confidence <= 1.0

    def test_less_than_2_points_raises(self):
        with pytest.raises(ValueError):
            estimate_by_pca(np.array([[1.0, 2.0]]))

    def test_horizontal_line_near_zero(self):
        c = _horizontal_line_contour(100)
        r = estimate_by_pca(c)
        assert abs(r.angle_deg) < 1.0

    def test_cv2_format_contour_accepted(self):
        # cv2 contours have shape (N, 1, 2)
        c = _horizontal_line_contour().reshape(-1, 1, 2)
        r = estimate_by_pca(c)
        assert isinstance(r, RotationEstimate)

    def test_params_contain_eigenvalues(self):
        r = estimate_by_pca(_tilted_contour(20.0))
        assert "eigenvalues" in r.params


# ─── estimate_by_moments ─────────────────────────────────────────────────────

class TestEstimateByMoments:
    def test_returns_rotation_estimate(self):
        r = estimate_by_moments(_gradient_gray())
        assert isinstance(r, RotationEstimate)

    def test_method_is_moments(self):
        assert estimate_by_moments(_gradient_gray()).method == "moments"

    def test_angle_in_range(self):
        r = estimate_by_moments(_gradient_gray())
        assert -90.0 < r.angle_deg <= 90.0

    def test_confidence_in_unit_interval(self):
        r = estimate_by_moments(_gradient_gray())
        assert 0.0 <= r.confidence <= 1.0

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            estimate_by_moments(np.empty((0, 0), dtype=np.uint8))

    def test_uniform_image_raises(self):
        with pytest.raises(ValueError):
            estimate_by_moments(_gray(value=128))

    def test_bgr_image_accepted(self):
        r = estimate_by_moments(_bgr())
        assert isinstance(r, RotationEstimate)

    def test_params_contain_moments(self):
        r = estimate_by_moments(_gradient_gray())
        assert "mu20" in r.params
        assert "mu02" in r.params


# ─── estimate_by_gradient ────────────────────────────────────────────────────

class TestEstimateByGradient:
    def test_returns_rotation_estimate(self):
        r = estimate_by_gradient(_gradient_gray())
        assert isinstance(r, RotationEstimate)

    def test_method_is_gradient(self):
        assert estimate_by_gradient(_gradient_gray()).method == "gradient"

    def test_angle_in_range(self):
        r = estimate_by_gradient(_gradient_gray())
        assert -90.0 < r.angle_deg <= 90.0

    def test_confidence_in_unit_interval(self):
        r = estimate_by_gradient(_gradient_gray())
        assert 0.0 <= r.confidence <= 1.0

    def test_n_bins_less_than_2_raises(self):
        with pytest.raises(ValueError):
            estimate_by_gradient(_gradient_gray(), n_bins=1)

    def test_bgr_accepted(self):
        r = estimate_by_gradient(_bgr())
        assert isinstance(r, RotationEstimate)

    def test_custom_n_bins(self):
        r = estimate_by_gradient(_gradient_gray(), n_bins=36)
        assert isinstance(r, RotationEstimate)
        assert r.params["n_bins"] == 36

    def test_different_n_bins_different_precision(self):
        r90 = estimate_by_gradient(_gradient_gray(), n_bins=90)
        r180 = estimate_by_gradient(_gradient_gray(), n_bins=180)
        # Both return valid RotationEstimates
        assert isinstance(r90, RotationEstimate)
        assert isinstance(r180, RotationEstimate)


# ─── refine_rotation ─────────────────────────────────────────────────────────

class TestRefineRotation:
    def test_returns_rotation_estimate(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0)
        assert isinstance(r, RotationEstimate)

    def test_method_is_refine(self):
        assert refine_rotation(_gradient_gray(), 0.0).method == "refine"

    def test_angle_in_range(self):
        r = refine_rotation(_gradient_gray(), initial_angle=5.0)
        assert -90.0 < r.angle_deg <= 90.0

    def test_confidence_in_unit_interval(self):
        r = refine_rotation(_gradient_gray(), initial_angle=0.0)
        assert 0.0 <= r.confidence <= 1.0

    def test_n_steps_less_than_2_raises(self):
        with pytest.raises(ValueError):
            refine_rotation(_gradient_gray(), initial_angle=0.0, n_steps=1)

    def test_search_range_zero_raises(self):
        with pytest.raises(ValueError):
            refine_rotation(_gradient_gray(), initial_angle=0.0, search_range=0.0)

    def test_search_range_negative_raises(self):
        with pytest.raises(ValueError):
            refine_rotation(_gradient_gray(), initial_angle=0.0, search_range=-1.0)

    def test_angle_within_search_range(self):
        initial = 5.0
        search_range = 3.0
        r = refine_rotation(_gradient_gray(), initial_angle=initial,
                            search_range=search_range, n_steps=10)
        assert initial - search_range - 1e-9 <= r.angle_deg <= initial + search_range + 1e-9

    def test_params_stored(self):
        r = refine_rotation(_gradient_gray(), initial_angle=2.0,
                            search_range=4.0, n_steps=10)
        assert r.params["initial_angle"] == pytest.approx(2.0)
        assert r.params["search_range"] == pytest.approx(4.0)
        assert r.params["n_steps"] == 10


# ─── estimate_rotation_pair ──────────────────────────────────────────────────

class TestEstimateRotationPair:
    def test_returns_tuple_of_two(self):
        result = estimate_rotation_pair(_gradient_gray(), _gradient_gray())
        assert len(result) == 2

    def test_both_are_rotation_estimate(self):
        r1, r2 = estimate_rotation_pair(_gradient_gray(), _gradient_gray())
        assert isinstance(r1, RotationEstimate)
        assert isinstance(r2, RotationEstimate)

    def test_default_method_gradient(self):
        r1, r2 = estimate_rotation_pair(_gradient_gray(), _gradient_gray())
        assert r1.method == "gradient"
        assert r2.method == "gradient"

    def test_moments_method(self):
        r1, r2 = estimate_rotation_pair(
            _gradient_gray(), _gradient_gray(), method="moments"
        )
        assert r1.method == "moments"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            estimate_rotation_pair(_gradient_gray(), _gradient_gray(), method="pca")

    def test_different_images_different_angles(self):
        # Two clearly different images can produce different angle estimates
        img1 = _gradient_gray(64, 64)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        img2[10:20, :] = 255  # horizontal stripe → different gradient orientation
        r1, r2 = estimate_rotation_pair(img1, img2)
        # Both should still be valid RotationEstimates
        assert -90.0 < r1.angle_deg <= 90.0
        assert -90.0 < r2.angle_deg <= 90.0


# ─── batch_estimate_rotations ────────────────────────────────────────────────

class TestBatchEstimateRotations:
    def test_empty_returns_empty(self):
        assert batch_estimate_rotations([]) == []

    def test_length_preserved(self):
        imgs = [_gradient_gray()] * 5
        result = batch_estimate_rotations(imgs)
        assert len(result) == 5

    def test_all_rotation_estimate(self):
        imgs = [_gradient_gray(), _bgr()]
        result = batch_estimate_rotations(imgs, method="gradient")
        assert all(isinstance(r, RotationEstimate) for r in result)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_estimate_rotations([_gradient_gray()], method="unknown")

    def test_moments_method_applied(self):
        imgs = [_gradient_gray()] * 3
        result = batch_estimate_rotations(imgs, method="moments")
        assert all(r.method == "moments" for r in result)

    def test_gradient_method_applied(self):
        imgs = [_gradient_gray()] * 3
        result = batch_estimate_rotations(imgs, method="gradient")
        assert all(r.method == "gradient" for r in result)

    def test_confidence_in_unit_interval(self):
        imgs = [_gradient_gray()] * 4
        for r in batch_estimate_rotations(imgs):
            assert 0.0 <= r.confidence <= 1.0

    def test_single_image(self):
        result = batch_estimate_rotations([_gradient_gray()])
        assert len(result) == 1
        assert isinstance(result[0], RotationEstimate)
