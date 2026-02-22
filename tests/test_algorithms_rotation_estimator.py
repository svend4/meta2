"""Тесты для puzzle_reconstruction.algorithms.rotation_estimator."""
import pytest
import numpy as np
from puzzle_reconstruction.algorithms.rotation_estimator import (
    RotationEstimate,
    estimate_by_pca,
    estimate_by_moments,
    estimate_by_gradient,
    refine_rotation,
    estimate_rotation_pair,
    batch_estimate_rotations,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=64, w=64, val=128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


def _rand_gray(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w), dtype=np.uint8)


def _rand_rgb(h=64, w=64, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _contour(n=50, angle_deg=0.0) -> np.ndarray:
    """Elliptic contour rotated by angle_deg."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle_rad = np.radians(angle_deg)
    x = 20 * np.cos(t)
    y = 8 * np.sin(t)
    cx = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    cy = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return np.stack([cx + 32, cy + 32], axis=1).astype(np.float64)


# ─── TestRotationEstimate ─────────────────────────────────────────────────────

class TestRotationEstimate:
    def test_basic_fields(self):
        e = RotationEstimate(angle_deg=15.0, confidence=0.8, method="pca")
        assert e.angle_deg == pytest.approx(15.0)
        assert e.confidence == pytest.approx(0.8)
        assert e.method == "pca"

    def test_params_default_empty(self):
        e = RotationEstimate(angle_deg=0.0, confidence=1.0, method="test")
        assert e.params == {}

    def test_params_stored(self):
        e = RotationEstimate(angle_deg=0.0, confidence=1.0, method="test",
                             params={"k": 42})
        assert e.params["k"] == 42

    def test_angle_zero_ok(self):
        e = RotationEstimate(angle_deg=0.0, confidence=1.0, method="pca")
        assert e.angle_deg == 0.0

    def test_angle_negative_ok(self):
        e = RotationEstimate(angle_deg=-45.0, confidence=0.5, method="pca")
        assert e.angle_deg == pytest.approx(-45.0)

    def test_confidence_zero_ok(self):
        e = RotationEstimate(angle_deg=0.0, confidence=0.0, method="pca")
        assert e.confidence == 0.0

    def test_confidence_one_ok(self):
        e = RotationEstimate(angle_deg=0.0, confidence=1.0, method="pca")
        assert e.confidence == 1.0


# ─── TestEstimateByPca ────────────────────────────────────────────────────────

class TestEstimateByPca:
    def test_returns_rotation_estimate(self):
        contour = _contour(50)
        e = estimate_by_pca(contour)
        assert isinstance(e, RotationEstimate)

    def test_method_pca(self):
        e = estimate_by_pca(_contour(50))
        assert e.method == "pca"

    def test_confidence_in_range(self):
        e = estimate_by_pca(_contour(50))
        assert 0.0 <= e.confidence <= 1.0

    def test_horizontal_ellipse_near_zero(self):
        # Ellipse major axis along x → angle ≈ 0
        e = estimate_by_pca(_contour(100, angle_deg=0.0))
        assert abs(e.angle_deg) < 30.0  # rough check

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            estimate_by_pca(np.array([[0.0, 0.0]]))

    def test_two_points_ok(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        e = estimate_by_pca(pts)
        assert isinstance(e, RotationEstimate)

    def test_3d_contour_shape(self):
        # OpenCV contour format (N,1,2)
        pts = _contour(50).reshape(-1, 1, 2)
        e = estimate_by_pca(pts)
        assert isinstance(e, RotationEstimate)

    def test_params_has_eigenvalues(self):
        e = estimate_by_pca(_contour(50))
        assert "eigenvalues" in e.params

    def test_elongated_high_confidence(self):
        # Very elongated ellipse → high PCA confidence
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        pts = np.stack([50 * np.cos(t), 2 * np.sin(t)], axis=1)
        e = estimate_by_pca(pts)
        assert e.confidence > 0.5


# ─── TestEstimateByMoments ────────────────────────────────────────────────────

class TestEstimateByMoments:
    def test_returns_rotation_estimate(self):
        e = estimate_by_moments(_rand_gray())
        assert isinstance(e, RotationEstimate)

    def test_method_moments(self):
        e = estimate_by_moments(_rand_gray())
        assert e.method == "moments"

    def test_confidence_in_range(self):
        e = estimate_by_moments(_rand_gray())
        assert 0.0 <= e.confidence <= 1.0

    def test_rgb_ok(self):
        e = estimate_by_moments(_rand_rgb())
        assert isinstance(e, RotationEstimate)

    def test_constant_image_raises(self):
        with pytest.raises(ValueError):
            estimate_by_moments(_gray(val=128))

    def test_params_has_moments(self):
        e = estimate_by_moments(_rand_gray())
        assert "mu20" in e.params
        assert "mu02" in e.params
        assert "mu11" in e.params

    def test_non_square_ok(self):
        img = _rand_gray(48, 80)
        e = estimate_by_moments(img)
        assert isinstance(e, RotationEstimate)

    def test_angle_in_range(self):
        e = estimate_by_moments(_rand_gray())
        assert -90.0 < e.angle_deg <= 90.0


# ─── TestEstimateByGradient ───────────────────────────────────────────────────

class TestEstimateByGradient:
    def test_returns_rotation_estimate(self):
        e = estimate_by_gradient(_rand_gray())
        assert isinstance(e, RotationEstimate)

    def test_method_gradient(self):
        e = estimate_by_gradient(_rand_gray())
        assert e.method == "gradient"

    def test_confidence_in_range(self):
        e = estimate_by_gradient(_rand_gray())
        assert 0.0 <= e.confidence <= 1.0

    def test_rgb_ok(self):
        e = estimate_by_gradient(_rand_rgb())
        assert isinstance(e, RotationEstimate)

    def test_n_bins_one_raises(self):
        with pytest.raises(ValueError):
            estimate_by_gradient(_rand_gray(), n_bins=1)

    def test_n_bins_two_ok(self):
        e = estimate_by_gradient(_rand_gray(), n_bins=2)
        assert isinstance(e, RotationEstimate)

    def test_custom_n_bins(self):
        e = estimate_by_gradient(_rand_gray(), n_bins=90)
        assert e.params["n_bins"] == 90

    def test_angle_in_range(self):
        e = estimate_by_gradient(_rand_gray())
        assert -90.0 < e.angle_deg <= 90.0

    def test_deterministic(self):
        img = _rand_gray()
        e1 = estimate_by_gradient(img)
        e2 = estimate_by_gradient(img)
        assert e1.angle_deg == pytest.approx(e2.angle_deg)


# ─── TestRefineRotation ───────────────────────────────────────────────────────

class TestRefineRotation:
    def test_returns_rotation_estimate(self):
        e = refine_rotation(_rand_gray(), initial_angle=0.0)
        assert isinstance(e, RotationEstimate)

    def test_method_refine(self):
        e = refine_rotation(_rand_gray(), initial_angle=5.0)
        assert e.method == "refine"

    def test_confidence_in_range(self):
        e = refine_rotation(_rand_gray(), initial_angle=0.0)
        assert 0.0 <= e.confidence <= 1.0

    def test_n_steps_one_raises(self):
        with pytest.raises(ValueError):
            refine_rotation(_rand_gray(), initial_angle=0.0, n_steps=1)

    def test_search_range_zero_raises(self):
        with pytest.raises(ValueError):
            refine_rotation(_rand_gray(), initial_angle=0.0, search_range=0.0)

    def test_search_range_neg_raises(self):
        with pytest.raises(ValueError):
            refine_rotation(_rand_gray(), initial_angle=0.0, search_range=-1.0)

    def test_n_steps_two_ok(self):
        e = refine_rotation(_rand_gray(), initial_angle=0.0, n_steps=2)
        assert isinstance(e, RotationEstimate)

    def test_params_stored(self):
        e = refine_rotation(_rand_gray(), initial_angle=5.0,
                            search_range=3.0, n_steps=10)
        assert e.params["initial_angle"] == pytest.approx(5.0)
        assert e.params["search_range"] == pytest.approx(3.0)
        assert e.params["n_steps"] == 10

    def test_rgb_ok(self):
        e = refine_rotation(_rand_rgb(), initial_angle=0.0)
        assert isinstance(e, RotationEstimate)


# ─── TestEstimateRotationPair ─────────────────────────────────────────────────

class TestEstimateRotationPair:
    def test_returns_tuple(self):
        r = estimate_rotation_pair(_rand_gray(), _rand_gray())
        assert isinstance(r, tuple)
        assert len(r) == 2

    def test_both_rotation_estimates(self):
        r = estimate_rotation_pair(_rand_gray(), _rand_gray())
        assert isinstance(r[0], RotationEstimate)
        assert isinstance(r[1], RotationEstimate)

    def test_method_gradient_default(self):
        r = estimate_rotation_pair(_rand_gray(), _rand_gray())
        assert r[0].method == "gradient"
        assert r[1].method == "gradient"

    def test_method_moments_ok(self):
        r = estimate_rotation_pair(_rand_gray(), _rand_gray(),
                                   method="moments")
        assert r[0].method == "moments"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            estimate_rotation_pair(_rand_gray(), _rand_gray(), method="pca")

    def test_rgb_pair_ok(self):
        r = estimate_rotation_pair(_rand_rgb(), _rand_rgb())
        assert isinstance(r[0], RotationEstimate)


# ─── TestBatchEstimateRotations ───────────────────────────────────────────────

class TestBatchEstimateRotations:
    def test_returns_list(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        result = batch_estimate_rotations(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_rand_gray(seed=i) for i in range(5)]
        assert len(batch_estimate_rotations(imgs)) == 5

    def test_empty_list(self):
        assert batch_estimate_rotations([]) == []

    def test_all_rotation_estimates(self):
        imgs = [_rand_gray(seed=i) for i in range(3)]
        for e in batch_estimate_rotations(imgs):
            assert isinstance(e, RotationEstimate)

    def test_method_moments(self):
        imgs = [_rand_gray(seed=i) for i in range(2)]
        result = batch_estimate_rotations(imgs, method="moments")
        assert all(e.method == "moments" for e in result)

    def test_invalid_method_raises(self):
        imgs = [_rand_gray()]
        with pytest.raises(ValueError):
            batch_estimate_rotations(imgs, method="unknown")

    def test_rgb_images_ok(self):
        imgs = [_rand_rgb(seed=i) for i in range(2)]
        result = batch_estimate_rotations(imgs)
        assert len(result) == 2
