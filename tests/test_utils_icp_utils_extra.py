"""Extra tests for puzzle_reconstruction/utils/icp_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.utils.icp_utils import (
    ICPConfig,
    centroid,
    center_points,
    scale_points,
    resample_uniform,
    nearest_neighbours,
    filter_correspondences,
    svd_rotation,
    svd_translation,
    compute_rmse,
    rmse_after_transform,
    has_converged,
    compose_transforms,
    invert_transform,
    PairAlignResult,
    batch_nearest_neighbours,
    transform_points,
    align_to_first,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pts(n=10) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, 2))


def _line(n=8) -> np.ndarray:
    return np.column_stack([np.linspace(0, 1, n), np.zeros(n)])


# ─── ICPConfig ────────────────────────────────────────────────────────────────

class TestICPConfigExtra:
    def test_default_max_iter(self):
        assert ICPConfig().max_iter == 50

    def test_default_tol(self):
        assert ICPConfig().tol == pytest.approx(1e-5)

    def test_default_allow_reflection(self):
        assert ICPConfig().allow_reflection is False

    def test_custom_values(self):
        cfg = ICPConfig(max_iter=100, tol=1e-3, allow_reflection=True)
        assert cfg.max_iter == 100
        assert cfg.tol == pytest.approx(1e-3)


# ─── centroid ─────────────────────────────────────────────────────────────────

class TestCentroidExtra:
    def test_returns_2d_array(self):
        c = centroid(_pts())
        assert c.shape == (2,)

    def test_value_correct(self):
        pts = np.array([[0.0, 0.0], [2.0, 2.0]])
        c = centroid(pts)
        assert c == pytest.approx([1.0, 1.0])

    def test_non_nx2_raises(self):
        with pytest.raises(ValueError):
            centroid(np.array([1.0, 2.0]))


# ─── center_points ────────────────────────────────────────────────────────────

class TestCenterPointsExtra:
    def test_returns_tuple(self):
        result = center_points(_pts())
        assert isinstance(result, tuple) and len(result) == 2

    def test_centered_mean_zero(self):
        centered, _ = center_points(_pts())
        assert np.abs(centered.mean(axis=0)).max() < 1e-9

    def test_centroid_returned(self):
        pts = np.array([[2.0, 4.0], [4.0, 6.0]])
        _, c = center_points(pts)
        assert c == pytest.approx([3.0, 5.0])


# ─── scale_points ─────────────────────────────────────────────────────────────

class TestScalePointsExtra:
    def test_returns_tuple(self):
        result = scale_points(_pts())
        assert isinstance(result, tuple) and len(result) == 2

    def test_scale_positive(self):
        _, s = scale_points(_pts())
        assert s > 0.0

    def test_degenerate_returns_one(self):
        pts = np.array([[1.0, 1.0], [1.0, 1.0]])
        _, s = scale_points(pts)
        assert s == pytest.approx(1.0)


# ─── resample_uniform ─────────────────────────────────────────────────────────

class TestResampleUniformExtra:
    def test_output_length(self):
        result = resample_uniform(_line(8), 16)
        assert len(result) == 16

    def test_empty_input(self):
        result = resample_uniform(np.empty((0, 2)), 5)
        assert len(result) == 0

    def test_single_point(self):
        result = resample_uniform(np.array([[1.0, 2.0]]), 4)
        assert result.shape == (4, 2)


# ─── nearest_neighbours ───────────────────────────────────────────────────────

class TestNearestNeighboursExtra:
    def test_returns_tuple(self):
        src = _pts(5)
        tgt = _pts(8)
        result = nearest_neighbours(src, tgt)
        assert isinstance(result, tuple) and len(result) == 2

    def test_indices_shape(self):
        src = _pts(5)
        tgt = _pts(8)
        idx, _ = nearest_neighbours(src, tgt)
        assert idx.shape == (5,)

    def test_distances_nonneg(self):
        _, dists = nearest_neighbours(_pts(5), _pts(8))
        assert (dists >= 0).all()

    def test_empty_src(self):
        idx, dists = nearest_neighbours(np.empty((0, 2)), _pts(5))
        assert len(idx) == 0


# ─── compute_rmse ─────────────────────────────────────────────────────────────

class TestComputeRmseExtra:
    def test_identical_is_zero(self):
        pts = _pts()
        assert compute_rmse(pts, pts) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(compute_rmse(_pts(), _pts()), float)

    def test_empty_is_zero(self):
        assert compute_rmse(np.empty((0, 2)), np.empty((0, 2))) == pytest.approx(0.0)


# ─── has_converged ────────────────────────────────────────────────────────────

class TestHasConvergedExtra:
    def test_small_diff_converged(self):
        assert has_converged(1.0, 1.0 + 1e-8, tol=1e-5) is True

    def test_large_diff_not_converged(self):
        assert has_converged(1.0, 0.5, tol=1e-5) is False


# ─── svd_rotation ─────────────────────────────────────────────────────────────

class TestSvdRotationExtra:
    def test_returns_2x2(self):
        R = svd_rotation(_pts(), _pts())
        assert R.shape == (2, 2)

    def test_det_positive(self):
        R = svd_rotation(_pts(), _pts())
        assert np.linalg.det(R) > 0

    def test_empty_returns_identity(self):
        R = svd_rotation(np.empty((0, 2)), np.empty((0, 2)))
        assert R == pytest.approx(np.eye(2))


# ─── compose and invert transforms ────────────────────────────────────────────

class TestTransformsExtra:
    def test_compose_identity(self):
        I = np.eye(2)
        z = np.zeros(2)
        R, t = compose_transforms(I, z, I, z)
        assert R == pytest.approx(I) and t == pytest.approx(z)

    def test_invert_identity(self):
        I = np.eye(2)
        z = np.zeros(2)
        R_inv, t_inv = invert_transform(I, z)
        assert R_inv == pytest.approx(I) and t_inv == pytest.approx(z)


# ─── transform_points ─────────────────────────────────────────────────────────

class TestTransformPointsExtra:
    def test_identity_unchanged(self):
        pts = _pts()
        result = transform_points(pts, np.eye(2), np.zeros(2))
        assert result == pytest.approx(pts)

    def test_translation(self):
        pts = np.array([[1.0, 2.0]])
        result = transform_points(pts, np.eye(2), np.array([3.0, 4.0]))
        assert np.allclose(result, [[4.0, 6.0]])


# ─── align_to_first ───────────────────────────────────────────────────────────

class TestAlignToFirstExtra:
    def test_returns_same_count(self):
        clouds = [_pts(5), _pts(5)]
        result = align_to_first(clouds)
        assert len(result) == 2

    def test_empty_input(self):
        assert align_to_first([]) == []

    def test_first_cloud_centroid_preserved(self):
        clouds = [_pts(5), _pts(5)]
        anchor = centroid(clouds[0])
        result = align_to_first(clouds)
        assert centroid(result[0]) == pytest.approx(anchor)
