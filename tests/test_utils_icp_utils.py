"""Tests for puzzle_reconstruction.utils.icp_utils"""
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


np.random.seed(42)


# ── centroid ─────────────────────────────────────────────────────────────────

def test_centroid_simple():
    pts = np.array([[0.0, 0.0], [2.0, 2.0]])
    c = centroid(pts)
    np.testing.assert_allclose(c, [1.0, 1.0])


def test_centroid_shape():
    pts = np.random.rand(10, 2)
    c = centroid(pts)
    assert c.shape == (2,)


def test_centroid_wrong_shape_raises():
    with pytest.raises(ValueError):
        centroid(np.array([1.0, 2.0, 3.0]))


def test_centroid_single_point():
    pts = np.array([[3.0, 7.0]])
    c = centroid(pts)
    np.testing.assert_allclose(c, [3.0, 7.0])


# ── center_points ────────────────────────────────────────────────────────────

def test_center_points_zero_mean():
    pts = np.random.rand(20, 2)
    centered, c = center_points(pts)
    np.testing.assert_allclose(centered.mean(axis=0), [0.0, 0.0], atol=1e-12)


def test_center_points_returns_centroid():
    pts = np.array([[1.0, 3.0], [3.0, 7.0]])
    centered, c = center_points(pts)
    np.testing.assert_allclose(c, [2.0, 5.0])


# ── scale_points ─────────────────────────────────────────────────────────────

def test_scale_points_rms_one():
    pts = np.random.rand(50, 2) * 100
    scaled, s = scale_points(pts)
    centered_scaled, _ = center_points(scaled)
    rms = float(np.sqrt(np.mean(np.sum(centered_scaled ** 2, axis=1))))
    assert rms == pytest.approx(1.0, abs=1e-6)


def test_scale_points_degenerate():
    pts = np.ones((10, 2))
    scaled, s = scale_points(pts)
    assert s == 1.0


def test_scale_points_positive_scale():
    pts = np.random.rand(10, 2)
    _, s = scale_points(pts)
    assert s > 0


# ── resample_uniform ─────────────────────────────────────────────────────────

def test_resample_uniform_output_shape():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = resample_uniform(pts, n=10)
    assert out.shape == (10, 2)


def test_resample_uniform_empty_pts():
    out = resample_uniform(np.empty((0, 2)), n=5)
    assert out.shape == (0, 2)


def test_resample_uniform_n_zero():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    out = resample_uniform(pts, n=0)
    assert out.shape == (0, 2)


def test_resample_uniform_single_point():
    pts = np.array([[5.0, 3.0]])
    out = resample_uniform(pts, n=4)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, [[5.0, 3.0]] * 4)


def test_resample_uniform_endpoints_preserved():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = resample_uniform(pts, n=3)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[-1], [2.0, 0.0], atol=1e-6)


# ── nearest_neighbours ───────────────────────────────────────────────────────

def test_nearest_neighbours_basic():
    # src[0]=[0,0] -> nearest tgt[0]=[0.1,0] (dist=0.1) vs tgt[1]=[5,0] (dist=5)
    # src[1]=[4,0] -> nearest tgt[1]=[5,0] (dist=1) vs tgt[0]=[0.1,0] (dist=3.9)
    src = np.array([[0.0, 0.0], [4.0, 0.0]])
    tgt = np.array([[0.1, 0.0], [5.0, 0.0]])
    idx, dists = nearest_neighbours(src, tgt)
    assert idx[0] == 0
    assert idx[1] == 1


def test_nearest_neighbours_shapes():
    src = np.random.rand(5, 2)
    tgt = np.random.rand(8, 2)
    idx, dists = nearest_neighbours(src, tgt)
    assert idx.shape == (5,)
    assert dists.shape == (5,)


def test_nearest_neighbours_empty():
    src = np.empty((0, 2))
    tgt = np.random.rand(5, 2)
    idx, dists = nearest_neighbours(src, tgt)
    assert len(idx) == 0
    assert len(dists) == 0


def test_nearest_neighbours_non_negative_distances():
    src = np.random.rand(10, 2)
    tgt = np.random.rand(10, 2)
    _, dists = nearest_neighbours(src, tgt)
    assert (dists >= 0).all()


# ── filter_correspondences ───────────────────────────────────────────────────

def test_filter_correspondences_no_filter():
    src = np.random.rand(5, 2)
    tgt = np.random.rand(5, 2)
    idx = np.arange(5)
    dists = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    s_out, t_out = filter_correspondences(src, tgt, idx, dists, max_dist=None)
    assert len(s_out) == 5


def test_filter_correspondences_with_max_dist():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tgt = np.array([[0.1, 0.0], [1.1, 0.0], [100.0, 0.0]])
    idx = np.arange(3)
    dists = np.array([0.1, 0.1, 98.0])
    s_out, t_out = filter_correspondences(src, tgt, idx, dists, max_dist=1.0)
    assert len(s_out) == 2


# ── svd_rotation ─────────────────────────────────────────────────────────────

def test_svd_rotation_identity():
    pts = np.random.rand(10, 2)
    R = svd_rotation(pts, pts)
    np.testing.assert_allclose(R, np.eye(2), atol=1e-8)


def test_svd_rotation_det_positive():
    src = np.random.rand(20, 2)
    tgt = src @ np.array([[0, -1], [1, 0]])  # 90-degree rotation
    R = svd_rotation(src, tgt)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


def test_svd_rotation_empty_returns_eye():
    R = svd_rotation(np.empty((0, 2)), np.empty((0, 2)))
    np.testing.assert_allclose(R, np.eye(2))


def test_svd_rotation_shape():
    src = np.random.rand(5, 2)
    tgt = np.random.rand(5, 2)
    R = svd_rotation(src, tgt)
    assert R.shape == (2, 2)


# ── compute_rmse ─────────────────────────────────────────────────────────────

def test_compute_rmse_identical():
    pts = np.random.rand(10, 2)
    assert compute_rmse(pts, pts) == pytest.approx(0.0, abs=1e-12)


def test_compute_rmse_known_value():
    src = np.array([[0.0, 0.0], [1.0, 0.0]])
    tgt = np.array([[0.0, 1.0], [1.0, 1.0]])
    assert compute_rmse(src, tgt) == pytest.approx(1.0)


def test_compute_rmse_empty():
    assert compute_rmse(np.empty((0, 2)), np.empty((0, 2))) == 0.0


def test_compute_rmse_non_negative():
    src = np.random.rand(20, 2)
    tgt = np.random.rand(20, 2)
    assert compute_rmse(src, tgt) >= 0.0


# ── rmse_after_transform ─────────────────────────────────────────────────────

def test_rmse_after_transform_identity():
    pts = np.random.rand(10, 2)
    R = np.eye(2)
    t = np.zeros(2)
    assert rmse_after_transform(pts, pts, R, t) == pytest.approx(0.0, abs=1e-12)


# ── has_converged ─────────────────────────────────────────────────────────────

def test_has_converged_true():
    assert has_converged(1.0, 1.0 + 1e-10, tol=1e-6) is True


def test_has_converged_false():
    assert has_converged(1.0, 0.5, tol=1e-6) is False


# ── compose_transforms ───────────────────────────────────────────────────────

def test_compose_transforms_identity():
    R1 = np.eye(2)
    t1 = np.zeros(2)
    R2, t2 = compose_transforms(R1, t1, R1, t1)
    np.testing.assert_allclose(R2, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(t2, np.zeros(2), atol=1e-10)


def test_compose_transforms_associativity():
    R1 = np.array([[0, -1], [1, 0]], dtype=float)
    t1 = np.array([1.0, 0.0])
    R2 = np.array([[0, 1], [-1, 0]], dtype=float)
    t2 = np.array([0.0, 1.0])
    R_out, t_out = compose_transforms(R1, t1, R2, t2)
    assert R_out.shape == (2, 2)
    assert t_out.shape == (2,)


# ── invert_transform ─────────────────────────────────────────────────────────

def test_invert_transform_round_trip():
    R = np.array([[0, -1], [1, 0]], dtype=float)
    t = np.array([2.0, 3.0])
    R_inv, t_inv = invert_transform(R, t)
    R_id, t_id = compose_transforms(R, t, R_inv, t_inv)
    np.testing.assert_allclose(R_id, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(t_id, np.zeros(2), atol=1e-10)


# ── transform_points ─────────────────────────────────────────────────────────

def test_transform_points_identity():
    pts = np.random.rand(10, 2)
    out = transform_points(pts, np.eye(2), np.zeros(2))
    np.testing.assert_allclose(out, pts, atol=1e-12)


def test_transform_points_translation():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    out = transform_points(pts, np.eye(2), np.array([3.0, 4.0]))
    np.testing.assert_allclose(out, [[3.0, 4.0], [4.0, 4.0]])


# ── align_to_first ───────────────────────────────────────────────────────────

def test_align_to_first_empty():
    assert align_to_first([]) == []


def test_align_to_first_single():
    cloud = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = align_to_first([cloud])
    assert len(result) == 1


def test_align_to_first_centroids_match():
    c1 = np.array([[0.0, 0.0], [2.0, 2.0]])
    c2 = np.array([[10.0, 10.0], [12.0, 12.0]])
    aligned = align_to_first([c1, c2])
    np.testing.assert_allclose(
        centroid(aligned[0]), centroid(aligned[1]), atol=1e-10
    )
