"""Additional tests for puzzle_reconstruction/matching/icp.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.matching.icp import (
    ICPResult,
    icp_align,
    contour_icp,
    align_fragment_edge,
    _nearest_neighbors,
    _best_fit_transform,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=40, r=1.0, cx=0.0, cy=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _line(n=20, length=10.0):
    return np.column_stack([np.linspace(0.0, length, n), np.zeros(n)])


def _rotate(pts, angle):
    c, s = math.cos(angle), math.sin(angle)
    return pts @ np.array([[c, -s], [s, c]]).T


# ─── TestICPResultExtra2 ──────────────────────────────────────────────────────

class TestICPResultExtra2:
    def test_identity_leaves_pts_unchanged(self):
        pts = _circle(20)
        res = ICPResult(R=np.eye(2), t=np.zeros(2), rmse=0.0, n_iter=0, converged=True)
        np.testing.assert_allclose(res.transform(pts), pts, atol=1e-12)

    def test_n_iter_zero_ok(self):
        res = ICPResult(R=np.eye(2), t=np.zeros(2), rmse=0.0, n_iter=0, converged=True)
        assert res.n_iter == 0

    def test_converged_false_stored(self):
        res = ICPResult(R=np.eye(2), t=np.zeros(2), rmse=5.0, n_iter=50, converged=False)
        assert res.converged is False

    def test_large_translation_transform(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        t = np.array([1000.0, -500.0])
        res = ICPResult(R=np.eye(2), t=t, rmse=0.0, n_iter=1, converged=True)
        out = res.transform(pts)
        np.testing.assert_allclose(out[0], t, atol=1e-9)

    def test_rmse_history_empty_by_default(self):
        res = ICPResult(R=np.eye(2), t=np.zeros(2), rmse=0.0, n_iter=0, converged=True)
        assert res.rmse_history == []

    def test_transform_single_point(self):
        pts = np.array([[3.0, 4.0]])
        res = ICPResult(R=np.eye(2), t=np.array([1.0, 2.0]), rmse=0.0, n_iter=1, converged=True)
        out = res.transform(pts)
        np.testing.assert_allclose(out, [[4.0, 6.0]], atol=1e-9)


# ─── TestICPAlignExtra2 ───────────────────────────────────────────────────────

class TestICPAlignExtra2:
    def test_identical_circles_converges(self):
        pts = _circle(50)
        res = icp_align(pts, pts, max_iter=20)
        assert res.converged

    def test_rmse_zero_for_identical(self):
        pts = _circle(30)
        res = icp_align(pts, pts.copy(), max_iter=50)
        assert res.rmse < 0.05

    def test_both_empty_converged_false(self):
        res = icp_align(np.empty((0, 2)), np.empty((0, 2)))
        assert res.converged is False

    def test_max_iter_1_n_iter_leq_1(self):
        pts = _circle(20)
        res = icp_align(pts, pts, max_iter=1)
        assert res.n_iter <= 1

    def test_r_shape(self):
        pts = _circle(30)
        res = icp_align(pts, pts + np.array([1.0, 0.5]))
        assert res.R.shape == (2, 2)

    def test_t_shape(self):
        pts = _circle(30)
        res = icp_align(pts, pts + np.array([1.0, 0.5]))
        assert res.t.shape == (2,)

    def test_init_r_identity_ok(self):
        pts = _circle(30)
        tgt = pts + np.array([0.5, 0.3])
        res = icp_align(pts, tgt, init_R=np.eye(2))
        assert isinstance(res, ICPResult)

    def test_small_rotation_low_rmse(self):
        pts = _circle(50, r=5.0)
        tgt = _rotate(pts, 0.05)
        res = icp_align(pts, tgt, max_iter=100)
        assert res.rmse < 0.5

    def test_history_length_equals_n_iter(self):
        pts = _circle(40)
        tgt = pts + np.array([2.0, 1.0])
        res = icp_align(pts, tgt, max_iter=10, track_history=True)
        assert len(res.rmse_history) == res.n_iter

    def test_line_pts_no_crash(self):
        src = _line(20)
        tgt = src + np.array([0.5, 0.0])
        res = icp_align(src, tgt)
        assert isinstance(res, ICPResult)

    def test_large_cloud_runs(self):
        pts = _circle(200, r=10.0)
        res = icp_align(pts, pts.copy(), max_iter=5)
        assert isinstance(res, ICPResult)


# ─── TestContourICPExtra2 ─────────────────────────────────────────────────────

class TestContourICPExtra2:
    def test_square_vs_circle_no_crash(self):
        circle = _circle(40)
        line = _line(40, length=20.0)
        res = contour_icp(circle, line, n_points=30)
        assert isinstance(res, ICPResult)

    def test_n_points_5(self):
        pts = _circle(60)
        res = contour_icp(pts, pts.copy(), n_points=5)
        assert isinstance(res, ICPResult)

    def test_n_points_200(self):
        pts = _circle(50)
        res = contour_icp(pts, pts.copy(), n_points=200)
        assert isinstance(res, ICPResult)

    def test_max_iter_1(self):
        pts = _circle(40)
        res = contour_icp(pts, pts.copy(), n_points=20, max_iter=1)
        assert isinstance(res, ICPResult)

    def test_rmse_nonneg(self):
        pts = _circle(40)
        tgt = pts + np.array([1.0, 0.5])
        res = contour_icp(pts, tgt, n_points=30)
        assert res.rmse >= 0.0

    def test_R_det_near_1(self):
        pts = _circle(40)
        tgt = _rotate(pts, 0.3)
        res = contour_icp(pts, tgt, n_points=40)
        assert abs(np.linalg.det(res.R) - 1.0) < 0.05


# ─── TestAlignFragmentEdgeExtra2 ─────────────────────────────────────────────

class TestAlignFragmentEdgeExtra2:
    def test_line_vs_line_small_shift(self):
        pts = _line(30)
        tgt = pts + np.array([0.2, 0.0])
        t, rmse = align_fragment_edge(pts, tgt)
        assert rmse < 1.0

    def test_t_finite(self):
        pts = _circle(50)
        t, rmse = align_fragment_edge(pts, pts.copy())
        assert np.all(np.isfinite(t))

    def test_returns_tuple(self):
        pts = _circle(30)
        out = align_fragment_edge(pts, pts.copy())
        assert len(out) == 2

    def test_large_offset_finite_rmse(self):
        pts = _circle(40)
        tgt = pts + np.array([200.0, 150.0])
        t, rmse = align_fragment_edge(pts, tgt)
        assert math.isfinite(rmse)

    def test_n_points_5(self):
        pts = _circle(50)
        t, rmse = align_fragment_edge(pts, pts.copy(), n_points=5)
        assert t.shape == (2,)
        assert rmse >= 0.0


# ─── TestNearestNeighborsExtra ────────────────────────────────────────────────

class TestNearestNeighborsExtra:
    def test_single_source_single_target(self):
        src = np.array([[0.0, 0.0]])
        tgt = np.array([[1.0, 0.0]])
        idx = _nearest_neighbors(src, tgt)
        assert idx[0] == 0

    def test_output_dtype_int(self):
        src = _circle(10)
        tgt = _circle(10)
        idx = _nearest_neighbors(src, tgt)
        assert idx.dtype in (np.int32, np.int64, np.intp)

    def test_all_map_to_first_when_one_target(self):
        src = _circle(20)
        tgt = np.array([[0.0, 0.0]])
        idx = _nearest_neighbors(src, tgt)
        assert np.all(idx == 0)

    def test_large_cloud_runs(self):
        src = _circle(200)
        tgt = _circle(200)
        idx = _nearest_neighbors(src, tgt)
        assert idx.shape == (200,)


# ─── TestBestFitTransformExtra ────────────────────────────────────────────────

class TestBestFitTransformExtra:
    def test_small_rotation_recovered(self):
        pts = _circle(50, r=5.0)
        angle = 0.2
        tgt = _rotate(pts, angle)
        R, t = _best_fit_transform(pts, tgt)
        assert abs(np.linalg.det(R) - 1.0) < 0.01

    def test_t_is_2d(self):
        pts = _circle(30)
        _, t = _best_fit_transform(pts, pts)
        assert t.shape == (2,)

    def test_R_orthogonal(self):
        pts = _circle(20)
        tgt = pts + np.array([3.0, 1.0])
        R, _ = _best_fit_transform(pts, tgt)
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-8)

    def test_two_point_identity(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        R, t = _best_fit_transform(pts, pts)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-8)
        np.testing.assert_allclose(t, np.zeros(2), atol=1e-8)
