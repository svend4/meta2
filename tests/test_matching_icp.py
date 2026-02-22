"""Tests for puzzle_reconstruction/matching/icp.py"""
import pytest
import numpy as np

from puzzle_reconstruction.matching.icp import (
    ICPResult,
    icp_align,
    contour_icp,
    align_fragment_edge,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_circle_pts(n=50, r=10.0, cx=0.0, cy=0.0):
    """Generate N 2D points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack([x, y])


def make_line_pts(n=30, length=20.0):
    """Generate N 2D points on a horizontal line."""
    x = np.linspace(0.0, length, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def rotate_pts(pts, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def translate_pts(pts, tx, ty):
    return pts + np.array([tx, ty])


# ─── ICPResult ────────────────────────────────────────────────────────────────

class TestICPResult:
    def test_basic_creation(self):
        result = ICPResult(
            R=np.eye(2),
            t=np.zeros(2),
            rmse=1.5,
            n_iter=10,
            converged=True,
        )
        assert result.rmse == pytest.approx(1.5)
        assert result.n_iter == 10
        assert result.converged is True

    def test_default_rmse_history_empty(self):
        result = ICPResult(
            R=np.eye(2), t=np.zeros(2),
            rmse=0.0, n_iter=0, converged=False,
        )
        assert result.rmse_history == []

    def test_custom_rmse_history(self):
        result = ICPResult(
            R=np.eye(2), t=np.zeros(2),
            rmse=0.1, n_iter=3, converged=True,
            rmse_history=[1.0, 0.5, 0.1],
        )
        assert len(result.rmse_history) == 3

    def test_transform_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ICPResult(
            R=np.eye(2), t=np.zeros(2),
            rmse=0.0, n_iter=0, converged=True,
        )
        transformed = result.transform(pts)
        np.testing.assert_allclose(transformed, pts, atol=1e-10)

    def test_transform_translation(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        t = np.array([3.0, 4.0])
        result = ICPResult(
            R=np.eye(2), t=t,
            rmse=0.0, n_iter=0, converged=True,
        )
        transformed = result.transform(pts)
        np.testing.assert_allclose(transformed, pts + t, atol=1e-10)

    def test_transform_rotation(self):
        pts = np.array([[1.0, 0.0]])
        angle = np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        result = ICPResult(
            R=R, t=np.zeros(2),
            rmse=0.0, n_iter=0, converged=True,
        )
        transformed = result.transform(pts)
        np.testing.assert_allclose(transformed, [[0.0, 1.0]], atol=1e-10)

    def test_transform_output_shape(self):
        pts = make_circle_pts(20)
        result = ICPResult(
            R=np.eye(2), t=np.zeros(2),
            rmse=0.0, n_iter=0, converged=True,
        )
        transformed = result.transform(pts)
        assert transformed.shape == pts.shape


# ─── icp_align ────────────────────────────────────────────────────────────────

class TestIcpAlign:
    def test_returns_icp_result(self):
        src = make_circle_pts(30)
        tgt = make_circle_pts(30)
        result = icp_align(src, tgt)
        assert isinstance(result, ICPResult)

    def test_identical_points_low_rmse(self):
        pts = make_circle_pts(40)
        result = icp_align(pts, pts.copy())
        assert result.rmse < 0.1

    def test_pure_translation_converges(self):
        src = make_circle_pts(40)
        tgt = translate_pts(src, 5.0, 3.0)
        result = icp_align(src, tgt)
        assert result.rmse < 0.5

    def test_empty_source_returns_inf(self):
        result = icp_align(np.empty((0, 2)), make_circle_pts(10))
        assert result.rmse == float("inf")
        assert result.converged is False

    def test_empty_target_returns_inf(self):
        result = icp_align(make_circle_pts(10), np.empty((0, 2)))
        assert result.rmse == float("inf")
        assert result.converged is False

    def test_empty_returns_zero_iterations(self):
        result = icp_align(np.empty((0, 2)), np.empty((0, 2)))
        assert result.n_iter == 0

    def test_r_shape(self):
        src = make_circle_pts(30)
        tgt = make_circle_pts(30)
        result = icp_align(src, tgt)
        assert result.R.shape == (2, 2)

    def test_t_shape(self):
        src = make_circle_pts(30)
        tgt = make_circle_pts(30)
        result = icp_align(src, tgt)
        assert result.t.shape == (2,)

    def test_n_iter_positive(self):
        src = make_circle_pts(30)
        tgt = make_circle_pts(30)
        result = icp_align(src, tgt)
        assert result.n_iter >= 1

    def test_max_iter_respected(self):
        src = make_circle_pts(30)
        tgt = make_circle_pts(30)
        result = icp_align(src, tgt, max_iter=3)
        assert result.n_iter <= 3

    def test_track_history(self):
        src = make_circle_pts(30)
        tgt = translate_pts(src, 2.0, 1.0)
        result = icp_align(src, tgt, track_history=True)
        assert len(result.rmse_history) == result.n_iter

    def test_no_history_by_default(self):
        src = make_circle_pts(30)
        tgt = make_circle_pts(30)
        result = icp_align(src, tgt, track_history=False)
        assert result.rmse_history == []

    def test_init_t_overrides_centroid_alignment(self):
        src = make_circle_pts(30)
        tgt = translate_pts(src, 5.0, 5.0)
        result = icp_align(src, tgt, init_t=np.array([5.0, 5.0]))
        assert result.rmse < 1.0

    def test_converged_flag_set(self):
        src = make_circle_pts(50)
        result = icp_align(src, src.copy(), max_iter=100, tol=1e-3)
        assert result.converged is True


# ─── contour_icp ──────────────────────────────────────────────────────────────

class TestContourIcp:
    def test_returns_icp_result(self):
        a = make_circle_pts(50)
        b = make_circle_pts(50)
        result = contour_icp(a, b)
        assert isinstance(result, ICPResult)

    def test_identical_contours_low_rmse(self):
        pts = make_circle_pts(60)
        result = contour_icp(pts, pts.copy())
        assert result.rmse < 1.0

    def test_translated_contour(self):
        a = make_circle_pts(60)
        b = translate_pts(a, 3.0, 2.0)
        result = contour_icp(a, b, n_points=50)
        assert result.rmse < 2.0

    def test_r_shape(self):
        a = make_circle_pts(50)
        b = make_circle_pts(50)
        result = contour_icp(a, b)
        assert result.R.shape == (2, 2)

    def test_try_mirror_false(self):
        a = make_circle_pts(50)
        b = make_circle_pts(50)
        result = contour_icp(a, b, try_mirror=False)
        assert isinstance(result, ICPResult)

    def test_custom_n_points(self):
        a = make_circle_pts(80)
        b = make_circle_pts(80)
        result = contour_icp(a, b, n_points=20)
        assert isinstance(result, ICPResult)

    def test_line_contours(self):
        a = make_line_pts(40)
        b = translate_pts(a, 1.0, 0.0)
        result = contour_icp(a, b, n_points=30)
        assert isinstance(result, ICPResult)


# ─── align_fragment_edge ──────────────────────────────────────────────────────

class TestAlignFragmentEdge:
    def test_returns_tuple(self):
        a = make_circle_pts(60)
        b = make_circle_pts(60)
        result = align_fragment_edge(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_translation_shape(self):
        a = make_circle_pts(60)
        b = make_circle_pts(60)
        t, rmse = align_fragment_edge(a, b)
        assert t.shape == (2,)

    def test_rmse_nonneg(self):
        a = make_circle_pts(60)
        b = make_circle_pts(60)
        _, rmse = align_fragment_edge(a, b)
        assert rmse >= 0.0

    def test_identical_edges_low_rmse(self):
        pts = make_line_pts(40, 30.0)
        _, rmse = align_fragment_edge(pts, pts.copy())
        assert rmse < 1.0

    def test_custom_n_points(self):
        a = make_circle_pts(60)
        b = make_circle_pts(60)
        t, rmse = align_fragment_edge(a, b, n_points=40)
        assert t.shape == (2,)
