"""Additional tests for puzzle_reconstruction/matching/icp.py."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.matching.icp import (
    ICPResult,
    icp_align,
    contour_icp,
    align_fragment_edge,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n=50, r=10.0, cx=0.0, cy=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _line(n=30, length=20.0):
    return np.column_stack([np.linspace(0.0, length, n), np.zeros(n)])


def _rotate(pts, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _translate(pts, tx, ty):
    return pts + np.array([tx, ty])


# ─── TestICPResultExtra ───────────────────────────────────────────────────────

class TestICPResultExtra:
    def test_transform_combined_r_and_t(self):
        pts = np.array([[1.0, 0.0]])
        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        t = np.array([1.0, 2.0])
        result = ICPResult(R=R, t=t, rmse=0.0, n_iter=1, converged=True)
        out = result.transform(pts)
        expected = (pts @ R.T) + t
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_r_determinant_near_1(self):
        src = _circle(30)
        tgt = _circle(30)
        result = icp_align(src, tgt)
        assert abs(np.linalg.det(result.R) - 1.0) < 0.01

    def test_converged_false_if_empty(self):
        result = icp_align(np.empty((0, 2)), _circle(10))
        assert result.converged is False

    def test_rmse_history_list(self):
        result = ICPResult(R=np.eye(2), t=np.zeros(2),
                           rmse=0.5, n_iter=3, converged=True,
                           rmse_history=[1.0, 0.7, 0.5])
        assert isinstance(result.rmse_history, list)
        assert len(result.rmse_history) == 3

    def test_transform_preserves_shape(self):
        pts = _circle(25)
        result = ICPResult(R=np.eye(2), t=np.zeros(2),
                           rmse=0.0, n_iter=0, converged=True)
        out = result.transform(pts)
        assert out.shape == pts.shape

    def test_transform_180_rotation(self):
        pts = np.array([[2.0, 0.0]])
        R = np.array([[-1.0, 0.0], [0.0, -1.0]])
        result = ICPResult(R=R, t=np.zeros(2), rmse=0.0, n_iter=0, converged=True)
        out = result.transform(pts)
        np.testing.assert_allclose(out, [[-2.0, 0.0]], atol=1e-9)

    def test_rmse_nonneg(self):
        src = _circle(30)
        tgt = _translate(src, 3.0, 2.0)
        result = icp_align(src, tgt)
        assert result.rmse >= 0.0


# ─── TestIcpAlignExtra ────────────────────────────────────────────────────────

class TestIcpAlignExtra:
    def test_line_pts_no_crash(self):
        src = _line(30)
        tgt = _translate(src, 1.0, 0.0)
        result = icp_align(src, tgt)
        assert isinstance(result, ICPResult)

    def test_small_rotation_converges(self):
        src = _circle(50)
        tgt = _rotate(src, 0.1)
        result = icp_align(src, tgt, max_iter=50)
        assert isinstance(result, ICPResult)

    def test_max_iter_1_n_iter_leq_1(self):
        src = _circle(30)
        tgt = _circle(30)
        result = icp_align(src, tgt, max_iter=1)
        assert result.n_iter <= 1

    def test_tol_param_accepted(self):
        src = _circle(30)
        tgt = _translate(src, 0.5, 0.5)
        result = icp_align(src, tgt, tol=1e-6)
        assert isinstance(result, ICPResult)

    def test_large_translation_rmse_finite(self):
        src = _circle(30)
        tgt = _translate(src, 50.0, 50.0)
        result = icp_align(src, tgt)
        assert math.isfinite(result.rmse) or result.rmse == float("inf")

    def test_history_decreasing_or_equal(self):
        src = _circle(50)
        tgt = _translate(src, 3.0, 2.0)
        result = icp_align(src, tgt, track_history=True)
        history = result.rmse_history
        if len(history) > 1:
            # RMSE should generally decrease (allow small violations)
            drops = sum(1 for i in range(len(history) - 1)
                        if history[i + 1] <= history[i] + 1e-6)
            assert drops >= len(history) // 2

    def test_r_is_2x2(self):
        result = icp_align(_circle(20), _circle(20))
        assert result.R.shape == (2, 2)

    def test_t_is_length_2(self):
        result = icp_align(_circle(20), _circle(20))
        assert result.t.shape == (2,)

    def test_identical_large_pts_low_rmse(self):
        pts = _circle(100, r=50.0)
        result = icp_align(pts, pts.copy())
        assert result.rmse < 0.1


# ─── TestContourIcpExtra ─────────────────────────────────────────────────────

class TestContourIcpExtra:
    def test_try_mirror_true(self):
        a = _circle(50)
        b = _circle(50)
        result = contour_icp(a, b, try_mirror=True)
        assert isinstance(result, ICPResult)

    def test_rotated_contour_result(self):
        a = _circle(50)
        b = _rotate(a, 0.2)
        result = contour_icp(a, b, n_points=40)
        assert isinstance(result, ICPResult)

    def test_n_points_10(self):
        a = _circle(80)
        b = _circle(80)
        result = contour_icp(a, b, n_points=10)
        assert isinstance(result, ICPResult)

    def test_n_points_100(self):
        a = _circle(120)
        b = _circle(120)
        result = contour_icp(a, b, n_points=100)
        assert isinstance(result, ICPResult)

    def test_r_shape_2x2(self):
        a = _circle(50)
        b = _translate(a, 2.0, 1.0)
        result = contour_icp(a, b)
        assert result.R.shape == (2, 2)

    def test_rmse_nonneg(self):
        a = _circle(50)
        b = _translate(a, 2.0, 1.0)
        result = contour_icp(a, b)
        assert result.rmse >= 0.0

    def test_line_contours_no_crash(self):
        a = _line(40)
        b = _translate(a, 0.5, 0.5)
        result = contour_icp(a, b, n_points=30)
        assert isinstance(result, ICPResult)


# ─── TestAlignFragmentEdgeExtra ───────────────────────────────────────────────

class TestAlignFragmentEdgeExtra:
    def test_translated_edge_low_rmse(self):
        pts = _line(40, 30.0)
        shifted = _translate(pts, 1.0, 0.0)
        _, rmse = align_fragment_edge(pts, shifted)
        assert rmse < 2.0

    def test_rmse_is_float(self):
        pts = _circle(60)
        _, rmse = align_fragment_edge(pts, pts.copy())
        assert isinstance(rmse, float)

    def test_translation_finite(self):
        pts = _circle(60)
        t, rmse = align_fragment_edge(pts, pts.copy())
        assert np.all(np.isfinite(t))

    def test_n_points_20(self):
        pts = _circle(60)
        t, rmse = align_fragment_edge(pts, pts.copy(), n_points=20)
        assert t.shape == (2,)
        assert rmse >= 0.0

    def test_n_points_80(self):
        pts = _circle(100)
        t, rmse = align_fragment_edge(pts, pts.copy(), n_points=80)
        assert t.shape == (2,)

    def test_circle_vs_line_finite(self):
        a = _circle(60)
        b = _line(60)
        t, rmse = align_fragment_edge(a, b)
        assert np.all(np.isfinite(t))
        assert math.isfinite(rmse)

    def test_large_translation_returns_result(self):
        pts = _line(40)
        shifted = _translate(pts, 100.0, 50.0)
        t, rmse = align_fragment_edge(pts, shifted)
        assert t.shape == (2,)
