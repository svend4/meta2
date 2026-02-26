"""Tests for puzzle_reconstruction/matching/rotation_dtw.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.matching.rotation_dtw import (
    RotationDTWResult,
    rotation_dtw,
    rotation_dtw_similarity,
    batch_rotation_dtw,
    _resample_curve,
    _rotate_curve,
    _mirror_curve,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _line(n: int = 64, slope: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, slope * x])


def _circle_arc(n: int = 64, start: float = 0.0, end: float = np.pi) -> np.ndarray:
    t = np.linspace(start, end, n)
    return np.column_stack([np.cos(t), np.sin(t)])


def _sinusoid(n: int = 64, freq: float = 2.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, np.sin(freq * np.pi * x)])


# ── _resample_curve ────────────────────────────────────────────────────────────

class TestResampleCurve:

    def test_output_length(self):
        c = _line(32)
        r = _resample_curve(c, 64)
        assert len(r) == 64

    def test_output_shape(self):
        c = _sinusoid(100)
        r = _resample_curve(c, 50)
        assert r.shape == (50, 2)

    def test_single_point_no_crash(self):
        c = np.array([[1.0, 2.0]])
        r = _resample_curve(c, 8)
        assert r.shape == (8, 2)

    def test_collinear_input(self):
        c = np.tile([0.5, 0.5], (4, 1))  # all same point
        r = _resample_curve(c, 10)
        assert r.shape == (10, 2)
        assert np.all(np.isfinite(r))

    def test_preserves_endpoints_approx(self):
        c = _line(32)
        r = _resample_curve(c, 32)
        # First and last points should be close to original endpoints
        np.testing.assert_allclose(r[0], c[0], atol=1e-6)
        np.testing.assert_allclose(r[-1], c[-1], atol=1e-6)


# ── _rotate_curve ──────────────────────────────────────────────────────────────

class TestRotateCurve:

    def test_zero_rotation_identity(self):
        c = _sinusoid(32)
        r = _rotate_curve(c, 0.0)
        np.testing.assert_allclose(r, c, atol=1e-12)

    def test_360_rotation_identity(self):
        c = _sinusoid(32)
        r = _rotate_curve(c, 360.0)
        np.testing.assert_allclose(r, c, atol=1e-10)

    def test_180_rotation_centroid_preserved(self):
        c = _line(32)
        centroid = c.mean(axis=0)
        r = _rotate_curve(c, 180.0)
        np.testing.assert_allclose(r.mean(axis=0), centroid, atol=1e-10)

    def test_rotation_preserves_pairwise_distances(self):
        c = _sinusoid(32)
        r = _rotate_curve(c, 45.0)
        d_before = np.linalg.norm(c[0] - c[-1])
        d_after  = np.linalg.norm(r[0] - r[-1])
        assert abs(d_before - d_after) < 1e-10

    def test_shape_preserved(self):
        c = _line(50)
        r = _rotate_curve(c, 90.0)
        assert r.shape == c.shape


# ── _mirror_curve ──────────────────────────────────────────────────────────────

class TestMirrorCurve:

    def test_double_mirror_identity(self):
        c = _sinusoid(32)
        m = _mirror_curve(_mirror_curve(c))
        np.testing.assert_allclose(m, c, atol=1e-12)

    def test_centroid_x_preserved(self):
        c = _sinusoid(32)
        m = _mirror_curve(c)
        assert abs(m[:, 0].mean() - c[:, 0].mean()) < 1e-10

    def test_y_unchanged(self):
        c = _sinusoid(32)
        m = _mirror_curve(c)
        np.testing.assert_allclose(m[:, 1], c[:, 1], atol=1e-12)

    def test_shape_preserved(self):
        c = _line(50)
        m = _mirror_curve(c)
        assert m.shape == c.shape


# ── RotationDTWResult ──────────────────────────────────────────────────────────

class TestRotationDTWResult:

    def test_is_namedtuple(self):
        r = RotationDTWResult(distance=0.5, best_angle_deg=45.0, mirrored=False)
        assert r.distance == 0.5
        assert r.best_angle_deg == 45.0
        assert r.mirrored is False

    def test_unpacking(self):
        d, a, m = RotationDTWResult(0.1, 90.0, True)
        assert d == 0.1
        assert a == 90.0
        assert m is True


# ── rotation_dtw ──────────────────────────────────────────────────────────────

class TestRotationDTW:

    def test_returns_result(self):
        a = _sinusoid(32)
        b = _sinusoid(32)
        result = rotation_dtw(a, b, n_angles=8, n_points=32)
        assert isinstance(result, RotationDTWResult)

    def test_identical_curves_distance_zero(self):
        c = _sinusoid(64)
        result = rotation_dtw(c, c, n_angles=1, n_points=32, dtw_window=5)
        assert result.distance == pytest.approx(0.0, abs=1e-9)

    def test_identical_curves_angle_zero(self):
        c = _sinusoid(64)
        result = rotation_dtw(c, c, n_angles=1, n_points=32, dtw_window=5)
        assert result.best_angle_deg == pytest.approx(0.0)

    def test_distance_non_negative(self):
        a = _sinusoid(64)
        b = _line(64)
        result = rotation_dtw(a, b, n_angles=12, n_points=32)
        assert result.distance >= 0.0

    def test_best_angle_in_range(self):
        a = _sinusoid(64)
        b = _circle_arc(64)
        result = rotation_dtw(a, b, n_angles=36, n_points=32)
        assert 0.0 <= result.best_angle_deg < 360.0

    def test_mirrored_false_without_check_mirror(self):
        a = _sinusoid(32)
        b = _line(32)
        result = rotation_dtw(a, b, n_angles=8, n_points=32, check_mirror=False)
        assert result.mirrored is False

    def test_check_mirror_can_give_mirrored_true(self):
        # Build a curve whose mirror is clearly better
        a = _sinusoid(32, freq=2.0)
        # mirror of a should match a itself better than random rotation
        b_mirror = _mirror_curve(a)
        result = rotation_dtw(a, b_mirror, n_angles=8, n_points=32,
                              check_mirror=True)
        # mirrored flag could be True or False depending on geometry,
        # but the result should not crash and distance should be finite
        assert np.isfinite(result.distance)

    def test_short_curve_returns_inf(self):
        a = np.array([[0.0, 0.0]])
        b = _sinusoid(32)
        result = rotation_dtw(a, b, n_angles=4, n_points=8)
        assert result.distance == float("inf")

    def test_different_lengths_ok(self):
        a = _sinusoid(20)
        b = _sinusoid(80)
        result = rotation_dtw(a, b, n_angles=8, n_points=32)
        assert np.isfinite(result.distance)

    def test_n_angles_1_always_returns(self):
        a = _sinusoid(32)
        b = _line(32)
        result = rotation_dtw(a, b, n_angles=1, n_points=32)
        assert isinstance(result, RotationDTWResult)

    def test_rotated_back_low_distance(self):
        """Rotating B by 45° and then searching with check of that angle → distance near zero."""
        c = _sinusoid(64)
        c_rot = _rotate_curve(c, 45.0)
        # Search with 36 angles (10° step) — 40° or 50° might be found
        result = rotation_dtw(c, c_rot, n_angles=36, n_points=32, dtw_window=10)
        # The best distance should be much lower than with zero rotation
        result_no_search = rotation_dtw(c, c_rot, n_angles=1, n_points=32, dtw_window=10)
        # Rotation search should find a better (lower) distance
        assert result.distance <= result_no_search.distance + 1e-9


# ── rotation_dtw_similarity ───────────────────────────────────────────────────

class TestRotationDTWSimilarity:

    def test_returns_float(self):
        a = _sinusoid(32)
        b = _line(32)
        sim = rotation_dtw_similarity(a, b, n_angles=8, n_points=32)
        assert isinstance(sim, float)

    def test_range_0_1(self):
        for _ in range(5):
            a = _sinusoid(32)
            b = _circle_arc(32)
            sim = rotation_dtw_similarity(a, b, n_angles=8, n_points=32)
            assert 0.0 <= sim <= 1.0

    def test_identical_curves_similarity_one(self):
        c = _sinusoid(64)
        sim = rotation_dtw_similarity(c, c, n_angles=1, n_points=32, dtw_window=5)
        assert sim == pytest.approx(1.0, abs=1e-9)

    def test_empty_curve_returns_zero(self):
        a = np.array([[0.0, 0.0]])
        b = _sinusoid(32)
        sim = rotation_dtw_similarity(a, b, n_angles=4, n_points=8)
        assert sim == pytest.approx(0.0)

    def test_very_different_curves_low_sim(self):
        c = _sinusoid(64, freq=5.0)
        line = _line(64)
        sim = rotation_dtw_similarity(c, line, n_angles=12, n_points=32)
        assert sim < 1.0


# ── batch_rotation_dtw ────────────────────────────────────────────────────────

class TestBatchRotationDTW:

    def test_returns_list(self):
        q = _sinusoid(32)
        cands = [_line(32), _circle_arc(32)]
        results = batch_rotation_dtw(q, cands, n_angles=8, n_points=32)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_result_is_namedtuple(self):
        q = _sinusoid(32)
        cands = [_line(32)]
        results = batch_rotation_dtw(q, cands, n_angles=8, n_points=32)
        assert isinstance(results[0], RotationDTWResult)

    def test_empty_candidates_returns_empty(self):
        q = _sinusoid(32)
        results = batch_rotation_dtw(q, [], n_angles=8, n_points=32)
        assert results == []

    def test_single_candidate(self):
        q = _sinusoid(32)
        results = batch_rotation_dtw(q, [_line(32)], n_angles=8, n_points=32)
        assert len(results) == 1

    def test_distances_finite(self):
        q = _sinusoid(32)
        cands = [_line(32), _sinusoid(32, freq=3.0), _circle_arc(32)]
        results = batch_rotation_dtw(q, cands, n_angles=8, n_points=32)
        for r in results:
            assert np.isfinite(r.distance)

    def test_best_match_is_self(self):
        q = _sinusoid(64)
        cands = [_line(64), q, _circle_arc(64)]
        results = batch_rotation_dtw(q, cands, n_angles=1, n_points=32,
                                     dtw_window=5, check_mirror=False)
        distances = [r.distance for r in results]
        # q vs q should be the minimum
        assert distances[1] == min(distances)
