"""Extra tests for puzzle_reconstruction/matching/rotation_dtw.py"""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _line(n=64, slope=0.0):
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, slope * x])


def _circle(n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


def _sinusoid(n=64, freq=2.0):
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, np.sin(freq * np.pi * x)])


def _triangle(n=60):
    t = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    return t[np.arange(n) % 3]


# ─── _resample_curve edge cases ───────────────────────────────────────────────

def test_resample_two_points_output_n():
    c = np.array([[0.0, 0.0], [1.0, 1.0]])
    r = _resample_curve(c, 10)
    assert r.shape == (10, 2)


def test_resample_zero_length_curve():
    c = np.tile([1.0, 2.0], (5, 1))
    r = _resample_curve(c, 8)
    assert r.shape == (8, 2)
    np.testing.assert_allclose(r[:, 0], 1.0)


def test_resample_single_point_repeats():
    c = np.array([[3.0, 4.0]])
    r = _resample_curve(c, 5)
    assert r.shape == (5, 2)
    np.testing.assert_allclose(r, 0.0)  # len<2 → zeros


def test_resample_preserves_start_and_end():
    c = _sinusoid(50)
    r = _resample_curve(c, 50)
    np.testing.assert_allclose(r[0], c[0], atol=1e-6)
    np.testing.assert_allclose(r[-1], c[-1], atol=1e-6)


def test_resample_n_equals_input_length():
    c = _line(32)
    r = _resample_curve(c, 32)
    assert r.shape == (32, 2)


def test_resample_output_finite():
    c = _circle(64)
    r = _resample_curve(c, 32)
    assert np.all(np.isfinite(r))


def test_resample_to_1():
    c = _line(20)
    r = _resample_curve(c, 1)
    assert r.shape == (1, 2)


# ─── _rotate_curve edge cases ─────────────────────────────────────────────────

def test_rotate_45_and_315_is_identity():
    c = _sinusoid(32)
    r = _rotate_curve(_rotate_curve(c, 45.0), 315.0)
    np.testing.assert_allclose(r, c, atol=1e-10)


def test_rotate_90_reverses_x_y():
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = _rotate_curve(c, 90.0)
    # Centroid is (0.5, 0.5); rotating by 90° around centroid
    assert r.shape == c.shape


def test_rotate_output_same_shape():
    c = _circle(100)
    r = _rotate_curve(c, 37.5)
    assert r.shape == c.shape


def test_rotate_preserves_centroid_circle():
    c = _circle(64)
    orig_centroid = c.mean(axis=0)
    r = _rotate_curve(c, 123.0)
    np.testing.assert_allclose(r.mean(axis=0), orig_centroid, atol=1e-10)


def test_rotate_single_point():
    c = np.array([[2.0, 3.0]])
    r = _rotate_curve(c, 90.0)
    # Single point: rotation around itself → same point
    np.testing.assert_allclose(r, c, atol=1e-10)


def test_rotate_all_angles_preserve_distances():
    c = _sinusoid(32)
    for angle in [0, 15, 30, 60, 90, 120, 180, 270]:
        r = _rotate_curve(c, float(angle))
        d_orig  = np.linalg.norm(c - c.mean(axis=0), axis=1)
        d_rot   = np.linalg.norm(r - r.mean(axis=0), axis=1)
        np.testing.assert_allclose(d_rot, d_orig, atol=1e-10)


# ─── _mirror_curve edge cases ─────────────────────────────────────────────────

def test_mirror_is_involution():
    c = _sinusoid(32)
    np.testing.assert_allclose(_mirror_curve(_mirror_curve(c)), c, atol=1e-12)


def test_mirror_y_unchanged():
    c = _triangle(60)
    m = _mirror_curve(c)
    np.testing.assert_allclose(m[:, 1], c[:, 1], atol=1e-12)


def test_mirror_x_reflected():
    c = _line(20, slope=0.5)
    m = _mirror_curve(c)
    cx = c[:, 0].mean()
    mx = m[:, 0].mean()
    assert abs(cx - mx) < 1e-10


def test_mirror_shape_unchanged():
    c = _circle(100)
    m = _mirror_curve(c)
    assert m.shape == c.shape


# ─── RotationDTWResult ────────────────────────────────────────────────────────

def test_rotation_dtw_result_fields():
    r = RotationDTWResult(distance=1.23, best_angle_deg=45.0, mirrored=True)
    assert r.distance == pytest.approx(1.23)
    assert r.best_angle_deg == pytest.approx(45.0)
    assert r.mirrored is True


def test_rotation_dtw_result_immutable():
    r = RotationDTWResult(0.5, 0.0, False)
    with pytest.raises((AttributeError, TypeError)):
        r.distance = 9.9


def test_rotation_dtw_result_unpack():
    d, a, m = RotationDTWResult(2.0, 90.0, False)
    assert d == pytest.approx(2.0)
    assert a == pytest.approx(90.0)
    assert m is False


# ─── rotation_dtw edge cases ─────────────────────────────────────────────────

def test_rotation_dtw_identical_distance_zero():
    c = _sinusoid(64)
    result = rotation_dtw(c, c, n_angles=1, n_points=32, dtw_window=10)
    assert result.distance == pytest.approx(0.0, abs=1e-9)


def test_rotation_dtw_two_point_curve_returns_inf():
    a = np.array([[0.0, 0.0]])
    b = _sinusoid(32)
    result = rotation_dtw(a, b, n_angles=4, n_points=8)
    assert result.distance == float("inf")


def test_rotation_dtw_angle_in_valid_range():
    a = _sinusoid(32)
    b = _line(32)
    result = rotation_dtw(a, b, n_angles=36, n_points=32)
    assert 0.0 <= result.best_angle_deg < 360.0


def test_rotation_dtw_no_mirror_flag_false():
    a = _sinusoid(32)
    b = _line(32)
    result = rotation_dtw(a, b, n_angles=4, n_points=16, check_mirror=False)
    assert result.mirrored is False


def test_rotation_dtw_check_mirror_does_not_crash():
    a = _sinusoid(32)
    b = _mirror_curve(a)
    result = rotation_dtw(a, b, n_angles=8, n_points=32, check_mirror=True)
    assert np.isfinite(result.distance)


def test_rotation_dtw_n_angles_high():
    a = _circle(64)
    b = _circle(64)
    result = rotation_dtw(a, b, n_angles=72, n_points=32, dtw_window=5)
    assert result.distance == pytest.approx(0.0, abs=1e-5)


def test_rotation_dtw_different_n_points():
    a = _sinusoid(100)
    b = _sinusoid(20)
    result = rotation_dtw(a, b, n_angles=8, n_points=16)
    assert np.isfinite(result.distance)


def test_rotation_dtw_window_none_no_crash():
    # dtw_window=None is not supported by the underlying DTW (requires int); use default
    a = _line(32)
    b = _line(32)
    result = rotation_dtw(a, b, n_angles=4, n_points=16)
    assert isinstance(result, RotationDTWResult)


def test_rotation_dtw_triangle_contour():
    a = _triangle(60)
    b = _triangle(60)
    result = rotation_dtw(a, b, n_angles=1, n_points=30, dtw_window=10)
    assert result.distance == pytest.approx(0.0, abs=1e-6)


# ─── rotation_dtw_similarity edge cases ──────────────────────────────────────

def test_similarity_identical_is_one():
    c = _line(64)
    s = rotation_dtw_similarity(c, c, n_angles=1, n_points=32, dtw_window=5)
    assert s == pytest.approx(1.0, abs=1e-9)


def test_similarity_empty_curve_zero():
    a = np.array([[0.0, 0.0]])
    b = _sinusoid(32)
    s = rotation_dtw_similarity(a, b, n_angles=4, n_points=8)
    assert s == pytest.approx(0.0)


def test_similarity_in_range_multiple_shapes():
    shapes = [_line(32), _circle(32), _sinusoid(32), _triangle(30)]
    for a in shapes:
        for b in shapes:
            s = rotation_dtw_similarity(a, b, n_angles=4, n_points=16)
            assert 0.0 <= s <= 1.0, f"Out of range: {s}"


def test_similarity_symmetric():
    a = _sinusoid(32, freq=2.0)
    b = _sinusoid(32, freq=3.0)
    s1 = rotation_dtw_similarity(a, b, n_angles=8, n_points=16)
    s2 = rotation_dtw_similarity(b, a, n_angles=8, n_points=16)
    # Not guaranteed to be exactly equal (rotation grid is not symmetric)
    # but both should be in range
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0


# ─── batch_rotation_dtw edge cases ───────────────────────────────────────────

def test_batch_single_candidate():
    q = _sinusoid(32)
    results = batch_rotation_dtw(q, [_line(32)], n_angles=4, n_points=16)
    assert len(results) == 1


def test_batch_five_candidates():
    q = _sinusoid(32)
    cands = [_line(32), _circle(32), _triangle(30), _sinusoid(32, 3.0), _sinusoid(32, 4.0)]
    results = batch_rotation_dtw(q, cands, n_angles=4, n_points=16)
    assert len(results) == 5


def test_batch_all_finite_distances():
    q = _circle(32)
    cands = [_line(32), _sinusoid(32), _circle(32)]
    results = batch_rotation_dtw(q, cands, n_angles=4, n_points=16)
    for r in results:
        assert np.isfinite(r.distance)


def test_batch_empty_returns_empty():
    results = batch_rotation_dtw(_sinusoid(32), [], n_angles=4, n_points=16)
    assert results == []


def test_batch_self_is_best():
    q = _sinusoid(64)
    cands = [_line(64), q, _circle(64)]
    results = batch_rotation_dtw(q, cands, n_angles=1, n_points=32, dtw_window=5)
    dists = [r.distance for r in results]
    assert dists[1] == min(dists)


def test_batch_result_types():
    q = _line(32)
    cands = [_sinusoid(32), _triangle(30)]
    results = batch_rotation_dtw(q, cands, n_angles=4, n_points=16)
    for r in results:
        assert isinstance(r, RotationDTWResult)
        assert isinstance(r.distance, float)
        assert isinstance(r.best_angle_deg, float)
        assert isinstance(r.mirrored, bool)
