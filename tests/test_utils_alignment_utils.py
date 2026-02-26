"""Tests for puzzle_reconstruction.utils.alignment_utils."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.alignment_utils import (
    AlignmentConfig,
    AlignmentResult,
    normalize_for_alignment,
    find_best_rotation,
    find_best_translation,
    compute_alignment_error,
    align_curves_procrustes,
    align_curves_icp,
    alignment_score,
    batch_align_curves,
    _resample,
    _rotation_matrix,
)


np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_circle(n=32, r=10.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def rotate_pts(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


# ── AlignmentConfig ───────────────────────────────────────────────────────────

def test_alignment_config_defaults():
    cfg = AlignmentConfig()
    assert cfg.n_samples == 64
    assert cfg.max_icp_iter == 50
    assert cfg.icp_tol > 0


def test_alignment_config_invalid_n_samples():
    with pytest.raises(ValueError):
        AlignmentConfig(n_samples=1)


def test_alignment_config_invalid_max_icp_iter():
    with pytest.raises(ValueError):
        AlignmentConfig(max_icp_iter=0)


def test_alignment_config_invalid_icp_tol():
    with pytest.raises(ValueError):
        AlignmentConfig(icp_tol=0.0)


# ── _resample ─────────────────────────────────────────────────────────────────

def test_resample_output_shape():
    pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    result = _resample(pts, 10)
    assert result.shape == (10, 2)


def test_resample_degenerate_single_point():
    pts = np.array([[3.0, 5.0]])
    result = _resample(pts, 8)
    assert result.shape == (8, 2)
    assert np.allclose(result, 3.0, atol=1e-9) or np.allclose(result[:, 0], 3.0)


def test_resample_collinear():
    pts = np.array([[0, 0], [5, 0]], dtype=float)
    result = _resample(pts, 6)
    assert result.shape == (6, 2)
    assert np.allclose(result[:, 1], 0.0)


# ── normalize_for_alignment ───────────────────────────────────────────────────

def test_normalize_zero_mean():
    pts = make_circle(20, 5.0)
    normed, centroid, scale = normalize_for_alignment(pts)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-9)


def test_normalize_unit_rms():
    pts = make_circle(20, 5.0)
    normed, centroid, scale = normalize_for_alignment(pts)
    rms = float(np.sqrt((normed ** 2).sum() / len(normed)))
    assert abs(rms - 1.0) < 1e-6


def test_normalize_returns_centroid_and_scale():
    pts = np.array([[10.0, 20.0], [12.0, 22.0]])
    normed, centroid, scale = normalize_for_alignment(pts)
    assert centroid.shape == (2,)
    assert scale > 0.0


def test_normalize_constant_pts():
    pts = np.ones((5, 2)) * 7.0
    normed, centroid, scale = normalize_for_alignment(pts)
    assert scale == 1.0  # fallback when all identical


# ── find_best_rotation ────────────────────────────────────────────────────────

def test_find_best_rotation_identity():
    pts = make_circle(16, 3.0)
    angle, R = find_best_rotation(pts, pts)
    assert abs(angle) < 1e-6 or abs(abs(angle) - 2 * np.pi) < 1e-6


def test_find_best_rotation_known_angle():
    pts = make_circle(32, 5.0)
    theta = math.pi / 4
    rotated = rotate_pts(pts, theta)
    angle, R = find_best_rotation(pts, rotated)
    assert abs(angle - theta) < 1e-3 or abs(angle + theta) < 1e-3


def test_find_best_rotation_returns_2x2():
    pts = np.random.randn(10, 2)
    angle, R = find_best_rotation(pts, pts)
    assert R.shape == (2, 2)


# ── find_best_translation ─────────────────────────────────────────────────────

def test_find_best_translation_returns_2d():
    src = np.random.randn(10, 2)
    tgt = src + np.array([3.0, -5.0])
    t = find_best_translation(src, tgt)
    assert t.shape == (2,)
    assert np.allclose(t, [3.0, -5.0], atol=1e-6)


# ── compute_alignment_error ───────────────────────────────────────────────────

def test_compute_alignment_error_identical():
    pts = make_circle(20, 4.0)
    assert compute_alignment_error(pts, pts) == 0.0


def test_compute_alignment_error_positive():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [2.0, 0.0]])
    err = compute_alignment_error(a, b)
    assert err > 0.0


def test_compute_alignment_error_shape_mismatch_returns_inf():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert compute_alignment_error(a, b) == float("inf")


# ── align_curves_procrustes ───────────────────────────────────────────────────

def test_procrustes_result_type():
    src = make_circle(16, 3.0)
    tgt = make_circle(16, 5.0)
    result = align_curves_procrustes(src, tgt)
    assert isinstance(result, AlignmentResult)


def test_procrustes_aligned_shape():
    src = make_circle(16, 3.0)
    tgt = make_circle(16, 5.0)
    result = align_curves_procrustes(src, tgt, AlignmentConfig(n_samples=32))
    assert result.aligned.shape[1] == 2


def test_procrustes_error_nonnegative():
    src = make_circle(16, 3.0)
    tgt = rotate_pts(make_circle(16, 3.0), 0.5)
    result = align_curves_procrustes(src, tgt)
    assert result.error >= 0.0


def test_procrustes_to_dict_keys():
    src = make_circle(16, 3.0)
    result = align_curves_procrustes(src, src)
    d = result.to_dict()
    assert "rotation" in d and "error" in d and "scale" in d


# ── align_curves_icp ──────────────────────────────────────────────────────────

def test_icp_result_type():
    src = make_circle(16, 3.0)
    tgt = make_circle(16, 3.0)
    result = align_curves_icp(src, tgt)
    assert isinstance(result, AlignmentResult)


def test_icp_scale_is_one():
    src = make_circle(16, 3.0)
    tgt = make_circle(16, 3.0)
    result = align_curves_icp(src, tgt)
    assert result.scale == 1.0


def test_icp_aligned_shape():
    src = make_circle(16, 3.0)
    result = align_curves_icp(src, src)
    assert result.aligned.shape == (64, 2)  # default n_samples=64


# ── alignment_score ───────────────────────────────────────────────────────────

def test_alignment_score_identical_zero_error():
    src = make_circle(16, 3.0)
    result = align_curves_procrustes(src, src)
    score = alignment_score(result, sigma=1.0)
    assert abs(score - 1.0) < 0.05  # near 1 when error ≈ 0


def test_alignment_score_in_01():
    src = make_circle(16, 3.0)
    tgt = make_circle(16, 5.0)
    result = align_curves_procrustes(src, tgt)
    score = alignment_score(result, sigma=1.0)
    assert 0.0 <= score <= 1.0


def test_alignment_score_invalid_sigma():
    result = AlignmentResult(0.0, np.zeros(2), 1.0, 0.5, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        alignment_score(result, sigma=-1.0)


# ── batch_align_curves ────────────────────────────────────────────────────────

def test_batch_align_length():
    curves = [make_circle(16, r) for r in [3, 4, 5]]
    results = batch_align_curves(curves, curves)
    assert len(results) == 3


def test_batch_align_length_mismatch():
    curves = [make_circle(16, 3)]
    with pytest.raises(ValueError):
        batch_align_curves(curves, [curves[0], curves[0]])


def test_batch_align_icp_method():
    src = [make_circle(16, 3)]
    tgt = [make_circle(16, 3)]
    results = batch_align_curves(src, tgt, method="icp")
    assert len(results) == 1
    assert results[0].scale == 1.0


def test_batch_align_invalid_method():
    src = [make_circle(16, 3)]
    tgt = [make_circle(16, 3)]
    with pytest.raises(ValueError):
        batch_align_curves(src, tgt, method="unknown")
