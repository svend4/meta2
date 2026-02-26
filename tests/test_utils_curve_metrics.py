"""Tests for puzzle_reconstruction.utils.curve_metrics."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.utils.curve_metrics import (
    CurveMetricConfig,
    CurveComparisonResult,
    _resample,
    curve_l2,
    curve_l2_mirror,
    hausdorff_distance,
    frechet_distance_approx,
    curve_length,
    length_ratio,
    compare_curves,
    batch_compare_curves,
)

np.random.seed(123)


# ── Helpers ────────────────────────────────────────────────────────────────────

def line(n=10, length=10.0):
    """Horizontal line from 0 to length."""
    x = np.linspace(0, length, n)
    y = np.zeros(n)
    return np.stack([x, y], axis=1)


def circle_arc(n=32, r=5.0):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([r*np.cos(t), r*np.sin(t)], axis=1)


# ── CurveMetricConfig ──────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = CurveMetricConfig()
    assert cfg.n_samples >= 2
    assert cfg.eps > 0


def test_config_invalid_n_samples():
    with pytest.raises(ValueError):
        CurveMetricConfig(n_samples=1)


def test_config_invalid_eps():
    with pytest.raises(ValueError):
        CurveMetricConfig(eps=0.0)


# ── _resample ─────────────────────────────────────────────────────────────────

def test_resample_shape():
    pts = line(5, 10.0)
    result = _resample(pts, 20)
    assert result.shape == (20, 2)


def test_resample_empty_pts():
    result = _resample(np.zeros((0, 2)), 5)
    assert result.shape == (5, 2)


def test_resample_single_pt():
    pts = np.array([[3.0, 4.0]])
    result = _resample(pts, 8)
    assert result.shape == (8, 2)
    assert np.allclose(result, [[3.0, 4.0]])


def test_resample_degenerate_zero_length():
    pts = np.ones((5, 2)) * 7.0
    result = _resample(pts, 4)
    assert result.shape == (4, 2)
    assert np.allclose(result, 7.0)


# ── curve_l2 ──────────────────────────────────────────────────────────────────

def test_curve_l2_identical():
    a = line(20, 10.0)
    assert curve_l2(a, a) == pytest.approx(0.0, abs=1e-9)


def test_curve_l2_non_negative():
    a = line(20, 10.0)
    b = line(20, 10.0) + np.array([0, 5])
    assert curve_l2(a, b) >= 0.0


def test_curve_l2_returns_float():
    a = line(10, 5.0)
    b = line(10, 5.0) + 1.0
    assert isinstance(curve_l2(a, b), float)


def test_curve_l2_custom_config():
    cfg = CurveMetricConfig(n_samples=16)
    a = line(10, 5.0)
    b = line(10, 5.0)
    assert curve_l2(a, b, cfg) == pytest.approx(0.0, abs=1e-9)


# ── curve_l2_mirror ───────────────────────────────────────────────────────────

def test_curve_l2_mirror_symmetric():
    a = line(10, 10.0)
    b = a[::-1]
    d_mirror = curve_l2_mirror(a, b)
    d_direct = curve_l2(a, b)
    assert d_mirror <= d_direct + 1e-9


def test_curve_l2_mirror_nonneg():
    a = line(10, 5.0)
    b = line(10, 6.0)
    assert curve_l2_mirror(a, b) >= 0.0


# ── hausdorff_distance ────────────────────────────────────────────────────────

def test_hausdorff_identical():
    a = line(16, 10.0)
    assert hausdorff_distance(a, a) == pytest.approx(0.0, abs=1e-6)


def test_hausdorff_non_negative():
    a = circle_arc(16)
    b = circle_arc(16, r=6.0)
    assert hausdorff_distance(a, b) >= 0.0


def test_hausdorff_symmetric():
    a = line(16, 10.0)
    b = line(16, 10.0) + np.array([0, 2])
    assert abs(hausdorff_distance(a, b) - hausdorff_distance(b, a)) < 1e-9


def test_hausdorff_returns_float():
    a = line(10, 5.0)
    b = line(12, 5.0)
    assert isinstance(hausdorff_distance(a, b), float)


# ── frechet_distance_approx ────────────────────────────────────────────────────

def test_frechet_identical():
    a = line(16, 10.0)
    assert frechet_distance_approx(a, a) == pytest.approx(0.0, abs=1e-6)


def test_frechet_positive():
    a = line(16, 10.0)
    b = line(16, 10.0) + np.array([0, 3])
    assert frechet_distance_approx(a, b) > 0.0


def test_frechet_non_negative():
    a = circle_arc(16)
    b = circle_arc(16, r=7.0)
    assert frechet_distance_approx(a, b) >= 0.0


def test_frechet_returns_float():
    a = line(8, 5.0)
    b = line(8, 5.0) + 0.5
    assert isinstance(frechet_distance_approx(a, b), float)


# ── curve_length ──────────────────────────────────────────────────────────────

def test_curve_length_line():
    pts = line(11, 10.0)
    assert abs(curve_length(pts) - 10.0) < 1e-9


def test_curve_length_single_pt():
    pts = np.array([[5.0, 5.0]])
    assert curve_length(pts) == 0.0


def test_curve_length_nonneg():
    pts = np.random.randn(20, 2)
    assert curve_length(pts) >= 0.0


def test_curve_length_empty():
    pts = np.zeros((0, 2))
    assert curve_length(pts) == 0.0


# ── length_ratio ──────────────────────────────────────────────────────────────

def test_length_ratio_identical():
    a = line(10, 10.0)
    assert length_ratio(a, a) == pytest.approx(1.0, abs=1e-9)


def test_length_ratio_in_01():
    a = line(10, 10.0)
    b = line(10, 5.0)
    r = length_ratio(a, b)
    assert 0.0 <= r <= 1.0


def test_length_ratio_zero_both():
    a = np.ones((5, 2))
    b = np.ones((5, 2)) * 2
    # both are degenerate → length 0 → ratio 0
    assert length_ratio(a, b) == 0.0


# ── compare_curves ────────────────────────────────────────────────────────────

def test_compare_curves_returns_result():
    a = line(16, 10.0)
    b = line(16, 10.0) + np.array([0, 1])
    result = compare_curves(a, b)
    assert isinstance(result, CurveComparisonResult)


def test_compare_curves_to_dict():
    a = line(16, 10.0)
    result = compare_curves(a, a)
    d = result.to_dict()
    assert all(k in d for k in ("l2", "hausdorff", "frechet", "length_ratio"))


def test_compare_curves_similarity_in_01():
    a = line(16, 10.0)
    b = line(16, 10.0) + np.array([0, 0.5])
    result = compare_curves(a, b)
    s = result.similarity(sigma=1.0)
    assert 0.0 <= s <= 1.0


def test_compare_curves_similarity_invalid_sigma():
    a = line(16, 10.0)
    result = compare_curves(a, a)
    with pytest.raises(ValueError):
        result.similarity(sigma=-1.0)


# ── batch_compare_curves ──────────────────────────────────────────────────────

def test_batch_compare_length():
    pairs = [(line(8, 5.0), line(8, 5.0)) for _ in range(4)]
    results = batch_compare_curves(pairs)
    assert len(results) == 4


def test_batch_compare_each_is_result():
    pairs = [(line(8, 5.0), line(8, 6.0))]
    results = batch_compare_curves(pairs)
    assert isinstance(results[0], CurveComparisonResult)


def test_batch_compare_empty():
    results = batch_compare_curves([])
    assert results == []
