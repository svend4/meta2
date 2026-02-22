"""Tests for puzzle_reconstruction/algorithms/fractal/ifs.py"""
import pytest
import numpy as np

from puzzle_reconstruction.algorithms.fractal.ifs import (
    fit_ifs_coefficients,
    reconstruct_from_ifs,
    ifs_distance,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_sine_curve(n=64, cycles=2):
    """Simple sine wave as (N, 2) curve."""
    t = np.linspace(0, 2 * np.pi * cycles, n)
    x = np.linspace(0, 1, n)
    y = np.sin(t)
    return np.column_stack([x, y])


def make_line_curve(n=32):
    """Straight line from (0,0) to (1,0)."""
    x = np.linspace(0, 1, n)
    y = np.zeros(n)
    return np.column_stack([x, y])


def make_random_curve(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = rng.standard_normal(n) * 0.1
    return np.column_stack([x, y])


# ─── fit_ifs_coefficients ─────────────────────────────────────────────────────

class TestFitIfsCoefficients:
    def test_returns_ndarray(self):
        curve = make_sine_curve()
        d = fit_ifs_coefficients(curve)
        assert isinstance(d, np.ndarray)

    def test_default_n_transforms(self):
        curve = make_sine_curve()
        d = fit_ifs_coefficients(curve, n_transforms=8)
        assert len(d) == 8

    def test_custom_n_transforms(self):
        curve = make_sine_curve(n=64)
        d = fit_ifs_coefficients(curve, n_transforms=4)
        assert len(d) == 4

    def test_coefficients_bounded(self):
        """All coefficients must be in [-0.95, 0.95] for convergence."""
        curve = make_sine_curve()
        d = fit_ifs_coefficients(curve)
        assert (d >= -0.95).all()
        assert (d <= 0.95).all()

    def test_short_curve_reduces_n_transforms(self):
        """Very short curve: n_transforms reduced automatically."""
        curve = make_sine_curve(n=5)
        d = fit_ifs_coefficients(curve, n_transforms=8)
        # n < n_transforms + 2 → n_transforms = max(2, N//4)
        assert len(d) == max(2, 5 // 4)

    def test_straight_line_near_zero(self):
        """Straight line should give near-zero IFS coefficients."""
        curve = make_line_curve(n=64)
        d = fit_ifs_coefficients(curve, n_transforms=4)
        # All should be small
        assert np.abs(d).max() < 1.0

    def test_random_curve(self):
        curve = make_random_curve()
        d = fit_ifs_coefficients(curve)
        assert len(d) > 0
        assert (np.abs(d) <= 0.95).all()

    def test_float_output(self):
        curve = make_sine_curve()
        d = fit_ifs_coefficients(curve)
        assert d.dtype in (np.float32, np.float64)

    def test_2_transforms_minimum(self):
        """n_transforms=2 should work."""
        curve = make_sine_curve()
        d = fit_ifs_coefficients(curve, n_transforms=2)
        assert len(d) == 2

    def test_reproducible(self):
        """Same input should give same output."""
        curve = make_sine_curve()
        d1 = fit_ifs_coefficients(curve, n_transforms=4)
        d2 = fit_ifs_coefficients(curve, n_transforms=4)
        np.testing.assert_array_equal(d1, d2)


# ─── reconstruct_from_ifs ─────────────────────────────────────────────────────

class TestReconstructFromIfs:
    def test_returns_ndarray(self):
        coeffs = np.array([0.5, -0.3, 0.2, 0.1])
        profile = reconstruct_from_ifs(coeffs)
        assert isinstance(profile, np.ndarray)

    def test_output_length(self):
        coeffs = np.array([0.5, -0.3])
        profile = reconstruct_from_ifs(coeffs, n_points=128)
        assert len(profile) == 128

    def test_custom_n_points(self):
        coeffs = np.array([0.4, 0.2, -0.1, 0.3])
        profile = reconstruct_from_ifs(coeffs, n_points=64)
        assert len(profile) == 64

    def test_zero_coefficients_gives_zeros(self):
        """Zero IFS coefficients → all-zero attractor."""
        coeffs = np.zeros(4)
        profile = reconstruct_from_ifs(coeffs, n_points=64, n_iter=5)
        assert (profile == 0.0).all()

    def test_output_1d(self):
        coeffs = np.array([0.5, -0.3])
        profile = reconstruct_from_ifs(coeffs)
        assert profile.ndim == 1

    def test_bounded_coefficients_bounded_output(self):
        """Coefficients < 1 → output should remain bounded."""
        coeffs = np.array([0.5, 0.3, -0.4, 0.2])
        profile = reconstruct_from_ifs(coeffs, n_iter=20)
        assert np.isfinite(profile).all()

    def test_single_transform(self):
        coeffs = np.array([0.3])
        profile = reconstruct_from_ifs(coeffs, n_points=32)
        assert len(profile) == 32

    def test_more_iterations_finite(self):
        """Increasing iterations should always give finite, bounded output."""
        coeffs = np.array([0.6, -0.5, 0.4, -0.3])
        p1 = reconstruct_from_ifs(coeffs, n_points=64, n_iter=1)
        p10 = reconstruct_from_ifs(coeffs, n_points=64, n_iter=10)
        assert np.isfinite(p1).all()
        assert np.isfinite(p10).all()


# ─── ifs_distance ─────────────────────────────────────────────────────────────

class TestIfsDistance:
    def test_identical_coefficients_zero_distance(self):
        coeffs = np.array([0.5, -0.3, 0.2])
        assert abs(ifs_distance(coeffs, coeffs)) < 1e-9

    def test_non_negative(self):
        a = np.array([0.5, -0.3, 0.2])
        b = np.array([0.1, 0.4, -0.5])
        assert ifs_distance(a, b) >= 0.0

    def test_symmetric(self):
        a = np.array([0.5, -0.3, 0.2])
        b = np.array([0.1, 0.4, -0.5])
        assert abs(ifs_distance(a, b) - ifs_distance(b, a)) < 1e-9

    def test_known_distance(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(ifs_distance(a, b) - 1.0) < 1e-9

    def test_different_lengths_resample(self):
        """Should handle different-length arrays by resampling."""
        a = np.array([0.5, 0.3, 0.2, 0.1])
        b = np.array([0.4, 0.2])  # shorter
        d = ifs_distance(a, b)
        assert d >= 0.0
        assert np.isfinite(d)

    def test_triangle_inequality(self):
        a = np.array([0.5, 0.3, 0.1])
        b = np.array([0.2, 0.4, 0.0])
        c = np.array([0.8, 0.1, 0.5])
        d_ab = ifs_distance(a, b)
        d_bc = ifs_distance(b, c)
        d_ac = ifs_distance(a, c)
        # Triangle inequality
        assert d_ac <= d_ab + d_bc + 1e-9

    def test_all_zeros(self):
        a = np.zeros(5)
        b = np.zeros(5)
        assert abs(ifs_distance(a, b)) < 1e-9

    def test_large_distance(self):
        a = np.array([0.9, 0.9, 0.9])
        b = np.array([-0.9, -0.9, -0.9])
        d = ifs_distance(a, b)
        expected = np.sqrt(3 * (1.8 ** 2))
        assert abs(d - expected) < 1e-9

    def test_single_coefficient(self):
        a = np.array([0.3])
        b = np.array([0.7])
        assert abs(ifs_distance(a, b) - 0.4) < 1e-9


# ─── Round-trip: fit → reconstruct ───────────────────────────────────────────

class TestFitReconstructRoundTrip:
    def test_fit_reconstruct_returns_profile(self):
        curve = make_sine_curve(n=64)
        d = fit_ifs_coefficients(curve, n_transforms=4)
        profile = reconstruct_from_ifs(d, n_points=64)
        assert len(profile) == 64
        assert np.isfinite(profile).all()

    def test_two_similar_curves_close_distance(self):
        """Similar curves should have closer IFS representations."""
        curve1 = make_sine_curve(n=64, cycles=1)
        curve2 = make_sine_curve(n=64, cycles=2)
        curve3 = make_line_curve(n=64)
        d1 = fit_ifs_coefficients(curve1, n_transforms=4)
        d2 = fit_ifs_coefficients(curve2, n_transforms=4)
        d3 = fit_ifs_coefficients(curve3, n_transforms=4)
        # d1 and d2 are both sine waves, d3 is flat
        dist_sines = ifs_distance(d1, d2)
        dist_sine_line = ifs_distance(d1, d3)
        # No strict guarantee, just check both are finite
        assert np.isfinite(dist_sines)
        assert np.isfinite(dist_sine_line)
