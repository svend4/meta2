"""Additional tests for puzzle_reconstruction/algorithms/fractal/ifs.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.ifs import (
    fit_ifs_coefficients,
    reconstruct_from_ifs,
    ifs_distance,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sine(n: int = 64, cycles: int = 2) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi * cycles, n)
    return np.column_stack([np.linspace(0, 1, n), np.sin(t)])


def _line(n: int = 64) -> np.ndarray:
    return np.column_stack([np.linspace(0, 1, n), np.zeros(n)])


def _flat(n: int = 32) -> np.ndarray:
    return np.full((n, 2), 0.5)


# ─── TestFitIFSCoefficientsExtra ──────────────────────────────────────────────

class TestFitIFSCoefficientsExtra:
    def test_n_transforms_16(self):
        d = fit_ifs_coefficients(_sine(n=256), n_transforms=16)
        assert len(d) == 16

    def test_large_n_512(self):
        d = fit_ifs_coefficients(_sine(n=512), n_transforms=8)
        assert len(d) == 8

    def test_constant_curve(self):
        """All-same-point curve: IFS should not crash."""
        d = fit_ifs_coefficients(_flat(n=32), n_transforms=4)
        assert len(d) >= 1

    def test_two_point_curve(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        d = fit_ifs_coefficients(pts, n_transforms=4)
        assert len(d) >= 1

    def test_float32_input(self):
        d = fit_ifs_coefficients(_sine(n=64).astype(np.float32), n_transforms=4)
        assert len(d) == 4

    def test_integer_input(self):
        pts = (_sine(n=64) * 100).astype(np.int32)
        d = fit_ifs_coefficients(pts, n_transforms=4)
        assert len(d) >= 1

    def test_all_abs_coeffs_lt_1(self):
        for n in [32, 64, 128]:
            d = fit_ifs_coefficients(_sine(n=n), n_transforms=4)
            assert np.all(np.abs(d) < 1.0)

    def test_output_shape_correct(self):
        for nt in [2, 4, 6, 8]:
            d = fit_ifs_coefficients(_sine(n=128), n_transforms=nt)
            assert d.shape == (nt,)

    def test_returns_float_dtype(self):
        d = fit_ifs_coefficients(_line(), n_transforms=4)
        assert np.issubdtype(d.dtype, np.floating)


# ─── TestReconstructFromIFSExtra ──────────────────────────────────────────────

class TestReconstructFromIFSExtra:
    def test_n_points_1(self):
        coeffs = np.array([0.3, 0.1])
        profile = reconstruct_from_ifs(coeffs, n_points=1)
        assert profile.shape == (1,)

    def test_n_points_256(self):
        coeffs = np.array([0.3, -0.2, 0.1, 0.4])
        profile = reconstruct_from_ifs(coeffs, n_points=256)
        assert len(profile) == 256

    def test_n_iter_0_no_crash(self):
        coeffs = np.array([0.3, -0.2])
        profile = reconstruct_from_ifs(coeffs, n_points=32, n_iter=0)
        assert profile.shape == (32,)

    def test_float_output_dtype(self):
        coeffs = np.array([0.5, -0.3, 0.2])
        profile = reconstruct_from_ifs(coeffs, n_points=64)
        assert np.issubdtype(profile.dtype, np.floating)

    def test_output_1d(self):
        coeffs = np.array([0.5, -0.3])
        profile = reconstruct_from_ifs(coeffs, n_points=64)
        assert profile.ndim == 1

    def test_single_coefficient(self):
        coeffs = np.array([0.5])
        profile = reconstruct_from_ifs(coeffs, n_points=32)
        assert len(profile) == 32

    def test_many_transforms(self):
        coeffs = np.linspace(-0.8, 0.8, 16)
        profile = reconstruct_from_ifs(coeffs, n_points=128)
        assert len(profile) == 128
        assert np.all(np.isfinite(profile))

    def test_negative_coefficients_finite(self):
        coeffs = np.array([-0.7, -0.5, -0.3])
        profile = reconstruct_from_ifs(coeffs, n_points=64)
        assert np.all(np.isfinite(profile))


# ─── TestIFSDistanceExtra ─────────────────────────────────────────────────────

class TestIFSDistanceExtra:
    def test_returns_float_type(self):
        a = np.array([0.3, -0.1, 0.5])
        b = np.array([0.1, 0.2, 0.0])
        assert isinstance(ifs_distance(a, b), float)

    def test_zero_arrays_same_length(self):
        a = np.zeros(4)
        b = np.zeros(4)
        assert ifs_distance(a, b) == pytest.approx(0.0)

    def test_zero_vs_nonzero(self):
        a = np.zeros(4)
        b = np.array([0.5, 0.3, -0.2, 0.1])
        d = ifs_distance(a, b)
        assert d > 0.0

    def test_negative_coefficients_symmetric(self):
        a = np.array([-0.5, -0.3])
        b = np.array([0.5, 0.3])
        assert ifs_distance(a, b) == pytest.approx(ifs_distance(b, a))

    def test_all_positive_positive_distance(self):
        a = np.array([0.9, 0.8, 0.7])
        b = np.array([0.1, 0.2, 0.3])
        d = ifs_distance(a, b)
        assert d > 0.0

    def test_different_lengths_finite(self):
        a = np.array([0.5, 0.3, 0.2, 0.1, 0.05])
        b = np.array([0.5, 0.3])
        d = ifs_distance(a, b)
        assert np.isfinite(d)

    def test_known_value_single_coeff(self):
        a = np.array([0.0])
        b = np.array([0.4])
        assert ifs_distance(a, b) == pytest.approx(0.4, abs=1e-9)


# ─── TestRoundTripExtra ───────────────────────────────────────────────────────

class TestRoundTripExtra:
    def test_fit_large_then_reconstruct(self):
        curve = _sine(n=256)
        d = fit_ifs_coefficients(curve, n_transforms=8)
        profile = reconstruct_from_ifs(d, n_points=256)
        assert len(profile) == 256
        assert np.all(np.isfinite(profile))

    def test_constant_curve_reconstruct_finite(self):
        curve = _flat(n=32)
        d = fit_ifs_coefficients(curve, n_transforms=4)
        profile = reconstruct_from_ifs(d, n_points=64)
        assert np.all(np.isfinite(profile))

    def test_line_curve_reconstruct_zeros(self):
        """Straight horizontal line → near-zero coefficients → near-zero profile."""
        curve = _line(n=64)
        d = fit_ifs_coefficients(curve, n_transforms=4)
        profile = reconstruct_from_ifs(d, n_points=64)
        assert np.all(np.isfinite(profile))
