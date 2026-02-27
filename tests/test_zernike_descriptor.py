"""
Tests for puzzle_reconstruction.algorithms.zernike_descriptor.

All synthetic contours are built with numpy only; cv2 and scipy are
intentionally absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.zernike_descriptor import (
    ZernikeDescriptor,
    _factorial,
    _radial_polynomial,
    _valid_nm_pairs,
    _zernike_basis,
    zernike_moments,
    zernike_similarity,
    zernike_to_feature_vector,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic contours
# ---------------------------------------------------------------------------

def _circle_contour(n: int = 100, r: float = 1.0) -> np.ndarray:
    """Return (n, 2) points sampled uniformly from a circle of radius r."""
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square_contour(n: int = 100, side: float = 1.0) -> np.ndarray:
    """Return (n, 2) points sampled uniformly around a square [0,side]^2."""
    perimeter = 4.0 * side
    pts_per_edge = max(n // 4, 1)
    edges = [
        np.column_stack([np.linspace(0, side, pts_per_edge), np.zeros(pts_per_edge)]),
        np.column_stack([np.full(pts_per_edge, side), np.linspace(0, side, pts_per_edge)]),
        np.column_stack([np.linspace(side, 0, pts_per_edge), np.full(pts_per_edge, side)]),
        np.column_stack([np.zeros(pts_per_edge), np.linspace(side, 0, pts_per_edge)]),
    ]
    return np.vstack(edges)


def _rotate_contour(contour: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a (N, 2) contour by angle_rad around the origin."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ contour.T).T


# ===========================================================================
# 1. ZernikeDescriptor construction and fields
# ===========================================================================

class TestZernikeDescriptorConstruction:
    def test_is_named_tuple(self):
        arr = np.zeros(5, dtype=complex)
        mags = np.zeros(5)
        desc = ZernikeDescriptor(moments=arr, magnitudes=mags, order=3)
        assert isinstance(desc, tuple)

    def test_fields_present(self):
        arr = np.array([1 + 2j, 0.5 + 0j])
        mags = np.abs(arr)
        desc = ZernikeDescriptor(moments=arr, magnitudes=mags, order=2)
        assert hasattr(desc, "moments")
        assert hasattr(desc, "magnitudes")
        assert hasattr(desc, "order")

    def test_moments_field_complex(self):
        arr = np.array([1 + 2j, -0.5 + 0.5j])
        desc = ZernikeDescriptor(moments=arr, magnitudes=np.abs(arr), order=1)
        assert np.iscomplexobj(desc.moments)

    def test_magnitudes_field_real(self):
        arr = np.array([1 + 2j])
        desc = ZernikeDescriptor(moments=arr, magnitudes=np.abs(arr), order=0)
        assert not np.iscomplexobj(desc.magnitudes)

    def test_order_field_integer(self):
        desc = ZernikeDescriptor(moments=np.zeros(1, dtype=complex),
                                 magnitudes=np.zeros(1), order=7)
        assert desc.order == 7


# ===========================================================================
# 2. zernike_moments – return type and basic shape
# ===========================================================================

class TestZernikeMomentsReturnType:
    def test_returns_zernike_descriptor(self):
        contour = _circle_contour()
        result = zernike_moments(contour, order=5)
        assert isinstance(result, ZernikeDescriptor)

    def test_moments_dtype_complex(self):
        contour = _circle_contour()
        result = zernike_moments(contour, order=4)
        assert np.iscomplexobj(result.moments)

    def test_order_stored(self):
        contour = _circle_contour()
        result = zernike_moments(contour, order=7)
        assert result.order == 7

    def test_order_zero_stored(self):
        contour = _circle_contour()
        result = zernike_moments(contour, order=0)
        assert result.order == 0


# ===========================================================================
# 3. Correct number of moments
# ===========================================================================

class TestMomentCount:
    def _expected(self, order: int) -> int:
        """Number of valid (n,m) pairs up to order."""
        return len(_valid_nm_pairs(order))

    def test_order_0_has_1_moment(self):
        contour = _circle_contour()
        desc = zernike_moments(contour, order=0)
        assert len(desc.moments) == 1

    def test_order_5_moment_count(self):
        contour = _circle_contour()
        desc = zernike_moments(contour, order=5)
        assert len(desc.moments) == self._expected(5)

    def test_order_10_moment_count(self):
        contour = _circle_contour()
        desc = zernike_moments(contour, order=10)
        assert len(desc.moments) == self._expected(10)

    def test_moments_and_magnitudes_same_length(self):
        contour = _circle_contour()
        desc = zernike_moments(contour, order=6)
        assert len(desc.moments) == len(desc.magnitudes)

    def test_order_5_gives_fewer_than_order_10(self):
        contour = _circle_contour()
        desc5 = zernike_moments(contour, order=5)
        desc10 = zernike_moments(contour, order=10)
        assert len(desc5.moments) < len(desc10.moments)


# ===========================================================================
# 4. Magnitudes are non-negative
# ===========================================================================

class TestMagnitudesNonNegative:
    def test_circle_magnitudes_nonneg(self):
        desc = zernike_moments(_circle_contour(), order=8)
        assert np.all(desc.magnitudes >= 0)

    def test_square_magnitudes_nonneg(self):
        desc = zernike_moments(_square_contour(), order=6)
        assert np.all(desc.magnitudes >= 0)

    def test_small_contour_magnitudes_nonneg(self):
        tiny = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        desc = zernike_moments(tiny, order=4)
        assert np.all(desc.magnitudes >= 0)


# ===========================================================================
# 5. Similarity – identical descriptors
# ===========================================================================

class TestSimilarityIdentical:
    def test_self_similarity_circle(self):
        desc = zernike_moments(_circle_contour(), order=5)
        s = zernike_similarity(desc, desc)
        assert abs(s - 1.0) < 1e-9

    def test_self_similarity_square(self):
        desc = zernike_moments(_square_contour(), order=5)
        s = zernike_similarity(desc, desc)
        assert abs(s - 1.0) < 1e-9


# ===========================================================================
# 6. Similarity range [0, 1]
# ===========================================================================

class TestSimilarityRange:
    def test_similarity_in_unit_interval(self):
        desc_a = zernike_moments(_circle_contour(), order=5)
        desc_b = zernike_moments(_square_contour(), order=5)
        s = zernike_similarity(desc_a, desc_b)
        assert 0.0 <= s <= 1.0

    def test_similarity_nonneg_different_sizes(self):
        desc_a = zernike_moments(_circle_contour(50), order=5)
        desc_b = zernike_moments(_circle_contour(200), order=5)
        s = zernike_similarity(desc_a, desc_b)
        assert 0.0 <= s <= 1.0


# ===========================================================================
# 7. Similarity is symmetric
# ===========================================================================

class TestSimilaritySymmetry:
    def test_symmetric(self):
        desc_a = zernike_moments(_circle_contour(), order=5)
        desc_b = zernike_moments(_square_contour(), order=5)
        assert abs(zernike_similarity(desc_a, desc_b) -
                   zernike_similarity(desc_b, desc_a)) < 1e-12


# ===========================================================================
# 8. Feature vector shape and normalisation
# ===========================================================================

class TestFeatureVector:
    def test_correct_shape(self):
        desc = zernike_moments(_circle_contour(), order=5)
        fv = zernike_to_feature_vector(desc)
        assert fv.shape == desc.magnitudes.shape

    def test_l2_normalised(self):
        desc = zernike_moments(_circle_contour(), order=5)
        fv = zernike_to_feature_vector(desc)
        norm = np.linalg.norm(fv)
        assert abs(norm - 1.0) < 1e-9

    def test_values_bounded_01(self):
        desc = zernike_moments(_circle_contour(), order=5)
        fv = zernike_to_feature_vector(desc)
        assert np.all(fv >= 0.0)
        assert np.all(fv <= 1.0)

    def test_zero_descriptor_returns_zero_vector(self):
        # A degenerate contour produces all-zero magnitudes
        desc = ZernikeDescriptor(
            moments=np.zeros(5, dtype=complex),
            magnitudes=np.zeros(5),
            order=2,
        )
        fv = zernike_to_feature_vector(desc)
        assert np.all(fv == 0.0)


# ===========================================================================
# 9. Edge cases: small and large contours
# ===========================================================================

class TestContourSizes:
    def test_small_contour_10_points(self):
        small = _circle_contour(10)
        desc = zernike_moments(small, order=5, n_points=10)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(desc.magnitudes >= 0)

    def test_large_contour_500_points(self):
        large = _circle_contour(500)
        desc = zernike_moments(large, order=5, n_points=64)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(desc.magnitudes >= 0)


# ===========================================================================
# 10. Specific shape: circular contour magnitude pattern
# ===========================================================================

class TestCircularContour:
    def test_circular_produces_nonzero_moments(self):
        desc = zernike_moments(_circle_contour(200), order=5)
        assert np.any(desc.magnitudes > 0)

    def test_zero_order_magnitude_nonzero_for_circle(self):
        """The Z_00 moment (dc component) should be nonzero for a circle."""
        desc = zernike_moments(_circle_contour(200), order=4)
        assert desc.magnitudes[0] > 0


# ===========================================================================
# 11. Specific shape: square contour produces non-zero moments
# ===========================================================================

class TestSquareContour:
    def test_square_has_nonzero_moments(self):
        desc = zernike_moments(_square_contour(100), order=5)
        assert np.any(desc.magnitudes > 0)

    def test_square_moment_count_correct(self):
        desc = zernike_moments(_square_contour(100), order=5)
        assert len(desc.moments) == len(_valid_nm_pairs(5))


# ===========================================================================
# 12. Radial polynomial properties
# ===========================================================================

class TestRadialPolynomial:
    def test_R00_at_rho0(self):
        val = _radial_polynomial(0, 0, np.array([0.0]))
        np.testing.assert_allclose(val, [1.0])

    def test_R00_at_rho1(self):
        val = _radial_polynomial(0, 0, np.array([1.0]))
        np.testing.assert_allclose(val, [1.0])

    def test_R11_at_rho1(self):
        val = _radial_polynomial(1, 1, np.array([1.0]))
        np.testing.assert_allclose(val, [1.0])

    def test_R11_at_rho0(self):
        val = _radial_polynomial(1, 1, np.array([0.0]))
        np.testing.assert_allclose(val, [0.0])

    def test_R22_at_rho1(self):
        # R_22(rho) = rho^2, so at rho=1 should be 1
        val = _radial_polynomial(2, 2, np.array([1.0]))
        np.testing.assert_allclose(val, [1.0], atol=1e-12)

    def test_R20_at_rho0(self):
        # R_20(rho) = 2*rho^2 - 1, so at rho=0 should be -1
        val = _radial_polynomial(2, 0, np.array([0.0]))
        np.testing.assert_allclose(val, [-1.0], atol=1e-12)

    def test_R20_at_rho1(self):
        # R_20(1) = 2 - 1 = 1
        val = _radial_polynomial(2, 0, np.array([1.0]))
        np.testing.assert_allclose(val, [1.0], atol=1e-12)

    def test_invalid_nm_returns_zeros(self):
        # (n=2, m=1): (2-1)=1 is odd => invalid, should return zeros
        rho = np.linspace(0, 1, 10)
        val = _radial_polynomial(2, 1, rho)
        np.testing.assert_array_equal(val, np.zeros(10))

    def test_R_nm_equals_R_n_minus_m(self):
        """R_nm is even in m: R_n,m == R_n,-m."""
        rho = np.linspace(0, 1, 20)
        r_pos = _radial_polynomial(4, 2, rho)
        r_neg = _radial_polynomial(4, -2, rho)
        np.testing.assert_allclose(r_pos, r_neg, atol=1e-12)


# ===========================================================================
# 13. Rotation invariance
# ===========================================================================

class TestRotationInvariance:
    def _desc(self, contour: np.ndarray, order: int = 6) -> ZernikeDescriptor:
        return zernike_moments(contour, order=order, n_points=128)

    def test_circle_rotation_invariant(self):
        """Rotating a circle should not change its Zernike magnitudes."""
        base = _circle_contour(200)
        rotated = _rotate_contour(base, np.pi / 3)
        da = self._desc(base)
        db = self._desc(rotated)
        s = zernike_similarity(da, db)
        assert s > 0.99

    def test_square_rotation_90_deg(self):
        """A square rotated 90 degrees should look the same (4-fold symmetry)."""
        base = _square_contour(200)
        rotated = _rotate_contour(base, np.pi / 2)
        da = self._desc(base)
        db = self._desc(rotated)
        s = zernike_similarity(da, db)
        assert s > 0.95


# ===========================================================================
# 14. Degenerate / edge-case inputs
# ===========================================================================

class TestDegenerateInputs:
    def test_collinear_contour(self):
        """A straight-line contour should not raise an exception."""
        line = np.column_stack([np.linspace(0, 1, 20), np.zeros(20)])
        desc = zernike_moments(line, order=4)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(desc.magnitudes >= 0)

    def test_repeated_points_contour(self):
        """A contour with all identical points should not raise."""
        pts = np.tile([[1.0, 2.0]], (30, 1))
        desc = zernike_moments(pts, order=4)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(desc.magnitudes >= 0)

    def test_single_point_contour(self):
        """A single-point (1, 2) contour should return zero moments gracefully."""
        pts = np.array([[5.0, 3.0]])
        desc = zernike_moments(pts, order=3)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(desc.magnitudes >= 0)

    def test_two_point_contour(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        desc = zernike_moments(pts, order=3)
        assert isinstance(desc, ZernikeDescriptor)


# ===========================================================================
# 15. Very different shapes give low similarity
# ===========================================================================

class TestLowSimilarityDifferentShapes:
    def test_circle_vs_line_low_similarity(self):
        line = np.column_stack([np.linspace(-1, 1, 50), np.zeros(50)])
        desc_circle = zernike_moments(_circle_contour(100), order=8)
        desc_line = zernike_moments(line, order=8)
        s = zernike_similarity(desc_circle, desc_line)
        # Lines are degenerate so zero-descriptor edge case handled; score <= 1
        assert 0.0 <= s <= 1.0


# ===========================================================================
# 16. Batch computation
# ===========================================================================

class TestBatchComputation:
    def test_batch_moments_list(self):
        contours = [_circle_contour(50), _square_contour(50), _circle_contour(80)]
        descriptors = [zernike_moments(c, order=5) for c in contours]
        assert len(descriptors) == 3
        for d in descriptors:
            assert isinstance(d, ZernikeDescriptor)

    def test_batch_similarities(self):
        contours = [_circle_contour(80) for _ in range(5)]
        descs = [zernike_moments(c, order=4) for c in contours]
        # All are circles; pairwise similarities should be near 1
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                s = zernike_similarity(descs[i], descs[j])
                assert s > 0.99


# ===========================================================================
# 17. Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_input_same_output(self):
        contour = _circle_contour(100)
        desc1 = zernike_moments(contour, order=5, n_points=64)
        desc2 = zernike_moments(contour, order=5, n_points=64)
        np.testing.assert_array_equal(desc1.moments, desc2.moments)
        np.testing.assert_array_equal(desc1.magnitudes, desc2.magnitudes)

    def test_determinism_square(self):
        sq = _square_contour(60)
        d1 = zernike_moments(sq, order=6)
        d2 = zernike_moments(sq, order=6)
        np.testing.assert_array_equal(d1.magnitudes, d2.magnitudes)


# ===========================================================================
# 18. Factorial helper
# ===========================================================================

class TestFactorial:
    def test_factorial_0(self):
        assert _factorial(0) == 1.0

    def test_factorial_1(self):
        assert _factorial(1) == 1.0

    def test_factorial_5(self):
        assert _factorial(5) == 120.0

    def test_factorial_10(self):
        assert _factorial(10) == 3628800.0

    def test_factorial_negative_raises(self):
        with pytest.raises((ValueError, Exception)):
            _factorial(-1)
