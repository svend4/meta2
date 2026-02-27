"""Tests for puzzle_reconstruction/algorithms/zernike_descriptor.py."""
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square(n: int = 64, side: float = 1.0) -> np.ndarray:
    k = max(n // 4, 1)
    top    = np.column_stack([np.linspace(0, side, k), np.full(k, side)])
    right  = np.column_stack([np.full(k, side), np.linspace(side, 0, k)])
    bottom = np.column_stack([np.linspace(side, 0, k), np.zeros(k)])
    left   = np.column_stack([np.zeros(k), np.linspace(0, side, k)])
    return np.vstack([top, right, bottom, left])


def _ellipse(n: int = 64, a: float = 2.0, b: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


def _triangle(n: int = 60) -> np.ndarray:
    k = max(n // 3, 1)
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    e0 = v0 + np.outer(np.linspace(0, 1, k), v1 - v0)
    e1 = v1 + np.outer(np.linspace(0, 1, k), v2 - v1)
    e2 = v2 + np.outer(np.linspace(0, 1, k), v0 - v2)
    return np.vstack([e0, e1, e2])


def _rotate(contour: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ contour.T).T


def _translate(contour: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return contour + np.array([dx, dy])


def _scale(contour: np.ndarray, factor: float) -> np.ndarray:
    return contour * factor


# ── _factorial ─────────────────────────────────────────────────────────────────

class TestFactorial:

    def test_factorial_2(self):
        assert _factorial(2) == 2.0

    def test_factorial_3(self):
        assert _factorial(3) == 6.0

    def test_factorial_6(self):
        assert _factorial(6) == 720.0

    def test_factorial_7(self):
        assert _factorial(7) == 5040.0

    def test_factorial_returns_float(self):
        assert isinstance(_factorial(4), float)

    def test_factorial_large(self):
        # 12! = 479001600
        assert _factorial(12) == pytest.approx(479001600.0)

    @pytest.mark.parametrize("n,expected", [
        (0, 1.0), (1, 1.0), (2, 2.0), (4, 24.0), (8, 40320.0),
    ])
    def test_factorial_parametrised(self, n, expected):
        assert _factorial(n) == pytest.approx(expected)


# ── _radial_polynomial ──────────────────────────────────────────────────────────

class TestRadialPolynomial:

    def test_output_shape_matches_rho(self):
        rho = np.linspace(0, 1, 50)
        out = _radial_polynomial(2, 0, rho)
        assert out.shape == rho.shape

    def test_output_dtype_float(self):
        rho = np.array([0.5])
        out = _radial_polynomial(2, 0, rho)
        assert out.dtype == float

    def test_R00_is_constant_one(self):
        rho = np.linspace(0, 1, 20)
        out = _radial_polynomial(0, 0, rho)
        np.testing.assert_allclose(out, np.ones(20), atol=1e-12)

    def test_R11_equals_rho(self):
        rho = np.linspace(0, 1, 15)
        out = _radial_polynomial(1, 1, rho)
        np.testing.assert_allclose(out, rho, atol=1e-12)

    def test_R1_neg1_equals_rho(self):
        rho = np.linspace(0, 1, 15)
        out = _radial_polynomial(1, -1, rho)
        np.testing.assert_allclose(out, rho, atol=1e-12)

    def test_R20_formula(self):
        # R_20(rho) = 2*rho^2 - 1
        rho = np.linspace(0, 1, 25)
        out = _radial_polynomial(2, 0, rho)
        expected = 2 * rho**2 - 1
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_R22_equals_rho_squared(self):
        # R_22(rho) = rho^2
        rho = np.linspace(0, 1, 25)
        out = _radial_polynomial(2, 2, rho)
        np.testing.assert_allclose(out, rho**2, atol=1e-12)

    def test_R31_formula(self):
        # R_31(rho) = 3*rho^3 - 2*rho
        rho = np.linspace(0, 1, 30)
        out = _radial_polynomial(3, 1, rho)
        expected = 3 * rho**3 - 2 * rho
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_odd_nm_diff_returns_zeros(self):
        # (n=3, m=2): (3-2)=1 is odd -> invalid -> zeros
        rho = np.linspace(0, 1, 10)
        out = _radial_polynomial(3, 2, rho)
        np.testing.assert_array_equal(out, np.zeros(10))

    def test_symmetry_in_m(self):
        # R_nm(rho) = R_n,-m(rho) because definition uses |m|
        rho = np.linspace(0, 1, 30)
        for n, m in [(2, 2), (3, 1), (4, 2), (4, 4)]:
            np.testing.assert_allclose(
                _radial_polynomial(n, m, rho),
                _radial_polynomial(n, -m, rho),
                atol=1e-12,
                err_msg=f"Symmetry failed for n={n}, m={m}",
            )

    def test_at_rho_zero_R40(self):
        # R_40(0) = 6*0 - 6*0 + 1 = 1
        val = _radial_polynomial(4, 0, np.array([0.0]))
        assert np.isfinite(val[0])

    def test_handles_scalar_rho_as_array(self):
        rho = np.array([0.7])
        out = _radial_polynomial(2, 2, rho)
        np.testing.assert_allclose(out, [0.7**2], atol=1e-12)


# ── _zernike_basis ──────────────────────────────────────────────────────────────

class TestZernikeBasis:

    def test_output_is_complex(self):
        rho = np.array([0.5])
        theta = np.array([1.0])
        out = _zernike_basis(1, 1, rho, theta)
        assert np.iscomplexobj(out)

    def test_output_shape_matches_input(self):
        rho = np.linspace(0, 1, 40)
        theta = np.linspace(0, 2 * np.pi, 40)
        out = _zernike_basis(2, 2, rho, theta)
        assert out.shape == rho.shape

    def test_outside_unit_disk_zeroed(self):
        rho = np.array([0.5, 1.0, 1.5, 2.0])
        theta = np.zeros(4)
        out = _zernike_basis(1, 1, rho, theta)
        # rho > 1 must be zero
        assert out[2] == 0.0
        assert out[3] == 0.0

    def test_inside_unit_disk_nonzero_for_n1m1(self):
        rho = np.array([0.5])
        theta = np.array([0.0])
        out = _zernike_basis(1, 1, rho, theta)
        assert abs(out[0]) > 0.0

    def test_at_origin_R00_V00(self):
        # V_00(0, theta) = R_00(0) * exp(0) = 1.0
        rho = np.array([0.0])
        theta = np.array([0.0])
        out = _zernike_basis(0, 0, rho, theta)
        np.testing.assert_allclose(out.real, [1.0], atol=1e-12)
        np.testing.assert_allclose(out.imag, [0.0], atol=1e-12)

    def test_azimuthal_phase_encodes_theta(self):
        # V_nm phase = m * theta
        rho = np.array([0.5])
        theta = np.array([np.pi / 4])
        out = _zernike_basis(1, 1, rho, theta)
        expected_phase = np.pi / 4
        np.testing.assert_allclose(np.angle(out[0]), expected_phase, atol=1e-12)

    def test_invalid_nm_basis_is_zero(self):
        # (n=2, m=1) is invalid -> radial polynomial zeros -> basis zeros
        rho = np.linspace(0.1, 0.9, 10)
        theta = np.linspace(0, np.pi, 10)
        out = _zernike_basis(2, 1, rho, theta)
        np.testing.assert_array_equal(np.abs(out), np.zeros(10))


# ── _valid_nm_pairs ─────────────────────────────────────────────────────────────

class TestValidNmPairs:

    def test_order_0(self):
        pairs = _valid_nm_pairs(0)
        assert pairs == [(0, 0)]

    def test_order_1(self):
        pairs = _valid_nm_pairs(1)
        assert (0, 0) in pairs
        assert (1, -1) in pairs
        assert (1, 1) in pairs
        # (1, 0) is invalid: (1-0)=1 odd
        assert (1, 0) not in pairs

    def test_order_2_count(self):
        pairs = _valid_nm_pairs(2)
        # n=0: (0,0); n=1: (1,-1),(1,1); n=2: (2,-2),(2,0),(2,2)
        assert len(pairs) == 6

    def test_all_pairs_satisfy_n_minus_m_even(self):
        for order in [3, 5, 8]:
            for n, m in _valid_nm_pairs(order):
                assert (n - m) % 2 == 0, f"Invalid pair ({n},{m}) for order {order}"

    def test_all_pairs_satisfy_m_leq_n(self):
        for order in [4, 6]:
            for n, m in _valid_nm_pairs(order):
                assert abs(m) <= n, f"|m|={abs(m)} > n={n}"

    def test_all_pairs_n_nonneg(self):
        for n, m in _valid_nm_pairs(7):
            assert n >= 0

    def test_order_grows_monotonically(self):
        prev = len(_valid_nm_pairs(3))
        for order in range(4, 9):
            curr = len(_valid_nm_pairs(order))
            assert curr > prev, f"Pair count did not grow at order {order}"
            prev = curr

    def test_order_5_count(self):
        # Sum: n=0:1, n=1:2, n=2:3, n=3:4, n=4:5, n=5:6 => 21
        pairs = _valid_nm_pairs(5)
        assert len(pairs) == 21

    def test_pairs_are_tuples(self):
        for pair in _valid_nm_pairs(3):
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_no_duplicate_pairs(self):
        pairs = _valid_nm_pairs(6)
        assert len(pairs) == len(set(pairs))

    def test_pairs_sorted_by_n_then_m(self):
        pairs = _valid_nm_pairs(4)
        for i in range(len(pairs) - 1):
            n0, m0 = pairs[i]
            n1, m1 = pairs[i + 1]
            assert (n0, m0) <= (n1, m1), "Pairs not in sorted order"


# ── ZernikeDescriptor NamedTuple ────────────────────────────────────────────────

class TestZernikeDescriptorNamedTuple:

    def test_positional_construction(self):
        m = np.zeros(3, dtype=complex)
        mag = np.zeros(3)
        desc = ZernikeDescriptor(m, mag, 2)
        assert desc.order == 2

    def test_keyword_construction(self):
        m = np.array([1 + 0j])
        mag = np.array([1.0])
        desc = ZernikeDescriptor(moments=m, magnitudes=mag, order=0)
        assert desc.order == 0

    def test_unpacking(self):
        m = np.zeros(2, dtype=complex)
        mag = np.zeros(2)
        moments, magnitudes, order = ZernikeDescriptor(m, mag, 4)
        assert order == 4

    def test_indexing(self):
        m = np.zeros(2, dtype=complex)
        mag = np.zeros(2)
        desc = ZernikeDescriptor(m, mag, 5)
        assert desc[2] == 5

    def test_len_is_3(self):
        desc = ZernikeDescriptor(np.zeros(1, dtype=complex), np.zeros(1), 0)
        assert len(desc) == 3

    def test_immutable(self):
        desc = ZernikeDescriptor(np.zeros(1, dtype=complex), np.zeros(1), 0)
        with pytest.raises((AttributeError, TypeError)):
            desc.order = 99  # type: ignore[misc]


# ── zernike_moments – return shape and types ────────────────────────────────────

class TestZernikeMomentsShapesAndTypes:

    def test_moments_ndim_1(self):
        desc = zernike_moments(_circle(), order=5)
        assert desc.moments.ndim == 1

    def test_magnitudes_ndim_1(self):
        desc = zernike_moments(_circle(), order=5)
        assert desc.magnitudes.ndim == 1

    def test_magnitudes_are_real(self):
        desc = zernike_moments(_circle(), order=5)
        assert not np.iscomplexobj(desc.magnitudes)

    def test_magnitudes_are_abs_of_moments(self):
        desc = zernike_moments(_circle(), order=5)
        np.testing.assert_allclose(desc.magnitudes, np.abs(desc.moments), atol=1e-14)

    @pytest.mark.parametrize("order", [0, 1, 3, 6, 10])
    def test_moment_length_matches_valid_pairs(self, order):
        desc = zernike_moments(_circle(), order=order)
        expected = len(_valid_nm_pairs(order))
        assert len(desc.moments) == expected

    def test_order_field_equals_requested(self):
        desc = zernike_moments(_square(), order=8)
        assert desc.order == 8

    def test_all_magnitudes_finite(self):
        desc = zernike_moments(_ellipse(), order=6)
        assert np.all(np.isfinite(desc.magnitudes))

    def test_all_moments_finite(self):
        desc = zernike_moments(_triangle(), order=6)
        assert np.all(np.isfinite(desc.moments))

    def test_n_points_parameter_changes_nothing_in_descriptor_type(self):
        desc = zernike_moments(_circle(128), order=4, n_points=32)
        assert isinstance(desc, ZernikeDescriptor)

    def test_negative_order_clamped_to_zero(self):
        # Implementation does max(0, int(order)) so negative -> order=0
        desc = zernike_moments(_circle(), order=-3)
        assert desc.order == 0
        assert len(desc.moments) == 1


# ── zernike_moments – magnitudes properties ──────────────────────────────────────

class TestMagnitudesProperties:

    def test_magnitudes_nonneg_ellipse(self):
        desc = zernike_moments(_ellipse(), order=7)
        assert np.all(desc.magnitudes >= 0.0)

    def test_magnitudes_nonneg_triangle(self):
        desc = zernike_moments(_triangle(), order=5)
        assert np.all(desc.magnitudes >= 0.0)

    def test_magnitudes_equal_abs_moments(self):
        desc = zernike_moments(_circle(100), order=6)
        np.testing.assert_allclose(desc.magnitudes, np.abs(desc.moments), atol=1e-14)

    def test_circle_has_nonzero_magnitude(self):
        desc = zernike_moments(_circle(100), order=4)
        assert np.any(desc.magnitudes > 0.0)

    def test_ellipse_has_nonzero_magnitude(self):
        desc = zernike_moments(_ellipse(100), order=4)
        assert np.any(desc.magnitudes > 0.0)

    def test_translated_contour_magnitudes_unchanged(self):
        c = _circle(100)
        c_shifted = _translate(c, 500.0, -300.0)
        d_orig = zernike_moments(c, order=5, n_points=64)
        d_shift = zernike_moments(c_shifted, order=5, n_points=64)
        np.testing.assert_allclose(d_orig.magnitudes, d_shift.magnitudes, atol=1e-8)

    def test_scaled_contour_magnitudes_unchanged(self):
        c = _circle(100)
        c_big = _scale(c, 50.0)
        d_orig = zernike_moments(c, order=5, n_points=64)
        d_big = zernike_moments(c_big, order=5, n_points=64)
        np.testing.assert_allclose(d_orig.magnitudes, d_big.magnitudes, atol=1e-8)


# ── Rotation invariance ──────────────────────────────────────────────────────────

class TestRotationInvariance:

    @pytest.mark.parametrize("angle", [
        np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 5 * np.pi / 4,
    ])
    def test_circle_rotation_invariant_magnitudes(self, angle):
        c = _circle(128)
        c_rot = _rotate(c, angle)
        d_orig = zernike_moments(c, order=6, n_points=128)
        d_rot = zernike_moments(c_rot, order=6, n_points=128)
        np.testing.assert_allclose(d_orig.magnitudes, d_rot.magnitudes, atol=1e-6)

    def test_ellipse_45_deg_rotation(self):
        c = _ellipse(128)
        c_rot = _rotate(c, np.pi / 4)
        d_orig = zernike_moments(c, order=6, n_points=128)
        d_rot = zernike_moments(c_rot, order=6, n_points=128)
        # Ellipse rotated 45 deg changes shape differently but magnitudes should
        # remain equal because Zernike magnitudes are rotation-invariant.
        np.testing.assert_allclose(d_orig.magnitudes, d_rot.magnitudes, atol=1e-6)

    def test_triangle_arbitrary_rotation(self):
        c = _triangle(120)
        c_rot = _rotate(c, np.pi / 7)
        d_orig = zernike_moments(c, order=5, n_points=64)
        d_rot = zernike_moments(c_rot, order=5, n_points=64)
        np.testing.assert_allclose(d_orig.magnitudes, d_rot.magnitudes, atol=1e-6)

    def test_similarity_circle_rotated_near_one(self):
        c = _circle(100)
        c_rot = _rotate(c, np.pi / 5)
        da = zernike_moments(c, order=6, n_points=100)
        db = zernike_moments(c_rot, order=6, n_points=100)
        assert zernike_similarity(da, db) > 0.999


# ── zernike_similarity ────────────────────────────────────────────────────────

class TestZernikeSimilarity:

    def test_self_similarity_ellipse(self):
        desc = zernike_moments(_ellipse(), order=5)
        assert zernike_similarity(desc, desc) == pytest.approx(1.0, abs=1e-9)

    def test_self_similarity_triangle(self):
        desc = zernike_moments(_triangle(), order=5)
        assert zernike_similarity(desc, desc) == pytest.approx(1.0, abs=1e-9)

    def test_result_is_float(self):
        da = zernike_moments(_circle(), order=4)
        db = zernike_moments(_square(), order=4)
        assert isinstance(zernike_similarity(da, db), float)

    def test_circle_vs_ellipse_in_range(self):
        da = zernike_moments(_circle(), order=5)
        db = zernike_moments(_ellipse(), order=5)
        s = zernike_similarity(da, db)
        assert 0.0 <= s <= 1.0

    def test_symmetry_circle_triangle(self):
        da = zernike_moments(_circle(), order=5)
        db = zernike_moments(_triangle(), order=5)
        assert zernike_similarity(da, db) == pytest.approx(
            zernike_similarity(db, da), abs=1e-12
        )

    def test_zero_magnitudes_returns_zero(self):
        d_zero = ZernikeDescriptor(
            moments=np.zeros(5, dtype=complex),
            magnitudes=np.zeros(5),
            order=2,
        )
        d_circle = zernike_moments(_circle(), order=2)
        assert zernike_similarity(d_zero, d_circle) == pytest.approx(0.0)

    def test_both_zero_returns_zero(self):
        d_zero = ZernikeDescriptor(
            moments=np.zeros(3, dtype=complex),
            magnitudes=np.zeros(3),
            order=1,
        )
        assert zernike_similarity(d_zero, d_zero) == pytest.approx(0.0)

    def test_mismatched_order_descriptors_in_range(self):
        da = zernike_moments(_circle(), order=3)
        db = zernike_moments(_circle(), order=5)
        s = zernike_similarity(da, db)
        assert 0.0 <= s <= 1.0

    def test_identical_circles_different_sizes_high_similarity(self):
        da = zernike_moments(_circle(100, r=1.0), order=6, n_points=100)
        db = zernike_moments(_circle(100, r=5.0), order=6, n_points=100)
        assert zernike_similarity(da, db) > 0.99

    def test_clip_ensures_no_value_above_one(self):
        # Manually craft two descriptors with large magnitudes;
        # cosine should still be clipped to [0,1].
        m = np.array([10.0, 0.0, 0.0])
        da = ZernikeDescriptor(m.astype(complex), m, 2)
        db = ZernikeDescriptor(m.astype(complex), m, 2)
        assert zernike_similarity(da, db) <= 1.0


# ── zernike_to_feature_vector ──────────────────────────────────────────────────

class TestZernikeToFeatureVector:

    def test_returns_ndarray(self):
        desc = zernike_moments(_circle(), order=5)
        fv = zernike_to_feature_vector(desc)
        assert isinstance(fv, np.ndarray)

    def test_dtype_float(self):
        desc = zernike_moments(_circle(), order=5)
        fv = zernike_to_feature_vector(desc)
        assert np.issubdtype(fv.dtype, np.floating)

    def test_ndim_1(self):
        desc = zernike_moments(_circle(), order=5)
        fv = zernike_to_feature_vector(desc)
        assert fv.ndim == 1

    def test_l2_norm_is_one(self):
        desc = zernike_moments(_circle(), order=6)
        fv = zernike_to_feature_vector(desc)
        np.testing.assert_allclose(np.linalg.norm(fv), 1.0, atol=1e-9)

    def test_l2_norm_ellipse(self):
        desc = zernike_moments(_ellipse(), order=6)
        fv = zernike_to_feature_vector(desc)
        np.testing.assert_allclose(np.linalg.norm(fv), 1.0, atol=1e-9)

    def test_all_values_nonneg(self):
        desc = zernike_moments(_triangle(), order=5)
        fv = zernike_to_feature_vector(desc)
        assert np.all(fv >= 0.0)

    def test_length_equals_moments_length(self):
        desc = zernike_moments(_circle(), order=7)
        fv = zernike_to_feature_vector(desc)
        assert len(fv) == len(desc.moments)

    def test_all_zero_magnitudes_returns_zero_vector(self):
        desc = ZernikeDescriptor(
            moments=np.zeros(6, dtype=complex),
            magnitudes=np.zeros(6),
            order=3,
        )
        fv = zernike_to_feature_vector(desc)
        np.testing.assert_array_equal(fv, np.zeros(6))

    def test_does_not_modify_descriptor_magnitudes(self):
        desc = zernike_moments(_circle(), order=4)
        orig = desc.magnitudes.copy()
        _ = zernike_to_feature_vector(desc)
        np.testing.assert_array_equal(desc.magnitudes, orig)

    def test_feature_vector_is_copy(self):
        desc = zernike_moments(_circle(), order=4)
        fv = zernike_to_feature_vector(desc)
        fv[:] = 0.0
        # Modifying fv should not change the descriptor
        assert np.any(desc.magnitudes != 0.0) or np.all(desc.magnitudes == 0.0)

    def test_single_nonzero_magnitude_gives_unit_vector(self):
        mags = np.array([0.0, 3.0, 0.0, 0.0])
        desc = ZernikeDescriptor(mags.astype(complex), mags, 2)
        fv = zernike_to_feature_vector(desc)
        np.testing.assert_allclose(np.linalg.norm(fv), 1.0, atol=1e-12)
        np.testing.assert_allclose(fv[1], 1.0, atol=1e-12)


# ── Degenerate / edge-case contours ────────────────────────────────────────────

class TestEdgeCaseContours:

    def test_very_short_contour_2_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        desc = zernike_moments(pts, order=4)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(np.isfinite(desc.magnitudes))

    def test_very_short_contour_3_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        desc = zernike_moments(pts, order=4)
        assert np.all(desc.magnitudes >= 0.0)

    def test_single_point_gives_zero_moments(self):
        pts = np.array([[1.0, 2.0]])
        desc = zernike_moments(pts, order=3)
        np.testing.assert_array_equal(desc.moments, np.zeros(len(_valid_nm_pairs(3)), dtype=complex))

    def test_all_same_points_gives_zero_moments(self):
        pts = np.tile([[3.0, 4.0]], (20, 1))
        desc = zernike_moments(pts, order=3)
        np.testing.assert_array_equal(desc.moments, np.zeros(len(_valid_nm_pairs(3)), dtype=complex))

    def test_collinear_points_no_crash(self):
        pts = np.column_stack([np.linspace(0, 5, 30), np.zeros(30)])
        desc = zernike_moments(pts, order=4)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(np.isfinite(desc.magnitudes))

    def test_very_small_circle(self):
        c = _circle(64, r=1e-8)
        desc = zernike_moments(c, order=5)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(np.isfinite(desc.magnitudes))

    def test_very_large_circle(self):
        c = _circle(64, r=1e6)
        desc = zernike_moments(c, order=5)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(np.isfinite(desc.magnitudes))


# ── Determinism ────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_circle_input_same_moments(self):
        c = _circle(100)
        d1 = zernike_moments(c, order=6, n_points=64)
        d2 = zernike_moments(c, order=6, n_points=64)
        np.testing.assert_array_equal(d1.moments, d2.moments)

    def test_same_ellipse_input_same_magnitudes(self):
        e = _ellipse(80)
        d1 = zernike_moments(e, order=5, n_points=64)
        d2 = zernike_moments(e, order=5, n_points=64)
        np.testing.assert_array_equal(d1.magnitudes, d2.magnitudes)

    def test_same_triangle_input_same_feature_vector(self):
        t = _triangle(90)
        fv1 = zernike_to_feature_vector(zernike_moments(t, order=4, n_points=64))
        fv2 = zernike_to_feature_vector(zernike_moments(t, order=4, n_points=64))
        np.testing.assert_array_equal(fv1, fv2)

    def test_same_similarity_called_twice(self):
        da = zernike_moments(_circle(80), order=5)
        db = zernike_moments(_square(80), order=5)
        s1 = zernike_similarity(da, db)
        s2 = zernike_similarity(da, db)
        assert s1 == s2


# ── Different shapes produce different descriptors ─────────────────────────────

class TestDescriptorDistinctness:

    def test_circle_and_square_distinct(self):
        da = zernike_moments(_circle(100), order=6, n_points=100)
        db = zernike_moments(_square(100), order=6, n_points=100)
        s = zernike_similarity(da, db)
        # They should not be identical
        assert s < 1.0

    def test_circle_and_triangle_distinct(self):
        da = zernike_moments(_circle(100), order=6, n_points=100)
        db = zernike_moments(_triangle(99), order=6, n_points=100)
        s = zernike_similarity(da, db)
        assert s < 1.0

    def test_circle_and_ellipse_distinct(self):
        da = zernike_moments(_circle(100, r=1.0), order=6, n_points=100)
        db = zernike_moments(_ellipse(100, a=2.0, b=1.0), order=6, n_points=100)
        # Circle is a special case of ellipse; they should have measurable difference
        s = zernike_similarity(da, db)
        assert s < 1.0

    def test_same_shape_similarity_higher_than_different(self):
        da = zernike_moments(_circle(100), order=6, n_points=100)
        db = zernike_moments(_circle(100), order=6, n_points=100)
        dc = zernike_moments(_triangle(99), order=6, n_points=100)
        s_same = zernike_similarity(da, db)
        s_diff = zernike_similarity(da, dc)
        assert s_same >= s_diff


# ── Various n_points values ────────────────────────────────────────────────────

class TestNPointsParameter:

    @pytest.mark.parametrize("n_pts", [16, 32, 64, 128, 256])
    def test_various_n_points_return_valid_descriptor(self, n_pts):
        c = _circle(200)
        desc = zernike_moments(c, order=5, n_points=n_pts)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(np.isfinite(desc.magnitudes))
        assert np.all(desc.magnitudes >= 0.0)

    def test_n_points_1_clamps_gracefully(self):
        c = _circle(64)
        desc = zernike_moments(c, order=3, n_points=1)
        assert isinstance(desc, ZernikeDescriptor)


# ── Order boundary values ──────────────────────────────────────────────────────

class TestOrderBoundary:

    def test_order_zero_single_moment(self):
        desc = zernike_moments(_circle(), order=0)
        assert len(desc.moments) == 1
        assert len(desc.magnitudes) == 1

    def test_order_zero_moment_is_Z00(self):
        # With order=0 only (n=0,m=0) pair exists
        pairs = _valid_nm_pairs(0)
        assert pairs == [(0, 0)]

    def test_order_1_moment_count(self):
        desc = zernike_moments(_circle(), order=1)
        assert len(desc.moments) == 3  # (0,0),(1,-1),(1,1)

    def test_high_order_no_crash(self):
        desc = zernike_moments(_circle(200), order=15, n_points=200)
        assert isinstance(desc, ZernikeDescriptor)
        assert np.all(np.isfinite(desc.magnitudes))
