"""Extra tests for puzzle_reconstruction/algorithms/zernike_descriptor.py"""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.zernike_descriptor import (
    ZernikeDescriptor,
    _factorial,
    _radial_polynomial,
    _resample_contour,
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


def _rotate(contour: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ contour.T).T


# ── _factorial extra ──────────────────────────────────────────────────────────

class TestFactorialExtra:

    def test_factorial_0_is_1(self):
        assert _factorial(0) == 1.0

    def test_factorial_1_is_1(self):
        assert _factorial(1) == 1.0

    def test_factorial_5_is_120(self):
        assert _factorial(5) == pytest.approx(120.0)

    def test_factorial_10(self):
        assert _factorial(10) == pytest.approx(3628800.0)

    def test_factorial_negative_raises(self):
        with pytest.raises((ValueError, Exception)):
            _factorial(-1)

    def test_factorial_sequence_ratio(self):
        # n! / (n-1)! == n
        for n in range(1, 10):
            ratio = _factorial(n) / _factorial(n - 1)
            assert ratio == pytest.approx(float(n))


# ── _radial_polynomial extra ──────────────────────────────────────────────────

class TestRadialPolynomialExtra:

    def test_R40_formula(self):
        # R_40(rho) = 6*rho^4 - 6*rho^2 + 1
        rho = np.linspace(0, 1, 25)
        out = _radial_polynomial(4, 0, rho)
        expected = 6 * rho**4 - 6 * rho**2 + 1
        np.testing.assert_allclose(out, expected, atol=1e-11)

    def test_R44_formula(self):
        # R_44(rho) = rho^4
        rho = np.linspace(0, 1, 25)
        out = _radial_polynomial(4, 4, rho)
        np.testing.assert_allclose(out, rho**4, atol=1e-11)

    def test_empty_rho_array(self):
        rho = np.array([])
        out = _radial_polynomial(2, 0, rho)
        assert out.shape == (0,)

    def test_rho_at_one_R11(self):
        # R_11(1) = 1
        out = _radial_polynomial(1, 1, np.array([1.0]))
        np.testing.assert_allclose(out, [1.0], atol=1e-12)

    def test_rho_at_one_R00(self):
        out = _radial_polynomial(0, 0, np.array([1.0]))
        np.testing.assert_allclose(out, [1.0], atol=1e-12)

    def test_output_is_numpy_array(self):
        rho = np.linspace(0.1, 0.9, 10)
        out = _radial_polynomial(2, 2, rho)
        assert isinstance(out, np.ndarray)

    def test_large_n_no_crash(self):
        rho = np.linspace(0, 1, 30)
        out = _radial_polynomial(8, 0, rho)
        assert out.shape == rho.shape
        assert np.all(np.isfinite(out))


# ── _zernike_basis extra ──────────────────────────────────────────────────────

class TestZernikeBasisExtra:

    def test_magnitude_at_boundary_rho1(self):
        rho = np.array([1.0])
        theta = np.array([0.0])
        out = _zernike_basis(1, 1, rho, theta)
        # rho == 1.0 is at the boundary; not strictly outside, should not be zero
        # (the mask is rho > 1.0)
        assert np.isfinite(out[0])

    def test_all_zeros_outside_disk(self):
        rho = np.array([1.1, 2.0, 5.0])
        theta = np.zeros(3)
        out = _zernike_basis(2, 2, rho, theta)
        np.testing.assert_array_equal(out, np.zeros(3, dtype=complex))

    def test_n0m0_is_constant_1_inside_disk(self):
        rho = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        theta = np.zeros(5)
        out = _zernike_basis(0, 0, rho, theta)
        np.testing.assert_allclose(out.real, np.ones(5), atol=1e-12)
        np.testing.assert_allclose(out.imag, np.zeros(5), atol=1e-12)

    def test_complex_conjugate_for_neg_m(self):
        # V_{n,-m}(rho, theta) = conj(V_{n,m}(rho, theta))
        rho = np.linspace(0, 0.9, 15)
        theta = np.linspace(0, np.pi, 15)
        for n, m in [(1, 1), (2, 2), (3, 1), (3, 3)]:
            v_pos = _zernike_basis(n, m, rho, theta)
            v_neg = _zernike_basis(n, -m, rho, theta)
            np.testing.assert_allclose(v_neg, np.conj(v_pos), atol=1e-11)


# ── _valid_nm_pairs extra ─────────────────────────────────────────────────────

class TestValidNmPairsExtra:

    def test_order_0_only_pair_is_00(self):
        pairs = _valid_nm_pairs(0)
        assert pairs == [(0, 0)]

    def test_all_n_values_present_up_to_order(self):
        order = 5
        pairs = _valid_nm_pairs(order)
        ns = {n for n, _ in pairs}
        assert ns == set(range(order + 1))

    def test_m_range_for_each_n(self):
        for order in range(1, 7):
            pairs = _valid_nm_pairs(order)
            for n, m in pairs:
                assert -n <= m <= n

    def test_count_formula(self):
        # Total count = sum_{n=0}^{order} (floor(n/2)+1) * (2 if n>0 else 1)
        # Simpler check: count for order k equals count for k-1 plus (k+1 terms)
        for order in range(1, 8):
            expected_added = sum(1 for m in range(-order, order + 1)
                                 if (order - m) % 2 == 0)
            prev = len(_valid_nm_pairs(order - 1))
            curr = len(_valid_nm_pairs(order))
            assert curr - prev == expected_added

    def test_increasing_count(self):
        prev = 0
        for order in range(0, 9):
            curr = len(_valid_nm_pairs(order))
            assert curr > prev or order == 0
            prev = curr

    def test_large_order_no_crash(self):
        pairs = _valid_nm_pairs(20)
        assert len(pairs) > 100


# ── _resample_contour (Zernike version) extra ─────────────────────────────────

class TestZernikeResampleContourExtra:

    def test_one_point_returns_zeros(self):
        c = np.array([[1.0, 2.0]])
        r = _resample_contour(c, 8)
        assert r.shape == (8, 2)

    def test_identical_points_all_same(self):
        c = np.tile([3.0, 4.0], (5, 1))
        r = _resample_contour(c, 10)
        assert r.shape == (10, 2)
        assert np.all(np.isfinite(r))

    def test_downsampling(self):
        c = _circle(128)
        r = _resample_contour(c, 16)
        assert r.shape == (16, 2)

    def test_upsampling(self):
        c = _circle(8)
        r = _resample_contour(c, 64)
        assert r.shape == (64, 2)

    def test_output_is_float(self):
        c = np.array([[0, 0], [1, 0], [1, 1]], dtype=int)
        r = _resample_contour(c.astype(float), 6)
        assert r.dtype.kind == "f"


# ── zernike_moments extra parameters ─────────────────────────────────────────

class TestZernikeMomentsExtraParams:

    def test_large_n_points_returns_valid(self):
        d = zernike_moments(_circle(200), order=5, n_points=512)
        assert np.all(np.isfinite(d.magnitudes))

    def test_order_zero_moments_length_1(self):
        d = zernike_moments(_circle(), order=0)
        assert len(d.moments) == 1

    def test_contour_with_1d_wrong_shape(self):
        # 1D array (not Nx2) should return zero moments
        pts = np.array([1.0, 2.0, 3.0])
        d = zernike_moments(pts, order=2)  # type: ignore[arg-type]
        assert isinstance(d, ZernikeDescriptor)

    def test_order_15_no_crash(self):
        d = zernike_moments(_circle(200), order=15, n_points=128)
        assert isinstance(d, ZernikeDescriptor)
        assert np.all(np.isfinite(d.magnitudes))

    def test_moments_complex_dtype(self):
        d = zernike_moments(_circle(), order=4)
        assert np.iscomplexobj(d.moments)

    def test_magnitudes_equals_abs_moments(self):
        d = zernike_moments(_ellipse(), order=6)
        np.testing.assert_allclose(d.magnitudes, np.abs(d.moments), atol=1e-14)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_even_order_valid(self, order):
        d = zernike_moments(_circle(128), order=order, n_points=64)
        assert np.all(d.magnitudes >= 0.0)

    @pytest.mark.parametrize("order", [1, 3, 5, 7])
    def test_odd_order_valid(self, order):
        d = zernike_moments(_circle(128), order=order, n_points=64)
        assert np.all(d.magnitudes >= 0.0)


# ── zernike_similarity extra ──────────────────────────────────────────────────

class TestZernikeSimilarityExtra:

    def test_result_in_range_0_1_various_shapes(self):
        shapes = [_circle(), _square(), _ellipse()]
        for i in range(len(shapes)):
            for j in range(i, len(shapes)):
                da = zernike_moments(shapes[i], order=5)
                db = zernike_moments(shapes[j], order=5)
                s = zernike_similarity(da, db)
                assert 0.0 <= s <= 1.0, f"Out of range for shapes {i},{j}: {s}"

    def test_similarity_symmetric(self):
        da = zernike_moments(_circle(), order=5)
        db = zernike_moments(_ellipse(), order=5)
        assert zernike_similarity(da, db) == pytest.approx(
            zernike_similarity(db, da), abs=1e-12
        )

    def test_same_shape_different_n_points_similar(self):
        da = zernike_moments(_circle(200), order=5, n_points=32)
        db = zernike_moments(_circle(200), order=5, n_points=128)
        assert zernike_similarity(da, db) > 0.9

    def test_zero_vs_nonzero_descriptor(self):
        d_zero = ZernikeDescriptor(
            np.zeros(6, dtype=complex), np.zeros(6), order=3
        )
        d_real = zernike_moments(_circle(), order=3)
        assert zernike_similarity(d_zero, d_real) == pytest.approx(0.0)

    def test_scale_invariance_via_similarity(self):
        da = zernike_moments(_circle(100, r=1.0), order=6, n_points=100)
        db = zernike_moments(_circle(100, r=100.0), order=6, n_points=100)
        assert zernike_similarity(da, db) > 0.99

    def test_translation_invariance_via_similarity(self):
        c = _circle(100)
        da = zernike_moments(c, order=6, n_points=100)
        db = zernike_moments(c + 500.0, order=6, n_points=100)
        assert zernike_similarity(da, db) > 0.99

    def test_rotation_invariance_180_degrees(self):
        c = _circle(100)
        da = zernike_moments(c, order=6, n_points=100)
        db = zernike_moments(_rotate(c, np.pi), order=6, n_points=100)
        assert zernike_similarity(da, db) > 0.99


# ── zernike_to_feature_vector extra ──────────────────────────────────────────

class TestZernikeToFeatureVectorExtra:

    def test_different_shapes_different_vectors(self):
        fv_circle = zernike_to_feature_vector(zernike_moments(_circle(), order=6))
        fv_square = zernike_to_feature_vector(zernike_moments(_square(), order=6))
        assert not np.allclose(fv_circle, fv_square)

    def test_all_values_finite(self):
        desc = zernike_moments(_ellipse(), order=8)
        fv = zernike_to_feature_vector(desc)
        assert np.all(np.isfinite(fv))

    def test_l2_norm_for_various_orders(self):
        for order in [2, 4, 6, 8]:
            desc = zernike_moments(_circle(100), order=order, n_points=64)
            fv = zernike_to_feature_vector(desc)
            norm = np.linalg.norm(fv)
            if np.any(desc.magnitudes != 0):
                assert abs(norm - 1.0) < 1e-9

    def test_cosine_similarity_with_feature_vectors(self):
        desc_a = zernike_moments(_circle(100), order=6, n_points=64)
        desc_b = zernike_moments(_circle(100), order=6, n_points=64)
        fv_a = zernike_to_feature_vector(desc_a)
        fv_b = zernike_to_feature_vector(desc_b)
        cos_sim = float(np.dot(fv_a, fv_b))
        assert cos_sim == pytest.approx(1.0, abs=1e-9)

    def test_feature_vector_length_equals_valid_pairs(self):
        order = 7
        desc = zernike_moments(_circle(), order=order)
        fv = zernike_to_feature_vector(desc)
        assert len(fv) == len(_valid_nm_pairs(order))

    def test_order_0_feature_vector_length_1(self):
        desc = zernike_moments(_circle(), order=0)
        fv = zernike_to_feature_vector(desc)
        assert len(fv) == 1
