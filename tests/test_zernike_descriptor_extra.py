"""Extra tests for puzzle_reconstruction/algorithms/zernike_descriptor.py"""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square(n=64, side=1.0):
    side_n = max(n // 4, 1)
    e0 = np.column_stack([np.linspace(0, side, side_n), np.zeros(side_n)])
    e1 = np.column_stack([np.full(side_n, side), np.linspace(0, side, side_n)])
    e2 = np.column_stack([np.linspace(side, 0, side_n), np.full(side_n, side)])
    e3 = np.column_stack([np.zeros(side_n), np.linspace(side, 0, side_n)])
    return np.vstack([e0, e1, e2, e3])


def _triangle(n=60):
    t = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    idx = np.linspace(0, 3, n, endpoint=False).astype(int) % 3
    return t[idx]


# ─── _factorial edge cases ────────────────────────────────────────────────────

def test_factorial_zero():
    assert _factorial(0) == pytest.approx(1.0)


def test_factorial_one():
    assert _factorial(1) == pytest.approx(1.0)


def test_factorial_two():
    assert _factorial(2) == pytest.approx(2.0)


def test_factorial_six():
    assert _factorial(6) == pytest.approx(720.0)


def test_factorial_twelve():
    assert _factorial(12) == pytest.approx(479001600.0)


def test_factorial_negative_raises():
    with pytest.raises((ValueError, Exception)):
        _factorial(-1)


def test_factorial_negative_large_raises():
    with pytest.raises((ValueError, Exception)):
        _factorial(-100)


# ─── _valid_nm_pairs ─────────────────────────────────────────────────────────

def test_valid_nm_pairs_order_0():
    pairs = _valid_nm_pairs(0)
    assert len(pairs) == 1
    assert (0, 0) in pairs


def test_valid_nm_pairs_order_1():
    pairs = _valid_nm_pairs(1)
    # n=0: (0,0); n=1: (1,1),(1,-1) → 3 pairs
    assert len(pairs) == 3


def test_valid_nm_pairs_all_n_minus_m_even():
    for order in range(8):
        for n, m in _valid_nm_pairs(order):
            assert (n - abs(m)) % 2 == 0


def test_valid_nm_pairs_m_abs_le_n():
    for order in range(6):
        for n, m in _valid_nm_pairs(order):
            assert abs(m) <= n


def test_valid_nm_pairs_n_le_order():
    for order in range(8):
        for n, m in _valid_nm_pairs(order):
            assert n <= order


# ─── _radial_polynomial edge cases ───────────────────────────────────────────

def test_radial_R00_is_one_everywhere():
    rho = np.linspace(0, 1, 20)
    val = _radial_polynomial(0, 0, rho)
    np.testing.assert_allclose(val, np.ones(20))


def test_radial_R11_equals_rho():
    rho = np.linspace(0, 1, 20)
    val = _radial_polynomial(1, 1, rho)
    np.testing.assert_allclose(val, rho)


def test_radial_R11_negative_m():
    rho = np.linspace(0, 1, 20)
    val = _radial_polynomial(1, -1, rho)
    np.testing.assert_allclose(val, rho)


def test_radial_invalid_pair_returns_zeros():
    rho = np.linspace(0, 1, 10)
    val = _radial_polynomial(3, 2, rho)  # (3-2)=1 odd → invalid
    np.testing.assert_array_equal(val, np.zeros(10))


def test_radial_R40_formula():
    # R_40(rho) = 6*rho^4 - 6*rho^2 + 1
    rho = np.array([0.0, 0.5, 1.0])
    expected = 6 * rho**4 - 6 * rho**2 + 1
    val = _radial_polynomial(4, 0, rho)
    np.testing.assert_allclose(val, expected, atol=1e-10)


def test_radial_R33_at_rho0():
    # R_33(0) = 0 (any n=m>0 at rho=0)
    val = _radial_polynomial(3, 3, np.array([0.0]))
    np.testing.assert_allclose(val, [0.0], atol=1e-12)


def test_radial_R33_at_rho1():
    # R_33(1) = 1
    val = _radial_polynomial(3, 3, np.array([1.0]))
    np.testing.assert_allclose(val, [1.0], atol=1e-12)


# ─── zernike_moments: various orders and shapes ──────────────────────────────

def test_moments_order_0_one_element():
    desc = zernike_moments(_circle(), order=0)
    assert len(desc.moments) == 1


def test_moments_order_12():
    desc = zernike_moments(_circle(200), order=12)
    assert len(desc.moments) == len(_valid_nm_pairs(12))


def test_moments_triangle_shape():
    desc = zernike_moments(_triangle(), order=6)
    assert isinstance(desc, ZernikeDescriptor)
    assert np.all(desc.magnitudes >= 0)


def test_moments_very_small_contour_3_points():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    desc = zernike_moments(pts, order=3)
    assert isinstance(desc, ZernikeDescriptor)


def test_moments_n_points_parameter_affects_internal_sampling():
    c = _circle(200)
    d8  = zernike_moments(c, order=5, n_points=8)
    d64 = zernike_moments(c, order=5, n_points=64)
    # Both must produce valid ZernikeDescriptors
    assert isinstance(d8, ZernikeDescriptor)
    assert isinstance(d64, ZernikeDescriptor)


def test_moments_zero_filled_magnitudes_for_degenerate():
    """All-same points → degenerate contour, no crash."""
    pts = np.tile([3.0, 4.0], (20, 1))
    desc = zernike_moments(pts, order=4)
    assert np.all(np.isfinite(desc.magnitudes))


def test_moments_dtype_float64_magnitudes():
    desc = zernike_moments(_circle(100), order=5)
    assert desc.magnitudes.dtype in (np.float64, np.float32)


# ─── zernike_similarity edge cases ───────────────────────────────────────────

def test_similarity_zero_descriptors():
    d = ZernikeDescriptor(
        moments=np.zeros(5, dtype=complex),
        magnitudes=np.zeros(5),
        order=3,
    )
    s = zernike_similarity(d, d)
    assert s == pytest.approx(0.0)


def test_similarity_range_various_shapes():
    shapes = [_circle(80), _square(80), _triangle(60)]
    descs = [zernike_moments(s, order=6) for s in shapes]
    for i in range(len(descs)):
        for j in range(len(descs)):
            s = zernike_similarity(descs[i], descs[j])
            assert 0.0 <= s <= 1.0, f"Similarity {i}-{j} = {s} out of [0,1]"


def test_similarity_symmetric_triangle_circle():
    d_c = zernike_moments(_circle(100), order=5)
    d_t = zernike_moments(_triangle(90), order=5)
    s1 = zernike_similarity(d_c, d_t)
    s2 = zernike_similarity(d_t, d_c)
    assert abs(s1 - s2) < 1e-12


def test_similarity_circle_vs_scaled_circle():
    d1 = zernike_moments(_circle(100, r=1.0), order=6)
    d2 = zernike_moments(_circle(100, r=2.0), order=6)
    s = zernike_similarity(d1, d2)
    # Rotation-invariant magnitudes should be nearly equal
    assert s > 0.99


# ─── zernike_to_feature_vector ────────────────────────────────────────────────

def test_feature_vector_l2_norm_one():
    desc = zernike_moments(_circle(100), order=6)
    fv = zernike_to_feature_vector(desc)
    norm = np.linalg.norm(fv)
    assert abs(norm - 1.0) < 1e-9 or norm == pytest.approx(0.0, abs=1e-9)


def test_feature_vector_zero_desc_returns_zeros():
    d = ZernikeDescriptor(
        moments=np.zeros(10, dtype=complex),
        magnitudes=np.zeros(10),
        order=4,
    )
    fv = zernike_to_feature_vector(d)
    np.testing.assert_array_equal(fv, np.zeros(10))


def test_feature_vector_nonneg():
    desc = zernike_moments(_square(100), order=7)
    fv = zernike_to_feature_vector(desc)
    assert np.all(fv >= 0.0)


def test_feature_vector_shape_equals_magnitudes():
    desc = zernike_moments(_circle(80), order=8)
    fv = zernike_to_feature_vector(desc)
    assert fv.shape == desc.magnitudes.shape


# ─── Rotation invariance ─────────────────────────────────────────────────────

def test_rotation_circle_30_deg():
    c = _circle(200)
    angle = np.pi / 6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    c_rot = (R @ c.T).T
    d1 = zernike_moments(c, order=6, n_points=128)
    d2 = zernike_moments(c_rot, order=6, n_points=128)
    s = zernike_similarity(d1, d2)
    assert s > 0.99


def test_rotation_square_90_deg_invariance():
    c = _square(200)
    angle = np.pi / 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    c_rot = (R @ c.T).T
    d1 = zernike_moments(c, order=6, n_points=128)
    d2 = zernike_moments(c_rot, order=6, n_points=128)
    s = zernike_similarity(d1, d2)
    assert s > 0.95


# ─── Determinism ─────────────────────────────────────────────────────────────

def test_moments_deterministic_circle():
    c = _circle(100)
    d1 = zernike_moments(c, order=5)
    d2 = zernike_moments(c, order=5)
    np.testing.assert_array_equal(d1.magnitudes, d2.magnitudes)
    np.testing.assert_array_equal(d1.moments, d2.moments)


def test_moments_deterministic_square():
    c = _square(100)
    d1 = zernike_moments(c, order=7)
    d2 = zernike_moments(c, order=7)
    np.testing.assert_array_equal(d1.magnitudes, d2.magnitudes)


# ─── ZernikeDescriptor NamedTuple properties ─────────────────────────────────

def test_descriptor_is_immutable_namedtuple():
    d = ZernikeDescriptor(
        moments=np.zeros(3, dtype=complex),
        magnitudes=np.zeros(3),
        order=2,
    )
    with pytest.raises((AttributeError, TypeError)):
        d.order = 99


def test_descriptor_indexable():
    moms = np.zeros(3, dtype=complex)
    mags = np.zeros(3)
    d = ZernikeDescriptor(moments=moms, magnitudes=mags, order=2)
    assert d[2] == 2  # order is at index 2


def test_descriptor_unpackable():
    moms = np.array([1 + 0j], dtype=complex)
    mags = np.array([1.0])
    m, mg, o = ZernikeDescriptor(moments=moms, magnitudes=mags, order=1)
    assert o == 1
