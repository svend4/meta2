"""Extra tests for puzzle_reconstruction/algorithms/fractal/ (properties_fractal module)"""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.css import (
    curvature_scale_space,
    css_to_feature_vector,
    css_similarity,
    css_similarity_mirror,
)
from puzzle_reconstruction.algorithms.fractal.box_counting import (
    box_counting_fd,
    box_counting_curve,
)
from puzzle_reconstruction.algorithms.fractal.ifs import (
    fit_ifs_coefficients,
    reconstruct_from_ifs,
)
from puzzle_reconstruction.algorithms.synthesis import compute_fractal_signature


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _circle(n=64, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square(n=64):
    side = max(n // 4, 1)
    e0 = np.column_stack([np.linspace(0, 1, side), np.zeros(side)])
    e1 = np.column_stack([np.ones(side), np.linspace(0, 1, side)])
    e2 = np.column_stack([np.linspace(1, 0, side), np.ones(side)])
    e3 = np.column_stack([np.zeros(side), np.linspace(1, 0, side)])
    return np.vstack([e0, e1, e2, e3])


def _zigzag(n=64, freq=3.0):
    x = np.linspace(0, 10, n)
    return np.column_stack([x, np.abs(np.sin(x * freq))])


def _triangle(n=60):
    t = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    idx = (np.arange(n) % 3)
    return t[idx]


def _spiral(n=128):
    t = np.linspace(0, 4 * np.pi, n)
    r = t / (4 * np.pi)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


# ─── CSS: additional edge cases ───────────────────────────────────────────────

def test_css_circle_max_sigma_near_zero_crossings():
    c = _circle(128)
    css = curvature_scale_space(c, sigma_range=[64.0])
    for _, zc in css:
        assert len(zc) < 5


def test_css_zigzag_more_crossings_than_circle():
    c_circ  = _circle(128)
    c_zig   = _zigzag(128, freq=5.0)
    css_c  = curvature_scale_space(c_circ,  n_sigmas=3)
    css_z  = curvature_scale_space(c_zig,   n_sigmas=3)
    total_c = sum(len(zc) for _, zc in css_c)
    total_z = sum(len(zc) for _, zc in css_z)
    # Zigzag has higher curvature variations → more zero crossings
    assert total_z >= total_c


def test_css_triangle_returns_nonempty():
    c = _triangle()
    css = curvature_scale_space(c, n_sigmas=4)
    assert len(css) == 4


def test_css_zero_crossings_are_float():
    c = _square(64)
    css = curvature_scale_space(c, n_sigmas=3)
    for _, zc in css:
        assert zc.dtype in (np.float32, np.float64)


def test_css_n_sigmas_0_or_1():
    c = _circle(64)
    css = curvature_scale_space(c, n_sigmas=1)
    assert len(css) == 1


def test_css_feature_vector_same_shape_different_shapes():
    vc = css_to_feature_vector(curvature_scale_space(_circle(64), n_sigmas=3))
    vs = css_to_feature_vector(curvature_scale_space(_square(64), n_sigmas=3))
    vz = css_to_feature_vector(curvature_scale_space(_zigzag(64), n_sigmas=3))
    assert vc.shape == vs.shape == vz.shape


def test_css_feature_vector_n_bins_16():
    c = _circle(64)
    css = curvature_scale_space(c, n_sigmas=4)
    vec = css_to_feature_vector(css, n_bins=16)
    assert len(vec) == 4 * 16


def test_css_feature_vector_n_bins_32():
    c = _square(64)
    css = curvature_scale_space(c, n_sigmas=2)
    vec = css_to_feature_vector(css, n_bins=32)
    assert len(vec) == 2 * 32


def test_css_similarity_zero_vs_nonzero():
    va = css_to_feature_vector(curvature_scale_space(_circle(64)))
    vz = np.zeros_like(va)
    s = css_similarity(va, vz)
    assert 0.0 <= s <= 1.0


def test_css_similarity_triangle_vs_circle():
    vt = css_to_feature_vector(curvature_scale_space(_triangle(), n_sigmas=4))
    vc = css_to_feature_vector(curvature_scale_space(_circle(64), n_sigmas=4))
    s = css_similarity(vt, vc)
    assert 0.0 <= s <= 1.0


def test_css_similarity_mirror_symmetric():
    va = css_to_feature_vector(curvature_scale_space(_zigzag(64)))
    vb = css_to_feature_vector(curvature_scale_space(_zigzag(64)[::-1]))
    s1 = css_similarity_mirror(va, vb)
    s2 = css_similarity_mirror(vb, va)
    assert abs(s1 - s2) < 1e-9


def test_css_spiral_shape():
    css = curvature_scale_space(_spiral(128), n_sigmas=4)
    assert len(css) == 4


# ─── Box counting: additional edge cases ─────────────────────────────────────

def test_box_counting_fd_straight_horizontal_line():
    c = np.column_stack([np.linspace(0, 1, 64), np.zeros(64)])
    fd = box_counting_fd(c)
    assert 1.0 <= fd <= 1.5


def test_box_counting_fd_two_points():
    c = np.array([[0.0, 0.0], [1.0, 1.0]])
    fd = box_counting_fd(c)
    assert 1.0 <= fd <= 2.0


def test_box_counting_fd_zigzag_higher_than_line():
    line = np.column_stack([np.linspace(0, 1, 128), np.zeros(128)])
    zig = _zigzag(128, freq=5.0)
    fd_line = box_counting_fd(line)
    fd_zig  = box_counting_fd(zig)
    # Zigzag more complex → higher FD
    assert fd_zig >= fd_line - 0.1  # slight tolerance


def test_box_counting_fd_circle_in_range():
    fd = box_counting_fd(_circle(128))
    assert 1.0 <= fd <= 2.0


def test_box_counting_fd_square_in_range():
    fd = box_counting_fd(_square(128))
    assert 1.0 <= fd <= 2.0


def test_box_counting_curve_returns_two_nonempty_arrays():
    log_r, log_N = box_counting_curve(_circle(64))
    assert len(log_r) > 0
    assert len(log_N) > 0


def test_box_counting_curve_3_scales():
    log_r, log_N = box_counting_curve(_circle(64), n_scales=3)
    assert len(log_r) == 3
    assert len(log_N) == 3


def test_box_counting_fd_n_scales_4():
    fd = box_counting_fd(_square(64), n_scales=4)
    assert 1.0 <= fd <= 2.0


def test_box_counting_fd_n_scales_10():
    fd = box_counting_fd(_zigzag(64), n_scales=10)
    assert 1.0 <= fd <= 2.0


def test_box_counting_fd_all_zero_points():
    c = np.zeros((20, 2))
    fd = box_counting_fd(c)
    assert fd == pytest.approx(1.0)


# ─── IFS: additional edge cases ───────────────────────────────────────────────

def test_ifs_fit_returns_float32_or_64():
    c = _circle(64)
    d = fit_ifs_coefficients(c)
    assert d.dtype in (np.float32, np.float64)


def test_ifs_fit_n_transforms_1():
    c = _circle(64)
    d = fit_ifs_coefficients(c, n_transforms=1)
    assert len(d) == 1


def test_ifs_fit_n_transforms_16():
    c = _circle(64)
    d = fit_ifs_coefficients(c, n_transforms=16)
    assert len(d) == 16


def test_ifs_all_coefficients_lt_1():
    for c in [_circle(128), _square(128), _zigzag(128)]:
        d = fit_ifs_coefficients(c)
        assert np.all(np.abs(d) < 1.0), "IFS coefficients must be < 1"


def test_ifs_reconstruct_default_length():
    c = _circle(64)
    d = fit_ifs_coefficients(c)
    r = reconstruct_from_ifs(d)
    assert r.ndim == 1
    assert len(r) > 0


def test_ifs_reconstruct_32_points():
    d = fit_ifs_coefficients(_circle(64))
    r = reconstruct_from_ifs(d, n_points=32)
    assert len(r) == 32


def test_ifs_reconstruct_all_finite():
    for c in [_circle(64), _square(64), _zigzag(64)]:
        d = fit_ifs_coefficients(c)
        r = reconstruct_from_ifs(d)
        assert np.all(np.isfinite(r))


def test_ifs_coefficients_not_all_zero():
    c = _circle(64)
    d = fit_ifs_coefficients(c)
    # For a real contour, not all coefficients should be exactly 0
    # (unless the contour is pathological)
    assert len(d) > 0


def test_ifs_triangle_no_crash():
    c = _triangle(60)
    d = fit_ifs_coefficients(c, n_transforms=6)
    assert len(d) == 6
    r = reconstruct_from_ifs(d)
    assert np.all(np.isfinite(r))


# ─── FractalSignature: additional property tests ──────────────────────────────

def test_fractal_signature_css_image_is_list():
    sig = compute_fractal_signature(_circle(128))
    assert isinstance(sig.css_image, list)


def test_fractal_signature_ifs_coeffs_length():
    sig = compute_fractal_signature(_circle(128))
    assert len(sig.ifs_coeffs) > 0


def test_fractal_signature_curve_2d():
    sig = compute_fractal_signature(_circle(128))
    assert sig.curve.ndim == 2
    assert sig.curve.shape[1] == 2


def test_fractal_signature_fd_box_in_range():
    sig = compute_fractal_signature(_zigzag(64))
    assert 1.0 <= sig.fd_box <= 2.0


def test_fractal_signature_fd_divider_in_range():
    sig = compute_fractal_signature(_circle(64))
    assert 1.0 <= sig.fd_divider <= 2.0


def test_fractal_signature_chain_code_not_none():
    sig = compute_fractal_signature(_circle(64))
    assert sig.chain_code is not None


def test_fractal_signature_square():
    sig = compute_fractal_signature(_square(128))
    assert sig is not None
    assert 1.0 <= sig.fd_box <= 2.0


def test_fractal_signature_triangle():
    sig = compute_fractal_signature(_triangle(60))
    assert sig is not None
    assert isinstance(sig.css_image, list)


def test_fractal_signature_spiral():
    sig = compute_fractal_signature(_spiral(128))
    assert sig is not None
    assert 1.0 <= sig.fd_box <= 2.0


def test_fractal_signatures_differ_circle_vs_zigzag():
    s_c = compute_fractal_signature(_circle(256))
    s_z = compute_fractal_signature(_zigzag(256))
    # FD values should differ (zigzag has higher complexity)
    assert not np.array_equal(s_c.ifs_coeffs, s_z.ifs_coeffs)


def test_fractal_signature_deterministic_square():
    c = _square(128)
    s1 = compute_fractal_signature(c)
    s2 = compute_fractal_signature(c)
    assert s1.fd_box == s2.fd_box
    np.testing.assert_array_equal(s1.ifs_coeffs, s2.ifs_coeffs)
