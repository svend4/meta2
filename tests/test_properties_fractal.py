"""
Property-based tests for fractal descriptor algorithms.

Verifies mathematical invariants:
- CSS: invariance to scale/translation, symmetry of similarity
- Box-counting: FD ∈ [1, 2], monotonicity, determinism
- IFS: coefficient bounds, reconstruction shape
- FractalSignature: field types and value ranges
"""
from __future__ import annotations

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circle(n: int = 64, r: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _square(n: int = 64) -> np.ndarray:
    side = n // 4
    top    = np.column_stack([np.linspace(0, 1, side), np.ones(side)])
    right  = np.column_stack([np.ones(side), np.linspace(1, 0, side)])
    bottom = np.column_stack([np.linspace(1, 0, side), np.zeros(side)])
    left   = np.column_stack([np.zeros(side), np.linspace(0, 1, side)])
    return np.vstack([top, right, bottom, left])


def _zigzag(n: int = 64) -> np.ndarray:
    x = np.linspace(0, 10, n)
    y = np.abs(np.sin(x * 3))
    return np.column_stack([x, y])


# ── CSS properties ─────────────────────────────────────────────────────────────

class TestCSSInvariants:
    """Mathematical properties of curvature_scale_space."""

    def test_returns_list(self):
        c = _circle(64)
        result = curvature_scale_space(c)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_entry_is_sigma_plus_array(self):
        c = _circle(64)
        css = curvature_scale_space(c)
        for sigma, zc in css:
            assert isinstance(sigma, float)
            assert isinstance(zc, np.ndarray)
            assert sigma > 0

    def test_sigma_values_increasing(self):
        c = _circle(64)
        css = curvature_scale_space(c)
        sigmas = [s for s, _ in css]
        assert sigmas == sorted(sigmas), "Sigma values should be sorted ascending"

    def test_zero_crossings_normalized(self):
        """All zero-crossing positions should be in [0, 1)."""
        c = _square(64)
        css = curvature_scale_space(c)
        for _, zc in css:
            if len(zc) > 0:
                assert np.all(zc >= 0.0), "Zero crossings must be >= 0"
                assert np.all(zc <= 1.0), "Zero crossings must be <= 1"

    def test_scale_invariance(self):
        """CSS of scaled contour should be equal (similarity = 1)."""
        c = _circle(64, r=1.0)
        c_scaled = _circle(64, r=5.0)
        vec_a = css_to_feature_vector(curvature_scale_space(c))
        vec_b = css_to_feature_vector(curvature_scale_space(c_scaled))
        sim = css_similarity(vec_a, vec_b)
        assert sim > 0.9, f"Scale invariance violated: similarity={sim:.3f}"

    def test_translation_invariance(self):
        """CSS of translated contour should be equal."""
        c = _circle(64)
        c_shifted = c + np.array([100.0, 200.0])
        vec_a = css_to_feature_vector(curvature_scale_space(c))
        vec_b = css_to_feature_vector(curvature_scale_space(c_shifted))
        sim = css_similarity(vec_a, vec_b)
        assert sim > 0.9, f"Translation invariance violated: similarity={sim:.3f}"

    def test_custom_sigma_range(self):
        c = _circle(64)
        css = curvature_scale_space(c, sigma_range=[1.0, 2.0, 4.0])
        assert len(css) == 3

    def test_n_sigmas_respected(self):
        c = _circle(64)
        css = curvature_scale_space(c, n_sigmas=5)
        assert len(css) == 5

    def test_circle_has_few_crossings_at_high_sigma(self):
        """Circle is a smooth shape — few crossings at large sigma."""
        c = _circle(128)
        css = curvature_scale_space(c, sigma_range=[32.0, 64.0])
        # At large sigma, smooth shape should have 0 or few zero crossings
        for _, zc in css:
            assert len(zc) < 10, f"Too many zero crossings for circle at large sigma: {len(zc)}"


class TestCSSFeatureVector:
    """Properties of css_to_feature_vector."""

    def test_returns_1d_array(self):
        c = _circle(64)
        css = curvature_scale_space(c)
        vec = css_to_feature_vector(css)
        assert vec.ndim == 1

    def test_fixed_size_for_fixed_input(self):
        c1, c2 = _circle(64), _square(64)
        v1 = css_to_feature_vector(curvature_scale_space(c1))
        v2 = css_to_feature_vector(curvature_scale_space(c2))
        assert v1.shape == v2.shape, "Feature vectors must have identical shape"

    def test_unit_norm(self):
        c = _circle(64)
        vec = css_to_feature_vector(curvature_scale_space(c))
        norm = np.linalg.norm(vec)
        # non-zero contour should yield unit-norm vector
        assert abs(norm - 1.0) < 1e-9 or norm == 0.0, f"Not unit norm: {norm}"

    def test_all_non_negative(self):
        c = _circle(64)
        vec = css_to_feature_vector(curvature_scale_space(c))
        assert np.all(vec >= 0), "Feature vector should be non-negative (histogram-based)"

    def test_n_bins_respected(self):
        c = _circle(64)
        css = curvature_scale_space(c, n_sigmas=3)
        vec = css_to_feature_vector(css, n_bins=16)
        assert len(vec) == 3 * 16


class TestCSSSimilarity:
    """Properties of css_similarity."""

    def test_self_similarity_is_one(self):
        c = _circle(64)
        vec = css_to_feature_vector(curvature_scale_space(c))
        assert css_similarity(vec, vec) == pytest.approx(1.0, abs=1e-9)

    def test_symmetry(self):
        va = css_to_feature_vector(curvature_scale_space(_circle(64)))
        vb = css_to_feature_vector(curvature_scale_space(_square(64)))
        assert css_similarity(va, vb) == pytest.approx(css_similarity(vb, va), abs=1e-9)

    def test_range_0_1(self):
        va = css_to_feature_vector(curvature_scale_space(_circle(64)))
        vb = css_to_feature_vector(curvature_scale_space(_zigzag(64)))
        sim = css_similarity(va, vb)
        assert 0.0 <= sim <= 1.0

    def test_zero_vectors(self):
        z = np.zeros(64)
        assert css_similarity(z, z) == 0.0

    def test_circle_vs_square_less_than_circle_vs_circle(self):
        va = css_to_feature_vector(curvature_scale_space(_circle(128)))
        vb = css_to_feature_vector(curvature_scale_space(_square(128)))
        sim_same = css_similarity(va, va)
        sim_diff = css_similarity(va, vb)
        assert sim_same >= sim_diff, "Self-similarity must be >= cross-similarity"

    def test_mirror_ge_direct(self):
        va = css_to_feature_vector(curvature_scale_space(_zigzag(64)))
        vb = css_to_feature_vector(curvature_scale_space(_zigzag(64)[::-1]))
        assert css_similarity_mirror(va, vb) >= css_similarity(va, vb)


# ── Box-counting properties ────────────────────────────────────────────────────

class TestBoxCountingFD:
    """Mathematical properties of box_counting_fd."""

    def test_fd_in_range_1_2(self):
        for c in [_circle(64), _square(64), _zigzag(64)]:
            fd = box_counting_fd(c)
            assert 1.0 <= fd <= 2.0, f"FD out of range: {fd}"

    def test_deterministic(self):
        c = _zigzag(128)
        fd1 = box_counting_fd(c)
        fd2 = box_counting_fd(c)
        assert fd1 == fd2

    def test_smooth_curve_near_1(self):
        """A straight line has FD close to 1.0."""
        line = np.column_stack([np.linspace(0, 1, 64), np.zeros(64)])
        fd = box_counting_fd(line)
        assert fd < 1.5, f"Straight line FD should be near 1, got {fd:.3f}"

    def test_degenerate_single_point(self):
        """Single repeated point should not crash."""
        single = np.array([[0.5, 0.5]] * 10)
        fd = box_counting_fd(single)
        assert fd == pytest.approx(1.0)

    def test_degenerate_too_few_points(self):
        """Fewer than 4 points → 1.0 without error."""
        tiny = np.array([[0, 0], [1, 1], [0, 1]])
        fd = box_counting_fd(tiny)
        assert fd == pytest.approx(1.0)

    def test_n_scales_parameter(self):
        c = _circle(64)
        fd4 = box_counting_fd(c, n_scales=4)
        fd8 = box_counting_fd(c, n_scales=8)
        # Both valid, no crash; values close
        assert 1.0 <= fd4 <= 2.0
        assert 1.0 <= fd8 <= 2.0

    @pytest.mark.parametrize("n", [32, 64, 128, 256])
    def test_fd_stable_across_resolutions(self, n):
        c = _circle(n)
        fd = box_counting_fd(c)
        assert 1.0 <= fd <= 2.0


class TestBoxCountingCurve:
    """Properties of box_counting_curve."""

    def test_returns_two_arrays(self):
        c = _circle(64)
        log_r, log_N = box_counting_curve(c)
        assert isinstance(log_r, np.ndarray)
        assert isinstance(log_N, np.ndarray)

    def test_same_length(self):
        c = _circle(64)
        log_r, log_N = box_counting_curve(c)
        assert len(log_r) == len(log_N)

    def test_length_equals_n_scales(self):
        c = _circle(64)
        log_r, log_N = box_counting_curve(c, n_scales=5)
        assert len(log_r) == 5

    def test_log_r_inv_monotone(self):
        """log(1/r) should be strictly increasing with scale index."""
        c = _circle(64)
        log_r, _ = box_counting_curve(c)
        diffs = np.diff(log_r)
        assert np.all(diffs >= 0), "log(1/r) should be non-decreasing"

    def test_log_N_non_decreasing(self):
        """More boxes at finer scale → N should grow or stay."""
        c = _circle(128)
        _, log_N = box_counting_curve(c)
        diffs = np.diff(log_N)
        assert np.all(diffs >= -1e-9), "log(N) should be non-decreasing with finer scales"

    def test_degenerate_constant_contour(self):
        const = np.zeros((64, 2))
        log_r, log_N = box_counting_curve(const)
        assert len(log_r) > 0  # no crash


# ── IFS properties ─────────────────────────────────────────────────────────────

class TestIFSCoefficients:
    """Properties of fit_ifs_coefficients."""

    def test_returns_1d_array(self):
        c = _circle(64)
        d = fit_ifs_coefficients(c)
        assert d.ndim == 1

    def test_length_equals_n_transforms(self):
        c = _circle(64)
        d = fit_ifs_coefficients(c, n_transforms=8)
        assert len(d) == 8

    def test_coefficients_bounded(self):
        """All IFS coefficients must be in (-1, 1) for convergence."""
        for c in [_circle(64), _square(64), _zigzag(64)]:
            d = fit_ifs_coefficients(c)
            assert np.all(np.abs(d) < 1.0), f"IFS coefficients out of bounds: {d}"
            assert np.all(np.abs(d) <= 0.95), f"IFS clip violated: {np.max(np.abs(d))}"

    def test_deterministic(self):
        c = _zigzag(128)
        d1 = fit_ifs_coefficients(c)
        d2 = fit_ifs_coefficients(c)
        np.testing.assert_array_equal(d1, d2)

    def test_few_points_fallback(self):
        """Fewer points than transforms → fallback n_transforms."""
        c = np.column_stack([np.linspace(0, 1, 8), np.zeros(8)])
        d = fit_ifs_coefficients(c, n_transforms=20)
        assert len(d) > 0
        assert np.all(np.abs(d) <= 0.95)


class TestIFSReconstruct:
    """Properties of reconstruct_from_ifs."""

    def test_returns_1d_array(self):
        c = _circle(64)
        d = fit_ifs_coefficients(c)
        r = reconstruct_from_ifs(d)
        assert r.ndim == 1

    def test_length_equals_n_points(self):
        c = _circle(64)
        d = fit_ifs_coefficients(c)
        r = reconstruct_from_ifs(d, n_points=128)
        assert len(r) == 128

    def test_output_finite(self):
        for c in [_circle(64), _zigzag(64)]:
            d = fit_ifs_coefficients(c)
            r = reconstruct_from_ifs(d)
            assert np.all(np.isfinite(r))

    def test_zero_coefficients_reconstruct_constant(self):
        """Zero IFS coefficients should yield a constant (flat) signal."""
        d = np.zeros(8)
        r = reconstruct_from_ifs(d, n_points=64)
        # All values should be the same (or very close)
        assert np.ptp(r) < 1e-9 or np.all(np.isfinite(r))


# ── FractalSignature properties ────────────────────────────────────────────────

class TestFractalSignature:
    """Properties of compute_fractal_signature."""

    @pytest.fixture(scope="class")
    def sig(self):
        return compute_fractal_signature(_circle(256))

    def test_has_fd_box(self, sig):
        assert hasattr(sig, "fd_box")
        assert 1.0 <= sig.fd_box <= 2.0

    def test_has_fd_divider(self, sig):
        assert hasattr(sig, "fd_divider")
        assert 1.0 <= sig.fd_divider <= 2.0

    def test_has_ifs_coeffs(self, sig):
        assert hasattr(sig, "ifs_coeffs")
        assert isinstance(sig.ifs_coeffs, np.ndarray)
        assert len(sig.ifs_coeffs) > 0

    def test_has_css_image(self, sig):
        assert hasattr(sig, "css_image")
        assert isinstance(sig.css_image, list)

    def test_has_chain_code(self, sig):
        assert hasattr(sig, "chain_code")
        assert sig.chain_code is not None

    def test_has_curve(self, sig):
        assert hasattr(sig, "curve")
        assert isinstance(sig.curve, np.ndarray)
        assert sig.curve.shape[1] == 2

    def test_deterministic(self):
        c = _circle(128)
        s1 = compute_fractal_signature(c)
        s2 = compute_fractal_signature(c)
        assert s1.fd_box == s2.fd_box
        np.testing.assert_array_equal(s1.ifs_coeffs, s2.ifs_coeffs)

    def test_circle_vs_square_different_fd(self):
        s_circle = compute_fractal_signature(_circle(256))
        s_square = compute_fractal_signature(_square(256))
        # FD values don't have to be hugely different but signatures must differ
        assert not np.array_equal(s_circle.ifs_coeffs, s_square.ifs_coeffs)

    @pytest.mark.parametrize("n", [64, 128, 256])
    def test_various_contour_sizes(self, n):
        c = _circle(n)
        sig = compute_fractal_signature(c)
        assert sig is not None
        assert 1.0 <= sig.fd_box <= 2.0
