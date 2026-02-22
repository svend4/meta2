"""Additional tests for test_fractal.py — IFS and CSS algorithms."""
import math
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.fractal.ifs import (
    fit_ifs_coefficients,
    reconstruct_from_ifs,
    ifs_distance,
)
from puzzle_reconstruction.algorithms.fractal.css import (
    curvature_scale_space,
    css_to_feature_vector,
    css_similarity,
    css_similarity_mirror,
    freeman_chain_code,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _circle(n: int = 128, r: float = 50.0) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _line(n: int = 64, length: float = 100.0) -> np.ndarray:
    return np.column_stack([np.linspace(0, length, n), np.zeros(n)])


def _square(n: int = 64) -> np.ndarray:
    s = n // 4
    pts = []
    for i in range(s):
        pts.append([i, 0.0])
    for i in range(s):
        pts.append([s, float(i)])
    for i in range(s):
        pts.append([s - i, float(s)])
    for i in range(s):
        pts.append([0.0, s - i])
    return np.array(pts, dtype=float)


# ─── TestFitIFSCoefficientsExtra ──────────────────────────────────────────────

class TestFitIFSCoefficientsExtra:
    def test_n_transforms_2(self):
        coeffs = fit_ifs_coefficients(_circle(), n_transforms=2)
        assert coeffs.shape == (2,)

    def test_n_transforms_4(self):
        coeffs = fit_ifs_coefficients(_line(), n_transforms=4)
        assert coeffs.shape == (4,)

    def test_n_transforms_16(self):
        coeffs = fit_ifs_coefficients(_circle(n=256), n_transforms=16)
        assert coeffs.shape[0] <= 16

    def test_all_coeffs_bounded_below_1(self):
        coeffs = fit_ifs_coefficients(_circle(n=128), n_transforms=8)
        assert np.all(np.abs(coeffs) < 1.0)

    def test_coeffs_clipped_to_0_95(self):
        """All |coefficients| must be ≤ 0.95 (clipping bound)."""
        rng = np.random.RandomState(7)
        pts = rng.randn(128, 2)
        coeffs = fit_ifs_coefficients(pts, n_transforms=8)
        assert np.all(np.abs(coeffs) <= 0.95 + 1e-10)

    def test_square_contour_works(self):
        coeffs = fit_ifs_coefficients(_square(), n_transforms=4)
        assert len(coeffs) >= 1

    def test_short_curve_adapts_n_transforms(self):
        """When N < n_transforms + 2, n_transforms is clamped internally."""
        pts = _line(n=6)
        coeffs = fit_ifs_coefficients(pts, n_transforms=8)
        assert len(coeffs) >= 1

    def test_returns_ndarray(self):
        coeffs = fit_ifs_coefficients(_line(), n_transforms=4)
        assert isinstance(coeffs, np.ndarray)


# ─── TestReconstructFromIFSExtra ──────────────────────────────────────────────

class TestReconstructFromIFSExtra:
    def test_n_points_64(self):
        coeffs = np.array([0.3, -0.4, 0.2, 0.1])
        profile = reconstruct_from_ifs(coeffs, n_points=64)
        assert profile.shape == (64,)

    def test_n_points_512(self):
        coeffs = np.array([0.3, -0.4, 0.2, 0.1])
        profile = reconstruct_from_ifs(coeffs, n_points=512)
        assert profile.shape == (512,)

    def test_zero_coeffs_returns_zeros(self):
        coeffs = np.zeros(4)
        profile = reconstruct_from_ifs(coeffs, n_points=128)
        assert np.allclose(profile, 0.0)

    def test_n_iter_1_no_crash(self):
        coeffs = np.array([0.5, -0.3])
        profile = reconstruct_from_ifs(coeffs, n_points=32, n_iter=1)
        assert profile.shape == (32,)

    def test_returns_ndarray(self):
        coeffs = np.array([0.2, -0.1, 0.3])
        assert isinstance(reconstruct_from_ifs(coeffs, n_points=64), np.ndarray)

    def test_finite_output(self):
        coeffs = np.array([0.3, -0.4, 0.2, 0.1])
        profile = reconstruct_from_ifs(coeffs, n_points=128)
        assert np.all(np.isfinite(profile))

    def test_round_trip_shape(self):
        """fit → reconstruct preserves n_points regardless of n_transforms."""
        curve = _circle(n=64)
        coeffs = fit_ifs_coefficients(curve, n_transforms=4)
        profile = reconstruct_from_ifs(coeffs, n_points=64)
        assert len(profile) == 64


# ─── TestIFSDistanceExtra ─────────────────────────────────────────────────────

class TestIFSDistanceExtra:
    def test_symmetric(self):
        a = np.array([0.3, -0.1, 0.5])
        b = np.array([0.1,  0.2, 0.0])
        assert ifs_distance(a, b) == pytest.approx(ifs_distance(b, a))

    def test_nonneg(self):
        a = np.array([0.3, -0.1, 0.5])
        b = np.array([0.1,  0.2, 0.0])
        assert ifs_distance(a, b) >= 0.0

    def test_zero_for_identical(self):
        a = np.array([0.3, -0.1, 0.5])
        assert ifs_distance(a, a) < 1e-10

    def test_different_lengths_nonneg(self):
        a = np.array([0.1, 0.2, 0.3, 0.4])
        b = np.array([0.5, 0.6])
        d = ifs_distance(a, b)
        assert d >= 0.0

    def test_returns_float(self):
        a = np.array([0.3, 0.1])
        b = np.array([0.1, 0.3])
        assert isinstance(ifs_distance(a, b), float)

    def test_triangle_inequality(self):
        a = np.array([0.5, -0.3, 0.2])
        b = np.array([0.1,  0.4, 0.1])
        c = np.array([0.0,  0.0, 0.9])
        assert ifs_distance(a, c) <= ifs_distance(a, b) + ifs_distance(b, c) + 1e-9


# ─── TestCSSExtra ─────────────────────────────────────────────────────────────

class TestCSSExtra:
    def test_n_sigmas_1(self):
        css = curvature_scale_space(_circle(n=64), n_sigmas=1)
        assert len(css) == 1

    def test_n_sigmas_10(self):
        css = curvature_scale_space(_circle(n=64), n_sigmas=10)
        assert len(css) == 10

    def test_sigma_values_are_float(self):
        css = curvature_scale_space(_circle(n=64), n_sigmas=4)
        for sigma, _ in css:
            assert isinstance(sigma, float)

    def test_zero_crossings_in_0_1(self):
        css = curvature_scale_space(_circle(n=64), n_sigmas=4)
        for _, zc in css:
            if len(zc) > 0:
                assert np.all(zc >= 0.0)
                assert np.all(zc <= 1.0 + 1e-10)

    def test_custom_sigma_range(self):
        css = curvature_scale_space(_circle(n=64), sigma_range=[1.0, 5.0, 10.0])
        assert len(css) == 3

    def test_square_returns_list(self):
        css = curvature_scale_space(_square(n=64), n_sigmas=4)
        assert isinstance(css, list)


# ─── TestCSSFeatureVectorExtra ────────────────────────────────────────────────

class TestCSSFeatureVectorExtra:
    def test_length_n_sigmas_times_n_bins(self):
        n_sigmas, n_bins = 5, 8
        css = curvature_scale_space(_circle(n=64), n_sigmas=n_sigmas)
        vec = css_to_feature_vector(css, n_bins=n_bins)
        assert len(vec) == n_sigmas * n_bins

    def test_n_bins_128(self):
        css = curvature_scale_space(_circle(n=64), n_sigmas=3)
        vec = css_to_feature_vector(css, n_bins=128)
        assert len(vec) == 3 * 128

    def test_norm_le_1(self):
        css = curvature_scale_space(_circle(n=64), n_sigmas=4)
        vec = css_to_feature_vector(css, n_bins=16)
        assert np.linalg.norm(vec) <= 1.0 + 1e-6

    def test_all_zero_for_degenerate(self):
        """All-same-point contour → all-zero CSS vector is valid (or norm=0)."""
        pts = np.ones((32, 2)) * 5.0
        css = curvature_scale_space(pts, n_sigmas=2)
        vec = css_to_feature_vector(css, n_bins=8)
        assert np.all(np.isfinite(vec))

    def test_returns_ndarray(self):
        css = curvature_scale_space(_circle(), n_sigmas=2)
        vec = css_to_feature_vector(css, n_bins=8)
        assert isinstance(vec, np.ndarray)


# ─── TestCSSSimilarityExtra ───────────────────────────────────────────────────

class TestCSSSimilarityExtra:
    def test_in_0_1(self):
        a = np.array([0.3, 0.4, 0.5, 0.6])
        a /= np.linalg.norm(a)
        b = np.array([0.1, 0.8, 0.2, 0.3])
        b /= np.linalg.norm(b)
        s = css_similarity(a, b)
        assert 0.0 <= s <= 1.0

    def test_symmetric(self):
        a = np.array([0.3, 0.4, 0.5])
        b = np.array([0.1, 0.8, 0.2])
        assert css_similarity(a, b) == pytest.approx(css_similarity(b, a))

    def test_identical_is_1(self):
        v = np.array([0.3, 0.4, 0.5])
        v /= np.linalg.norm(v)
        assert css_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_returns_0(self):
        a = np.zeros(5)
        b = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
        assert css_similarity(a, b) == pytest.approx(0.0)

    def test_different_lengths_accepted(self):
        a = np.array([0.1, 0.2, 0.3, 0.4])
        b = np.array([0.3, 0.2])
        s = css_similarity(a, b)
        assert 0.0 <= s <= 1.0

    def test_mirror_geq_direct(self):
        v = np.array([0.3, 0.4, 0.5, 0.6])
        v /= np.linalg.norm(v)
        u = v[::-1].copy()
        direct = css_similarity(v, u)
        mirror = css_similarity_mirror(v, u)
        assert mirror >= direct - 1e-9

    def test_mirror_reversed_identical_is_1(self):
        """Mirror similarity of a vector with its reverse should be 1."""
        v = np.array([0.3, 0.4, 0.5, 0.6])
        v /= np.linalg.norm(v)
        s = css_similarity_mirror(v, v[::-1].copy())
        assert s == pytest.approx(1.0, abs=1e-6)


# ─── TestFreemanChainCodeExtra ────────────────────────────────────────────────

class TestFreemanChainCodeExtra:
    def test_empty_for_single_point(self):
        pts = np.array([[0.0, 0.0]])
        assert freeman_chain_code(pts) == ""

    def test_only_valid_chars(self):
        code = freeman_chain_code(_square(n=64))
        assert all(c in "01234567" for c in code)

    def test_returns_str(self):
        code = freeman_chain_code(_circle(n=32))
        assert isinstance(code, str)

    def test_nonempty_for_valid_contour(self):
        code = freeman_chain_code(_circle(n=32))
        assert len(code) > 0

    def test_horizontal_line_contains_only_e_w(self):
        """Horizontal line → direction 0 (east) or 4 (west) only."""
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        code = freeman_chain_code(pts)
        assert all(c in "04" for c in code)

    def test_square_code_length(self):
        pts = _square(n=64)
        code = freeman_chain_code(pts)
        assert len(code) > 0
